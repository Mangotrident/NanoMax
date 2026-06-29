import threading
import numpy as np
import mne
from typing import List, Tuple, Optional
from src.features.base import BaseDecoderPipeline
from src.preprocessing.preprocessor import EEGPreprocessor
import logging

logger = logging.getLogger(__name__)


class StreamingBuffer:
    """
    Thread-safe rolling buffer that stores real-time incoming EEG samples.
    """

    def __init__(self, n_channels: int = 22, max_len: int = 4000, sfreq: float = 250.0):
        self.n_channels = n_channels
        self.max_len = max_len
        self.sfreq = sfreq

        self.lock = threading.Lock()
        # Initialize circular buffer: shape (max_len, n_channels)
        self.buffer = np.zeros((self.max_len, self.n_channels), dtype=np.float32)
        self.pointer = 0  # Points to the next write position
        self.total_written = 0

    def append(self, chunk: np.ndarray):
        """
        Appends a chunk of new samples to the buffer.
        chunk shape: (n_samples, n_channels)
        """
        n_samples = chunk.shape[0]
        if n_samples == 0:
            return

        with self.lock:
            if n_samples >= self.max_len:
                # Chunk is larger than buffer, keep the tail end of the chunk
                self.buffer = chunk[-self.max_len :].astype(np.float32)
                self.pointer = 0
                self.total_written += n_samples
                return

            # Write chunk in parts if it wraps around the circular boundary
            space_left = self.max_len - self.pointer
            if n_samples <= space_left:
                self.buffer[self.pointer : self.pointer + n_samples] = chunk
                self.pointer = (self.pointer + n_samples) % self.max_len
            else:
                self.buffer[self.pointer :] = chunk[:space_left]
                self.buffer[: n_samples - space_left] = chunk[space_left:]
                self.pointer = n_samples - space_left

            self.total_written += n_samples

    def get_latest_window(self, window_size_samples: int) -> np.ndarray:
        """
        Retrieves the latest window_size_samples from the buffer.
        Returns: (n_channels, window_size_samples)
        """
        with self.lock:
            if window_size_samples > self.max_len:
                raise ValueError(
                    f"Requested window size {window_size_samples} exceeds buffer limit {self.max_len}"
                )

            if self.total_written < window_size_samples:
                # Buffer is not full enough yet, return whatever is there padded with zeros
                window = np.zeros(
                    (window_size_samples, self.n_channels), dtype=np.float32
                )
                if self.total_written > 0:
                    start_idx = self.pointer - self.total_written
                    if start_idx >= 0:
                        window[-self.total_written :] = self.buffer[
                            start_idx : self.pointer
                        ]
                    else:
                        window[-self.total_written :] = np.concatenate(
                            [self.buffer[start_idx:], self.buffer[: self.pointer]],
                            axis=0,
                        )
                return window.T

            # Normal circular extraction
            start_idx = (self.pointer - window_size_samples) % self.max_len
            if start_idx < self.pointer:
                window = self.buffer[start_idx : self.pointer]
            else:
                window = np.concatenate(
                    [self.buffer[start_idx:], self.buffer[: self.pointer]], axis=0
                )

            return window.T


class RealTimeInferenceEngine:
    """
    Manages online sliding window inference by reading from a StreamingBuffer,
    preprocessing, and decoding.
    """

    def __init__(
        self,
        pipeline: BaseDecoderPipeline,
        buffer: StreamingBuffer,
        preprocessor: EEGPreprocessor,
        window_size_sec: float = 4.0,
        sfreq: float = 250.0,
        ch_names: Optional[List[str]] = None,
    ):
        self.pipeline = pipeline
        self.buffer = buffer
        self.preprocessor = preprocessor
        self.window_size_samples = int(window_size_sec * sfreq)
        self.sfreq = sfreq

        # Channel names fallback if not provided
        self.ch_names = ch_names or [
            f"EEG-{i}" for i in range(1, buffer.n_channels + 1)
        ]

        # MNE info object for on-the-fly Epoch creation
        self.info = mne.create_info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=["eeg"] * len(self.ch_names),
        )

    def process_window(self, window_data: np.ndarray) -> np.ndarray:
        """
        Applies pre-processing to the sliding window data.
        window_data shape: (n_channels, window_size_samples)
        """
        # Create a raw array for the single window
        # We divide by 1e6 because our preprocessor expects Volts, and window_data is in microvolts
        raw = mne.io.RawArray(window_data * 1e-6, self.info, verbose=False)

        # Preprocessor expects a subject run object or Raw, let's create a Mock
        class MockRun:
            def __init__(self, raw_obj):
                self.raw = raw_obj
                self.run_type = "online"

        # Notch Filter
        if self.preprocessor.notch_freq is not None:
            raw.notch_filter(
                self.preprocessor.notch_freq,
                picks="all",
                verbose=self.preprocessor.verbose,
            )

        # Bandpass Filter
        if self.preprocessor.l_freq is not None or self.preprocessor.h_freq is not None:
            raw.filter(
                l_freq=self.preprocessor.l_freq,
                h_freq=self.preprocessor.h_freq,
                picks="all",
                verbose=self.preprocessor.verbose,
            )

        # CAR
        if self.preprocessor.apply_car:
            raw.set_eeg_reference(
                ref_channels="average",
                projection=False,
                verbose=self.preprocessor.verbose,
            )

        # Extract data: shape (n_channels, window_size_samples)
        processed_data = raw.get_data()

        # Normalize
        processed_data = self.preprocessor.normalize_epochs(
            np.expand_dims(processed_data, axis=0)
        )[0]

        return processed_data

    def run_inference_step(self) -> Tuple[int, np.ndarray]:
        """
        Pulls latest window, processes, and runs inference.
        Returns: (predicted_class_index, class_probabilities)
        """
        # 1. Pull latest window from the buffer
        window_data = self.buffer.get_latest_window(self.window_size_samples)

        # 2. Process window
        processed_window = self.process_window(window_data)

        # 3. Add batch dimension: shape (1, n_channels, n_times)
        batched_data = np.expand_dims(processed_window, axis=0)

        # 4. Predict
        pred = self.pipeline.predict(batched_data)[0]
        probs = self.pipeline.predict_proba(batched_data)[0]

        return int(pred), probs
