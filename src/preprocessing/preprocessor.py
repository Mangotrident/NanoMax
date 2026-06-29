import mne
import numpy as np
from typing import Tuple, Optional
import logging
from src.data.loader import SubjectData

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """
    Modular and configurable preprocessing pipeline for EEG signals using MNE-Python.
    Supports bandpass filtering, notch filtering, common average referencing (CAR),
    ICA artifact removal, epoching, z-score normalization, and artifact rejection.
    """

    def __init__(
        self,
        l_freq: float = 8.0,
        h_freq: float = 30.0,
        notch_freq: float = 50.0,
        apply_car: bool = True,
        apply_ica: bool = False,
        ica_n_components: int = 15,
        tmin: float = 2.0,  # Start epoch relative to trial start (cue is at 2.0s)
        tmax: float = 6.0,  # End epoch relative to trial start (motor imagery ends at 6.0s)
        reject_threshold: Optional[float] = 100e-6,  # 100 uV peak-to-peak threshold
        verbose: str = "WARNING",
    ):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.apply_car = apply_car
        self.apply_ica = apply_ica
        self.ica_n_components = ica_n_components
        self.tmin = tmin
        self.tmax = tmax
        self.reject_threshold = reject_threshold
        self.verbose = verbose

    def _create_mne_raw(self, subject_data: SubjectData) -> mne.io.RawArray:
        """
        Converts custom SubjectData to mne.io.RawArray, scaling signal values to Volts.
        """
        # Convert microvolts to Volts
        scaled_signals = (
            subject_data.signals.T * 1e-6
        )  # shape must be (n_channels, n_times)

        info = mne.create_info(
            ch_names=subject_data.ch_names,
            sfreq=subject_data.fs,
            ch_types=["eeg"] * 22 + ["eog"] * 3,
        )

        raw = mne.io.RawArray(scaled_signals, info, verbose=self.verbose)
        return raw

    def process(
        self, subject_data: SubjectData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the full preprocessing pipeline on SubjectData.
        Returns:
            epochs_data: preprocessed EEG epochs of shape (N_clean_epochs, 22, N_times)
            labels: corresponding class labels of shape (N_clean_epochs,)
            artifacts_mask: boolean mask indicating which original trials were rejected (True = rejected)
        """
        # 1. Create Raw Array
        raw = self._create_mne_raw(subject_data)

        # Set montage for EEG channels
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore", verbose=self.verbose)

        # 2. Notch Filter
        if self.notch_freq is not None:
            raw.notch_filter(self.notch_freq, picks="all", verbose=self.verbose)

        # 3. Bandpass Filter
        if self.l_freq is not None or self.h_freq is not None:
            raw.filter(
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                picks="all",
                verbose=self.verbose,
            )

        # 4. Common Average Reference (CAR)
        if self.apply_car:
            raw.set_eeg_reference(
                ref_channels="average", projection=False, verbose=self.verbose
            )

        # 5. Optional ICA
        if self.apply_ica:
            logger.info("Fitting ICA for artifact rejection...")
            ica = mne.preprocessing.ICA(
                n_components=self.ica_n_components,
                random_state=42,
                max_iter=800,
                verbose=self.verbose,
            )
            ica.fit(raw, picks="eeg", verbose=self.verbose)

            # Find components correlating with EOG channels to exclude eye movements
            eog_indices = []
            for eog_ch in ["EOG1", "EOG2", "EOG3"]:
                if eog_ch in raw.ch_names:
                    indices, scores = ica.find_bads_eog(
                        raw, ch_name=eog_ch, threshold=3.0, verbose=self.verbose
                    )
                    eog_indices.extend(indices)

            # Remove duplicate indices
            ica.exclude = list(set(eog_indices))
            logger.info(f"ICA excluded components: {ica.exclude}")
            ica.apply(raw, verbose=self.verbose)

        # 6. Epoch Extraction
        # Create events array: shape (N_trials, 3)
        # Event structure: [sample_index, 0, event_id]
        events = np.zeros((len(subject_data.trial_onsets), 3), dtype=int)
        events[:, 0] = subject_data.trial_onsets
        events[:, 2] = subject_data.labels

        reject = (
            dict(eeg=self.reject_threshold)
            if self.reject_threshold is not None
            else None
        )

        # We only pick EEG channels for classification
        picks = mne.pick_types(raw.info, eeg=True, eog=False)

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=None,  # Handles all unique event codes
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,  # No baseline correction to avoid distortion
            picks=picks,
            preload=True,
            reject=reject,
            verbose=self.verbose,
        )

        # Get epochs data: shape (N_epochs, 22, N_times)
        # Multiply by 1e6 to convert back to microvolts or keep as is (V is standard, let's keep as Volts!)
        # Keeping as Volts makes features more stable, but let's standardise to Volts.
        epochs_data = epochs.get_data(verbose=self.verbose)
        labels = epochs.events[:, 2]

        # Calculate which trials were kept
        kept_indices = epochs.selection
        total_original_trials = len(subject_data.trial_onsets)

        # Create a boolean mask of rejected trials
        artifacts_mask = np.ones(total_original_trials, dtype=bool)
        artifacts_mask[kept_indices] = False

        logger.info(
            f"Epoching complete. Clean epochs: {len(labels)} / {total_original_trials}"
        )

        return epochs_data, labels, artifacts_mask

    def normalize_epochs(self, epochs_data: np.ndarray) -> np.ndarray:
        """
        Z-score normalizes each epoch across time steps.
        epochs_data shape: (N_epochs, N_channels, N_times)
        """
        mean = np.mean(epochs_data, axis=-1, keepdims=True)
        std = np.std(epochs_data, axis=-1, keepdims=True)
        # Prevent division by zero
        std[std == 0] = 1.0
        return (epochs_data - mean) / std
