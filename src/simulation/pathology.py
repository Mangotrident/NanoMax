import numpy as np
import scipy.signal
import logging

logger = logging.getLogger(__name__)


class PathologicalSimulator:
    """
    Simulates Cerebral Palsy (CP)-like neural signal degradation by injecting configurable
    pathological artifacts (EMG tremor, low-frequency baseline drift, electrode shift,
    channel dropouts, high impedance, and motion spikes).
    """

    def __init__(
        self,
        fs: float = 250.0,
        emg_amplitude: float = 0.0,  # EMG muscle artifact scale (in microvolts)
        drift_amplitude: float = 0.0,  # Baseline low-frequency drift scale (in microvolts)
        electrode_shift_prob: float = 0.0,  # Probability of channel swapping
        gaussian_noise_std: float = 0.0,  # Standard deviation of white noise (in microvolts)
        dropout_prob: float = 0.0,  # Probability of channel dropout
        motion_spike_rate: float = 0.0,  # Average motion spikes per minute
        motion_spike_amplitude: float = 0.0,  # Amplitude of motion spikes (in microvolts)
        impedance_shift_prob: float = 0.0,  # Probability of a sudden channel variance shift
    ):
        self.fs = fs
        self.emg_amplitude = emg_amplitude
        self.drift_amplitude = drift_amplitude
        self.electrode_shift_prob = electrode_shift_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.dropout_prob = dropout_prob
        self.motion_spike_rate = motion_spike_rate
        self.motion_spike_amplitude = motion_spike_amplitude
        self.impedance_shift_prob = impedance_shift_prob

    def inject_emg_contamination(self, signals: np.ndarray) -> np.ndarray:
        """
        Injects high-frequency muscle activity noise (30-100 Hz).
        """
        if self.emg_amplitude <= 0:
            return signals

        n_samples, n_channels = signals.shape
        # Create bandpass filtered white noise
        nyq = 0.5 * self.fs
        # If Nyquist limit is 125 Hz, we can filter to 30-100 Hz
        low = 30.0 / nyq
        high = min(100.0, self.fs * 0.45) / nyq
        b, a = scipy.signal.butter(4, [low, high], btype="band")

        emg_noise = np.random.randn(n_samples, n_channels)
        emg_noise = scipy.signal.filtfilt(b, a, emg_noise, axis=0)

        # Normalize and scale
        emg_noise = emg_noise / (np.std(emg_noise, axis=0) + 1e-8)
        return signals + self.emg_amplitude * emg_noise

    def inject_baseline_drift(self, signals: np.ndarray) -> np.ndarray:
        """
        Injects low-frequency baseline wander (e.g. 0.05 - 0.5 Hz).
        """
        if self.drift_amplitude <= 0:
            return signals

        n_samples, n_channels = signals.shape
        t = np.arange(n_samples) / self.fs

        # Sum of slow sine waves with random phases
        drifts = np.zeros((n_samples, n_channels))
        rng = np.random.default_rng(42)

        for c in range(n_channels):
            # Frequency components between 0.05 and 0.5 Hz
            freqs = rng.uniform(0.05, 0.5, size=3)
            phases = rng.uniform(0, 2 * np.pi, size=3)
            drift_c = np.zeros(n_samples)
            for f, p in zip(freqs, phases):
                drift_c += np.sin(2 * np.pi * f * t + p)
            drifts[:, c] = drift_c / 3.0

        return signals + self.drift_amplitude * drifts

    def inject_electrode_shift(self, signals: np.ndarray) -> np.ndarray:
        """
        Simulates displacement by swapping adjacent channels with a given probability.
        """
        if self.electrode_shift_prob <= 0:
            return signals

        n_samples, n_channels = signals.shape
        signals_shifted = signals.copy()

        rng = np.random.default_rng(42)
        # Iterate over pairs of channels and swap them with the given probability
        for c in range(0, n_channels - 1, 2):
            if rng.random() < self.electrode_shift_prob:
                # Swap channel c and c+1
                temp = signals_shifted[:, c].copy()
                signals_shifted[:, c] = signals_shifted[:, c + 1]
                signals_shifted[:, c + 1] = temp

        return signals_shifted

    def inject_gaussian_noise(self, signals: np.ndarray) -> np.ndarray:
        """
        Adds high-impedance thermal Gaussian white noise.
        """
        if self.gaussian_noise_std <= 0:
            return signals

        n_samples, n_channels = signals.shape
        noise = np.random.normal(
            0, self.gaussian_noise_std, size=(n_samples, n_channels)
        )
        return signals + noise

    def inject_channel_dropout(self, signals: np.ndarray) -> np.ndarray:
        """
        Simulates bad electrode contacts by zeroing or flat-lining random channels.
        """
        if self.dropout_prob <= 0:
            return signals

        n_samples, n_channels = signals.shape
        signals_dropout = signals.copy()

        rng = np.random.default_rng(42)
        for c in range(n_channels):
            if rng.random() < self.dropout_prob:
                # Zero out the entire channel
                signals_dropout[:, c] = 0.0

        return signals_dropout

    def inject_motion_spikes(self, signals: np.ndarray) -> np.ndarray:
        """
        Adds abrupt high-amplitude voltage steps or spikes (e.g. from coughing or swallowing).
        """
        if self.motion_spike_rate <= 0 or self.motion_spike_amplitude <= 0:
            return signals

        n_samples, n_channels = signals.shape
        signals_spikes = signals.copy()
        duration_min = n_samples / (self.fs * 60.0)
        n_spikes = int(np.random.poisson(self.motion_spike_rate * duration_min))

        if n_spikes == 0:
            return signals

        # Generate spikes
        rng = np.random.default_rng(42)
        spike_onsets = rng.choice(n_samples - 50, size=n_spikes, replace=False)

        # A spike is modeled as a short half-sine wave pulse (duration ~200ms)
        pulse_len = 50
        t = np.sin(np.linspace(0, np.pi, pulse_len))

        for onset in spike_onsets:
            # Add spike to random channels
            ch_idx = rng.choice(
                n_channels, size=rng.integers(1, n_channels // 2), replace=False
            )
            direction = rng.choice([-1, 1])
            for ch in ch_idx:
                signals_spikes[onset : onset + pulse_len, ch] += (
                    direction * self.motion_spike_amplitude * t
                )

        return signals_spikes

    def inject_impedance_shifts(self, signals: np.ndarray) -> np.ndarray:
        """
        Simulates impedance shifts that double the variance of random channels.
        """
        if self.impedance_shift_prob <= 0:
            return signals

        n_samples, n_channels = signals.shape
        signals_shifted = signals.copy()

        rng = np.random.default_rng(42)
        for c in range(n_channels):
            if rng.random() < self.impedance_shift_prob:
                # Multiply the channel signals by a random gain to simulate impedance variance change
                gain = rng.uniform(2.0, 5.0)
                signals_shifted[:, c] *= gain

        return signals_shifted

    def simulate(self, signals: np.ndarray) -> np.ndarray:
        """
        Applies the configured pathological effects sequentially to the signals.
        signals shape: (N_samples, N_channels)
        """
        out = signals.copy()
        out = self.inject_electrode_shift(out)
        out = self.inject_channel_dropout(out)
        out = self.inject_impedance_shifts(out)
        out = self.inject_emg_contamination(out)
        out = self.inject_baseline_drift(out)
        out = self.inject_gaussian_noise(out)
        out = self.inject_motion_spikes(out)
        return out
