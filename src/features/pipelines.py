import numpy as np
import scipy.signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
import pyriemann.estimation
import pyriemann.tangentspace
from typing import Tuple, List, Dict, Optional
from src.features.base import BaseDecoderPipeline
import logging

logger = logging.getLogger(__name__)


class CSP_LDA_Pipeline(BaseDecoderPipeline):
    """
    Pipeline 1: Common Spatial Patterns (CSP) + Log-Variance + LDA.
    """

    def __init__(self, n_components: int = 4, reg: Optional[str] = "ledoit_wolf"):
        self.n_components = n_components
        self.reg = reg
        self.csp = CSP(
            n_components=n_components,
            reg=reg,
            log=True,
            cov_est="concat",
            transform_into="average_power",
        )
        self.lda = LinearDiscriminantAnalysis()
        self.pipeline = Pipeline(
            [("csp", self.csp), ("scaler", StandardScaler()), ("lda", self.lda)]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP_LDA_Pipeline":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)


class FBCSP_Pipeline(BaseDecoderPipeline):
    """
    Pipeline 2: Filter Bank Common Spatial Patterns (FBCSP) + Feature Selection + LDA.
    """

    def __init__(
        self,
        fs: float = 250.0,
        bands: Optional[List[Tuple[float, float]]] = None,
        n_components: int = 4,
        k_features: int = 8,
    ):
        self.fs = fs
        # Standard filter banks covering theta, alpha, beta bands
        self.bands = bands or [
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 16.0),
            (16.0, 20.0),
            (20.0, 24.0),
            (24.0, 28.0),
            (28.0, 32.0),
            (32.0, 36.0),
            (36.0, 40.0),
        ]
        self.n_components = n_components
        self.k_features = k_features

        # We will create a list of CSP objects, one for each frequency band
        self.csps = [
            CSP(
                n_components=n_components,
                reg="ledoit_wolf",
                log=True,
                cov_est="concat",
                transform_into="average_power",
            )
            for _ in self.bands
        ]
        self.selector = SelectKBest(score_func=f_classif, k=k_features)
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis()

    def _bandpass_filter_epochs(
        self, X: np.ndarray, l_freq: float, h_freq: float
    ) -> np.ndarray:
        """
        Applies a zero-phase Butterworth bandpass filter to EEG epochs.
        X shape: (N_trials, N_channels, N_times)
        """
        nyq = 0.5 * self.fs
        low = l_freq / nyq
        high = h_freq / nyq
        b, a = scipy.signal.butter(4, [low, high], btype="band")
        # Filter along the time axis (last axis)
        return scipy.signal.filtfilt(b, a, X, axis=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSP_Pipeline":
        fb_features = []
        for i, (l_freq, h_freq) in enumerate(self.bands):
            X_filt = self._bandpass_filter_epochs(X, l_freq, h_freq)
            # Fit CSP for this band
            self.csps[i].fit(X_filt, y)
            feat = self.csps[i].transform(X_filt)  # shape: (N_trials, n_components)
            fb_features.append(feat)

        # Concatenate features from all bands: (N_trials, n_components * n_bands)
        X_feat = np.hstack(fb_features)

        # Select best K features using ANOVA
        X_selected = self.selector.fit_transform(X_feat, y)
        X_scaled = self.scaler.fit_transform(X_selected)

        # Fit classifier
        self.lda.fit(X_scaled, y)
        return self

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        fb_features = []
        for i, (l_freq, h_freq) in enumerate(self.bands):
            X_filt = self._bandpass_filter_epochs(X, l_freq, h_freq)
            feat = self.csps[i].transform(X_filt)
            fb_features.append(feat)
        X_feat = np.hstack(fb_features)
        X_selected = self.selector.transform(X_feat)
        return self.scaler.transform(X_selected)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_feat = self._extract_features(X)
        return self.lda.predict(X_feat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_feat = self._extract_features(X)
        return self.lda.predict_proba(X_feat)


class Bandpower_Pipeline(BaseDecoderPipeline):
    """
    Pipeline 3: Bandpower Features (Power Spectral Density in Alpha/Beta bands) + SVM.
    """

    def __init__(
        self, fs: float = 250.0, bands: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.fs = fs
        self.bands = bands or {"alpha": (8.0, 12.0), "beta": (13.0, 30.0)}
        self.scaler = StandardScaler()
        self.clf = SVC(probability=True, kernel="linear", C=1.0)

    def _extract_bandpower(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts average bandpower in specified bands for all channels.
        X shape: (N_trials, N_channels, N_times)
        Returns: (N_trials, N_channels * N_bands)
        """
        n_trials, n_channels, n_times = X.shape
        features = []

        # We compute PSD for each trial using Welch's method
        # n_perseg should be smaller than n_times, e.g. 256
        n_perseg = min(256, n_times)
        freqs, psds = scipy.signal.welch(X, fs=self.fs, nperseg=n_perseg, axis=-1)
        # psds shape: (N_trials, N_channels, N_freqs)

        for band_name, (l_freq, h_freq) in self.bands.items():
            freq_idx = np.where((freqs >= l_freq) & (freqs <= h_freq))[0]
            if len(freq_idx) == 0:
                # Fallback to closest frequency bin
                freq_idx = [np.argmin(np.abs(freqs - (l_freq + h_freq) / 2))]

            # Average power in this band: shape (N_trials, N_channels)
            bp = np.mean(psds[:, :, freq_idx], axis=-1)
            # Log transform is standard for power features
            bp = np.log10(bp + 1e-10)
            features.append(bp)

        # Concatenate: shape (N_trials, N_channels * N_bands)
        return np.hstack(features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Bandpower_Pipeline":
        X_feat = self._extract_bandpower(X)
        X_scaled = self.scaler.fit_transform(X_feat)
        self.clf.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_feat = self._extract_bandpower(X)
        X_scaled = self.scaler.transform(X_feat)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_feat = self._extract_bandpower(X)
        X_scaled = self.scaler.transform(X_feat)
        return self.clf.predict_proba(X_scaled)


class Riemannian_Pipeline(BaseDecoderPipeline):
    """
    Pipeline 4: Riemannian Covariance + Tangent Space Mapping + Logistic Regression / LDA.
    """

    def __init__(self, estimator: str = "oas", metric: str = "riemann"):
        self.cov_estimator = pyriemann.estimation.Covariances(estimator=estimator)
        self.tsm = pyriemann.tangentspace.TangentSpace(metric=metric)
        self.scaler = StandardScaler()
        self.clf = LinearDiscriminantAnalysis()

        self.pipeline = Pipeline(
            [
                ("cov", self.cov_estimator),
                ("tsm", self.tsm),
                ("scaler", self.scaler),
                ("clf", self.clf),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Riemannian_Pipeline":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)
