from abc import ABC, abstractmethod
import numpy as np


class BaseDecoderPipeline(ABC):
    """
    Common interface for all motor intent classification pipelines.
    Enables plug-and-play evaluation of both classical ML and deep learning models.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDecoderPipeline":
        """
        Fits the pipeline on the epoched EEG signals and corresponding labels.
        Args:
            X: Epoched EEG signals of shape (N_trials, N_channels, N_times)
            y: Labels of shape (N_trials,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the input epoched EEG signals.
        Args:
            X: Epoched EEG signals of shape (N_trials, N_channels, N_times)
        Returns:
            y_pred: Predicted class labels of shape (N_trials,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the input epoched EEG signals.
        Args:
            X: Epoched EEG signals of shape (N_trials, N_channels, N_times)
        Returns:
            y_prob: Predicted class probabilities of shape (N_trials, N_classes)
        """
        pass
