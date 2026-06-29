import os
import scipy.io
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubjectData:
    """
    Data container for a single subject and session.
    Holds continuous signals, channel names, trial onset indices, labels, artifacts, and sampling rate.
    """

    def __init__(
        self,
        subject_id: str,
        session_type: str,  # 'T' (train) or 'E' (evaluation)
        signals: np.ndarray,  # Shape: (N_samples, N_channels)
        trial_onsets: np.ndarray,  # Shape: (N_trials,)
        labels: np.ndarray,  # Shape: (N_trials,)
        artifacts: np.ndarray,  # Shape: (N_trials,)
        fs: float,
        ch_names: List[str],
    ):
        self.subject_id = subject_id
        self.session_type = session_type
        self.signals = signals
        self.trial_onsets = trial_onsets
        self.labels = labels
        self.artifacts = artifacts
        self.fs = fs
        self.ch_names = ch_names

    @property
    def num_trials(self) -> int:
        return len(self.trial_onsets)

    @property
    def num_channels(self) -> int:
        return self.signals.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.signals.shape[0] / self.fs


class BCI2aLoader:
    """
    Parser for BCI Competition IV Dataset 2a MATLAB (.mat) files.
    """

    CHANNELS_22_EEG = [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    ]
    CHANNELS_3_EOG = ["EOG1", "EOG2", "EOG3"]
    ALL_CHANNELS = CHANNELS_22_EEG + CHANNELS_3_EOG

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to directory containing the MAT files.
        """
        self.data_dir = data_dir

    def get_subject_filepath(self, subject_id: int, session_type: str) -> str:
        """
        Returns the absolute filepath for the given subject and session.
        Example: A01T.mat or A01E.mat.
        """
        sub_str = f"A{subject_id:02d}"
        filename = f"{sub_str}{session_type.upper()}.mat"
        return os.path.join(self.data_dir, filename)

    def load_subject_session(self, subject_id: int, session_type: str) -> SubjectData:
        """
        Loads and parses a MAT file for a single subject and session (T or E).
        """
        filepath = self.get_subject_filepath(subject_id, session_type)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        logger.info(f"Loading file: {filepath}")
        mat = scipy.io.loadmat(filepath)

        # 'data' is structured as a 1x9 object array
        data_cell = mat["data"][0]

        all_signals = []
        all_trials = []
        all_labels = []
        all_artifacts = []

        fs = 250.0  # Default frequency for Dataset 2a
        sample_offset = 0

        for run_idx, run_data in enumerate(data_cell):
            # Extract raw continuous signal for the run
            # run_data['X'] is a 1x1 array containing a 2D numpy array
            X_run = run_data["X"][0, 0]  # Shape: (N_samples, N_channels)

            # Extract trial onset indices, label classes, and artifact indicators
            trial_run = run_data["trial"][0, 0]  # Shape: (N_trials, 1) or (0, 1)
            y_run = run_data["y"][0, 0]  # Shape: (N_trials, 1) or (0, 1)

            # Extract artifact flags. If missing, assume all 0
            if "artifacts" in run_data.dtype.names:
                artifacts_run = run_data["artifacts"][0, 0]
            else:
                artifacts_run = np.zeros_like(y_run)

            # Retrieve actual sampling rate if available
            if "fs" in run_data.dtype.names:
                fs = float(run_data["fs"][0, 0][0, 0])

            # Convert labels to 0-based index: 1-4 becomes 0-3
            # We filter out runs that have no trials (e.g. calibration runs 0, 1, 2)
            if trial_run.size > 0:
                # Add sample offset to trial indices since we will concatenate runs
                trial_indices = (
                    trial_run.flatten() - 1 + sample_offset
                )  # Convert to 0-based index and offset
                labels = y_run.flatten() - 1  # Map [1, 2, 3, 4] -> [0, 1, 2, 3]
                artifacts = artifacts_run.flatten()

                all_trials.append(trial_indices)
                all_labels.append(labels)
                all_artifacts.append(artifacts)

            all_signals.append(X_run)
            sample_offset += X_run.shape[0]

        # Concatenate signals along time axis
        signals = np.vstack(all_signals)
        trial_onsets = np.concatenate(all_trials)
        labels = np.concatenate(all_labels)
        artifacts = np.concatenate(all_artifacts)

        return SubjectData(
            subject_id=f"A{subject_id:02d}",
            session_type=session_type.upper(),
            signals=signals,
            trial_onsets=trial_onsets,
            labels=labels,
            artifacts=artifacts,
            fs=fs,
            ch_names=self.ALL_CHANNELS,
        )

    def load_all_subjects(
        self, subjects: List[int] = list(range(1, 10))
    ) -> Dict[str, Dict[str, SubjectData]]:
        """
        Loads training ('T') and evaluation ('E') sessions for a list of subjects.
        Returns a dict: { 'A01': {'T': SubjectData, 'E': SubjectData}, ... }
        """
        data = {}
        for sub in subjects:
            sub_key = f"A{sub:02d}"
            data[sub_key] = {}
            for session in ["T", "E"]:
                try:
                    data[sub_key][session] = self.load_subject_session(sub, session)
                except Exception as e:
                    logger.error(
                        f"Error loading Subject {sub_key} Session {session}: {e}"
                    )
        return data
