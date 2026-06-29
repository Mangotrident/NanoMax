import torch
import numpy as np
from typing import Tuple, Union, Optional, Any


class EEGDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for epoched EEG signals.
    Provides samples as (X_epoch, label), where X_epoch has shape (channels, time_steps)
    and label is a scalar integer.
    """

    def __init__(
        self,
        epochs: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        transform: Optional[Any] = None,
    ):
        """
        Args:
            epochs: shape (num_trials, channels, time_steps)
            labels: shape (num_trials,)
            transform: Optional callable transformation to apply to each trial
        """
        if isinstance(epochs, np.ndarray):
            self.epochs = torch.tensor(epochs, dtype=torch.float32)
        else:
            self.epochs = epochs.clone().detach().to(torch.float32)

        if isinstance(labels, np.ndarray):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels.clone().detach().to(torch.long)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.epochs[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y
