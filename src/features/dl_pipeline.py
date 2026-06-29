import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Type, Dict, Any, Optional
from src.features.base import BaseDecoderPipeline
from src.data.dataset import EEGDataset
import logging

logger = logging.getLogger(__name__)


class PyTorchPipeline(BaseDecoderPipeline):
    """
    Generic wrapper for PyTorch deep learning models to implement the BaseDecoderPipeline interface.
    Handles training, inference, batching, optimization, and hardware acceleration.
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Dict[str, Any],
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"PyTorchPipeline using device: {self.device}")

        self.model = self.model_class(**self.model_kwargs).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchPipeline":
        """
        Trains the PyTorch model on the input dataset.
        """
        self.model.train()
        dataset = EEGDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

            acc = correct / total
            avg_loss = epoch_loss / total
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}"
                )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for the input data.
        """
        self.model.eval()
        dataset = EEGDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = outputs.max(1)
                preds.append(predicted.cpu().numpy())

        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the input data.
        """
        self.model.eval()
        dataset = EEGDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probabilities = torch.softmax(outputs, dim=1)
                probs.append(probabilities.cpu().numpy())

        return np.concatenate(probs, axis=0)

    def export_onnx(self, file_path: str):
        """
        Exports the trained PyTorch model to ONNX format.
        """
        self.model.eval()
        # Input shape: (batch_size, 1, n_channels, n_times)
        n_channels = self.model_kwargs.get("n_channels", 22)
        n_times = self.model_kwargs.get("n_times", 1000)
        dummy_input = torch.randn(1, 1, n_channels, n_times, device=self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            file_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"Model successfully exported to ONNX: {file_path}")

    def export_torchscript(self, file_path: str):
        """
        Exports the trained PyTorch model to TorchScript format.
        """
        self.model.eval()
        n_channels = self.model_kwargs.get("n_channels", 22)
        n_times = self.model_kwargs.get("n_times", 1000)
        dummy_input = torch.randn(1, 1, n_channels, n_times, device=self.device)

        traced_cell = torch.jit.trace(self.model, dummy_input)
        traced_cell.save(file_path)
        logger.info(f"Model successfully exported to TorchScript: {file_path}")
