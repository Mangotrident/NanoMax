import numpy as np
import os
import psutil
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
)
from typing import Dict, Any, Optional, List
import torch


class MetricsEvaluator:
    """
    Evaluates classification performance, computing standard and BCI-specific metrics.
    Also tracks computational costs such as inference latency and memory footprints.
    """

    @staticmethod
    def calculate_multiclass_specificity(
        y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4
    ) -> float:
        """
        Calculates the average specificity across all classes in a One-vs-Rest fashion.
        """
        specificities = []
        for c in range(n_classes):
            # True Negatives (TN) and False Positives (FP)
            tn = np.sum((y_true != c) & (y_pred != c))
            fp = np.sum((y_true != c) & (y_pred == c))
            if (tn + fp) > 0:
                specificities.append(tn / (tn + fp))
            else:
                specificities.append(0.0)
        return float(np.mean(specificities))

    @classmethod
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        inference_times: Optional[List[float]] = None,
        n_classes: int = 4,
    ) -> Dict[str, Any]:
        """
        Computes all required performance and system metrics.
        """
        # Accuracy & Balanced Accuracy
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 (macro averaged)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        # Specificity
        spec = self.calculate_multiclass_specificity(y_true, y_pred, n_classes)

        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # ROC AUC (One-vs-Rest)
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = (
                0.5  # Fallback if classes are missing or predictions are degenerate
            )

        # PR AUC (One-vs-Rest average precision)
        try:
            # One-hot encode y_true
            y_true_oh = np.eye(n_classes)[y_true]
            pr_auc = average_precision_score(y_true_oh, y_prob, average="macro")
        except Exception:
            pr_auc = 0.0

        # Latency (inference time per trial in milliseconds)
        avg_latency = 0.0
        if inference_times is not None and len(inference_times) > 0:
            avg_latency = float(np.mean(inference_times)) * 1000.0  # seconds -> ms

        # Memory usage of current process in MB
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024**2)

        # GPU utilization
        gpu_util = 0.0
        if torch.cuda.is_available():
            # Simply get active memory percent or static mock
            gpu_util = torch.cuda.memory_allocated() / (1024**2)  # allocated MB
        elif torch.backends.mps.is_available():
            # For Apple Silicon, we don't have direct memory query in torch, return 0.0 or a placeholder
            gpu_util = 0.0

        return {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
            "f1_score": float(f1),
            "cohen_kappa": float(kappa),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "inference_latency_ms": avg_latency,
            "memory_usage_mb": float(mem_mb),
            "gpu_utilization_mb": float(gpu_util),
        }
