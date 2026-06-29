import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List
from src.features.base import BaseDecoderPipeline
from src.features.dl_pipeline import PyTorchPipeline
from src.data.dataset import EEGDataset
from src.training.evaluator import MetricsEvaluator
import logging

logger = logging.getLogger(__name__)


class ModelAdapter:
    """
    Handles patient-specific adaptation (transfer learning) from healthy baseline decoders
    to noisy target subject EEG recordings.
    """

    def __init__(self):
        pass

    def linear_probe(
        self,
        pipeline: BaseDecoderPipeline,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        epochs: int = 15,
        lr: float = 1e-3,
    ) -> BaseDecoderPipeline:
        """
        Linear probing: Freezes the feature extractor, retrains only the classifier on calibration data.
        """
        adapted_pipeline = copy.deepcopy(pipeline)

        if isinstance(adapted_pipeline, PyTorchPipeline):
            # PyTorch: freeze all layers except classifier
            adapted_pipeline.model.train()
            for name, param in adapted_pipeline.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            dataset = EEGDataset(X_cal, y_cal)
            loader = DataLoader(dataset, batch_size=min(16, len(X_cal)), shuffle=True)

            # Optimize only active parameters
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, adapted_pipeline.model.parameters()),
                lr=lr,
            )
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(adapted_pipeline.device)
                    batch_y = batch_y.to(adapted_pipeline.device)

                    optimizer.zero_grad()
                    outputs = adapted_pipeline.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
        else:
            # Classical pipeline: extract features using fitted CSP/Covariance, refit Scikit-Learn classifier
            logger.info(
                "Classical linear probing: Refitting classifier head on calibration data."
            )
            # For Pipeline (e.g. Scikit-Learn Pipeline)
            if hasattr(adapted_pipeline, "pipeline"):
                # Extract features from the first steps (everything except 'lda' or 'clf')
                steps = list(adapted_pipeline.pipeline.named_steps.keys())
                clf_step = steps[-1]

                # Transform data through all steps except the classifier
                X_trans = X_cal
                for step_name in steps[:-1]:
                    X_trans = adapted_pipeline.pipeline.named_steps[
                        step_name
                    ].transform(X_trans)

                # Refit only the classifier
                adapted_pipeline.pipeline.named_steps[clf_step].fit(X_trans, y_cal)
            elif hasattr(adapted_pipeline, "clf"):
                # For Bandpower_Pipeline
                X_trans = adapted_pipeline._extract_bandpower(X_cal)
                X_trans = adapted_pipeline.scaler.transform(X_trans)
                adapted_pipeline.clf.fit(X_trans, y_cal)
            else:
                raise NotImplementedError(
                    "Classical pipeline structure not recognized for linear probing."
                )

        return adapted_pipeline

    def fine_tune(
        self,
        pipeline: BaseDecoderPipeline,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        epochs: int = 15,
        lr: float = 1e-4,
    ) -> BaseDecoderPipeline:
        """
        Fine-tuning: Updates all layers with a lower learning rate.
        """
        adapted_pipeline = copy.deepcopy(pipeline)

        if isinstance(adapted_pipeline, PyTorchPipeline):
            # PyTorch: make all layers trainable
            adapted_pipeline.model.train()
            for param in adapted_pipeline.model.parameters():
                param.requires_grad = True

            dataset = EEGDataset(X_cal, y_cal)
            loader = DataLoader(dataset, batch_size=min(16, len(X_cal)), shuffle=True)

            # Lower learning rate optimizer for all parameters
            optimizer = optim.Adam(adapted_pipeline.model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(adapted_pipeline.device)
                    batch_y = batch_y.to(adapted_pipeline.device)

                    optimizer.zero_grad()
                    outputs = adapted_pipeline.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
        else:
            # Classical pipeline: refit entire model
            logger.info(
                "Classical fine-tuning: Refitting entire pipeline on calibration data."
            )
            adapted_pipeline.fit(X_cal, y_cal)

        return adapted_pipeline

    def interpolate_weights(
        self,
        baseline_pipeline: BaseDecoderPipeline,
        target_pipeline: BaseDecoderPipeline,
        alpha: float = 0.5,
    ) -> BaseDecoderPipeline:
        """
        Interpolates weights between baseline (group) model and target-specific model.
        W_adapted = alpha * W_target + (1 - alpha) * W_baseline
        """
        adapted_pipeline = copy.deepcopy(target_pipeline)

        if isinstance(adapted_pipeline, PyTorchPipeline) and isinstance(
            baseline_pipeline, PyTorchPipeline
        ):
            state_base = baseline_pipeline.model.state_dict()
            state_target = target_pipeline.model.state_dict()
            state_adapted = {}

            for k in state_base.keys():
                state_adapted[k] = alpha * state_target[k].to(
                    adapted_pipeline.device
                ) + (1.0 - alpha) * state_base[k].to(adapted_pipeline.device)

            adapted_pipeline.model.load_state_dict(state_adapted)
        else:
            # Classical pipeline: interpolate classifier coefficients if LDA/linear SVM
            logger.info(
                "Classical weight interpolation: Interpolating LDA coefficients."
            )
            if hasattr(adapted_pipeline, "pipeline") and hasattr(
                baseline_pipeline, "pipeline"
            ):
                steps = list(adapted_pipeline.pipeline.named_steps.keys())
                clf_step = steps[-1]
                clf_adapted = adapted_pipeline.pipeline.named_steps[clf_step]
                clf_base = baseline_pipeline.pipeline.named_steps[clf_step]

                if hasattr(clf_adapted, "coef_") and hasattr(clf_base, "coef_"):
                    clf_adapted.coef_ = (
                        alpha * clf_adapted.coef_ + (1 - alpha) * clf_base.coef_
                    )
                    clf_adapted.intercept_ = (
                        alpha * clf_adapted.intercept_
                        + (1 - alpha) * clf_base.intercept_
                    )
            elif hasattr(adapted_pipeline, "clf") and hasattr(baseline_pipeline, "clf"):
                # For Bandpower_Pipeline
                clf_adapted = adapted_pipeline.clf
                clf_base = baseline_pipeline.clf
                if hasattr(clf_adapted, "coef_") and hasattr(clf_base, "coef_"):
                    # SVC linear kernel has coef_
                    clf_adapted.coef_ = (
                        alpha * clf_adapted.coef_ + (1 - alpha) * clf_base.coef_
                    )
                    clf_adapted.intercept_ = (
                        alpha * clf_adapted.intercept_
                        + (1 - alpha) * clf_base.intercept_
                    )

        return adapted_pipeline

    def _get_stratified_indices(self, y: np.ndarray, k: int) -> np.ndarray:
        """
        Samples k indices from y such that classes are balanced (stratified).
        """
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        samples_per_class = k // n_classes
        remainder = k % n_classes

        indices = []
        rng = np.random.default_rng(42)
        for i, c in enumerate(unique_classes):
            c_indices = np.where(y == c)[0]
            n_samples = samples_per_class + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(c_indices))
            sampled = rng.choice(c_indices, size=n_samples, replace=False)
            indices.extend(sampled)
        return np.array(indices)

    def run_few_shot_calibration(
        self,
        baseline_pipeline: BaseDecoderPipeline,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        shot_sizes: List[int] = [10, 20, 30, 50],
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmarks transfer learning performance under few-shot calibration regimes.
        Evaluates baseline, linear probing, fine-tuning, and weight interpolation.
        """
        results = {}

        # 1. Baseline Performance (0 trials calibration)
        preds_base = baseline_pipeline.predict(X_test)
        probs_base = baseline_pipeline.predict_proba(X_test)
        metrics_base = MetricsEvaluator.evaluate(y_test, preds_base, probs_base)
        results["baseline"] = {
            "accuracy": metrics_base["accuracy"],
            "cohen_kappa": metrics_base["cohen_kappa"],
        }

        for k in shot_sizes:
            if k > len(X_cal):
                continue

            # Stratified sample few-shot calibration trials
            indices = self._get_stratified_indices(y_cal, k)
            X_shot = X_cal[indices]
            y_shot = y_cal[indices]

            # --- Linear Probing ---
            lp_model = self.linear_probe(baseline_pipeline, X_shot, y_shot, epochs=15)
            preds_lp = lp_model.predict(X_test)
            probs_lp = lp_model.predict_proba(X_test)
            metrics_lp = MetricsEvaluator.evaluate(y_test, preds_lp, probs_lp)

            # --- Fine Tuning ---
            ft_model = self.fine_tune(baseline_pipeline, X_shot, y_shot, epochs=15)
            preds_ft = ft_model.predict(X_test)
            probs_ft = ft_model.predict_proba(X_test)
            metrics_ft = MetricsEvaluator.evaluate(y_test, preds_ft, probs_ft)

            # --- Weight Interpolation (blending target-fit and baseline) ---
            # First, train a target-fit model on the shot trials
            target_fit = copy.deepcopy(baseline_pipeline).fit(X_shot, y_shot)
            wi_model = self.interpolate_weights(
                baseline_pipeline, target_fit, alpha=0.5
            )
            preds_wi = wi_model.predict(X_test)
            probs_wi = wi_model.predict_proba(X_test)
            metrics_wi = MetricsEvaluator.evaluate(y_test, preds_wi, probs_wi)

            results[f"shot_{k}_linear_probe"] = {
                "accuracy": metrics_lp["accuracy"],
                "cohen_kappa": metrics_lp["cohen_kappa"],
            }
            results[f"shot_{k}_fine_tune"] = {
                "accuracy": metrics_ft["accuracy"],
                "cohen_kappa": metrics_ft["cohen_kappa"],
            }
            results[f"shot_{k}_weight_interpolation"] = {
                "accuracy": metrics_wi["accuracy"],
                "cohen_kappa": metrics_wi["cohen_kappa"],
            }

        return results
