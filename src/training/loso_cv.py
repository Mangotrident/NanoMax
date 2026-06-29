import os
import joblib
import torch
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, Any, Type, List, Optional, Tuple
import logging

from src.data.loader import BCI2aLoader
from src.preprocessing.preprocessor import EEGPreprocessor
from src.training.evaluator import MetricsEvaluator
from src.features.base import BaseDecoderPipeline

logger = logging.getLogger(__name__)


class LOSOCrossValidator:
    """
    Executes Leave-One-Subject-Out (LOSO) Cross-Validation across subjects.
    Handles data pooling, model training, evaluation, and checkpoint saving.
    """

    def __init__(
        self,
        data_dir: str,
        checkpoint_dir: str = "models_checkpoints",
        results_dir: str = "results",
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.loader = BCI2aLoader(data_dir)
        self.preprocessor = EEGPreprocessor(**(preprocessor_kwargs or {}))
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    def _get_pooled_data(
        self, subjects: List[int], sessions: List[str] = ["T", "E"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and preprocesses data for multiple subjects, pooling them into single arrays.
        """
        all_epochs = []
        all_labels = []

        for sub in subjects:
            for sess in sessions:
                try:
                    sub_data = self.loader.load_subject_session(sub, sess)
                    epochs, labels, _ = self.preprocessor.process(sub_data)
                    epochs = self.preprocessor.normalize_epochs(epochs)

                    all_epochs.append(epochs)
                    all_labels.append(labels)
                except Exception as e:
                    logger.warning(
                        f"Could not load Subject A{sub:02d} Session {sess}: {e}"
                    )

        return np.concatenate(all_epochs, axis=0), np.concatenate(all_labels, axis=0)

    def run_loso(
        self,
        pipeline_class: Type[BaseDecoderPipeline],
        pipeline_kwargs: Dict[str, Any],
        pipeline_name: str,
        subjects: List[int] = list(range(1, 10)),
    ) -> Dict[str, Any]:
        """
        Runs LOSO-CV across the specified subjects.
        For each subject:
            1. Train on all other subjects (pooling both T and E sessions).
            2. Evaluate on the left-out subject's E (evaluation) session.
            3. Save model checkpoint.
        """
        logger.info(f"Starting LOSO Cross-Validation for pipeline: {pipeline_name}")
        results = {}

        for test_sub in subjects:
            test_sub_str = f"A{test_sub:02d}"
            logger.info(f"--- Left-out Subject: {test_sub_str} ---")

            # 1. Pool training data (all subjects EXCEPT the test subject)
            train_subs = [s for s in subjects if s != test_sub]
            logger.info(f"Pooling training data from subjects: {train_subs}")
            X_train, y_train = self._get_pooled_data(train_subs, sessions=["T", "E"])

            # 2. Load test data (only the test subject's E session for strict evaluation)
            logger.info(f"Loading test data for subject: {test_sub_str} (Session E)")
            try:
                test_data = self.loader.load_subject_session(test_sub, "E")
                X_test, y_test, _ = self.preprocessor.process(test_data)
                X_test = self.preprocessor.normalize_epochs(X_test)
            except Exception as e:
                logger.error(f"Error loading test subject {test_sub_str}: {e}")
                continue

            # 3. Instantiate and fit pipeline
            logger.info(f"Fitting pipeline {pipeline_name}...")
            pipeline = pipeline_class(**pipeline_kwargs)

            start_train = time.time()
            pipeline.fit(X_train, y_train)
            train_duration = time.time() - start_train
            logger.info(f"Training completed in {train_duration:.2f}s")

            # 4. Predict and evaluate
            logger.info(f"Evaluating model on Subject {test_sub_str}...")
            inference_times = []
            preds = []
            probs = []

            for epoch in X_test:
                # Add batch dimension to simulate single-trial online inference
                epoch_batched = np.expand_dims(epoch, axis=0)

                t_start = time.time()
                pred = pipeline.predict(epoch_batched)[0]
                t_end = time.time()

                prob = pipeline.predict_proba(epoch_batched)[0]

                inference_times.append(t_end - t_start)
                preds.append(pred)
                probs.append(prob)

            preds = np.array(preds)
            probs = np.array(probs)

            # Compute metrics
            metrics = MetricsEvaluator.evaluate(
                y_test, preds, probs, inference_times=inference_times, n_classes=4
            )
            metrics["train_time_sec"] = train_duration

            logger.info(
                f"Subject {test_sub_str} Results - Accuracy: {metrics['accuracy']:.4f} - Kappa: {metrics['cohen_kappa']:.4f}"
            )
            results[test_sub_str] = metrics

            # 5. Save model checkpoint
            checkpoint_filename = f"{pipeline_name}_loso_leftout_{test_sub_str}"
            self.save_checkpoint(pipeline, checkpoint_filename)

        # Compile and save aggregate results
        self.save_aggregate_results(results, pipeline_name)
        return results

    def save_checkpoint(self, pipeline: BaseDecoderPipeline, filename: str):
        """
        Saves model weights or object to disk.
        """
        # If it's a PyTorch pipeline, we can save both the full pipeline and the model state_dict
        from src.features.dl_pipeline import PyTorchPipeline

        if isinstance(pipeline, PyTorchPipeline):
            # Save PyTorch state dict
            torch_path = os.path.join(self.checkpoint_dir, f"{filename}.pt")
            torch.save(pipeline.model.state_dict(), torch_path)
            logger.info(f"Saved PyTorch weights to {torch_path}")

            # Save the pipeline wrapper skeleton via joblib
            # Note: We must temporarily move the model off GPU/MPS to CPU or exclude the model state to make it portable
            model_device = pipeline.device
            pipeline.model.to("cpu")
            pkl_path = os.path.join(self.checkpoint_dir, f"{filename}_pipeline.pkl")
            joblib.dump(pipeline, pkl_path)
            pipeline.model.to(model_device)

            # Export model to ONNX format
            onnx_path = os.path.join(self.checkpoint_dir, f"{filename}.onnx")
            try:
                pipeline.export_onnx(onnx_path)
            except Exception as e:
                logger.error(
                    f"Failed to export ONNX model for checkpoint {filename}: {e}"
                )

            # Export model to TorchScript format
            ts_path = os.path.join(self.checkpoint_dir, f"{filename}_ts.pt")
            try:
                pipeline.export_torchscript(ts_path)
            except Exception as e:
                logger.error(
                    f"Failed to export TorchScript model for checkpoint {filename}: {e}"
                )
        else:
            # Classical pipeline, save using joblib
            pkl_path = os.path.join(self.checkpoint_dir, f"{filename}.pkl")
            joblib.dump(pipeline, pkl_path)
            logger.info(f"Saved classical pipeline to {pkl_path}")

    def save_aggregate_results(self, results: Dict[str, Any], pipeline_name: str):
        """
        Saves cross-subject evaluation metrics as CSV and JSON.
        """
        df = pd.DataFrame(results).T
        # Add mean row
        mean_row = df.mean()
        mean_row.name = "mean"
        df = pd.concat([df, pd.DataFrame([mean_row])])

        csv_path = os.path.join(self.results_dir, f"{pipeline_name}_loso_results.csv")
        df.to_csv(csv_path)
        logger.info(f"Saved CSV results to {csv_path}")

        json_path = os.path.join(self.results_dir, f"{pipeline_name}_loso_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved JSON results to {json_path}")
