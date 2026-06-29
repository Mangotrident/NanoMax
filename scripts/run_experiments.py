import argparse
import logging
import os
import pandas as pd
from src.training.loso_cv import LOSOCrossValidator
from src.features.pipelines import (
    CSP_LDA_Pipeline,
    Riemannian_Pipeline,
    FBCSP_Pipeline,
    Bandpower_Pipeline,
)
from src.features.dl_pipeline import PyTorchPipeline
from src.models.deep_learning import EEGNet, ShallowConvNet, DeepConvNet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_experiments(data_dir: str, fast: bool = False):
    """
    Runs baseline experiments comparing all pipelines.
    """
    subjects = [1, 9] if fast else list(range(1, 10))
    dl_epochs = 2 if fast else 30

    # Initialize LOSO validator
    validator = LOSOCrossValidator(
        data_dir=data_dir, checkpoint_dir="models_checkpoints", results_dir="results"
    )

    # Define pipelines to evaluate
    pipelines = {
        "CSP_LDA": {"class": CSP_LDA_Pipeline, "kwargs": {"n_components": 4}},
        "FBCSP": {
            "class": FBCSP_Pipeline,
            "kwargs": {"fs": 250.0, "n_components": 4, "k_features": 8},
        },
        "Bandpower": {"class": Bandpower_Pipeline, "kwargs": {"fs": 250.0}},
        "Riemannian": {
            "class": Riemannian_Pipeline,
            "kwargs": {"estimator": "oas", "metric": "riemann"},
        },
        "EEGNet": {
            "class": PyTorchPipeline,
            "kwargs": {
                "model_class": EEGNet,
                "model_kwargs": {"n_channels": 22, "n_classes": 4, "n_times": 1001},
                "epochs": dl_epochs,
                "batch_size": 32,
                "lr": 1e-3,
            },
        },
        "ShallowConvNet": {
            "class": PyTorchPipeline,
            "kwargs": {
                "model_class": ShallowConvNet,
                "model_kwargs": {"n_channels": 22, "n_classes": 4, "n_times": 1001},
                "epochs": dl_epochs,
                "batch_size": 32,
                "lr": 1e-3,
            },
        },
        "DeepConvNet": {
            "class": PyTorchPipeline,
            "kwargs": {
                "model_class": DeepConvNet,
                "model_kwargs": {"n_channels": 22, "n_classes": 4, "n_times": 1001},
                "epochs": dl_epochs,
                "batch_size": 32,
                "lr": 1e-3,
            },
        },
    }

    all_summary_results = []

    for name, p_info in pipelines.items():
        logger.info("==================================================")
        logger.info(f"Running experiments for {name}")
        logger.info("==================================================")

        try:
            results = validator.run_loso(
                pipeline_class=p_info["class"],
                pipeline_kwargs=p_info["kwargs"],
                pipeline_name=name,
                subjects=subjects,
            )

            # Extract mean metrics
            df = pd.DataFrame(results).T
            mean_metrics = df.mean().to_dict()
            mean_metrics["pipeline"] = name
            all_summary_results.append(mean_metrics)

        except Exception as e:
            logger.error(f"Failed to run experiments for {name}: {e}", exc_info=True)

    # Save a global comparative table
    if all_summary_results:
        summary_df = pd.DataFrame(all_summary_results)
        # Move pipeline to the first column
        cols = ["pipeline"] + [col for col in summary_df.columns if col != "pipeline"]
        summary_df = summary_df[cols]

        summary_csv = os.path.join("results", "global_comparison_results.csv")
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Global comparative analysis written to {summary_csv}")

        # Print a nice markdown table
        print("\n=== GLOBAL PERFORMANCE COMPARISON ===")
        print(summary_df.to_markdown(index=False))
        print("======================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline EEG motor imagery experiments."
    )
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Directory containing MAT files."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a fast validation experiment (only 2 subjects, 2 epochs).",
    )
    args = parser.parse_args()

    run_experiments(args.data_dir, args.fast)
