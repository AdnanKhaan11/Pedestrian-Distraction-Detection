"""
This file evaluates your trained posture classifier on held-out test data.
It computes classification metrics, saves them to disk, and logs them to MLflow.
The outputs are structured so they are easy to track with DAGsHub and DVC later.
This keeps evaluation separated from training, which is better for MLOps design.
"""

from pathlib import Path
import os

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.entity.config_entity import PathsConfig, TrainingConfig
from src.utils.common import save_json, save_text
from src.utils.logger import get_logger


class PostureEvaluator:
    """
    Evaluate posture model performance.
    """

    def __init__(
        self,
        paths_config: PathsConfig,
        training_config: TrainingConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.paths_config = paths_config
        self.training_config = training_config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

    def _setup_mlflow(self) -> None:
        """
        Configure MLflow tracking for evaluation.
        """
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self.training_config.tracking.tracking_uri,
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.training_config.tracking.experiment_name_posture)

    def evaluate(self, model: torch.nn.Module, test_loader, device: str) -> dict:
        """
        Evaluate the model on test data.
        """
        self._setup_mlflow()

        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_inputs, batch_labels in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                y_true.extend(batch_labels.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred).tolist()
        class_report = classification_report(y_true, y_pred, zero_division=0)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": conf_matrix,
        }

        metrics_path = (
            self.paths_config.metrics_dir
            / self.training_config.posture_training.outputs.metrics_file_name
        )
        save_json(metrics_path, metrics)
        save_text(
            self.paths_config.metrics_dir / "posture_classification_report.txt",
            class_report,
        )

        with mlflow.start_run(run_name="posture_evaluation_run"):
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)

        self.logger.info("Posture evaluation metrics: %s", metrics)
        return metrics
