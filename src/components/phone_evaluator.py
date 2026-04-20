"""
This file evaluates the trained YOLO phone detector using Ultralytics validation.
It stores validation summaries on disk and logs key information for experiment tracking.
This is cleaner than mixing validation inside training scripts and helps later debugging.
The outputs are also easy to keep under DVC if needed.
"""

from pathlib import Path
import os

import mlflow

from src.entity.config_entity import PathsConfig, PhoneDetectorConfig, TrainingConfig
from src.utils.common import save_json
from src.utils.logger import get_logger


class PhoneEvaluator:
    """
    Evaluate the phone detector.
    """

    def __init__(
        self,
        paths_config: PathsConfig,
        phone_detector_config: PhoneDetectorConfig,
        training_config: TrainingConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.paths_config = paths_config
        self.phone_detector_config = phone_detector_config
        self.training_config = training_config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

    def _setup_mlflow(self) -> None:
        """
        Configure MLflow tracking for phone evaluation.
        """
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self.training_config.tracking.tracking_uri,
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.training_config.tracking.experiment_name_phone)

    def evaluate(self, data_yaml_path: Path, model_path: Path | None = None) -> dict:
        """
        Validate the phone detector on the dataset yaml.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Please install it before running phone evaluation."
            ) from exc

        if not data_yaml_path.exists():
            raise FileNotFoundError(f"YOLO dataset yaml not found: {data_yaml_path}")

        self._setup_mlflow()

        if model_path is None:
            model_path = self.phone_detector_config.weights.default_weight_file

        if not model_path.exists():
            raise FileNotFoundError(f"Phone detector weights not found: {model_path}")

        model = YOLO(str(model_path))
        validation_result = model.val(data=str(data_yaml_path))

        summary = {
            "model_path": str(model_path),
            "data_yaml_path": str(data_yaml_path),
            "validation_result_repr": str(validation_result),
        }

        save_json(
            self.paths_config.metrics_dir / "phone_evaluation_summary.json", summary
        )

        with mlflow.start_run(run_name="phone_evaluation_run"):
            mlflow.log_param("model_path", str(model_path))
            mlflow.log_param("data_yaml_path", str(data_yaml_path))

        self.logger.info("Phone evaluation summary: %s", summary)
        return summary
