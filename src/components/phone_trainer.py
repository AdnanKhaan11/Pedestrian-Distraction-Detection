"""
This file trains or fine-tunes your YOLO phone detector using the cleaned MLOps structure.
It follows your current project idea of using Ultralytics YOLO with a dataset yaml file.
It also saves training metadata for MLflow, DAGsHub, and DVC-friendly artifact tracking.
This keeps phone-model training separate from posture-model training.
"""

from pathlib import Path
import os

import mlflow

from src.entity.config_entity import PathsConfig, PhoneDetectorConfig, TrainingConfig
from src.utils.common import save_json, save_yaml
from src.utils.logger import get_logger


class PhoneTrainer:
    """
    Train or fine-tune the phone detector.
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
        Configure MLflow tracking for phone detector training.
        """
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self.training_config.tracking.tracking_uri,
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.training_config.tracking.experiment_name_phone)

    def train(self, data_yaml_path: Path) -> dict:
        """
        Train the YOLO phone detector using a dataset yaml file.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Please install it before running phone training."
            ) from exc

        if not data_yaml_path.exists():
            raise FileNotFoundError(f"YOLO dataset yaml not found: {data_yaml_path}")

        self._setup_mlflow()

        if self.training_config.phone_training.use_existing_weights:
            model_path = self.training_config.phone_training.existing_weights_path
        else:
            model_path = self.phone_detector_config.weights.fallback_weight_file

        if not model_path.exists():
            raise FileNotFoundError(
                f"Phone detector start weights not found: {model_path}"
            )

        self.logger.info("Starting phone detector training from: %s", model_path)
        model = YOLO(str(model_path))

        with mlflow.start_run(run_name="phone_training_run"):
            mlflow.log_param("start_weights", str(model_path))
            mlflow.log_param(
                "epochs", self.training_config.phone_training.hyperparameters.epochs
            )
            mlflow.log_param(
                "batch_size",
                self.training_config.phone_training.hyperparameters.batch_size,
            )
            mlflow.log_param(
                "image_size",
                self.training_config.phone_training.hyperparameters.image_size,
            )

            results = model.train(
                data=str(data_yaml_path),
                epochs=self.training_config.phone_training.hyperparameters.epochs,
                imgsz=self.training_config.phone_training.hyperparameters.image_size,
                batch=self.training_config.phone_training.hyperparameters.batch_size,
                device=self.training_config.general.device_preference,
            )

            summary = {
                "data_yaml_path": str(data_yaml_path),
                "start_weights": str(model_path),
                "epochs": self.training_config.phone_training.hyperparameters.epochs,
                "batch_size": self.training_config.phone_training.hyperparameters.batch_size,
                "image_size": self.training_config.phone_training.hyperparameters.image_size,
                "results_repr": str(results),
            }

            save_json(
                self.paths_config.metrics_dir
                / self.training_config.phone_training.outputs.metrics_file_name,
                summary,
            )

            dvc_meta = {
                "artifact_type": "phone_model",
                "data_yaml_path": str(data_yaml_path),
                "start_weights": str(model_path),
                "default_target_path": str(
                    self.paths_config.phone_weights_dir
                    / self.training_config.phone_training.outputs.save_best_model_as
                ),
            }
            save_yaml(
                self.paths_config.metrics_dir / "phone_dvc_manifest.yaml", dvc_meta
            )

        self.logger.info("Phone detector training completed.")
        return summary
