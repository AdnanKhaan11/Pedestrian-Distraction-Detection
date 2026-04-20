"""
This file runs the full phone-detector training workflow in one clean pipeline.
It connects phone model training and phone model evaluation in the correct order.
This is the production-style replacement for your old YOLO training script flow.
It is simple to run, easier to debug, and better for MLOps tracking.
"""

from pathlib import Path

from src.components.phone_evaluator import PhoneEvaluator
from src.components.phone_trainer import PhoneTrainer
from src.entity.config_entity import PathsConfig, PhoneDetectorConfig, TrainingConfig
from src.utils.logger import get_logger


class PhoneTrainingPipeline:
    """
    End-to-end pipeline for phone detector training and evaluation.
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

        self.trainer = PhoneTrainer(
            paths_config=paths_config,
            phone_detector_config=phone_detector_config,
            training_config=training_config,
            log_dir=log_dir,
            log_level=log_level,
        )

        self.evaluator = PhoneEvaluator(
            paths_config=paths_config,
            phone_detector_config=phone_detector_config,
            training_config=training_config,
            log_dir=log_dir,
            log_level=log_level,
        )

    def run(self, data_yaml_path: Path) -> dict:
        """
        Train phone detector first, then evaluate it.
        """
        self.logger.info(
            "Starting phone training pipeline with data yaml: %s", data_yaml_path
        )

        training_summary = self.trainer.train(data_yaml_path=data_yaml_path)
        evaluation_summary = self.evaluator.evaluate(data_yaml_path=data_yaml_path)

        result = {
            "training_summary": training_summary,
            "evaluation_summary": evaluation_summary,
        }

        self.logger.info("Phone training pipeline completed.")
        return result
