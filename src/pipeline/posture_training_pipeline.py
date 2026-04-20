"""
This file runs the full posture training workflow in the correct order.
It connects posture training and posture evaluation into one clean pipeline.
This is the production-style replacement for the old training script flow.
It is designed to be simple, reproducible, and easy to test step by step later.
"""

from pathlib import Path

from src.components.posture_evaluator import PostureEvaluator
from src.components.posture_trainer import PostureTrainer
from src.entity.config_entity import PathsConfig, PostureModelConfig, TrainingConfig
from src.utils.logger import get_logger


class PostureTrainingPipeline:
    """
    End-to-end pipeline for posture model training and evaluation.
    """

    def __init__(
        self,
        paths_config: PathsConfig,
        posture_model_config: PostureModelConfig,
        training_config: TrainingConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.paths_config = paths_config
        self.posture_model_config = posture_model_config
        self.training_config = training_config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

        self.trainer = PostureTrainer(
            paths_config=paths_config,
            posture_model_config=posture_model_config,
            training_config=training_config,
            log_dir=log_dir,
            log_level=log_level,
        )

        self.evaluator = PostureEvaluator(
            paths_config=paths_config,
            training_config=training_config,
            log_dir=log_dir,
            log_level=log_level,
        )

    def run(self) -> dict:
        """
        Train first, then evaluate on test data.
        """
        self.logger.info("Starting posture training pipeline.")

        training_output = self.trainer.train()

        model = training_output["model"]
        test_loader = training_output["test_loader"]

        evaluation_metrics = self.evaluator.evaluate(
            model=model,
            test_loader=test_loader,
            device=self.trainer.device,
        )

        result = {
            "training_history": training_output["history"],
            "best_val_loss": training_output["best_val_loss"],
            "evaluation_metrics": evaluation_metrics,
        }

        self.logger.info("Posture training pipeline completed.")
        return result
