"""
This script is the command-line entrypoint for posture model training.
It loads all configs, starts the posture training pipeline, and prints final output.
This is cleaner than keeping training logic inside notebooks or mixed scripts.
It is also better for MLflow, DAGsHub tracking, and repeatable execution.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager
from src.pipeline.posture_training_pipeline import PostureTrainingPipeline
from src.utils.common import create_directories, seed_everything
from src.utils.logger import get_logger


def main() -> None:
    config_manager = ConfigurationManager()

    project_config = config_manager.get_project_config()
    paths_config = config_manager.get_paths_config()
    posture_model_config = config_manager.get_posture_model_config()
    training_config = config_manager.get_training_config()

    create_directories(
        [
            paths_config.logs_dir,
            paths_config.metrics_dir,
            paths_config.posture_weights_dir,
        ]
    )

    seed_everything(project_config.random_seed)
    logger = get_logger(
        "train_posture_model", log_dir=paths_config.logs_dir, level="INFO"
    )

    logger.info("Starting posture training script.")
    logger.info("MLflow tracking URI will be read from config or environment variable.")

    pipeline = PostureTrainingPipeline(
        paths_config=paths_config,
        posture_model_config=posture_model_config,
        training_config=training_config,
        log_dir=paths_config.logs_dir,
        log_level="INFO",
    )

    result = pipeline.run()

    logger.info("Posture training script completed successfully.")
    logger.info("Final result summary: %s", result)


if __name__ == "__main__":
    main()
