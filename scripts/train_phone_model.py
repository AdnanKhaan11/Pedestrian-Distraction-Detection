"""
This script is the command-line entrypoint for phone detector training.
It loads configs, starts the phone training pipeline, and logs progress cleanly.
This is better than using one large experimental script and fits MLOps practice.
It is designed to work with MLflow, DAGsHub, and later DVC-based artifact tracking.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager
from src.pipeline.phone_training_pipeline import PhoneTrainingPipeline
from src.utils.common import create_directories, seed_everything
from src.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the phone detector model.")
    parser.add_argument(
        "--data-yaml",
        type=str,
        required=False,
        default="data/processed/phone_dataset/data.yaml",
        help="Path to YOLO dataset yaml file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_manager = ConfigurationManager()

    project_config = config_manager.get_project_config()
    paths_config = config_manager.get_paths_config()
    phone_detector_config = config_manager.get_phone_detector_config()
    training_config = config_manager.get_training_config()

    create_directories(
        [
            paths_config.logs_dir,
            paths_config.metrics_dir,
            paths_config.phone_weights_dir,
        ]
    )

    seed_everything(project_config.random_seed)
    logger = get_logger(
        "train_phone_model", log_dir=paths_config.logs_dir, level="INFO"
    )

    data_yaml_path = Path(args.data_yaml)
    logger.info("Starting phone training script.")
    logger.info("Using data yaml: %s", data_yaml_path)

    pipeline = PhoneTrainingPipeline(
        paths_config=paths_config,
        phone_detector_config=phone_detector_config,
        training_config=training_config,
        log_dir=paths_config.logs_dir,
        log_level="INFO",
    )

    result = pipeline.run(data_yaml_path=data_yaml_path)

    logger.info("Phone training script completed successfully.")
    logger.info("Final result summary: %s", result)


if __name__ == "__main__":
    main()
