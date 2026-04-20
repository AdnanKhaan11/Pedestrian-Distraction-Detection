"""
This script is the command-line entrypoint for full image inference.
It loads configs, initializes the serving predictor, and runs one prediction.
This is useful for quick manual testing before API integration.
It is also a clean reproducible entrypoint for your new MLOps project.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager
from src.serving.predictor import Predictor
from src.utils.common import create_directories
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pedestrian distraction inference on one image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_manager = ConfigurationManager()
    paths_config = config_manager.get_paths_config()

    create_directories(
        [
            paths_config.logs_dir,
            paths_config.frontend_result_dir,
            paths_config.metrics_dir,
        ]
    )

    logger = get_logger("run_inference", log_dir=paths_config.logs_dir, level="INFO")
    logger.info("Starting inference script for image: %s", args.image)

    predictor = Predictor(log_dir=paths_config.logs_dir, log_level="INFO")
    result = predictor.predict_image(Path(args.image), save_rendered_output=True)

    logger.info("Inference completed successfully.")
    logger.info("Prediction result: %s", result)

    print(result)


if __name__ == "__main__":
    main()
