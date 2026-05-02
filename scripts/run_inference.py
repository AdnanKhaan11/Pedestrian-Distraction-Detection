"""
This script is the command-line entrypoint for full image inference.
It loads configs, initializes the serving predictor, and runs one prediction.
This is useful for quick manual testing before API integration.
It is also a clean reproducible entrypoint for your new MLOps project.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager
from src.serving.predictor import Predictor
from src.utils.common import create_directories
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pedestrian distraction inference on one image or video."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        default=None,
        help="Path to input image",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=False,
        default=None,
        help="Path to input video",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Process every Nth frame from video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image is None and args.video is None:
        print("Error: provide --image or --video")
        sys.exit(1)

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

    predictor = Predictor(log_dir=paths_config.logs_dir, log_level="INFO")

    if args.image:
        logger.info("Starting inference on image: %s", args.image)
        result = predictor.predict_image(Path(args.image), save_rendered_output=True)
        logger.info("Inference completed successfully.")
        logger.info("Prediction result: %s", result)
        print(result)

    elif args.video:
        logger.info("Starting inference on video: %s", args.video)
        logger.info("Frame step: %s", args.frame_step)
        result = predictor.predict_video(
            Path(args.video),
            frame_step=args.frame_step,
            save_rendered_output=True,
        )
        logger.info("Video inference completed.")
        logger.info("Total frames processed: %s", result["total_frames_processed"])
        print(result)


if __name__ == "__main__":
    main()
