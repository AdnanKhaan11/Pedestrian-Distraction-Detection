"""
This file provides a clean serving wrapper around the inference pipeline.
It is designed for backend usage, especially inside FastAPI endpoints later.
The predictor loads configs, initializes the pipeline, and returns structured results.
This keeps API code small and avoids mixing business logic inside route functions.
"""

from pathlib import Path

import cv2

from src.config.configuration import ConfigurationManager
from src.pipeline.inference_pipeline import InferencePipeline
from src.utils.common import create_directories, save_json
from src.utils.logger import get_logger


class Predictor:
    """
    Service wrapper used by backend/API code.
    """

    def __init__(self, log_dir: Path | None = None, log_level: str = "INFO") -> None:
        self.config_manager = ConfigurationManager()

        self.paths_config = self.config_manager.get_paths_config()
        self.mmpose_config = self.config_manager.get_mmpose_config()
        self.posture_model_config = self.config_manager.get_posture_model_config()
        self.phone_detector_config = self.config_manager.get_phone_detector_config()
        self.inference_config = self.config_manager.get_inference_config()

        self.logger = get_logger(
            self.__class__.__name__,
            log_dir=log_dir or self.paths_config.logs_dir,
            level=log_level,
        )

        create_directories(
            [
                self.paths_config.predictions_dir,
                self.paths_config.frontend_result_dir,
                self.paths_config.metrics_dir,
            ]
        )

        self.pipeline = InferencePipeline(
            mmpose_config=self.mmpose_config,
            posture_model_config=self.posture_model_config,
            phone_detector_config=self.phone_detector_config,
            inference_config=self.inference_config,
            log_dir=log_dir or self.paths_config.logs_dir,
            log_level=log_level,
        )

    def predict_image(
        self, image_path: Path, save_rendered_output: bool = True
    ) -> dict:
        """
        Run full inference on one image and optionally save rendered output.
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = self.pipeline.run_on_frame(frame=frame, draw_visualizer=False)

        saved_result_path = None
        if save_rendered_output:
            output_name = f"pred_{image_path.stem}.jpg"
            output_path = self.paths_config.frontend_result_dir / output_name
            save_ok = cv2.imwrite(str(output_path), result["frame"])
            if not save_ok:
                raise IOError(f"Could not save rendered output image: {output_path}")
            saved_result_path = str(output_path)

        response = {
            "num_persons": result["num_persons"],
            "person_results": [
                {
                    "posture": person_result["posture"],
                    "phone": person_result["phone"],
                    "state": person_result["state"],
                    "display_text": person_result["display_text"],
                    "score_text": person_result["score_text"],
                }
                for person_result in result["person_results"]
            ],
            "saved_result_path": saved_result_path,
        }

        save_json(self.paths_config.metrics_dir / "latest_prediction.json", response)
        self.logger.info("Prediction response: %s", response)
        return response
