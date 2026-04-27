"""
This file loads the YOLO phone detection model for runtime inference.
It supports both your trained phone detector and the fallback pretrained YOLO model.
The class keeps model-loading logic separate from prediction logic.
This makes the project cleaner, easier to debug, and easier to track in MLOps pipelines.
"""

from pathlib import Path

from src.entity.config_entity import PhoneDetectorConfig
from src.utils.logger import get_logger


class PhoneModelLoader:
    """
    Load and manage YOLO phone detection models.

    Supported modes:
    - trained custom phone detector
    - fallback pretrained detector
    """

    def __init__(
        self,
        config: PhoneDetectorConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.config = config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

        self._trained_model = None
        self._fallback_model = None

    def _import_yolo(self):
        """
        Import YOLO only when needed.
        """
        try:
            from ultralytics import YOLO

            return YOLO
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Unable to import Ultralytics YOLO. Ensure the package is installed "
                "and the environment has permission to access Ultralytics settings."
            ) from exc

    def load_trained_model(self):
        """
        Load your trained phone detector model.
        """
        model_path = Path(self.config.weights.default_weight_file)

        if not model_path.exists():
            raise FileNotFoundError(f"Trained phone detector not found: {model_path}")

        YOLO = self._import_yolo()
        self._trained_model = YOLO(str(model_path))

        self.logger.info("Loaded trained phone detector from: %s", model_path)
        return self._trained_model

    def load_fallback_model(self):
        """
        Load the fallback pretrained YOLO model.
        """
        model_path = Path(self.config.weights.fallback_weight_file)

        if not model_path.exists():
            raise FileNotFoundError(f"Fallback phone detector not found: {model_path}")

        YOLO = self._import_yolo()
        self._fallback_model = YOLO(str(model_path))

        self.logger.info("Loaded fallback phone detector from: %s", model_path)
        return self._fallback_model

    def get_model(self, use_trained: bool = True):
        """
        Return the requested model and load it if needed.
        """
        if use_trained:
            if self._trained_model is None:
                self.load_trained_model()
            return self._trained_model

        if self._fallback_model is None:
            self.load_fallback_model()
        return self._fallback_model
