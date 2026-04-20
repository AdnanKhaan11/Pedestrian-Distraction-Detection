"""
This file runs posture classification using the cleaned posture model service.
It receives pose feature tensors and returns a posture decision in a simple format.
The goal is to keep posture prediction logic separate from MMPose and phone detection.
This mirrors your current project logic but in a modular and reusable way.
"""

from pathlib import Path

import numpy as np

from src.components.posture_model import PostureModelService
from src.entity.config_entity import PostureModelConfig
from src.utils.logger import get_logger


class PostureDetector:
    """
    Simple wrapper around the posture classifier.
    """

    def __init__(
        self,
        config: PostureModelConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.config = config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )
        self.model_service = PostureModelService(
            config=config,
            log_dir=log_dir,
            log_level=log_level,
        )

    def load(self) -> None:
        """
        Load posture model weights into memory.
        """
        self.model_service.load_model()

    def predict(self, feature_tensor: np.ndarray) -> dict:
        """
        Predict posture label from pose feature tensor.

        Expected input shape:
        (N, C, D, H, W)

        Returns:
        {
            "score_text": "0.91",
            "class_signal": 0 or 1,
            "label": "not_using" or "using_or_suspicious",
            "probabilities": [...]
        }
        """
        score_text, class_signal, probabilities = self.model_service.predict_numpy(
            feature_tensor
        )

        label = (
            self.config.inference.class_names[1]
            if class_signal == 1
            else self.config.inference.class_names[0]
        )

        result = {
            "score_text": score_text,
            "class_signal": class_signal,
            "label": label,
            "probabilities": probabilities.tolist(),
        }

        self.logger.info("Posture prediction result: %s", result)
        return result
