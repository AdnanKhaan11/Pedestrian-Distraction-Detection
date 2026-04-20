"""
This file runs phone detection on hand crops using YOLO.
It supports both your custom trained model and the fallback pretrained model.
The file only handles phone prediction logic, not posture logic or pose extraction.
This keeps runtime inference clean and matches your original project behavior.
"""

from pathlib import Path

import numpy as np
import torch

from src.components.phone_model_loader import PhoneModelLoader
from src.entity.config_entity import PhoneDetectorConfig
from src.utils.common import resolve_device
from src.utils.logger import get_logger


class PhoneDetector:
    """
    Detect phone presence inside a cropped hand frame.
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

        self.device = resolve_device("auto")
        self.loader = PhoneModelLoader(
            config=config, log_dir=log_dir, log_level=log_level
        )

    def load(self, use_trained: bool = True):
        """
        Load requested YOLO model.
        """
        return self.loader.get_model(use_trained=use_trained)

    def detect_phone(
        self,
        model,
        frame_rgb: np.ndarray,
        use_trained: bool = True,
    ) -> tuple[str, np.ndarray | None]:
        """
        Run phone detection on one RGB hand crop.

        Returns:
        - detection text
        - relative xyxy phone box or None
        """
        if frame_rgb is None:
            return "", None

        if frame_rgb.ndim != 3:
            raise ValueError(
                f"Expected RGB frame with 3 dims, got shape: {frame_rgb.shape}"
            )

        tensor_frame = torch.from_numpy(frame_rgb).float() / 255.0
        tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0).to(self.device)

        results = model(tensor_frame)
        first_result = results[0]

        result_classes = first_result.boxes.cls.cpu().numpy().astype(np.int32)

        phone_class_index = (
            self.config.inference.trained_model_phone_class_index
            if use_trained
            else self.config.inference.fallback_model_phone_class_index
        )

        confidence_threshold = (
            self.config.inference.confidence_threshold_trained
            if use_trained
            else self.config.inference.confidence_threshold_fallback
        )

        if not np.any(result_classes == phone_class_index):
            return "", None

        confidences = (
            first_result.boxes.conf.cpu()
            .numpy()
            .astype(np.float32)[result_classes == phone_class_index]
        )
        boxes = (
            first_result.boxes.data.cpu()
            .numpy()
            .astype(np.float32)[result_classes == phone_class_index]
        )

        max_conf_index = int(np.argmax(confidences))
        best_confidence = float(confidences[max_conf_index])

        if best_confidence < confidence_threshold:
            return "", None

        detection_text = f"Phone: {best_confidence:.3f}"
        relative_xyxy = boxes[max_conf_index][:4]

        return detection_text, relative_xyxy

    def predict(self, frame_rgb: np.ndarray, use_trained: bool = True) -> dict:
        """
        Full phone detection entrypoint for one hand crop.

        Returns:
        {
            "detected": bool,
            "text": str,
            "relative_xyxy": list or None,
            "used_trained_model": bool
        }
        """
        model = self.load(use_trained=use_trained)
        detection_text, relative_xyxy = self.detect_phone(
            model=model,
            frame_rgb=frame_rgb,
            use_trained=use_trained,
        )

        result = {
            "detected": relative_xyxy is not None,
            "text": detection_text,
            "relative_xyxy": (
                relative_xyxy.tolist() if relative_xyxy is not None else None
            ),
            "used_trained_model": use_trained,
        }

        self.logger.info("Phone detection result: %s", result)
        return result
