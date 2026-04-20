from pathlib import Path

import numpy as np

from src.entity.config_entity import MMPoseConfig, PhoneDetectorConfig
from src.utils.logger import get_logger
from src.utils.opencv_utils import crop_frame


class HandCropper:
    """
    Crop hand regions using pose keypoints.

    This is the cleaned version of the hand-cropping logic used in:
    - processing.py
    - yolo11_data_gather.py

    It supports:
    - left hand crop
    - right hand crop
    - merged hand crop when hands are close
    - primary/secondary hand selection based on distance to face
    """

    def __init__(
        self,
        mmpose_config: MMPoseConfig,
        phone_detector_config: PhoneDetectorConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.mmpose_config = mmpose_config
        self.phone_detector_config = phone_detector_config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

    def _extract_points(self, keypoints: np.ndarray) -> dict:
        """
        Read important landmark points from one person keypoints array.
        """
        kp_cfg = self.mmpose_config.keypoints

        return {
            "face_center": keypoints[kp_cfg.face_center_index][:2],
            "left_elbow": keypoints[kp_cfg.left_elbow_index][:2],
            "right_elbow": keypoints[kp_cfg.right_elbow_index][:2],
            "left_wrist": keypoints[kp_cfg.left_wrist_index][:2],
            "right_wrist": keypoints[kp_cfg.right_wrist_index][:2],
        }

    def _compute_hand_hw(self, frame: np.ndarray, xyxy: np.ndarray) -> tuple[int, int]:
        """
        Decide hand crop size based on person bbox size.
        """
        bbox_width = abs(xyxy[2] - xyxy[0])

        if bbox_width / (frame.shape[1] + np.finfo(np.float32).eps) < 0.6:
            edge = int(
                bbox_width
                * self.phone_detector_config.hand_crop_logic.far_body_hand_ratio
            )
        else:
            edge = int(
                frame.shape[1]
                * self.phone_detector_config.hand_crop_logic.near_body_hand_ratio
            )

        return edge, edge

    def compute_hand_centers(
        self, keypoints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate hand centers using elbow-to-wrist direction.
        """
        points = self._extract_points(keypoints)

        left_arm_vector = points["left_wrist"] - points["left_elbow"]
        right_arm_vector = points["right_wrist"] - points["right_elbow"]

        extension_ratio = (
            self.phone_detector_config.hand_crop_logic.hand_extension_ratio
        )

        left_hand_center = points["left_wrist"] + left_arm_vector * extension_ratio
        right_hand_center = points["right_wrist"] + right_arm_vector * extension_ratio

        return left_hand_center, right_hand_center

    def crop_candidate_hands(
        self, frame: np.ndarray, keypoints: np.ndarray, xyxy: np.ndarray
    ) -> dict:
        """
        Generate candidate hand crops.

        Returns a dictionary that contains:
        - left crop
        - right crop
        - merged crop if hands are close
        """
        hand_hw = self._compute_hand_hw(frame, xyxy)
        bbox_width = abs(xyxy[2] - xyxy[0])

        left_hand_center, right_hand_center = self.compute_hand_centers(keypoints)

        distance_between_hands = np.linalg.norm(left_hand_center - right_hand_center)
        merge_ratio_threshold = (
            self.phone_detector_config.hand_crop_logic.merge_hand_distance_ratio
        )

        result = {
            "left": None,
            "right": None,
            "merged": None,
            "hand_hw": hand_hw,
        }

        if distance_between_hands > merge_ratio_threshold * bbox_width:
            result["left"] = crop_frame(frame, left_hand_center, hand_hw)
            result["right"] = crop_frame(frame, right_hand_center, hand_hw)
        else:
            merged_center = (left_hand_center + right_hand_center) // 2
            result["merged"] = crop_frame(frame, merged_center, hand_hw)

        return result

    def get_priority_hand_crops(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        xyxy: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, list[int]] | None, tuple[np.ndarray, list[int]] | None, float
    ]:
        """
        Return:
        - primary hand crop
        - secondary hand crop
        - spare distance ratio

        This follows your current runtime logic:
        choose the hand closer to the face first.
        """
        points = self._extract_points(keypoints)
        face_center = points["face_center"]

        hand_crops = self.crop_candidate_hands(frame, keypoints, xyxy)

        if hand_crops["merged"] is not None:
            return hand_crops["merged"], None, 1.0

        left_crop = hand_crops["left"]
        right_crop = hand_crops["right"]

        if left_crop is None and right_crop is None:
            return None, None, 1.0

        left_wrist = points["left_wrist"]
        right_wrist = points["right_wrist"]

        left_face_distance = np.linalg.norm(left_wrist - face_center)
        right_face_distance = np.linalg.norm(right_wrist - face_center)

        try:
            spare_ratio = (
                left_face_distance / (right_face_distance + np.finfo(np.float32).eps)
                if left_face_distance <= right_face_distance
                else right_face_distance
                / (left_face_distance + np.finfo(np.float32).eps)
            )
        except Exception:
            spare_ratio = 1.0

        if left_face_distance < right_face_distance:
            primary_crop = left_crop
            secondary_crop = right_crop
        else:
            primary_crop = right_crop
            secondary_crop = left_crop

        return primary_crop, secondary_crop, float(spare_ratio)
