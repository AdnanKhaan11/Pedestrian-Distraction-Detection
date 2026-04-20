import itertools
from pathlib import Path
from typing import Any

import numpy as np

from src.entity.config_entity import MMPoseConfig, PostureModelConfig
from src.utils.helpers import calc_keypoint_angle, normalize_pose_channels
from src.utils.logger import get_logger


class PoseFeatureGenerator:
    """
    Convert raw keypoints from MMPose into the structured angle-score tensor
    used by your posture classifier.

    This is the cleaned version of the logic from:
    - keypoint_config.py
    - calculations.py
    - annotate_image.py
    - processing.py
    """

    def __init__(
        self,
        mmpose_config: MMPoseConfig,
        posture_model_config: PostureModelConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.mmpose_config = mmpose_config
        self.posture_model_config = posture_model_config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

        # These names match the first 13 keypoints used in your current project logic
        self.keypoint_names = {
            0: "Body-Chin",
            1: "Body-Left_eye",
            2: "Body-Right_eye",
            3: "Body-Left_ear",
            4: "Body-Right_ear",
            5: "Body-Left_shoulder",
            6: "Body-Right_shoulder",
            7: "Body-Left_elbow",
            8: "Body-Right_elbow",
            9: "Body-Left_wrist",
            10: "Body-Right_wrist",
            11: "Body-Left_hip",
            12: "Body-Right_hip",
        }

        self.keypoint_indexes = {name: idx for idx, name in self.keypoint_names.items()}
        self.target_structure = self._get_cube_angles(
            use_str=True, num=self.mmpose_config.keypoints.use_first_n_keypoints
        )

    def _get_cube_angles(
        self, use_str: bool = True, num: int = 13
    ) -> list[list[list[Any]]]:
        """
        Rebuild the angle cube structure from the old keypoint_config logic.
        """
        key_source = (
            self.keypoint_indexes.keys() if use_str else self.keypoint_indexes.values()
        )
        keys = list(key_source)[:num]
        edge_combinations = list(itertools.combinations(keys, 2))
        sorted_angles = [
            [edge, corner]
            for corner in keys
            for edge in edge_combinations
            if corner not in edge
        ]

        row = num - 1
        col = num - 2
        depth = (num + 1) // 2

        init_value = [("", ""), ""] if use_str else [(0, 0), 0]
        cube = [
            [[init_value for _ in range(depth)] for _ in range(col)] for _ in range(row)
        ]

        ij_order = [(i, j) for i in range(row) for j in range(i, col)]
        ij_order += [(i, j) for j in range(col) for i in range(j + 1, row)]

        idx = 0
        for k in range(depth):
            for i, j in ij_order:
                cube[i][j][k] = sorted_angles[idx]
                idx = (idx + 1) % len(sorted_angles)

        return cube

    def translate_one_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Translate one person's landmarks into 2-channel 3D feature tensor.

        Output shape:
        (channels=2, height=12, width=11, depth=7)

        Channel 0:
        - angle values

        Channel 1:
        - angle scores
        """
        num = self.mmpose_config.keypoints.use_first_n_keypoints
        row = num - 1
        col = num - 2
        depth = (num + 1) // 2

        angle_channel = np.zeros((row, col, depth), dtype=np.float32)
        score_channel = np.zeros((row, col, depth), dtype=np.float32)

        for i in range(row):
            for j in range(col):
                for k in range(depth):
                    edge_names, middle_name = self.target_structure[i][j][k]
                    angle_value, angle_score = calc_keypoint_angle(
                        landmarks=landmarks,
                        name_to_index=self.keypoint_indexes,
                        edge_keypoint_names=tuple(edge_names),
                        mid_keypoint_name=middle_name,
                    )
                    angle_channel[i, j, k] = angle_value
                    score_channel[i, j, k] = angle_score

        return np.stack([angle_channel, score_channel], axis=0)

    def build_feature_tensor(
        self, landmarks: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        """
        Convert one person's landmarks into model-ready tensor.

        Final output shape after transpose:
        (1, C, D, H, W)
        """
        feature_tensor = self.translate_one_landmarks(landmarks)  # (C, H, W, D)
        feature_tensor = np.expand_dims(feature_tensor, axis=0)  # (N, C, H, W, D)

        if (
            self.posture_model_config.feature_engineering.normalize_per_channel
            and normalize
        ):
            # Convert to N, C, D, H, W before channel normalization logic
            feature_tensor = np.transpose(feature_tensor, (0, 1, 4, 2, 3))
            feature_tensor = normalize_pose_channels(feature_tensor)
        else:
            feature_tensor = np.transpose(feature_tensor, (0, 1, 4, 2, 3))

        return feature_tensor.astype(np.float32)

    def build_batch_feature_tensor(
        self, landmarks_batch: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        """
        Convert many persons into model-ready pose tensor batch.

        Input:
        - (N, K, 3)

        Output:
        - (N, C, D, H, W)
        """
        batch_features = [
            self.translate_one_landmarks(one_person) for one_person in landmarks_batch
        ]
        batch_features = np.stack(batch_features, axis=0)  # (N, C, H, W, D)
        batch_features = np.transpose(
            batch_features, (0, 1, 4, 2, 3)
        )  # (N, C, D, H, W)

        if (
            self.posture_model_config.feature_engineering.normalize_per_channel
            and normalize
        ):
            batch_features = normalize_pose_channels(batch_features)

        return batch_features.astype(np.float32)
