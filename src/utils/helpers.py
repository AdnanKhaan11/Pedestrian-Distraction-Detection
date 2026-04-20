import math
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch

from src.config.constants import SUPPORTED_IMAGE_SUFFIXES, SUPPORTED_VIDEO_SUFFIXES


def list_image_files(directory: Path) -> list[Path]:
    """
    List supported image files recursively.
    """
    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
    )


def list_video_files(directory: Path) -> list[Path]:
    """
    List supported video files recursively.
    """
    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
        ]
    )


def ensure_clean_directory(directory: Path) -> None:
    """
    Create directory if missing.
    Does not delete existing contents.
    """
    directory.mkdir(parents=True, exist_ok=True)


def split_list(
    items: list,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split a list into train/val/test parts.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    items = list(items)

    if shuffle:
        random.seed(seed)
        random.shuffle(items)

    total_count = len(items)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]

    return train_items, val_items, test_items


def load_image_rgb(image_path: Path, image_size: int | None = None) -> np.ndarray:
    """
    Load image as RGB numpy array.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_size is not None:
        image = cv2.resize(
            image, (image_size, image_size), interpolation=cv2.INTER_AREA
        )

    return image


def image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Convert RGB image to PyTorch tensor in CHW format.
    """
    image_float = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_float, (2, 0, 1))
    return torch.tensor(chw, dtype=torch.float32)


def normalize_pose_channels(x: np.ndarray) -> np.ndarray:
    """
    Normalize pose features channel by channel.

    Expected shape:
    (N, C, D, H, W)
    or a compatible 5D pose feature tensor.

    This logic is based on your current posture model code.
    """
    if x.ndim != 5:
        raise ValueError(f"Expected 5D pose tensor, got shape: {x.shape}")

    x = x.astype(np.float32).copy()

    for channel_index in range(x.shape[1]):
        channel = x[:, channel_index, :, :, :]
        mean_value = np.mean(channel)
        std_value = np.std(channel)

        if std_value < np.finfo(np.float32).eps:
            std_value = 1.0

        x[:, channel_index, :, :, :] = (channel - mean_value) / std_value

    return x


def calc_angle(edge_points: list[list[float]], mid_point: list[float]) -> float:
    """
    Calculate angle from two edge points and one middle point.

    This is the cleaned version of your current angle calculation logic.
    """
    p1, p2 = [np.array(point, dtype=np.float32) for point in edge_points]
    midpoint = np.array(mid_point, dtype=np.float32)

    radians = np.arctan2(p2[1] - midpoint[1], p2[0] - midpoint[0]) - np.arctan2(
        p1[1] - midpoint[1], p1[0] - midpoint[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)


def calc_keypoint_angle(
    landmarks: np.ndarray,
    name_to_index: dict[str, int],
    edge_keypoint_names: tuple[str, str],
    mid_keypoint_name: str,
) -> tuple[float, float]:
    """
    Calculate one angle and one angle-score using three keypoints.

    This keeps the same idea from your current project:
    - angle from coordinates
    - score from geometric relation of confidence values
    """
    name_1, name_2 = edge_keypoint_names
    name_mid = mid_keypoint_name

    if name_1 == "" and name_2 == "" and name_mid == "":
        return 0.0, 0.0

    idx_1 = name_to_index[name_1]
    idx_2 = name_to_index[name_2]
    idx_mid = name_to_index[name_mid]

    coord_1 = landmarks[idx_1][:2]
    coord_2 = landmarks[idx_2][:2]
    coord_mid = landmarks[idx_mid][:2]

    score_1 = landmarks[idx_1][2]
    score_2 = landmarks[idx_2][2]
    score_mid = landmarks[idx_mid][2]

    angle_score = float(np.cbrt(score_1 * score_2 * score_mid))
    angle_value = calc_angle([coord_1, coord_2], coord_mid)

    return angle_value, angle_score
