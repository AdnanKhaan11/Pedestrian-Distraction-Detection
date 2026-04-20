import re
from pathlib import Path


def parse_pose_data_filename(file_name: str, extension: str | None = None) -> dict:
    """
    Parse your posture dataset file naming convention.

    Expected style:
    20241114_1643_hyz_N_TC-Head_000_012.npy

    Extracted fields:
    - capture_date
    - capture_time
    - human_model_name
    - label
    - extensions
    - weight
    - frame_number
    """
    if extension is not None:
        file_name = file_name.replace(extension, "")

    parts = re.split(r"_", file_name)

    if len(parts) not in range(6, 8):
        raise ValueError(f"Invalid posture filename format: {file_name}")

    return {
        "capture_date": int(parts[0]),
        "capture_time": int(parts[1]),
        "human_model_name": str(parts[2]),
        "label": str(parts[3]),
        "extensions": str(parts[4]),
        "weight": float(parts[5][:1] + "." + parts[5][1:]),
        "frame_number": int(parts[6]) if len(parts) > 6 else None,
    }


def get_video_properties(video_name: str, item: str | None = None) -> dict | int | str:
    """
    Parse your phone-data video naming convention.

    Expected style without extension:
    20250512_1427_model_position_greeninfo_Lphone

    Your current project uses:
    - date
    - time
    - model name
    - position
    - color info
    - hand info

    Returns full dict or one selected item.
    """
    if video_name.endswith(".mp4"):
        raise ValueError("video_name must not include file extension")

    values = tuple(video_name.split("_"))
    keys = ["date", "time", "model name", "position", "color info", "hand info"]

    if len(values) < len(keys):
        raise ValueError(f"Invalid video filename format: {video_name}")

    properties = {key: value for key, value in zip(keys, values)}

    hand_prefix = properties["hand info"][0]
    if hand_prefix not in {"L", "R", "B"}:
        raise ValueError(f"Invalid hand code in video name: {video_name}")

    properties["hand index"] = {"L": 0, "R": 1, "B": 2}[hand_prefix]
    properties["hand item"] = properties["hand info"][1:]
    del properties["hand info"]

    if item is None:
        return properties

    valid_items = list(properties.keys()) + ["hex id"]
    if item not in valid_items:
        raise KeyError(f"Invalid item '{item}'. Supported: {valid_items}")

    if item == "hex id":
        return f"{properties['date'][:8]}_{properties['time']}"

    return properties[item]


def video_name_to_hand_index(video_name: str) -> int:
    """
    Small helper used in your current phone dataset gathering logic.

    Returns:
    - 0 for left hand
    - 1 for right hand
    - 2 for both hands
    """
    properties = tuple(video_name.split("_"))
    hand_prefix = properties[-1][0]

    if hand_prefix not in {"L", "R", "B"}:
        raise ValueError(f"Invalid hand code in video name: {video_name}")

    return {"L": 0, "R": 1, "B": 2}[hand_prefix]


def get_file_stem(path_like: str | Path) -> str:
    """
    Return file name without extension.
    """
    return Path(path_like).stem
