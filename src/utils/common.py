import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def read_yaml(path_to_yaml: Path) -> dict:
    """
    Read a YAML file and return its data as a dictionary.
    """
    if not path_to_yaml.exists():
        raise FileNotFoundError(f"YAML file not found: {path_to_yaml}")

    with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)

    if data is None:
        raise ValueError(f"YAML file is empty: {path_to_yaml}")

    return data


def save_yaml(path_to_yaml: Path, data: dict) -> None:
    """
    Save dictionary data to a YAML file.
    """
    path_to_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_yaml, "w", encoding="utf-8") as yaml_file:
        yaml.safe_dump(data, yaml_file, sort_keys=False)


def create_directories(paths: list[Path]) -> None:
    """
    Create many directories safely.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """
    Save dictionary data to JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """
    Load dictionary data from JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_text(path: Path, content: str) -> None:
    """
    Save plain text to a file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as text_file:
        text_file.write(content)


def seed_everything(seed: int) -> None:
    """
    Make training and data processing more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_preference: str = "auto") -> str:
    """
    Resolve device safely for CPU/GPU environments.

    Supported values:
    - auto
    - cpu
    - cuda
    """
    device_preference = device_preference.lower().strip()

    if device_preference == "cpu":
        return "cpu"

    if device_preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    raise ValueError(f"Unsupported device preference: {device_preference}")
