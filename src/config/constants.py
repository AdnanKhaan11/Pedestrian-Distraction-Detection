# What this file does
# This file stores shared project constants.

# It includes:

# path constants
# environment variable names
# common file names
# runtime state constants from your current project logic
# So later code does not need magic numbers or hardcoded strings


from pathlib import Path

# Root of the new MLOps project
ROOT_DIR = Path(__file__).resolve().parents[2]

# Config directory
CONFIGS_DIR = ROOT_DIR / "configs"

# YAML config files
CONFIG_FILE_PATH = CONFIGS_DIR / "config.yaml"
PATHS_FILE_PATH = CONFIGS_DIR / "paths.yaml"
MMPOSE_CONFIG_FILE_PATH = CONFIGS_DIR / "mmpose_config.yaml"
POSTURE_MODEL_CONFIG_FILE_PATH = CONFIGS_DIR / "posture_model_config.yaml"
PHONE_DETECTOR_CONFIG_FILE_PATH = CONFIGS_DIR / "phone_detector_config.yaml"
TRAINING_CONFIG_FILE_PATH = CONFIGS_DIR / "training_config.yaml"
INFERENCE_CONFIG_FILE_PATH = CONFIGS_DIR / "inference_config.yaml"

# Default environment variable names
ENV_DAGSHUB_USERNAME = "DAGSHUB_USERNAME"
ENV_DAGSHUB_TOKEN = "DAGSHUB_TOKEN"
ENV_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"

# Common file names
DEFAULT_LOG_FILE_NAME = "running_logs.log"
DEFAULT_POSTURE_BEST_MODEL_NAME = "best_posture_model.pth"
DEFAULT_PHONE_BEST_MODEL_NAME = "best.pt"

# Supported image suffixes
SUPPORTED_IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Supported video suffixes
SUPPORTED_VIDEO_SUFFIXES = [".mp4", ".avi", ".mov", ".mkv"]

# State labels used in runtime logic
STATE_TO_BE_CLASSIFIED = 0
STATE_OUT_OF_FRAME = 1
STATE_BACKSIDE = 2
STATE_SUSPICIOUS = 4
STATE_USING = 32
STATE_NOT_USING = 128

STATE_DISPLAY = {
    STATE_TO_BE_CLASSIFIED: {"color": "gray", "text": "-"},
    STATE_OUT_OF_FRAME: {"color": "gray", "text": "Out of frame"},
    STATE_BACKSIDE: {"color": "gray", "text": "Back"},
    STATE_SUSPICIOUS: {"color": "orange", "text": "?"},
    STATE_USING: {"color": "pink", "text": "+"},
    STATE_NOT_USING: {"color": "green", "text": "-"},
}
