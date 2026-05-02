"""
ML Model Bridge for Backend.

This file is IMPORT BRIDGE ONLY.
- Does not contain ML logic
- Imports existing Predictor from src/
- Exposes load_pipeline() to be called at startup
"""

from dataclasses import replace
import sys
from pathlib import Path

# Add parent directory to path so src/ can be imported
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.serving.predictor import Predictor


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def load_pipeline() -> Predictor:
    """
    Load the ML inference pipeline.

    This function is called ONCE at application startup.
    The result is stored in app.state.pipeline and reused for all requests.

    Returns:
        Predictor: Ready-to-use inference pipeline (wraps InferencePipeline)

    Raises:
        Exception: If any model files are missing or invalid
    """
    pipeline = Predictor(log_level="INFO")

    resolved_mmpose_config = replace(
        pipeline.mmpose_config,
        detector=replace(
            pipeline.mmpose_config.detector,
            config_file=str(
                _resolve_project_path(pipeline.mmpose_config.detector.config_file)
            ),
            checkpoint_file=_resolve_project_path(
                pipeline.mmpose_config.detector.checkpoint_file
            ),
        ),
        pose_estimator=replace(
            pipeline.mmpose_config.pose_estimator,
            config_file=str(
                _resolve_project_path(pipeline.mmpose_config.pose_estimator.config_file)
            ),
            checkpoint_file=_resolve_project_path(
                pipeline.mmpose_config.pose_estimator.checkpoint_file
            ),
        ),
    )
    pipeline.mmpose_config = resolved_mmpose_config
    pipeline.pipeline.mmpose_config = resolved_mmpose_config
    pipeline.pipeline.mmpose_loader.config = resolved_mmpose_config
    pipeline.pipeline.runtime_detector.mmpose_config = resolved_mmpose_config

    resolved_phone_detector_config = replace(
        pipeline.phone_detector_config,
        weights=replace(
            pipeline.phone_detector_config.weights,
            default_weight_file=_resolve_project_path(
                pipeline.phone_detector_config.weights.default_weight_file
            ),
            fallback_weight_file=_resolve_project_path(
                pipeline.phone_detector_config.weights.fallback_weight_file
            ),
        ),
    )
    pipeline.phone_detector_config = resolved_phone_detector_config
    pipeline.pipeline.phone_detector_config = resolved_phone_detector_config
    pipeline.pipeline.runtime_detector.phone_detector_config = (
        resolved_phone_detector_config
    )
    pipeline.pipeline.runtime_detector.hand_cropper.phone_detector_config = (
        resolved_phone_detector_config
    )
    pipeline.pipeline.runtime_detector.phone_detector.config = (
        resolved_phone_detector_config
    )
    pipeline.pipeline.runtime_detector.phone_detector.loader.config = (
        resolved_phone_detector_config
    )

    return pipeline
