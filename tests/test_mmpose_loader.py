import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.components.mmpose_loader import MMPoseLoader  # noqa: E402
from src.config.configuration import ConfigurationManager  # noqa: E402


def _build_loader(
    detector_config_file: Path,
    detector_checkpoint_file: Path,
    pose_config_file: Path,
    pose_checkpoint_file: Path,
) -> MMPoseLoader:
    manager = ConfigurationManager()
    config = manager.get_mmpose_config()
    detector = replace(
        config.detector,
        config_file=str(detector_config_file),
        checkpoint_file=detector_checkpoint_file,
    )
    pose_estimator = replace(
        config.pose_estimator,
        config_file=str(pose_config_file),
        checkpoint_file=pose_checkpoint_file,
    )
    config = replace(config, detector=detector, pose_estimator=pose_estimator)
    return MMPoseLoader(config)


def test_mmpose_loader_validates_required_files(tmp_path: Path):
    loader = _build_loader(
        detector_config_file=tmp_path / "missing_detector.py",
        detector_checkpoint_file=tmp_path / "missing_detector.pth",
        pose_config_file=tmp_path / "missing_pose.py",
        pose_checkpoint_file=tmp_path / "missing_pose.pth",
    )

    with pytest.raises(FileNotFoundError, match="Person detector config file not found"):
        loader.load()


def test_mmpose_loader_properties_require_load():
    manager = ConfigurationManager()
    loader = MMPoseLoader(manager.get_mmpose_config())

    with pytest.raises(RuntimeError, match="BBox detector is not loaded yet"):
        _ = loader.bbox_detector

    with pytest.raises(RuntimeError, match="Pose estimator is not loaded yet"):
        _ = loader.pose_estimator

    with pytest.raises(RuntimeError, match="Visualizer is not loaded yet"):
        _ = loader.visualizer


def test_mmpose_loader_runtime_bundle_uses_loaded_objects(tmp_path: Path):
    detector_config_file = tmp_path / "detector.py"
    detector_checkpoint_file = tmp_path / "detector.pth"
    pose_config_file = tmp_path / "pose.py"
    pose_checkpoint_file = tmp_path / "pose.pth"

    detector_config_file.write_text("# detector config", encoding="utf-8")
    detector_checkpoint_file.write_bytes(b"detector")
    pose_config_file.write_text("# pose config", encoding="utf-8")
    pose_checkpoint_file.write_bytes(b"pose")

    loader = _build_loader(
        detector_config_file=detector_config_file,
        detector_checkpoint_file=detector_checkpoint_file,
        pose_config_file=pose_config_file,
        pose_checkpoint_file=pose_checkpoint_file,
    )

    loader._bbox_detector = object()
    loader._pose_estimator = object()
    loader._visualizer = object()

    bundle = loader.get_runtime_bundle()

    assert bundle["bbox_detector_model"] is loader._bbox_detector
    assert bundle["pose_estimator_model"] is loader._pose_estimator
    assert bundle["visualizer"] is loader._visualizer
    assert bundle["device"] == loader.device


def test_mmpose_loader_wraps_import_failures(tmp_path: Path):
    detector_config_file = tmp_path / "detector.py"
    detector_checkpoint_file = tmp_path / "detector.pth"
    pose_config_file = tmp_path / "pose.py"
    pose_checkpoint_file = tmp_path / "pose.pth"

    detector_config_file.write_text("# detector config", encoding="utf-8")
    detector_checkpoint_file.write_bytes(b"detector")
    pose_config_file.write_text("# pose config", encoding="utf-8")
    pose_checkpoint_file.write_bytes(b"pose")

    loader = _build_loader(
        detector_config_file=detector_config_file,
        detector_checkpoint_file=detector_checkpoint_file,
        pose_config_file=pose_config_file,
        pose_checkpoint_file=pose_checkpoint_file,
    )

    original_import = __import__

    def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"mmdet.apis", "mmpose.apis", "mmpose.registry", "mmpose.utils"}:
            raise PermissionError(name)
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_mock_import):
        with pytest.raises(RuntimeError, match="Unable to import MMPose/MMDetection"):
            loader.load()
