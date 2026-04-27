import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager  # noqa: E402
from src.components.phone_model_loader import PhoneModelLoader  # noqa: E402


class DummyYOLO:
    calls: list[str] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.__class__.calls.append(model_path)


def _build_loader_with_paths(
    trained_path: Path,
    fallback_path: Path,
) -> PhoneModelLoader:
    manager = ConfigurationManager()
    config = manager.get_phone_detector_config()
    updated_weights = replace(
        config.weights,
        default_weight_file=trained_path,
        fallback_weight_file=fallback_path,
    )
    updated_config = replace(config, weights=updated_weights)
    return PhoneModelLoader(updated_config)


def test_phone_model_loader_caches_models(tmp_path: Path):
    DummyYOLO.calls = []
    trained_path = tmp_path / "trained.pt"
    fallback_path = tmp_path / "fallback.pt"
    trained_path.write_bytes(b"trained")
    fallback_path.write_bytes(b"fallback")

    loader = _build_loader_with_paths(
        trained_path=trained_path,
        fallback_path=fallback_path,
    )
    loader._import_yolo = lambda: DummyYOLO

    trained_model_1 = loader.get_model(use_trained=True)
    trained_model_2 = loader.get_model(use_trained=True)
    fallback_model_1 = loader.get_model(use_trained=False)
    fallback_model_2 = loader.get_model(use_trained=False)

    assert trained_model_1 is trained_model_2
    assert fallback_model_1 is fallback_model_2
    assert DummyYOLO.calls == [str(trained_path), str(fallback_path)]


def test_phone_model_loader_raises_for_missing_weight_files(tmp_path: Path):
    loader = _build_loader_with_paths(
        trained_path=tmp_path / "missing-trained.pt",
        fallback_path=tmp_path / "missing-fallback.pt",
    )

    with pytest.raises(FileNotFoundError, match="Trained phone detector not found"):
        loader.get_model(use_trained=True)

    with pytest.raises(FileNotFoundError, match="Fallback phone detector not found"):
        loader.get_model(use_trained=False)


def test_phone_model_loader_wraps_ultralytics_import_errors(tmp_path: Path):
    trained_path = tmp_path / "trained.pt"
    fallback_path = tmp_path / "fallback.pt"
    trained_path.write_bytes(b"trained")
    fallback_path.write_bytes(b"fallback")

    loader = _build_loader_with_paths(
        trained_path=trained_path,
        fallback_path=fallback_path,
    )

    original_import = __import__

    def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics":
            raise PermissionError("settings.yaml")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_mock_import):
        with pytest.raises(RuntimeError, match="Unable to import Ultralytics YOLO"):
            loader.load_trained_model()
