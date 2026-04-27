import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_paths_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        logs_dir=tmp_path / "logs",
        predictions_dir=tmp_path / "predictions",
        frontend_result_dir=tmp_path / "results",
        metrics_dir=tmp_path / "metrics",
    )


def _build_fake_config_manager(tmp_path: Path):
    paths_config = _build_paths_config(tmp_path)
    return SimpleNamespace(
        get_paths_config=lambda: paths_config,
        get_mmpose_config=lambda: SimpleNamespace(),
        get_posture_model_config=lambda: SimpleNamespace(),
        get_phone_detector_config=lambda: SimpleNamespace(),
        get_inference_config=lambda: SimpleNamespace(),
    )


def test_predictor_predict_image_returns_structured_response(monkeypatch, tmp_path: Path):
    predictor_module = pytest.importorskip("src.serving.predictor")

    fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run_on_frame(self, frame, draw_visualizer=False):
            assert draw_visualizer is False
            return {
                "frame": frame,
                "num_persons": 1,
                "person_results": [
                    {
                        "posture": "distracted",
                        "phone": True,
                        "state": 32,
                        "display_text": "+",
                        "score_text": "0.91",
                    }
                ],
            }

    saved_json_calls = []

    monkeypatch.setattr(
        predictor_module,
        "ConfigurationManager",
        lambda: _build_fake_config_manager(tmp_path),
    )
    monkeypatch.setattr(predictor_module, "InferencePipeline", FakePipeline)
    monkeypatch.setattr(predictor_module, "create_directories", lambda paths: None)
    monkeypatch.setattr(
        predictor_module, "save_json", lambda path, data: saved_json_calls.append((path, data))
    )
    monkeypatch.setattr(predictor_module.cv2, "imread", lambda _: fake_frame.copy())
    monkeypatch.setattr(predictor_module.cv2, "imwrite", lambda *_: True)

    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"image")

    predictor = predictor_module.Predictor(log_dir=tmp_path / "logs")
    response = predictor.predict_image(image_path=image_path, save_rendered_output=True)

    assert response["num_persons"] == 1
    assert response["person_results"][0]["phone"] is True
    assert response["person_results"][0]["score_text"] == "0.91"
    assert response["saved_result_path"] is not None
    assert saved_json_calls


def test_predictor_rejects_missing_or_unreadable_images(monkeypatch, tmp_path: Path):
    predictor_module = pytest.importorskip("src.serving.predictor")

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run_on_frame(self, frame, draw_visualizer=False):
            return {"frame": frame, "num_persons": 0, "person_results": []}

    monkeypatch.setattr(
        predictor_module,
        "ConfigurationManager",
        lambda: _build_fake_config_manager(tmp_path),
    )
    monkeypatch.setattr(predictor_module, "InferencePipeline", FakePipeline)
    monkeypatch.setattr(predictor_module, "create_directories", lambda paths: None)
    monkeypatch.setattr(predictor_module, "save_json", lambda *args, **kwargs: None)

    predictor = predictor_module.Predictor(log_dir=tmp_path / "logs")

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        predictor.predict_image(tmp_path / "missing.jpg")

    unreadable_image = tmp_path / "bad.jpg"
    unreadable_image.write_bytes(b"bad")
    monkeypatch.setattr(predictor_module.cv2, "imread", lambda _: None)

    with pytest.raises(ValueError, match="Could not read image"):
        predictor.predict_image(unreadable_image)


def test_predictor_raises_when_rendered_output_cannot_be_saved(
    monkeypatch, tmp_path: Path
):
    predictor_module = pytest.importorskip("src.serving.predictor")

    fake_frame = np.zeros((16, 16, 3), dtype=np.uint8)

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run_on_frame(self, frame, draw_visualizer=False):
            return {"frame": frame, "num_persons": 0, "person_results": []}

    monkeypatch.setattr(
        predictor_module,
        "ConfigurationManager",
        lambda: _build_fake_config_manager(tmp_path),
    )
    monkeypatch.setattr(predictor_module, "InferencePipeline", FakePipeline)
    monkeypatch.setattr(predictor_module, "create_directories", lambda paths: None)
    monkeypatch.setattr(predictor_module, "save_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(predictor_module.cv2, "imread", lambda _: fake_frame.copy())
    monkeypatch.setattr(predictor_module.cv2, "imwrite", lambda *_: False)

    predictor = predictor_module.Predictor(log_dir=tmp_path / "logs")
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"image")

    with pytest.raises(IOError, match="Could not save rendered output image"):
        predictor.predict_image(image_path=image_path, save_rendered_output=True)
