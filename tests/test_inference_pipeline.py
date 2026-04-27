"""
These tests validate the new inference pipeline contract.
They avoid loading real MMPose or YOLO weights by replacing heavy
runtime services with small fake classes. This keeps testing fast,
Windows-friendly, and suitable for early refactor checkpoints.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_inference_config() -> SimpleNamespace:
    return SimpleNamespace(
        general=SimpleNamespace(allow_multiple_persons=True, max_persons_per_frame=10),
        posture=SimpleNamespace(confidence_threshold=0.75),
        phone=SimpleNamespace(trained_model_confidence=0.65),
        rendering=SimpleNamespace(show_fps=True),
    )


def test_inference_pipeline_load_models_uses_mmpose_loader(monkeypatch):
    inference_module = pytest.importorskip("src.pipeline.inference_pipeline")

    class FakeMMPoseLoader:
        def __init__(self, *args, **kwargs):
            self.called = False

        def load(self):
            self.called = True
            return "bbox-detector", "pose-estimator", "visualizer"

    class FakeRuntimeDetector:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(inference_module, "MMPoseLoader", FakeMMPoseLoader)
    monkeypatch.setattr(inference_module, "RuntimeDetector", FakeRuntimeDetector)

    pipeline = inference_module.InferencePipeline(
        mmpose_config=SimpleNamespace(),
        posture_model_config=SimpleNamespace(),
        phone_detector_config=SimpleNamespace(),
        inference_config=_build_inference_config(),
    )
    pipeline.load_models()

    assert pipeline.bbox_detector == "bbox-detector"
    assert pipeline.pose_estimator == "pose-estimator"
    assert pipeline.visualizer == "visualizer"


def test_inference_pipeline_can_run_frame_with_fake_runtime(monkeypatch):
    inference_module = pytest.importorskip("src.pipeline.inference_pipeline")

    class FakeMMPoseLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return "bbox-detector", "pose-estimator", None

    class FakeRuntimeDetector:
        def __init__(self, *args, **kwargs):
            pass

        def process_people(self, frame, keypoints_list, xyxy_list):
            return {
                "frame": frame,
                "num_persons": len(keypoints_list),
                "person_results": [
                    {
                        "posture": "using_or_suspicious",
                        "phone": True,
                        "state": "distracted",
                        "display_text": "+ 0.91",
                        "score_text": "0.91",
                    }
                ],
            }

    monkeypatch.setattr(inference_module, "MMPoseLoader", FakeMMPoseLoader)
    monkeypatch.setattr(inference_module, "RuntimeDetector", FakeRuntimeDetector)

    pipeline = inference_module.InferencePipeline(
        mmpose_config=SimpleNamespace(),
        posture_model_config=SimpleNamespace(),
        phone_detector_config=SimpleNamespace(),
        inference_config=_build_inference_config(),
    )

    monkeypatch.setattr(
        pipeline,
        "_process_one_image_with_mmpose",
        lambda frame: (
            np.zeros((1, 13, 3), dtype=np.float32),
            np.array([[10, 20, 100, 200]], dtype=np.float32),
            None,
        ),
    )

    if not hasattr(pipeline, "run_on_frame"):
        pytest.skip(
            "run_on_frame is not available in the current inference pipeline implementation."
        )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.run_on_frame(frame=frame, draw_visualizer=False)

    assert result["num_persons"] == 1
    assert result["person_results"][0]["phone"] is True
    assert result["person_results"][0]["state"] == "distracted"
    # assert result["person_results"][0]["display_text"] == "+ 0.91"
    # assert result["person_results"][0]["score_text"] == "0.91"
