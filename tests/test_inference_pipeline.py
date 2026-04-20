import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager  # noqa: E402
from src.pipeline.inference_pipeline import InferencePipeline  # noqa: E402


def test_inference_pipeline_instantiation():
    manager = ConfigurationManager()
    mmpose_config = manager.get_mmpose_config()
    posture_config = manager.get_posture_model_config()
    phone_config = manager.get_phone_detector_config()
    inference_config = manager.get_inference_config()

    pipeline = InferencePipeline(
        mmpose_config=mmpose_config,
        posture_model_config=posture_config,
        phone_detector_config=phone_config,
        inference_config=inference_config,
    )

    assert pipeline.mmpose_config is mmpose_config
    assert pipeline.posture_model_config is posture_config
    assert pipeline.phone_detector_config is phone_config
    assert pipeline.inference_config is inference_config


def test_inference_pipeline_dummy_frame_shape():
    manager = ConfigurationManager()
    mmpose_config = manager.get_mmpose_config()
    posture_config = manager.get_posture_model_config()
    phone_config = manager.get_phone_detector_config()
    inference_config = manager.get_inference_config()

    pipeline = InferencePipeline(
        mmpose_config=mmpose_config,
        posture_model_config=posture_config,
        phone_detector_config=phone_config,
        inference_config=inference_config,
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.run_on_frame(dummy_frame, draw_visualizer=False)

    assert isinstance(result, dict)
    assert "person_results" in result
    assert "num_persons" in result
    assert isinstance(result["person_results"], list)
