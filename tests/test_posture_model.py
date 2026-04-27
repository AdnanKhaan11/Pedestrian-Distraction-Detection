import sys
from pathlib import Path

import torch
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager  # noqa: E402
from src.components.posture_model import PostureModelService  # noqa: E402
from src.models.posture_cnn import MLP3d  # noqa: E402


def test_posture_cnn_builds_and_forward_pass():
    manager = ConfigurationManager()
    posture_config = manager.get_posture_model_config()
    model = MLP3d(
        input_channel_num=posture_config.input_channels,
        output_class_num=posture_config.output_classes,
    )

    dummy_input = torch.randn(
        1,
        posture_config.input_channels,
        posture_config.input_shape.depth,
        posture_config.input_shape.height,
        posture_config.input_shape.width,
    )
    output = model(dummy_input)

    assert output.shape == (1, posture_config.output_classes)
    assert torch.is_tensor(output)


def test_posture_model_checkpoint_format():
    manager = ConfigurationManager()
    posture_config = manager.get_posture_model_config()
    checkpoint_path = posture_config.weights.default_weight_file

    # If the checkpoint exists, it should contain a model_state_dict
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert isinstance(checkpoint, dict)
        assert "model_state_dict" in checkpoint
        model_state_dict = checkpoint["model_state_dict"]
        assert isinstance(model_state_dict, dict)


def test_posture_model_service_loads_checkpoint_and_predicts(tmp_path: Path):
    manager = ConfigurationManager()
    posture_config = manager.get_posture_model_config()
    service = PostureModelService(posture_config)
    model = service.build_model()

    checkpoint_path = tmp_path / "posture_service_test.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    loaded_model = service.load_model(checkpoint_path)
    dummy_input = torch.randn(
        1,
        posture_config.input_channels,
        posture_config.input_shape.depth,
        posture_config.input_shape.height,
        posture_config.input_shape.width,
    )

    score_text, class_signal, probabilities = service.predict_tensor(dummy_input)

    assert isinstance(loaded_model, MLP3d)
    assert isinstance(score_text, str)
    assert class_signal in (0, 1)
    assert probabilities.shape == (posture_config.output_classes,)


def test_posture_model_service_rejects_batched_runtime_input():
    manager = ConfigurationManager()
    posture_config = manager.get_posture_model_config()
    service = PostureModelService(posture_config)
    service.model = service.build_model().eval()

    batched_input = torch.randn(
        2,
        posture_config.input_channels,
        posture_config.input_shape.depth,
        posture_config.input_shape.height,
        posture_config.input_shape.width,
    )

    with pytest.raises(ValueError, match="exactly one sample"):
        service.predict_tensor(batched_input)
