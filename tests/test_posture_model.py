import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager  # noqa: E402
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
