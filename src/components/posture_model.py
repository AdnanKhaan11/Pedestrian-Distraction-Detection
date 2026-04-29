from pathlib import Path

import numpy as np
import torch

from src.entity.config_entity import PostureModelConfig
from src.models.posture_cnn import MLP3d
from src.utils.common import resolve_device
from src.utils.logger import get_logger


class PostureModelService:
    """
    Load and run the posture classifier.

    This service wraps:
    - model construction
    - checkpoint loading
    - inference
    - confidence thresholding

    It uses the exact same MLP3d architecture as your current project.
    """

    def __init__(
        self,
        config: PostureModelConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.config = config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

        self.device = resolve_device("auto")
        self.model: MLP3d | None = None

    def build_model(self) -> MLP3d:
        """
        Build the posture CNN architecture.
        """
        model = MLP3d(
            input_channel_num=self.config.input_channels,
            output_class_num=self.config.output_classes,
            input_shape=(
                self.config.input_shape.depth,
                self.config.input_shape.height,
                self.config.input_shape.width,
            ),
            conv_kernel_size=tuple(self.config.architecture.conv_kernel_size),
            pool_kernel_size=self.config.architecture.pool_kernel_size,
            activation_name=self.config.architecture.activation,
            fc_dims=self.config.architecture.fc_dims,
        )
        return model

    def load_model(self, weight_path: Path | None = None) -> MLP3d:
        """
        Load posture model weights from checkpoint.
        """
        if weight_path is None:
            weight_path = self.config.weights.default_weight_file

        weight_path = Path(weight_path)
        if not weight_path.is_absolute():
            ROOT_DIR = Path(__file__).resolve().parents[2]
            weight_path = ROOT_DIR / weight_path
        weight_path = weight_path.resolve()
        print("DEBUG PATH:", weight_path)

        if not weight_path.exists():
            raise FileNotFoundError(
                f"Posture model checkpoint not found: {weight_path}"
            )

        model = self.build_model()
        checkpoint = torch.load(weight_path, map_location=self.device)

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(
                "Invalid posture checkpoint format. Expected a dictionary with 'model_state_dict'."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self.device)

        self.model = model
        self.logger.info("Posture model loaded from: %s", weight_path)
        return model

    def predict_tensor(self, input_tensor: torch.Tensor) -> tuple[str, int, np.ndarray]:
        """
        Predict from one already-prepared tensor.

        Expected tensor shape:
        (1, C, D, H, W)
        """
        if self.model is None:
            self.load_model()

        if input_tensor.ndim != 5:
            raise ValueError(
                f"Expected posture tensor shape (N, C, D, H, W), got: {input_tensor.shape}"
            )

        if input_tensor.shape[0] != 1:
            raise ValueError(
                "PostureModelService expects exactly one sample for runtime inference. "
                f"Received batch size: {input_tensor.shape[0]}"
            )

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.sigmoid(outputs[0])

        score_not_using = float(probabilities[0].item())
        score_using = float(probabilities[1].item())

        threshold = self.config.inference.confidence_threshold
        prediction_is_using = (score_using > score_not_using) and (
            score_using > threshold
        )

        class_signal = 1 if prediction_is_using else 0
        display_score = score_using if prediction_is_using else score_not_using
        score_text = f"{display_score:.2f}"

        return score_text, class_signal, probabilities.cpu().numpy()

    def predict_numpy(self, input_array: np.ndarray) -> tuple[str, int, np.ndarray]:
        """
        Predict from numpy tensor.

        Expected numpy shape:
        (1, C, D, H, W)
        """
        tensor = torch.tensor(input_array, dtype=torch.float32)
        return self.predict_tensor(tensor)
