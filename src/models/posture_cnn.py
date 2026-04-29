from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResPool3d(nn.Module):
    """
    Residual pooling block from your current project.

    It keeps important max-pooled responses,
    then adds them back to the original tensor.
    """

    __constants__ = ["kernel_size", "stride", "padding", "dilation", "ceil_mode"]
    ceil_mode: bool

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, input_tensor: Tensor) -> Tensor:
        max_pooled, indices = F.max_pool3d(
            input_tensor,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=True,
        )

        output_shape = input_tensor.shape
        output_tensor = torch.zeros(
            output_shape,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

        output_tensor = (
            output_tensor.view(-1)
            .scatter_(
                0,
                indices.view(-1),
                max_pooled.view(-1),
            )
            .view(output_shape)
        )

        return input_tensor + output_tensor


class FocalLoss(nn.Module):
    """
    Optional focal loss helper for posture training.
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        if self.reduction not in ["mean", "sum"]:
            raise ValueError("reduction must be either 'mean' or 'sum'")

    def forward(self, inputs: Tensor, labels: Tensor) -> Tensor:
        probs = torch.softmax(inputs, dim=-1)

        alpha_tensor = torch.tensor(
            [
                [self.alpha] if label.item() == 1 else [1 - self.alpha]
                for label in labels
            ],
            dtype=inputs.dtype,
            device=inputs.device,
        )

        pt = probs.gather(dim=-1, index=labels.unsqueeze(1))
        loss = (alpha_tensor * ((1 - pt) ** self.gamma)) * (-torch.log(pt + 1e-12))

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class L2Regularization(nn.Module):
    """
    Optional L2 regularization helper for posture training.
    """

    def __init__(self, l2_lambda: float = 0.0001) -> None:
        super().__init__()
        self.l2_lambda = l2_lambda

    def forward(self, model: nn.Module) -> Tensor:
        device = next(model.parameters()).device
        l2_reg = torch.tensor(0.0, device=device)

        for parameter in model.parameters():
            l2_reg += torch.sum(parameter**2)

        return self.l2_lambda * l2_reg


class MCLoss(nn.Module):
    """
    Combined loss from your current project design:
    - Cross entropy
    - Focal loss
    - L2 regularization
    """

    def __init__(
        self,
        w1: float = 0.6,
        w2: float = 0.3,
        w3: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        l2_lambda: float = 0.0001,
    ) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.l2 = L2Regularization(l2_lambda=l2_lambda)

    def forward(
        self, outputs: Tensor, labels: Tensor, mlp3d_instance: nn.Module
    ) -> Tensor:
        ce_loss = F.cross_entropy(outputs, labels)
        focal_loss = self.focal(outputs, labels)
        l2_reg = self.l2(mlp3d_instance)
        total_loss = self.w1 * ce_loss + self.w2 * focal_loss + self.w3 * l2_reg
        return total_loss


class MLP3d(nn.Module):
    """
    Main posture classifier copied cleanly from your current project.

    Input shape:
    (N, C=2, D=7, H=12, W=11)
    """

    def __init__(
        self,
        input_channel_num: int,
        output_class_num: int,
        input_shape: Tuple[int, int, int] = (7, 12, 11),
        conv_kernel_size: Union[int, Tuple[int, int, int]] = (3, 5, 5),
        pool_kernel_size: Union[int, Tuple[int, int, int]] = 2,
        activation_name: str = "SiLU",
        fc_dims: Sequence[int] = (7392, 1848, 256),
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.activation_name = activation_name
        self.fc_dims = list(fc_dims)

        self.conv_layers = nn.Sequential(
            nn.Conv3d(
                in_channels=input_channel_num,
                out_channels=8,
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=8),
            self._build_activation(),
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=16),
            self._build_activation(),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=self.kernel_size,
                padding="same",
            ),
            nn.BatchNorm3d(num_features=32),
            self._build_activation(),
            ResPool3d(
                kernel_size=self.pool_kernel_size,
                stride=self.pool_kernel_size,
                padding=0,
            ),
        )

        flattened_features = self._infer_flattened_features(input_channel_num)
        fc_layer_sizes = [flattened_features, *self.fc_dims, output_class_num]

        fc_modules: list[nn.Module] = []
        for idx in range(len(fc_layer_sizes) - 1):
            fc_modules.append(
                nn.Linear(
                    in_features=fc_layer_sizes[idx],
                    out_features=fc_layer_sizes[idx + 1],
                )
            )
            if idx < len(fc_layer_sizes) - 2:
                fc_modules.append(self._build_activation())

        self.fc_layers = nn.Sequential(*fc_modules)

    def _build_activation(self) -> nn.Module:
        activation_map = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "leakyrelu": nn.LeakyReLU,
        }
        normalized_name = self.activation_name.lower().strip()
        activation_cls = activation_map.get(normalized_name)
        if activation_cls is None:
            supported = ", ".join(sorted(activation_map))
            raise ValueError(
                f"Unsupported activation '{self.activation_name}'. Supported: {supported}"
            )
        return activation_cls()

    def _infer_flattened_features(self, input_channel_num: int) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                input_channel_num,
                self.input_shape[0],
                self.input_shape[1],
                self.input_shape[2],
            )
            conv_output = self.conv_layers(dummy_input)
        return int(torch.flatten(conv_output, start_dim=1).shape[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x
