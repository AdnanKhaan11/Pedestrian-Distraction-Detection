"""
This file trains your posture classifier using the same MLP3d logic from the current project.
It reads saved pose-feature .npy files, builds train/val/test loaders, and tracks training cleanly.
It also logs metrics with logger and MLflow, which can be connected to DAGsHub.
The saved artifacts are disk-friendly and DVC-friendly for later versioning.
"""

from pathlib import Path
import os
from typing import Tuple

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from src.entity.config_entity import PathsConfig, PostureModelConfig, TrainingConfig
from src.models.posture_cnn import MCLoss, MLP3d
from src.utils.common import resolve_device, save_json, save_yaml
from src.utils.logger import get_logger
from src.utils.naming_utils import parse_pose_data_filename


class PostureTrainer:
    """
    Train the posture model using .npy pose feature files.
    """

    def __init__(
        self,
        paths_config: PathsConfig,
        posture_model_config: PostureModelConfig,
        training_config: TrainingConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.paths_config = paths_config
        self.posture_model_config = posture_model_config
        self.training_config = training_config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

        self.device = resolve_device(self.training_config.general.device_preference)
        self.model = MLP3d(
            input_channel_num=self.posture_model_config.input_channels,
            output_class_num=self.posture_model_config.output_classes,
            input_shape=(
                self.posture_model_config.input_shape.depth,
                self.posture_model_config.input_shape.height,
                self.posture_model_config.input_shape.width,
            ),
            conv_kernel_size=tuple(
                self.posture_model_config.architecture.conv_kernel_size
            ),
            pool_kernel_size=self.posture_model_config.architecture.pool_kernel_size,
            activation_name=self.posture_model_config.architecture.activation,
            fc_dims=self.posture_model_config.architecture.fc_dims,
        ).to(self.device)

    def _setup_mlflow(self) -> None:
        """
        Configure MLflow tracking. DAGsHub can be used through the tracking URI.
        """
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self.training_config.tracking.tracking_uri,
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.training_config.tracking.experiment_name_posture)

    def _collect_feature_files(self) -> list[Path]:
        """
        Find all .npy posture feature files.
        """
        source_dir = self.training_config.posture_training.data.source_feature_dir
        feature_files = sorted(source_dir.rglob("*.npy"))

        if not feature_files:
            raise FileNotFoundError(
                f"No posture feature .npy files found in: {source_dir}"
            )

        self.logger.info("Found %d posture feature files.", len(feature_files))
        return feature_files

    def _load_dataset_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all posture feature arrays and convert labels from filename.
        """
        feature_files = self._collect_feature_files()

        inputs_list = []
        labels_list = []

        for file_path in feature_files:
            file_info = parse_pose_data_filename(file_path.name, extension=".npy")
            label_str = file_info["label"]
            label_value = 1 if label_str.startswith("U") else 0

            feature_array = np.load(file_path)

            if feature_array.ndim != 5:
                raise ValueError(
                    f"Expected posture feature array with 5 dims (N, C, D, H, W), got {feature_array.shape} "
                    f"for file {file_path}"
                )

            file_labels = np.full(
                shape=(feature_array.shape[0],), fill_value=label_value, dtype=np.int64
            )

            inputs_list.append(feature_array.astype(np.float32))
            labels_list.append(file_labels)

        inputs = np.concatenate(inputs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        self.logger.info(
            "Combined posture dataset shape: X=%s, y=%s", inputs.shape, labels.shape
        )
        return inputs, labels

    def _split_dataset(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        """
        Split arrays into train/val/test using config ratios.
        """
        total_samples = inputs.shape[0]
        indices = np.arange(total_samples)

        if self.training_config.posture_training.data.shuffle:
            np.random.shuffle(indices)

        inputs = inputs[indices]
        labels = labels[indices]

        train_ratio = self.training_config.posture_training.data.train_split
        val_ratio = self.training_config.posture_training.data.val_split

        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        x_train, y_train = inputs[:train_end], labels[:train_end]
        x_val, y_val = inputs[train_end:val_end], labels[train_end:val_end]
        x_test, y_test = inputs[val_end:], labels[val_end:]

        self.logger.info(
            "Posture split sizes -> train=%d, val=%d, test=%d",
            len(x_train),
            len(x_val),
            len(x_test),
        )
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def _to_dataloader(
        self, inputs: np.ndarray, labels: np.ndarray, shuffle: bool
    ) -> DataLoader:
        """
        Convert numpy arrays to PyTorch DataLoader.
        """
        x_tensor = torch.tensor(inputs, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)

        loader = DataLoader(
            dataset,
            batch_size=self.training_config.posture_training.hyperparameters.batch_size,
            shuffle=shuffle,
            num_workers=self.training_config.general.num_workers,
            pin_memory=self.training_config.general.pin_memory,
        )
        return loader

    def _save_dataset_manifest(self, x_train, x_val, x_test) -> None:
        """
        Save a small manifest file. This is useful for DVC tracking later.
        """
        manifest = {
            "task": "posture_training",
            "train_samples": int(len(x_train)),
            "val_samples": int(len(x_val)),
            "test_samples": int(len(x_test)),
            "feature_source_dir": str(
                self.training_config.posture_training.data.source_feature_dir
            ),
        }
        save_json(
            self.paths_config.metrics_dir / "posture_dataset_manifest.json", manifest
        )

    def train(self) -> dict:
        """
        Run full posture training.
        """
        self._setup_mlflow()

        inputs, labels = self._load_dataset_arrays()
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self._split_dataset(
            inputs, labels
        )
        self._save_dataset_manifest(x_train, x_val, x_test)

        train_loader = self._to_dataloader(x_train, y_train, shuffle=True)
        val_loader = self._to_dataloader(x_val, y_val, shuffle=False)
        test_loader = self._to_dataloader(x_test, y_test, shuffle=False)

        if self.training_config.posture_training.use_existing_weights:
            existing_weight_path = (
                self.training_config.posture_training.existing_weights_path
            )
            if existing_weight_path.exists():
                checkpoint = torch.load(existing_weight_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    self.logger.info(
                        "Loaded existing posture weights from: %s", existing_weight_path
                    )

        criterion = MCLoss()
        optimizer = Adam(
            self.model.parameters(),
            lr=self.training_config.posture_training.hyperparameters.learning_rate,
            weight_decay=self.training_config.posture_training.hyperparameters.weight_decay,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
        }

        with mlflow.start_run(run_name="posture_training_run"):
            mlflow.log_param("device", self.device)
            mlflow.log_param(
                "epochs", self.training_config.posture_training.hyperparameters.epochs
            )
            mlflow.log_param(
                "batch_size",
                self.training_config.posture_training.hyperparameters.batch_size,
            )
            mlflow.log_param(
                "learning_rate",
                self.training_config.posture_training.hyperparameters.learning_rate,
            )

            for epoch in range(
                self.training_config.posture_training.hyperparameters.epochs
            ):
                self.model.train()
                running_train_loss = 0.0

                for batch_inputs, batch_labels in train_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_inputs)
                    loss = criterion(outputs, batch_labels, self.model)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item() * len(batch_inputs)

                train_loss = running_train_loss / max(len(train_loader.dataset), 1)

                self.model.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for batch_inputs, batch_labels in val_loader:
                        batch_inputs = batch_inputs.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        outputs = self.model(batch_inputs)
                        loss = criterion(outputs, batch_labels, self.model)
                        running_val_loss += loss.item() * len(batch_inputs)

                val_loss = running_val_loss / max(len(val_loader.dataset), 1)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                self.logger.info(
                    "Epoch %d | train_loss=%.6f | val_loss=%.6f",
                    epoch + 1,
                    train_loss,
                    val_loss,
                )

                if (
                    val_loss
                    < best_val_loss
                    - self.training_config.posture_training.hyperparameters.min_delta
                ):
                    best_val_loss = val_loss
                    patience_counter = 0

                    best_path = (
                        self.paths_config.posture_weights_dir
                        / self.training_config.posture_training.outputs.save_best_model_as
                    )
                    torch.save({"model_state_dict": self.model.state_dict()}, best_path)
                    self.logger.info("Saved new best posture model to: %s", best_path)
                else:
                    patience_counter += 1

                if (
                    patience_counter
                    >= self.training_config.posture_training.hyperparameters.early_stopping_patience
                ):
                    self.logger.info("Early stopping triggered for posture training.")
                    break

            last_path = (
                self.paths_config.posture_weights_dir
                / self.training_config.posture_training.outputs.save_last_model_as
            )
            torch.save({"model_state_dict": self.model.state_dict()}, last_path)

            history_path = (
                self.paths_config.metrics_dir
                / self.training_config.posture_training.outputs.history_file_name
            )
            save_json(history_path, history)

            dvc_meta = {
                "artifact_type": "posture_model",
                "best_model_path": str(
                    self.paths_config.posture_weights_dir
                    / self.training_config.posture_training.outputs.save_best_model_as
                ),
                "last_model_path": str(last_path),
                "history_path": str(history_path),
            }
            save_yaml(
                self.paths_config.metrics_dir / "posture_dvc_manifest.yaml", dvc_meta
            )

        return {
            "history": history,
            "test_loader": test_loader,
            "best_val_loss": best_val_loss,
            "model": self.model,
        }
