"""
This file builds processed datasets for posture and phone-related training stages.
It keeps your data preparation logic separate from model training logic.
Right now it prepares clean folder-based outputs and metadata summaries.
Later it can be extended for train/val/test split generation and manifest files.
"""

from pathlib import Path

from src.entity.config_entity import PathsConfig
from src.utils.common import create_directories, save_json
from src.utils.helpers import list_image_files, split_list
from src.utils.logger import get_logger


class DatasetBuilder:
    """
    Build simple dataset metadata and split summaries for downstream pipelines.
    """

    def __init__(
        self,
        paths_config: PathsConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.paths_config = paths_config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

    def build_image_dataset_manifest(
        self,
        source_dir: Path,
        manifest_name: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict:
        """
        Create a simple dataset manifest from a folder of images.
        """
        images = list_image_files(source_dir)

        train_items, val_items, test_items = split_list(
            items=images,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            shuffle=True,
            seed=42,
        )

        manifest = {
            "source_dir": str(source_dir),
            "total_images": len(images),
            "train_count": len(train_items),
            "val_count": len(val_items),
            "test_count": len(test_items),
            "train_files": [str(path) for path in train_items],
            "val_files": [str(path) for path in val_items],
            "test_files": [str(path) for path in test_items],
        }

        create_directories([self.paths_config.metrics_dir])
        save_json(self.paths_config.metrics_dir / manifest_name, manifest)

        self.logger.info("Saved dataset manifest: %s", manifest_name)
        return manifest

    def run(self) -> dict:
        """
        Build manifests for available processed datasets.
        """
        create_directories(
            [
                self.paths_config.posture_feature_dir,
                self.paths_config.hand_crop_dir,
                self.paths_config.phone_dataset_dir,
                self.paths_config.metrics_dir,
            ]
        )

        posture_manifest = self.build_image_dataset_manifest(
            source_dir=self.paths_config.raw_image_dir,
            manifest_name="raw_image_manifest.json",
        )

        phone_manifest = self.build_image_dataset_manifest(
            source_dir=self.paths_config.phone_dataset_dir,
            manifest_name="phone_dataset_manifest.json",
        )

        summary = {
            "posture_manifest_total": posture_manifest["total_images"],
            "phone_manifest_total": phone_manifest["total_images"],
        }

        self.logger.info("Dataset builder summary: %s", summary)
        return summary
