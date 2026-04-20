"""
This file prepares raw project data folders for the new MLOps pipeline.
It does not train or infer anything. It only checks that expected raw data paths exist
and creates missing folders safely. This keeps data preparation clean and organized.
It is the first step before frame extraction or dataset generation.
"""

from pathlib import Path

from src.entity.config_entity import PathsConfig
from src.utils.common import create_directories
from src.utils.helpers import list_image_files, list_video_files
from src.utils.logger import get_logger


class DataIngestion:
    """
    Prepare and validate the raw data area of the project.
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

    def create_required_directories(self) -> None:
        """
        Create all raw/interim/processed folders that the data pipeline needs.
        """
        create_directories(
            [
                self.paths_config.raw_data_dir,
                self.paths_config.interim_data_dir,
                self.paths_config.processed_data_dir,
                self.paths_config.raw_video_dir,
                self.paths_config.raw_image_dir,
                self.paths_config.posture_feature_dir,
                self.paths_config.hand_crop_dir,
                self.paths_config.phone_dataset_dir,
                self.paths_config.metrics_dir,
                self.paths_config.predictions_dir,
            ]
        )
        self.logger.info("Required data directories are ready.")

    def summarize_available_data(self) -> dict:
        """
        Count currently available raw videos and raw images.
        """
        videos = list_video_files(self.paths_config.raw_video_dir)
        images = list_image_files(self.paths_config.raw_image_dir)

        summary = {
            "raw_video_count": len(videos),
            "raw_image_count": len(images),
            "raw_video_dir": str(self.paths_config.raw_video_dir),
            "raw_image_dir": str(self.paths_config.raw_image_dir),
        }

        self.logger.info("Data summary: %s", summary)
        return summary

    def run(self) -> dict:
        """
        Run the ingestion stage.
        """
        self.create_required_directories()
        return self.summarize_available_data()
