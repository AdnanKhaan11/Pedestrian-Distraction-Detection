"""
This file runs the full data preparation pipeline in the correct order.
It connects ingestion, frame extraction, and dataset metadata creation.
The purpose is to keep data workflow reproducible and easy to track in MLOps.
This is the clean pipeline wrapper for your project's raw-data preparation stage.
"""

from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.components.dataset_builder import DatasetBuilder
from src.components.frame_extractor import FrameExtractor
from src.entity.config_entity import PathsConfig
from src.utils.logger import get_logger


class DataPipeline:
    """
    Run all major data-preparation steps in sequence.
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

        self.ingestion = DataIngestion(
            paths_config=paths_config,
            log_dir=log_dir,
            log_level=log_level,
        )
        self.frame_extractor = FrameExtractor(
            raw_video_dir=paths_config.raw_video_dir,
            output_dir=paths_config.raw_image_dir,
            log_dir=log_dir,
            log_level=log_level,
        )
        self.dataset_builder = DatasetBuilder(
            paths_config=paths_config,
            log_dir=log_dir,
            log_level=log_level,
        )

    def run(self, frame_step_size: int = 10) -> dict:
        """
        Run the full data pipeline.
        """
        self.logger.info("Starting data pipeline.")

        ingestion_summary = self.ingestion.run()
        extraction_summary = self.frame_extractor.run(step_size=frame_step_size)
        dataset_summary = self.dataset_builder.run()

        summary = {
            "ingestion": ingestion_summary,
            "frame_extraction": extraction_summary,
            "dataset_builder": dataset_summary,
        }

        self.logger.info("Data pipeline completed.")
        return summary
