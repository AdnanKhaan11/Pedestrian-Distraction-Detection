from pathlib import Path

from src.entity.config_entity import MMPoseConfig
from src.utils.common import resolve_device
from src.utils.logger import get_logger


class MMPoseLoader:
    """
    Load and manage MMPose runtime models.

    This class wraps:
    - person detector
    - pose estimator
    - optional visualizer

    It keeps loading logic separate from inference logic.
    """

    def __init__(
        self, config: MMPoseConfig, log_dir: Path | None = None, log_level: str = "INFO"
    ) -> None:
        self.config = config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

        self.device = resolve_device(self.config.device_preference)

        self._bbox_detector = None
        self._pose_estimator = None
        self._visualizer = None

    def _validate_files(self) -> None:
        """
        Make sure config and checkpoint files exist before loading.
        """
        detector_config = Path(self.config.detector.config_file)
        detector_checkpoint = Path(self.config.detector.checkpoint_file)
        pose_config = Path(self.config.pose_estimator.config_file)
        pose_checkpoint = Path(self.config.pose_estimator.checkpoint_file)

        if not detector_config.exists():
            raise FileNotFoundError(
                f"Person detector config file not found: {detector_config}"
            )

        if not detector_checkpoint.exists():
            raise FileNotFoundError(
                f"Person detector checkpoint not found: {detector_checkpoint}"
            )

        if not pose_config.exists():
            raise FileNotFoundError(
                f"Pose estimator config file not found: {pose_config}"
            )

        if not pose_checkpoint.exists():
            raise FileNotFoundError(
                f"Pose estimator checkpoint not found: {pose_checkpoint}"
            )

    def load(self):
        """
        Load MMPose models and visualizer.

        Important:
        Imports are done inside this method so the project does not fail
        immediately on import if mmpose/mmdet are not installed yet.
        """
        self._validate_files()

        try:
            from mmdet.apis import init_detector
            from mmpose.apis import init_model as init_pose_estimator
            from mmpose.registry import VISUALIZERS
            from mmpose.utils import register_all_modules, adapt_mmdet_pipeline
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Unable to import MMPose/MMDetection. Ensure the packages are installed "
                "and the environment can access their dependencies."
            ) from exc

        register_all_modules()

        self.logger.info("Loading MMPose detector on device: %s", self.device)
        bbox_detector = init_detector(
            self.config.detector.config_file,
            str(self.config.detector.checkpoint_file),
            device=self.device,
        )
        bbox_detector.cfg = adapt_mmdet_pipeline(bbox_detector.cfg)

        self.logger.info("Loading MMPose pose estimator on device: %s", self.device)
        pose_estimator = init_pose_estimator(
            self.config.pose_estimator.config_file,
            str(self.config.pose_estimator.checkpoint_file),
            device=self.device,
            cfg_options=dict(
                model=dict(
                    test_cfg=dict(output_heatmaps=self.config.visualizer.draw_heatmap)
                )
            ),
        )

        # Apply visualizer settings from config
        pose_estimator.cfg.visualizer.radius = self.config.visualizer.radius
        pose_estimator.cfg.visualizer.alpha = self.config.visualizer.alpha
        pose_estimator.cfg.visualizer.line_width = self.config.visualizer.thickness

        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(
            pose_estimator.dataset_meta,
            skeleton_style=self.config.visualizer.skeleton_style,
        )

        self._bbox_detector = bbox_detector
        self._pose_estimator = pose_estimator
        self._visualizer = visualizer

        self.logger.info("MMPose models loaded successfully.")
        return self._bbox_detector, self._pose_estimator, self._visualizer

    @property
    def bbox_detector(self):
        if self._bbox_detector is None:
            raise RuntimeError("BBox detector is not loaded yet. Call load() first.")
        return self._bbox_detector

    @property
    def pose_estimator(self):
        if self._pose_estimator is None:
            raise RuntimeError("Pose estimator is not loaded yet. Call load() first.")
        return self._pose_estimator

    @property
    def visualizer(self):
        if self._visualizer is None:
            raise RuntimeError("Visualizer is not loaded yet. Call load() first.")
        return self._visualizer

    def get_runtime_bundle(self) -> dict:
        """
        Return a clean dictionary bundle for downstream pipeline code.
        """
        if (
            self._bbox_detector is None
            or self._pose_estimator is None
            or self._visualizer is None
        ):
            self.load()

        return {
            "bbox_detector_model": self._bbox_detector,
            "pose_estimator_model": self._pose_estimator,
            "visualizer": self._visualizer,
            "device": self.device,
        }
