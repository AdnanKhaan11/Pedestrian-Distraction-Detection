"""
This file runs the cleaned end-to-end inference pipeline for one frame or one image.
It connects MMPose loading, runtime person processing, and final results packaging.
This is the modular pipeline replacement for the mixed logic previously spread across
main.py, ai_engine.py, and processing.py in your current project.
"""

from pathlib import Path
from typing import Any

import numpy as np

from src.components.mmpose_loader import MMPoseLoader
from src.components.runtime_detector import RuntimeDetector
from src.entity.config_entity import (
    InferenceConfig,
    MMPoseConfig,
    PhoneDetectorConfig,
    PostureModelConfig,
)
from src.utils.logger import get_logger


class InferencePipeline:
    """
    Run full runtime inference on one frame.
    """

    def __init__(
        self,
        mmpose_config: MMPoseConfig,
        posture_model_config: PostureModelConfig,
        phone_detector_config: PhoneDetectorConfig,
        inference_config: InferenceConfig,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.mmpose_config = mmpose_config
        self.posture_model_config = posture_model_config
        self.phone_detector_config = phone_detector_config
        self.inference_config = inference_config
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

        self.mmpose_loader = MMPoseLoader(
            config=mmpose_config,
            log_dir=log_dir,
            log_level=log_level,
        )
        self.runtime_detector = RuntimeDetector(
            mmpose_config=mmpose_config,
            posture_model_config=posture_model_config,
            phone_detector_config=phone_detector_config,
            inference_config=inference_config,
            log_dir=log_dir,
            log_level=log_level,
        )

        self.bbox_detector = None
        self.pose_estimator = None
        self.visualizer = None

    def load_models(self) -> None:
        """
        Load MMPose models used for frame-level inference.
        """
        self.bbox_detector, self.pose_estimator, self.visualizer = (
            self.mmpose_loader.load()
        )
        self.logger.info("Inference pipeline models loaded.")

    def _process_one_image_with_mmpose(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """
        Run person detection and pose estimation on one frame.

        Returns:
        - keypoints_list: (num_people, num_keypoints, 3)
        - xyxy_list: (num_people, 4)
        - data_samples: raw MMPose result for optional visualization
        """
        if self.bbox_detector is None or self.pose_estimator is None:
            self.load_models()

        try:
            from mmdet.apis import inference_detector
            from mmpose.apis import inference_topdown
            from mmpose.evaluation.functional import nms
            from mmpose.structures import merge_data_samples
        except ImportError as exc:
            raise ImportError(
                "MMPose/MMDetection runtime packages are missing. "
                "Please install them before running inference."
            ) from exc

        det_result = inference_detector(self.bbox_detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()

        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]),
            axis=1,
        )

        bboxes = bboxes[
            np.logical_and(
                pred_instance.labels == self.mmpose_config.detector.category_id,
                pred_instance.scores
                > self.mmpose_config.detector.bbox_threshold_multi_person,
            )
        ]
        bboxes = bboxes[nms(bboxes, self.mmpose_config.detector.nms_threshold), :4]

        pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)

        raw_predictions = data_samples.get("pred_instances", None)
        keypoints_coords = raw_predictions.keypoints
        keypoint_scores = np.expand_dims(raw_predictions.keypoint_scores, axis=-1)
        keypoints_list = np.concatenate((keypoints_coords, keypoint_scores), axis=-1)
        xyxy_list = raw_predictions.bboxes

        return keypoints_list, xyxy_list, data_samples

    def run_on_frame(
        self,
        frame: np.ndarray,
        draw_visualizer: bool = False,
        runtime_parameters: dict | None = None,
    ) -> dict:
        """
        Run full inference on one frame.

        Args:
            frame: Input frame as numpy array
            draw_visualizer: Whether to draw visualization
            runtime_parameters: Optional runtime state dict (maintains face announce timing across frames)

        Returns:
        {
            "frame": updated frame,
            "person_results": [...],
            "num_persons": int
        }
        """
        original_frame = frame.copy()
        keypoints_list, xyxy_list, data_samples = self._process_one_image_with_mmpose(
            frame
        )

        if draw_visualizer and self.visualizer is not None:
            self.visualizer.add_datasample(
                name="inference_result",
                image=frame,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=self.mmpose_config.visualizer.draw_heatmap,
                draw_bbox=self.mmpose_config.visualizer.draw_bbox,
                show=False,
                wait_time=0,
                kpt_thr=self.mmpose_config.visualizer.keypoint_threshold,
            )

        if runtime_parameters is None:
            runtime_parameters = {
                "time_last_record_framerate": 0.0,
                "time_last_announce_face": 0.0,
                "path_runtime_handframes": None,
            }

        person_results = []
        for keypoints, xyxy in zip(keypoints_list, xyxy_list):
            one_result = self.runtime_detector.process_one_person(
                frame=frame,
                original_frame=original_frame,
                keypoints=keypoints,
                xyxy=xyxy,
                runtime_parameters=runtime_parameters,
            )
            one_result["xyxy"] = xyxy.tolist()
            person_results.append(one_result)

        result = {
            "frame": frame,
            "person_results": person_results,
            "num_persons": len(person_results),
        }

        self.logger.info(
            "Inference pipeline processed %d persons.", len(person_results)
        )
        return result
