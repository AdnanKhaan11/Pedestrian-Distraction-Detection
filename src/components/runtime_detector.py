"""
This file is the main runtime orchestrator for one person inside one frame.
It connects pose feature generation, posture classification, hand cropping, phone detection,
and final fusion into one clean modular flow. This is the clean replacement for the large
processOnePerson logic from your current project while keeping the same decision behavior.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from src.components.distraction_fusion import DistractionFusion
from src.components.hand_cropper import HandCropper
from src.components.phone_detector import PhoneDetector
from src.components.pose_feature_generator import PoseFeatureGenerator
from src.components.posture_detector import PostureDetector
from src.config.constants import (
    STATE_BACKSIDE,
    STATE_NOT_USING,
    STATE_OUT_OF_FRAME,
    STATE_SUSPICIOUS,
    STATE_TO_BE_CLASSIFIED,
)
from src.entity.config_entity import (
    InferenceConfig,
    MMPoseConfig,
    PhoneDetectorConfig,
    PostureModelConfig,
)
from src.utils.logger import get_logger
from src.utils.opencv_utils import (
    crop_frame,
    relative_to_absolute,
    render_detection_rectangle,
    resize_frame_to_square,
)


class RuntimeDetector:
    """
    End-to-end runtime detector for one person.
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

        self.pose_feature_generator = PoseFeatureGenerator(
            mmpose_config=mmpose_config,
            posture_model_config=posture_model_config,
            log_dir=log_dir,
            log_level=log_level,
        )
        self.posture_detector = PostureDetector(
            config=posture_model_config,
            log_dir=log_dir,
            log_level=log_level,
        )
        self.hand_cropper = HandCropper(
            mmpose_config=mmpose_config,
            phone_detector_config=phone_detector_config,
            log_dir=log_dir,
            log_level=log_level,
        )
        self.phone_detector = PhoneDetector(
            config=phone_detector_config,
            log_dir=log_dir,
            log_level=log_level,
        )
        self.fusion = DistractionFusion(log_dir=log_dir, log_level=log_level)

        self.posture_detector.load()

    def _initial_state_checks(
        self, keypoints: np.ndarray, xyxy: np.ndarray
    ) -> tuple[int, str]:
        """
        Reproduce your current lightweight state machine checks
        before posture classification.
        """
        state = STATE_TO_BE_CLASSIFIED
        score_text = ""

        keypoint_score_threshold = (
            self.inference_config.posture.keypoint_score_threshold
        )
        missing_threshold = (
            self.inference_config.posture.out_of_frame_missing_keypoints_threshold
        )

        visible_keypoints = keypoints[:13, 2]
        if np.sum(visible_keypoints < keypoint_score_threshold) >= missing_threshold:
            return STATE_OUT_OF_FRAME, score_text

        left_shoulder_x = keypoints[self.mmpose_config.keypoints.left_shoulder_index][0]
        right_shoulder_x = keypoints[self.mmpose_config.keypoints.right_shoulder_index][
            0
        ]
        left_shoulder_score = keypoints[
            self.mmpose_config.keypoints.left_shoulder_index
        ][2]
        right_shoulder_score = keypoints[
            self.mmpose_config.keypoints.right_shoulder_index
        ][2]

        backside_ratio = (left_shoulder_x - right_shoulder_x) / (
            (xyxy[2] - xyxy[0]) + np.finfo(np.float32).eps
        )

        if (
            right_shoulder_score > keypoint_score_threshold
            and left_shoulder_score > keypoint_score_threshold
            and backside_ratio < self.inference_config.posture.backside_ratio_threshold
        ):
            numeric_value = (
                (right_shoulder_score + left_shoulder_score) / 2.0 + 1.0
            ) / 2.0
            score_text = f"{numeric_value:.2f}"
            return STATE_BACKSIDE, score_text

        return state, score_text

    def _run_posture_stage(self, keypoints: np.ndarray) -> tuple[int, str]:
        """
        Convert keypoints to pose features and run posture classifier.
        """
        feature_tensor = self.pose_feature_generator.build_feature_tensor(
            keypoints, normalize=True
        )
        posture_result = self.posture_detector.predict(feature_tensor)

        state = (
            STATE_SUSPICIOUS if posture_result["class_signal"] == 1 else STATE_NOT_USING
        )
        score_text = posture_result["score_text"]

        return state, score_text

    def _run_phone_stage(
        self,
        frame: np.ndarray,
        original_frame: np.ndarray,
        keypoints: np.ndarray,
        xyxy: np.ndarray,
    ) -> tuple[bool, np.ndarray | None]:
        """
        Crop hands, run phone detection, and return phone status.
        """
        use_trained_model = (
            self.phone_detector_config.inference.use_trained_model_by_default
        )
        primary_crop, secondary_crop, spare_ratio = (
            self.hand_cropper.get_priority_hand_crops(
                frame=original_frame,
                keypoints=keypoints,
                xyxy=xyxy,
            )
        )

        for crop_index, crop_item in enumerate([primary_crop, secondary_crop]):
            if crop_item is None:
                continue

            hand_frame, hand_xyxy = crop_item

            if hand_frame is None or hand_xyxy is None or hand_frame.size == 0:
                continue

            subframe_wh = (
                abs(hand_xyxy[2] - hand_xyxy[0]),
                abs(hand_xyxy[3] - hand_xyxy[1]),
            )

            resized_hand = resize_frame_to_square(
                frame=hand_frame,
                edge_length=(
                    self.phone_detector_config.inference.image_size
                    if use_trained_model
                    else 640
                ),
                ratio_threshold=0.5625,
            )

            rgb_hand = cv2.cvtColor(resized_hand, cv2.COLOR_BGR2RGB)
            phone_result = self.phone_detector.predict(
                rgb_hand, use_trained=use_trained_model
            )

            render_detection_rectangle(
                frame=frame,
                text=f"Hand {crop_index}",
                xyxy=hand_xyxy,
                color="green" if not phone_result["detected"] else "pink",
            )

            if phone_result["detected"]:
                relative_xyxy = np.array(
                    phone_result["relative_xyxy"], dtype=np.float32
                )

                from_mother_wh = (
                    (
                        self.phone_detector_config.inference.image_size
                        if use_trained_model
                        else 640
                    ),
                    (
                        self.phone_detector_config.inference.image_size
                        if use_trained_model
                        else 640
                    ),
                )

                absolute_xyxy = relative_to_absolute(
                    from_mother_wh=from_mother_wh,
                    to_mother_wh=subframe_wh,
                    from_child_xyxy=relative_xyxy,
                    to_mother_xy=(hand_xyxy[0], hand_xyxy[1]),
                )

                render_detection_rectangle(
                    frame=frame,
                    text=phone_result["text"],
                    xyxy=absolute_xyxy,
                    color="pink",
                )

                return True, None

            if spare_ratio < self.inference_config.phone.spare_ratio_threshold:
                break

        return False, None

    def process_one_person(
        self,
        frame: np.ndarray,
        original_frame: np.ndarray,
        keypoints: np.ndarray,
        xyxy: np.ndarray,
        runtime_parameters: dict,
    ) -> dict:
        """
        Main entrypoint for one person in one frame.

        Returns a structure similar to your current runtime response.
        """
        start_posture = time.time()
        initial_state, initial_score_text = self._initial_state_checks(keypoints, xyxy)

        if initial_state in {STATE_OUT_OF_FRAME, STATE_BACKSIDE}:
            posture_state = initial_state
            posture_score_text = initial_score_text
        else:
            posture_state, posture_score_text = self._run_posture_stage(keypoints)

        posture_time = time.time() - start_posture

        phone_detected = False
        announced_face_frame = None
        face_xyxy = None
        phone_time = 0.0

        if posture_state == STATE_SUSPICIOUS:
            start_phone = time.time()
            phone_detected, announced_face_frame = self._run_phone_stage(
                frame=frame,
                original_frame=original_frame,
                keypoints=keypoints,
                xyxy=xyxy,
            )
            phone_time = time.time() - start_phone

        fusion_result = self.fusion.fuse(
            base_state=posture_state,
            posture_score_text=posture_score_text,
            phone_detected=phone_detected,
        )

        render_detection_rectangle(
            frame=frame,
            text=f"{fusion_result['display_text']} {fusion_result['score_text']}".strip(),
            xyxy=xyxy,
            color=fusion_result["display_color"],
        )

        # Optional face crop announce, similar to your old runtime logic
        if fusion_result["final_label"] == "distracted":
            face_center = keypoints[self.mmpose_config.keypoints.face_center_index][:2]
            bbox_width = abs(xyxy[2] - xyxy[0])
            left_ear_x = keypoints[self.mmpose_config.keypoints.left_ear_index][0]
            right_ear_x = keypoints[self.mmpose_config.keypoints.right_ear_index][0]

            face_len = max(
                abs(int((right_ear_x - left_ear_x) * 1.1)), int(0.3 * bbox_width)
            )
            face_frame, face_xyxy = crop_frame(
                original_frame, face_center, (face_len, face_len)
            )

            if face_frame is not None and face_xyxy is not None:
                last_announce_time = runtime_parameters.get(
                    "time_last_announce_face", 0.0
                )
                current_frame_time = runtime_parameters.get(
                    "time_last_record_framerate", time.time()
                )

                if (
                    current_frame_time - last_announce_time
                    > self.inference_config.phone.face_announce_interval_seconds
                ):
                    announced_face_frame = face_frame

                render_detection_rectangle(
                    frame=frame,
                    text="Face",
                    xyxy=face_xyxy,
                    color="white",
                )

        result = {
            "performance": (posture_time, phone_time),
            "announced_face_frame": announced_face_frame,
            "posture": fusion_result["final_label"],
            "phone": phone_detected,
            "state": fusion_result["state"],
            "display_text": fusion_result["display_text"],
            "score_text": fusion_result["score_text"],
            # "face_xyxy": face_xyxy.tolist() if face_xyxy is not None else None,
            "face_xyxy": (
                face_xyxy.tolist()
                if hasattr(face_xyxy, "tolist")
                else face_xyxy if face_xyxy is not None else None
            ),
        }

        self.logger.info("Runtime one-person result: %s", result)
        return result
