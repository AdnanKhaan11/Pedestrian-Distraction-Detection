# What this file does
# This is the config loader for the whole project.

# It:

# reads YAML files
# validates that they exist
# converts them into clean dataclass objects
# keeps the rest of the codebase simple
# This is the heart of the config layer.


from pathlib import Path
from typing import Any, Dict

import yaml

from src.config.constants import (
    CONFIG_FILE_PATH,
    INFERENCE_CONFIG_FILE_PATH,
    MMPOSE_CONFIG_FILE_PATH,
    PATHS_FILE_PATH,
    PHONE_DETECTOR_CONFIG_FILE_PATH,
    POSTURE_MODEL_CONFIG_FILE_PATH,
    TRAINING_CONFIG_FILE_PATH,
)
from src.entity.config_entity import (
    GeneralTrainingConfig,
    InferenceAPIConfig,
    InferenceConfig,
    InferenceFrontendConfig,
    InferenceGeneralConfig,
    InferencePhoneConfig,
    InferencePostureConfig,
    InferenceRenderingConfig,
    InferenceVideoConfig,
    LabelConfig,
    MMPoseConfig,
    MMPoseDetectorConfig,
    MMPoseEstimatorConfig,
    MMPoseKeypointConfig,
    MMPoseVisualizerConfig,
    PathsConfig,
    PhoneDetectorConfig,
    PhoneHandCropLogicConfig,
    PhoneInferenceConfig,
    PhoneRuntimeConfig,
    PhoneTrainingConfig,
    PhoneTrainingDataConfig,
    PhoneTrainingHyperConfig,
    PhoneTrainingOutputConfig,
    PhoneWeightsConfig,
    PostureArchitectureConfig,
    PostureFeatureEngineeringConfig,
    PostureInferenceConfig,
    PostureInputShapeConfig,
    PostureModelConfig,
    PostureTrainingConfig,
    PostureTrainingDataConfig,
    PostureTrainingHyperConfig,
    PostureTrainingOutputConfig,
    PostureWeightsConfig,
    ProjectConfig,
    RuntimeConfig,
    SystemConfig,
    TrackingConfig,
    TrainingConfig,
    TrainingTrackingConfig,
)


class ConfigurationManager:
    """
    This class loads all YAML config files
    and converts them into clean dataclass objects.
    """

    def __init__(
        self,
        config_file_path: Path = CONFIG_FILE_PATH,
        paths_file_path: Path = PATHS_FILE_PATH,
        mmpose_config_file_path: Path = MMPOSE_CONFIG_FILE_PATH,
        posture_model_config_file_path: Path = POSTURE_MODEL_CONFIG_FILE_PATH,
        phone_detector_config_file_path: Path = PHONE_DETECTOR_CONFIG_FILE_PATH,
        training_config_file_path: Path = TRAINING_CONFIG_FILE_PATH,
        inference_config_file_path: Path = INFERENCE_CONFIG_FILE_PATH,
    ) -> None:
        self.config_data = self._read_yaml(config_file_path)
        self.paths_data = self._read_yaml(paths_file_path)
        self.mmpose_data = self._read_yaml(mmpose_config_file_path)
        self.posture_model_data = self._read_yaml(posture_model_config_file_path)
        self.phone_detector_data = self._read_yaml(phone_detector_config_file_path)
        self.training_data = self._read_yaml(training_config_file_path)
        self.inference_data = self._read_yaml(inference_config_file_path)

    @staticmethod
    def _read_yaml(file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file)

        if data is None:
            raise ValueError(f"Config file is empty: {file_path}")

        return data

    @staticmethod
    def _to_path(path_str: str) -> Path:
        return Path(path_str)

    def get_project_config(self) -> ProjectConfig:
        project = self.config_data["project"]
        return ProjectConfig(
            name=project["name"],
            version=project["version"],
            description=project["description"],
            author=project["author"],
            random_seed=project["random_seed"],
        )

    def get_system_config(self) -> SystemConfig:
        system = self.config_data["system"]
        return SystemConfig(
            device=system["device"],
            num_workers=system["num_workers"],
            pin_memory=system["pin_memory"],
            log_level=system["log_level"],
        )

    def get_tracking_config(self) -> TrackingConfig:
        tracking = self.config_data["tracking"]
        return TrackingConfig(
            enable_mlflow=tracking["enable_mlflow"],
            experiment_name=tracking["experiment_name"],
            tracking_uri=tracking["tracking_uri"],
        )

    def get_runtime_config(self) -> RuntimeConfig:
        runtime = self.config_data["runtime"]
        return RuntimeConfig(
            use_pretrained_phone_detector_fallback=runtime[
                "use_pretrained_phone_detector_fallback"
            ],
            save_prediction_images=runtime["save_prediction_images"],
            save_runtime_hand_crops=runtime["save_runtime_hand_crops"],
            enable_debug_visuals=runtime["enable_debug_visuals"],
            allow_multiple_persons=runtime["allow_multiple_persons"],
        )

    def get_label_config(self) -> LabelConfig:
        labels = self.config_data["labels"]
        return LabelConfig(
            posture_classes=labels["posture_classes"],
            phone_classes=labels["phone_classes"],
            final_classes=labels["final_classes"],
        )

    def get_paths_config(self) -> PathsConfig:
        paths = self.paths_data["paths"]
        files = self.paths_data["files"]

        return PathsConfig(
            artifacts_root=self._to_path(paths["artifacts_root"]),
            logs_dir=self._to_path(paths["logs_dir"]),
            data_root=self._to_path(paths["data_root"]),
            raw_data_dir=self._to_path(paths["raw_data_dir"]),
            interim_data_dir=self._to_path(paths["interim_data_dir"]),
            processed_data_dir=self._to_path(paths["processed_data_dir"]),
            mmpose_checkpoint_dir=self._to_path(paths["mmpose_checkpoint_dir"]),
            posture_model_dir=self._to_path(paths["posture_model_dir"]),
            posture_weights_dir=self._to_path(paths["posture_weights_dir"]),
            posture_archive_dir=self._to_path(paths["posture_archive_dir"]),
            phone_model_dir=self._to_path(paths["phone_model_dir"]),
            phone_pretrained_dir=self._to_path(paths["phone_pretrained_dir"]),
            phone_weights_dir=self._to_path(paths["phone_weights_dir"]),
            phone_archive_dir=self._to_path(paths["phone_archive_dir"]),
            metrics_dir=self._to_path(paths["metrics_dir"]),
            predictions_dir=self._to_path(paths["predictions_dir"]),
            raw_video_dir=self._to_path(paths["raw_video_dir"]),
            raw_image_dir=self._to_path(paths["raw_image_dir"]),
            posture_feature_dir=self._to_path(paths["posture_feature_dir"]),
            hand_crop_dir=self._to_path(paths["hand_crop_dir"]),
            phone_dataset_dir=self._to_path(paths["phone_dataset_dir"]),
            frontend_upload_dir=self._to_path(paths["frontend_upload_dir"]),
            frontend_result_dir=self._to_path(paths["frontend_result_dir"]),
            person_detector_checkpoint=self._to_path(
                files["person_detector_checkpoint"]
            ),
            pose_estimator_checkpoint=self._to_path(files["pose_estimator_checkpoint"]),
            default_posture_model=self._to_path(files["default_posture_model"]),
            archived_posture_model=self._to_path(files["archived_posture_model"]),
            default_phone_model=self._to_path(files["default_phone_model"]),
            fallback_phone_model=self._to_path(files["fallback_phone_model"]),
        )

    def get_mmpose_config(self) -> MMPoseConfig:
        mmpose = self.mmpose_data["mmpose"]

        detector = MMPoseDetectorConfig(
            config_file=mmpose["detector"]["config_file"],
            checkpoint_file=self._to_path(mmpose["detector"]["checkpoint_file"]),
            category_id=mmpose["detector"]["category_id"],
            bbox_threshold_multi_person=mmpose["detector"][
                "bbox_threshold_multi_person"
            ],
            bbox_threshold_single_person=mmpose["detector"][
                "bbox_threshold_single_person"
            ],
            nms_threshold=mmpose["detector"]["nms_threshold"],
        )

        pose_estimator = MMPoseEstimatorConfig(
            config_file=mmpose["pose_estimator"]["config_file"],
            checkpoint_file=self._to_path(mmpose["pose_estimator"]["checkpoint_file"]),
        )

        visualizer = MMPoseVisualizerConfig(
            draw_bbox=mmpose["visualizer"]["draw_bbox"],
            draw_heatmap=mmpose["visualizer"]["draw_heatmap"],
            keypoint_threshold=mmpose["visualizer"]["keypoint_threshold"],
            radius=mmpose["visualizer"]["radius"],
            alpha=mmpose["visualizer"]["alpha"],
            thickness=mmpose["visualizer"]["thickness"],
            skeleton_style=mmpose["visualizer"]["skeleton_style"],
        )

        keypoints = MMPoseKeypointConfig(
            use_first_n_keypoints=mmpose["keypoints"]["use_first_n_keypoints"],
            left_shoulder_index=mmpose["keypoints"]["left_shoulder_index"],
            right_shoulder_index=mmpose["keypoints"]["right_shoulder_index"],
            left_elbow_index=mmpose["keypoints"]["left_elbow_index"],
            right_elbow_index=mmpose["keypoints"]["right_elbow_index"],
            left_wrist_index=mmpose["keypoints"]["left_wrist_index"],
            right_wrist_index=mmpose["keypoints"]["right_wrist_index"],
            left_ear_index=mmpose["keypoints"]["left_ear_index"],
            right_ear_index=mmpose["keypoints"]["right_ear_index"],
            face_center_index=mmpose["keypoints"]["face_center_index"],
        )

        return MMPoseConfig(
            device_preference=mmpose["device_preference"],
            detector=detector,
            pose_estimator=pose_estimator,
            visualizer=visualizer,
            keypoints=keypoints,
        )

    def get_posture_model_config(self) -> PostureModelConfig:
        posture = self.posture_model_data["posture_model"]

        input_shape = PostureInputShapeConfig(
            channels=posture["input_shape"]["channels"],
            depth=posture["input_shape"]["depth"],
            height=posture["input_shape"]["height"],
            width=posture["input_shape"]["width"],
        )

        architecture = PostureArchitectureConfig(
            conv_kernel_size=posture["architecture"]["conv_kernel_size"],
            pool_kernel_size=posture["architecture"]["pool_kernel_size"],
            activation=posture["architecture"]["activation"],
            fc_dims=posture["architecture"]["fc_dims"],
        )

        inference = PostureInferenceConfig(
            confidence_threshold=posture["inference"]["confidence_threshold"],
            class_names=posture["inference"]["class_names"],
        )

        weights = PostureWeightsConfig(
            default_weight_file=self._to_path(
                posture["weights"]["default_weight_file"]
            ),
            archive_weight_file=self._to_path(
                posture["weights"]["archive_weight_file"]
            ),
        )

        feature_engineering = PostureFeatureEngineeringConfig(
            feature_type=posture["feature_engineering"]["feature_type"],
            normalize_per_channel=posture["feature_engineering"][
                "normalize_per_channel"
            ],
            transpose_to_ncdhw=posture["feature_engineering"]["transpose_to_ncdhw"],
        )

        return PostureModelConfig(
            model_name=posture["model_name"],
            task_type=posture["task_type"],
            input_channels=posture["input_channels"],
            output_classes=posture["output_classes"],
            input_shape=input_shape,
            architecture=architecture,
            inference=inference,
            weights=weights,
            feature_engineering=feature_engineering,
        )

    def get_phone_detector_config(self) -> PhoneDetectorConfig:
        phone = self.phone_detector_data["phone_detector"]

        weights = PhoneWeightsConfig(
            default_weight_file=self._to_path(phone["weights"]["default_weight_file"]),
            fallback_weight_file=self._to_path(
                phone["weights"]["fallback_weight_file"]
            ),
        )

        inference = PhoneInferenceConfig(
            image_size=phone["inference"]["image_size"],
            confidence_threshold_trained=phone["inference"][
                "confidence_threshold_trained"
            ],
            confidence_threshold_fallback=phone["inference"][
                "confidence_threshold_fallback"
            ],
            use_trained_model_by_default=phone["inference"][
                "use_trained_model_by_default"
            ],
            trained_model_phone_class_index=phone["inference"][
                "trained_model_phone_class_index"
            ],
            fallback_model_phone_class_index=phone["inference"][
                "fallback_model_phone_class_index"
            ],
        )

        hand_crop_logic = PhoneHandCropLogicConfig(
            far_body_hand_ratio=phone["hand_crop_logic"]["far_body_hand_ratio"],
            near_body_hand_ratio=phone["hand_crop_logic"]["near_body_hand_ratio"],
            merge_hand_distance_ratio=phone["hand_crop_logic"][
                "merge_hand_distance_ratio"
            ],
            hand_extension_ratio=phone["hand_crop_logic"]["hand_extension_ratio"],
            spare_ratio_threshold=phone["hand_crop_logic"]["spare_ratio_threshold"],
        )

        runtime = PhoneRuntimeConfig(
            save_runtime_handframes=phone["runtime"]["save_runtime_handframes"],
            runtime_handframe_format=phone["runtime"]["runtime_handframe_format"],
        )

        return PhoneDetectorConfig(
            framework=phone["framework"],
            task_type=phone["task_type"],
            weights=weights,
            inference=inference,
            hand_crop_logic=hand_crop_logic,
            runtime=runtime,
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.training_data["training"]

        general = GeneralTrainingConfig(
            random_seed=training["general"]["random_seed"],
            device_preference=training["general"]["device_preference"],
            num_workers=training["general"]["num_workers"],
            pin_memory=training["general"]["pin_memory"],
        )

        posture_training = PostureTrainingConfig(
            enabled=training["posture_training"]["enabled"],
            use_existing_weights=training["posture_training"]["use_existing_weights"],
            existing_weights_path=self._to_path(
                training["posture_training"]["existing_weights_path"]
            ),
            data=PostureTrainingDataConfig(
                source_feature_dir=self._to_path(
                    training["posture_training"]["data"]["source_feature_dir"]
                ),
                train_split=training["posture_training"]["data"]["train_split"],
                val_split=training["posture_training"]["data"]["val_split"],
                test_split=training["posture_training"]["data"]["test_split"],
                shuffle=training["posture_training"]["data"]["shuffle"],
            ),
            hyperparameters=PostureTrainingHyperConfig(
                epochs=training["posture_training"]["hyperparameters"]["epochs"],
                batch_size=training["posture_training"]["hyperparameters"][
                    "batch_size"
                ],
                learning_rate=training["posture_training"]["hyperparameters"][
                    "learning_rate"
                ],
                weight_decay=training["posture_training"]["hyperparameters"][
                    "weight_decay"
                ],
                early_stopping_patience=training["posture_training"]["hyperparameters"][
                    "early_stopping_patience"
                ],
                min_delta=training["posture_training"]["hyperparameters"]["min_delta"],
            ),
            outputs=PostureTrainingOutputConfig(
                save_best_model_as=training["posture_training"]["outputs"][
                    "save_best_model_as"
                ],
                save_last_model_as=training["posture_training"]["outputs"][
                    "save_last_model_as"
                ],
                metrics_file_name=training["posture_training"]["outputs"][
                    "metrics_file_name"
                ],
                history_file_name=training["posture_training"]["outputs"][
                    "history_file_name"
                ],
            ),
        )

        phone_training = PhoneTrainingConfig(
            enabled=training["phone_training"]["enabled"],
            use_existing_weights=training["phone_training"]["use_existing_weights"],
            existing_weights_path=self._to_path(
                training["phone_training"]["existing_weights_path"]
            ),
            data=PhoneTrainingDataConfig(
                source_dataset_dir=self._to_path(
                    training["phone_training"]["data"]["source_dataset_dir"]
                ),
            ),
            hyperparameters=PhoneTrainingHyperConfig(
                epochs=training["phone_training"]["hyperparameters"]["epochs"],
                batch_size=training["phone_training"]["hyperparameters"]["batch_size"],
                image_size=training["phone_training"]["hyperparameters"]["image_size"],
                learning_rate=training["phone_training"]["hyperparameters"][
                    "learning_rate"
                ],
            ),
            outputs=PhoneTrainingOutputConfig(
                save_best_model_as=training["phone_training"]["outputs"][
                    "save_best_model_as"
                ],
                metrics_file_name=training["phone_training"]["outputs"][
                    "metrics_file_name"
                ],
            ),
        )

        tracking = TrainingTrackingConfig(
            enable_mlflow=training["tracking"]["enable_mlflow"],
            experiment_name_posture=training["tracking"]["experiment_name_posture"],
            experiment_name_phone=training["tracking"]["experiment_name_phone"],
            tracking_uri=training["tracking"]["tracking_uri"],
        )

        return TrainingConfig(
            general=general,
            posture_training=posture_training,
            phone_training=phone_training,
            tracking=tracking,
        )

    def get_inference_config(self) -> InferenceConfig:
        inference = self.inference_data["inference"]

        return InferenceConfig(
            general=InferenceGeneralConfig(
                device_preference=inference["general"]["device_preference"],
                allow_multiple_persons=inference["general"]["allow_multiple_persons"],
                max_persons_per_frame=inference["general"]["max_persons_per_frame"],
                save_prediction_images=inference["general"]["save_prediction_images"],
                save_runtime_hand_crops=inference["general"]["save_runtime_hand_crops"],
            ),
            video=InferenceVideoConfig(
                default_frame_width=inference["video"]["default_frame_width"],
                default_frame_height=inference["video"]["default_frame_height"],
                websocket_frame_width=inference["video"]["websocket_frame_width"],
                websocket_frame_height=inference["video"]["websocket_frame_height"],
            ),
            posture=InferencePostureConfig(
                confidence_threshold=inference["posture"]["confidence_threshold"],
                out_of_frame_missing_keypoints_threshold=inference["posture"][
                    "out_of_frame_missing_keypoints_threshold"
                ],
                keypoint_score_threshold=inference["posture"][
                    "keypoint_score_threshold"
                ],
                backside_ratio_threshold=inference["posture"][
                    "backside_ratio_threshold"
                ],
            ),
            phone=InferencePhoneConfig(
                trained_model_confidence=inference["phone"]["trained_model_confidence"],
                fallback_model_confidence=inference["phone"][
                    "fallback_model_confidence"
                ],
                face_announce_interval_seconds=inference["phone"][
                    "face_announce_interval_seconds"
                ],
                spare_ratio_threshold=inference["phone"]["spare_ratio_threshold"],
            ),
            rendering=InferenceRenderingConfig(
                show_fps=inference["rendering"]["show_fps"],
                show_current_time=inference["rendering"]["show_current_time"],
                show_last_announce_time=inference["rendering"][
                    "show_last_announce_time"
                ],
                show_person_bbox=inference["rendering"]["show_person_bbox"],
                show_hand_bbox=inference["rendering"]["show_hand_bbox"],
                show_face_bbox=inference["rendering"]["show_face_bbox"],
                window_title=inference["rendering"]["window_title"],
            ),
            api=InferenceAPIConfig(
                host=inference["api"]["host"],
                port=inference["api"]["port"],
                reload=inference["api"]["reload"],
            ),
            frontend=InferenceFrontendConfig(
                app_title=inference["frontend"]["app_title"],
                upload_folder=self._to_path(inference["frontend"]["upload_folder"]),
                result_folder=self._to_path(inference["frontend"]["result_folder"]),
            ),
        )
