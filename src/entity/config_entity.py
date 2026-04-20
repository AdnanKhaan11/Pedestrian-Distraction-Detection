# What this file does
# This file turns YAML settings into structured config objects.

# Why this is useful:

# cleaner code
# less bugs
# easier auto-complete
# easier debugging
# easier to explain in FYP presentation


from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    description: str
    author: str
    random_seed: int


@dataclass(frozen=True)
class SystemConfig:
    device: str
    num_workers: int
    pin_memory: bool
    log_level: str


@dataclass(frozen=True)
class TrackingConfig:
    enable_mlflow: bool
    experiment_name: str
    tracking_uri: str


@dataclass(frozen=True)
class RuntimeConfig:
    use_pretrained_phone_detector_fallback: bool
    save_prediction_images: bool
    save_runtime_hand_crops: bool
    enable_debug_visuals: bool
    allow_multiple_persons: bool


@dataclass(frozen=True)
class LabelConfig:
    posture_classes: List[str]
    phone_classes: List[str]
    final_classes: List[str]


@dataclass(frozen=True)
class PathsConfig:
    artifacts_root: Path
    logs_dir: Path
    data_root: Path
    raw_data_dir: Path
    interim_data_dir: Path
    processed_data_dir: Path
    mmpose_checkpoint_dir: Path
    posture_model_dir: Path
    posture_weights_dir: Path
    posture_archive_dir: Path
    phone_model_dir: Path
    phone_pretrained_dir: Path
    phone_weights_dir: Path
    phone_archive_dir: Path
    metrics_dir: Path
    predictions_dir: Path
    raw_video_dir: Path
    raw_image_dir: Path
    posture_feature_dir: Path
    hand_crop_dir: Path
    phone_dataset_dir: Path
    frontend_upload_dir: Path
    frontend_result_dir: Path
    person_detector_checkpoint: Path
    pose_estimator_checkpoint: Path
    default_posture_model: Path
    archived_posture_model: Path
    default_phone_model: Path
    fallback_phone_model: Path


@dataclass(frozen=True)
class MMPoseDetectorConfig:
    config_file: str
    checkpoint_file: Path
    category_id: int
    bbox_threshold_multi_person: float
    bbox_threshold_single_person: float
    nms_threshold: float


@dataclass(frozen=True)
class MMPoseEstimatorConfig:
    config_file: str
    checkpoint_file: Path


@dataclass(frozen=True)
class MMPoseVisualizerConfig:
    draw_bbox: bool
    draw_heatmap: bool
    keypoint_threshold: float
    radius: int
    alpha: float
    thickness: int
    skeleton_style: str


@dataclass(frozen=True)
class MMPoseKeypointConfig:
    use_first_n_keypoints: int
    left_shoulder_index: int
    right_shoulder_index: int
    left_elbow_index: int
    right_elbow_index: int
    left_wrist_index: int
    right_wrist_index: int
    left_ear_index: int
    right_ear_index: int
    face_center_index: int


@dataclass(frozen=True)
class MMPoseConfig:
    device_preference: str
    detector: MMPoseDetectorConfig
    pose_estimator: MMPoseEstimatorConfig
    visualizer: MMPoseVisualizerConfig
    keypoints: MMPoseKeypointConfig


@dataclass(frozen=True)
class PostureInputShapeConfig:
    channels: int
    depth: int
    height: int
    width: int


@dataclass(frozen=True)
class PostureArchitectureConfig:
    conv_kernel_size: List[int]
    pool_kernel_size: int
    activation: str
    fc_dims: List[int]


@dataclass(frozen=True)
class PostureInferenceConfig:
    confidence_threshold: float
    class_names: List[str]


@dataclass(frozen=True)
class PostureWeightsConfig:
    default_weight_file: Path
    archive_weight_file: Path


@dataclass(frozen=True)
class PostureFeatureEngineeringConfig:
    feature_type: str
    normalize_per_channel: bool
    transpose_to_ncdhw: bool


@dataclass(frozen=True)
class PostureModelConfig:
    model_name: str
    task_type: str
    input_channels: int
    output_classes: int
    input_shape: PostureInputShapeConfig
    architecture: PostureArchitectureConfig
    inference: PostureInferenceConfig
    weights: PostureWeightsConfig
    feature_engineering: PostureFeatureEngineeringConfig


@dataclass(frozen=True)
class PhoneWeightsConfig:
    default_weight_file: Path
    fallback_weight_file: Path


@dataclass(frozen=True)
class PhoneInferenceConfig:
    image_size: int
    confidence_threshold_trained: float
    confidence_threshold_fallback: float
    use_trained_model_by_default: bool
    trained_model_phone_class_index: int
    fallback_model_phone_class_index: int


@dataclass(frozen=True)
class PhoneHandCropLogicConfig:
    far_body_hand_ratio: float
    near_body_hand_ratio: float
    merge_hand_distance_ratio: float
    hand_extension_ratio: float
    spare_ratio_threshold: float


@dataclass(frozen=True)
class PhoneRuntimeConfig:
    save_runtime_handframes: bool
    runtime_handframe_format: str


@dataclass(frozen=True)
class PhoneDetectorConfig:
    framework: str
    task_type: str
    weights: PhoneWeightsConfig
    inference: PhoneInferenceConfig
    hand_crop_logic: PhoneHandCropLogicConfig
    runtime: PhoneRuntimeConfig


@dataclass(frozen=True)
class GeneralTrainingConfig:
    random_seed: int
    device_preference: str
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class PostureTrainingDataConfig:
    source_feature_dir: Path
    train_split: float
    val_split: float
    test_split: float
    shuffle: bool


@dataclass(frozen=True)
class PostureTrainingHyperConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    min_delta: float


@dataclass(frozen=True)
class PostureTrainingOutputConfig:
    save_best_model_as: str
    save_last_model_as: str
    metrics_file_name: str
    history_file_name: str


@dataclass(frozen=True)
class PostureTrainingConfig:
    enabled: bool
    use_existing_weights: bool
    existing_weights_path: Path
    data: PostureTrainingDataConfig
    hyperparameters: PostureTrainingHyperConfig
    outputs: PostureTrainingOutputConfig


@dataclass(frozen=True)
class PhoneTrainingDataConfig:
    source_dataset_dir: Path


@dataclass(frozen=True)
class PhoneTrainingHyperConfig:
    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float


@dataclass(frozen=True)
class PhoneTrainingOutputConfig:
    save_best_model_as: str
    metrics_file_name: str


@dataclass(frozen=True)
class PhoneTrainingConfig:
    enabled: bool
    use_existing_weights: bool
    existing_weights_path: Path
    data: PhoneTrainingDataConfig
    hyperparameters: PhoneTrainingHyperConfig
    outputs: PhoneTrainingOutputConfig


@dataclass(frozen=True)
class TrainingTrackingConfig:
    enable_mlflow: bool
    experiment_name_posture: str
    experiment_name_phone: str
    tracking_uri: str


@dataclass(frozen=True)
class TrainingConfig:
    general: GeneralTrainingConfig
    posture_training: PostureTrainingConfig
    phone_training: PhoneTrainingConfig
    tracking: TrainingTrackingConfig


@dataclass(frozen=True)
class InferenceGeneralConfig:
    device_preference: str
    allow_multiple_persons: bool
    max_persons_per_frame: int
    save_prediction_images: bool
    save_runtime_hand_crops: bool


@dataclass(frozen=True)
class InferenceVideoConfig:
    default_frame_width: int
    default_frame_height: int
    websocket_frame_width: int
    websocket_frame_height: int


@dataclass(frozen=True)
class InferencePostureConfig:
    confidence_threshold: float
    out_of_frame_missing_keypoints_threshold: int
    keypoint_score_threshold: float
    backside_ratio_threshold: float


@dataclass(frozen=True)
class InferencePhoneConfig:
    trained_model_confidence: float
    fallback_model_confidence: float
    face_announce_interval_seconds: int
    spare_ratio_threshold: float


@dataclass(frozen=True)
class InferenceRenderingConfig:
    show_fps: bool
    show_current_time: bool
    show_last_announce_time: bool
    show_person_bbox: bool
    show_hand_bbox: bool
    show_face_bbox: bool
    window_title: str


@dataclass(frozen=True)
class InferenceAPIConfig:
    host: str
    port: int
    reload: bool


@dataclass(frozen=True)
class InferenceFrontendConfig:
    app_title: str
    upload_folder: Path
    result_folder: Path


@dataclass(frozen=True)
class InferenceConfig:
    general: InferenceGeneralConfig
    video: InferenceVideoConfig
    posture: InferencePostureConfig
    phone: InferencePhoneConfig
    rendering: InferenceRenderingConfig
    api: InferenceAPIConfig
    frontend: InferenceFrontendConfig
