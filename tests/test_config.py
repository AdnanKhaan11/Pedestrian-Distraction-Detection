"""
These tests validate the new YAML-driven configuration layer.
They are written for the refactored MLOps structure under `src/`
instead of the old research-style folders. If the refactor files
have not been copied yet, the tests skip cleanly instead of failing.
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_configuration_manager_loads_core_configs_from_yaml(tmp_path: Path):
    configuration_module = pytest.importorskip("src.config.configuration")

    config_file = tmp_path / "configs" / "config.yaml"
    paths_file = tmp_path / "configs" / "paths.yaml"
    mmpose_file = tmp_path / "configs" / "mmpose_config.yaml"
    posture_file = tmp_path / "configs" / "posture_model_config.yaml"
    phone_file = tmp_path / "configs" / "phone_detector_config.yaml"
    training_file = tmp_path / "configs" / "training_config.yaml"
    inference_file = tmp_path / "configs" / "inference_config.yaml"

    _write_yaml(
        config_file,
        {
            "project": {
                "name": "pedestrian-distraction-detection",
                "version": "1.0.0",
                "description": "Test project",
                "author": "FYP Team",
                "random_seed": 42,
            },
            "system": {
                "device": "cpu",
                "num_workers": 0,
                "pin_memory": False,
                "log_level": "INFO",
            },
            "tracking": {
                "enable_mlflow": True,
                "experiment_name": "pedestrian_distraction_experiment",
                "tracking_uri": "https://dagshub.com/example/repo.mlflow",
            },
            "runtime": {
                "use_pretrained_phone_detector_fallback": True,
                "save_prediction_images": True,
                "save_runtime_hand_crops": False,
                "enable_debug_visuals": False,
                "allow_multiple_persons": True,
            },
            "labels": {
                "posture_classes": ["not_using", "using_or_suspicious"],
                "phone_classes": ["phone"],
                "final_classes": ["not_distracted", "distracted"],
            },
        },
    )

    _write_yaml(
        paths_file,
        {
            "paths": {
                "artifacts_root": "artifacts",
                "logs_dir": "logs",
                "data_root": "data",
                "raw_data_dir": "data/raw",
                "interim_data_dir": "data/interim",
                "processed_data_dir": "data/processed",
                "mmpose_checkpoint_dir": "artifacts/mmpose/checkpoints",
                "posture_model_dir": "artifacts/posture_classifier",
                "posture_weights_dir": "artifacts/posture_classifier/weights",
                "posture_archive_dir": "artifacts/posture_classifier/archive",
                "phone_model_dir": "artifacts/phone_detector",
                "phone_pretrained_dir": "artifacts/phone_detector/pretrained",
                "phone_weights_dir": "artifacts/phone_detector/weights",
                "phone_archive_dir": "artifacts/phone_detector/archive",
                "metrics_dir": "artifacts/metrics",
                "predictions_dir": "artifacts/predictions",
                "raw_video_dir": "data/raw/videos",
                "raw_image_dir": "data/raw/images",
                "posture_feature_dir": "data/processed/posture_features",
                "hand_crop_dir": "data/interim/hand_crops",
                "phone_dataset_dir": "data/processed/phone_dataset",
                "frontend_upload_dir": "app/static/uploads",
                "frontend_result_dir": "app/static/results",
            },
            "files": {
                "person_detector_checkpoint": "artifacts/mmpose/checkpoints/detector.pth",
                "pose_estimator_checkpoint": "artifacts/mmpose/checkpoints/pose.pth",
                "default_posture_model": "artifacts/posture_classifier/weights/best_posture_model.pth",
                "archived_posture_model": "artifacts/posture_classifier/archive/posture_old.pth",
                "default_phone_model": "artifacts/phone_detector/weights/best.pt",
                "fallback_phone_model": "artifacts/phone_detector/pretrained/yolo11n.pt",
            },
        },
    )

    _write_yaml(
        mmpose_file,
        {
            "mmpose": {
                "device_preference": "auto",
                "detector": {
                    "config_file": "model_config/configs/rtmdet_nano_320-8xb32_coco-person.py",
                    "checkpoint_file": "artifacts/mmpose/checkpoints/detector.pth",
                    "category_id": 0,
                    "bbox_threshold_multi_person": 0.3,
                    "bbox_threshold_single_person": 0.85,
                    "nms_threshold": 0.3,
                },
                "pose_estimator": {
                    "config_file": "model_config/configs/rtmpose-t_8xb256-420e_coco-256x192.py",
                    "checkpoint_file": "artifacts/mmpose/checkpoints/pose.pth",
                },
                "visualizer": {
                    "draw_bbox": True,
                    "radius": 3,
                    "alpha": 0.8,
                    "thickness": 1,
                    "skeleton_style": "mmpose",
                    "draw_heatmap": False,
                    "keypoint_threshold": 0.3,
                },
                "keypoints": {
                    "use_first_n_keypoints": 13,
                    "left_shoulder_index": 5,
                    "right_shoulder_index": 6,
                    "left_elbow_index": 7,
                    "right_elbow_index": 8,
                    "left_wrist_index": 9,
                    "right_wrist_index": 10,
                    "left_ear_index": 3,
                    "right_ear_index": 4,
                    "face_center_index": 0,
                },
            }
        },
    )

    _write_yaml(
        posture_file,
        {
            "posture_model": {
                "model_name": "MLP3d",
                "task_type": "binary_classification",
                "input_channels": 2,
                "output_classes": 2,
                "input_shape": {
                    "channels": 2,
                    "depth": 7,
                    "height": 12,
                    "width": 11,
                },
                "architecture": {
                    "conv_kernel_size": [3, 5, 5],
                    "pool_kernel_size": 2,
                    "activation": "SiLU",
                    "fc_dims": [7392, 1848, 256],
                },
                "inference": {
                    "confidence_threshold": 0.75,
                    "class_names": ["not_using", "using_or_suspicious"],
                },
                "weights": {
                    "default_weight_file": "artifacts/posture_classifier/weights/best_posture_model.pth",
                    "archive_weight_file": "artifacts/posture_classifier/archive/posture_old.pth",
                },
                "feature_engineering": {
                    "feature_type": "angle_score_cube",
                    "normalize_per_channel": True,
                    "transpose_to_ncdhw": True,
                },
            }
        },
    )

    _write_yaml(
        phone_file,
        {
            "phone_detector": {
                "framework": "ultralytics_yolo",
                "task_type": "phone_detection",
                "weights": {
                    "default_weight_file": "artifacts/phone_detector/weights/best.pt",
                    "fallback_weight_file": "artifacts/phone_detector/pretrained/yolo11n.pt",
                },
                "inference": {
                    "image_size": 128,
                    "confidence_threshold_trained": 0.65,
                    "confidence_threshold_fallback": 0.35,
                    "use_trained_model_by_default": True,
                    "trained_model_phone_class_index": 0,
                    "fallback_model_phone_class_index": 67,
                },
                "hand_crop_logic": {
                    "far_body_hand_ratio": 0.70,
                    "near_body_hand_ratio": 0.45,
                    "merge_hand_distance_ratio": 0.21,
                    "hand_extension_ratio": 0.80,
                    "spare_ratio_threshold": 0.45,
                },
                "runtime": {
                    "save_runtime_handframes": False,
                    "runtime_handframe_format": "png",
                },
            }
        },
    )

    _write_yaml(
        training_file,
        {
            "training": {
                "general": {
                    "random_seed": 42,
                    "device_preference": "auto",
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "posture_training": {
                    "enabled": True,
                    "use_existing_weights": True,
                    "existing_weights_path": "artifacts/posture_classifier/weights/best_posture_model.pth",
                    "data": {
                        "source_feature_dir": "data/processed/posture_features",
                        "train_split": 0.7,
                        "val_split": 0.15,
                        "test_split": 0.15,
                        "shuffle": True,
                    },
                    "hyperparameters": {
                        "epochs": 50,
                        "batch_size": 8,
                        "learning_rate": 0.001,
                        "weight_decay": 0.0001,
                        "early_stopping_patience": 5,
                        "min_delta": 0.0001,
                    },
                    "outputs": {
                        "save_best_model_as": "best_posture_model.pth",
                        "save_last_model_as": "last_posture_model.pth",
                        "metrics_file_name": "posture_metrics.json",
                        "history_file_name": "posture_training_history.json",
                    },
                },
                "phone_training": {
                    "enabled": True,
                    "use_existing_weights": True,
                    "existing_weights_path": "artifacts/phone_detector/weights/best.pt",
                    "data": {
                        "source_dataset_dir": "data/processed/phone_dataset",
                    },
                    "hyperparameters": {
                        "epochs": 100,
                        "batch_size": 16,
                        "image_size": 128,
                        "learning_rate": 0.001,
                    },
                    "outputs": {
                        "save_best_model_as": "best.pt",
                        "metrics_file_name": "phone_metrics.json",
                    },
                },
                "tracking": {
                    "enable_mlflow": True,
                    "experiment_name_posture": "posture_classifier_training",
                    "experiment_name_phone": "phone_detector_training",
                    "tracking_uri": "https://dagshub.com/example/repo.mlflow",
                },
            }
        },
    )

    _write_yaml(
        inference_file,
        {
            "inference": {
                "general": {
                    "device_preference": "auto",
                    "allow_multiple_persons": True,
                    "max_persons_per_frame": 10,
                    "save_prediction_images": True,
                    "save_runtime_hand_crops": False,
                },
                "video": {
                    "default_frame_width": 640,
                    "default_frame_height": 480,
                    "websocket_frame_width": 384,
                    "websocket_frame_height": 288,
                },
                "posture": {
                    "confidence_threshold": 0.75,
                    "out_of_frame_missing_keypoints_threshold": 5,
                    "keypoint_score_threshold": 0.30,
                    "backside_ratio_threshold": -0.20,
                },
                "phone": {
                    "trained_model_confidence": 0.65,
                    "fallback_model_confidence": 0.35,
                    "face_announce_interval_seconds": 5,
                    "spare_ratio_threshold": 0.45,
                },
                "rendering": {
                    "show_fps": True,
                    "show_current_time": True,
                    "show_last_announce_time": True,
                    "show_person_bbox": True,
                    "show_hand_bbox": True,
                    "show_face_bbox": True,
                    "window_title": "Pedestrian Distraction Detection",
                },
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "reload": True,
                },
                "frontend": {
                    "app_title": "Pedestrian Distraction Detection",
                    "upload_folder": "app/static/uploads",
                    "result_folder": "app/static/results",
                },
            }
        },
    )

    manager = configuration_module.ConfigurationManager(
        config_file_path=config_file,
        paths_file_path=paths_file,
        mmpose_config_file_path=mmpose_file,
        posture_model_config_file_path=posture_file,
        phone_detector_config_file_path=phone_file,
        training_config_file_path=training_file,
        inference_config_file_path=inference_file,
    )

    project_config = manager.get_project_config()
    paths_config = manager.get_paths_config()
    posture_config = manager.get_posture_model_config()
    inference_config = manager.get_inference_config()

    assert project_config.name == "pedestrian-distraction-detection"
    assert project_config.random_seed == 42
    assert paths_config.logs_dir.name == "logs"
    assert posture_config.model_name == "MLP3d"
    assert posture_config.input_shape.depth == 7
    assert inference_config.general.max_persons_per_frame == 10


def test_config_constants_point_to_expected_yaml_names():
    constants_module = pytest.importorskip("src.config.constants")

    assert constants_module.CONFIG_FILE_PATH.name == "config.yaml"
    assert constants_module.PATHS_FILE_PATH.name == "paths.yaml"
    assert constants_module.MMPOSE_CONFIG_FILE_PATH.name == "mmpose_config.yaml"
    assert (
        constants_module.POSTURE_MODEL_CONFIG_FILE_PATH.name
        == "posture_model_config.yaml"
    )
    assert (
        constants_module.PHONE_DETECTOR_CONFIG_FILE_PATH.name
        == "phone_detector_config.yaml"
    )
    assert constants_module.TRAINING_CONFIG_FILE_PATH.name == "training_config.yaml"
    assert constants_module.INFERENCE_CONFIG_FILE_PATH.name == "inference_config.yaml"
