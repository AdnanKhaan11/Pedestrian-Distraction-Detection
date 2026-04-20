from pathlib import Path

FOLDERS = [
    "configs",
    "src",
    "src/config",
    "src/entity",
    "src/utils",
    "src/components",
    "src/pipeline",
    "src/models",
    "src/serving",
    "scripts",
    "app",
    "app/templates",
    "app/static",
    "app/static/css",
    "app/static/js",
    "app/static/uploads",
    "app/static/results",
    "tests",
    "docker",
    ".github/workflows",
    "artifacts/mmpose/checkpoints",
    "artifacts/posture_classifier/weights",
    "artifacts/posture_classifier/archive",
    "artifacts/phone_detector/pretrained",
    "artifacts/phone_detector/weights",
    "artifacts/phone_detector/archive",
    "artifacts/metrics",
    "artifacts/predictions",
    "logs",
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
]

FILES = [
    "configs/config.yaml",
    "configs/paths.yaml",
    "configs/mmpose_config.yaml",
    "configs/posture_model_config.yaml",
    "configs/phone_detector_config.yaml",
    "configs/training_config.yaml",
    "configs/inference_config.yaml",
    "src/__init__.py",
    "src/config/__init__.py",
    "src/config/constants.py",
    "src/config/configuration.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/common.py",
    "src/utils/helpers.py",
    "src/utils/opencv_utils.py",
    "src/utils/naming_utils.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/frame_extractor.py",
    "src/components/hand_cropper.py",
    "src/components/pose_feature_generator.py",
    "src/components/dataset_builder.py",
    "src/components/mmpose_loader.py",
    "src/components/posture_model.py",
    "src/components/posture_trainer.py",
    "src/components/posture_evaluator.py",
    "src/components/phone_model_loader.py",
    "src/components/phone_trainer.py",
    "src/components/phone_evaluator.py",
    "src/components/posture_detector.py",
    "src/components/phone_detector.py",
    "src/components/distraction_fusion.py",
    "src/components/runtime_detector.py",
    "src/pipeline/__init__.py",
    "src/pipeline/data_pipeline.py",
    "src/pipeline/posture_training_pipeline.py",
    "src/pipeline/phone_training_pipeline.py",
    "src/pipeline/inference_pipeline.py",
    "src/models/__init__.py",
    "src/models/posture_cnn.py",
    "src/serving/__init__.py",
    "src/serving/predictor.py",
    "src/serving/schemas.py",
    "scripts/run_data_pipeline.py",
    "scripts/train_posture_model.py",
    "scripts/train_phone_model.py",
    "scripts/run_inference.py",
    "app/api.py",
    "app/templates/index.html",
    "app/static/css/style.css",
    "app/static/js/main.js",
    "tests/test_config.py",
    "tests/test_posture_model.py",
    "tests/test_inference_pipeline.py",
    "docker/Dockerfile",
    "docker/docker-compose.yml",
    ".github/workflows/ci.yml",
    "requirements.txt",
    "README.md",
]


def create_structure():
    root = Path(".")

    print("\nUpdating project structure...\n")

    for folder in FOLDERS:
        folder_path = root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"OK folder: {folder_path}")

    for file in FILES:
        file_path = root / file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.touch()
            print(f"Created file: {file_path}")
        else:
            print(f"OK file: {file_path}")

    print("\nStructure is ready.\n")


if __name__ == "__main__":
    create_structure()
