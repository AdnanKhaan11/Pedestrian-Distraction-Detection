import os
from pathlib import Path

# -------------------------------
# 📁 FOLDERS
# -------------------------------
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
    "tests",
    "docker",
    ".github/workflows",
    "artifacts/posture",
    "artifacts/phone",
    "artifacts/metrics",
    "artifacts/predictions",
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
]

# -------------------------------
# 📄 FILES
# -------------------------------
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
    "app/streamlit_app.py",
    "tests/test_config.py",
    "tests/test_posture_model.py",
    "tests/test_inference_pipeline.py",
    "docker/Dockerfile",
    "docker/docker-compose.yml",
    ".github/workflows/ci.yml",
    "requirements.txt",
    "README.md",
]


# -------------------------------
# 🚀 CREATE STRUCTURE
# -------------------------------
def create_structure():
    root = Path(".")  # 👈 current folder (IMPORTANT CHANGE)

    print("\n🚀 Updating existing project structure...\n")

    # Create folders
    for folder in FOLDERS:
        folder_path = root / folder
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {folder_path}")
        else:
            print(f"✔️ Exists: {folder_path}")

    # Create files
    for file in FILES:
        file_path = root / file
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.touch()
            print(f"📄 Created: {file_path}")
        else:
            print(f"✔️ Exists: {file_path}")

    print("\n✅ Structure updated (no overwrite, Git-safe)\n")


# -------------------------------
# ▶️ RUN
# -------------------------------
if __name__ == "__main__":
    create_structure()
