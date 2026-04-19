import os
from pathlib import Path

PROJECT_NAME = "pedestrian-phone-detection"

PROJECT_FOLDERS = [
    "configs",
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/components",
    "src/pipeline",
    "src/utils",
    "src/config",
    "src/entity",
    "models/checkpoints",
    "logs",
    "artifacts",
    "app",
    "scripts",
    "tests",
    "docker",
    ".github/workflows",
]

PROJECT_FILES = [
    "README.md",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    ".env",
    "configs/config.yaml",
    "configs/model_config.yaml",
    "configs/training_config.yaml",
    "configs/paths.yaml",
    "notebooks/experiments.ipynb",
    "src/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_preprocessing.py",
    "src/components/dataset_builder.py",
    "src/components/model_loader.py",
    "src/components/trainer.py",
    "src/components/evaluator.py",
    "src/components/detector.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/inference_pipeline.py",
    "src/utils/logger.py",
    "src/utils/common.py",
    "src/utils/helpers.py",
    "src/config/configuration.py",
    "src/config/constants.py",
    "src/entity/config_entity.py",
    "app/app.py",
    "app/api.py",
    "scripts/train.py",
    "scripts/inference.py",
    "tests/test_pipeline.py",
    "docker/Dockerfile",
    "docker/docker-compose.yml",
    ".github/workflows/ci.yml",
]


def create_structure():
    print("\n🚀 Creating project structure...\n")

    for folder in PROJECT_FOLDERS:
        Path(folder).mkdir(parents=True, exist_ok=True)

    for file in PROJECT_FILES:
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.touch()

    print("\n✅ Structure created successfully!")


if __name__ == "__main__":
    create_structure()
