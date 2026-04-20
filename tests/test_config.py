import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager  # noqa: E402


def test_configuration_manager_loads_yaml_files():
    manager = ConfigurationManager()
    project_config = manager.get_project_config()
    paths_config = manager.get_paths_config()
    posture_config = manager.get_posture_model_config()
    phone_config = manager.get_phone_detector_config()

    assert project_config.name
    assert hasattr(paths_config, "artifacts_root")
    assert isinstance(paths_config.artifacts_root, Path)
    assert hasattr(posture_config, "model_name")
    assert hasattr(phone_config, "framework")
