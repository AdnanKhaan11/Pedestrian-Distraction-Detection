"""
Backend configuration using Pydantic Settings.

Reads from .env file and environment variables.
All backend-specific settings are defined here.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Main backend configuration.
    Uses Pydantic BaseSettings to read from .env and environment variables.
    """

    # =====================================================
    # MongoDB Configuration
    # =====================================================
    MONGODB_URI: str
    DB_NAME: str = "pedestrian_detection"

    # =====================================================
    # API Configuration
    # =====================================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    RELOAD: bool = True
    SECRET_KEY: str = "change-me-in-production"

    # =====================================================
    # CORS Configuration
    # =====================================================
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    def get_cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    # =====================================================
    # Project Paths (relative to project root)
    # =====================================================
    PROJECT_ROOT: Path = Path("./")
    CONFIGS_DIR: Path = Path("./configs")
    ARTIFACTS_DIR: Path = Path("./artifacts")

    def get_absolute_configs_dir(self) -> Path:
        """Get absolute path to configs directory."""
        if self.CONFIGS_DIR.is_absolute():
            return self.CONFIGS_DIR
        return self.PROJECT_ROOT / self.CONFIGS_DIR

    def get_absolute_artifacts_dir(self) -> Path:
        """Get absolute path to artifacts directory."""
        if self.ARTIFACTS_DIR.is_absolute():
            return self.ARTIFACTS_DIR
        return self.PROJECT_ROOT / self.ARTIFACTS_DIR

    # =====================================================
    # System Configuration
    # =====================================================
    DEVICE: str = "auto"  # auto, cpu, cuda
    LOG_LEVEL: str = "INFO"

    # =====================================================
    # Inference Settings
    # =====================================================
    FRAME_INTERVAL_MS: int = 500
    MAX_PERSONS_PER_FRAME: int = 10

    # =====================================================
    # Face Embedding Configuration
    # =====================================================
    FACE_EMBEDDING_MODEL: str = "insightface"  # Currently only insightface
    FACE_SIMILARITY_THRESHOLD: float = 0.85
    FACE_EMBEDDING_DIM: int = 512

    # =====================================================
    # Alert Severity Thresholds
    # =====================================================
    # HIGH alert: posture confidence > this AND phone detected
    ALERT_HIGH_POSTURE_CONFIDENCE: float = 0.90
    ALERT_HIGH_PHONE_REQUIRED: bool = True

    # MEDIUM alert: posture confidence > this AND phone detected
    ALERT_MEDIUM_POSTURE_CONFIDENCE: float = 0.75
    ALERT_MEDIUM_PHONE_REQUIRED: bool = True

    # LOW alert: posture state is this
    ALERT_LOW_POSTURE_STATE: str = "suspicious"

    # =====================================================
    # Database Configuration
    # =====================================================
    DETECTION_RETENTION_DAYS: int = 30

    # =====================================================
    # Training Configuration
    # =====================================================
    ALLOW_CONCURRENT_TRAINING: bool = False

    # =====================================================
    # Debug Configuration
    # =====================================================
    DEBUG: bool = False

    class Config:
        """Pydantic config for Settings."""

        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from .env


# Global settings instance
settings = Settings()
