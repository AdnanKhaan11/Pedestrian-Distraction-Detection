"""
Pydantic models for system settings.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """System-wide settings (singleton in MongoDB)."""

    # Detection thresholds
    detection_confidence_threshold: float = Field(
        0.75, ge=0.0, le=1.0, description="Minimum confidence for any detection"
    )
    alert_confidence_threshold: float = Field(
        0.80, ge=0.0, le=1.0, description="Minimum confidence to trigger alert"
    )

    # Face matching threshold
    face_similarity_threshold: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for face deduplication",
    )

    # Alert deduplication
    alert_dedup_window_seconds: int = Field(
        60, ge=1, le=3600, description="Time window for alert deduplication per face"
    )

    # UI settings
    auto_refresh_interval_ms: int = Field(
        5000,
        ge=1000,
        le=60000,
        description="Dashboard refresh interval in milliseconds",
    )
    notifications_enabled: bool = Field(
        True, description="Whether to show browser notifications"
    )

    # Resource limits
    usage_threshold_percent: int = Field(
        80, ge=0, le=100, description="Alert threshold for resource usage"
    )
    max_concurrent_cameras: int = Field(
        4, ge=1, le=16, description="Maximum concurrent camera streams"
    )

    # Model and camera configuration
    active_model: str = Field(
        "posture_classifier_v1", description="Active posture model name"
    )
    camera_resolution: str = Field(
        "640x480", description="Camera resolution (WIDTHxHEIGHT)"
    )
    frame_sample_rate: int = Field(
        2, ge=1, le=60, description="Frames per second to process"
    )

    # Metadata
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}
