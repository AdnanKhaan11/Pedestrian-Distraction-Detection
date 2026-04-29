"""
Pydantic models for alert results.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class Alert(BaseModel):
    """Alert document structure."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique alert ID"
    )
    detection_id: str = Field(..., description="Related detection ID")
    face_id: str | None = Field(
        None, description="Associated face ID (if face processed)"
    )

    severity: str = Field(..., description="HIGH, MEDIUM, or LOW")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")

    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When alert was created"
    )

    resolved: bool = Field(False, description="Whether alert has been resolved")
    resolved_at: datetime | None = Field(None, description="When alert was resolved")

    snapshot_base64: str | None = Field(None, description="Frame snapshot as Base64")

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}
