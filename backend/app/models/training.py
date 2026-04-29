"""
Pydantic models for training requests and responses.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """Training job request."""

    model_type: str = Field(
        ..., description="Model to train: 'posture_classifier' or 'phone_detector'"
    )
    epochs: int = Field(50, ge=1, le=500, description="Number of training epochs")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="Learning rate")
    batch_size: int = Field(32, ge=4, le=256, description="Batch size")


class TrainingStatus(BaseModel):
    """Current training job status."""

    job_id: str = Field(..., description="Unique training job ID")
    status: str = Field(
        ..., description="Job status: running, completed, error, cancelled"
    )
    model_type: str = Field(..., description="Model being/trained")
    started_at: datetime = Field(..., description="When job started")
    current_epoch: int = Field(0, ge=0, description="Current epoch (0 if not started)")
    total_epochs: int = Field(..., ge=1, description="Total epochs to run")
    progress_percent: float = Field(
        0.0, ge=0.0, le=100.0, description="Completion percentage"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TrainingLog(BaseModel):
    """Single log message for WebSocket streaming."""

    type: str = Field(
        ..., description="Message type: 'progress', 'log', 'complete', 'error'"
    )
    epoch: Optional[int] = Field(None, description="Current epoch (for progress)")
    total_epochs: Optional[int] = Field(None, description="Total epochs (for progress)")
    loss: Optional[float] = Field(None, description="Loss value (for progress)")
    accuracy: Optional[float] = Field(None, description="Accuracy value (for progress)")
    message: Optional[str] = Field(None, description="Log message text")
    metrics: Optional[dict[str, Any]] = Field(
        None, description="Final metrics on completion"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class TrainingJob(BaseModel):
    """Training job database document."""

    job_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique job ID"
    )
    model_type: str = Field(..., description="'posture_classifier' or 'phone_detector'")
    status: str = Field(
        "queued", description="queued, running, completed, error, cancelled"
    )

    epochs: int = Field(..., description="Configured epochs")
    learning_rate: float = Field(..., description="Configured learning rate")
    batch_size: int = Field(..., description="Configured batch size")

    current_epoch: int = Field(0, description="Current progress epoch")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

    logs: list[str] = Field(default_factory=list, description="All log lines")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Final metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
