"""
This file defines clean request and response schemas for serving.
These schemas are useful for FastAPI validation and for keeping API data structured.
They make backend responses easier to understand and easier to test later.
This is a small but important part of a production-ready MLOps project.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    message: str = Field(..., description="API health message")


class PersonPrediction(BaseModel):
    posture: str = Field(..., description="Final posture/distraction label")
    phone: bool = Field(..., description="Whether a phone was detected")
    state: int = Field(..., description="Internal runtime state code")
    display_text: str = Field(..., description="Short display label")
    score_text: str = Field(..., description="Confidence text used in UI")


class InferenceResponse(BaseModel):
    num_persons: int = Field(..., description="Number of persons processed")
    person_results: List[PersonPrediction] = Field(
        ..., description="List of person-level predictions"
    )
    saved_result_path: Optional[str] = Field(
        default=None, description="Saved rendered output image path"
    )
