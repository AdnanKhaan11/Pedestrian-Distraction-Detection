"""
Settings API routes.

GET /api/settings — returns the single settings document
PUT /api/settings — validates and updates the settings document (upsert)
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from app.database import Database
from app.models.settings import Settings

router = APIRouter(prefix="/api/settings", tags=["Settings"])

SETTINGS_ID = "default"  # Singleton document ID


@router.get("")
async def get_settings() -> Settings:
    """
    Get system settings (singleton document).

    Returns:
    ```
    {
      "detection_confidence_threshold": 0.75,
      "alert_confidence_threshold": 0.80,
      "face_similarity_threshold": 0.85,
      "alert_dedup_window_seconds": 60,
      "auto_refresh_interval_ms": 5000,
      "notifications_enabled": true,
      "usage_threshold_percent": 80,
      "active_model": "posture_classifier_v1",
      "camera_resolution": "640x480",
      "max_concurrent_cameras": 4,
      "frame_sample_rate": 2,
      "updated_at": "2024-01-01T12:00:00Z"
    }
    ```
    """
    db = Database.get_database()

    settings_doc = await db.settings.find_one({"_id": SETTINGS_ID})

    if not settings_doc:
        # Return defaults (shouldn't happen if seed_data ran, but handle gracefully)
        return Settings()

    # Convert MongoDB document to Pydantic model
    # Remove _id field before passing to model
    settings_doc.pop("_id", None)
    return Settings(**settings_doc)


@router.put("")
async def update_settings(settings: Settings) -> dict[str, Any]:
    """
    Update system settings (upsert pattern - always updates the same single document).

    Request body:
    ```
    {
      "detection_confidence_threshold": 0.75,
      "alert_confidence_threshold": 0.80,
      "face_similarity_threshold": 0.85,
      "alert_dedup_window_seconds": 60,
      "auto_refresh_interval_ms": 5000,
      "notifications_enabled": true,
      "usage_threshold_percent": 80,
      "active_model": "posture_classifier_v1",
      "camera_resolution": "640x480",
      "max_concurrent_cameras": 4,
      "frame_sample_rate": 2
    }
    ```

    Returns:
    ```
    {
      "success": true,
      "message": "Settings updated",
      "updated_at": "2024-01-01T12:00:01Z"
    }
    ```
    """
    db = Database.get_database()

    # Always update the same document (upsert pattern)
    settings_doc = {
        **settings.dict(),
        "updated_at": datetime.utcnow(),
    }

    result = await db.settings.update_one(
        {"_id": SETTINGS_ID},
        {"$set": settings_doc},
        upsert=True,  # Create if doesn't exist, update if exists
    )

    return {
        "success": True,
        "message": "Settings updated successfully",
        "updated_at": settings_doc["updated_at"].isoformat(),
    }
