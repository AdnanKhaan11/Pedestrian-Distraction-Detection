"""
Alert service for creating and managing violation alerts.

Handles:
- Severity determination based on confidences
- 60-second deduplication per face_id
- MongoDB alert storage
"""

from datetime import datetime, timedelta
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import Database
from app.models.alert import Alert
from app.models.detection import DetectionResult, PedestrianResult
from app.services.dashboard_service import DashboardService


class AlertService:
    """Service for alert creation and management."""

    HIGH_POSTURE_THRESHOLD = 0.90
    HIGH_PHONE_THRESHOLD = 0.90
    MEDIUM_POSTURE_THRESHOLD = 0.75
    DEDUP_WINDOW_SECONDS = 60
    ALERT_RETENTION_HOURS = 24

    @staticmethod
    def _determine_severity(pedestrian: PedestrianResult) -> str:
        """Determine alert severity based on confidence scores."""
        posture_conf = pedestrian.posture_confidence
        phone_detected = pedestrian.phone_detected
        phone_conf = pedestrian.phone_confidence
        fusion_state = pedestrian.fusion_state

        if (
            posture_conf > AlertService.HIGH_POSTURE_THRESHOLD
            and phone_detected
            and phone_conf > AlertService.HIGH_PHONE_THRESHOLD
        ):
            return "HIGH"

        if posture_conf > AlertService.MEDIUM_POSTURE_THRESHOLD and phone_detected:
            return "MEDIUM"

        if fusion_state == "SUSPICIOUS":
            return "LOW"

        return "LOW"

    @staticmethod
    async def _check_recent_alert_exists(
        face_id: str, db: AsyncIOMotorDatabase
    ) -> bool:
        """Check if an unresolved alert exists for this face within the dedup window."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(
                seconds=AlertService.DEDUP_WINDOW_SECONDS
            )
            existing = await db.alerts.find_one(
                {
                    "face_id": face_id,
                    "resolved": False,
                    "timestamp": {"$gte": cutoff_time},
                }
            )
            return existing is not None
        except Exception as e:
            print(f"Warning: failed to check for recent alert: {e}")
            return False

    @staticmethod
    async def create_alert(
        detection_result: DetectionResult,
        pedestrian: PedestrianResult,
        face_id: Optional[str] = None,
        snapshot_base64: Optional[str] = None,
    ) -> Optional[str]:
        """Create an alert for a violation."""
        db: AsyncIOMotorDatabase = Database.get_database()

        if face_id and await AlertService._check_recent_alert_exists(face_id, db):
            print(f"Warning: alert dedup skipped recent unresolved alert for {face_id}")
            return None

        alert = Alert(
            detection_id=detection_result.detection_id,
            face_id=face_id,
            severity=AlertService._determine_severity(pedestrian),
            title="Cell Phone Usage Detected",
            description="Pedestrian confirmed using cell phone while crossing road",
            confidence=max(pedestrian.phone_confidence, pedestrian.posture_confidence),
            timestamp=detection_result.timestamp,
            snapshot_base64=snapshot_base64,
        )

        try:
            await db.alerts.insert_one(
                {
                    "_id": alert.alert_id,
                    "detection_id": alert.detection_id,
                    "face_id": alert.face_id,
                    "severity": alert.severity,
                    "title": alert.title,
                    "description": alert.description,
                    "confidence": alert.confidence,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at,
                    "snapshot_base64": alert.snapshot_base64,
                }
            )
            DashboardService.clear_cache()
            return alert.alert_id
        except Exception as e:
            print(f"Warning: failed to save alert: {e}")
            return None

    @staticmethod
    async def cleanup_expired_alerts() -> int:
        """Delete alerts older than the retention window."""
        try:
            db: AsyncIOMotorDatabase = Database.get_database()
            cutoff_time = datetime.utcnow() - timedelta(
                hours=AlertService.ALERT_RETENTION_HOURS
            )
            result = await db.alerts.delete_many({"timestamp": {"$lt": cutoff_time}})
            if result.deleted_count > 0:
                DashboardService.clear_cache()
            return result.deleted_count
        except Exception as e:
            print(f"Warning: failed to cleanup expired alerts: {e}")
            return 0

    @staticmethod
    async def resolve_alert(alert_id: str) -> bool:
        """Mark an alert as resolved."""
        try:
            db: AsyncIOMotorDatabase = Database.get_database()
            result = await db.alerts.update_one(
                {"_id": alert_id},
                {"$set": {"resolved": True, "resolved_at": datetime.utcnow()}},
            )
            if result.modified_count > 0:
                DashboardService.clear_cache()
            return result.modified_count > 0
        except Exception as e:
            print(f"Warning: failed to resolve alert: {e}")
            return False

    @staticmethod
    async def delete_alert(alert_id: str) -> bool:
        """Delete an alert."""
        try:
            db: AsyncIOMotorDatabase = Database.get_database()
            result = await db.alerts.delete_one({"_id": alert_id})
            if result.deleted_count > 0:
                DashboardService.clear_cache()
            return result.deleted_count > 0
        except Exception as e:
            print(f"Warning: failed to delete alert: {e}")
            return False
