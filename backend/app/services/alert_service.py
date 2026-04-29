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


class AlertService:
    """Service for alert creation and management."""

    # Severity thresholds
    HIGH_POSTURE_THRESHOLD = 0.90
    HIGH_PHONE_THRESHOLD = 0.90
    MEDIUM_POSTURE_THRESHOLD = 0.75

    # Dedup window in seconds
    DEDUP_WINDOW_SECONDS = 60

    @staticmethod
    def _determine_severity(pedestrian: PedestrianResult) -> str:
        """
        Determine alert severity based on confidence scores.

        Args:
            pedestrian: PedestrianResult with confidence metrics

        Returns:
            "HIGH", "MEDIUM", or "LOW"
        """
        posture_conf = pedestrian.posture_confidence
        phone_detected = pedestrian.phone_detected
        phone_conf = pedestrian.phone_confidence
        fusion_state = pedestrian.fusion_state

        # HIGH: High confidence posture + phone detected + high phone confidence
        if (
            posture_conf > AlertService.HIGH_POSTURE_THRESHOLD
            and phone_detected
            and phone_conf > AlertService.HIGH_PHONE_THRESHOLD
        ):
            return "HIGH"

        # MEDIUM: Moderate posture confidence + phone detected
        if posture_conf > AlertService.MEDIUM_POSTURE_THRESHOLD and phone_detected:
            return "MEDIUM"

        # LOW: Suspicious fusion state
        if fusion_state == "SUSPICIOUS":
            return "LOW"

        # Default (shouldn't reach here if is_violation=True)
        return "LOW"

    @staticmethod
    async def _check_recent_alert_exists(
        face_id: str, db: AsyncIOMotorDatabase
    ) -> bool:
        """
        Check if an unresolved alert already exists for this face within dedup window.

        Args:
            face_id: Face ID to check
            db: AsyncIOMotorDatabase

        Returns:
            True if recent unresolved alert exists, False otherwise
        """
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
            print(f"⚠️  Failed to check for recent alert: {e}")
            return False

    @staticmethod
    async def create_alert(
        detection_result: DetectionResult,
        pedestrian: PedestrianResult,
        face_id: Optional[str] = None,
        snapshot_base64: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create an alert for a violation.

        Args:
            detection_result: Full DetectionResult
            pedestrian: PedestrianResult with violation
            face_id: Optional face ID (from face service)
            snapshot_base64: Optional frame snapshot

        Returns:
            alert_id if created, None if skipped (dedup) or error
        """
        db: AsyncIOMotorDatabase = Database.get_database()

        # Check dedup window only if we have a face_id
        if face_id:
            if await AlertService._check_recent_alert_exists(face_id, db):
                print(
                    f"⚠️  Alert dedup: Recent unresolved alert exists for face {face_id}"
                )
                return None

        # Determine severity
        severity = AlertService._determine_severity(pedestrian)

        # Create alert object
        alert = Alert(
            detection_id=detection_result.detection_id,
            face_id=face_id,
            severity=severity,
            title="Cell Phone Usage Detected",
            description="Pedestrian confirmed using cell phone while crossing road",
            confidence=max(pedestrian.phone_confidence, pedestrian.posture_confidence),
            timestamp=detection_result.timestamp,
            snapshot_base64=snapshot_base64,
        )

        # Save to MongoDB
        try:
            result = await db.alerts.insert_one(
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
            return alert.alert_id

        except Exception as e:
            print(f"⚠️  Failed to save alert: {e}")
            return None

    @staticmethod
    async def resolve_alert(alert_id: str) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if resolved, False if not found or error
        """
        try:
            db: AsyncIOMotorDatabase = Database.get_database()

            result = await db.alerts.update_one(
                {"_id": alert_id},
                {
                    "$set": {
                        "resolved": True,
                        "resolved_at": datetime.utcnow(),
                    }
                },
            )

            return result.modified_count > 0

        except Exception as e:
            print(f"⚠️  Failed to resolve alert: {e}")
            return False

    @staticmethod
    async def delete_alert(alert_id: str) -> bool:
        """
        Delete an alert.

        Args:
            alert_id: Alert ID to delete

        Returns:
            True if deleted, False if not found or error
        """
        try:
            db: AsyncIOMotorDatabase = Database.get_database()

            result = await db.alerts.delete_one({"_id": alert_id})

            return result.deleted_count > 0

        except Exception as e:
            print(f"⚠️  Failed to delete alert: {e}")
            return False
