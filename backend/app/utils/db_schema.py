"""
MongoDB collections schema definitions and initialization.

Defines the structure and indexes for all MongoDB collections.
"""

from typing import Dict, Any


class CollectionSchema:
    """Schema definitions for all MongoDB collections."""

    @staticmethod
    def get_detections_schema() -> Dict[str, Any]:
        """
        Detections collection schema.

        Stores all detection results (violations and non-violations).
        """
        return {
            "name": "detections",
            "indexes": [
                {"keys": [("timestamp", -1)]},  # For time-range queries
                {"keys": [("is_violation", 1)]},  # For filtering violations
                {"keys": [("session_id", 1)]},  # For session grouping
                {"keys": [("timestamp", -1), ("is_violation", 1)]},  # Compound
            ],
            "description": "Stores all detection results from camera feeds",
        }

    @staticmethod
    def get_alerts_schema() -> Dict[str, Any]:
        """
        Alerts collection schema.

        Stores alerts triggered by violations.
        """
        return {
            "name": "alerts",
            "indexes": [
                {"keys": [("timestamp", -1)]},  # Recent alerts
                {
                    "keys": [("timestamp", 1)],
                    "expireAfterSeconds": 86400,
                },  # Auto-delete after 24 hours
                {"keys": [("resolved", 1)]},  # Unresolved alerts
                {"keys": [("severity", 1)]},  # Filter by severity
                {"keys": [("face_id", 1)]},  # Alerts for a person
                {"keys": [("timestamp", -1), ("resolved", 1)]},  # Recent unresolved
            ],
            "description": "Stores alerts for violations detected",
        }

    @staticmethod
    def get_faces_schema() -> Dict[str, Any]:
        """
        Faces collection schema.

        Stores unique face embeddings and crops.
        """
        return {
            "name": "faces",
            "indexes": [
                {"keys": [("last_seen", -1)]},  # Most recent faces
                {"keys": [("violation_count", -1)]},  # Most violating people
                {"keys": [("first_seen", -1)]},  # Faces by creation time
            ],
            "description": "Stores unique face embeddings and cropped images",
        }

    @staticmethod
    def get_settings_schema() -> Dict[str, Any]:
        """
        Settings collection schema.

        Singleton collection for system-wide settings.
        """
        return {
            "name": "settings",
            "indexes": [
                {"keys": [("_id", 1)], "unique": True},  # Singleton pattern
            ],
            "description": "System-wide settings (singleton)",
        }

    @staticmethod
    def get_devices_schema() -> Dict[str, Any]:
        """
        Devices collection schema.

        Stores registered camera/detection devices.
        """
        return {
            "name": "devices",
            "indexes": [
                {"keys": [("device_id", 1)], "unique": True},
                {"keys": [("is_active", 1)]},
            ],
            "description": "Registered detection devices/cameras",
        }

    @staticmethod
    def get_training_logs_schema() -> Dict[str, Any]:
        """
        Training logs collection schema.

        Stores training job history and logs.
        """
        return {
            "name": "training_logs",
            "indexes": [
                {"keys": [("job_id", 1)], "unique": True},
                {"keys": [("started_at", -1)]},
                {"keys": [("status", 1)]},
            ],
            "description": "Training job logs and history",
        }

    @staticmethod
    def get_all_schemas() -> list:
        """Get all collection schemas."""
        return [
            CollectionSchema.get_detections_schema(),
            CollectionSchema.get_alerts_schema(),
            CollectionSchema.get_faces_schema(),
            CollectionSchema.get_settings_schema(),
            CollectionSchema.get_devices_schema(),
            CollectionSchema.get_training_logs_schema(),
        ]
