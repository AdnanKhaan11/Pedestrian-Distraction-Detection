"""
Database seed data initialization.

Creates default settings and initial data.
"""

import logging
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class SeedData:
    """Handles seeding of default data."""

    @staticmethod
    async def seed_default_settings(db: AsyncIOMotorDatabase) -> None:
        """
        Seed default settings document if it doesn't exist.

        Settings are stored as a singleton (one document).
        """
        settings_collection = db["settings"]

        # Check if settings already exist
        existing_settings = await settings_collection.find_one({})

        if existing_settings:
            logger.info("✅ Default settings already exist")
            return

        # Default settings document
        default_settings = {
            "_id": "system_settings",
            "detection_confidence_threshold": 0.75,
            "alert_confidence_threshold": 0.80,
            "face_similarity_threshold": 0.85,
            "auto_refresh_interval_ms": 5000,
            "notifications_enabled": True,
            "usage_threshold_percent": 80,
            "active_model": "posture_classifier_v1",
            "camera_resolution": "640x480",
            "max_concurrent_cameras": 4,
            "frame_interval_ms": 500,
            # Alert severity thresholds
            "alert_high_posture_confidence": 0.90,
            "alert_high_phone_required": True,
            "alert_medium_posture_confidence": 0.75,
            "alert_medium_phone_required": True,
            "alert_low_posture_state": "suspicious",
            # Face embedding
            "face_embedding_model": "insightface",
            "face_embedding_dim": 512,
            # System configuration
            "system_enabled": True,
            "debug_mode": False,
            # Timestamps
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        try:
            result = await settings_collection.insert_one(default_settings)
            logger.info(f"✅ Default settings created: {result.inserted_id}")
        except Exception as e:
            logger.error(f"Failed to seed default settings: {e}")
            raise

    @staticmethod
    async def seed_default_device(db: AsyncIOMotorDatabase) -> None:
        """
        Seed a default camera device if none exist.

        This is optional for testing purposes.
        """
        devices_collection = db["devices"]

        # Check if devices exist
        existing_device = await devices_collection.find_one({})

        if existing_device:
            logger.info("✅ Devices already exist")
            return

        # Create a test device
        default_device = {
            "device_id": "default_camera_001",
            "device_name": "Main Entrance Camera",
            "device_type": "webcam",
            "location": "Main Entrance",
            "is_active": True,
            "ip_address": "0.0.0.0",
            "port": 8000,
            "status": "online",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "total_detections": 0,
            "total_violations": 0,
        }

        try:
            result = await devices_collection.insert_one(default_device)
            logger.info(f"✅ Default device created: {result.inserted_id}")
        except Exception as e:
            logger.error(f"Failed to seed default device: {e}")
            # Don't raise, this is optional

    @staticmethod
    async def seed_all(db: AsyncIOMotorDatabase) -> None:
        """
        Seed all default data.
        """
        logger.info("🌱 Seeding default data...")

        await SeedData.seed_default_settings(db)
        await SeedData.seed_default_device(db)

        logger.info("✅ Seeding complete")
