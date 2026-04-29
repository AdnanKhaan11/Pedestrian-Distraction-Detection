"""
Database verification test endpoints.

These endpoints test MongoDB connection and basic operations.
Should only be used during development/debugging.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import settings
from app.database import get_db
from app.utils.db_init import DatabaseInitializer

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["testing"])


@router.get("/test-db")
async def test_database_connection(
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> Dict[str, Any]:
    """
    Test MongoDB connection by performing insert and read operations.

    Returns:
        {
            "status": "success",
            "inserted": true,
            "data": {document}
        }

    Raises:
        Exception: If database operations fail
    """
    try:
        logger.info("Starting database verification test...")

        # Verify database is connected
        if db is None:
            logger.error("Database not connected")
            return {
                "status": "error",
                "message": "Database not connected",
                "data": None,
            }

        # =====================================================
        # INSERT: Create test document
        # =====================================================
        test_document = {
            "device_id": "test_device",
            "timestamp": datetime.utcnow(),
            "is_violation": False,
            "severity": "low",
            "test_run": True,
            "created_at": datetime.utcnow().isoformat(),
        }

        logger.info("Inserting test document into 'detections' collection...")

        detections_collection = db["detections"]
        insert_result = await detections_collection.insert_one(test_document)

        if not insert_result.inserted_id:
            logger.error("Failed to insert test document")
            return {
                "status": "error",
                "message": "Failed to insert test document",
                "inserted": False,
                "data": None,
            }

        logger.info(f"✅ Test document inserted with ID: {insert_result.inserted_id}")

        # =====================================================
        # FETCH: Retrieve the latest document
        # =====================================================
        logger.info("Fetching latest document from 'detections' collection...")

        latest_document = await detections_collection.find_one(
            {"_id": insert_result.inserted_id}
        )

        if not latest_document:
            logger.error("Failed to fetch inserted document")
            return {
                "status": "error",
                "message": "Failed to fetch inserted document",
                "inserted": True,
                "data": None,
            }

        logger.info(f"✅ Successfully fetched document: {latest_document}")

        # Convert ObjectId to string for JSON serialization
        latest_document["_id"] = str(latest_document["_id"])

        # =====================================================
        # CLEANUP: Delete test document (optional)
        # =====================================================
        # Uncomment if you want to auto-clean test data
        # await detections_collection.delete_one({"_id": insert_result.inserted_id})
        # logger.info("Test document cleaned up")

        return {
            "status": "success",
            "message": "Database verification successful",
            "inserted": True,
            "data": latest_document,
            "database": settings.DB_NAME,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Database verification failed: {str(e)}",
            "inserted": False,
            "data": None,
        }


@router.get("/health")
async def health_check_with_db(
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> Dict[str, Any]:
    """
    Enhanced health check including database status.
    """
    try:
        # Try to ping the database
        await db.command("ping")
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"

    return {
        "status": "healthy",
        "service": "pedestrian-distraction-detection-backend",
        "version": "1.0.0",
        "database_status": db_status,
        "database_name": settings.DB_NAME,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/db-status")
async def database_status_report(
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get detailed database status including all collections and their document counts.

    Returns collection names and document counts for monitoring.
    """
    try:
        health = await DatabaseInitializer.check_database_health(db)

        return {
            "status": "success",
            "database": settings.DB_NAME,
            "health": health,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Database status check failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
