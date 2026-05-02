"""
Device API routes.
GET /api/devices — list all registered cameras/devices
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter

from app.database import Database

router = APIRouter(prefix="/api/devices", tags=["Devices"])


@router.get("")
async def list_devices() -> dict[str, Any]:
    """Get all registered devices."""
    try:
        db = Database.get_database()
        devices_cursor = db.devices.find({}).sort("last_seen", -1)
        devices = await devices_cursor.to_list(length=100)

        device_list = []
        for device in devices:
            device["device_id"] = str(device.pop("_id", ""))
            if isinstance(device.get("last_seen"), datetime):
                device["last_seen"] = device["last_seen"].isoformat()
            device_list.append(device)

        return {"total": len(device_list), "devices": device_list}

    except Exception as e:
        return {
            "error": f"Failed to fetch devices: {str(e)}",
            "total": 0,
            "devices": [],
        }
