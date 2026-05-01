"""
Alert API routes.

GET /api/alerts — list with filtering and pagination
PATCH /api/alerts/{alert_id}/resolve — mark as resolved
DELETE /api/alerts/{alert_id} — remove alert
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Query

from app.database import Database
from app.services.alert_service import AlertService

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("")
async def list_alerts(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(10, ge=1, le=100, description="Results per page"),
    severity: Optional[str] = Query(
        None, description="Filter by severity: HIGH, MEDIUM, LOW"
    ),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    from_date: Optional[str] = Query(None, description="ISO8601 start date"),
    to_date: Optional[str] = Query(None, description="ISO8601 end date"),
) -> dict[str, Any]:
    """
    Get paginated list of alerts with optional filtering.

    Query params:
    - page: Page number (default: 1)
    - limit: Results per page (default: 10, max: 100)
    - severity: "HIGH", "MEDIUM", or "LOW" (optional)
    - resolved: true/false (optional)
    - from_date: ISO8601 date string (optional)
    - to_date: ISO8601 date string (optional)

    Returns:
        {
            "total": total count,
            "page": current page,
            "limit": results per page,
            "alerts": [
                {
                    "alert_id": "...",
                    "detection_id": "...",
                    "face_id": "...",
                    "severity": "HIGH",
                    "title": "...",
                    "description": "...",
                    "confidence": 0.92,
                    "timestamp": "ISO8601",
                    "resolved": false,
                    "resolved_at": null
                },
                ...
            ]
        }
    """
    try:
        db = Database.get_database()

        # Build filter query
        query_filter = {}

        if severity:
            query_filter["severity"] = severity.upper()

        if resolved is not None:
            query_filter["resolved"] = resolved

        if from_date or to_date:
            date_range = {}
            if from_date:
                try:
                    date_range["$gte"] = datetime.fromisoformat(
                        from_date.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            if to_date:
                try:
                    date_range["$lte"] = datetime.fromisoformat(
                        to_date.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            if date_range:
                query_filter["timestamp"] = date_range

        # Get total count
        total = await db.alerts.count_documents(query_filter)

        # Calculate skip
        skip = (page - 1) * limit

        # Get alerts
        alerts_cursor = (
            db.alerts.find(query_filter, {"snapshot_base64": 0})  # Exclude large field
            .sort("timestamp", -1)
            .skip(skip)
            .limit(limit)
        )

        alerts = await alerts_cursor.to_list(length=limit)

        # Convert _id to alert_id
        alert_list = []
        for alert in alerts:
            alert["alert_id"] = str(alert.pop("_id"))
            alert_list.append(alert)

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "alerts": alert_list,
        }

    except Exception as e:
        return {
            "error": f"Failed to fetch alerts: {str(e)}",
            "total": 0,
            "page": page,
            "limit": limit,
            "alerts": [],
        }


@router.patch("/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> dict[str, Any]:
    """
    Mark an alert as resolved.

    Args:
        alert_id: Alert ID to resolve

    Returns:
        { "success": true/false, "message": "..." }
    """
    success = await AlertService.resolve_alert(alert_id)

    if success:
        return {"success": True, "message": f"Alert {alert_id} resolved"}
    else:
        return {
            "success": False,
            "message": f"Alert {alert_id} not found or update failed",
        }


@router.delete("/{alert_id}")
async def delete_alert(alert_id: str) -> dict[str, Any]:
    """
    Delete an alert.

    Args:
        alert_id: Alert ID to delete

    Returns:
        { "success": true/false, "message": "..." }
    """
    success = await AlertService.delete_alert(alert_id)

    if success:
        return {"success": True, "message": f"Alert {alert_id} deleted"}
    else:
        return {"success": False, "message": f"Alert {alert_id} not found"}
