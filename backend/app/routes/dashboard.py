"""
Dashboard API routes.

GET /api/dashboard/stats — system statistics and alert counts
GET /api/dashboard/timeline — hourly detection trends for last 24 hours
"""

from typing import Any

from fastapi import APIRouter

from app.services.dashboard_service import DashboardService

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/stats")
async def get_dashboard_stats() -> dict[str, Any]:
    """
    Get dashboard statistics (cached for 5 seconds).

    Returns:
    ```
    {
      "detections_today": 47,
      "violations_today": 8,
      "unresolved_alerts": 3,
      "unique_violators_today": 6,
      "active_devices": 1,
      "system_status": "online",
      "model_status": "loaded",
      "last_updated": "2024-01-01T12:00:00Z"
    }
    ```
    """
    return await DashboardService.get_stats()


@router.get("/timeline")
async def get_dashboard_timeline() -> dict[str, Any]:
    """
    Get hourly detection and violation trends for last 24 hours (cached for 5 seconds).

    Returns:
    ```
    {
      "labels": ["00:00", "01:00", ..., "23:00"],
      "detections": [0, 0, 2, 5, 12, 3, ...],
      "violations": [0, 0, 0, 1, 3, 0, ...]
    }
    ```
    """
    return await DashboardService.get_timeline()
