"""
Dashboard service for aggregating detection, violation, and alert statistics.

Handles:
- Real-time aggregation queries using MongoDB pipelines
- Today's detection and violation counts
- Hourly trends for last 24 hours
- Unique violators tracking
- Alert status summaries
- Result caching for 5 seconds
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import Database


class DashboardCache:
    """Simple in-memory cache for dashboard stats."""

    _stats_cache: Optional[dict[str, Any]] = None
    _timeline_cache: Optional[dict[str, Any]] = None
    _stats_cached_at: Optional[datetime] = None
    _timeline_cached_at: Optional[datetime] = None
    _cache_ttl_seconds = 5

    @classmethod
    def is_stats_expired(cls) -> bool:
        """Check if stats cache is expired."""
        if cls._stats_cached_at is None:
            return True
        elapsed = (datetime.utcnow() - cls._stats_cached_at).total_seconds()
        return elapsed > cls._cache_ttl_seconds

    @classmethod
    def is_timeline_expired(cls) -> bool:
        """Check if timeline cache is expired."""
        if cls._timeline_cached_at is None:
            return True
        elapsed = (datetime.utcnow() - cls._timeline_cached_at).total_seconds()
        return elapsed > cls._cache_ttl_seconds

    @classmethod
    def set_stats(cls, stats: dict[str, Any]):
        """Cache stats with timestamp."""
        cls._stats_cache = stats
        cls._stats_cached_at = datetime.utcnow()

    @classmethod
    def get_stats(cls) -> Optional[dict[str, Any]]:
        """Get cached stats if not expired."""
        if cls.is_stats_expired():
            return None
        return cls._stats_cache

    @classmethod
    def set_timeline(cls, timeline: dict[str, Any]):
        """Cache timeline with timestamp."""
        cls._timeline_cache = timeline
        cls._timeline_cached_at = datetime.utcnow()

    @classmethod
    def get_timeline(cls) -> Optional[dict[str, Any]]:
        """Get cached timeline if not expired."""
        if cls.is_timeline_expired():
            return None
        return cls._timeline_cache


class DashboardService:
    """Service for dashboard statistics and aggregations."""

    @staticmethod
    async def get_stats() -> dict[str, Any]:
        """
        Get dashboard statistics with caching.

        Returns aggregated stats:
        - detections_today: total detections since midnight UTC
        - violations_today: total violations since midnight UTC
        - unresolved_alerts: count of unresolved alerts
        - unique_violators_today: distinct face_ids with violations today
        - active_devices: count of devices
        - system_status: "online" (always, for now)
        - model_status: "loaded" (always, for now)
        - last_updated: current timestamp
        """
        # Check cache
        cached = DashboardCache.get_stats()
        if cached:
            return cached

        db: AsyncIOMotorDatabase = Database.get_database()

        # Calculate midnight UTC for "today"
        now = datetime.utcnow()
        today_midnight = datetime(now.year, now.month, now.day, 0, 0, 0)

        # Query 1: Total detections today
        detections_today = await db.detections.count_documents(
            {"timestamp": {"$gte": today_midnight}}
        )

        # Query 2: Total violations today
        violations_today = await db.detections.count_documents(
            {
                "timestamp": {"$gte": today_midnight},
                "is_violation": True,
            }
        )

        # Query 3: Unresolved alerts (all-time)
        unresolved_alerts = await db.alerts.count_documents({"resolved": False})

        # Query 4: Unique violators today (distinct face_ids in violations)
        unique_violators_pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": today_midnight},
                    "is_violation": True,
                    "pedestrians": {"$exists": True},
                }
            },
            {"$unwind": "$pedestrians"},
            {
                "$group": {
                    "_id": "$pedestrians.face_id",
                }
            },
            {"$count": "unique_faces"},
        ]
        unique_violators_result = await db.detections.aggregate(
            unique_violators_pipeline
        ).to_list(None)
        unique_violators = (
            unique_violators_result[0]["unique_faces"] if unique_violators_result else 0
        )

        # Query 5: Active devices (count devices where is_active=True)
        active_devices = await db.devices.count_documents({"is_active": True})

        stats = {
            "detections_today": detections_today,
            "violations_today": violations_today,
            "unresolved_alerts": unresolved_alerts,
            "unique_violators_today": unique_violators,
            "active_devices": active_devices,
            "system_status": "online",
            "model_status": "loaded",
            "last_updated": datetime.utcnow().isoformat(),
        }

        # Cache result
        DashboardCache.set_stats(stats)

        return stats

    @staticmethod
    async def get_timeline() -> dict[str, Any]:
        """
        Get hourly detection and violation counts for last 24 hours.

        Returns:
        {
          "labels": ["00:00", "01:00", ..., "23:00"],
          "detections": [0, 2, 5, ...],  # detections per hour
          "violations": [0, 0, 1, ...]   # violations per hour
        }
        """
        # Check cache
        cached = DashboardCache.get_timeline()
        if cached:
            return cached

        db: AsyncIOMotorDatabase = Database.get_database()

        now = datetime.utcnow()
        today_midnight = datetime(now.year, now.month, now.day, 0, 0, 0)

        # Query: Group detections by hour for last 24 hours
        hourly_pipeline = [
            {"$match": {"timestamp": {"$gte": today_midnight}}},
            {
                "$group": {
                    "_id": {
                        "hour": {"$hour": "$timestamp"},
                        "is_violation": "$is_violation",
                    },
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id.hour": 1}},
        ]

        results = await db.detections.aggregate(hourly_pipeline).to_list(None)

        # Initialize arrays for all 24 hours
        labels = [f"{h:02d}:00" for h in range(24)]
        detections = [0] * 24
        violations = [0] * 24

        # Populate from results
        for result in results:
            hour = result["_id"]["hour"]
            is_violation = result["_id"]["is_violation"]
            count = result["count"]

            if is_violation:
                violations[hour] = count
            else:
                detections[hour] = count

        # Combine detections and violations (total in detections, violations separate)
        for i in range(24):
            detections[i] += violations[i]

        timeline = {
            "labels": labels,
            "detections": detections,
            "violations": violations,
        }

        # Cache result
        DashboardCache.set_timeline(timeline)

        return timeline

    @staticmethod
    async def get_recent_alerts(limit: int = 5) -> list[dict[str, Any]]:
        """Get recent unresolved alerts."""
        db: AsyncIOMotorDatabase = Database.get_database()

        alerts = (
            await db.alerts.find({"resolved": False})
            .sort("timestamp", -1)
            .limit(limit)
            .to_list(None)
        )

        return alerts or []

    @staticmethod
    def clear_cache():
        """Clear all dashboard caches (useful for testing)."""
        DashboardCache._stats_cache = None
        DashboardCache._timeline_cache = None
        DashboardCache._stats_cached_at = None
        DashboardCache._timeline_cached_at = None
