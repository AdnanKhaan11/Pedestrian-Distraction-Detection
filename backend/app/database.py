"""
MongoDB database connection using Motor (async driver).

Provides:
- Async MongoDB client initialization
- Database connection management
- Dependency injection for routes
"""

from typing import Optional

# from motor.motor_asyncio import AsyncClient, AsyncDatabase
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings


from pymongo import MongoClient
import ssl

import os

print("MONGO URI BEING USED:", os.getenv("MONGODB_URI"))


class Database:
    """
    MongoDB connection manager using Motor (async).

    Usage:
        db = Database()
        await db.connect()
        # Use db.client, db.database
        await db.disconnect()
    """

    client: Optional[AsyncIOMotorClient] = None
    database: Optional[AsyncIOMotorDatabase] = None
    app: Optional[object] = None  # Store FastAPI app for accessing app.state

    @classmethod
    async def connect(cls, app=None) -> None:
        """
        Establish async MongoDB connection.

        Called during FastAPI lifespan startup.

        Args:
            app: FastAPI app instance (for storing in app.state)
        """
        if app:
            cls.app = app

        try:
            cls.client = AsyncIOMotorClient(settings.MONGODB_URI)
            cls.database = cls.client[settings.DB_NAME]

            # Verify connection with a ping
            await cls.database.command("ping")
            print(f"Connected to MongoDB: {settings.DB_NAME}")

        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            raise

    @classmethod
    async def disconnect(cls) -> None:
        """
        Close MongoDB connection.

        Called during FastAPI lifespan shutdown.
        """
        if cls.client:
            cls.client.close()
            print("Disconnected from MongoDB")

    @classmethod
    def get_database(cls) -> AsyncIOMotorDatabase:
        """
        Get the current database instance.

        Use as a FastAPI dependency:
            async def some_route(db: AsyncIOMotorDatabase = Depends(Database.get_database)):
                ...
        """
        if cls.database is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return cls.database


async def get_db() -> AsyncIOMotorDatabase:
    """
    FastAPI dependency for injecting database into routes.

    Usage:
        @app.get("/example")
        async def example_route(db: AsyncIOMotorDatabase = Depends(get_db)):
            collection = db["collection_name"]
            ...
    """
    return Database.get_database()
