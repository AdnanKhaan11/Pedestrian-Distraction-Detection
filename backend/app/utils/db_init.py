"""
Database initialization utilities.

Creates MongoDB collections and indexes.
"""

import logging
from typing import List

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.utils.db_schema import CollectionSchema

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization, collections, and indexes."""

    @staticmethod
    async def initialize_database(db: AsyncIOMotorDatabase) -> None:
        """
        Initialize all collections and indexes.

        Args:
            db: Motor AsyncIOMotorDatabase instance
        """
        logger.info("🔧 Initializing MongoDB database...")

        # Get existing collections
        existing_collections = await db.list_collection_names()
        logger.info(f"Existing collections: {existing_collections}")

        # Initialize each collection
        schemas = CollectionSchema.get_all_schemas()

        for schema in schemas:
            collection_name = schema["name"]

            # Create collection if it doesn't exist
            if collection_name not in existing_collections:
                logger.info(f"  Creating collection: {collection_name}")
                await db.create_collection(collection_name)
            else:
                logger.info(f"  Collection exists: {collection_name}")

            # Create indexes
            collection = db[collection_name]
            for index in schema.get("indexes", []):
                await DatabaseInitializer._create_index(collection, index)

        logger.info("✅ Database initialization complete")

    @staticmethod
    async def _create_index(collection, index_spec: dict) -> None:
        """
        Create an index on a collection.

        Args:
            collection: Motor collection object
            index_spec: Index specification with 'keys' and optional 'unique'
        """
        keys = index_spec.get("keys", [])
        options = {k: v for k, v in index_spec.items() if k != "keys"}

        try:
            index_name = await collection.create_index(keys, **options)
            logger.debug(f"    Index created: {index_name}")
        except Exception as e:
            # Index may already exist, which is fine
            logger.debug(f"    Index creation note: {e}")

    @staticmethod
    async def check_database_health(db: AsyncIOMotorDatabase) -> dict:
        """
        Check database health and collection status.

        Returns:
            dict with health status
        """
        try:
            # Ping the database
            await db.command("ping")

            # Count documents in each collection
            collections_status = {}
            for schema in CollectionSchema.get_all_schemas():
                collection_name = schema["name"]
                collection = db[collection_name]
                count = await collection.count_documents({})
                collections_status[collection_name] = count

            return {
                "status": "healthy",
                "collections": collections_status,
            }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }
