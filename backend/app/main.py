"""
FastAPI Application Entry Point.

Initializes:
- FastAPI app with CORS middleware
- Database connection via lifespan events
- Database initialization (collections + indexes)
- Default settings seeding
- ML model loading via app.state
- Health check endpoint
- Route includes (populated as steps progress)
- Run with Uvicorn
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ✅ Correct config import (renamed to avoid conflict)
# from app.config import settings as config_settings
from backend.app.config import settings as config_settings

# from app.database import Database
from backend.app.database import Database
from backend.app.ml.pipeline import load_pipeline
from backend.app.utils.db_init import DatabaseInitializer
from backend.app.utils.seed_data import SeedData


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("🚀 STARTING PEDESTRIAN DISTRACTION DETECTION BACKEND")
    print("=" * 60)

    # 1. MongoDB
    print("\n[1/5] Connecting to MongoDB...")
    await Database.connect(app=app)

    # 2. DB Init
    print("[2/5] Initializing database collections and indexes...")
    db = Database.get_database()
    await DatabaseInitializer.initialize_database(db)

    # 3. Seed
    print("[3/5] Seeding default settings...")
    await SeedData.seed_all(db)

    # 4. ML Model
    print("[4/5] Loading ML models...")
    try:
        app.state.pipeline = load_pipeline()
        print("✅ ML models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ML models: {e}")
        raise

    # 5. Ready
    print("[5/5] Backend ready for requests")
    print(f"    API URL: http://{config_settings.API_HOST}:{config_settings.API_PORT}")
    print(
        f"    Docs: http://{config_settings.API_HOST}:{config_settings.API_PORT}/docs"
    )
    print("\n" + "=" * 60)

    yield

    # Shutdown
    print("\n" + "=" * 60)
    print("🛑 SHUTTING DOWN BACKEND")
    print("=" * 60)

    print("Disconnecting from MongoDB...")
    await Database.disconnect()

    print("✅ Shutdown complete\n")


# ✅ Create app
app = FastAPI(
    title="Pedestrian Distraction Detection API",
    description="Real-time pedestrian phone usage detection backend",
    version="1.0.0",
    lifespan=lifespan,
)

# ✅ CORS (FIXED — use config_settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config_settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Health API
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "pedestrian-distraction-detection-backend",
        "version": "1.0.0",
        "database_connected": Database.database is not None,
        "model_loaded": hasattr(app.state, "pipeline"),
        "device": config_settings.DEVICE,
    }


# ✅ IMPORTANT: rename settings router
from backend.app.routes import (
    alerts,
    dashboard,
    detect,
    faces,
    settings as settings_router,
    test,
    training,
)

app.include_router(test.router)
app.include_router(detect.router)
app.include_router(faces.router)
app.include_router(alerts.router)
app.include_router(training.router)
app.include_router(dashboard.router)
app.include_router(settings_router.router)


# ✅ Run server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config_settings.API_HOST,
        port=config_settings.API_PORT,
        reload=config_settings.RELOAD,
    )
