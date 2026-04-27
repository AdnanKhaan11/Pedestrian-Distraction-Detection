"""
Production-ready FastAPI backend for Pedestrian Phone Detection System.

This module:
- Initializes FastAPI with modular architecture
- Integrates ML predictor (unchanged)
- Provides authentication, detection, face recognition, and reporting APIs
- Serves frontend with proper error handling
- Implements CORS and security middleware
"""

import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config.configuration import ConfigurationManager
from src.serving.predictor import Predictor
from src.serving.schemas import InferenceResponse
from src.utils.common import create_directories
from src.utils.logger import get_logger

# Import modular routes
from app.core.config import settings
from app.core.security import generate_unique_filename
from app.db.database import init_db, get_db
from app.routes import auth, detect, face, reports
from app.schemas.models import HealthResponse

# Initialize FastAPI app
app = FastAPI(
    title="Pedestrian Phone Detection API - v2.0",
    version="2.0.0",
    description="Production-ready backend for pedestrian phone detection with face recognition and reporting.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

config_manager = ConfigurationManager()
paths_config = config_manager.get_paths_config()
inference_config = config_manager.get_inference_config()

create_directories(
    [
        paths_config.logs_dir,
        paths_config.frontend_upload_dir,
        paths_config.frontend_result_dir,
        paths_config.metrics_dir,
        settings.UPLOAD_DIR,
        settings.RESULT_DIR,
        settings.EMBEDDINGS_DIR,
    ]
)

logger = get_logger("api", log_dir=paths_config.logs_dir, level="INFO")
predictor = Predictor(log_dir=paths_config.logs_dir, log_level="INFO")

# Initialize database
init_db()
logger.info("Database initialized")

# ============================================================================
# STATIC FILES & TEMPLATES
# ============================================================================

templates = Jinja2Templates(directory=str(ROOT_DIR / "app" / "templates"))
app.mount(
    "/static", StaticFiles(directory=str(ROOT_DIR / "app" / "static")), name="static"
)

# ============================================================================
# ROUTE INCLUSION
# ============================================================================

app.include_router(auth.router)
app.include_router(detect.router)
app.include_router(face.router)
app.include_router(reports.router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Serve the main frontend page.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_title": inference_config.frontend.app_title,
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic API health endpoint.
    """
    return HealthResponse(status="ok", message="API is running")


@app.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    """
    Run inference on one uploaded image.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file name provided.")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        upload_path = paths_config.frontend_upload_dir / file.filename
        with open(upload_path, "wb") as output_file:
            output_file.write(await file.read())

        result = predictor.predict_image(upload_path, save_rendered_output=True)
        return InferenceResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train/posture")
async def train_posture():
    """
    Trigger posture training script from the UI.
    """
    try:
        script_path = ROOT_DIR / "scripts" / "train_posture_model.py"
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=False,
        )

        return JSONResponse(
            {
                "status": "completed" if process.returncode == 0 else "failed",
                "return_code": process.returncode,
                "stdout": process.stdout[-4000:],
                "stderr": process.stderr[-4000:],
            }
        )
    except Exception as exc:
        logger.exception("Posture training trigger failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/train/phone")
async def train_phone():
    """
    Trigger phone detector training script from the UI.
    """
    try:
        script_path = ROOT_DIR / "scripts" / "train_phone_model.py"
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=False,
        )

        return JSONResponse(
            {
                "status": "completed" if process.returncode == 0 else "failed",
                "return_code": process.returncode,
                "stdout": process.stdout[-4000:],
                "stderr": process.stderr[-4000:],
            }
        )
    except Exception as exc:
        logger.exception("Phone training trigger failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/about")
async def about():
    """
    Provide project information for the frontend.
    """
    return {
        "project_name": "Pedestrian Distraction Detection",
        "backend": "FastAPI",
        "tracking": "MLflow + DAGsHub",
        "pipeline": [
            "MMPose person detection",
            "Pose keypoint extraction",
            "Posture classification",
            "Phone detection on hand crops",
            "Final distraction decision",
        ],
    }
