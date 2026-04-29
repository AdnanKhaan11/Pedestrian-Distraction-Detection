"""
ML Model Bridge for Backend.

This file is IMPORT BRIDGE ONLY.
- Does not contain ML logic
- Imports existing Predictor from src/
- Exposes load_pipeline() to be called at startup
"""

import sys
from pathlib import Path

# Add parent directory to path so src/ can be imported
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.serving.predictor import Predictor


def load_pipeline() -> Predictor:
    """
    Load the ML inference pipeline.

    This function is called ONCE at application startup.
    The result is stored in app.state.pipeline and reused for all requests.

    Returns:
        Predictor: Ready-to-use inference pipeline (wraps InferencePipeline)

    Raises:
        Exception: If any model files are missing or invalid
    """
    pipeline = Predictor(log_level="INFO")
    return pipeline
