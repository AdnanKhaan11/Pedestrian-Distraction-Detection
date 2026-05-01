"""
Inference service for running ML pipeline and processing detections.

Handles:
- Base64 frame decoding
- ML pipeline invocation
- Output mapping to Pydantic models
- Numpy type conversion
- MongoDB storage
- Face processing for violations
- Alert creation for violations
"""

import asyncio
import base64
import time
from typing import Any

import cv2
import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import Database
from app.models.detection import (
    BoundingBox,
    DetectionResult,
    FaceRegion,
    PedestrianResult,
)
from app.services.alert_service import AlertService
from app.services.dashboard_service import DashboardService
from app.services.face_service import FaceService


class InferenceService:
    """Service for running inference and storing results."""

    SESSION_TTL_SECONDS = 1800
    _session_runtime_params: dict[str, dict] = {}

    @classmethod
    def _get_session_runtime_params(cls, session_id: str) -> dict:
        """Get or create runtime parameters for a session."""
        now = time.time()
        expired_sessions = [
            key
            for key, value in cls._session_runtime_params.items()
            if now - value.get("last_accessed_at", 0.0) > cls.SESSION_TTL_SECONDS
        ]
        for key in expired_sessions:
            cls._session_runtime_params.pop(key, None)

        if session_id not in cls._session_runtime_params:
            cls._session_runtime_params[session_id] = {
                "time_last_record_framerate": 0.0,
                "time_last_announce_face": 0.0,
                "path_runtime_handframes": None,
                "last_accessed_at": now,
            }
        cls._session_runtime_params[session_id]["last_accessed_at"] = now
        return cls._session_runtime_params[session_id]

    @staticmethod
    def _convert_to_python_type(value: Any) -> Any:
        """
        Recursively convert numpy types to Python native types.

        JSON serialization fails on numpy.float32, numpy.int64, etc.
        This ensures all values are Python native types.
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {
                k: InferenceService._convert_to_python_type(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [InferenceService._convert_to_python_type(v) for v in value]
        return value

    @staticmethod
    async def _enrich_violation_pedestrian(
        frame: np.ndarray | None,
        detection: DetectionResult,
        pedestrian: PedestrianResult,
    ) -> None:
        """Resolve face linkage before creating alerts and saving detections."""
        face_id = None

        if frame is not None:
            try:
                face_id = await FaceService.process_face(frame, pedestrian, detection)
            except Exception as e:
                print(f"Warning: face processing error: {e}")

        pedestrian.face_id = face_id

        try:
            await AlertService.create_alert(
                detection_result=detection,
                pedestrian=pedestrian,
                face_id=face_id,
            )
        except Exception as e:
            print(f"Warning: alert creation error: {e}")

    @staticmethod
    def decode_base64_frame(frame_base64: str) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Decode Base64 JPEG frame to numpy BGR array.

        Args:
            frame_base64: Base64-encoded JPEG string

        Returns:
            (frame_array, (height, width))

        Raises:
            ValueError: If decoding fails
        """
        try:
            frame_bytes = base64.b64decode(frame_base64)
            frame_array = cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if frame_array is None:
                raise ValueError("Failed to decode image from Base64 JPEG")
            height, width = frame_array.shape[:2]
            return frame_array, (height, width)
        except Exception as e:
            raise ValueError(f"Base64 decoding failed: {e}")

    @staticmethod
    def _compute_normalized_bbox(xyxy: list, height: int, width: int) -> BoundingBox:
        """Convert pixel coordinates to normalized coordinates."""
        x1, y1, x2, y2 = xyxy
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        x1_norm = max(0.0, min(1.0, x1 / width))
        y1_norm = max(0.0, min(1.0, y1 / height))
        x2_norm = max(0.0, min(1.0, x2 / width))
        y2_norm = max(0.0, min(1.0, y2 / height))

        return BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            x1_norm=x1_norm,
            y1_norm=y1_norm,
            x2_norm=x2_norm,
            y2_norm=y2_norm,
        )

    @staticmethod
    def _build_pedestrian_result(
        person_result: dict, height: int, width: int
    ) -> PedestrianResult:
        """Map raw ML pipeline output to PedestrianResult."""
        xyxy = person_result.get("xyxy", [0, 0, 100, 100])
        bbox = InferenceService._compute_normalized_bbox(xyxy, height, width)

        face_xyxy = person_result.get("face_xyxy")
        face_region = None
        if face_xyxy is not None:
            try:
                x1, y1, x2, y2 = face_xyxy
                face_region = FaceRegion(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
            except (ValueError, TypeError):
                pass

        posture_state = person_result.get("posture", "GOOD")
        phone_detected = person_result.get("phone", False)
        fusion_state = person_result.get("state", "NORMAL")
        is_violation = fusion_state == "USING"

        # Get real confidence from pipeline output, fallback to reasonable defaults
        posture_confidence = person_result.get("posture_confidence", 0.85)
        phone_confidence = person_result.get("phone_confidence", 0.90)

        # Validate confidence values are in [0, 1]
        posture_confidence = max(0.0, min(1.0, float(posture_confidence)))
        phone_confidence = max(0.0, min(1.0, float(phone_confidence)))

        return PedestrianResult(
            bbox=bbox,
            posture_state=posture_state,
            posture_confidence=posture_confidence,
            phone_detected=phone_detected,
            phone_confidence=phone_confidence,
            fusion_state=fusion_state,
            face_region=face_region,
            is_violation=is_violation,
        )

    async def run_inference(
        self,
        frame_base64: str,
        session_id: str,
        frame_id: int = 0,
    ) -> DetectionResult:
        """Run full inference pipeline on a frame and store the result."""
        start_time = time.time()

        if not Database.app or not hasattr(Database.app.state, "pipeline"):
            return DetectionResult(
                session_id=session_id,
                frame_id=frame_id,
                is_violation=False,
                overall_confidence=0.0,
                processing_time_ms=0.0,
                error_message="ML pipeline not loaded",
            )

        pipeline = Database.app.state.pipeline

        try:
            frame, (height, width) = self.decode_base64_frame(frame_base64)
        except ValueError as e:
            return DetectionResult(
                session_id=session_id,
                frame_id=frame_id,
                is_violation=False,
                overall_confidence=0.0,
                processing_time_ms=0.0,
                error_message=str(e),
            )

        runtime_params = self._get_session_runtime_params(session_id)

        try:
            ml_result = pipeline.pipeline.run_on_frame(
                frame=frame,
                draw_visualizer=False,
                runtime_parameters=runtime_params,
            )

            # Validate pipeline output format
            if not isinstance(ml_result, dict):
                raise ValueError(f"Pipeline returned non-dict: {type(ml_result)}")
            if "person_results" not in ml_result:
                raise ValueError("Pipeline output missing 'person_results' key")

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return DetectionResult(
                session_id=session_id,
                frame_id=frame_id,
                is_violation=False,
                overall_confidence=0.0,
                processing_time_ms=processing_time,
                error_message=f"ML pipeline error: {str(e)}",
            )

        pedestrians = []
        overall_confidence = 0.0

        if ml_result.get("person_results"):
            for person_result in ml_result["person_results"]:
                ped = self._build_pedestrian_result(person_result, height, width)
                pedestrians.append(ped)

                if ped.is_violation:
                    overall_confidence = max(overall_confidence, ped.phone_confidence)

        is_violation = any(p.is_violation for p in pedestrians)
        processing_time = (time.time() - start_time) * 1000

        detection = DetectionResult(
            session_id=session_id,
            frame_id=frame_id,
            is_violation=is_violation,
            overall_confidence=overall_confidence,
            processing_time_ms=processing_time,
            pedestrians=pedestrians,
        )

        # CRITICAL FIX: Enrich violation pedestrians (modifies pedestrians in-place with face_id)
        # This must happen BEFORE storing to DB to ensure face_id is in the document
        if is_violation:
            violation_tasks = [
                self._enrich_violation_pedestrian(frame, detection, ped)
                for ped in pedestrians
                if ped.is_violation
            ]
            if violation_tasks:
                await asyncio.gather(*violation_tasks)

        try:
            db: AsyncIOMotorDatabase = Database.get_database()
            await db.detections.insert_one(
                {
                    "detection_id": detection.detection_id,
                    "timestamp": detection.timestamp,
                    "session_id": detection.session_id,
                    "frame_id": detection.frame_id,
                    "is_violation": detection.is_violation,
                    "overall_confidence": detection.overall_confidence,
                    "processing_time_ms": detection.processing_time_ms,
                    "pedestrians": [p.dict() for p in detection.pedestrians],
                    "error_message": detection.error_message,
                }
            )
            DashboardService.clear_cache()
        except Exception as e:
            print(f"Warning: failed to save detection to MongoDB: {e}")

        return detection
