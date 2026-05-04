"""
Face cropping, embedding, and deduplication service.

Handles:
- Face cropping from original frame using face_region coordinates
- 128-D embedding generation
- Cosine similarity deduplication
- MongoDB face record updates/inserts
"""

import base64
from datetime import datetime
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import Database
from app.models.detection import DetectionResult, PedestrianResult
from app.utils.embedding_utils import cosine_similarity, generate_face_embedding


class FaceService:
    """Service for face processing and deduplication."""

    DEFAULT_SIMILARITY_THRESHOLD = 0.85

    @staticmethod
    def _crop_face_from_frame(
        frame: np.ndarray, face_region: dict
    ) -> Optional[np.ndarray]:
        """
        Extract face crop from frame using bounding box coordinates.

        Args:
            frame: Original BGR frame
            face_region: Dict with x1, y1, x2, y2 keys (pixel coords)

        Returns:
            Face crop as BGR numpy array, or None if crop invalid
        """
        try:
            x1 = max(0, int(face_region.get("x1", 0)))
            y1 = max(0, int(face_region.get("y1", 0)))
            x2 = min(frame.shape[1], int(face_region.get("x2", frame.shape[1])))
            y2 = min(frame.shape[0], int(face_region.get("y2", frame.shape[0])))

            if x2 <= x1 or y2 <= y1:
                print(
                    f"Warning: invalid crop coordinates x1={x1} y1={y1} x2={x2} y2={y2} — skipping"
                )
                return None

            face_crop = frame[y1:y2, x1:x2]

            if face_crop is None or face_crop.size == 0:
                print("Warning: face crop is empty after slicing frame")
                return None

            face_crop_resized = cv2.resize(
                face_crop, (160, 160), interpolation=cv2.INTER_LINEAR
            )

            return face_crop_resized

        except Exception as e:
            print(f"Warning: _crop_face_from_frame failed: {e}")
            return None

    @staticmethod
    def _face_crop_to_base64(face_crop: np.ndarray) -> str:
        """
        Encode face crop as Base64 JPEG string.

        Args:
            face_crop: Face image as numpy BGR array

        Returns:
            Base64-encoded JPEG string
        """
        try:
            success, buffer = cv2.imencode(
                ".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not success:
                print("Warning: cv2.imencode failed to encode face crop")
                return ""
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print(f"Warning: _face_crop_to_base64 failed: {e}")
            return ""

    @staticmethod
    async def _get_similarity_threshold(db: AsyncIOMotorDatabase) -> float:
        """
        Read face similarity threshold from MongoDB settings.

        Falls back to default if not found.
        """
        try:
            settings = await db.settings.find_one({"_id": "default"})
            if settings and "face_similarity_threshold" in settings:
                return float(settings["face_similarity_threshold"])
        except Exception as e:
            print(f"Warning: could not read similarity threshold from DB: {e}")
        return FaceService.DEFAULT_SIMILARITY_THRESHOLD

    @staticmethod
    async def _find_matching_face(
        embedding: list[float],
        threshold: float,
        db: AsyncIOMotorDatabase,
        max_age_days: int = 30,
    ) -> Optional[dict]:
        """
        Find existing face with similar embedding in MongoDB.

        Args:
            embedding: New face embedding (128-D)
            threshold: Similarity threshold (0.85 recommended)
            db: AsyncIOMotorDatabase
            max_age_days: Only query faces seen in last N days

        Returns:
            Existing face document if found, None otherwise
        """
        try:
            from datetime import timedelta

            cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)

            faces = await db.faces.find({"last_seen": {"$gte": cutoff_time}}).to_list(
                length=1000
            )

            best_match = None
            best_similarity = 0.0

            for face in faces:
                stored_embedding = face.get("embedding", [])
                similarity = cosine_similarity(embedding, stored_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity >= threshold:
                        best_match = face

            return best_match if best_similarity >= threshold else None

        except Exception as e:
            print(f"Warning: _find_matching_face failed: {e}")
            return None

    @staticmethod
    async def process_face(
        original_frame: np.ndarray,
        pedestrian: PedestrianResult,
        detection_result: DetectionResult,
    ) -> Optional[str]:
        """
        Process face from detection: crop, embed, deduplicate, save to DB.

        Args:
            original_frame: Original frame as numpy BGR array
            pedestrian: PedestrianResult with face_region
            detection_result: Full DetectionResult for context

        Returns:
            face_id (string) of stored/updated face, or None if processing failed
        """
        db: AsyncIOMotorDatabase = Database.get_database()

        # FIX 1: Log clearly when face_region is missing instead of silent return
        if not pedestrian.face_region:
            print("Warning: pedestrian has no face_region — cannot crop or save face")
            return None

        # Crop face from frame
        face_crop = FaceService._crop_face_from_frame(
            original_frame,
            {
                "x1": pedestrian.face_region.x1,
                "y1": pedestrian.face_region.y1,
                "x2": pedestrian.face_region.x2,
                "y2": pedestrian.face_region.y2,
            },
        )

        if face_crop is None:
            print("Warning: face crop returned None — skipping save")
            return None

        # Generate embedding
        try:
            embedding = generate_face_embedding(face_crop)
        except Exception as e:
            print(f"Warning: embedding generation failed: {e}")
            return None

        # FIX 2: Convert numpy array to plain Python list before MongoDB insert
        # MongoDB BSON cannot serialize numpy arrays — this was silently failing
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # Find similar face
        threshold = await FaceService._get_similarity_threshold(db)
        matching_face = await FaceService._find_matching_face(embedding, threshold, db)

        # Encode face to Base64
        face_base64 = FaceService._face_crop_to_base64(face_crop)

        if not face_base64:
            print(
                "Warning: face base64 encoding produced empty string — image may not display in UI"
            )

        now = datetime.utcnow()

        # FIX 3: Extract session_id and violation_type for frontend display
        session_id = detection_result.session_id or "unknown"
        violation_type = pedestrian.posture_state or "distracted"

        if matching_face:
            # Update existing face record
            face_id = matching_face["_id"]

            try:
                await db.faces.update_one(
                    {"_id": face_id},
                    {
                        "$set": {
                            "last_seen": now,
                            "image_base64": face_base64,
                            # FIX 4: Keep status as active on re-detection
                            # Only mark resolved manually from frontend
                            "status": matching_face.get("status", "active"),
                        },
                        "$inc": {
                            "detection_count": 1,
                            "violation_count": (
                                1 if detection_result.is_violation else 0
                            ),
                        },
                        "$push": {
                            "detection_ids": detection_result.detection_id,
                        },
                    },
                )
                print(f"Info: updated existing face record — face_id={face_id}")
            except Exception as e:
                print(f"Warning: failed to update face in MongoDB: {e}")

            return face_id

        else:
            # Create new face record
            face_id = str(uuid4())

            try:
                await db.faces.insert_one(
                    {
                        "_id": face_id,
                        # FIX 5: Added all fields the frontend Violators UI expects
                        "embedding": embedding,
                        "image_base64": face_base64,
                        "first_seen": now,
                        "last_seen": now,
                        "detection_count": 1,
                        "violation_count": 1 if detection_result.is_violation else 0,
                        "detection_ids": [detection_result.detection_id],
                        "session_id": session_id,
                        "status": "active",
                        "resolved": False,
                        "violation_type": violation_type,
                        "confidence": pedestrian.posture_confidence,
                        "zone": session_id,
                        "camera_id": session_id,
                    }
                )
                print(f"Info: new face saved to MongoDB — face_id={face_id}")
            except Exception as e:
                print(f"Warning: failed to insert face into MongoDB: {e}")
                return None

            return face_id
