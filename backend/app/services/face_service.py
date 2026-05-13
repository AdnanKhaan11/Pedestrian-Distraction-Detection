"""
Face embedding, deduplication, and MongoDB storage service.

Handles:
- Decoding model face crop from announced_face_b64 (no re-cropping needed)
- Embedding generation for deduplication
- Cosine similarity deduplication
- MongoDB face record updates/inserts

Note: Face cropping is done by the ML model (runtime_detector.py).
      facenet_pytorch has been removed — the model handles face detection.
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
    def _decode_face_b64(face_b64: str) -> Optional[np.ndarray]:
        """
        Decode base64 JPEG face crop (produced by the ML model) into a numpy BGR array.

        Args:
            face_b64: Base64-encoded JPEG string from pedestrian.announced_face_b64

        Returns:
            Face image as BGR numpy array resized to 160x160, or None if decoding fails
        """
        try:
            face_bytes = base64.b64decode(face_b64)
            face_array = cv2.imdecode(
                np.frombuffer(face_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if face_array is None or face_array.size == 0:
                print("Warning: decoded face array is empty")
                return None
            # Resize to 160x160 — standard input size for embedding model
            return cv2.resize(face_array, (160, 160), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Warning: _decode_face_b64 failed: {e}")
            return None

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
            embedding: New face embedding vector
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
        original_frame: Optional[
            np.ndarray
        ],  # kept for signature compat, no longer used
        pedestrian: PedestrianResult,
        detection_result: DetectionResult,
    ) -> Optional[str]:
        """
        Process face from detection: decode model crop, embed, deduplicate, save to DB.

        The face crop comes directly from the ML model via pedestrian.announced_face_b64.
        No re-cropping from the original frame is needed.

        Args:
            original_frame: Not used — kept for backward compatibility only
            pedestrian: PedestrianResult with announced_face_b64 from the ML model
            detection_result: Full DetectionResult for context

        Returns:
            face_id (string) of stored/updated face, or None if processing failed
        """
        db: AsyncIOMotorDatabase = Database.get_database()

        # Use the face crop produced by the ML model directly
        if not pedestrian.announced_face_b64:
            print(
                "Warning: pedestrian has no announced_face_b64 — face was not announced this frame"
            )
            return None

        # Decode the model's face crop from base64 to numpy array
        face_crop = FaceService._decode_face_b64(pedestrian.announced_face_b64)

        if face_crop is None:
            print("Warning: face crop decoding returned None — skipping save")
            return None

        # Generate embedding for deduplication
        try:
            embedding = generate_face_embedding(face_crop)
        except Exception as e:
            print(f"Warning: embedding generation failed: {e}")
            return None

        # Convert numpy array to plain Python list before MongoDB insert
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # Find similar face in DB
        threshold = await FaceService._get_similarity_threshold(db)
        matching_face = await FaceService._find_matching_face(embedding, threshold, db)

        # Use the base64 string directly — already encoded by the model
        face_base64 = pedestrian.announced_face_b64

        now = datetime.utcnow()
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
                        "face_id": face_id,
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
