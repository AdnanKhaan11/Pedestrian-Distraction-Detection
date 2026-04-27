"""
Face recognition service for detecting and recognizing faces.
"""

import base64
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import generate_unique_filename
from app.db.models import RegisteredFace, User


class FaceRecognitionService:
    """Service for face detection and recognition."""

    def __init__(self):
        """Initialize face recognition service."""
        self.model = settings.FACE_RECOGNITION_MODEL
        self.threshold = settings.FACE_DISTANCE_THRESHOLD
        self.min_face_size = settings.MIN_FACE_SIZE
        self.embeddings_dir = settings.EMBEDDINGS_DIR

    def register_face(
        self,
        image_data: bytes,
        person_name: str,
        user_id: int,
        db: Session,
    ) -> dict:
        """
        Register a new face for a person.

        Args:
            image_data: Image bytes
            person_name: Name of the person
            user_id: User ID
            db: Database session

        Returns:
            Dictionary with registration details
        """
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image data")

        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_image, model=settings.FACE_DETECTION_MODEL
        )

        if len(face_locations) == 0:
            raise ValueError("No face detected in image")

        if len(face_locations) > 1:
            raise ValueError(
                "Multiple faces detected, please provide image with one face"
            )

        # Generate embedding
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if len(face_encodings) == 0:
            raise ValueError("Could not generate face embedding")

        embedding = face_encodings[0]

        # Save embedding and image
        embedding_filename = generate_unique_filename(f"{person_name}_embedding.pkl")
        embedding_path = self.embeddings_dir / embedding_filename

        with open(embedding_path, "wb") as f:
            pickle.dump(embedding, f)

        # Save original image
        image_filename = generate_unique_filename(f"{person_name}_face.jpg")
        image_path = self.embeddings_dir / image_filename
        cv2.imwrite(str(image_path), image)

        # Store in database
        registered_face = RegisteredFace(
            user_id=user_id,
            person_name=person_name,
            embedding_path=str(embedding_path),
            image_path=str(image_path),
            confidence=0.95,  # Initial confidence
            is_active=True,
        )

        db.add(registered_face)
        db.commit()
        db.refresh(registered_face)

        return {
            "id": registered_face.id,
            "person_name": person_name,
            "confidence": registered_face.confidence,
            "message": f"Face registered successfully for {person_name}",
        }

    def recognize_faces(
        self,
        image_data: bytes,
        user_id: int,
        db: Session,
        threshold: Optional[float] = None,
    ) -> dict:
        """
        Recognize faces in an image.

        Args:
            image_data: Image bytes
            user_id: User ID
            db: Database session
            threshold: Optional custom threshold

        Returns:
            Dictionary with recognition results
        """
        if threshold is None:
            threshold = self.threshold

        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image data")

        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_image, model=settings.FACE_DETECTION_MODEL
        )

        if len(face_locations) == 0:
            return {
                "faces_found": 0,
                "matches": [],
                "processing_time": 0,
            }

        # Generate encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Get registered faces for this user
        registered = (
            db.query(RegisteredFace)
            .filter((RegisteredFace.user_id == user_id) & (RegisteredFace.is_active))
            .all()
        )

        matches = []

        for face_encoding in face_encodings:
            best_match = None
            best_distance = float("inf")

            for reg_face in registered:
                try:
                    with open(reg_face.embedding_path, "rb") as f:
                        stored_embedding = pickle.load(f)

                    distance = face_recognition.face_distance(
                        [stored_embedding], face_encoding
                    )[0]

                    if distance < best_distance:
                        best_distance = distance
                        best_match = reg_face

                except Exception:
                    continue

            if best_match and best_distance <= threshold:
                confidence = 1.0 - best_distance
                matches.append(
                    {
                        "person_name": best_match.person_name,
                        "confidence": float(confidence),
                        "distance": float(best_distance),
                    }
                )

        return {
            "faces_found": len(face_locations),
            "matches": matches,
            "processing_time": 0,
        }

    def recognize_face_by_image_path(
        self,
        image_path: Path,
        user_id: int,
        db: Session,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Recognize a face from an image path (convenience method).

        Args:
            image_path: Path to image file
            user_id: User ID
            db: Database session
            threshold: Optional custom threshold

        Returns:
            Tuple of (person_name, confidence) or None if no match
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        # Convert to bytes for recognition
        _, image_data = cv2.imencode(".jpg", image)

        result = self.recognize_faces(image_data.tobytes(), user_id, db, threshold)

        if result["matches"]:
            top_match = max(result["matches"], key=lambda x: x["confidence"])
            return (top_match["person_name"], top_match["confidence"])

        return None

    def list_registered_faces(
        self,
        user_id: int,
        db: Session,
    ) -> List[dict]:
        """
        List all registered faces for a user.

        Args:
            user_id: User ID
            db: Database session

        Returns:
            List of registered faces
        """
        faces = (
            db.query(RegisteredFace)
            .filter((RegisteredFace.user_id == user_id) & (RegisteredFace.is_active))
            .all()
        )

        return [
            {
                "id": face.id,
                "person_name": face.person_name,
                "confidence": face.confidence,
                "created_at": face.created_at.isoformat(),
            }
            for face in faces
        ]

    def delete_registered_face(
        self,
        user_id: int,
        face_id: int,
        db: Session,
    ) -> bool:
        """
        Delete a registered face.

        Args:
            user_id: User ID
            face_id: Face ID
            db: Database session

        Returns:
            True if deleted
        """
        face = (
            db.query(RegisteredFace)
            .filter(
                (RegisteredFace.id == face_id) & (RegisteredFace.user_id == user_id)
            )
            .first()
        )

        if not face:
            raise ValueError("Face not found")

        db.delete(face)
        db.commit()
        return True
