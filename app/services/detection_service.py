"""
Detection service for interfacing with the ML predictor.
"""

import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import generate_unique_filename
from app.db.models import Report
from src.serving.predictor import Predictor


class DetectionService:
    """Service for running predictions using the ML predictor."""

    def __init__(self):
        """Initialize the detection service with predictor."""
        self.predictor = Predictor(log_dir=settings.LOGS_DIR, log_level="INFO")
        self.settings = settings

    def detect_image(
        self,
        image_path: Path,
        user_id: int,
        db: Session,
        save_result: bool = True,
        recognized_person: Optional[str] = None,
        recognition_confidence: Optional[float] = None,
    ) -> dict:
        """
        Run detection on an image file.

        Args:
            image_path: Path to image file
            user_id: User ID for database record
            db: Database session
            save_result: Whether to save rendered output
            recognized_person: Name of recognized person (if any)
            recognition_confidence: Confidence score of recognition

        Returns:
            Dictionary with detection results
        """
        start_time = time.time()

        try:
            # Run prediction using the predictor
            result = self.predictor.predict_image(
                image_path, save_rendered_output=save_result
            )

            processing_time = (
                time.time() - start_time
            ) * 1000  # Convert to milliseconds

            # Determine alert status
            alert_triggered = False
            alert_reason = None

            if result["num_persons"] > 0:
                for person in result["person_results"]:
                    if person["phone"]:
                        alert_triggered = True
                        if recognized_person:
                            alert_reason = (
                                f"KNOWN_PERSON_USING_PHONE: {recognized_person}"
                            )
                        else:
                            alert_reason = "PHONE_USAGE_DETECTED"
                        break

            # Save report to database
            alert_level = (
                "critical"
                if (alert_triggered and recognized_person)
                else ("high" if alert_triggered else "low")
            )

            report = Report(
                user_id=user_id,
                image_path=str(image_path),
                result_image_path=result.get("saved_result_path"),
                num_persons=result["num_persons"],
                phone_detected=any(p["phone"] for p in result["person_results"]),
                distraction_detected=any(
                    p["posture"] == "distracted" for p in result["person_results"]
                ),
                face_recognized=recognized_person is not None,
                recognized_person=recognized_person,
                recognition_confidence=recognition_confidence,
                alert_level=alert_level,
                detection_data=json.dumps(result),
            )

            db.add(report)
            db.commit()
            db.refresh(report)

            return {
                "num_persons": result["num_persons"],
                "person_results": result["person_results"],
                "saved_result_path": result.get("saved_result_path"),
                "processing_time": processing_time,
                "alert_triggered": alert_triggered,
                "alert_reason": alert_reason,
                "report_id": report.id,
            }

        except Exception as e:
            raise Exception(f"Detection failed: {str(e)}")

    def detect_frame(
        self,
        frame_base64: str,
        user_id: int,
        db: Session,
        recognized_person: Optional[str] = None,
        recognition_confidence: Optional[float] = None,
    ) -> dict:
        """
        Run detection on a base64 encoded frame (optimized for real-time).

        Args:
            frame_base64: Base64 encoded image
            user_id: User ID
            db: Database session
            recognized_person: Name of recognized person
            recognition_confidence: Recognition confidence

        Returns:
            Dictionary with detection results
        """
        start_time = time.time()

        try:
            # Decode base64
            image_data = base64.b64decode(frame_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Invalid image data")

            # Save to temporary file for predictor
            temp_filename = generate_unique_filename("frame.jpg")
            temp_path = self.settings.UPLOAD_DIR / temp_filename

            success = cv2.imwrite(str(temp_path), frame)
            if not success:
                raise IOError(f"Failed to save temporary frame: {temp_path}")

            # Run detection
            result = self.predictor.predict_image(temp_path, save_rendered_output=False)

            processing_time = (time.time() - start_time) * 1000

            # Determine alert
            alert_triggered = False
            alert_reason = None

            if result["num_persons"] > 0:
                for person in result["person_results"]:
                    if person["phone"]:
                        alert_triggered = True
                        if recognized_person:
                            alert_reason = (
                                f"KNOWN_PERSON_USING_PHONE: {recognized_person}"
                            )
                        else:
                            alert_reason = "PHONE_USAGE_DETECTED"
                        break

            # Clean up temp file
            try:
                temp_path.unlink()
            except Exception:
                pass

            return {
                "num_persons": result["num_persons"],
                "person_results": result["person_results"],
                "alert_triggered": alert_triggered,
                "alert_reason": alert_reason,
                "processing_time": processing_time,
            }

        except Exception as e:
            raise Exception(f"Frame detection failed: {str(e)}")

    @staticmethod
    def get_reports(
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 50,
        alert_only: bool = False,
    ) -> tuple[list[Report], int]:
        """
        Get detection reports for a user.

        Args:
            db: Database session
            user_id: User ID
            skip: Skip this many results
            limit: Limit results to this many
            alert_only: Only return alert reports

        Returns:
            Tuple of (reports list, total count)
        """
        query = db.query(Report).filter(Report.user_id == user_id)

        if alert_only:
            query = query.filter(Report.alert_level.in_(["high", "critical"]))

        total = query.count()
        reports = (
            query.order_by(Report.created_at.desc()).offset(skip).limit(limit).all()
        )

        return reports, total

    @staticmethod
    def delete_report(db: Session, user_id: int, report_id: int) -> bool:
        """
        Delete a report.

        Args:
            db: Database session
            user_id: User ID
            report_id: Report ID

        Returns:
            True if deleted
        """
        report = (
            db.query(Report)
            .filter((Report.id == report_id) & (Report.user_id == user_id))
            .first()
        )

        if not report:
            raise ValueError("Report not found")

        db.delete(report)
        db.commit()
        return True
