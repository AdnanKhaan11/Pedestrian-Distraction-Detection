"""
Detection API routes.

POST /api/detect — single frame detection
WebSocket /ws/stream — live frame streaming
"""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.models.detection import DetectionResult
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/api", tags=["Detection"])


class DetectRequest(BaseModel):
    """Request body for POST /api/detect endpoint."""

    image: str  # Base64-encoded JPEG
    session_id: str = "default_session"


@router.post("/detect")
async def detect_single_frame(request: DetectRequest) -> dict[str, Any]:
    """
    Run inference on a single frame.

    Request body:
    {
        "image": "<base64_jpeg_string>",
        "session_id": "cam_001"
    }

    Returns:
        DetectionResult as JSON
    """
    service = InferenceService()
    result = await service.run_inference(
        frame_base64=request.image,
        session_id=request.session_id,
        frame_id=0,
    )

    return result.dict()


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for live frame streaming.

    Accepts messages:
    {
        "frame": "<base64_jpeg_string>",
        "session_id": "cam_001",
        "timestamp": "2024-01-01T12:00:00Z"
    }

    Sends back DetectionResult JSON.
    Handles disconnect and errors gracefully.
    """
    await websocket.accept()

    frame_id = 0
    service = InferenceService()

    try:
        while True:
            # Add backpressure throttling to prevent client overload
            await asyncio.sleep(0.03)  # ~33 FPS max

            # Receive frame
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except json.JSONDecodeError as e:
                await websocket.send_json(
                    {"error": f"Invalid JSON in message: {str(e)}"}
                )
                continue

            frame_base64 = message.get("frame", "")
            session_id = message.get("session_id", "unknown")

            if not frame_base64:
                await websocket.send_json(
                    {"error": "Missing 'frame' (base64 JPEG) in message"}
                )
                continue

            # Run inference
            try:
                result = await service.run_inference(
                    frame_base64=frame_base64,
                    session_id=session_id,
                    frame_id=frame_id,
                )
                frame_id += 1

                # Send result back
                await websocket.send_json(result.dict())

            except Exception as e:
                # Send error but don't close connection
                await websocket.send_json(
                    {
                        "error": f"Inference failed: {str(e)}",
                        "frame_id": frame_id,
                    }
                )
                frame_id += 1

    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected: session={session_id}")

    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011, reason="Server error")
        except Exception:
            pass
