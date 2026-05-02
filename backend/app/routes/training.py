"""
Training API routes.

POST /api/train — start training job
GET /api/train/status — current/last job status
GET /api/train/history — past training jobs
DELETE /api/train/current — cancel job
WebSocket /ws/train-logs — stream training logs
"""

from typing import Any, Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.database import Database
from app.models.training import TrainRequest, TrainingStatus
from app.services.training_service import TrainingJobManager, TrainingService

router = APIRouter(tags=["Training"])


@router.post("/api/train")
async def start_training(request: TrainRequest) -> dict[str, Any]:
    """
    Start a new training job for the specified model.

    Body:
    ```
    {
      "model_type": "posture_classifier" | "phone_detector",
      "epochs": 50,
      "learning_rate": 0.001,
      "batch_size": 32
    }
    ```

    Returns:
    ```
    {
      "job_id": "uuid",
      "status": "started"
    }
    ```
    """
    try:
        # Check if training already running
        if await TrainingService.is_training_running():
            return {
                "success": False,
                "error": "Training job already running. Cancel it first.",
            }

        # Start training
        job_id = await TrainingService.start_training(
            model_type=request.model_type,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
        )

        return {
            "success": True,
            "job_id": job_id,
            "status": "started",
        }

    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
        }
    except RuntimeError as e:
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/api/train/status")
async def get_training_status() -> dict[str, Any]:
    """
    Get status of current or last training job.

    Returns:
    ```
    {
      "job_id": "uuid",
      "status": "running" | "completed" | "error" | "cancelled" | "none",
      "model_type": "posture_classifier",
      "started_at": "ISO8601",
      "current_epoch": 25,
      "total_epochs": 50,
      "progress_percent": 50.0
    }
    ```
    """
    status = await TrainingService.get_current_job_status()

    if not status:
        return {
            "job_id": None,
            "status": "none",
            "model_type": None,
            "started_at": None,
            "current_epoch": 0,
            "total_epochs": 0,
            "progress_percent": 0.0,
        }

    return {
        "job_id": str(status.get("job_id")),
        "status": status.get("status", "unknown"),
        "model_type": status.get("model_type"),
        "started_at": status.get("started_at"),
        "current_epoch": status.get("current_epoch", 0),
        "total_epochs": status.get("epochs", 0),
        "progress_percent": (
            status.get("current_epoch", 0) / status.get("epochs", 1) * 100
            if status.get("epochs", 0) > 0
            else 0.0
        ),
    }


@router.get("/api/train/history")
async def get_training_history(
    limit: int = Query(20, ge=1, le=100, description="Number of past jobs to return")
) -> dict[str, Any]:
    """
    Get list of past training jobs.

    Returns:
    ```
    {
      "jobs": [
        {
          "job_id": "uuid",
          "model_type": "posture_classifier",
          "status": "completed",
          "started_at": "ISO8601",
          "completed_at": "ISO8601",
          "epochs": 50,
          "current_epoch": 50,
          "metrics": {
            "final_accuracy": 0.9178,
            "final_loss": 0.21
          }
        }
      ]
    }
    ```
    """
    try:
        db = Database.get_database()
        cursor = db.training_logs.find({}).sort("started_at", -1).limit(limit)
        history = await cursor.to_list(length=limit)
        for item in history:
            item["_id"] = str(item["_id"])
        return {"history": history, "total": len(history)}
    except Exception as e:
        return {"history": [], "total": 0, "error": str(e)}


@router.delete("/api/train/current")
async def cancel_training() -> dict[str, Any]:
    """
    Cancel currently running training job.

    Returns:
    ```
    {
      "success": true,
      "message": "Training cancelled"
    }
    ```
    """
    result = await TrainingService.cancel_training()

    if result:
        return {
            "success": True,
            "message": "Training cancelled",
        }
    else:
        return {
            "success": False,
            "message": "No training job running",
        }


@router.websocket("/ws/train-logs")
async def websocket_training_logs(websocket: WebSocket):
    """
    WebSocket endpoint for streaming training logs.

    Messages:
    ```
    {
      "type": "progress",
      "epoch": 5,
      "total_epochs": 50,
      "loss": 0.312,
      "accuracy": 0.78
    }

    {
      "type": "log",
      "message": "Epoch 5/50 complete"
    }

    {
      "type": "complete",
      "metrics": {
        "final_accuracy": 0.9178,
        "final_loss": 0.21
      }
    }

    {
      "type": "error",
      "message": "CUDA out of memory"
    }
    ```
    """
    await websocket.accept()

    # Register this client
    manager = TrainingJobManager.get_instance()
    manager.add_websocket_client(websocket)

    try:
        # Keep connection alive
        while True:
            # Receive any message from client (for keep-alive)
            data = await websocket.receive_text()
            # Clients can send "ping" or similar to keep connection alive
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')

    except WebSocketDisconnect:
        manager.remove_websocket_client(websocket)
        print("✅ WebSocket training logs client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        manager.remove_websocket_client(websocket)
