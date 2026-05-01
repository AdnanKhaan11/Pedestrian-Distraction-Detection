"""
Training service for managing background model training jobs.

Handles:
- Job lifecycle management (start, cancel, status)
- Subprocess execution with line-by-line log capture
- MongoDB persistence of training logs and metrics
- WebSocket broadcasting to connected clients
- GPU/resource management (prevent concurrent jobs)
"""

import asyncio
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import Database
from app.models.training import TrainingJob, TrainingLog


class TrainingJobManager:
    """Singleton to manage current training job and WebSocket connections."""

    _instance: Optional["TrainingJobManager"] = None
    _current_job_id: Optional[str] = None
    _current_process: Optional[Any] = None
    _websocket_clients: list[Any] = []  # WebSocket connections
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def set_current_job(self, job_id: str):
        """Mark job as currently running."""
        async with self._lock:
            self._current_job_id = job_id

    async def clear_current_job(self):
        """Mark no job is running."""
        async with self._lock:
            self._current_job_id = None

    async def set_current_process(self, process: Any):
        """Track the active training subprocess."""
        async with self._lock:
            self._current_process = process

    async def clear_current_process(self):
        """Forget the active training subprocess."""
        async with self._lock:
            self._current_process = None

    async def terminate_current_process(self) -> bool:
        """Terminate the active training subprocess if it is still running."""
        async with self._lock:
            process = self._current_process
            if process is None or process.returncode is not None:
                return False
            process.terminate()
            return True

    async def get_current_job_id(self) -> Optional[str]:
        """Get ID of currently running job."""
        async with self._lock:
            return self._current_job_id

    def add_websocket_client(self, websocket):
        """Register WebSocket client for log streaming."""
        self._websocket_clients.append(websocket)

    def remove_websocket_client(self, websocket):
        """Unregister WebSocket client."""
        if websocket in self._websocket_clients:
            self._websocket_clients.remove(websocket)

    async def broadcast_log(self, log: TrainingLog):
        """Send log message to all connected WebSocket clients."""
        message = log.dict(exclude_none=True)
        disconnected = []
        for client in self._websocket_clients:
            try:
                await client.send_json(message)
            except Exception:
                # Client disconnected
                disconnected.append(client)

        # Clean up disconnected clients
        for client in disconnected:
            self.remove_websocket_client(client)


class TrainingService:
    """Service for background model training with logging and WebSocket streaming."""

    SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "scripts"

    SCRIPT_MAPPING = {
        "posture_classifier": "train_posture_model.py",
        "phone_detector": "train_phone_model.py",
    }

    @staticmethod
    async def get_current_job_status() -> Optional[dict[str, Any]]:
        """Get status of currently running training job."""
        db: AsyncIOMotorDatabase = Database.get_database()
        manager = TrainingJobManager.get_instance()

        job_id = await manager.get_current_job_id()
        if not job_id:
            return None

        job = await db.training_logs.find_one({"_id": job_id})
        return job

    @staticmethod
    async def is_training_running() -> bool:
        """Check if any training job is currently running."""
        db: AsyncIOMotorDatabase = Database.get_database()
        running_job = await db.training_logs.find_one({"status": "running"})
        return running_job is not None

    @staticmethod
    async def start_training(
        model_type: str, epochs: int, learning_rate: float, batch_size: int
    ) -> str:
        """
        Start a new training job in the background.

        Args:
            model_type: "posture_classifier" or "phone_detector"
            epochs: Number of epochs
            learning_rate: Learning rate
            batch_size: Batch size

        Returns:
            job_id

        Raises:
            RuntimeError: If training already running
            ValueError: If model_type invalid
        """
        db: AsyncIOMotorDatabase = Database.get_database()
        manager = TrainingJobManager.get_instance()

        # Validate model type
        if model_type not in TrainingService.SCRIPT_MAPPING:
            raise ValueError(f"Invalid model_type: {model_type}")

        # Check if already running
        if await TrainingService.is_training_running():
            raise RuntimeError("Training job already running")

        # Create job document
        job = TrainingJob(
            model_type=model_type,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            status="running",
        )

        # Insert into MongoDB
        job_data = job.dict()
        job_data["_id"] = job.job_id
        await db.training_logs.insert_one(job_data)
        job_id = job.job_id

        # Set as current job
        await manager.set_current_job(job_id)

        # Launch training in background (fire-and-forget)
        asyncio.create_task(
            TrainingService._run_training_subprocess(
                job_id, model_type, epochs, learning_rate, batch_size
            )
        )

        return str(job_id)

    @staticmethod
    async def _run_training_subprocess(
        job_id: str, model_type: str, epochs: int, learning_rate: float, batch_size: int
    ):
        """
        Run training subprocess with output capture and log streaming.

        This runs in a background task and should not block.
        """
        db: AsyncIOMotorDatabase = Database.get_database()
        manager = TrainingJobManager.get_instance()

        script_name = TrainingService.SCRIPT_MAPPING[model_type]
        script_path = TrainingService.SCRIPTS_DIR / script_name

        if not script_path.exists():
            error_msg = f"Training script not found: {script_path}"
            print(f"❌ {error_msg}")
            await db.training_logs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "error",
                        "error_message": error_msg,
                        "completed_at": datetime.utcnow(),
                    }
                },
            )
            await manager.clear_current_job()
            return

        try:
            # Build command
            # TODO: Pass epochs, learning_rate, batch_size to script if it supports them
            command = [
                sys.executable,
                str(script_path),
            ]

            print(f"🚀 Starting training: {' '.join(command)}")

            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Redirect stderr to stdout
                cwd=str(TrainingService.SCRIPTS_DIR.parent),
            )
            await manager.set_current_process(process)

            # Capture output line by line
            logs = []
            current_epoch = 0
            total_epochs = epochs
            loss = None
            accuracy = None

            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str:
                    continue

                print(f"[{model_type}] {line_str}")

                # Store raw log
                logs.append(line_str)
                logs = logs[-500:]

                # Try to parse progress from log line
                # Look for patterns like "Epoch 5/50" or "accuracy: 0.78"
                epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", line_str, re.IGNORECASE)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))

                loss_match = re.search(r"loss[:\s]+([0-9.]+)", line_str, re.IGNORECASE)
                if loss_match:
                    loss = float(loss_match.group(1))

                acc_match = re.search(
                    r"accuracy[:\s]+([0-9.]+)", line_str, re.IGNORECASE
                )
                if acc_match:
                    accuracy = float(acc_match.group(1))

                # Broadcast progress if detected
                if epoch_match or loss_match or acc_match:
                    progress_log = TrainingLog(
                        type="progress",
                        epoch=current_epoch if epoch_match else None,
                        total_epochs=total_epochs if epoch_match else None,
                        loss=loss,
                        accuracy=accuracy,
                    )
                    await manager.broadcast_log(progress_log)

                # Broadcast raw log line
                log_entry = TrainingLog(type="log", message=line_str)
                await manager.broadcast_log(log_entry)

                # Update MongoDB with current progress
                progress_pct = (
                    (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
                )
                await db.training_logs.update_one(
                    {"_id": job_id},
                    {
                        "$set": {
                            "current_epoch": current_epoch,
                            "logs": logs,
                        }
                    },
                )

            # Wait for process to complete
            return_code = await process.wait()

            if return_code == 0:
                # Success - extract metrics from logs
                metrics = await TrainingService._extract_metrics(logs)

                # Broadcast completion
                complete_log = TrainingLog(
                    type="complete",
                    metrics=metrics,
                )
                await manager.broadcast_log(complete_log)

                # Update job as completed
                await db.training_logs.update_one(
                    {"_id": job_id},
                    {
                        "$set": {
                            "status": "completed",
                            "completed_at": datetime.utcnow(),
                            "metrics": metrics,
                            "logs": logs,
                        }
                    },
                )

                print(f"✅ Training completed: {job_id}")
            else:
                # Failure
                error_msg = f"Training script exited with code {return_code}"
                print(f"❌ {error_msg}")

                error_log = TrainingLog(
                    type="error",
                    message=error_msg,
                )
                await manager.broadcast_log(error_log)

                await db.training_logs.update_one(
                    {"_id": job_id},
                    {
                        "$set": {
                            "status": "error",
                            "completed_at": datetime.utcnow(),
                            "error_message": error_msg,
                            "logs": logs,
                        }
                    },
                )

        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            print(f"❌ {error_msg}")

            error_log = TrainingLog(type="error", message=error_msg)
            await manager.broadcast_log(error_log)

            await db.training_logs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "error",
                        "completed_at": datetime.utcnow(),
                        "error_message": error_msg,
                        "logs": logs,
                    }
                },
            )

        finally:
            await manager.clear_current_process()
            await manager.clear_current_job()

    @staticmethod
    async def _extract_metrics(logs: list[str]) -> dict[str, Any]:
        """Extract final metrics from training logs."""
        metrics = {}

        # Look for final accuracy/loss in logs (reverse search for latest)
        for line in reversed(logs):
            if "accuracy" in line.lower():
                match = re.search(r"accuracy[:\s]+([0-9.]+)", line, re.IGNORECASE)
                if match and "final_accuracy" not in metrics:
                    metrics["final_accuracy"] = float(match.group(1))

            if "loss" in line.lower():
                match = re.search(r"loss[:\s]+([0-9.]+)", line, re.IGNORECASE)
                if match and "final_loss" not in metrics:
                    metrics["final_loss"] = float(match.group(1))

        return metrics

    @staticmethod
    async def cancel_training() -> bool:
        """Cancel currently running training job."""
        db: AsyncIOMotorDatabase = Database.get_database()
        manager = TrainingJobManager.get_instance()

        job_id = await manager.get_current_job_id()
        if not job_id:
            return False

        # Terminate the subprocess (critical fix)
        process_terminated = await manager.terminate_current_process()

        # Update status to cancelled
        await db.training_logs.update_one(
            {"_id": job_id},
            {
                "$set": {
                    "status": "cancelled",
                    "completed_at": datetime.utcnow(),
                }
            },
        )

        # Broadcast cancellation
        message = "Training cancelled by user"
        if not process_terminated:
            message += " (no running subprocess found)"
        log = TrainingLog(type="log", message=message)
        await manager.broadcast_log(log)

        await manager.clear_current_process()
        await manager.clear_current_job()

        return True

    @staticmethod
    async def get_training_history(limit: int = 20) -> list[dict[str, Any]]:
        """Get past training jobs."""
        db: AsyncIOMotorDatabase = Database.get_database()

        jobs = (
            await db.training_logs.find(
                {"status": {"$in": ["completed", "error", "cancelled"]}}
            )
            .sort("started_at", -1)
            .limit(limit)
            .to_list(None)
        )

        return jobs or []
