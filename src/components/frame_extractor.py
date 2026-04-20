"""
This file extracts frames from raw videos and saves them as images.
It is based on your current project idea of converting videos into usable training data.
The extractor is simple, safe, and configurable for CPU-friendly processing.
This is useful before MMPose feature extraction or phone hand-crop generation.
"""

from pathlib import Path

import cv2

from src.utils.common import create_directories
from src.utils.helpers import list_video_files
from src.utils.logger import get_logger


class FrameExtractor:
    """
    Extract frames from videos at a fixed step interval.
    """

    def __init__(
        self,
        raw_video_dir: Path,
        output_dir: Path,
        log_dir: Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.raw_video_dir = raw_video_dir
        self.output_dir = output_dir
        self.logger = get_logger(
            self.__class__.__name__, log_dir=log_dir, level=log_level
        )

    def extract_frames_from_video(
        self,
        video_path: Path,
        step_size: int = 10,
        image_extension: str = ".jpg",
    ) -> int:
        """
        Extract every Nth frame from one video.
        """
        create_directories([self.output_dir])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        saved_count = 0
        frame_index = 0
        video_stem = video_path.stem
        video_output_dir = self.output_dir / video_stem
        create_directories([video_output_dir])

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1
            if frame_index % step_size != 0:
                continue

            output_name = f"{video_stem}_frame_{frame_index:06d}{image_extension}"
            output_path = video_output_dir / output_name
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        cap.release()
        self.logger.info("Extracted %d frames from %s", saved_count, video_path.name)
        return saved_count

    def run(self, step_size: int = 10) -> dict:
        """
        Extract frames from all videos inside the raw video folder.
        """
        videos = list_video_files(self.raw_video_dir)

        summary = {
            "videos_found": len(videos),
            "videos_processed": 0,
            "frames_saved": 0,
        }

        for video_path in videos:
            frames_saved = self.extract_frames_from_video(
                video_path=video_path, step_size=step_size
            )
            summary["videos_processed"] += 1
            summary["frames_saved"] += frames_saved

        self.logger.info("Frame extraction summary: %s", summary)
        return summary
