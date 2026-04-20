import logging
import sys
from pathlib import Path

from src.config.constants import DEFAULT_LOG_FILE_NAME


def get_logger(
    name: str, log_dir: Path | None = None, level: str = "INFO"
) -> logging.Logger:
    """
    Create and return a reusable logger.

    Why this helper exists:
    - keeps logs consistent
    - writes logs to console
    - optionally writes logs to file
    - avoids duplicate handlers
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / DEFAULT_LOG_FILE_NAME, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
