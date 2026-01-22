"""
Logging configuration for orchestrator runs.

We provide a logger that:
- logs to console
- optionally logs to a file inside the run logs directory
- avoids duplicate handlers when called multiple times
"""

from __future__ import annotations

import logging
from pathlib import Path


def build_run_logger(
    logger_name: str,
    *,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Build a standard run logger.

    Parameters
    ----------
    logger_name:
        Logger name (stable, human-readable).
    level:
        Logging level.
    log_file:
        Optional file path for file logging.

    Returns
    -------
    logging.Logger
        Configured logger with handlers attached exactly once.
    """
    logger = logging.getLogger(str(logger_name))
    logger.setLevel(level)

    # Prevent propagation to root logger (avoids duplicate prints).
    logger.propagate = False

    # If handlers exist, assume logger already configured.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler.
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional).
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger