"""
Structured logging events.

We keep these helpers small so logs are consistent across pipelines.
"""

from __future__ import annotations

import logging
from typing import Tuple


def log_step_start(logger: logging.Logger, *, step_name: str, tags: Tuple[str, ...] = ()) -> None:
    """Log a standard step-start message."""
    logger.info("STEP_START | %s | tags=%s", step_name, list(tags))


def log_step_end(logger: logging.Logger, *, step_name: str, ok: bool) -> None:
    """Log a standard step-end message."""
    logger.info("STEP_END   | %s | ok=%s", step_name, bool(ok))