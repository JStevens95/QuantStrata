# src/orchestrator/logging/events.py

from __future__ import annotations

import logging
from typing import Optional, Tuple


def log_step_start(
    logger: Optional[logging.Logger],
    *,
    step_name: str,
    tags: Tuple[str, ...] = (),
) -> None:
    if logger is None:
        return
    logger.info("STEP_START | %s | tags=%s", step_name, list(tags))


def log_step_end(
    logger: Optional[logging.Logger],
    *,
    step_name: str,
    ok: bool,
    tags: Tuple[str, ...] = (),
    elapsed_s: Optional[float] = None,
) -> None:
    if logger is None:
        return

    if elapsed_s is None:
        logger.info("STEP_END | %s | ok=%s | tags=%s", step_name, ok, list(tags))
    else:
        logger.info("STEP_END | %s | ok=%s | elapsed_s=%.6f | tags=%s", step_name, ok, float(elapsed_s), list(tags))


def log_step_error(
    logger: Optional[logging.Logger],
    *,
    step_name: str,
    exc: BaseException,
    tags: Tuple[str, ...] = (),
    elapsed_s: Optional[float] = None,
) -> None:
    if logger is None:
        return

    if elapsed_s is None:
        logger.exception("STEP_ERROR | %s | tags=%s | exc=%s", step_name, list(tags), type(exc).__name__)
    else:
        logger.exception(
            "STEP_ERROR | %s | elapsed_s=%.6f | tags=%s | exc=%s",
            step_name,
            float(elapsed_s),
            list(tags),
            type(exc).__name__,
        )