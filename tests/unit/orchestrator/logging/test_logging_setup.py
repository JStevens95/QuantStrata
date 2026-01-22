from __future__ import annotations

import logging

from src.orchestrator.logging.setup import build_run_logger


def test_build_run_logger_is_idempotent(tmp_path) -> None:
    log_file = tmp_path / "run.log"

    logger1 = build_run_logger("QuantStrata.Orchestrator.Test", level=logging.INFO, log_file=log_file)
    n_handlers_first = len(logger1.handlers)

    # Calling again should NOT add new handlers.
    logger2 = build_run_logger("QuantStrata.Orchestrator.Test", level=logging.INFO, log_file=log_file)
    n_handlers_second = len(logger2.handlers)

    assert logger1 is logger2
    assert n_handlers_first == n_handlers_second