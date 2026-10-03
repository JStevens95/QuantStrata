"""
Structured logging with contextual run identifiers.

A forty-job run produces forty interleaved streams of log output. Without
identifiers attached to every record, that output is unreadable, and the usual
remedy -- threading a pre-configured logger through every function -- puts an
infrastructure parameter into the signature of code that has nothing to do
with logging.

Instead the identifiers live in :class:`~contextvars.ContextVar` slots. Any
module can call :func:`get_logger` and its records are automatically stamped
with the current run, job and stage.

Crossing a process boundary
---------------------------
Context variables do not survive ``fork`` or ``spawn``. A worker must re-bind
them, which is what :func:`context_payload` and :func:`apply_context_payload`
are for: the parent serialises the context into the job payload, and the
worker applies it before doing any work.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, Final

__all__ = [
    "ContextFilter",
    "apply_context_payload",
    "bound_context",
    "configure_logging",
    "context_payload",
    "current_context",
    "get_logger",
]

#: Identifier of the current run. ``None`` outside a run.
RUN_ID: ContextVar[str | None] = ContextVar("rade_xl_run_id", default=None)

#: Identifier of the current job within a job set. ``None`` for a single run.
JOB_ID: ContextVar[str | None] = ContextVar("rade_xl_job_id", default=None)

#: Name of the pipeline stage currently executing. ``None`` between stages.
STAGE: ContextVar[str | None] = ContextVar("rade_xl_stage", default=None)

_CONTEXT_VARIABLES: Final = {"run_id": RUN_ID, "job_id": JOB_ID, "stage": STAGE}

#: Root logger name. Every framework logger is a descendant, so a user can
#: raise or lower the framework's verbosity with a single call without
#: touching their own loggers.
ROOT_LOGGER_NAME: Final = "rade_xl"

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-7s [%(rade_xl_context)s] %(name)s: %(message)s"
_TIME_FORMAT: Final = "%Y-%m-%d %H:%M:%S"


class ContextFilter(logging.Filter):
    """
    Attach the current run, job and stage identifiers to every record.

    Implemented as a filter rather than an adapter so that records emitted by
    code which knows nothing about this module -- including a third-party
    library logging under the framework's logger tree -- are stamped too.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add a ``rade_xl_context`` attribute to the record.

        Parameters
        ----------
        record
            The record being emitted. Mutated in place.

        Returns
        -------
        bool
            Always ``True``: this filter annotates, it never discards.
        """
        identifiers = current_context()
        # A compact single field rather than three, so the format string stays
        # readable and a line with no context does not carry empty brackets.
        record.rade_xl_context = (
            " ".join(f"{key}={value}" for key, value in identifiers.items()) or "-"
        )
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Return the framework logger for a module.

    Parameters
    ----------
    name
        Usually ``__name__``. A leading ``rade_xl`` is not duplicated, and any
        other name is placed under the framework's logger tree so that
        configuring one logger configures all of them.

    Returns
    -------
    logging.Logger
        A logger beneath :data:`ROOT_LOGGER_NAME`.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    # Strip the repository's import prefix so `src.rade_xl.core.spec.run` and
    # `rade_xl.core.spec.run` produce the same logger name. The package is
    # importable under both paths, and a log stream should not reveal which.
    trimmed = name.removeprefix("src.")
    if trimmed.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(trimmed)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{trimmed}")


def configure_logging(
    *,
    level: int = logging.INFO,
    stream: IO[str] | None = None,
    force: bool = False,
) -> logging.Logger:
    """
    Install a handler on the framework's root logger.

    Called explicitly, never at import. Importing ``rade_xl`` must not change
    how an application's logging behaves -- a library that configures logging
    on import is a library that silently redirects somebody else's output.

    Parameters
    ----------
    level
        Threshold for the framework's logger.
    stream
        Destination. Defaults to standard error, so log output does not
        contaminate a program's results on standard output.
    force
        Replace any handler this function previously installed. Without it the
        call is idempotent, so repeated calls cannot produce duplicated lines.

    Returns
    -------
    logging.Logger
        The configured framework root logger.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    # Do not propagate to the root logger: an application that has configured
    # its own root handler would otherwise see every framework line twice.
    logger.propagate = False

    existing = [handler for handler in logger.handlers if getattr(handler, "_rade_xl", False)]
    if existing and not force:
        return logger
    for handler in existing:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_TIME_FORMAT))
    handler.addFilter(ContextFilter())
    # Marked so a later call can recognise its own handler and leave any
    # handler the application added alone.
    handler._rade_xl = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def current_context() -> dict[str, str]:
    """
    Return the identifiers currently bound, omitting those that are unset.

    Returns
    -------
    dict of str to str
        Mapping with any of ``run_id``, ``job_id`` and ``stage`` that are set.
    """
    return {
        name: value
        for name, variable in _CONTEXT_VARIABLES.items()
        if (value := variable.get()) is not None
    }


@contextmanager
def bound_context(
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    stage: str | None = None,
) -> Iterator[None]:
    """
    Bind identifiers for the duration of a block.

    Only the arguments given are changed; the rest keep their current values.
    That is what lets a stage bind ``stage`` without needing to know, or
    repeat, the run it belongs to.

    Every variable is restored on exit, including when the block raises, so a
    failed stage cannot leave its name attached to subsequent log lines.

    Parameters
    ----------
    run_id, job_id, stage
        Values to bind. ``None`` leaves the existing value in place.

    Yields
    ------
    None
    """
    updates = {"run_id": run_id, "job_id": job_id, "stage": stage}
    tokens = [
        _CONTEXT_VARIABLES[name].set(value) for name, value in updates.items() if value is not None
    ]
    try:
        yield
    finally:
        # Reset in reverse order so nested bindings unwind correctly.
        for token in reversed(tokens):
            token.var.reset(token)


def context_payload() -> dict[str, str]:
    """
    Serialise the current context for transport to a worker process.

    Returns
    -------
    dict of str to str
        A plain, picklable mapping suitable for inclusion in a job payload.
    """
    return current_context()


def apply_context_payload(payload: Mapping[str, str]) -> None:
    """
    Re-bind identifiers in a worker process.

    Unlike :func:`bound_context` this does not restore anything, because a
    worker's context should last for the whole of its task.

    This **replaces** the context rather than merging into it: an identifier
    absent from the payload is cleared. That matters because a process pool
    reuses its workers. If applying a payload merged, a worker that handled
    job ``EURUSD`` and then a run with no job identifier would keep logging
    ``EURUSD`` -- misattributing work to a job that had already finished,
    which is precisely the failure this mechanism exists to prevent.

    Unknown keys are ignored rather than raising: a newer parent sending an
    identifier an older worker does not recognise should degrade to slightly
    less informative logs, not to a crash.

    Parameters
    ----------
    payload
        A mapping produced by :func:`context_payload`. Pass an empty mapping
        to clear the context entirely.
    """
    for name, variable in _CONTEXT_VARIABLES.items():
        variable.set(payload.get(name))
