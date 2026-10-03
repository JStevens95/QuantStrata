"""
Experiment tracking, behind one interface, with a no-op default.

The governing rule: **tracking is never load-bearing.** A model that trained
successfully but could not reach a tracking server has still trained
successfully, and the bundle on disk is the system of record -- not the
tracker. So every implementation here swallows its own failures and warns,
and the default implementation does nothing at all.

That default matters more than it sounds. If the framework's baseline
behaviour required a tracking backend, every test, every notebook and every
quick experiment would need one configured. Making
:class:`NullTracker` the default means tracking is something you add, not
something you disable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ..core.runtime.hashing import canonical_json
from ..core.runtime.logging import get_logger

__all__ = ["JsonlTracker", "NullTracker"]

_LOGGER = get_logger(__name__)


class NullTracker:
    """
    A tracker that records nothing.

    Satisfies :class:`~rade_xl.core.runtime.context.Tracker`. The default, so
    that a run needs no tracking infrastructure to proceed.

    Every method is empty rather than raising ``NotImplementedError``: this is
    a working implementation of "do not track", not an incomplete one.
    """

    def log_params(self, params: Mapping[str, object]) -> None:
        """
        Discard the run's configuration.

        Parameters
        ----------
        params
            Flattened specification values.
        """

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """
        Discard metrics.

        Parameters
        ----------
        metrics
            Metric name to value.
        step
            Epoch or boosting round, or ``None``.
        """

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """
        Discard an artifact reference.

        Parameters
        ----------
        path
            The file that was written.
        name
            Logical name.
        """

    def finish(self, *, succeeded: bool) -> None:
        """
        Do nothing on completion.

        Parameters
        ----------
        succeeded
            Whether the run completed.
        """


class JsonlTracker:
    """
    A tracker that appends events to a local JSON Lines file.

    Satisfies :class:`~rade_xl.core.runtime.context.Tracker`. Useful when a
    hosted tracker is unavailable or inappropriate, and as a reference for
    what an implementation has to do.

    Parameters
    ----------
    path
        File to append to. Its parent directory is created if absent.

    Notes
    -----
    One line per event, so a reader can follow a run in progress with ``tail
    -f`` and a crashed run still leaves everything written before the crash.
    Writes are appends with no locking, which is safe for the one-process-per
    -run case this is intended for; a job set should give each job its own
    file, which is what a per-job
    :class:`~rade_xl.core.runtime.context.RunContext` naturally produces.
    """

    def __init__(self, path: Path) -> None:
        """
        Prepare the event file.

        Parameters
        ----------
        path
            File to append events to.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_params(self, params: Mapping[str, object]) -> None:
        """
        Append the run's configuration.

        Parameters
        ----------
        params
            Flattened specification values.
        """
        self._write({"event": "params", "params": json.loads(canonical_json(dict(params)))})

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """
        Append metrics.

        Parameters
        ----------
        metrics
            Metric name to value.
        step
            Epoch or boosting round, or ``None`` for a final metric.
        """
        self._write({"event": "metrics", "step": step, "metrics": dict(metrics)})

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """
        Append an artifact reference.

        Parameters
        ----------
        path
            The file that was written.
        name
            Logical name, defaulting to the file name.
        """
        self._write({"event": "artifact", "name": name or path.name, "path": str(path)})

    def finish(self, *, succeeded: bool) -> None:
        """
        Append the terminal event.

        Parameters
        ----------
        succeeded
            Whether the run completed.
        """
        self._write({"event": "finish", "succeeded": succeeded})

    def _write(self, payload: Mapping[str, object]) -> None:
        """
        Append one event, swallowing I/O failures.

        Parameters
        ----------
        payload
            A JSON-encodable event.
        """
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
        except OSError:
            # Caught here as well as in RunContext, because this class may be
            # used directly. See the module docstring: tracking never fails a
            # run, and that has to hold wherever the tracker is called from.
            _LOGGER.warning("could not append a tracking event to %s", self.path, exc_info=True)
