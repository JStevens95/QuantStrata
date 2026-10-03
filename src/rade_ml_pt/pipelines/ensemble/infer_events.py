"""Lifecycle events emitted by :class:`EnsembleInferencePipeline`.

The pipeline runs in two contexts:

* **Headless** (CLI / scripts / tests) — no UI; events are silenced.
* **UI** (Rade Analytics *Inference Console*) — every event is pushed
  into the page's :data:`activity_log_store` so the analyst sees a
  live narration of *file loaded → inputs ready → forward pass →
  predictions ready*.

Both call paths share the **same** pipeline.  They differ only in
whether they pass an ``on_event`` callback when constructing the
pipeline.

Public surface
--------------

* :class:`ActivityEntry` — the single dict shape every event takes.
  Mirrors the shape consumed by
  :func:`src.ui.apps.rade_analytics.layouts.inference.render_activity_entries`
  exactly; UI callbacks pass these through to the Store with no
  re-shaping.

* :data:`EmitFn` — function type the pipeline accepts.  Anything
  that takes one ``ActivityEntry`` and returns ``None`` qualifies
  (``list.append`` works as-is, which is how
  :class:`EventCollector` is implemented).

* :func:`event` — factory that fills in the ``id`` (``uuid4``) and
  ``ts`` (UTC ISO-8601, second precision) so call-sites only have
  to specify *what happened*, not *when*.

* :data:`noop_emit` — singleton no-op used as the default so the
  pipeline body always has a callable to invoke.

* :class:`EventCollector` — convenience wrapper that buffers events
  in an in-memory list, with a thread-safe ``append`` and a typed
  ``snapshot()`` accessor for tests / Stage-2 backend wrappers.

Stage / status vocabulary
-------------------------

The UI maps these to icons in
``src/ui/apps/rade_analytics/layouts/inference.py::_STATUS_ICON``.
Keep them in lock-step:

* **stage** — ``"ingest"`` · ``"validate"`` · ``"inference"``
* **status** — ``"ok"`` · ``"fail"`` · ``"running"`` · ``"pending"``
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional


# ─────────────────────────────────────────────────────────────────────
# Vocabularies — kept as module-level constants so the UI layout, the
# pipeline emit sites and the tests can all import the *same* tokens.
# Drift here is what would silently break the activity log render.
# ─────────────────────────────────────────────────────────────────────

Stage  = Literal["ingest", "validate", "inference"]
Status = Literal["ok", "fail", "running", "pending"]

STAGE_INGEST    : Stage = "ingest"
STAGE_VALIDATE  : Stage = "validate"
STAGE_INFERENCE : Stage = "inference"

STATUS_OK      : Status = "ok"
STATUS_FAIL    : Status = "fail"
STATUS_RUNNING : Status = "running"
STATUS_PENDING : Status = "pending"


# ─────────────────────────────────────────────────────────────────────
# Event shape
# ─────────────────────────────────────────────────────────────────────


# A plain dict is the wire shape (so it round-trips through Dash
# Stores via JSON without any custom (de)serialisation).  We expose
# the schema both as a TypedDict-like alias for documentation and as
# a Pydantic-free dataclass for headless callers that prefer typed
# attribute access.
ActivityEntry = Dict[str, Any]
"""Wire shape.  Keys: ``id``, ``stage``, ``phase``, ``status``,
``ts`` and *optionally* ``target`` and ``detail``.  Identical to the
Dash Store payload consumed by ``render_activity_entries``."""


@dataclass(frozen=True)
class TypedActivityEntry:
    """Type-friendly mirror of :data:`ActivityEntry`.

    Pure convenience for headless callers / tests that prefer
    attribute access; ``to_dict()`` round-trips it back to the wire
    shape with no information loss.  The pipeline only ever emits
    plain dicts.
    """

    stage:   Stage
    phase:   str
    status:  Status
    target:  Optional[str] = None
    detail:  Optional[str] = None
    id:      str           = field(default_factory=lambda: uuid.uuid4().hex)
    ts:      str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    def to_dict(self) -> ActivityEntry:
        d: ActivityEntry = {
            "id":     self.id,
            "stage":  self.stage,
            "phase":  self.phase,
            "status": self.status,
            "ts":     self.ts,
        }
        if self.target is not None:
            d["target"] = self.target
        if self.detail is not None:
            d["detail"] = self.detail
        return d


# ─────────────────────────────────────────────────────────────────────
# Emit callback contract
# ─────────────────────────────────────────────────────────────────────


EmitFn = Callable[[ActivityEntry], None]
"""Type accepted by :class:`EnsembleInferencePipeline`.

Implementations must be **non-blocking** and **side-effect-only**;
the pipeline calls them inline in the hot loop and never inspects
return values.  ``list.append`` qualifies.  An async queue ``put``
qualifies.  A thread-safe ``EventCollector`` qualifies.
"""


def noop_emit(_entry: ActivityEntry) -> None:
    """Default emitter — drops every event on the floor.

    Used when no UI is present (tests, CLI, batch jobs).  Keeps the
    pipeline body free of ``if on_event is not None`` checks.
    """
    return None


def event(
    stage:   Stage,
    phase:   str,
    *,
    status:  Status              = STATUS_OK,
    target:  Optional[str]       = None,
    detail:  Optional[str]       = None,
) -> ActivityEntry:
    """Construct an :data:`ActivityEntry` with auto-filled ``id``/``ts``.

    Parameters
    ----------
    stage
        One of :data:`STAGE_INGEST`, :data:`STAGE_VALIDATE`,
        :data:`STAGE_INFERENCE`.
    phase
        Short, human-readable phase label.  Becomes the bold middle
        column in the UI feed (see ``rade-activity-phase``).
    status
        Defaults to :data:`STATUS_OK`.  Use :data:`STATUS_RUNNING`
        for *started* events, :data:`STATUS_FAIL` for failures.
    target
        Optional second column — usually a filename, a cluster id,
        or a risk-factor name.  Renders as a small monospace pill.
    detail
        Optional sub-text for failures — e.g. a stripped traceback
        or a validation error.

    Returns
    -------
    ActivityEntry
        Wire-shape dict ready for :data:`EmitFn` consumption.
    """
    return TypedActivityEntry(
        stage=stage,
        phase=phase,
        status=status,
        target=target,
        detail=detail,
    ).to_dict()


# ─────────────────────────────────────────────────────────────────────
# Test / backend helper — buffer events in memory
# ─────────────────────────────────────────────────────────────────────


class EventCollector:
    """Thread-safe in-memory buffer that doubles as an :data:`EmitFn`.

    Two usage patterns:

    * **Tests** — instantiate, pass the instance directly as
      ``on_event`` (it's callable), then ``snapshot()`` after the
      pipeline returns to assert event order / counts.

    * **Backend wrapper** — same construction, but the wrapper
      passes the ``snapshot()`` payload back to the UI as the
      activity-log slice for the run.

    Thread-safety
    -------------

    The ensemble session loads inference state in a
    ``ThreadPoolExecutor`` (``EnsembleSession._load_parallel``), so
    multiple threads can call ``__call__`` concurrently.  We guard
    the underlying list with a single mutex; appends are O(1) and
    contention is negligible at the event volumes the UI will see
    (<200 / run).
    """

    def __init__(self) -> None:
        self._events: List[ActivityEntry] = []
        self._lock = threading.Lock()

    def __call__(self, entry: ActivityEntry) -> None:
        with self._lock:
            self._events.append(entry)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def snapshot(self) -> List[ActivityEntry]:
        """Return a *copy* of the buffered events in emit order.

        Always returns a fresh list — callers can mutate / serialise
        without racing with concurrent ``__call__`` writes.
        """
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """Drop everything buffered so far (testing helper)."""
        with self._lock:
            self._events.clear()


__all__ = [
    "ActivityEntry",
    "EmitFn",
    "EventCollector",
    "Stage",
    "Status",
    "STAGE_INFERENCE",
    "STAGE_INGEST",
    "STAGE_VALIDATE",
    "STATUS_FAIL",
    "STATUS_OK",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "TypedActivityEntry",
    "event",
    "noop_emit",
]
