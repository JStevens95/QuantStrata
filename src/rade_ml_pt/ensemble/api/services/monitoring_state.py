"""Monitoring run-state holder for the PRISM API.

Holds the live :class:`EnsembleMonitoringPipeline` between HTTP requests
so the staged workflow (``load → load_scenarios → validate_scenarios →
run → promote_to_predictions``) survives the stateless boundary between
calls.

Mirrors :mod:`services.inference_state` deliberately — same singleton
pattern, same status enum shape, same background-thread dispatch —
so the M.4 router can keep its mental model identical to the
inference router.

State model — Option A (single-user, single-process)
----------------------------------------------------
One active :class:`MonitoringRunState` at a time on the
:class:`MonitoringStateManager`.  Calling ``create_run`` while a prior
run exists replaces it.  Sufficient for the single-user dashboard;
Option B (multi-user, dict-keyed) is the same mechanical refactor
documented on :class:`InferenceStateManager`.

State machine
-------------
::

    created → loading → loaded → scenarios_loaded → validated
                                                       ↓
                                                    running
                                                       ↓
                                                    complete  ──▶ promoting
                                                                       ↓
                                                                   promoted
    (any) → failed

``/run`` is gated on ``status == validated``.  ``/promote`` is gated
on ``status == complete``.  Both endpoints dispatch onto a background
thread (Stage 9 of the inference router precedent) so the HTTP
request returns immediately and the UI polls ``/status`` until a
terminal state is reached.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.monitoring.run_paths import (
    MONITORING_RUNS_DIRNAME,
    MONITORING_SUBDIRNAME,
)
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    ActivityEntry,
    EventCollector,
)
from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Status enum (str-valued for transparent JSON encoding)
# ──────────────────────────────────────────────────────────────────────

STATUS_CREATED:    str = "created"
STATUS_LOADING:    str = "loading"
STATUS_LOADED:     str = "loaded"
STATUS_SCENARIOS:  str = "scenarios_loaded"
STATUS_VALIDATED:  str = "validated"
STATUS_RUNNING:    str = "running"
STATUS_COMPLETE:   str = "complete"
STATUS_PROMOTING:  str = "promoting"
STATUS_PROMOTED:   str = "promoted"
STATUS_FAILED:     str = "failed"

# Terminal states — used by the router to gate which transitions are
# valid from any given snapshot.  Promoted is terminal w.r.t. this run
# but a fresh run can always be created on top of it.
TERMINAL_STATES = frozenset({STATUS_COMPLETE, STATUS_PROMOTED, STATUS_FAILED})


# ──────────────────────────────────────────────────────────────────────
# Per-run state
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MonitoringRunState:
    """One monitoring run's mutable state.

    Constructed by :meth:`MonitoringStateManager.create_run`; thereafter
    only lifecycle fields (``status``, ``last_error``, ``artifacts_dir``,
    ``manifest_path``, ``predictions_dir``) are mutated.

    Attributes
    ----------
    run_id
        Stable identifier for this monitoring run.  The router's
        ``/load`` endpoint mints a provisional id immediately;
        ``pipeline.run()`` overwrites it with the canonical monitoring
        run id once the underlying :class:`EnsembleMonitoringPipeline`
        decides the on-disk path.  Why provisional → final: load()
        doesn't know the run id yet (that's resolved at run() time)
        but the router needs SOMETHING to return on the load response.
    ensemble_version
        Ensemble version this run was constructed against.
    pipeline
        Live :class:`EnsembleMonitoringPipeline` instance — the
        underlying inference pipeline is composed inside it.
    activity_log
        Thread-safe event buffer.  Captures both monitoring stage
        events and any inference stage events emitted by the
        composed pipeline (monitoring forwards on_event to inference).
    status
        Current lifecycle state — one of the ``STATUS_*`` constants.
    created_at
        UTC ISO-8601 timestamp at construction.
    last_error
        Detail message for the most recent failure.  ``None`` when
        no failure has occurred.
    artifacts_dir
        Resolved per-run monitoring artifacts directory.  Set when
        the drift run completes; equals
        ``<base>/monitoring_runs/<run_id>``.
    manifest_path
        Convenience pointer to the run's ``manifest.json``.  Populated
        when the drift run completes.
    predictions_dir
        Convenience pointer to the predictions sub-directory
        (``<manifest_path.parent>/inference/``).  Populated when
        promote-to-predictions completes.
    """

    run_id:           str
    ensemble_version: str
    pipeline:         EnsembleMonitoringPipeline
    activity_log:     EventCollector
    status:           str           = STATUS_CREATED
    created_at:       str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    last_error:       Optional[str] = None
    artifacts_dir:    Optional[str] = None
    manifest_path:    Optional[str] = None
    predictions_dir:  Optional[str] = None

    # Background-execution machinery — populated when ``/run`` or
    # ``/promote`` dispatches onto a worker thread.  Excluded from
    # ``repr`` so log lines stay readable.
    _thread:          Optional[threading.Thread] = field(default=None, repr=False)

    @property
    def is_alive(self) -> bool:
        """True iff a background run / promote thread is currently executing."""
        return self._thread is not None and self._thread.is_alive()

    def transition(self, new_status: str, *, error: Optional[str] = None) -> None:
        """Move the run to ``new_status`` and clear / set ``last_error``.

        Single mutator for the run lifecycle so the
        ``status ⇒ last_error`` invariant (set only when FAILED) is
        enforced in one place.
        """
        self.status = new_status
        self.last_error = error if new_status == STATUS_FAILED else None


# ──────────────────────────────────────────────────────────────────────
# State manager
# ──────────────────────────────────────────────────────────────────────

class MonitoringStateManager:
    """Single source of truth for the API's active monitoring run.

    Today (Option A) holds **one** :class:`MonitoringRunState` at a
    time.  Same Option-A → Option-B migration story as
    :class:`InferenceStateManager`; ``run_id`` is already plumbed
    through every accessor.
    """

    def __init__(self) -> None:
        self._active: Optional[MonitoringRunState] = None
        self._lock = threading.Lock()

    # ── Construction / lookup ─────────────────────────────────────

    def create_run(
        self,
        ensemble_config:  EnsembleConfig,
        ensemble_version: str,
    ) -> MonitoringRunState:
        """Construct a new monitoring pipeline + state object.

        Any previously-active run is silently replaced.  The
        ``run_id`` returned here is provisional — see the docstring
        on :attr:`MonitoringRunState.run_id` for the
        provisional-vs-final reasoning.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        provisional_run_id = f"{ensemble_version}__monitor__{ts}"

        # EventCollector captures monitoring stage events AND any
        # inference stage events emitted by the composed pipeline
        # (EnsembleMonitoringPipeline forwards on_event into its
        # inner EnsembleInferencePipeline).
        activity_log = EventCollector()

        pipeline = EnsembleMonitoringPipeline(
            ensemble_config  = ensemble_config,
            ensemble_version = ensemble_version,
            session          = None,
            on_event         = activity_log,
        )

        state = MonitoringRunState(
            run_id           = provisional_run_id,
            ensemble_version = ensemble_version,
            pipeline         = pipeline,
            activity_log     = activity_log,
        )

        with self._lock:
            self._active = state

        logger.info(
            "MonitoringStateManager: created run %s (provisional)",
            provisional_run_id,
        )
        return state

    def get_run(self, run_id: Optional[str] = None) -> MonitoringRunState:
        """Look up the active run by ID.

        When ``run_id`` is None the currently-active run is returned
        (v1 single-user convention).
        """
        with self._lock:
            state = self._active

        if state is None:
            raise RuntimeError(
                "No active monitoring run — call POST /monitoring/load first."
            )
        if run_id is not None and run_id != state.run_id:
            raise RuntimeError(
                f"Monitoring run '{run_id}' not found; active run is "
                f"'{state.run_id}'."
            )
        return state

    @property
    def has_active_run(self) -> bool:
        """True iff at least one run has been created."""
        with self._lock:
            return self._active is not None

    # ── Activity-log helpers (HTTP polling) ───────────────────────

    def events_since(
        self,
        run_id: Optional[str],
        cursor: int,
    ) -> Tuple[List[ActivityEntry], int]:
        """Return events emitted since ``cursor`` for one run.

        Same cursor protocol as
        :meth:`InferenceStateManager.events_since`.
        """
        state    = self.get_run(run_id)
        snapshot = state.activity_log.snapshot()
        if cursor < 0:
            cursor = 0
        if cursor > len(snapshot):
            cursor = len(snapshot)
        return snapshot[cursor:], len(snapshot)

    # ── Background-execution helpers ──────────────────────────────

    def start_run_in_background(
        self,
        target: Callable[[], None],
    ) -> threading.Thread:
        """Spawn the background run / promote thread for the active run.

        Single-thread-per-run by construction.  The ``target`` closure
        is responsible for the terminal status transition (COMPLETE /
        PROMOTED / FAILED).  Daemon mode is on so SIGTERM doesn't
        hang on a long monitoring run.

        Raises
        ------
        RuntimeError
            If a background thread is already running — prevents
            double-dispatch from a refreshed browser tab.
        """
        state = self.get_run()
        if state.is_alive:
            raise RuntimeError(
                f"Run {state.run_id} already has a background thread running."
            )

        thread = threading.Thread(
            target = target,
            name   = f"monitor-{state.run_id}",
            daemon = True,
        )
        state._thread = thread        # noqa: SLF001 — module-internal
        thread.start()
        logger.info(
            "MonitoringStateManager: started background thread for run %s",
            state.run_id,
        )
        return thread


# ──────────────────────────────────────────────────────────────────────
# Helpers used by router and dependency
# ──────────────────────────────────────────────────────────────────────

def build_monitoring_ensemble_config(
    registry_dir:    str,
    artifacts_dir:   str,
    new_scenario_dir: Optional[str] = None,
) -> EnsembleConfig:
    """Construct a minimal :class:`EnsembleConfig` for an API-driven monitoring run.

    Symmetric with :func:`build_ensemble_config` in
    :mod:`services.inference_state`.  Carries ``input_mode =
    "new_scenarios"`` because monitoring only supports the
    new-scenarios input mode today (new-trades is deferred).
    """
    metadata = {"inference": {"input_mode": "new_scenarios"}}
    if new_scenario_dir is not None:
        metadata["inference"]["new_scenario_dir"] = new_scenario_dir

    return EnsembleConfig(
        registry_dir  = registry_dir,
        artifacts_dir = artifacts_dir,
        metadata      = metadata,
    )


def per_run_monitoring_artifacts_dir(base_artifacts_dir: str, run_id: str) -> str:
    """Convention for where per-run monitoring artifacts go on disk.

    ``<base_artifacts_dir>/monitoring_runs/<run_id>``.  Mirrors
    inference's :func:`per_run_artifacts_dir` exactly — same
    "one subdir per run, named for the family" layout.

    Centralised here so the router, the state manager, and the
    monitoring reader agree on the layout without cross-coupling.
    """
    return str(Path(base_artifacts_dir) / MONITORING_RUNS_DIRNAME / run_id)


def manifest_path_for_run(base_artifacts_dir: str, run_id: str) -> Path:
    """Resolve the canonical manifest.json path for a monitoring run.

    ``<base>/monitoring_runs/<run_id>/monitoring/manifest.json``.
    Same constants the writer uses
    (:mod:`monitoring.run_paths`).
    """
    return (
        Path(base_artifacts_dir)
        / MONITORING_RUNS_DIRNAME
        / run_id
        / MONITORING_SUBDIRNAME
        / "manifest.json"
    )


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton
# ──────────────────────────────────────────────────────────────────────

_manager: Optional[MonitoringStateManager] = None


def set_monitoring_state_manager(manager: MonitoringStateManager) -> None:
    """Inject the manager at app-lifespan startup."""
    global _manager
    _manager = manager


def get_monitoring_state_manager() -> MonitoringStateManager:
    """FastAPI ``Depends`` — returns the singleton manager."""
    if _manager is None:
        raise RuntimeError(
            "MonitoringStateManager not initialised. Server not ready."
        )
    return _manager


__all__ = [
    # ── Status constants ─────────────────────────────────────────
    "STATUS_CREATED", "STATUS_LOADING", "STATUS_LOADED",
    "STATUS_SCENARIOS", "STATUS_VALIDATED", "STATUS_RUNNING",
    "STATUS_COMPLETE", "STATUS_PROMOTING", "STATUS_PROMOTED",
    "STATUS_FAILED", "TERMINAL_STATES",
    # ── State machinery ─────────────────────────────────────────
    "MonitoringRunState",
    "MonitoringStateManager",
    "set_monitoring_state_manager",
    "get_monitoring_state_manager",
    # ── Helpers ─────────────────────────────────────────────────
    "build_monitoring_ensemble_config",
    "per_run_monitoring_artifacts_dir",
    "manifest_path_for_run",
]
