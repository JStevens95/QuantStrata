"""Inference run-state holder for the PRISM API.

Holds the live :class:`EnsembleInferencePipeline` between HTTP requests
so the staged workflow (``load → load_scenarios → validate →
run_inference``) survives the stateless boundary between calls.

State model — Option A (single-user, single-process)
----------------------------------------------------
The API today holds **one** :class:`InferenceRunState` at a time on
the :class:`InferenceStateManager`.  Calling ``create_run`` while a
prior run exists replaces it (the prior pipeline and its activity log
are dropped).  Sufficient for the single-user dashboard use case.

Migrating to multi-user (Option B) is a small change
----------------------------------------------------
All public methods of :class:`InferenceStateManager` already take a
``run_id`` argument; today there is only one and the manager validates
that callers pass it through correctly.  Moving to per-user / per-run
isolation is then a localised refactor:

  * Replace the single ``_active`` slot with
    ``_runs: Dict[str, InferenceRunState]``.
  * Drop the "single active run" assertion in
    :meth:`InferenceStateManager.create_run`.
  * Surface ``run_id`` in the API by upgrading each router method to
    take it from the URL / a header instead of the singleton.

The router and dependency call sites do **not** need to change.

Threading
---------
Stage 9 (background-task execution for ``/run``) will dispatch
:meth:`EnsembleInferencePipeline.run_inference` on a worker thread.
:class:`EventCollector` is already thread-safe so the activity-log
contract holds; the rest of the state object is read-only after
``create_run`` returns, save for ``status`` / ``last_error`` /
``artifacts_dir`` which are written exactly once each.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.pipelines.ensemble.infer import EnsembleInferencePipeline
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    ActivityEntry,
    EventCollector,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Status enum (str-valued so it round-trips through JSON cleanly)
# ──────────────────────────────────────────────────────────────────────

# Lifecycle of one run.  String-typed for transparent JSON encoding;
# kept as module-level constants rather than an Enum because the wire
# shape is just ``str`` and the router doesn't need branching logic on
# the type.
STATUS_CREATED:     str = "created"
STATUS_LOADING:     str = "loading"
STATUS_LOADED:      str = "loaded"
STATUS_SCENARIOS:   str = "scenarios_loaded"
STATUS_VALIDATED:   str = "validated"
STATUS_RUNNING:     str = "running"
STATUS_COMPLETE:    str = "complete"
STATUS_FAILED:      str = "failed"


# ──────────────────────────────────────────────────────────────────────
# Per-run state
# ──────────────────────────────────────────────────────────────────────

@dataclass
class InferenceRunState:
    """One inference run's mutable state.

    Built once by :meth:`InferenceStateManager.create_run`; thereafter
    only the lifecycle fields (``status``, ``last_error``,
    ``artifacts_dir``, ``manifest_path``) are mutated.

    Attributes
    ----------
    run_id
        Stable identifier for this run.  Today derived from
        ``<ensemble_version>__<UTC_yyyymmdd_HHMMSS>``; not a UUID so
        run IDs are human-greppable in logs and on disk.
    ensemble_version
        Ensemble version this run was constructed against.
    pipeline
        Live pipeline instance.  All staged operations (load,
        load_scenarios, validate_scenarios, run_inference) are
        invoked through this object.
    activity_log
        Thread-safe buffer of events emitted by the pipeline.  Passed
        in as ``on_event`` at pipeline construction; readable via
        ``activity_log.snapshot()`` for HTTP polling.
    status
        Current lifecycle state — one of the ``STATUS_*`` constants.
    created_at
        UTC ISO-8601 timestamp at construction.
    last_error
        Detail message for the most recent failure (set whenever
        ``status`` transitions to ``STATUS_FAILED``).  ``None`` when
        no failure has occurred.
    artifacts_dir
        Resolved per-run artifacts directory.  Set when the run
        actually executes (``run_inference``) so the API can locate
        manifest / parquet outputs afterwards.
    manifest_path
        Convenience pointer to ``<artifacts_dir>/inference/manifest.json``
        populated once the run completes successfully.  ``None`` until
        then.
    """

    run_id:           str
    ensemble_version: str
    pipeline:         EnsembleInferencePipeline
    activity_log:     EventCollector
    status:           str           = STATUS_CREATED
    created_at:       str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    last_error:       Optional[str] = None
    artifacts_dir:    Optional[str] = None
    manifest_path:    Optional[str] = None

    # Background-execution machinery (Stage 9).  Populated when the
    # ``/run`` router dispatches ``pipeline.run_inference()`` onto a
    # worker thread.  Excluded from ``repr`` so log lines stay readable.
    _thread:          Optional[threading.Thread] = field(default=None, repr=False)

    @property
    def is_alive(self) -> bool:
        """True iff a background run thread is currently executing.

        Distinct from ``status == STATUS_RUNNING`` because the worker
        transitions the run to COMPLETE / FAILED *inside* the thread —
        there's a small window where ``status`` is already terminal but
        ``Thread.is_alive()`` is still True until the worker unwinds.
        ``is_alive`` is the authoritative concurrency probe; ``status``
        is the user-visible state.
        """
        return self._thread is not None and self._thread.is_alive()

    def transition(self, new_status: str, *, error: Optional[str] = None) -> None:
        """Move the run to ``new_status`` and clear / set ``last_error``.

        Single mutator for the run lifecycle so the status / error
        invariant (``last_error`` is only set when status ⇒ FAILED) is
        enforced in one place.

        Parameters
        ----------
        new_status
            Target lifecycle state (one of the ``STATUS_*`` constants).
        error
            Detail message — only respected when transitioning to
            :data:`STATUS_FAILED`.  Pass ``None`` to clear.
        """
        self.status = new_status
        self.last_error = error if new_status == STATUS_FAILED else None


# ──────────────────────────────────────────────────────────────────────
# State manager
# ──────────────────────────────────────────────────────────────────────

class InferenceStateManager:
    """Single source of truth for the API's active inference run.

    Today (Option A) holds **one** :class:`InferenceRunState` at a time.
    All accessors take ``run_id`` explicitly so the future Option B
    refactor (per-run dict) is mechanical.
    """

    def __init__(self) -> None:
        # The single active run, or None when the server hasn't been
        # asked to construct one yet.  Switching to a dict for Option
        # B replaces only this slot.
        self._active: Optional[InferenceRunState] = None
        self._lock = threading.Lock()

    # ── Construction / lookup ─────────────────────────────────────

    def create_run(
        self,
        ensemble_config:  EnsembleConfig,
        ensemble_version: str,
    ) -> InferenceRunState:
        """Construct a new pipeline + state object and make it the active run.

        Any previously-active run is silently replaced — the prior
        pipeline and its activity log are dropped.  Sufficient for the
        single-user dashboard.  Option B replaces this with a dict
        insertion keyed on ``run_id``.

        Parameters
        ----------
        ensemble_config
            Already-populated :class:`EnsembleConfig`.  Constructed by
            the router from the API settings (``registry_dir``,
            ``artifacts_dir``).
        ensemble_version
            Ensemble version or tag (passed through to the pipeline's
            ``ensemble_version`` constructor argument).

        Returns
        -------
        InferenceRunState
            The freshly registered run.  Use ``state.run_id`` for any
            subsequent operation against this run.
        """
        ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{ensemble_version}__{ts}"

        # Construct the activity-log collector first; the pipeline
        # captures the callable in __init__ and emits into it for
        # every subsequent stage.
        activity_log = EventCollector()

        pipeline = EnsembleInferencePipeline(
            ensemble_config  = ensemble_config,
            ensemble_version = ensemble_version,
            session          = None,           # cold-load path for v1
            on_event         = activity_log,
        )

        state = InferenceRunState(
            run_id           = run_id,
            ensemble_version = ensemble_version,
            pipeline         = pipeline,
            activity_log     = activity_log,
        )

        with self._lock:
            self._active = state

        logger.info("InferenceStateManager: created run %s", run_id)
        return state

    def get_run(self, run_id: Optional[str] = None) -> InferenceRunState:
        """Look up a run by ID.

        When ``run_id`` is None (the v1 single-user convention) the
        currently-active run is returned.  Once Option B lands this
        argument becomes mandatory; callers that pass ``None`` will
        need to be updated then.

        Raises
        ------
        RuntimeError
            If no run has been created yet, or the requested
            ``run_id`` doesn't match the active run.
        """
        with self._lock:
            state = self._active

        if state is None:
            raise RuntimeError(
                "No active inference run — call POST /inference/load first."
            )
        if run_id is not None and run_id != state.run_id:
            raise RuntimeError(
                f"Inference run '{run_id}' not found; active run is "
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
        """Return events emitted since the given cursor for one run.

        Used by ``GET /inference/events?cursor=N`` polling — the
        cursor is an opaque integer offset into the run's activity
        log.  The next cursor (= current log length) is returned
        alongside the slice so callers can chain polls.

        Parameters
        ----------
        run_id
            The run to read events from.  ``None`` ⇒ active run.
        cursor
            Number of events the caller has already seen.  Clamped to
            ``[0, len(log)]``.

        Returns
        -------
        events : list of ActivityEntry
            Events strictly after ``cursor`` in emit order.
        next_cursor : int
            Total events emitted so far.  Pass back on the next poll.
        """
        state    = self.get_run(run_id)
        snapshot = state.activity_log.snapshot()
        if cursor < 0:
            cursor = 0
        if cursor > len(snapshot):
            cursor = len(snapshot)
        return snapshot[cursor:], len(snapshot)

    # ── Background-execution helpers (Stage 9) ────────────────────

    def start_run_in_background(
        self,
        target: Callable[[], None],
    ) -> threading.Thread:
        """Spawn the background run thread for the active run.

        Single-thread-per-run by construction (Option A).  The caller
        has already transitioned ``state.status`` to ``STATUS_RUNNING``
        before invoking this; the ``target`` closure is responsible
        for the terminal transition (``STATUS_COMPLETE`` /
        ``STATUS_FAILED``) once the pipeline returns or raises.

        Daemon mode is on so a ``SIGTERM`` doesn't hang the server
        waiting for a long-running inference to finish — partial runs
        are recoverable from the on-disk manifest by future filesystem
        rehydration logic (Stage 9.5).

        Raises
        ------
        RuntimeError
            If a background thread is already running for this run —
            prevents accidental double-dispatch from a refreshed UI.
        """
        state = self.get_run()
        if state.is_alive:
            raise RuntimeError(
                f"Run {state.run_id} already has a background thread running."
            )

        thread = threading.Thread(
            target = target,
            name   = f"infer-{state.run_id}",
            daemon = True,
        )
        state._thread = thread        # noqa: SLF001 — module-internal
        thread.start()
        logger.info(
            "InferenceStateManager: started background thread for run %s",
            state.run_id,
        )
        return thread


# ──────────────────────────────────────────────────────────────────────
# Helpers used by router and dependency
# ──────────────────────────────────────────────────────────────────────

def build_ensemble_config(
    registry_dir:    str,
    artifacts_dir:   str,
    new_scenario_dir: Optional[str] = None,
) -> EnsembleConfig:
    """Construct a minimal :class:`EnsembleConfig` for an API-driven run.

    The configuration the pipeline needs at runtime is small —
    ``registry_dir`` (for cold-load) and ``artifacts_dir`` (where
    ``post_infer`` writes the manifest + parquets).  Any further
    metadata (``input_mode``, ``new_scenario_dir`` default) is plumbed
    through the ``metadata`` dict.

    Parameters
    ----------
    registry_dir
        Where the ensemble version + member models live on disk.
    artifacts_dir
        Per-run artifacts root (e.g.
        ``<base_artifacts>/inference_runs/<run_id>``).  ``post_infer``
        writes ``inference/manifest.json`` and friends under this path.
    new_scenario_dir
        Optional default new-scenario directory; the API's
        ``/scenarios`` call passes its own path explicitly so this is
        only used by the convenience ``run()`` orchestrator.

    Returns
    -------
    EnsembleConfig
        Minimal config sufficient for ``load → load_scenarios →
        validate_scenarios → run_inference``.
    """
    metadata = {"inference": {"input_mode": "new_scenarios"}}
    if new_scenario_dir is not None:
        metadata["inference"]["new_scenario_dir"] = new_scenario_dir

    return EnsembleConfig(
        registry_dir  = registry_dir,
        artifacts_dir = artifacts_dir,
        metadata      = metadata,
    )


def per_run_artifacts_dir(base_artifacts_dir: str, run_id: str) -> str:
    """Convention for where per-run artifacts go on disk.

    ``<base_artifacts_dir>/inference_runs/<run_id>``.  The pipeline's
    ``post_infer`` further nests an ``inference/`` directory underneath
    for the manifest + parquets, so the final manifest path is
    ``<base_artifacts_dir>/inference_runs/<run_id>/inference/manifest.json``.

    Centralised here so the router, the state manager, and any future
    listing endpoint agree on the layout.
    """
    return str(Path(base_artifacts_dir) / "inference_runs" / run_id)


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton
# ──────────────────────────────────────────────────────────────────────

_manager: Optional[InferenceStateManager] = None


def set_inference_state_manager(manager: InferenceStateManager) -> None:
    """Inject the manager at app-lifespan startup."""
    global _manager
    _manager = manager


def get_inference_state_manager() -> InferenceStateManager:
    """FastAPI ``Depends`` — returns the singleton manager."""
    if _manager is None:
        raise RuntimeError(
            "InferenceStateManager not initialised. Server not ready."
        )
    return _manager
