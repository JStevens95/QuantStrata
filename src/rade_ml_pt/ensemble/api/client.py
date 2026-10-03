"""Typed Python client for the PRISM / Rade API.

Wraps every endpoint defined under :mod:`src.rade_ml_pt.ensemble.api.routers`.
Responses are validated with the same Pydantic models the server emits,
so the client and server cannot drift silently — if the wire format
changes, the model import is the single place that needs updating.

Design choices
--------------

* **Hand-written, not generated.**  The surface is small and the server
  is in the same repo, so maintaining an OpenAPI codegen step buys us
  nothing.  Every method here is a 3-line wrapper over ``httpx.get``.

* **httpx.Client** (sync) under the hood.  Supports sessions, HTTP/2,
  connection pooling, configurable timeouts and retries via transports.

* **Usable as a context manager.**  ``with RadeApiClient(...) as c:`` closes
  the underlying connection pool deterministically.

* **Binary endpoint (``/predictions``)** has two flavours — raw bytes
  and a decoded ``dict[str, np.ndarray]`` — so callers can stream and
  cache however they prefer.

Usage
-----

.. code-block:: python

    from src.rade_ml_pt.ensemble.api.client import RadeApiClient

    with RadeApiClient("http://localhost:8000") as c:
        overview = c.overview()
        ts = c.portfolio("test")
        preds = c.predictions(cluster_id="cluster_0", split="test")
"""
from __future__ import annotations

import io
import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

import httpx

if TYPE_CHECKING:
    # Only referenced in type annotations — numpy stays optional at
    # runtime so the client itself can be imported in numpy-free
    # contexts (e.g. a pure FastAPI-to-FastAPI forwarder).
    import numpy as np

from src.rade_ml_pt.ensemble.api.models.cluster_timeseries import (
    ClusterTimeseriesResponse,
)
from src.rade_ml_pt.ensemble.api.models.clusters import ClustersResponse
from src.rade_ml_pt.ensemble.api.models.elementary_pnl import (
    ElementaryPnlResponse,
)
from src.rade_ml_pt.ensemble.api.models.governance import (
    GovernanceRegistryResponse,
)
from src.rade_ml_pt.ensemble.api.models.graph_stats import GraphStatsResponse
from src.rade_ml_pt.ensemble.api.models.inference import (
    ClusterSummaryResponse,
    ClusterTradesResponse,
    EventsResponse,
    LoadResponse,
    LoadScenariosResponse,
    ManifestResponse,
    PortfolioResponse,
    RunResponse,
    RunsListResponse,
    ScenariosSnapshotResponse,
    StatusResponse,
    ValidateResponse,
    ValidationSnapshotResponse,
)
from src.rade_ml_pt.ensemble.api.models.group_correlations import (
    GroupCorrelationsResponse,
)
from src.rade_ml_pt.ensemble.api.models.meta import HealthResponse, VersionsResponse
from src.rade_ml_pt.ensemble.api.models.metrics import (
    EnsembleMetricsResponse,
    PerMemberMetricsResponse,
)
from src.rade_ml_pt.ensemble.api.models.overview import OverviewResponse
from src.rade_ml_pt.ensemble.api.models.portfolio import PortfolioTimeseriesResponse
from src.rade_ml_pt.ensemble.api.models.quality import (
    CompletenessResponse,
    FeatureSummaryResponse,
)
from src.rade_ml_pt.ensemble.api.models.trade_graph import TradeGraphResponse
from src.rade_ml_pt.ensemble.api.models.trades import TradesResponse
from src.rade_ml_pt.ensemble.api.models.training_curves import (
    TrainingCurvesResponse,
)

logger = logging.getLogger(__name__)


# Default timeouts (seconds).  ``DEFAULT_TIMEOUT`` covers every cheap
# GET (sub-second on a warm server, low-second under load).
# ``LOAD_TIMEOUT`` is sized for the worst-case ``POST /load`` — the
# single synchronous, registry-walking, all-clusters-context-reading
# endpoint.  It scales with portfolio size: ~1-2 min for typical
# ensembles, up to ~10 min for very large ones with eager
# ``cluster_assets`` loads.  Callers can still override per-call via
# the ``timeout`` kwarg on the underlying helpers.
DEFAULT_TIMEOUT: float = 60.0
LOAD_TIMEOUT:    float = 900.0  # 15 min — generous ceiling for cold-load.


# ── Errors ────────────────────────────────────────────────────────────

class RadeApiError(RuntimeError):
    """Raised when a Rade API call returns a non-2xx status."""

    def __init__(self, status_code: int, detail: Any, url: str):
        self.status_code = status_code
        self.detail = detail
        self.url = url
        super().__init__(
            f"Rade API error {status_code} on {url}: {detail}"
        )


# ── Client ────────────────────────────────────────────────────────────

class RadeApiClient:
    """Synchronous HTTP client for the Rade API.

    Parameters
    ----------
    base_url
        Root of the API, e.g. ``"http://localhost:8000"``.  The client
        appends ``/prism/v1/...`` internally.
    timeout
        Default request timeout in seconds (``httpx.Timeout`` accepts
        float or tuple).  Applies to all methods unless overridden by a
        transport.
    http2
        Negotiate HTTP/2 if the server supports it.  Small latency win
        when calling many small endpoints in one tab render.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        http2: bool = False,
    ):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            http2=http2,
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RadeApiClient":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # ── Internal helpers ──────────────────────────────────────────

    def _get_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """GET *path*, validating status and returning parsed JSON.

        ``timeout`` (seconds) overrides the client-wide default for
        this call only — useful for the rare heavy GET (e.g. a NPZ
        download) that needs more than the default.  Pass ``None``
        (the default) to use the client's configured timeout.
        """
        # Drop ``None`` query params so they don't appear as empty
        # strings on the wire — keeps server-side filters clean.
        clean = (
            {k: v for k, v in params.items() if v is not None}
            if params
            else None
        )
        request_kwargs: Dict[str, Any] = {"params": clean}
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        r = self._client.get(path, **request_kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RadeApiError(r.status_code, detail, str(r.request.url))
        return r.json()

    def _get_bytes(self, path: str, params: Optional[Dict[str, Any]] = None) -> bytes:
        """GET *path*, returning raw body bytes (for the NPZ endpoint)."""
        clean = (
            {k: v for k, v in params.items() if v is not None}
            if params
            else None
        )
        r = self._client.get(path, params=clean)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RadeApiError(r.status_code, detail, str(r.request.url))
        return r.content

    def _post_json(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Any:
        """POST *path* with optional JSON body, returning parsed JSON.

        Used by the inference control-plane endpoints (``/load``,
        ``/scenarios``, ``/validate``, ``/run``).  ``json=None`` is
        valid — FastAPI treats it the same as ``{}`` for endpoints
        with a Pydantic body that has all-default fields.

        ``timeout`` (seconds) overrides the client-wide default for
        this call only.  ``POST /load`` and any future heavy synchronous
        endpoint pass an explicit, larger value to avoid spurious
        ``httpx.ReadTimeout`` while the server is still working
        (the worker is unaffected — only the client gives up).

        Error handling mirrors :meth:`_get_json`: HTTP 4xx / 5xx are
        surfaced as :class:`RadeApiError` carrying the server-side
        ``detail`` string when present.
        """
        request_kwargs: Dict[str, Any] = {"json": json}
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        r = self._client.post(path, **request_kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise RadeApiError(r.status_code, detail, str(r.request.url))
        return r.json()

    # ── Meta ──────────────────────────────────────────────────────

    def health(self) -> HealthResponse:
        return HealthResponse(**self._get_json("/health"))

    def versions(self) -> VersionsResponse:
        return VersionsResponse(**self._get_json("/versions"))

    # ── Overview ──────────────────────────────────────────────────

    def overview(self) -> OverviewResponse:
        return OverviewResponse(**self._get_json("/prism/v1/overview"))

    # ── Portfolio ─────────────────────────────────────────────────

    def portfolio(
        self, split: str, *, space: str = "scaled",
    ) -> PortfolioTimeseriesResponse:
        """Portfolio-level timeseries for one split, one PnL space.

        ``space`` defaults to ``"scaled"`` (legacy behaviour); pass
        ``"original"`` to receive inverse-scaled, notional-sign-
        restored PnL.  See the server-side ``/prism/v1/portfolio``
        endpoint for the NaN-handling contract.
        """
        return PortfolioTimeseriesResponse(
            **self._get_json(
                "/prism/v1/portfolio",
                {"split": split, "space": space},
            )
        )

    # ── Metrics ───────────────────────────────────────────────────

    def ensemble_metrics(self) -> EnsembleMetricsResponse:
        return EnsembleMetricsResponse(
            **self._get_json("/prism/v1/metrics/ensemble")
        )

    def per_member_metrics(
        self,
        *,
        split: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> PerMemberMetricsResponse:
        return PerMemberMetricsResponse(
            **self._get_json(
                "/prism/v1/metrics/per-member",
                {"split": split, "cluster_id": cluster_id},
            )
        )

    # ── Clusters ──────────────────────────────────────────────────

    def clusters(self, *, cluster_id: Optional[str] = None) -> ClustersResponse:
        return ClustersResponse(
            **self._get_json("/prism/v1/clusters", {"cluster_id": cluster_id})
        )

    # ── Cluster timeseries ────────────────────────────────────────

    def cluster_timeseries(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
        space:      str            = "scaled",
    ) -> ClusterTimeseriesResponse:
        """Per-cluster timeseries for one split, one PnL space.

        ``space`` defaults to ``"scaled"`` (legacy behaviour); pass
        ``"original"`` to receive inverse-scaled, notional-sign-
        restored PnL on every nested cluster.
        """
        return ClusterTimeseriesResponse(
            **self._get_json(
                "/prism/v1/cluster-timeseries",
                {"split": split, "cluster_id": cluster_id, "space": space},
            )
        )

    # ── Trades ────────────────────────────────────────────────────

    def trades(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> TradesResponse:
        return TradesResponse(
            **self._get_json(
                "/prism/v1/trades",
                {"split": split, "cluster_id": cluster_id},
            )
        )

    # ── Group correlations ────────────────────────────────────────

    def group_correlations(
        self,
        split: str,
        *,
        attribute: Optional[str] = None,
    ) -> GroupCorrelationsResponse:
        return GroupCorrelationsResponse(
            **self._get_json(
                "/prism/v1/group-correlations",
                {"split": split, "attribute": attribute},
            )
        )

    # ── Graph stats ───────────────────────────────────────────────

    def graph_stats(self, *, cluster_id: Optional[str] = None) -> GraphStatsResponse:
        return GraphStatsResponse(
            **self._get_json(
                "/prism/v1/graph-stats",
                {"cluster_id": cluster_id},
            )
        )

    # ── Trade graph (nodes + edges for one cluster) ───────────────

    def trade_graph(self, *, cluster_id: str) -> TradeGraphResponse:
        """Nodes + edges for one cluster's trade graph.

        Cluster-scoped and required — the endpoint does not return an
        aggregated ensemble payload.  See :mod:`..models.trade_graph`
        for the schema.
        """
        return TradeGraphResponse(
            **self._get_json(
                "/prism/v1/trade-graph",
                {"cluster_id": cluster_id},
            )
        )

    # ── Training curves (per-cluster, per-epoch) ──────────────────

    def training_curves(self, *, cluster_id: str) -> TrainingCurvesResponse:
        """Per-epoch training curves for one cluster member.

        Always returns ``train_loss``; additional series depend on what
        the trainer emitted (``val_loss``, ``mae``, …).  See
        :mod:`..models.training_curves` for the schema.
        """
        return TrainingCurvesResponse(
            **self._get_json(
                f"/prism/v1/clusters/{cluster_id}/training-curves"
            )
        )

    # ── Elementary PnL (per-scenario raw PnL of selected elementary trades) ──

    def elementary_pnl(
        self,
        *,
        cluster_id: str,
        trade_ids: list[str],
    ) -> ElementaryPnlResponse:
        """Per-scenario PnL for one or more elementary trades in a cluster.

        Elementary trades are model inputs (atomic legs / hedge
        instruments), so the response carries raw PnL values only —
        no predictions or targets.  ``trade_ids`` is required and
        capped server-side at 50 ids per request.

        See :mod:`..models.elementary_pnl` for the schema.
        """
        # ``httpx`` flattens a list value into repeated query
        # parameters (``?trade_ids=a&trade_ids=b&…``) which is exactly
        # what FastAPI's ``Query(...)`` typed-as-list expects.
        return ElementaryPnlResponse(
            **self._get_json(
                f"/prism/v1/clusters/{cluster_id}/elementary-pnl",
                {"trade_ids": list(trade_ids)},
            )
        )

    # ── Quality ───────────────────────────────────────────────────

    def completeness(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> CompletenessResponse:
        return CompletenessResponse(
            **self._get_json(
                "/prism/v1/quality/completeness",
                {"split": split, "cluster_id": cluster_id},
            )
        )

    def feature_summary(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> FeatureSummaryResponse:
        return FeatureSummaryResponse(
            **self._get_json(
                "/prism/v1/quality/feature-summary",
                {"split": split, "cluster_id": cluster_id},
            )
        )

    # ── Governance (cross-version registry) ──────────────────────

    def governance_registry(self) -> GovernanceRegistryResponse:
        """Return one row per registered ensemble version.

        Cross-version endpoint — unlike every other method on this
        client, the response is independent of the active version
        served by the server (it covers the whole registry).  See
        :mod:`..models.governance` for the schema.
        """
        return GovernanceRegistryResponse(
            **self._get_json("/prism/v1/governance/registry")
        )

    # ── Predictions (binary NPZ) ──────────────────────────────────

    def predictions_bytes(
        self, *, cluster_id: str, split: str, space: str = "scaled",
    ) -> bytes:
        """Raw NPZ payload for one (cluster, split, space).

        ``space`` defaults to ``"scaled"`` (legacy behaviour); pass
        ``"original"`` to fetch the Phase 3.1 inverse-scaled shard.
        """
        return self._get_bytes(
            "/prism/v1/predictions",
            {"cluster_id": cluster_id, "split": split, "space": space},
        )

    def predictions(
        self, *, cluster_id: str, split: str, space: str = "scaled",
    ) -> Dict[str, "np.ndarray"]:
        """Decoded NPZ content: ``{"predictions": ndarray, "targets": ndarray}``.

        Requires :mod:`numpy` at runtime of the caller — imported
        lazily here so the client itself can be used in numpy-free
        contexts (e.g. a pure FastAPI-to-FastAPI forwarder).

        ``space`` defaults to ``"scaled"`` for backwards compatibility;
        pass ``"original"`` to fetch the inverse-scaled, notional-sign-
        restored shard introduced in Phase 3.1.
        """
        import numpy as np  # noqa: F811  (shadows the TYPE_CHECKING import on purpose)

        raw = self.predictions_bytes(
            cluster_id=cluster_id, split=split, space=space,
        )
        with np.load(io.BytesIO(raw)) as z:
            return {
                "predictions": z["predictions"].copy(),
                "targets": z["targets"].copy(),
            }

    # ==================================================================
    # Inference (Stage 11) — control plane + data plane
    #
    # Mirrors the seven control-plane and seven data-plane endpoints
    # exposed under ``/prism/v1/inference``.  The control plane drives
    # the live staged workflow (``load → scenarios → validate → run``)
    # plus the polling probes the dashboard's callbacks use to hydrate
    # the activity log and status indicator.  The data plane reads
    # historical run artifacts on disk for the "previous runs" picker
    # and cross-session deep-link views.
    #
    # Every method validates the response with the same Pydantic
    # model the server emits — wire-format drift is caught at the
    # import line, not at runtime.
    # ==================================================================

    # ── Inference control plane ───────────────────────────────────

    def load_ensemble(self, *, timeout: float = LOAD_TIMEOUT) -> LoadResponse:
        """Trigger a cold-load of the ensemble + per-cluster contexts.

        Side-effect on the server: replaces any prior run state.
        Returns once the load completes (the call is synchronous, not
        threaded — only ``/run`` is dispatched on a worker thread).

        ``timeout`` defaults to :data:`LOAD_TIMEOUT` (15 minutes),
        which is generous enough for the worst-case registry walk
        on a large ensemble.  Pass a larger value if your ensemble
        exceeds that ceiling; the client-wide default is intentionally
        not used here because it would spuriously fail mid-load.
        """
        return LoadResponse(
            **self._post_json("/prism/v1/inference/load", timeout=timeout)
        )

    def load_scenarios(self, new_scenario_dir: str) -> LoadScenariosResponse:
        """Parse a folder of shock CSVs into the active run state.

        ``new_scenario_dir`` must be readable by the API host
        process — multipart upload is a deferred Stage.  Resets
        any prior validation result on the server side.
        """
        return LoadScenariosResponse(
            **self._post_json(
                "/prism/v1/inference/scenarios",
                json={"new_scenario_dir": new_scenario_dir},
            )
        )

    def validate_scenarios(self) -> ValidateResponse:
        """Compute per-cluster routing + run-level errors / warnings.

        The returned ``ValidateResponse.is_valid`` is the gate the
        UI uses to enable the ``Run`` button.  A non-empty ``errors``
        list leaves the server in ``scenarios_loaded`` state so the
        caller can fix and re-validate without re-uploading.
        """
        return ValidateResponse(**self._post_json("/prism/v1/inference/validate"))

    def run_inference(
        self,
        *,
        artifacts_dir: Optional[str] = None,
    ) -> RunResponse:
        """Dispatch the inference run onto a background worker thread.

        Returns immediately with ``status='running'`` — the caller
        polls :meth:`inference_status` and :meth:`inference_events`
        until terminal state.

        Parameters
        ----------
        artifacts_dir
            Optional override for the per-run artifacts directory.
            ``None`` (the default) lets the server pick
            ``<settings.artifacts_dir>/inference_runs/<run_id>``
            — the path the result reader can discover.  Supply
            this only for one-off / debugging runs that should
            land outside the discoverable tree.
        """
        body = {"artifacts_dir": artifacts_dir} if artifacts_dir is not None else None
        return RunResponse(**self._post_json("/prism/v1/inference/run", json=body))

    def inference_status(self) -> StatusResponse:
        """Cheap probe used to gate the next-button in the UI.

        Designed for 1-2 Hz polling without appreciable load.
        Returns ``has_active_run=False`` cleanly when no run has
        been created yet.
        """
        return StatusResponse(**self._get_json("/prism/v1/inference/status"))

    def inference_events(self, *, cursor: int = 0) -> EventsResponse:
        """Cursor-paginated tail of the activity log.

        Pass back ``response.next_cursor`` on the next poll to
        chain efficiently — each call returns only the events
        strictly after ``cursor``.
        """
        return EventsResponse(
            **self._get_json(
                "/prism/v1/inference/events",
                {"cursor": cursor},
            )
        )

    def inference_manifest(self) -> ManifestResponse:
        """Return the active run's ``manifest.json``.

        Available only after ``status='complete'``; the server
        returns 409 while a run is still in flight and 404 if
        nothing has run yet.
        """
        return ManifestResponse(**self._get_json("/prism/v1/inference/manifest"))

    # ── Inference data plane (historical runs) ────────────────────

    def list_inference_runs(self) -> RunsListResponse:
        """Enumerate every inference run on disk, most recent first.

        Sorts by ``run_id`` descending — given the server's
        ``<ensemble_version>__<UTC>`` format, this is also
        descending temporal order within an ensemble version.
        """
        return RunsListResponse(**self._get_json("/prism/v1/inference/runs"))

    def inference_run_manifest(self, run_id: str) -> ManifestResponse:
        """Read ``manifest.json`` for any historical run.

        Counterpart to :meth:`inference_manifest` (which is scoped
        to the *active* run on the singleton state manager).
        """
        return ManifestResponse(
            **self._get_json(f"/prism/v1/inference/runs/{run_id}/manifest")
        )

    def inference_portfolio(self, run_id: str) -> PortfolioResponse:
        """Portfolio-level long-format summary parquet for ``run_id``.

        Columns: ``scenario_label``, ``sum_pnl_scaled``,
        ``sum_pnl_original``, ``n_clusters``.  One row per
        scenario.
        """
        return PortfolioResponse(
            **self._get_json(f"/prism/v1/inference/runs/{run_id}/portfolio")
        )

    def inference_clusters_summary(self, run_id: str) -> ClusterSummaryResponse:
        """Cluster-level long-format summary parquet for ``run_id``.

        One row per (cluster, scenario) pair with sum / mean /
        std / min / max columns for both scaled and original
        spaces.
        """
        return ClusterSummaryResponse(
            **self._get_json(f"/prism/v1/inference/runs/{run_id}/clusters")
        )

    def inference_cluster_trades(
        self,
        run_id:     str,
        cluster_id: str,
        *,
        space:      str = "original",
    ) -> ClusterTradesResponse:
        """Per-cluster trade-level wide parquet (matrix + axis labels).

        Wide-format on the wire because typical sizes are ~30k
        cells — a row-per-cell long-format response would be
        5-10× heavier on the wire.

        Parameters
        ----------
        space
            ``"original"`` (default) returns inverse-scaled +
            notional-restored PnLs; ``"scaled"`` returns model
            output space (z-scored, sign-unrestored).  The
            dashboard offers a toggle between the two.
        """
        return ClusterTradesResponse(
            **self._get_json(
                f"/prism/v1/inference/runs/{run_id}/clusters/{cluster_id}/trades",
                {"space": space},
            )
        )

    def inference_validation(self, run_id: str) -> ValidationSnapshotResponse:
        """Return the ``ValidationReport`` snapshot embedded in ``run_id``'s manifest.

        Cheaper than the active-run ``/validate`` because no
        on-disk parquet is involved — ``post_infer`` captures the
        report into the manifest at write time.
        """
        return ValidationSnapshotResponse(
            **self._get_json(f"/prism/v1/inference/runs/{run_id}/validation")
        )

    def inference_scenarios(self, run_id: str) -> ScenariosSnapshotResponse:
        """Return the ``LoadedScenariosReport`` snapshot embedded in the manifest."""
        return ScenariosSnapshotResponse(
            **self._get_json(f"/prism/v1/inference/runs/{run_id}/scenarios")
        )
