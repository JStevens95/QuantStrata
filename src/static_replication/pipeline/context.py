"""
Run context — the orchestrator's working ledger for one pipeline run.

Why a context object?
---------------------
The orchestrator sequences stages; each stage produces a typed artifact the next
stage consumes.  Storing those artifacts on a single :class:`RunContext` (rather
than as a dozen private attributes on the orchestrator) gives you:

* **Partial runs** — call ``load_portfolio()`` then ``resolve_factors()`` without
  re-fetching from Sage/STAR.
* **Debuggability** — inspect ``orch.ctx.universe`` in a notebook after one stage.
* **Re-slicing** — re-cluster from an existing ``portfolio_data`` without repricing.

Stage outputs vs bundled view
-----------------------------
Stages 3a–3c write **separate** context fields (``assets``, ``elementary_trades``,
``elementary_trade_pnl``) so you can inspect or re-run one step without losing the
others.  ``assemble_portfolio_data`` optionally bundles them into a single
:class:`~src.static_replication.core.types.PortfolioData` for slicing/clustering.

Config vs context
-----------------
* :class:`~src.static_replication.config.config.OrchestratorConfig` — immutable
  recipe loaded from YAML (``factor_config``, ``portfolio_config``, …).  Never
  mutated during a run.
* :class:`RunContext` — mutable outputs and timings for **one** invocation.

Field population order mirrors :meth:`Orchestrator.run` — see orchestrator docstring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.static_replication.config.config import OrchestratorConfig
from src.static_replication.core.registry import RiskFactorRegistry
from src.static_replication.core.types import (
    ElementaryTrade,
    ElementaryTradePnL,
    PortfolioData,
    PortfolioLoadResult,
    ReplicationJob,
    RiskFactorUniverse,
)


@dataclass
class RunContext:
    """
    Mutable state shared across all orchestrator stages for a single run.

    Each optional field starts as ``None`` / empty and is filled by exactly one
    stage method on :class:`~src.static_replication.pipeline.orchestrator.Orchestrator`.
    """

    # ------------------------------------------------------------------
    # Fixed inputs — set once at orchestrator construction, never replaced
    # ------------------------------------------------------------------

    config: OrchestratorConfig
    """Pipeline recipe (YAML-loaded).  Read-only for the lifetime of the run."""

    portfolio_client: Any
    """Sage (work) or CSV/file mock (dev).  Must satisfy ``SageClient`` protocol."""

    market_data_client: Any
    """STAR (work) or CSV/file mock (dev).  Must satisfy ``StarClient`` protocol."""

    registry: RiskFactorRegistry
    """Maps asset class string (``"FX"``, ``"IR"``) → ``RiskFactorBuilder`` instance."""

    # ------------------------------------------------------------------
    # Stage 1 — portfolio ingest  (Orchestrator.load_portfolio)
    # ------------------------------------------------------------------

    portfolio_load: Optional[PortfolioLoadResult] = None
    """
    Formatted attributes, aligned PnL, and match diagnostics.

    Produced by ``portfolio.loader.load_portfolio`` (formatters → validation → matching).
    """

    validation_warnings: List[str] = field(default_factory=list)
    """
    Non-fatal issues from the validation stage (e.g. missing optional columns).

    Copied out of ``portfolio_load.diagnostics`` for easy inspection.
    """

    # ------------------------------------------------------------------
    # Stage 2 — risk factor resolution  (Orchestrator.resolve_factors)
    # ------------------------------------------------------------------

    universe: Optional[RiskFactorUniverse] = None
    """
    De-duplicated risk-factor universe for the whole portfolio.

    Expected shape once wired (see ``portfolio/resolution/resolver.py``):

    * ``specs: dict[str, RiskFactorSpec]`` — every primary + dependency factor
    * ``factors_by_trade: dict[str, list[str]]`` — trade_id → primary factor ids
    """

    # ------------------------------------------------------------------
    # Stage 3a — market data  (Orchestrator.build_assets)
    # ------------------------------------------------------------------

    assets: Optional[Dict[str, Any]] = None
    """
    Loaded market-data objects keyed by risk factor id.

    ``factor_id → Asset`` (``FxAsset`` / ``IrAsset``).  Populated by STAR/API calls
    via the risk-factor registry; consumed by trade generation and PnL.
    """

    # ------------------------------------------------------------------
    # Stage 3b — elementary trades  (Orchestrator.generate_trades)
    # ------------------------------------------------------------------

    elementary_trades: Optional[Dict[str, List[ElementaryTrade]]] = None
    """
    Elementary (replicating) trade universe keyed by risk factor id.

    ``factor_id → list[ElementaryTrade]``.  Built once at portfolio level.
    """

    # ------------------------------------------------------------------
    # Stage 3c — elementary scenario PnL  (Orchestrator.compute_pnl)
    # ------------------------------------------------------------------

    elementary_trade_pnl: Optional[Dict[str, Dict[str, ElementaryTradePnL]]] = None
    """
    Scenario PnL for every elementary trade, keyed by factor then trade id.

    ``factor_id → trade_id → ElementaryTradePnL``.  Assumes all vectors share the
    same scenario count.
    """

    # ------------------------------------------------------------------
    # Stage 3d — bundled portfolio view  (Orchestrator.assemble_portfolio_data)
    # ------------------------------------------------------------------

    portfolio_data: Optional[PortfolioData] = None
    """
    Optional **bundle** of assets + trades + PnL + target portfolio frames.

    Assembled from the three stage fields above plus ``portfolio_load``.  Used by
    ``build_jobs`` for ``PortfolioData.slice`` — cluster jobs do not reprice.
    """

    # ------------------------------------------------------------------
    # Stage 4 — cluster jobs  (Orchestrator.build_jobs)
    # ------------------------------------------------------------------

    jobs: List[ReplicationJob] = field(default_factory=list)
    """
    One :class:`~src.static_replication.core.types.ReplicationJob` per cluster — the
    handoff format for downstream ML (rade_ml / hybrid GNN-RNN).
    """

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    timings_ms: Dict[str, float] = field(default_factory=dict)
    """Wall-clock milliseconds per stage name (filled by ``Orchestrator._timed``)."""

    # ------------------------------------------------------------------
    # Convenience accessors — avoid repeated ``if portfolio_load is None`` checks
    # ------------------------------------------------------------------

    @property
    def attributes(self):
        """
        Normalised portfolio attributes (one row per trade), or ``None`` if stage 1
        has not run yet.
        """
        if self.portfolio_load is None:
            return None
        return self.portfolio_load.attributes

    @property
    def target_pnl(self):
        """
        Target portfolio PnL frame (trades × scenarios), or ``None`` if stage 1 has
        not run yet.
        """
        if self.portfolio_load is None:
            return None
        return self.portfolio_load.pnl

    @property
    def cob_date(self) -> Optional[str]:
        """
        COB date string from ``config.asset_config`` (``YYYYMMDD`` or ISO).

        Used when building per-factor ``AssetConfig`` for STAR lookups.
        """
        asset_cfg = self.config.asset_config or {}
        cob = asset_cfg.get("cob_date")
        return str(cob) if cob is not None else None

    @property
    def n_trades(self) -> int:
        """Number of trades after portfolio load, or 0 if not loaded."""
        attrs = self.attributes
        return len(attrs) if attrs is not None else 0

    @property
    def n_factors(self) -> int:
        """Number of resolved risk factors, or 0 if universe not built."""
        if self.universe is None:
            return 0
        specs = getattr(self.universe, "specs", None)
        return len(specs) if specs is not None else 0

    @property
    def n_assets(self) -> int:
        """Number of loaded asset objects, or 0 if build_assets has not run."""
        return len(self.assets) if self.assets is not None else 0

    @property
    def n_elementary_trades(self) -> int:
        """Total elementary trades across all factors, or 0 if not generated yet."""
        if self.elementary_trades is None:
            return 0
        return sum(len(trades) for trades in self.elementary_trades.values())

    @property
    def run_id(self) -> str:
        """Unique run id for artifact folders (base label + datetime suffix from config)."""
        return self.config.run_id

    def summary(self) -> str:
        """Single-line summary for logging at end of run or after a partial stage."""
        return (
            f"RunContext(run_id={self.run_id!r}, trades={self.n_trades}, "
            f"factors={self.n_factors}, assets={self.n_assets}, "
            f"elementary={self.n_elementary_trades}, jobs={len(self.jobs)}, "
            f"stages={list(self.timings_ms.keys())})"
        )
