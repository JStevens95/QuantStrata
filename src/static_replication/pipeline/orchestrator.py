"""
Pipeline orchestrator — sequences stages, owns :class:`RunContext`, exposes partial runs.

Entry points
------------
Full pipeline::

    config = OrchestratorConfig.from_yaml("configs/pipeline_config.yaml")
    orch = Orchestrator.from_config(config, portfolio_client, market_data_client)
    ctx = orch.run()

Single stage while wiring portfolio / resolution::

    orch = Orchestrator.from_config(...)
    orch.load_portfolio()      # stage 1
    orch.resolve_factors()     # stage 2
    print(orch.ctx.universe)   # inspect without re-calling Sage

Design rules
------------
* Methods on this class are **thin controllers**: check prerequisites → delegate to
  ``portfolio/``, ``data/``, ``trade/`` modules → store result on ``self.ctx``.
* No pandas transforms, no Sage/STAR calls, no pricing maths here.
* Unimplemented stages raise :class:`~src.static_replication.core.exceptions.PipelineError`
  with a clear message until the corresponding module is wired.

Stage order (full ``run()``)
----------------------------
1. ``load_portfolio``     — ingest + format + validate + match
2. ``resolve_factors``    — ``factor_config`` → risk factor universe
3. ``build_assets``       — registry + STAR → loaded Asset instances
4. ``generate_trades``    — elementary trade grid per factor
5. ``compute_pnl``        — scenario PnL for every elementary trade
6. ``assemble_portfolio_data`` — bundle into :class:`PortfolioData`
7. ``build_jobs``         — cluster + slice → :class:`ReplicationJob` list

Stages 3a–3c write **separate** fields on :class:`RunContext` (``assets``,
``elementary_trades``, ``elementary_trade_pnl``).  Stage 3d (``assemble_portfolio_data``)
bundles them into :class:`PortfolioData` for clustering only.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from src.static_replication.config.config import OrchestratorConfig
from src.static_replication.core.exceptions import PipelineError
from src.static_replication.core.registry import RiskFactorRegistry
from src.static_replication.core.types import (
    ElementaryTrade,
    ElementaryTradePnL,
    PortfolioData,
    PortfolioLoadResult,
    ReplicationJob,
)
from src.static_replication.data.builders import FxRiskFactorBuilder, IrRiskFactorBuilder
from src.static_replication.pipeline.context import RunContext

logger = logging.getLogger(__name__)


def _import_portfolio_loader():
    """
    Lazy import so the orchestrator module loads even before ``portfolio/loader.py``
    exists.  Returns the callable or ``None``.
    """
    try:
        from src.static_replication.portfolio.loader import load_portfolio
        return load_portfolio
    except ImportError:
        return None


def _import_factor_resolver():
    """
    Lazy import for ``portfolio.resolution.resolver.resolve_factors``.

    Returns the callable or ``None`` if the resolution package is not pasted yet.
    """
    try:
        from src.static_replication.portfolio.resolution.resolver import resolve_factors
        return resolve_factors
    except ImportError:
        return None


class Orchestrator:
    """
    Top-level driver for static-replication preprocessing.

    Parameters
    ----------
    config :
        Typed pipeline recipe (``OrchestratorConfig.from_yaml(...)``).
    portfolio_client :
        Work Sage client or dev CSV mock; must expose portfolio attribute/PnL fetch.
    market_data_client :
        Work STAR client or dev mock; passed to FX/IR risk-factor builders.
    registry :
        Optional pre-built registry (tests).  Defaults to FX + IR builders.
    trade_generator, pnl_engine :
        Optional overrides for elementary trade generation and batch PnL.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        portfolio_client: Any,
        market_data_client: Any,
        registry: Optional[RiskFactorRegistry] = None,
        trade_generator: Any = None,
        pnl_engine: Any = None,
    ) -> None:
        # --- clients & config (immutable for the run) ---
        self._trade_generator = trade_generator
        self._pnl_engine = pnl_engine

        # --- working ledger: every stage reads/writes ctx fields ---
        self.ctx = RunContext(
            config=config,
            portfolio_client=portfolio_client,
            market_data_client=market_data_client,
            registry=registry or self._build_default_registry(market_data_client),
        )

    # ==================================================================
    # Construction helpers
    # ==================================================================

    @classmethod
    def from_config(
        cls,
        config: OrchestratorConfig,
        portfolio_client: Any,
        market_data_client: Any,
        *,
        registry: Optional[RiskFactorRegistry] = None,
        trade_generator: Any = None,
        pnl_engine: Any = None,
    ) -> "Orchestrator":
        """
        Preferred constructor: YAML config + two clients → ready orchestrator.

        Example::

            config = OrchestratorConfig.from_yaml("configs/pipeline_config.yaml")
            orch = Orchestrator.from_config(config, sage_client, star_client)
        """
        return cls(
            config=config,
            portfolio_client=portfolio_client,
            market_data_client=market_data_client,
            registry=registry,
            trade_generator=trade_generator,
            pnl_engine=pnl_engine,
        )

    @staticmethod
    def _build_default_registry(market_data_client: Any) -> RiskFactorRegistry:
        """
        Register the standard FX and IR builders against the market-data client.

        Each builder wraps ``FxAsset.load`` / ``IrAsset.load`` (STAR API calls).
        """
        registry = RiskFactorRegistry()
        registry.register(FxRiskFactorBuilder(market_data_client))
        registry.register(IrRiskFactorBuilder(market_data_client))
        return registry

    # ==================================================================
    # Timing — records wall-clock ms on ctx.timings_ms per stage
    # ==================================================================

    @contextmanager
    def _timed(self, stage_name: str):
        """Context manager: log start/finish and store elapsed ms on the context."""
        logger.info("stage start: %s", stage_name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.ctx.timings_ms[stage_name] = elapsed_ms
            logger.info("stage done:  %s (%.1f ms)", stage_name, elapsed_ms)

    # ==================================================================
    # Prerequisite guards — enable standalone stage calls with clear errors
    # ==================================================================

    def _require_portfolio_load(self) -> PortfolioLoadResult:
        """Ensure stage 1 completed before running stage 2+."""
        if self.ctx.portfolio_load is None:
            raise PipelineError(
                "Portfolio not loaded — call load_portfolio() first."
            )
        return self.ctx.portfolio_load

    def _require_universe(self) -> Any:
        """Ensure stage 2 completed before building assets / trades."""
        if self.ctx.universe is None:
            raise PipelineError(
                "Risk factors not resolved — call resolve_factors() first."
            )
        return self.ctx.universe

    def _require_portfolio_data(self) -> PortfolioData:
        """Ensure the bundled PortfolioData exists (assemble step or auto-build)."""
        if self.ctx.portfolio_data is None:
            # Allow clustering after separate stage fields are populated.
            self.assemble_portfolio_data()
        return self.ctx.portfolio_data  # type: ignore[return-value]

    def _require_assets(self) -> Dict[str, Any]:
        """Ensure build_assets() has populated ctx.assets."""
        if self.ctx.assets is None:
            raise PipelineError("Assets not built — call build_assets() first.")
        return self.ctx.assets

    def _require_elementary_trades(self) -> Dict[str, List[ElementaryTrade]]:
        """Ensure generate_trades() has populated ctx.elementary_trades."""
        if self.ctx.elementary_trades is None:
            raise PipelineError("Elementary trades not generated — call generate_trades() first.")
        return self.ctx.elementary_trades

    def _require_elementary_pnl(self) -> Dict[str, Dict[str, ElementaryTradePnL]]:
        """Ensure compute_pnl() has populated ctx.elementary_trade_pnl."""
        if self.ctx.elementary_trade_pnl is None:
            raise PipelineError("Elementary PnL not computed — call compute_pnl() first.")
        return self.ctx.elementary_trade_pnl

    def _require_callable(self, fn: Any, module_hint: str) -> Any:
        """Raise a actionable PipelineError when a delegate module is not wired yet."""
        if fn is None:
            raise PipelineError(
                f"Stage not wired — implement {module_hint} and ensure it is importable."
            )
        return fn

    # ==================================================================
    # Stage 1 — portfolio ingest
    # ==================================================================

    def load_portfolio(self) -> PortfolioLoadResult:
        """
        Load, format, validate, and match portfolio attributes + target PnL.

        Delegates to ``portfolio.loader.load_portfolio`` which chains:

        * client fetch (Sage or file mock)
        * ``portfolio.formatters`` — pivot / trade_id / column normalisation
        * ``portfolio.validation`` — invariant checks
        * ``portfolio.matching`` — align attributes ↔ PnL → ``MatchResult``

        Returns
        -------
        PortfolioLoadResult
            Stored on ``self.ctx.portfolio_load``.
        """
        load_fn = self._require_callable(
            _import_portfolio_loader(),
            "portfolio/loader.py (facade over formatters, validation, matching)",
        )

        with self._timed("load_portfolio"):
            # Pass portfolio-specific config block; loader knows how to call the client.
            result = load_fn(
                client=self.ctx.portfolio_client,
                portfolio_config=self.ctx.config.portfolio_config,
            )

            # Persist on the context ledger for downstream stages and debugging.
            self.ctx.portfolio_load = result

            # Surface validation warnings at context top-level for easy logging.
            self.ctx.validation_warnings = list(
                result.diagnostics.get("warnings", [])
            )

            logger.info("portfolio loaded: %s", result.summary())
            return result

    # ==================================================================
    # Stage 2 — risk factor resolution
    # ==================================================================

    def resolve_factors(self) -> Any:
        """
        Resolve the de-duplicated risk-factor universe from portfolio attributes.

        Applies ``config.factor_config`` rules (``fx_config`` file mapping,
        ``ir_config`` attribute lookup, etc.).

        Prerequisites
        -------------
        load_portfolio()

        Returns
        -------
        RiskFactorUniverse
            Stored on ``self.ctx.universe``.  Expected attributes: ``specs``,
            ``factors_by_trade`` (see ``RunContext.universe`` docstring).
        """
        load_result = self._require_portfolio_load()
        resolve_fn = self._require_callable(
            _import_factor_resolver(),
            "portfolio/resolution/resolver.py (apply config.factor_config)",
        )

        with self._timed("resolve_factors"):
            universe = resolve_fn(
                attributes=load_result.attributes,
                factor_config=self.ctx.config.factor_config,
            )

            self.ctx.universe = universe
            n_specs = len(getattr(universe, "specs", {}) or {})
            logger.info("resolved %d risk factor specs", n_specs)
            return universe

    # ==================================================================
    # Stage 3 — market data + elementary universe + PnL (portfolio-level)
    # ==================================================================

    def build_assets(self) -> Dict[str, Any]:
        """
        Load market data for every resolved factor via the registry.

        Iterates ``universe.specs``, dispatches to ``FxRiskFactorBuilder`` or
        ``IrRiskFactorBuilder`` based on each spec's ``asset_class``.

        Prerequisites
        -------------
        resolve_factors()

        Returns
        -------
        dict
            ``factor_id → loaded Asset``, stored on ``self.ctx.assets``.
        """
        universe = self._require_universe()
        specs = getattr(universe, "specs", None)
        if not specs:
            raise PipelineError("Risk factor universe has no specs — check resolver output.")

        assets: Dict[str, Any] = {}

        with self._timed("build_assets"):
            for factor_id, spec in specs.items():
                # Registry key must match builder.asset_class ("FX", "IR", …).
                asset_class = getattr(spec, "asset_class", None) or spec.get("asset_class")  # type: ignore[union-attr]
                if asset_class is None:
                    raise PipelineError(f"Factor spec {factor_id!r} missing asset_class.")

                builder = self.ctx.registry.get(str(asset_class))

                # Merge global asset defaults with any per-factor overrides on the spec.
                factor_config = self._factor_build_config(factor_id, spec)

                logger.debug("building asset %s (%s)", factor_id, asset_class)
                assets[factor_id] = builder.build(factor_id, factor_config)

            # One context field per stage — inspect ctx.assets without running PnL.
            self.ctx.assets = assets
            logger.info("built %d assets", len(assets))
            return assets

    def _factor_build_config(self, factor_id: str, spec: Any) -> Dict[str, Any]:
        """
        Build the ``factor_config`` dict passed to ``RiskFactorBuilder.build``.

        Combines orchestrator-level ``asset_config`` defaults with fields from
        the resolved spec (asset class, dependency metadata, etc.).
        """
        base = dict(self.ctx.config.asset_config or {})
        asset_class = getattr(spec, "asset_class", None)
        if asset_class is None and isinstance(spec, dict):
            asset_class = spec.get("asset_class")

        cfg: Dict[str, Any] = {
            **base,
            "asset_class": asset_class,
            "asset_name": factor_id,
        }

        # Optional per-spec build hints (populated by resolver when implemented).
        extra = getattr(spec, "meta", None)
        if extra is not None:
            cfg["extra"] = dict(extra)
        elif isinstance(spec, dict) and "meta" in spec:
            cfg["extra"] = dict(spec["meta"])

        return cfg

    def generate_trades(self) -> Dict[str, List[Any]]:
        """
        Generate the elementary trade universe at portfolio level.

        Prerequisites
        -------------
        build_assets()

        Notes
        -----
        Wired via ``TradeGenerator`` when ``self._trade_generator`` is provided
        or the default generator is importable.
        """
        self._require_universe()
        assets = self._require_assets()

        if self._trade_generator is None:
            try:
                from src.static_replication.trade.generator import TradeGenerator
                self._trade_generator = TradeGenerator()
            except ImportError as exc:
                raise PipelineError(
                    "Trade generator not available — wire trade/generator.py"
                ) from exc

        with self._timed("generate_trades"):
            trades = self._trade_generator.generate(
                assets=assets,
                trade_config=self.ctx.config.trade_config,
            )
            self.ctx.elementary_trades = trades
            n_total = sum(len(v) for v in trades.values())
            logger.info("generated %d elementary trades across %d factors", n_total, len(trades))
            return trades

    def compute_pnl(self) -> Any:
        """
        Compute scenario PnL for all elementary trades (portfolio-level, once).

        Prerequisites
        -------------
        generate_trades()

        Notes
        -----
        Requires a ``PnLEngine`` implementation (``self._pnl_engine`` or future
        ``pricing/pnl_engine.py``).
        """
        if self._pnl_engine is None:
            raise PipelineError(
                "PnL engine not configured — inject pnl_engine=... or implement "
                "pricing/pnl_engine.py"
            )

        trades = self._require_elementary_trades()
        assets = self._require_assets()

        with self._timed("compute_pnl"):
            pnl = self._pnl_engine.compute_batch(trades=trades, risk_factors=assets)
            self.ctx.elementary_trade_pnl = pnl
            return pnl

    def assemble_portfolio_data(self) -> PortfolioData:
        """
        Bundle separate stage outputs into one :class:`PortfolioData` view.

        Reads ``ctx.assets``, ``ctx.elementary_trades``, ``ctx.elementary_trade_pnl``,
        and ``ctx.portfolio_load`` — does **not** re-fetch from Sage/STAR.

        Clustering uses this bundle for ``PortfolioData.slice`` only.

        Prerequisites
        -------------
        build_assets(), generate_trades(), compute_pnl(), and load_portfolio().
        """
        load_result = self._require_portfolio_load()
        assets = self._require_assets()
        trades = self._require_elementary_trades()
        pnl = self._require_elementary_pnl()

        with self._timed("assemble_portfolio_data"):
            cob = self.ctx.cob_date
            as_of = None
            if cob is not None:
                try:
                    from datetime import date
                    as_of = date.fromisoformat(str(cob)[:10].replace("/", "-"))
                except ValueError:
                    as_of = None

            portfolio_data = PortfolioData(
                assets=assets,
                elementary_trades=trades,
                elementary_trade_pnl=pnl,
                portfolio_attributes=load_result.attributes,
                portfolio_pnl=load_result.pnl,
                as_of_date=as_of,
            )
            portfolio_data.validate()

            self.ctx.portfolio_data = portfolio_data
            logger.info(
                "PortfolioData assembled: %d factors, %d trades, %d scenarios",
                portfolio_data.n_factors,
                portfolio_data.n_trades,
                portfolio_data.n_scenarios,
            )
            return portfolio_data

    # ==================================================================
    # Stage 4 — cluster + ML handoff jobs
    # ==================================================================

    def build_jobs(self) -> List[ReplicationJob]:
        """
        Resolve clusters and slice ``PortfolioData`` into ``ReplicationJob`` objects.

        Uses ``config.cluster_config`` (``cluster_key`` columns) and
        ``AttributeClusterResolver`` to map clusters → factor ids, then
        ``PortfolioData.slice`` for each cluster.

        Prerequisites
        -------------
        assemble_portfolio_data()
        """
        portfolio_data = self._require_portfolio_data()
        cluster_cfg = self.ctx.config.cluster_config or {}
        cluster_key = cluster_cfg.get("cluster_key", ["AssetClassCode"])

        with self._timed("build_jobs"):
            from src.static_replication.data.cluster_resolver import AttributeClusterResolver

            resolver = AttributeClusterResolver()
            factor_ids = list(portfolio_data.assets.keys())

            # cluster_id → list of factor ids belonging to that cluster
            cluster_map = resolver.resolve(
                portfolio_attributes=portfolio_data.portfolio_attributes,
                available_factor_ids=factor_ids,
                cluster_key=cluster_key,
            )

            jobs: List[ReplicationJob] = []
            for cluster_id, rf_ids in cluster_map.items():
                # Slice pre-computed portfolio state — no STAR/Sage calls here.
                slice_ = portfolio_data.slice(rf_ids)
                jobs.append(ReplicationJob(
                    cluster_id=str(cluster_id),
                    paths=None,  # PathResolver can fill artifact paths later
                    assets=slice_.assets,
                    elementary_trades=slice_.elementary_trades,
                    elementary_trade_pnl=slice_.elementary_trade_pnl,
                    metadata={"cluster_key": cluster_key, "risk_factor_ids": rf_ids},
                ))

            self.ctx.jobs = jobs
            logger.info("built %d replication jobs", len(jobs))
            return jobs

    # ==================================================================
    # Full run
    # ==================================================================

    def run(self) -> RunContext:
        """
        Execute all pipeline stages in order and return the populated context.

        Stages that are not yet wired (missing ``portfolio/`` modules, PnL engine,
        etc.) raise :class:`PipelineError` with an actionable message.  Completed
        stage outputs remain on ``self.ctx`` for inspection.

        Returns
        -------
        RunContext
            Fully populated when every stage succeeds.
        """
        logger.info("pipeline run start (run_id=%s)", self.ctx.config.run_id)
        try:
            # --- phase 1: portfolio + factors (wire these first) ---
            self.load_portfolio()
            self.resolve_factors()

            # --- phase 2: portfolio-level pricing (uncomment as modules land) ---
            self.build_assets()
            self.generate_trades()
            self.compute_pnl()
            self.assemble_portfolio_data()

            # --- phase 3: cluster handoff ---
            self.build_jobs()

        except PipelineError:
            # Already actionable — re-raise without wrapping.
            raise
        except NotImplementedError as exc:
            raise PipelineError(str(exc)) from exc
        except Exception as exc:
            logger.exception("pipeline failed")
            raise PipelineError(f"Pipeline failed: {exc}") from exc

        logger.info("pipeline run complete: %s", self.ctx.summary())
        return self.ctx
