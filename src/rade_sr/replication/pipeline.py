"""
Portfolio-first preprocessing pipeline.

Phase 1 — portfolio-wide (executed once):
    1. Load all assets:      asset_config  →  Dict[risk_factor, Asset]
    2. Generate all trades:  assets        →  Dict[risk_factor, List[ElementaryTrade]]
    3. Compute all PnL:      trades+assets →  Dict[risk_factor, {trade_id: PnL}]
    4. Build PortfolioData (single source of truth)

Phase 2 — cluster resolution + population:
    5. Resolve clusters:     cluster_config → Dict[cluster_id, [risk_factors]]
    6. For each cluster:
       a. Resolve paths
       b. Slice PortfolioData by the cluster's risk factors → ReplicationJob

A cluster can contain one or many risk factors depending on the
clustering method (simple key grouping = 1:1, k-means = many:1).
Asset objects are self-contained — they load their own dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from src.rade_sr.core.types import (
    ElementaryTrade,
    PortfolioData,
    ReplicationJob,
)
from src.rade_sr.core.protocols import (
    ClusterResolver,
    PathResolver,
    PnLEngine,
    TradeGenerator,
)
from src.rade_sr.core.exceptions import PipelineError
from src.rade_sr.replication.registry import RiskFactorRegistry

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline.

    Parameters
    ----------
    cluster_config : dict
        Passed to ``ClusterResolver.resolve()``.  Typical keys:
        ``cluster_key``, ``cluster_key_values``, ``method``, etc.
    path_config : dict
        Root directory + filename conventions for cluster artifacts.
    asset_config : dict
        ``{risk_factor: {asset_class, pair/currency, ...}}``.
        Defines which risk factors to load.
    trade_config : dict
        Trade generation parameters: notional, expiry grid, payoff types.
    fail_fast : bool
        If True, abort the pipeline on the first stage failure.
    """
    cluster_config: Dict[str, Any] = field(default_factory=dict)
    path_config: Dict[str, Any] = field(default_factory=dict)
    asset_config: Dict[str, Any] = field(default_factory=dict)
    trade_config: Dict[str, Any] = field(default_factory=dict)
    fail_fast: bool = True


class PreprocessingPipeline:
    """Portfolio-first preprocessing pipeline for static replication.

    Depends ONLY on Protocol types from core/protocols.py.  Any stage
    can be swapped by passing a different object at construction time.
    """

    def __init__(
        self,
        cluster_resolver: ClusterResolver,
        path_resolver: PathResolver,
        trade_generator: TradeGenerator,
        pnl_engine: PnLEngine,
        registry: Optional[Type[RiskFactorRegistry]] = None,
        hooks: Optional[List[Callable[[ReplicationJob], None]]] = None,
        market_data_client: Optional[Any] = None,
    ) -> None:
        self._cluster_resolver = cluster_resolver
        self._path_resolver = path_resolver
        self._trade_generator = trade_generator
        self._pnl_engine = pnl_engine
        self._registry = registry or RiskFactorRegistry
        self._hooks = hooks or []
        # Injected MarketDataClient (mock or real). Flows to builders → assets.
        self._market_data_client = market_data_client

    # ── Public entry point ────────────────────────────────────────────

    def run(self, config: PipelineConfig) -> List[ReplicationJob]:
        """Execute the full pipeline by chaining the public stage methods.

        Returns one ReplicationJob per cluster. Equivalent to calling
        ``load_assets`` → ``generate_trades`` → ``compute_pnl`` →
        ``assemble_jobs`` in sequence.
        """
        assets = self.load_assets(config)
        trades = self.generate_trades(assets, config)
        pnls = self.compute_pnl(trades, assets)
        jobs = self.assemble_jobs(assets, trades, pnls, config)
        logger.info("Pipeline complete: %d jobs assembled.", len(jobs))
        return jobs

    # ── Public stage methods (run individually for step-by-step control) ──

    def load_assets(self, config: PipelineConfig) -> Dict[str, Any]:
        """[Stage 2] Load one self-contained Asset per risk factor."""
        assets = self._load_all_assets(config)
        logger.info("Loaded %d assets: %s", len(assets), list(assets.keys()))
        return assets

    def generate_trades(
        self, assets: Dict[str, Any], config: PipelineConfig,
    ) -> Dict[str, List[ElementaryTrade]]:
        """[Stage 3] Generate the elementary trade universe from the assets."""
        trades = self._generate_all_trades(assets, config)
        n = sum(len(ts) for ts in trades.values())
        logger.info("Generated %d elementary trades across %d factors", n, len(trades))
        return trades

    def compute_pnl(
        self, trades: Dict[str, List[ElementaryTrade]], assets: Dict[str, Any],
    ) -> Dict[str, Any]:
        """[Stage 4] Price every elementary trade across all shock scenarios."""
        pnls = self._pnl_engine.compute_batch(trades, assets)
        n = sum(len(p) for p in pnls.values())
        logger.info("Computed PnL for %d trades across %d factors", n, len(pnls))
        return pnls

    def assemble_jobs(
        self,
        assets: Dict[str, Any],
        trades: Dict[str, List[ElementaryTrade]],
        pnls: Dict[str, Any],
        config: PipelineConfig,
    ) -> List[ReplicationJob]:
        """[Stage 5] Resolve clusters and slice the portfolio into ReplicationJobs."""
        portfolio = PortfolioData(
            assets=assets, elementary_trades=trades, trade_pnls=pnls,
        )
        cluster_map = self._cluster_resolver.resolve(config.cluster_config)
        logger.info(
            "Resolved %d clusters: %s", len(cluster_map), list(cluster_map.keys()),
        )
        return self._populate_clusters(cluster_map, portfolio, config)

    # ── Phase 1 helpers ───────────────────────────────────────────────

    def _load_all_assets(self, config: PipelineConfig) -> Dict[str, Any]:
        """Load one Asset per unique risk factor in the portfolio."""
        import src.rade_sr.replication.builders  # noqa: F401

        from src.rade_sr.replication.builders import MARKET_DATA_CLIENT_KEY

        assets: Dict[str, Any] = {}
        for factor_id, factor_cfg in config.asset_config.items():
            try:
                builder = self._registry.get(factor_cfg["asset_class"])
                # Inject the shared market-data client (if any) without mutating
                # the caller's config dict.
                if self._market_data_client is not None:
                    factor_cfg = {
                        **factor_cfg,
                        MARKET_DATA_CLIENT_KEY: self._market_data_client,
                    }
                assets[factor_id] = builder.build(factor_id, factor_cfg)
            except Exception as exc:
                logger.error("Failed to load asset %s: %s", factor_id, exc)
                if config.fail_fast:
                    raise PipelineError(
                        f"Asset loading aborted at factor {factor_id!r}"
                    ) from exc
        return assets

    def _generate_all_trades(
        self,
        assets: Dict[str, Any],
        config: PipelineConfig,
    ) -> Dict[str, List[ElementaryTrade]]:
        """Generate all elementary trades across all risk factors."""
        return self._trade_generator.generate(assets, config.trade_config)

    # ── Phase 2 helpers ───────────────────────────────────────────────

    def _populate_clusters(
        self,
        cluster_map: Dict[str, List[str]],
        portfolio: PortfolioData,
        config: PipelineConfig,
    ) -> List[ReplicationJob]:
        """Populate each cluster by slicing PortfolioData.

        Validates that all risk factors in a cluster share the same
        asset class — mixing asset classes (e.g. FX + IR) in a single
        cluster is not supported.

        Parameters
        ----------
        cluster_map : dict[str, list[str]]
            ``cluster_id → [risk_factor_names]`` from the resolver.
        """
        jobs: List[ReplicationJob] = []
        for cluster_id, risk_factors in cluster_map.items():
            self._validate_asset_class_homogeneity(
                cluster_id, risk_factors, portfolio,
            )

            s = portfolio.slice(risk_factors)
            paths = self._path_resolver.resolve(cluster_id, config.path_config)

            job = ReplicationJob(
                cluster_id=cluster_id,
                cluster_paths=paths,
                assets=s.assets,
                elementary_trades=s.elementary_trades,
                trade_pnls=s.trade_pnls,
            )
            self._run_hooks(job)
            jobs.append(job)

        return jobs

    @staticmethod
    def _validate_asset_class_homogeneity(
        cluster_id: str,
        risk_factors: List[str],
        portfolio: PortfolioData,
    ) -> None:
        """Ensure all risk factors in a cluster share the same asset class.

        Raises
        ------
        PipelineError
            If the cluster contains risk factors from different asset classes.
        """
        asset_classes = set()
        for rf in risk_factors:
            asset = portfolio.assets.get(rf)
            if asset is not None and hasattr(asset, "asset_class"):
                asset_classes.add(asset.asset_class)

        if len(asset_classes) > 1:
            raise PipelineError(
                f"Cluster {cluster_id!r} contains mixed asset classes "
                f"{asset_classes}. A cluster must be homogeneous — "
                f"all risk factors must share the same asset class."
            )

    def _run_hooks(self, job: ReplicationJob) -> None:
        """Execute post-assembly hooks."""
        for hook in self._hooks:
            try:
                hook(job)
            except Exception as exc:
                logger.warning(
                    "Hook %s raised for cluster %s: %s",
                    getattr(hook, "__name__", repr(hook)),
                    job.cluster_id,
                    exc,
                )
