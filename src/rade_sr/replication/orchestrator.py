"""
Orchestrator — wires all components and runs the preprocessing pipeline.

Top-level entry point that imports concrete stage implementations,
constructs them, and hands them to the PreprocessingPipeline.

Usage
-----
::

    from src.rade_sr.replication.orchestrator import build_pipeline, run_preprocessing
    from src.rade_sr.config.loader import ConfigLoader

    config = ConfigLoader("/path/to/config").load_pipeline_config()
    pipeline = build_pipeline(cluster_manager, pricer)
    jobs = run_preprocessing(pipeline, config)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from src.rade_sr.core.types import ReplicationJob
from src.rade_sr.replication.pipeline import (
    PipelineConfig,
    PreprocessingPipeline,
)
from src.rade_sr.replication.cluster_resolver import ClusterResolver
from src.rade_sr.replication.path_resolver import ConventionPathResolver
from src.rade_sr.replication.trade_generator import (
    RiskFactorAwareTradeGenerator,
)
from src.rade_sr.replication.pnl_engine import VectorisedPnLEngine

import src.rade_sr.replication.builders  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of an end-to-end ``run_from_source`` call.

    Attributes
    ----------
    inputs : PortfolioInputs
        Step 0 ingestion output (normalized attributes, asset_config,
        dependency graph, target PnL).
    jobs : list[ReplicationJob]
        One assembled job per cluster.
    """
    inputs: Any
    jobs: List[ReplicationJob] = field(default_factory=list)
    jobs_manifest_path: Optional[str] = None


def default_trade_config() -> dict:
    """Sensible default elementary-trade universe (FX + IR)."""
    return {
        "notional": 1.0,
        "maturities": [0.25, 0.5, 1.0, 2.0],
        "fx": {
            "strike_pcts": [0.90, 0.95, 1.00, 1.05, 1.10],
            "include_forwards": True,
            "include_vanillas": True,
            "include_digitals": True,
        },
        "rates": {
            "strike_offsets_bp": [-100, -50, 0, 50, 100],
            "include_swaps": True,
            "include_swaptions": True,
            "include_caps_floors": False,
            "vol_type": "normal",
        },
    }


@dataclass
class RunConfig:
    """Single configuration object that drives a preprocessing run.

    This is the "model configuration" you pass to :class:`Orchestrator`. It
    captures every knob; the external dependencies (portfolio source, market-data
    client, pricer) are injected separately so the same config is reusable across
    environments.

    Parameters
    ----------
    trade_config : dict
        Elementary-trade universe (strike/tenor grids per asset class).
        Defaults to :func:`default_trade_config`.
    cluster_config : dict
        Clustering config. Default ``{"method": "one_to_one"}`` (one cluster per
        risk factor). Use ``{"method": "explicit", "cluster_map": {...}}`` to group.
    output_dir : str or None
        If set, writes the per-cluster artifacts + ``jobs.pkl`` handoff here.
    max_workers : int
        PnL-engine concurrency (1 = serial; safe default).
    path_config : dict or None
        Override the artifact path convention (advanced; derived from
        ``output_dir`` by default).
    """
    trade_config: dict = field(default_factory=default_trade_config)
    cluster_config: dict = field(default_factory=lambda: {"method": "one_to_one"})
    output_dir: Optional[str] = None
    max_workers: int = 1
    path_config: Optional[dict] = None


class Orchestrator:
    """Config-driven, step-by-step entry point for the rade_sr pipeline.

    Initiate with a :class:`RunConfig` plus the injected providers, then either
    run everything at once::

        orch = Orchestrator(
            config=RunConfig(output_dir="data/rade_sr/run_001"),
            portfolio_source=my_source,        # PortfolioSource  (SWAP POINT)
            market_data_client=my_client,      # MarketDataClient (SWAP POINT)
            pricer=my_pricer,                  # optional; defaults to OptionPricer
        )
        result = orch.run()

    ...or run the stages one at a time and inspect between them (best for
    debugging / a tight deadline)::

        orch.load_portfolio()    # [1] raw CSVs → normalised trades + risk factors
        orch.load_assets()       # [2] market data → one Asset per risk factor
        orch.generate_trades()   # [3] assets → elementary trade universe
        orch.compute_pnl()       # [4] price every trade across scenarios
        orch.build_clusters()    # [5] group into clusters → ReplicationJobs
        orch.write_artifacts()   #     write the rade_ml_pt handoff (needs output_dir)

    Each stage caches its result on the instance (``orch.inputs``, ``orch.assets``,
    ``orch.trades``, ``orch.pnls``, ``orch.jobs``) and requires the previous one.
    """

    def __init__(
        self,
        config: RunConfig,
        portfolio_source: Any,
        market_data_client: Any,
        pricer: Any = None,
        cluster_resolver: Any = None,
        resolvers: Optional[dict] = None,
        hooks: Optional[List[Callable[[ReplicationJob], None]]] = None,
    ) -> None:
        self.config = config
        self.portfolio_source = portfolio_source
        self.market_data_client = market_data_client
        self.pricer = pricer
        self._cluster_resolver_override = cluster_resolver
        self.resolvers = resolvers
        self.hooks = hooks

        # Stage outputs (populated as you go; inspect at any point).
        self.inputs: Any = None
        self.assets: Optional[dict] = None
        self.trades: Optional[dict] = None
        self.pnls: Optional[dict] = None
        self.jobs: Optional[List[ReplicationJob]] = None
        self.jobs_manifest_path: Optional[str] = None

        # Lazily built once the portfolio (hence risk factors) is known.
        self._pipeline: Optional[PreprocessingPipeline] = None
        self._pipeline_config: Optional[PipelineConfig] = None

    # ── Stage 1 ───────────────────────────────────────────────────────
    def load_portfolio(self) -> Any:
        """[1] Pull raw frames from the source and normalise them (ingestion)."""
        from src.rade_sr.replication.ingest import ingest_portfolio

        raw_attributes = self.portfolio_source.load_attributes()
        raw_pnl = self.portfolio_source.load_pnl()
        self.inputs = ingest_portfolio(raw_attributes, raw_pnl, resolvers=self.resolvers)
        logger.info(
            "Ingested portfolio: %d trades, %d risk factors",
            len(self.inputs.attributes), len(self.inputs.asset_config),
        )
        self._build_pipeline()
        return self.inputs

    # ── Stage 2 ───────────────────────────────────────────────────────
    def load_assets(self) -> dict:
        """[2] Load one self-contained Asset per risk factor (via the client)."""
        self._require("inputs", "load_portfolio")
        self.assets = self._pipeline.load_assets(self._pipeline_config)
        return self.assets

    # ── Stage 3 ───────────────────────────────────────────────────────
    def generate_trades(self) -> dict:
        """[3] Generate the elementary trade universe from the loaded assets."""
        self._require("assets", "load_assets")
        self.trades = self._pipeline.generate_trades(self.assets, self._pipeline_config)
        return self.trades

    # ── Stage 4 ───────────────────────────────────────────────────────
    def compute_pnl(self) -> dict:
        """[4] Price every elementary trade across all shock scenarios."""
        self._require("trades", "generate_trades")
        self.pnls = self._pipeline.compute_pnl(self.trades, self.assets)
        return self.pnls

    # ── Stage 5 ───────────────────────────────────────────────────────
    def build_clusters(self) -> List[ReplicationJob]:
        """[5] Resolve clusters and slice into ReplicationJobs."""
        self._require("pnls", "compute_pnl")
        self.jobs = self._pipeline.assemble_jobs(
            self.assets, self.trades, self.pnls, self._pipeline_config,
        )
        return self.jobs

    # ── Stage 6 (optional) ────────────────────────────────────────────
    def write_artifacts(self) -> str:
        """Attach target data per cluster and write the rade_ml_pt handoff."""
        self._require("jobs", "build_clusters")
        if not self.config.output_dir:
            raise ValueError("RunConfig.output_dir must be set to write artifacts")
        self.jobs_manifest_path = _write_artifacts(
            self.jobs, self.inputs, self.config.output_dir,
        )
        return self.jobs_manifest_path

    # ── One-shot ──────────────────────────────────────────────────────
    def run(self) -> "OrchestrationResult":
        """Run every stage in order (and write artifacts if output_dir is set)."""
        self.load_portfolio()
        self.load_assets()
        self.generate_trades()
        self.compute_pnl()
        self.build_clusters()
        if self.config.output_dir:
            self.write_artifacts()
        return OrchestrationResult(
            inputs=self.inputs, jobs=self.jobs,
            jobs_manifest_path=self.jobs_manifest_path,
        )

    # ── Internals ─────────────────────────────────────────────────────
    def _build_pipeline(self) -> None:
        """Construct the pipeline + PipelineConfig once the portfolio is known."""
        from src.rade_sr.replication.cluster_resolver import KeyGroupingClusterResolver

        cfg = self.config
        pricer = self.pricer
        if pricer is None:
            from src.rade_sr.pricing.option_pricer import OptionPricer
            pricer = OptionPricer()

        cluster_resolver = self._cluster_resolver_override or KeyGroupingClusterResolver(
            list(self.inputs.asset_config)
        )

        self._pipeline = PreprocessingPipeline(
            cluster_resolver=cluster_resolver,
            path_resolver=ConventionPathResolver(),
            trade_generator=RiskFactorAwareTradeGenerator(),
            pnl_engine=VectorisedPnLEngine(pricer, max_workers=cfg.max_workers),
            hooks=self.hooks,
            market_data_client=self.market_data_client,
        )

        root = str(Path(cfg.output_dir) / "clusters") if cfg.output_dir else "data/rade_sr/clusters"
        self._pipeline_config = PipelineConfig(
            asset_config=self.inputs.asset_config,
            trade_config=cfg.trade_config,
            cluster_config=cfg.cluster_config,
            path_config=cfg.path_config or _default_path_config(root=root),
        )

    def _require(self, attr: str, stage: str) -> None:
        if getattr(self, attr) is None:
            raise RuntimeError(
                f"Stage prerequisite missing: call {stage}() before this stage."
            )


def _default_path_config(root: str = "data/rade_sr/clusters") -> dict:
    """Default artifact path convention (paths are resolved, not yet written)."""
    return {
        "root": root,
        "target_pnl_file": "target_pnl.parquet",
        "target_attr_file": "target_attributes.pkl",
        "elem_pnl_file": "elementary_pnl.parquet",
        "elem_attr_file": "elementary_attributes.pkl",
    }


def build_pipeline(
    cluster_manager: Any,
    pricer: Any,
    max_workers: int = 4,
    hooks: Optional[List[Callable[[ReplicationJob], None]]] = None,
    market_data_client: Optional[Any] = None,
) -> PreprocessingPipeline:
    """Construct a fully wired PreprocessingPipeline around your ClusterManager.

    Parameters
    ----------
    cluster_manager : Any
        Your existing ClusterManager instance (wrapped by ClusterResolver).
    pricer : Any
        Batch pricer (e.g. ``OptionPricer``).
    max_workers : int
        PnL-engine concurrency.
    hooks : list[Callable] or None
        Post-assembly hooks for validation, logging, caching.
    market_data_client : MarketDataClient or None
        Injected market-data client (mock or real). Flows to the asset builders.

    Returns
    -------
    PreprocessingPipeline
        Ready to call ``.run(config)``.
    """
    return PreprocessingPipeline(
        cluster_resolver=ClusterResolver(cluster_manager),
        path_resolver=ConventionPathResolver(),
        trade_generator=RiskFactorAwareTradeGenerator(),
        pnl_engine=VectorisedPnLEngine(pricer, max_workers=max_workers),
        hooks=hooks,
        market_data_client=market_data_client,
    )


def run_from_source(
    portfolio_source: Any,
    market_data_client: Any,
    trade_config: dict,
    cluster_config: Optional[dict] = None,
    path_config: Optional[dict] = None,
    pricer: Any = None,
    cluster_resolver: Any = None,
    resolvers: Optional[dict] = None,
    max_workers: int = 1,
    hooks: Optional[List[Callable[[ReplicationJob], None]]] = None,
    output_dir: Optional[str] = None,
) -> "OrchestrationResult":
    """End-to-end run: ingest a portfolio source, then run the pipeline.

    This is the clean, modular entry point. Both external dependencies are
    injected as protocols:

    - ``portfolio_source`` : a :class:`~rade_sr.sources.portfolio_source.PortfolioSource`
      (mock, file, or your API) supplying the raw attribute + PnL frames.
    - ``market_data_client`` : a :class:`~rade_sr.sources.market_data.MarketDataClient`
      (mock or your API) supplying spot / curves / vol / shocks.

    Swapping mock → production is purely a matter of passing different
    implementations here; no pipeline code changes.

    Parameters
    ----------
    trade_config : dict
        Elementary trade generation grids (see GUIDE.md).
    cluster_config : dict or None
        Clustering config. Defaults to one-cluster-per-risk-factor.
    path_config : dict or None
        Artifact path config (used when artifact saving is enabled).
    pricer : Any or None
        Batch pricer; defaults to a fresh ``OptionPricer``.
    cluster_resolver : Any or None
        Override the cluster resolver. Defaults to a config-driven
        ``KeyGroupingClusterResolver`` over the ingested risk factors.
    resolvers : dict or None
        Risk-factor resolver registry for ingestion (defaults to FX).
    max_workers : int
        PnL-engine concurrency (1 = serial; safe default for threads/processes).
    output_dir : str or None
        If set, attach per-cluster target data, write the four artifact files per
        cluster + a ``jobs.pkl`` manifest under this directory (the
        ``rade_ml_pt.build_dataset`` handoff), and populate
        ``OrchestrationResult.jobs_manifest_path``.

    Returns
    -------
    OrchestrationResult
        Bundle of the ingested inputs and the assembled jobs.

    Notes
    -----
    This is a thin functional wrapper around :class:`Orchestrator`. Use the
    Orchestrator directly if you want to run stages one at a time.
    """
    config = RunConfig(
        trade_config=trade_config,
        cluster_config=cluster_config or {"method": "one_to_one"},
        output_dir=output_dir,
        max_workers=max_workers,
        path_config=path_config,
    )
    orch = Orchestrator(
        config=config,
        portfolio_source=portfolio_source,
        market_data_client=market_data_client,
        pricer=pricer,
        cluster_resolver=cluster_resolver,
        resolvers=resolvers,
        hooks=hooks,
    )
    return orch.run()


def _write_artifacts(jobs: List[ReplicationJob], inputs: Any, output_dir: str) -> str:
    """Attach target data per cluster and write the rade_ml_pt handoff artifacts."""
    from src.rade_sr.replication.artifacts import (
        attach_target_data,
        save_cluster_artifacts,
        save_jobs_manifest,
    )

    for job in jobs:
        attach_target_data(job, inputs)
        save_cluster_artifacts(job)

    manifest_path = Path(output_dir) / "jobs.pkl"
    save_jobs_manifest(jobs, manifest_path)
    return str(manifest_path)


def run_preprocessing(
    pipeline: PreprocessingPipeline,
    config: PipelineConfig,
) -> List[ReplicationJob]:
    """Execute the preprocessing pipeline and return assembled jobs.

    Parameters
    ----------
    pipeline : PreprocessingPipeline
        Constructed via ``build_pipeline()``.
    config : PipelineConfig
        Loaded via ``ConfigLoader.load_pipeline_config()``.

    Returns
    -------
    list[ReplicationJob]
        One fully assembled job per cluster.
    """
    logger.info("Starting preprocessing pipeline")
    jobs = pipeline.run(config)
    logger.info(
        "Pipeline complete: %d jobs, %d total trades",
        len(jobs), sum(j.n_trades for j in jobs),
    )
    return jobs


# ── Example configuration (reference only) ────────────────────────────

EXAMPLE_CONFIG = PipelineConfig(
    cluster_config={
        "n_clusters": 20,
        "method": "user_defined",
        "cluster_key": "desk",
        "cluster_key_values": {
            "cluster_0": ["FX_FLOW"],
            "cluster_1": ["FX_EXOTIC"],
        },
    },
    path_config={
        "root": "/data/replication",
        "target_pnl_file": "target_pnl.parquet",
        "target_attr_file": "target_attributes.parquet",
        "elem_pnl_file": "elem_pnl.parquet",
        "elem_attr_file": "elem_attributes.parquet",
    },
    asset_config={
        "eurusd": {"asset_class": "fx", "pair": "EURUSD"},
        "gbpusd": {"asset_class": "fx", "pair": "GBPUSD"},
        "eur_6m": {"asset_class": "rates", "currency": "EUR"},
    },
    trade_config={
        "notional": 1_000_000,
        "expiry_grid": [0.25, 0.5, 1.0, 2.0],
        "payoff_types": ["call", "put", "forward"],
    },
)
