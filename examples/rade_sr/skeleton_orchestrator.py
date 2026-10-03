"""
rade_sr — SKELETON / BLUEPRINT (read this top-to-bottom)
========================================================

This single file is a *teaching skeleton* of the whole pipeline. It is NOT the
real implementation — it shows the SHAPE: the stages, how they wire together,
and exactly where you can swap a component (mock ↔ your firm's systems).

Read the data-flow first, then the protocols (the swap points), then the
Orchestrator at the bottom.

DATA FLOW (what goes in, what comes out)
----------------------------------------
    raw portfolio CSVs ─┐
                        ├─► [1] INGEST ───► normalised trades + risk-factor list
    market-data API ────┘                          │
                                                    ▼
                                  [2] LOAD ASSETS  (one per risk factor:
                                       spot, vol, IR curves, shocks)
                                                    │
                                                    ▼
                                  [3] GENERATE elementary trades
                                       (vanillas / digitals / forwards)
                                                    │
                                                    ▼
                                  [4] PRICE → PnL matrix (trades × scenarios)
                                                    │
                                                    ▼
                                  [5] CLUSTER + WRITE artifacts
                                       (4 files/cluster + jobs.pkl)
                                                    │
                                                    ▼
                                        rade_ml_pt.build_dataset()

WHERE EACH SKELETON PIECE REALLY LIVES
--------------------------------------
    PortfolioSource     → src/rade_sr/sources/portfolio_source.py
    MarketDataClient    → src/rade_sr/sources/market_data.py  (mock)
                          src/rade_sr/api/client.py            (real, YOU wire)
    ingest              → src/rade_sr/replication/ingest.py
    RiskFactorBuilder   → src/rade_sr/replication/builders.py + assets/fx.py, rates.py
    TradeGenerator      → src/rade_sr/replication/trade_generator.py + instruments/
    Pricer / PnLEngine  → src/rade_sr/pricing/option_pricer.py + replication/pnl_engine.py
    ClusterResolver     → src/rade_sr/replication/cluster_resolver.py
    ArtifactWriter      → src/rade_sr/replication/artifacts.py
    Orchestrator        → src/rade_sr/replication/orchestrator.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


# ═══════════════════════════════════════════════════════════════════════════
#  THE SWAP POINTS  (Protocols = interchangeable components)
#  Anything below that is a `Protocol` can be replaced with your own class,
#  as long as it has the same methods. That is the whole extensibility story.
# ═══════════════════════════════════════════════════════════════════════════

class PortfolioSource(Protocol):
    """WHERE YOUR PORTFOLIO COMES FROM.  Swap: Mock ↔ File ↔ your API."""
    def load_attributes(self) -> Any: ...   # raw trade attributes (DataFrame)
    def load_pnl(self) -> Any: ...           # raw scenario PnL       (DataFrame)


class MarketDataClient(Protocol):
    """WHERE MARKET DATA COMES FROM.  Swap: Mock ↔ your InternalAPIClient.

    These are the ONLY methods that touch your firm's systems. Implement them
    and the whole pipeline runs on real data. (FX shown; IR analogous.)
    """
    def get_fx_spot(self, pair: str, cob: Any = None) -> dict: ...
    def get_fx_atm_vol(self, pair: str, cob: Any = None) -> dict: ...
    def get_fx_smile_vol(self, pair: str, cob: Any = None) -> dict: ...
    def get_fx_shocks(self, pair: str, cob: Any = None) -> dict: ...
    # ... plus get_fx_forward_points + 5 IR methods (see api/client.py)


class RiskFactorBuilder(Protocol):
    """Turns one risk-factor config into a loaded Asset. Registered per asset class."""
    def build(self, factor_id: str, factor_config: dict) -> "Asset": ...


class TradeGenerator(Protocol):
    """Turns loaded assets + grids into the elementary trade universe."""
    def generate(self, assets: Dict[str, "Asset"], trade_config: dict) -> Dict[str, list]: ...


class Pricer(Protocol):
    """Prices a batch of trades across all scenarios. Swap: rade_sr ↔ your pricer."""
    def price_trades(self, trades: list, asset: "Asset") -> Any: ...  # → (n_trades, n_scen)


class ClusterResolver(Protocol):
    """Decides how risk factors group into clusters. Swap: key-grouping ↔ your ClusterManager."""
    def resolve(self, cluster_config: dict) -> Dict[str, List[str]]: ...  # cluster_id → [factors]


# ═══════════════════════════════════════════════════════════════════════════
#  THE DATA OBJECTS (typed contracts passed between stages)
#  Real versions: src/rade_sr/core/types.py
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Asset:
    """One risk factor's market data (spot, vol surface, IR curves, shocks)."""
    factor_id: str
    # spot, vol_surface, domestic_ir, foreign_ir, shocks ... (see assets/fx.py)


@dataclass
class ElementaryTrade:
    """One simple replicating instrument (e.g. EURUSD 1Y call @ 1.05)."""
    trade_id: str
    factor_id: str
    payoff_type: str          # "vanilla" | "digital" | "forward"
    parameters: dict = field(default_factory=dict)


@dataclass
class ReplicationJob:
    """One cluster's complete output — what gets written to disk."""
    cluster_id: str
    elementary_trades: Dict[str, List[ElementaryTrade]]
    trade_pnls: Dict[str, Any]            # elementary PnL matrices
    target_pnl: Any = None                # real-trade PnL for this cluster
    target_attributes: Any = None


# ═══════════════════════════════════════════════════════════════════════════
#  THE STAGES (concrete skeletons — bodies elided with `...`)
#  Each maps 1:1 to a real module. This shows the *interface*, not the maths.
# ═══════════════════════════════════════════════════════════════════════════

def ingest(raw_attributes: Any, raw_pnl: Any) -> "PortfolioInputs":
    """[1] Normalise raw CSVs → one row/trade + {factor_id: config}.
    REAL: replication/ingest.py :: ingest_portfolio()"""
    ...


@dataclass
class PortfolioInputs:
    attributes: Any                       # normalised, one row per trade
    asset_config: Dict[str, dict]         # {"EURUSD": {"asset_class": "fx", ...}}
    target_pnl: Any                       # real-trade PnL (trades × scenarios)
    risk_factor_by_trade: Any             # trade_id → factor_id


class RiskFactorAwareTradeGenerator:
    """[3] REAL: replication/trade_generator.py"""
    def generate(self, assets: Dict[str, Asset], trade_config: dict) -> Dict[str, list]:
        ...


class VectorisedPnLEngine:
    """[4] Calls the injected Pricer once per factor group.
    REAL: replication/pnl_engine.py"""
    def __init__(self, pricer: Pricer, max_workers: int = 1) -> None:
        self.pricer = pricer
    def compute_batch(self, trades: Dict[str, list], assets: Dict[str, Asset]) -> dict:
        ...


def write_artifacts(jobs: List[ReplicationJob], output_dir: str) -> str:
    """[5] Write 4 files/cluster + jobs.pkl. REAL: replication/artifacts.py"""
    ...


# ═══════════════════════════════════════════════════════════════════════════
#  THE PIPELINE  (runs the 5 stages; depends ONLY on the protocols above)
#  REAL: src/rade_sr/replication/pipeline.py
# ═══════════════════════════════════════════════════════════════════════════

class PreprocessingPipeline:
    def __init__(
        self,
        cluster_resolver: ClusterResolver,
        trade_generator: TradeGenerator,
        pnl_engine: VectorisedPnLEngine,
        market_data_client: MarketDataClient,    # ← injected, flows to builders
        registry: Any,                            # asset_class → RiskFactorBuilder
    ) -> None:
        self.cluster_resolver = cluster_resolver
        self.trade_generator = trade_generator
        self.pnl_engine = pnl_engine
        self.market_data_client = market_data_client
        self.registry = registry

    def run(self, asset_config: Dict[str, dict], trade_config: dict,
            cluster_config: dict) -> List[ReplicationJob]:
        # [2] load one Asset per risk factor (builder gets the market-data client)
        assets: Dict[str, Asset] = {}
        for factor_id, cfg in asset_config.items():
            builder = self.registry.get(cfg["asset_class"])
            assets[factor_id] = builder.build(factor_id, {**cfg,
                                              "market_data_client": self.market_data_client})

        # [3] generate elementary trades, [4] price them
        trades = self.trade_generator.generate(assets, trade_config)
        pnls = self.pnl_engine.compute_batch(trades, assets)

        # [5] group into clusters → one ReplicationJob each
        cluster_map = self.cluster_resolver.resolve(cluster_config)
        jobs = [self._build_job(cid, factors, trades, pnls)
                for cid, factors in cluster_map.items()]
        return jobs

    def _build_job(self, cluster_id, factors, trades, pnls) -> ReplicationJob:
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  THE ORCHESTRATOR  (config in, artifacts out — the single entry point)
#  REAL: src/rade_sr/replication/orchestrator.py
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    """Your single 'model configuration'. All knobs, no environment specifics."""
    trade_config: dict
    cluster_config: dict = field(default_factory=lambda: {"method": "one_to_one"})
    output_dir: Optional[str] = None
    max_workers: int = 1


class Orchestrator:
    """Wires everything and runs it. The 3 constructor args are your SWAP POINTS."""
    def __init__(
        self,
        config: RunConfig,
        portfolio_source: PortfolioSource,     # ← SWAP: Mock / File / your API
        market_data_client: MarketDataClient,  # ← SWAP: Mock / your InternalAPIClient
        pricer: Optional[Pricer] = None,        # ← SWAP: rade_sr's / your own (optional)
    ) -> None:
        self.config = config
        self.portfolio_source = portfolio_source
        self.market_data_client = market_data_client
        self.pricer = pricer

    def run(self):
        cfg = self.config

        # [1] INGEST — pull raw frames from whatever source, normalise them
        inputs = ingest(self.portfolio_source.load_attributes(),
                        self.portfolio_source.load_pnl())

        # build the pipeline with the chosen (interchangeable) components
        pipeline = PreprocessingPipeline(
            cluster_resolver=_make_cluster_resolver(cfg.cluster_config, inputs),
            trade_generator=RiskFactorAwareTradeGenerator(),
            pnl_engine=VectorisedPnLEngine(self.pricer or _default_pricer(),
                                           max_workers=cfg.max_workers),
            market_data_client=self.market_data_client,
            registry=_builder_registry(),
        )

        # [2]–[5] run the stages
        jobs = pipeline.run(inputs.asset_config, cfg.trade_config, cfg.cluster_config)

        # [5] write the rade_ml_pt handoff (if an output dir was given)
        manifest = write_artifacts(jobs, cfg.output_dir) if cfg.output_dir else None
        return jobs, manifest


# helpers that pick the concrete implementations (the "factory" layer) -----------
def _make_cluster_resolver(cluster_config, inputs): ...   # key-grouping or your ClusterManager
def _default_pricer(): ...                                # rade_sr's OptionPricer
def _builder_registry(): ...                              # {"fx": FXBuilder, "rates": IRBuilder}


# ═══════════════════════════════════════════════════════════════════════════
#  HOW YOU'D WIRE IT  (this is the only code you write per environment)
# ═══════════════════════════════════════════════════════════════════════════

def wiring_example() -> None:
    # --- pseudocode; see examples/rade_sr/01_run_pipeline_mock.py for a runnable one ---
    #
    # from src.rade_sr.sources import FilePortfolioSource
    # from src.rade_sr.api.client import InternalAPIClient   # you implement 10 methods
    #
    # orch = Orchestrator(
    #     config=RunConfig(trade_config={...}, output_dir="data/rade_sr/run_001"),
    #     portfolio_source=FilePortfolioSource("attrs.csv", "pnl.csv"),   # SWAP POINT
    #     market_data_client=InternalAPIClient(base_url=...),             # SWAP POINT
    # )
    # jobs, manifest_path = orch.run()
    # # → feed manifest_path (jobs.pkl) to rade_ml_pt.build_dataset()
    ...
