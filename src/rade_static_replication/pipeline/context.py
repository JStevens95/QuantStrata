"""
Run context — the orchestrator's working ledger.

A mutable record of one run: the config, the clients/registry, and each stage's output
contract. Every stage reads the contracts it needs and writes exactly one. Keeping
state here (rather than threading many returns) makes partial runs, inspection, and
re-slicing trivial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.clients.base import MarketDataClient, PortfolioClient
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    BasePriceSet,
    ClusterSet,
    ElementaryUniverse,
    FactorDataSet,
    PnLResult,
    Portfolio,
    RawPortfolio,
    RiskFactorUniverse,
)


@dataclass
class RunContext:
    """Mutable per-run state shared across stages."""
    config: OrchestratorConfig
    portfolio_client: PortfolioClient
    market_data_client: MarketDataClient
    registry: Registry

    raw_portfolio: Optional[RawPortfolio] = None
    portfolio: Optional[Portfolio] = None
    warnings: List[str] = field(default_factory=list)
    universe: Optional[RiskFactorUniverse] = None
    factor_data: Optional[FactorDataSet] = None
    elementary: Optional[ElementaryUniverse] = None
    base_prices: Optional[BasePriceSet] = None
    pnl: Optional[PnLResult] = None
    clusters: Optional[ClusterSet] = None

    timings_ms: dict = field(default_factory=dict)

    @property
    def cob_date(self) -> str:
        return self.config.cob_date
