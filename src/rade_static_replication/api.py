"""
Public facade — one call to run the whole pipeline.

::

    from src.rade_static_replication import run, load_config
    from src.rade_static_replication.clients.mock import MockPortfolioClient, MockMarketDataClient

    ctx = run(load_config("configs/orchestrator.yaml"),
              MockPortfolioClient(), MockMarketDataClient())

Returns the populated :class:`RunContext`; artifacts are written to the configured
run directory. For step-by-step control, construct :class:`Orchestrator` directly.
"""
from __future__ import annotations

from typing import Optional

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.clients.base import MarketDataClient, PortfolioClient
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.pipeline.context import RunContext
from src.rade_static_replication.pipeline.orchestrator import Orchestrator


def run(
    config: OrchestratorConfig,
    portfolio_client: PortfolioClient,
    market_data_client: MarketDataClient,
    registry: Optional[Registry] = None,
) -> RunContext:
    """Run the full static-replication preprocessing pipeline."""
    return Orchestrator(config, portfolio_client, market_data_client, registry).run()
