"""
Data-source seams for rade_sr.

This package isolates the two places the pipeline talks to the outside world, so
mock data can be swapped for real API calls by implementing a single protocol:

  - :class:`MarketDataClient`  — spot / curves / vol / shocks per risk factor.
    The asset ``_fetch_*`` hooks delegate to this. Swap = implement the protocol.
  - :class:`PortfolioSource`   — the raw portfolio attribute + scenario-PnL frames.
    Step 0 ingestion consumes whatever this returns. Swap = implement the protocol.

Mock implementations (:class:`MockMarketDataClient`, :class:`MockPortfolioSource`)
let the whole module run offline; ``InternalAPIClient`` / ``ApiPortfolioSource``
are the real-environment wiring stubs.
"""
from __future__ import annotations

from .market_data import MarketDataClient, MockMarketDataClient
from .portfolio_source import (
    ApiPortfolioSource,
    FilePortfolioSource,
    MockPortfolioSource,
    PortfolioSource,
    make_portfolio_source,
)

__all__ = [
    "MarketDataClient",
    "MockMarketDataClient",
    "PortfolioSource",
    "MockPortfolioSource",
    "FilePortfolioSource",
    "ApiPortfolioSource",
    "make_portfolio_source",
]
