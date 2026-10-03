"""Stage 1 — load the raw portfolio via the portfolio client."""
from __future__ import annotations

from src.rade_static_replication.clients.base import PortfolioClient
from src.rade_static_replication.domain.contracts import RawPortfolio


def load_portfolio(client: PortfolioClient, cob_date: str) -> RawPortfolio:
    """Fetch the untouched portfolio payload from the client."""
    return client.load(cob_date)
