"""Stage 2 — normalise + validate the portfolio."""
from __future__ import annotations

from typing import List, Tuple

from src.rade_static_replication.domain.contracts import Portfolio, RawPortfolio
from src.rade_static_replication.portfolio.normalise import normalise as _normalise
from src.rade_static_replication.portfolio.validate import validate as _validate


def normalise_portfolio(raw: RawPortfolio) -> Tuple[Portfolio, List[str]]:
    """Collapse the exploded export to one row per trade and validate it."""
    portfolio = _normalise(raw)
    warnings = _validate(portfolio)
    return portfolio, warnings
