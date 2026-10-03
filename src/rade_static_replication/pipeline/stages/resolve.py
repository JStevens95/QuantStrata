"""Stage 3 — resolve the risk-factor universe."""
from __future__ import annotations

from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import Portfolio, RiskFactorUniverse
from src.rade_static_replication.portfolio.resolution import resolve as _resolve


def resolve_universe(portfolio: Portfolio, config: OrchestratorConfig) -> RiskFactorUniverse:
    """Resolve every primary + dependency risk factor for the portfolio."""
    return _resolve(portfolio, config)
