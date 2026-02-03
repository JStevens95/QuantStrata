"""
Deep Hedging Core Components

Generic components that are independent of the specific market model:
- Transaction cost models
- Risk measures
- Types and configurations
- Protocols for hedging environments
"""

from src.deep_hedging.core.costs import (
    TransactionCostModel,
    ProportionalCost,
    FixedCost,
    MarketImpactCost,
    CombinedCost,
)
from src.deep_hedging.core.risk_measures import (
    RiskMeasure,
    VarianceRisk,
    CVaRRisk,
    EntropicRisk,
    MeanVarianceRisk,
)
from src.deep_hedging.core.types import (
    HedgingConfig,
    HedgingState,
    HedgingResult,
    HedgingEpisode,
)
from src.deep_hedging.core.protocols import HedgingEnvironment

__all__ = [
    # Costs
    "TransactionCostModel",
    "ProportionalCost",
    "FixedCost",
    "MarketImpactCost",
    "CombinedCost",
    # Risk measures
    "RiskMeasure",
    "VarianceRisk",
    "CVaRRisk",
    "EntropicRisk",
    "MeanVarianceRisk",
    # Types
    "HedgingConfig",
    "HedgingState",
    "HedgingResult",
    "HedgingEpisode",
    # Protocols
    "HedgingEnvironment",
]
