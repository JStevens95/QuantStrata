"""
Core layer — innermost. Types, protocols, and exceptions.

This module has ZERO imports from any other rade_sr layer.
If a type is needed here that currently lives elsewhere, move it here.
"""
from src.rade_sr.core.types import (
    ClusterPaths,
    ElementaryTrade,
    ElementaryTradePnL,
    PortfolioData,
    PortfolioSlice,
    ReplicationJob,
)
from src.rade_sr.core.protocols import (
    ClusterResolver,
    PathResolver,
    RiskFactorBuilder,
    TradeGenerator,
    PnLEngine,
)
from src.rade_sr.core.exceptions import (
    StaticReplicationError,
    UnknownAssetClassError,
    PipelineError,
    PnLComputationError,
)

__all__ = [
    "ClusterPaths",
    "ElementaryTrade",
    "ElementaryTradePnL",
    "PortfolioData",
    "PortfolioSlice",
    "ReplicationJob",
    "ClusterResolver",
    "PathResolver",
    "RiskFactorBuilder",
    "TradeGenerator",
    "PnLEngine",
    "StaticReplicationError",
    "UnknownAssetClassError",
    "PipelineError",
    "PnLComputationError",
]
