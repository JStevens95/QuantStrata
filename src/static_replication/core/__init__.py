"""
Core Layer.

Contains the core logic for the RADE Static Replication model, including the main classes and functions that implement
the model's functionality. This layer is responsible for processing data, performing calculations and generating
outputs based on the RADE Static Replication approach.
"""
from src.static_replication.config.config import OrchestratorConfig
from src.static_replication.core.types import (
    ClusterPaths, ElementaryTrade, ElementaryTradePnL, PortfolioSlice, PortfolioData, ReplicationJob
)
from src.static_replication.core.protocols import (
    ClusterResolver, PathResolver, RiskFactorBuilder, TradeGenerator, PnLEngine
)
from src.static_replication.core.exceptions import (
    StaticReplicationError, UnknownAssetClassError, PipelineError, PnLComputationError
)

__all__ = [
    "ClusterPaths", "ElementaryTrade", "ElementaryTradePnL", "PortfolioSlice", "PortfolioData", "ReplicationJob",
    "ClusterResolver", "PathResolver", "RiskFactorBuilder", "TradeGenerator", "PnLEngine", "StaticReplicationError",
    "UnknownAssetClassError", "PipelineError", "PnLComputationError"
]