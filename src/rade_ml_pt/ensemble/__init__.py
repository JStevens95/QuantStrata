"""
Ensemble model: combine N trained members with trade routing and aggregation.

Public API::

    from rade_ml_pt.ensemble import (
        EnsembleConfig,
        EnsembleModel,
        EnsembleBuilder,
        EnsembleRegistry,
        TradeRouter,
    )
"""
from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.model import EnsembleModel
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.rade_ml_pt.ensemble.router import TradeRouter

__all__ = [
    "EnsembleConfig",
    "EnsembleModel",
    "EnsembleBuilder",
    "EnsembleRegistry",
    "TradeRouter",
]
