"""
Hedging Environments

Model-specific hedging environments that implement the HedgingEnvironment protocol.

Available Environments
----------------------
- GBMHedgingEnv: Hedging under Geometric Brownian Motion dynamics
- MultiAssetHedgingEnv: Multi-asset portfolio hedging with correlations
- HistoricalHedgingEnv: Model-agnostic hedging on historical data

All environments provide:
- Market simulation (price paths)
- P&L accounting
- Greeks computation
- Episode recording

Usage:
    from src.deep_hedging.environments import (
        GBMHedgingEnv,
        MultiAssetHedgingEnv,
        HistoricalHedgingEnv,
    )
    
    # Single asset GBM
    env = GBMHedgingEnv(config=...)
    
    # Multi-asset with correlations
    env = MultiAssetHedgingEnv(config=MultiAssetHedgingConfig(n_assets=3))
    
    # Historical data
    env = HistoricalHedgingEnv(market_data=historical_data)
"""

from src.deep_hedging.environments.gbm import GBMHedgingEnv, create_gbm_env
from src.deep_hedging.environments.multi_asset import (
    MultiAssetHedgingEnv,
    MultiAssetHedgingConfig,
)
from src.deep_hedging.environments.historical import (
    HistoricalHedgingEnv,
    HistoricalHedgingConfig,
    HistoricalMarketDataInterface,
)

__all__ = [
    # GBM (single asset)
    "GBMHedgingEnv",
    "create_gbm_env",
    # Multi-asset
    "MultiAssetHedgingEnv",
    "MultiAssetHedgingConfig",
    # Historical
    "HistoricalHedgingEnv",
    "HistoricalHedgingConfig",
    "HistoricalMarketDataInterface",
]
