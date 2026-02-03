"""
Hedging Environments

Model-specific hedging environments that implement the HedgingEnvironment protocol.

Available Environments
----------------------
- GBMHedgingEnv: Hedging under Geometric Brownian Motion dynamics
- HestonHedgingEnv: Hedging under Heston stochastic volatility (future)

All environments provide:
- Market simulation (price paths)
- P&L accounting
- Greeks computation
- Episode recording
"""

from src.deep_hedging.environments.gbm import GBMHedgingEnv, create_gbm_env

__all__ = [
    "GBMHedgingEnv",
    "create_gbm_env",
]
