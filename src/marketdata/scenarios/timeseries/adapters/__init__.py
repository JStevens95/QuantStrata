"""
Dynamics adapters for time series generation.

This module provides adapters that transform correlated Gaussian shocks
into risk factor paths using various stochastic process models.

Protocol
--------
All adapters implement the DynamicsAdapter protocol:
    - simulate(initial_value, n_time, n_scenarios, shocks, dt) -> np.ndarray

Available Adapters
------------------
- GBMAdapter: Geometric Brownian Motion
- HestonAdapter: Heston stochastic volatility
- OUAdapter: Ornstein-Uhlenbeck (mean-reverting)
- FactorAdapter: Factor model dynamics
"""

from src.marketdata.scenarios.timeseries.adapters.base import DynamicsAdapter
from src.marketdata.scenarios.timeseries.adapters.gbm import GBMAdapter
from src.marketdata.scenarios.timeseries.adapters.heston import HestonAdapter
from src.marketdata.scenarios.timeseries.adapters.ou import OUAdapter
from src.marketdata.scenarios.timeseries.adapters.factor import FactorAdapter

__all__ = [
    "DynamicsAdapter",
    "GBMAdapter",
    "HestonAdapter",
    "OUAdapter",
    "FactorAdapter",
]
