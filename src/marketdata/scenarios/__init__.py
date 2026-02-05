"""
Market Data Scenarios Module.

This module provides scenario generation capabilities for risk management:

Submodules
----------
- shocks: Deterministic shock objects (SpotShock, VolShock, ParallelRateShock)
- generator: ScenarioDriver for correlated Gaussian shocks
- interfaces: Protocols (MarketView, ScenarioShock, ScenarioPack)
- runner: Basic scenario runner utilities
- timeseries: Monte Carlo time series generation for risk factors

Architecture
------------
This module sits in marketdata/ because it produces market data objects:
- Shocks transform Market → MarketView
- TimeseriesGenerator produces MarketDataset

The risk/scenarios/ module consumes these for portfolio PnL calculation.

Example: Deterministic Shocks
-----------------------------
>>> from src.marketdata.scenarios.shocks import SpotShock
>>> from src.marketdata.core.ids import MarketId
>>>
>>> shock = SpotShock(
...     name="spot_down_10",
...     spot_id=MarketId("FX", "SPOT", "EURUSD"),
...     bump=-0.10,
...     bump_mode="relative",
... )
>>> shocked_market = shock.apply(base_market)

Example: Monte Carlo Time Series
--------------------------------
>>> from src.marketdata.scenarios.timeseries import (
...     TimeseriesGenerator,
...     TimeseriesConfig,
...     RiskFactorSpec,
...     GBMDynamicsSpec,
... )
>>>
>>> config = TimeseriesConfig(
...     factors=[RiskFactorSpec(...)],
...     correlation=np.eye(1),
...     start_date="2024-01-01",
...     end_date="2024-12-31",
...     freq="D",
...     n_scenarios=10000,
... )
>>> dataset = TimeseriesGenerator(config).generate(seed=42)
"""

# Re-export key classes for convenience
from src.marketdata.scenarios.shocks import (
    SpotShock,
    VolShock,
    ParallelRateShock,
    CompositeShock,
)
from src.marketdata.scenarios.interfaces import (
    MarketView,
    ScenarioShock,
    ScenarioPack,
)
from src.marketdata.scenarios.generator import (
    ScenarioSpec,
    ScenarioDriver,
)

# Time series generation (Monte Carlo)
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator,
    TimeseriesConfig,
    RiskFactorSpec,
    GenerationResult,
    GBMDynamicsSpec,
    HestonDynamicsSpec,
    OUDynamicsSpec,
    FactorDynamicsSpec,
)

__all__ = [
    # Shock objects
    "SpotShock",
    "VolShock",
    "ParallelRateShock",
    "CompositeShock",
    # Protocols
    "MarketView",
    "ScenarioShock",
    "ScenarioPack",
    # Correlated shock driver
    "ScenarioSpec",
    "ScenarioDriver",
    # Time series generation
    "TimeseriesGenerator",
    "TimeseriesConfig",
    "RiskFactorSpec",
    "GenerationResult",
    "GBMDynamicsSpec",
    "HestonDynamicsSpec",
    "OUDynamicsSpec",
    "FactorDynamicsSpec",
]
