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
- historical: Non-parametric historical simulation (bootstrap, filtered)

Architecture
------------
This module sits in marketdata/ because it produces market data objects:
- Shocks transform Market → MarketView
- TimeseriesGenerator produces MarketDataset (parametric Monte Carlo)
- HistoricalSimulator produces MarketDataset (non-parametric resampling)

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

Example: Historical Simulation
------------------------------
>>> from src.marketdata.scenarios.historical import (
...     HistoricalSimulator,
...     HistoricalConfig,
... )
>>>
>>> config = HistoricalConfig(
...     historical_returns=returns,  # (n_assets, n_obs) array
...     asset_ids=["FX.SPOT.EUR", "FX.SPOT.GBP"],
...     method="filtered_block",
...     current_volatility=np.array([0.08, 0.10]),
...     block_length=20,
... )
>>> simulator = HistoricalSimulator(config)
>>> dataset = simulator.generate_dataset(
...     initial_values={"FX.SPOT.EUR": 1.10, "FX.SPOT.GBP": 1.25},
...     n_scenarios=10000,
...     horizon=252,
...     start_date="2024-01-01",
... )
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

# Historical simulation
from src.marketdata.scenarios.historical import (
    HistoricalSimulator,
    HistoricalConfig,
    BootstrapConfig,
    BlockBootstrap,
    StationaryBootstrap,
    FilteredConfig,
    FilteredHistorical,
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
    # Time series generation (Monte Carlo)
    "TimeseriesGenerator",
    "TimeseriesConfig",
    "RiskFactorSpec",
    "GenerationResult",
    "GBMDynamicsSpec",
    "HestonDynamicsSpec",
    "OUDynamicsSpec",
    "FactorDynamicsSpec",
    # Historical simulation
    "HistoricalSimulator",
    "HistoricalConfig",
    "BootstrapConfig",
    "BlockBootstrap",
    "StationaryBootstrap",
    "FilteredConfig",
    "FilteredHistorical",
]
