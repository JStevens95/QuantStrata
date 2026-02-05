"""
Historical Simulation Module.

Provides non-parametric scenario generation from historical data:

1. **Bootstrap**: Resample historical returns with replacement
2. **Filtered**: Volatility-adjust historical returns to current volatility
3. **Regime-Aware**: Select historical periods matching current market regime

Example
-------
>>> from src.marketdata.scenarios.historical import (
...     HistoricalSimulator,
...     BootstrapConfig,
...     FilteredConfig,
...     RegimeConfig,
... )
>>>
>>> # Simple bootstrap
>>> config = BootstrapConfig(
...     historical_returns=historical_returns,  # (n_assets, n_days) array
...     block_length=20,  # Block bootstrap with 20-day blocks
... )
>>> simulator = HistoricalSimulator(config)
>>> scenarios = simulator.generate(n_scenarios=10000, horizon=252)
"""

from src.marketdata.scenarios.historical.bootstrap import (
    BootstrapConfig,
    BlockBootstrap,
    StationaryBootstrap,
)
from src.marketdata.scenarios.historical.filtered import (
    FilteredConfig,
    FilteredHistorical,
)
from src.marketdata.scenarios.historical.simulator import (
    HistoricalSimulator,
    HistoricalConfig,
)

__all__ = [
    # Bootstrap
    "BootstrapConfig",
    "BlockBootstrap",
    "StationaryBootstrap",
    # Filtered
    "FilteredConfig",
    "FilteredHistorical",
    # Unified
    "HistoricalSimulator",
    "HistoricalConfig",
]
