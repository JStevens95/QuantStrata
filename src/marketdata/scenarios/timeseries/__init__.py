"""
Monte Carlo Time Series Generation for Risk Factor Simulation.

This module provides production-grade time series generation for risk factors,
enabling realistic scenario simulation for hedge fund risk management workflows.

Location in Architecture
------------------------
This module lives under `marketdata/scenarios/` because:
1. It produces `MarketDataset` - a market data structure
2. It builds on the existing `ScenarioDriver` correlation framework
3. It's consumed by `risk/scenarios/` for portfolio PnL calculation

Key Components
--------------
- TimeseriesGenerator: Main orchestrator for generating correlated risk factor paths
- RiskFactorSpec: Configuration for individual risk factors
- TimeseriesConfig: Top-level configuration for time series generation
- GenerationResult: Container for raw generated paths

Supported Dynamics
------------------
- GBM: Geometric Brownian Motion for spot prices (FX, equity, commodity)
- Heston: Stochastic volatility for more realistic vol dynamics
- OrnsteinUhlenbeck: Mean-reverting process for rates, spreads, vol factors
- Factor: Factor model dynamics with loadings (for curves, surfaces)

Mathematical Framework
----------------------
Each risk factor follows a stochastic differential equation:

**GBM** (Geometric Brownian Motion):
    dS_t = μ S_t dt + σ S_t dW_t

**Heston** (Stochastic Volatility):
    dS_t = μ S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
    Corr(dW^S, dW^V) = ρ

**Ornstein-Uhlenbeck** (Mean-Reverting):
    dX_t = κ(θ - X_t) dt + σ dW_t

**Factor** (Factor Model with Loadings):
    dF_t = κ(θ - F_t) dt + σ dW_t
    ΔR(τ) = λ(τ) × F_t

Correlation Handling
--------------------
Cross-factor correlation is handled via Cholesky decomposition:
1. Generate independent shocks Z[t,s,f] ~ N(0,I)
2. Apply correlation: Z_corr = Z @ L^T where Σ = LL^T

Example
-------
>>> import numpy as np
>>> from src.marketdata.scenarios.timeseries import (
...     TimeseriesGenerator,
...     TimeseriesConfig,
...     RiskFactorSpec,
...     GBMDynamicsSpec,
...     HestonDynamicsSpec,
... )
>>> from src.marketdata.core.ids import MarketId
>>>
>>> # Define correlated risk factors
>>> factors = [
...     RiskFactorSpec(
...         market_id=MarketId("FX", "SPOT", "EURUSD"),
...         initial_value=1.08,
...         dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
...     ),
...     RiskFactorSpec(
...         market_id=MarketId("EQ", "SPOT", "SPX"),
...         initial_value=4500.0,
...         dynamics=HestonDynamicsSpec(
...             drift=0.05, kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho_internal=-0.7
...         ),
...     ),
... ]
>>>
>>> # 30% correlation between EUR/USD and S&P 500
>>> correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
>>>
>>> config = TimeseriesConfig(
...     factors=factors,
...     correlation=correlation,
...     start_date="2024-01-01",
...     end_date="2024-12-31",
...     freq="D",
...     n_scenarios=10000,
... )
>>>
>>> generator = TimeseriesGenerator(config)
>>> dataset = generator.generate(seed=42)
>>>
>>> # Get market snapshot
>>> market = dataset.snapshot(time_idx=100, scenario_idx=0)
>>> fx_spot = market.quote(MarketId("FX", "SPOT", "EURUSD"))
"""

from src.marketdata.scenarios.timeseries.config import (
    TimeseriesConfig,
    RiskFactorSpec,
    GBMDynamicsSpec,
    HestonDynamicsSpec,
    OUDynamicsSpec,
    FactorDynamicsSpec,
    DynamicsSpec,
)
from src.marketdata.scenarios.timeseries.generator import TimeseriesGenerator, GenerationResult
from src.marketdata.scenarios.timeseries.adapters import (
    DynamicsAdapter,
    GBMAdapter,
    HestonAdapter,
    OUAdapter,
    FactorAdapter,
)

__all__ = [
    # Main classes
    "TimeseriesGenerator",
    "GenerationResult",
    # Configuration
    "TimeseriesConfig",
    "RiskFactorSpec",
    # Dynamics specifications
    "GBMDynamicsSpec",
    "HestonDynamicsSpec",
    "OUDynamicsSpec",
    "FactorDynamicsSpec",
    "DynamicsSpec",
    # Adapters
    "DynamicsAdapter",
    "GBMAdapter",
    "HestonAdapter",
    "OUAdapter",
    "FactorAdapter",
]
