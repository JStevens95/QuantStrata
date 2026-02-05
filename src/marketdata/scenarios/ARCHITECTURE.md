# Scenario Generation Architecture

## Overview

This module provides a comprehensive scenario generation framework for hedge fund risk management. It supports multiple generation methods that can be combined.

## Generation Methods

### 1. Monte Carlo Simulation (`timeseries/`)

**Status**: ✅ Implemented

Basic Monte Carlo with parametric dynamics:
- GBM (Geometric Brownian Motion)
- Heston (Stochastic Volatility)  
- Ornstein-Uhlenbeck (Mean-Reverting)
- Factor models

**Use when**: You have calibrated parameters and want parametric scenarios.

```python
from src.marketdata.scenarios.timeseries import TimeseriesGenerator, TimeseriesConfig
generator = TimeseriesGenerator(config)
dataset = generator.generate(seed=42)
```

### 2. Historical Simulation (`historical/`)

**Status**: ✅ Implemented

Bootstrap or replay historical returns:
- **Bootstrap**: Resample historical returns with replacement (block, stationary)
- **Filtered Historical**: Volatility-adjust historical returns to current vol (EWMA, GARCH)
- **Regime-Aware**: 🔲 To implement

**Use when**: You have sufficient historical data and want non-parametric scenarios.

```python
from src.marketdata.scenarios.historical import HistoricalSimulator, HistoricalConfig

config = HistoricalConfig(
    historical_returns=returns,  # (n_assets, n_obs) array
    asset_ids=["FX.SPOT.EUR", "FX.SPOT.GBP"],
    method="filtered_block",  # or "bootstrap", "block", "stationary", "filtered"
    current_volatility=np.array([0.08, 0.10]),
    block_length=20,
)
simulator = HistoricalSimulator(config)
dataset = simulator.generate_dataset(
    initial_values={"FX.SPOT.EUR": 1.10, "FX.SPOT.GBP": 1.25},
    n_scenarios=10000,
    horizon=252,
    start_date="2024-01-01",
)
```

### 3. Advanced Correlation (`correlation/`)

**Status**: ✅ Implemented

Beyond simple Cholesky:
- **Copulas**: Gaussian, Student-t (symmetric tails), Clayton (lower tail), Gumbel (upper tail)
- **Dynamic Correlation**: DCC-GARCH for time-varying correlation
- **Factor Models**: 🔲 To implement

**Use when**: Simple correlation is insufficient (e.g., tail dependence matters).

```python
from src.marketdata.scenarios.correlation import (
    StudentTCopula,
    ClaytonCopula,
    DynamicCorrelation,
    DCCConfig,
)
import numpy as np
from scipy.stats import norm

# Student-t copula for tail dependence
corr = np.array([[1.0, 0.6], [0.6, 1.0]])
copula = StudentTCopula(correlation=corr, df=4)
uniform_samples = copula.sample(n_scenarios=10000, seed=42)
print(f"Tail dependence: {copula.tail_dependence():.4f}")  # ~0.25

# Convert to correlated normals
z_correlated = norm.ppf(uniform_samples)

# Dynamic correlation (DCC-GARCH)
dcc_config = DCCConfig(historical_returns=returns, alpha=0.02, beta=0.95)
dcc = DynamicCorrelation(dcc_config)
current_corr = dcc.current_correlation
forecast_corr = dcc.forecast_correlation(h=20)
```

### 4. Advanced Dynamics (`advanced/`)

**Status**: 🔲 Adapters needed

Integrate existing models:
- **Merton Jump-Diffusion**: For crash risk / fat tails
- **SABR**: For rates/FX with realistic smile dynamics
- **Regime-Switching**: Multi-state Markov models
- **Neural SDE**: ML-learned dynamics

**Use when**: Standard dynamics don't capture market behavior.

### 5. Neural SDE / ML-Based (`advanced/neural_sde.py`)

**Status**: 🔲 Integration needed (model exists in `models/neural_sde/`)

ML-based path generation:
- Unconditional generation from trained model
- Conditional generation (pinned endpoints)
- Stress scenario generation

**Use when**: You have trained a Neural SDE on historical data.

## Decision Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                    Which Method to Use?                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Do you have sufficient historical data?                        │
│      │                                                          │
│      ├── YES ──► Historical Simulation                          │
│      │           (bootstrap, filtered, regime-aware)            │
│      │                                                          │
│      └── NO ───► Parametric Monte Carlo                         │
│                                                                 │
│  Do you need tail dependence?                                   │
│      │                                                          │
│      ├── YES ──► Copulas (t-copula, Clayton)                   │
│      │                                                          │
│      └── NO ───► Cholesky correlation                          │
│                                                                 │
│  Do you need fat tails / jumps?                                 │
│      │                                                          │
│      ├── YES ──► Jump-Diffusion (Merton) or                    │
│      │           Neural SDE (learned dynamics)                  │
│      │                                                          │
│      └── NO ───► GBM or Heston                                 │
│                                                                 │
│  Do you need regime changes?                                    │
│      │                                                          │
│      ├── YES ──► Regime-Switching or                           │
│      │           Regime-Aware Historical                        │
│      │                                                          │
│      └── NO ───► Single-regime dynamics                        │
│                                                                 │
│  Do you have a trained Neural SDE?                              │
│      │                                                          │
│      ├── YES ──► Neural SDE generation                         │
│      │           (can do conditional, stress scenarios)         │
│      │                                                          │
│      └── NO ───► Parametric dynamics                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Unified Interface

The `UnifiedScenarioGenerator` will provide a single interface:

```python
from src.marketdata.scenarios.unified import UnifiedScenarioGenerator

config = UnifiedScenarioConfig(
    method="monte_carlo",  # or "historical", "neural_sde"
    
    # Method-specific config
    monte_carlo=MonteCarloConfig(
        factors=[...],
        correlation=np.array([...]),
        dynamics="heston",  # or "gbm", "merton", "sabr"
    ),
    
    # OR
    historical=HistoricalConfig(
        data_source="returns_db",
        method="filtered",  # or "bootstrap", "regime_aware"
    ),
    
    # OR  
    neural_sde=NeuralSDEConfig(
        model_path="models/trained_nsde.pt",
        conditional=True,
    ),
    
    # Common
    start_date="2024-01-01",
    end_date="2024-12-31",
    n_scenarios=10000,
)

generator = UnifiedScenarioGenerator(config)
dataset = generator.generate(seed=42)
```

## Correlation Methods

### Cholesky (Current)
- Simple and fast
- Assumes Gaussian dependence
- Tail independence

### Gaussian Copula
- More flexible marginals
- Still has tail independence
- Good for normal market conditions

### t-Copula
- Captures tail dependence
- Better for stress scenarios
- Needs degrees of freedom parameter

### Clayton Copula
- Asymmetric tail dependence
- Strong lower tail (crash) dependence
- Good for equity portfolios

## Integration with Existing Models

The framework integrates with existing models in `src/models/`:

```python
# Example: Using Merton jump-diffusion
from src.models.jump_diffusion.merton import MertonDynamics, MertonParameters

params = MertonParameters(
    sigma=0.2,      # Diffusion vol
    lambda_=0.5,    # Jump intensity
    mu_j=-0.1,      # Mean jump (negative = crash)
    sigma_j=0.2,    # Jump vol
)

# Will be wrapped by MertonAdapter for TimeseriesGenerator
```

## Example: Multi-Method Comparison

```python
import numpy as np
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator, TimeseriesConfig, RiskFactorSpec, GBMDynamicsSpec
)

# Same config, different methods
factor = RiskFactorSpec(
    market_id=MarketId("EQ", "SPOT", "SPX"),
    initial_value=4500.0,
    dynamics=GBMDynamicsSpec(drift=0.05, vol=0.20),
)

config = TimeseriesConfig(
    factors=[factor],
    correlation=np.array([[1.0]]),
    start_date="2024-01-01",
    end_date="2024-12-31",
    freq="D",
    n_scenarios=10000,
)

# Method 1: Basic Monte Carlo (current)
mc_generator = TimeseriesGenerator(config)
mc_dataset = mc_generator.generate(seed=42)

# Method 2: With jump-diffusion (to implement)
# jd_generator = JumpDiffusionGenerator(config, merton_params)
# jd_dataset = jd_generator.generate(seed=42)

# Method 3: Neural SDE (to integrate)
# nsde_generator = NeuralSDEGenerator(trained_model)
# nsde_dataset = nsde_generator.generate(seed=42)

# Compare terminal distributions
mc_terminal = mc_dataset.panels[factor.market_id].data[-1, :]
print(f"MC: mean={mc_terminal.mean():.0f}, std={mc_terminal.std():.0f}")
```
