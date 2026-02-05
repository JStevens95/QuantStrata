# Scenario Generation Architecture

This document describes the two-tier scenario generation architecture in QuantStrata,
designed to support both rapid prototyping and production hedge fund workflows.

## Overview

QuantStrata provides two complementary approaches to scenario generation:

| Generator | Use Case | Output | Complexity |
|-----------|----------|--------|------------|
| `TimeseriesGenerator` | Single factors, quick prototyping | `[T, S]` scalars | Simple |
| `FactorModelGenerator` | Full term structures, production VaR | Curves & surfaces | Advanced |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCENARIO GENERATION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Tier 1: TimeseriesGenerator (Scalar Factors)                               │
│  ─────────────────────────────────────────────                               │
│  ├── GBM → FX Spot, Equity, Commodities                                     │
│  ├── Heston → Equity with stochastic vol                                    │
│  ├── OU → Rate levels, spreads, vol factors                                 │
│  └── Output: [T, S] per factor                                              │
│                                                                              │
│  Tier 2: FactorModelGenerator (Term Structures)                             │
│  ─────────────────────────────────────────────────                           │
│  ├── SpotFactorSpec → FX Spot [T, S]                                        │
│  ├── CurveFactorSpec → Yield curves [T, S, n_tenors]                        │
│  ├── VolSurfaceFactorSpec → Vol surfaces [T, S, n_exp, n_strike]            │
│  └── PCA-based factor loadings                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tier 1: TimeseriesGenerator

The `TimeseriesGenerator` is designed for rapid prototyping and simple risk factor
simulation. Each factor produces a scalar value per (time, scenario).

### Supported Dynamics

| Model | SDE | Use Case |
|-------|-----|----------|
| **GBM** | `dS = μS dt + σS dW` | FX, equity, commodities |
| **Heston** | `dS = μS dt + √V S dW^S`, `dV = κ(θ-V)dt + ξ√V dW^V` | Equity with vol clustering |
| **OU** | `dX = κ(θ-X) dt + σ dW` | Rates, spreads, vol factors |

### Example

```python
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator,
    TimeseriesConfig,
    RiskFactorSpec,
    GBMDynamicsSpec,
    OUDynamicsSpec,
)
from src.marketdata.core.ids import MarketId
import numpy as np

# Define factors
factors = [
    RiskFactorSpec(
        market_id=MarketId("FX", "SPOT", "EURUSD"),
        initial_value=1.08,
        dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
        name="EUR/USD",
    ),
    RiskFactorSpec(
        market_id=MarketId("IR", "LEVEL", "USD"),
        initial_value=0.05,
        dynamics=OUDynamicsSpec(mean=0.04, kappa=0.5, vol=0.01),
        name="USD Rate",
    ),
]

# Correlation matrix
correlation = np.array([
    [1.0, -0.2],
    [-0.2, 1.0],
])

# Configure and generate
config = TimeseriesConfig(
    factors=factors,
    correlation=correlation,
    start_date="2024-01-01",
    end_date="2024-12-31",
    freq="D",
    n_scenarios=10000,
)

generator = TimeseriesGenerator(config)
dataset = generator.generate(seed=42)

# Access market snapshot
market = dataset.snapshot(time_idx=100, scenario_idx=0)
fx_spot = market.quote(factors[0].market_id)
```

### When to Use

- Quick prototyping and testing
- Simple VaR with scalar risk factors
- Educational examples
- When full term structures aren't needed

---

## Tier 2: FactorModelGenerator

The `FactorModelGenerator` is designed for production hedge fund workflows that
require full term structure simulation for yield curves and volatility surfaces.

### Mathematical Framework

#### Factor Model for Yield Curves

Yield curves are driven by PCA-style factors (typically 3: level, slope, curvature):

```
r(t, τ) = r₀(τ) + Σᵢ fᵢ(t) × λᵢ(τ)

where:
  r₀(τ)  = Initial zero rate at tenor τ
  fᵢ(t)  = Factor i value at time t (OU process)
  λᵢ(τ)  = Loading of factor i at tenor τ
```

**Typical Factor Loadings (from historical PCA):**

| Factor | Short End | Belly | Long End | Economic Interpretation |
|--------|-----------|-------|----------|-------------------------|
| Level | +1.0 | +1.0 | +1.0 | Parallel shift |
| Slope | -0.8 | 0.0 | +0.8 | 2s10s twist |
| Curvature | +0.3 | -0.5 | +0.3 | Butterfly |

#### Factor Model for Volatility Surfaces

Vol surfaces are driven by factors that affect different parts of the smile:

```
σ(t, T, K) = σ₀(T, K) + Σᵢ vᵢ(t) × βᵢ(T, K)

where:
  σ₀(T, K)  = Initial vol at expiry T, strike K
  vᵢ(t)    = Vol factor i value (OU process)
  βᵢ(T, K)  = Loading at each (expiry, strike) point
```

**Typical Vol Factors:**

| Factor | Wings | ATM | Economic Interpretation |
|--------|-------|-----|-------------------------|
| ATM | +1.0 | +1.0 | Overall vol level |
| Skew | +/- | 0.0 | Risk reversal (25D RR) |
| Smile | + | 0.0 | Butterfly (25D BF) |

### Example

```python
from src.marketdata.scenarios.timeseries import (
    FactorModelGenerator,
    CurveFactorSpec,
    VolSurfaceFactorSpec,
    SpotFactorSpec,
    FactorDynamics,
)
from src.marketdata.core.ids import MarketId
import numpy as np

# Define FX spot
spot_spec = SpotFactorSpec(
    market_id=MarketId("FX", "SPOT", "EURUSD"),
    initial_value=1.0850,
    dynamics=FactorDynamics(dynamics_type="gbm", vol=0.085),
)

# Define USD yield curve with PCA factors
usd_curve = CurveFactorSpec(
    market_id=MarketId("IR", "CURVE", "USD"),
    tenors=np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]),
    initial_rates=np.array([0.054, 0.053, 0.052, 0.048, 0.042, 0.040, 0.043]),
    factor_loadings={
        "level": np.ones(7) * 0.01,
        "slope": np.array([-0.012, -0.008, -0.004, 0.0, 0.004, 0.008, 0.01]),
    },
    factor_dynamics={
        "level": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=0.3, vol=0.5),
        "slope": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=0.5, vol=0.3),
    },
)

# Define vol surface
vol_spec = VolSurfaceFactorSpec(
    market_id=MarketId("FX", "VOL", "EURUSD"),
    expiries=np.array([0.25, 0.5, 1.0, 2.0]),
    strikes=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
    initial_vols=np.array([
        [0.12, 0.095, 0.085, 0.090, 0.105],
        [0.115, 0.092, 0.083, 0.087, 0.100],
        [0.110, 0.090, 0.082, 0.085, 0.095],
        [0.105, 0.088, 0.082, 0.084, 0.092],
    ]),
    factor_loadings={
        "atm": np.ones((4, 5)) * 0.1,
        "skew": np.array([[-0.02, -0.01, 0, 0.01, 0.02]] * 4),
    },
    factor_dynamics={
        "atm": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=2.0, vol=0.8),
        "skew": FactorDynamics(dynamics_type="ou", mean=0.0, kappa=3.0, vol=0.5),
    },
)

# Build correlation matrix (3 factors: spot, curve_level, curve_slope)
# In production, estimate from historical returns
correlation = np.array([
    [1.0, -0.15, 0.05],
    [-0.15, 1.0, 0.3],
    [0.05, 0.3, 1.0],
])

# Generate scenarios
generator = FactorModelGenerator(
    spots=[spot_spec],
    curves=[usd_curve],
    vol_surfaces=[vol_spec],
    correlation_matrix=correlation,
)

result = generator.generate(n_time=252, n_scenarios=10000, seed=42)

# Access paths
spot_paths = result.spot_paths[spot_spec.market_id]       # [253, 10000]
curve_paths = result.curve_paths[usd_curve.market_id]     # [253, 10000, 7]
vol_paths = result.vol_paths[vol_spec.market_id]          # [253, 10000, 4, 5]

# Convert to MarketDataset
dataset = result.to_dataset()
```

### When to Use

- Production VaR/ES computation
- Full revaluation scenarios
- XVA simulations
- Stress testing with term structure dynamics
- Any workflow requiring realistic curve/surface evolution

---

## Correlation Handling

Both generators use Cholesky decomposition for correlation:

```python
# Given correlation matrix Σ = LL^T
# Generate independent shocks Z ~ N(0, I)
# Correlated shocks: Z_corr = Z @ L^T
```

### Building Correlation Matrices

**Economic relationships to consider:**

| Factor Pair | Typical Correlation | Rationale |
|-------------|---------------------|-----------|
| EUR/USD ↔ GBP/USD | 0.5 - 0.7 | Both dollar pairs |
| Equity ↔ USD rates | -0.1 to -0.3 | Higher rates → lower equity |
| FX spot ↔ FX vol | -0.3 to -0.5 | Risk-off: vol up, risky CCY down |
| USD rates ↔ EUR rates | 0.5 - 0.7 | Global rate environment |
| Rate level ↔ slope | 0.2 - 0.4 | Curve typically steepens in easing |

**Ensuring Positive Semi-Definiteness:**

```python
import numpy as np

def nearest_psd(A):
    """Find nearest positive semi-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

# Verify PSD
eigvals = np.linalg.eigvalsh(correlation)
if eigvals.min() < 0:
    correlation = nearest_psd(correlation)
```

---

## Historical Simulation

For scenarios based on historical data rather than Monte Carlo, use the
historical simulation module:

```python
from src.marketdata.scenarios.historical import (
    HistoricalSimulator,
    HistoricalConfig,
)

config = HistoricalConfig(
    method="filtered",  # or "block_bootstrap", "stationary_bootstrap"
    n_scenarios=10000,
    vol_model="ewma",   # for filtered historical
    decay=0.94,
)

simulator = HistoricalSimulator(config)
dataset = simulator.generate(historical_data, seed=42)
```

---

## Copulas and Advanced Correlation

For non-Gaussian dependence structures:

```python
from src.marketdata.scenarios.correlation import (
    StudentTCopula,
    ClaytonCopula,
    DynamicCorrelation,
)

# Student-t copula (symmetric tail dependence)
copula = StudentTCopula(correlation_matrix=corr, df=5)
samples = copula.sample(n=10000, seed=42)

# Clayton copula (lower tail dependence)
copula = ClaytonCopula(theta=2.0)

# Time-varying correlation (DCC-GARCH)
dcc = DynamicCorrelation(method="dcc")
time_varying_corr = dcc.fit(returns).forecast(horizon=10)
```

---

## Integration with Risk Workflows

### VaR Computation

```python
from src.marketdata.scenarios.timeseries import FactorModelGenerator
from src.risk.scenarios.runner import ScenarioRunner

# Generate scenarios
generator = FactorModelGenerator(spots=..., curves=..., vol_surfaces=...)
scenarios = generator.generate(n_time=10, n_scenarios=10000)
dataset = scenarios.to_dataset()

# Run full revaluation
runner = ScenarioRunner(portfolio, pricer_registry)
pnl = runner.compute_pnl(dataset)

# Calculate risk metrics
var_95 = np.percentile(pnl, 5)
es_95 = np.mean(pnl[pnl <= var_95])
```

### Stress Testing

```python
# Define stress scenario parameters
stress_vol = 0.15  # Vol spike
stress_corr_bump = 0.3  # Correlation spike

# Modify factor dynamics
stress_spot = SpotFactorSpec(
    ...,
    dynamics=FactorDynamics(dynamics_type="gbm", vol=stress_vol),
)

# Generate stress scenarios
stress_generator = FactorModelGenerator(
    spots=[stress_spot],
    curves=[stress_curve],
    correlation_matrix=stress_corr,
)
```

---

## API Reference

### TimeseriesGenerator

```python
class TimeseriesGenerator:
    def __init__(self, config: TimeseriesConfig)
    def generate(self, seed: int = 42) -> MarketDataset
    def generate_paths(self, seed: int = 42) -> GenerationResult
    def compute_statistics(self, result: GenerationResult) -> Dict
    def compute_realized_correlation(self, result: GenerationResult) -> np.ndarray
```

### FactorModelGenerator

```python
class FactorModelGenerator:
    def __init__(
        self,
        spots: List[SpotFactorSpec] = None,
        curves: List[CurveFactorSpec] = None,
        vol_surfaces: List[VolSurfaceFactorSpec] = None,
        correlation_matrix: np.ndarray = None,
        dt: float = 1/252,
    )
    def generate(
        self,
        n_time: int,
        n_scenarios: int,
        seed: int = 42,
        start_date: str = "2024-01-01",
    ) -> FactorModelResult
```

### Dynamics Specifications

```python
@dataclass
class FactorDynamics:
    dynamics_type: str  # "gbm" or "ou"
    mean: float = 0.0   # OU long-term mean
    kappa: float = 1.0  # OU mean reversion speed
    vol: float = 0.01   # Volatility
    drift: float = 0.0  # GBM drift

@dataclass
class CurveFactorSpec:
    market_id: MarketId
    tenors: np.ndarray
    initial_rates: np.ndarray
    factor_loadings: Dict[str, np.ndarray]
    factor_dynamics: Dict[str, FactorDynamics]

@dataclass
class VolSurfaceFactorSpec:
    market_id: MarketId
    expiries: np.ndarray
    strikes: np.ndarray
    initial_vols: np.ndarray  # Shape: (n_exp, n_strike)
    factor_loadings: Dict[str, np.ndarray]  # Each: (n_exp, n_strike)
    factor_dynamics: Dict[str, FactorDynamics]
    vol_floor: float = 0.001
```

---

## Best Practices

1. **Start Simple**: Use `TimeseriesGenerator` for prototyping, then upgrade to
   `FactorModelGenerator` for production.

2. **Calibrate from Data**: Estimate factor loadings and correlation from
   historical PCA, not arbitrary values.

3. **Validate Correlation**: Always check realized correlation against input:
   ```python
   realized = generator.compute_realized_correlation(result)
   error = np.abs(input_corr - realized).max()
   assert error < 0.05, f"Correlation error: {error}"
   ```

4. **Set Appropriate Horizons**: VaR horizon should match risk limit (1-day, 10-day).

5. **Use Sufficient Scenarios**: 10,000+ scenarios for stable VaR estimates.

6. **Handle Term Structure**: For options/bonds, use `FactorModelGenerator` to
   capture curve/surface dynamics.

---

## See Also

- `examples/risk/fx_option_scenario_pnl.py` - Complete production example
- `examples/fundamentals/07_timeseries_generation.py` - TimeseriesGenerator tutorial
- `src/marketdata/scenarios/timeseries/` - Implementation
- `src/marketdata/scenarios/historical/` - Historical simulation
- `src/marketdata/scenarios/correlation/` - Advanced correlation models
