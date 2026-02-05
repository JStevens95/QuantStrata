# Time Series Generation for Risk Factor Simulation

## Overview

The `src.marketdata.scenarios.timeseries` module provides production-grade Monte Carlo time series generation for risk factors. This enables hedge fund-quality scenario simulation for VaR, stress testing, and P&L attribution workflows.

## Architecture

```
src/marketdata/scenarios/
├── __init__.py              # Scenarios module exports
├── generator.py             # ScenarioDriver (correlated Gaussian shocks)
├── interfaces.py            # Protocols (MarketView, ScenarioShock)
├── shocks.py                # Deterministic shocks (SpotShock, VolShock)
├── runner.py                # Basic scenario runner
└── timeseries/              # Monte Carlo time series generation
    ├── __init__.py          # Public API exports
    ├── config.py            # Configuration dataclasses
    ├── generator.py         # TimeseriesGenerator
    └── adapters/            # Dynamics model adapters
        ├── base.py          # DynamicsAdapter protocol
        ├── gbm.py           # Geometric Brownian Motion
        ├── heston.py        # Heston stochastic volatility
        ├── ou.py            # Ornstein-Uhlenbeck (mean-reverting)
        └── factor.py        # Factor model with loadings
```

### Why This Location?

The time series generation lives under `marketdata/scenarios/` because:
1. **Output is market data** - It produces `MarketDataset`, not portfolio PnL
2. **Uses existing infrastructure** - Builds on `ScenarioDriver` Cholesky pattern
3. **Consumed by risk** - `risk/scenarios/` uses this for portfolio scenario analysis

### Design Philosophy

This module follows hedge fund best practices:

1. **Separation of Concerns**: Risk scenario generation is distinct from market data provisioning
2. **Correlation First**: Cross-asset correlation is handled at the driver level via Cholesky decomposition
3. **Dynamics Agnostic**: The generator delegates to adapters, allowing easy extension
4. **MarketDataset Output**: Direct integration with pricing and risk workflows

### Relationship to Other Modules

| Module | Purpose | Relationship |
|--------|---------|--------------|
| `marketdata/scenarios/generator.py` | Correlated Gaussian shocks | TimeseriesGenerator uses same Cholesky approach |
| `marketdata/scenarios/shocks.py` | Deterministic shocks (SpotShock, VolShock) | Complementary - for stress scenarios |
| `marketdata/providers/synthetic/` | Mock market data for testing | Different use case - not correlated Monte Carlo |
| `models/dynamics/` | Raw dynamics simulators | Adapters wrap these for the generator interface |

---

## Mathematical Framework

### Correlation Handling

Given $n$ risk factors, we define a correlation matrix $\Sigma \in \mathbb{R}^{n \times n}$.

The generator:
1. Creates independent standard normal shocks $Z_{t,s,f} \sim N(0, I)$
2. Computes Cholesky decomposition $\Sigma = LL^T$
3. Applies correlation: $\tilde{Z} = Z \cdot L^T$
4. Transforms correlated shocks to paths via factor-specific dynamics

### Supported Dynamics

#### Geometric Brownian Motion (GBM)

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

**Exact discretization**:
$$S_{t+\Delta t} = S_t \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} \, Z\right]$$

**Use cases**: FX spot, equity spot, commodity prices

**Parameters**:
- $\mu$: Drift (annual). For risk-neutral: $\mu = r - q$
- $\sigma$: Volatility (annual). Typical: 0.05 - 0.50

#### Heston Stochastic Volatility

$$dS_t = \mu S_t \, dt + \sqrt{V_t} S_t \, dW_t^S$$
$$dV_t = \kappa(\theta - V_t) \, dt + \xi \sqrt{V_t} \, dW_t^V$$
$$\text{Corr}(dW^S, dW^V) = \rho$$

**Use cases**: Equity with realistic vol dynamics, options pricing

**Parameters**:
- $\kappa$: Mean reversion speed (typical: 1-5)
- $\theta$: Long-term variance (typical: 0.01-0.10)
- $\xi$: Vol of vol (typical: 0.1-1.0)
- $V_0$: Initial variance
- $\rho$: Spot-variance correlation (typical: -0.9 to -0.3 for equities)

**Feller Condition**: If $2\kappa\theta > \xi^2$, variance stays positive almost surely.

#### Ornstein-Uhlenbeck (Mean-Reverting)

$$dX_t = \kappa(\theta - X_t) \, dt + \sigma \, dW_t$$

**Exact discretization**:
$$X_{t+\Delta t} = \theta + (X_t - \theta)e^{-\kappa\Delta t} + \sigma_{eff} Z$$

where $\sigma_{eff} = \sigma\sqrt{\frac{1 - e^{-2\kappa\Delta t}}{2\kappa}}$

**Use cases**: Interest rates, credit spreads, vol factors

**Parameters**:
- $\theta$: Long-term mean
- $\kappa$: Mean reversion speed
- $\sigma$: Volatility

**Properties**:
- Half-life: $t_{1/2} = \ln(2)/\kappa$
- Stationary variance: $\sigma^2/(2\kappa)$

#### Factor Model

Factor follows OU dynamics with tenor loadings:
$$dF_t = \kappa(\theta - F_t) \, dt + \sigma \, dW_t$$
$$\Delta R(\tau) = \lambda(\tau) \cdot F_t$$

**Use cases**: PCA-based yield curve factors (Level, Slope, Curvature)

---

## API Reference

### TimeseriesConfig

Top-level configuration for the generator.

```python
@dataclass
class TimeseriesConfig:
    factors: Sequence[RiskFactorSpec]    # Risk factors to simulate
    correlation: np.ndarray              # Correlation matrix (n x n)
    start_date: str                      # ISO date "YYYY-MM-DD"
    end_date: str                        # ISO date "YYYY-MM-DD"
    freq: Literal["D", "W", "M", "B"]   # Frequency
    n_scenarios: int = 1000              # Number of MC scenarios
    dt: Optional[float] = None           # Override time step (years)
```

### RiskFactorSpec

Specification for a single risk factor.

```python
@dataclass
class RiskFactorSpec:
    market_id: MarketId       # Market identifier
    initial_value: float      # Starting value
    dynamics: DynamicsSpec    # Dynamics specification
    name: Optional[str]       # Human-readable name
```

### DynamicsSpec Types

```python
GBMDynamicsSpec(drift: float, vol: float)

HestonDynamicsSpec(
    drift: float,
    kappa: float,      # Mean reversion speed
    theta: float,      # Long-term variance
    xi: float,         # Vol of vol
    v0: float,         # Initial variance
    rho_internal: float,  # Spot-variance correlation
)

OUDynamicsSpec(mean: float, kappa: float, vol: float)

FactorDynamicsSpec(
    mean: float,
    kappa: float,
    vol: float,
    loadings: dict[str, float],  # Tenor -> loading
)
```

### TimeseriesGenerator

Main orchestrator class.

```python
class TimeseriesGenerator:
    def __init__(self, config: TimeseriesConfig): ...
    
    def generate(self, seed: Optional[int] = None) -> MarketDataset:
        """Generate MarketDataset with correlated risk factor paths."""
    
    def generate_paths(self, seed: Optional[int] = None) -> GenerationResult:
        """Generate raw paths without building MarketDataset."""
    
    def compute_statistics(self, result: GenerationResult) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for generated paths."""
    
    def compute_realized_correlation(
        self, result: GenerationResult, log_returns: bool = True
    ) -> np.ndarray:
        """Compute realized correlation matrix from paths."""
```

### GenerationResult

Container for raw generated paths.

```python
@dataclass
class GenerationResult:
    paths: Dict[MarketId, np.ndarray]           # Shape (n_time+1, n_scenarios)
    variance_paths: Dict[MarketId, np.ndarray]  # For Heston factors
    dates: List[str]
    n_scenarios: int
    seed: int
```

---

## Usage Examples

### Example 1: Single GBM Factor

```python
import numpy as np
from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator,
    TimeseriesConfig,
    RiskFactorSpec,
    GBMDynamicsSpec,
)

# Define EUR/USD with 8% annual volatility
eurusd = RiskFactorSpec(
    market_id=MarketId("FX", "SPOT", "EURUSD"),
    initial_value=1.08,
    dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
)

config = TimeseriesConfig(
    factors=[eurusd],
    correlation=np.array([[1.0]]),
    start_date="2024-01-01",
    end_date="2024-12-31",
    freq="D",
    n_scenarios=10000,
)

generator = TimeseriesGenerator(config)
dataset = generator.generate(seed=42)

# Access market snapshot at time 100, scenario 0
market = dataset.snapshot(time_idx=100, scenario_idx=0)
spot = market.quote(eurusd.market_id)
print(f"EUR/USD at t=100, s=0: {spot:.4f}")
```

### Example 2: Correlated Multi-Asset Portfolio

```python
import numpy as np
from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator,
    TimeseriesConfig,
    RiskFactorSpec,
    GBMDynamicsSpec,
    HestonDynamicsSpec,
    OUDynamicsSpec,
)

# Define diverse risk factors
factors = [
    # FX (GBM)
    RiskFactorSpec(
        market_id=MarketId("FX", "SPOT", "EURUSD"),
        initial_value=1.08,
        dynamics=GBMDynamicsSpec(drift=0.0, vol=0.08),
        name="EUR/USD",
    ),
    # Equity with stochastic vol (Heston)
    RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "SPX"),
        initial_value=4500.0,
        dynamics=HestonDynamicsSpec(
            drift=0.05,        # 5% risk-neutral drift
            kappa=2.0,         # Mean reversion
            theta=0.04,        # 20% long-term vol
            xi=0.3,            # Vol of vol
            v0=0.04,           # 20% initial vol
            rho_internal=-0.7, # Leverage effect
        ),
        name="S&P 500",
    ),
    # Rate level (mean-reverting)
    RiskFactorSpec(
        market_id=MarketId("IR", "LEVEL", "USD"),
        initial_value=0.05,
        dynamics=OUDynamicsSpec(mean=0.04, kappa=0.5, vol=0.01),
        name="USD Rate",
    ),
]

# Correlation matrix reflecting typical market behavior:
# - EUR/USD and SPX: 0.30 (risk-on correlation)
# - EUR/USD and USD Rate: -0.20 (higher rates → stronger USD)
# - SPX and USD Rate: -0.10 (higher rates → lower equity)
correlation = np.array([
    [ 1.00,  0.30, -0.20],
    [ 0.30,  1.00, -0.10],
    [-0.20, -0.10,  1.00],
])

config = TimeseriesConfig(
    factors=factors,
    correlation=correlation,
    start_date="2024-01-01",
    end_date="2024-12-31",
    freq="D",
    n_scenarios=10000,
)

generator = TimeseriesGenerator(config)
result = generator.generate_paths(seed=42)

# Validate realized correlation
realized = generator.compute_realized_correlation(result)
print("Input correlation:")
print(correlation)
print("\nRealized correlation:")
print(realized.round(4))
print(f"\nMax error: {np.abs(correlation - realized).max():.4f}")
```

### Example 3: Rate Curve Factors (Level/Slope/Curvature)

```python
import numpy as np
from src.marketdata.core.ids import MarketId
from src.marketdata.scenarios.timeseries import (
    TimeseriesGenerator,
    TimeseriesConfig,
    RiskFactorSpec,
    OUDynamicsSpec,
)

# PCA-style rate curve factors
factors = [
    RiskFactorSpec(
        market_id=MarketId("IR", "LEVEL", "USD"),
        initial_value=0.0,  # Shock factor starts at zero
        dynamics=OUDynamicsSpec(mean=0.0, kappa=0.1, vol=0.005),
        name="Level",
    ),
    RiskFactorSpec(
        market_id=MarketId("IR", "SLOPE", "USD"),
        initial_value=0.0,
        dynamics=OUDynamicsSpec(mean=0.0, kappa=0.2, vol=0.003),
        name="Slope",
    ),
    RiskFactorSpec(
        market_id=MarketId("IR", "CURVE", "USD"),
        initial_value=0.0,
        dynamics=OUDynamicsSpec(mean=0.0, kappa=0.3, vol=0.002),
        name="Curvature",
    ),
]

# Factors are typically weakly correlated
correlation = np.array([
    [1.00, 0.15, -0.10],
    [0.15, 1.00, -0.20],
    [-0.10, -0.20, 1.00],
])

config = TimeseriesConfig(
    factors=factors,
    correlation=correlation,
    start_date="2024-01-01",
    end_date="2024-12-31",
    freq="D",
    n_scenarios=5000,
)

generator = TimeseriesGenerator(config)
result = generator.generate_paths(seed=42)
stats = generator.compute_statistics(result)

for factor in factors:
    s = stats[factor.display_name]
    print(f"{factor.display_name}: terminal_std = {s['terminal_std']*10000:.2f} bp")
```

---

## Integration with Risk Workflows

### VaR Calculation

```python
from src.marketdata.scenarios.timeseries import TimeseriesGenerator, TimeseriesConfig
from src.portfolio import Portfolio
from src.pricers.registry import get_pricer

# Generate scenarios
generator = TimeseriesGenerator(config)
dataset = generator.generate(seed=42)

# Price portfolio across all scenarios
pnl = []
for scenario_idx in range(dataset.n_scenarios):
    market = dataset.snapshot(time_idx=-1, scenario_idx=scenario_idx)
    portfolio_value = sum(
        get_pricer(trade).price(trade, market)
        for trade in portfolio.trades
    )
    pnl.append(portfolio_value - initial_value)

# Calculate VaR
pnl = np.array(pnl)
var_95 = np.percentile(pnl, 5)
var_99 = np.percentile(pnl, 1)
cvar_95 = pnl[pnl <= var_95].mean()
```

### Stress Testing

```python
# Generate stressed scenarios with modified parameters
stress_factors = [
    RiskFactorSpec(
        market_id=MarketId("EQ", "SPOT", "SPX"),
        initial_value=4500.0,
        dynamics=HestonDynamicsSpec(
            drift=-0.10,      # Negative drift (recession)
            kappa=5.0,        # Fast mean reversion
            theta=0.09,       # 30% long-term vol
            xi=0.8,           # High vol of vol
            v0=0.16,          # 40% starting vol
            rho_internal=-0.9, # Extreme leverage
        ),
    ),
]

stress_config = TimeseriesConfig(
    factors=stress_factors,
    correlation=np.array([[1.0]]),
    start_date="2024-01-01",
    end_date="2024-03-31",  # 90-day stress horizon
    freq="D",
    n_scenarios=10000,
)

stress_dataset = TimeseriesGenerator(stress_config).generate(seed=42)
```

---

## Comparison with Alternatives

### vs. `marketdata/providers/synthetic/generators/`

| Aspect | `providers/synthetic/` | `marketdata/scenarios/timeseries/` |
|--------|------------------------|--------------------------|
| **Purpose** | Mock market data for testing | Monte Carlo risk scenarios |
| **Correlation** | Not handled | Cross-asset via Cholesky |
| **Output** | Individual Panels | Full MarketDataset |
| **Dynamics** | GBM only (per-asset) | GBM, Heston, OU, Factor |
| **Use Case** | Unit tests, examples | VaR, stress testing, P&L attribution |

### vs. `marketdata/scenarios/ScenarioDriver`

| Aspect | `ScenarioDriver` | `TimeseriesGenerator` |
|--------|------------------|----------------------|
| **Output** | Raw shocks Z[t,s,f] | Transformed paths + MarketDataset |
| **Dynamics** | None (just correlation) | Full SDE simulation |
| **Level** | Low-level building block | High-level orchestrator |

---

## Best Practices

### 1. Validate Correlation Matrix

Always ensure your correlation matrix is positive semi-definite:

```python
eigenvalues = np.linalg.eigvalsh(correlation)
assert np.all(eigenvalues >= -1e-10), "Correlation matrix not PSD"
```

### 2. Check Feller Condition for Heston

```python
heston_spec = HestonDynamicsSpec(...)
if not heston_spec.feller_satisfied:
    print(f"Warning: Feller condition violated. Ratio: {heston_spec.feller_ratio:.2f}")
```

### 3. Validate Realized Correlation

```python
result = generator.generate_paths(seed=42)
realized = generator.compute_realized_correlation(result)
error = np.abs(config.correlation - realized).max()
if error > 0.05:
    print(f"Warning: Realized correlation differs by {error:.2%}")
```

### 4. Use Reproducible Seeds

```python
# Production: use fixed seeds for audit trail
dataset = generator.generate(seed=20240101)

# Research: use random seeds for diversity
import secrets
dataset = generator.generate(seed=secrets.randbelow(2**31))
```

---

## Extension Guide

### Adding a New Dynamics Model

1. Create adapter in `adapters/`:

```python
# adapters/my_dynamics.py
from dataclasses import dataclass
from src.marketdata.scenarios.timeseries.config import MyDynamicsSpec

@dataclass(frozen=True, slots=True)
class MyDynamicsAdapter:
    spec: MyDynamicsSpec
    
    @property
    def requires_variance_paths(self) -> bool:
        return False  # or True if you produce variance
    
    def simulate(
        self,
        initial_value: float,
        n_time: int,
        n_scenarios: int,
        shocks: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        # Transform shocks to paths
        ...
```

2. Add spec in `config.py`:

```python
@dataclass(frozen=True, slots=True)
class MyDynamicsSpec:
    param1: float
    param2: float
```

3. Register in `generator.py`:

```python
if isinstance(dynamics, MyDynamicsSpec):
    adapters[mkt_id] = MyDynamicsAdapter(spec=dynamics)
```

4. Export in `__init__.py`:

```python
from src.marketdata.scenarios.timeseries.config import MyDynamicsSpec
from src.marketdata.scenarios.timeseries.adapters.my_dynamics import MyDynamicsAdapter

__all__ = [..., "MyDynamicsSpec", "MyDynamicsAdapter"]
```
