# Portfolio Optimisation Reference

Technical specification and API reference for portfolio optimisation components.

## Overview

The portfolio optimisation module provides:
- **Mean-Variance Optimisation**: Markowitz efficient frontier
- **Risk Parity**: Equal risk contribution portfolios
- **Black-Litterman**: Bayesian return estimation with views
- **Covariance Estimation**: Robust estimation with shrinkage

---

## Mean-Variance Optimisation

### Module: `src.portfolio.optimization.mean_variance`

### MeanVarianceOptimizer

```python
from src.portfolio.optimization import MeanVarianceOptimizer, MVConstraints

optimizer = MeanVarianceOptimizer(risk_free_rate=0.02)

# Optimise for maximum Sharpe ratio
result = optimizer.optimize(
    expected_returns=returns,    # (n_assets,) array
    covariance=cov_matrix,       # (n_assets, n_assets) array
)

# Optimise with target return
result = optimizer.optimize(
    expected_returns=returns,
    covariance=cov_matrix,
    target_return=0.10,
)

# Optimise with target volatility
result = optimizer.optimize(
    expected_returns=returns,
    covariance=cov_matrix,
    target_volatility=0.15,
)
```

### MVConstraints

```python
constraints = MVConstraints(
    long_only=True,           # No short selling
    max_weight=0.3,           # Max 30% per asset
    min_weight=0.0,           # Min weight
    max_sector_weight={       # Sector constraints
        "tech": 0.4,
        "finance": 0.3,
    },
)

result = optimizer.optimize(
    expected_returns=returns,
    covariance=cov_matrix,
    constraints=constraints,
)
```

### MVOptimizationResult

```python
@dataclass
class MVOptimizationResult:
    weights: np.ndarray           # Optimal weights
    expected_return: float        # Portfolio return
    volatility: float             # Portfolio volatility
    sharpe_ratio: float           # Sharpe ratio
```

### Efficient Frontier

```python
frontier = optimizer.efficient_frontier(
    expected_returns=returns,
    covariance=cov_matrix,
    n_points=20,
    constraints=constraints,
)

# frontier is List[MVOptimizationResult]
for portfolio in frontier:
    print(f"Return: {portfolio.expected_return:.2%}, "
          f"Vol: {portfolio.volatility:.2%}, "
          f"Sharpe: {portfolio.sharpe_ratio:.2f}")
```

### Minimum Variance Portfolio

```python
result = optimizer.min_variance(covariance=cov_matrix)
```

---

## Risk Parity

### Module: `src.portfolio.optimization.risk_parity`

### RiskParityOptimizer

```python
from src.portfolio.optimization import RiskParityOptimizer

optimizer = RiskParityOptimizer(max_iterations=1000, tolerance=1e-8)

# Equal risk contribution
result = optimizer.optimize(covariance=cov_matrix)

# Custom risk budgets
result = optimizer.optimize(
    covariance=cov_matrix,
    risk_budgets=np.array([0.3, 0.3, 0.2, 0.2]),  # Sum to 1
)

# With leverage
result = optimizer.optimize(
    covariance=cov_matrix,
    leverage=1.5,  # Weights sum to 1.5
)
```

### RiskParityResult

```python
@dataclass
class RiskParityResult:
    weights: np.ndarray           # Optimal weights
    risk_contributions: np.ndarray # Risk contribution per asset
    volatility: float             # Portfolio volatility
```

### Hierarchical Risk Parity (HRP)

```python
result = optimizer.hierarchical(covariance=cov_matrix)
```

---

## Black-Litterman Model

### Module: `src.portfolio.optimization.black_litterman`

### BlackLittermanModel

```python
from src.portfolio.optimization import BlackLittermanModel

model = BlackLittermanModel(
    market_caps=market_caps,      # Market capitalizations
    covariance=cov_matrix,        # Covariance matrix
    risk_aversion=2.5,            # Risk aversion parameter
    tau=0.05,                     # Uncertainty in equilibrium
    risk_free_rate=0.02,
)

# Equilibrium returns (no views)
eq_returns = model.equilibrium_returns
market_weights = model.market_weights
```

### Adding Views

```python
# Views format: (asset_index, return) for absolute
#              ([long_assets, short_assets], spread) for relative

# Absolute view: Asset 0 returns 12%
views = [(0, 0.12)]
confidences = [0.5]  # 50% confidence

# Relative view: Asset 0 outperforms Asset 2 by 2%
views = [([0, 2], 0.02)]
confidences = [0.6]

# Multiple views
views = [
    (0, 0.12),           # Asset 0 returns 12%
    ([1, 3], 0.03),      # Asset 1 beats Asset 3 by 3%
]
confidences = [0.5, 0.7]

# Compute posterior
result = model.posterior(views=views, confidences=confidences)
```

### BlackLittermanResult

```python
@dataclass
class BlackLittermanResult:
    posterior_returns: np.ndarray      # Posterior expected returns
    posterior_covariance: np.ndarray   # Posterior covariance
    optimal_weights: np.ndarray        # Optimal portfolio weights
    equilibrium_returns: np.ndarray    # Prior (equilibrium) returns
```

---

## Covariance Estimation

### Module: `src.portfolio.optimization.covariance`

### CovarianceEstimator

```python
from src.portfolio.optimization import CovarianceEstimator

estimator = CovarianceEstimator(annualization=252)

# Sample covariance
cov = estimator.sample(returns, annualize=True)

# Exponentially weighted (more weight on recent)
cov = estimator.ewm(returns, halflife=60)

# Constant correlation model
cov = estimator.constant_correlation(returns)
```

### ShrinkageEstimator (Ledoit-Wolf)

```python
from src.portfolio.optimization import ShrinkageEstimator

estimator = ShrinkageEstimator(
    shrinkage_target="identity",  # or "diagonal", "constant_correlation"
    annualization=252,
)

result = estimator.estimate(returns)

print(f"Optimal shrinkage: {result.shrinkage_intensity:.2%}")
cov = result.covariance
```

### Shrinkage Targets

| Target | Description |
|--------|-------------|
| `identity` | Shrink toward identity matrix |
| `diagonal` | Shrink toward diagonal (sample variances) |
| `constant_correlation` | Shrink toward constant correlation |

### ShrinkageResult

```python
@dataclass
class ShrinkageResult:
    covariance: np.ndarray        # Shrunk covariance
    shrinkage_intensity: float    # Optimal intensity (0-1)
```

---

## Complete Example

```python
import numpy as np
from src.portfolio.optimization import (
    MeanVarianceOptimizer, MVConstraints,
    RiskParityOptimizer,
    BlackLittermanModel,
    ShrinkageEstimator,
)

# Asset data
returns_data = ...  # (n_obs, n_assets) array
market_caps = np.array([500e9, 400e9, 300e9, 200e9, 100e9])

# 1. Robust covariance estimation
estimator = ShrinkageEstimator(shrinkage_target="constant_correlation")
cov_result = estimator.estimate(returns_data)
cov = cov_result.covariance

# 2. Expected returns via Black-Litterman
bl = BlackLittermanModel(market_caps=market_caps, covariance=cov)

# Add views
views = [(0, 0.15), ([1, 3], 0.02)]
confidences = [0.6, 0.5]

bl_result = bl.posterior(views=views, confidences=confidences)
expected_returns = bl_result.posterior_returns

# 3. Mean-variance optimisation
constraints = MVConstraints(long_only=True, max_weight=0.3)
mv_opt = MeanVarianceOptimizer(risk_free_rate=0.02)

mv_result = mv_opt.optimize(
    expected_returns=expected_returns,
    covariance=bl_result.posterior_covariance,
    constraints=constraints,
)

print("Mean-Variance Portfolio:")
print(f"  Expected Return: {mv_result.expected_return:.2%}")
print(f"  Volatility: {mv_result.volatility:.2%}")
print(f"  Sharpe Ratio: {mv_result.sharpe_ratio:.2f}")

# 4. Risk parity portfolio
rp_opt = RiskParityOptimizer()
rp_result = rp_opt.optimize(covariance=cov)

print("\nRisk Parity Portfolio:")
print(f"  Risk Contributions: {rp_result.risk_contributions}")
print(f"  Volatility: {rp_result.volatility:.2%}")

# 5. Efficient frontier
frontier = mv_opt.efficient_frontier(
    expected_returns=expected_returns,
    covariance=cov,
    n_points=20,
)

# Plot frontier
import matplotlib.pyplot as plt
vols = [p.volatility for p in frontier]
rets = [p.expected_return for p in frontier]
plt.plot(vols, rets)
plt.xlabel("Volatility")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.show()
```
