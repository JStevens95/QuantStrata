# Longstaff-Schwartz Monte Carlo (LSM) User Guide

This guide explains how to use the Longstaff-Schwartz algorithm in QuantStrata for pricing American options via Monte Carlo simulation.

---

## Overview

The Longstaff-Schwartz (LSM) algorithm prices American options by combining Monte Carlo simulation with regression. It works backward through time, using least-squares regression to estimate continuation values.

**Key advantages:**
- Works with any underlying dynamics (GBM, Heston, jump-diffusion)
- Handles high-dimensional problems
- Provides exercise boundary estimates

---

## Quick Start

```python
from src.models.numeric.monte_carlo.lsm import price_american_put_lsm

# Price an American put
result = price_american_put_lsm(
    spot0=100.0,       # Initial spot price
    strike=100.0,      # Strike price
    maturity=1.0,      # Time to maturity (years)
    r=0.05,            # Risk-free rate
    sigma=0.2,         # Volatility
    n_paths=100000,    # Number of MC paths
    n_steps=50,        # Number of time steps
    seed=42,           # Random seed for reproducibility
)

print(f"Price: {result.price:.4f}")
print(f"Std Error: {result.std_error:.4f}")
print(f"95% CI: {result.confidence_interval_95}")
```

---

## Core Components

### LSMResult

The result container holds all pricing information:

```python
@dataclass
class LSMResult:
    price: float                    # Option price
    std_error: float               # Standard error
    exercise_boundary: np.ndarray  # Estimated exercise boundary
    n_paths: int                   # Number of paths used
    n_steps: int                   # Number of time steps
    basis_type: BasisType          # Basis function type
    basis_degree: int              # Polynomial degree
```

Access the 95% confidence interval:

```python
lower, upper = result.confidence_interval_95
print(f"Price is between {lower:.4f} and {upper:.4f} with 95% confidence")
```

### Basis Functions

LSM uses polynomial basis functions for regression. Three types are available:

```python
from src.models.numeric.monte_carlo.lsm import BasisType

# Available options
BasisType.POLYNOMIAL  # 1, x, x², x³, ...
BasisType.LAGUERRE    # Laguerre polynomials (recommended)
BasisType.CHEBYSHEV   # Chebyshev polynomials
```

**Laguerre polynomials** are recommended (default) as they perform well across a range of option parameters.

---

## Pricing American Puts

### Basic Usage

```python
from src.models.numeric.monte_carlo.lsm import price_american_put_lsm

result = price_american_put_lsm(
    spot0=100,
    strike=100,
    maturity=1.0,
    r=0.05,
    sigma=0.2,
)
```

### With Custom Parameters

```python
result = price_american_put_lsm(
    spot0=100,
    strike=110,            # ITM put
    maturity=0.5,
    r=0.08,                # Higher rate
    sigma=0.3,             # Higher vol
    n_paths=200000,        # More paths for accuracy
    n_steps=100,           # More steps for early exercise
    basis_type=BasisType.LAGUERRE,
    basis_degree=4,        # Higher degree for complex payoffs
    seed=123,
)
```

---

## Pricing American Calls (with Dividends)

American calls on non-dividend paying stocks should not be exercised early. For dividend-paying assets:

```python
from src.models.numeric.monte_carlo.lsm import price_american_call_lsm

result = price_american_call_lsm(
    spot0=100,
    strike=100,
    maturity=1.0,
    r=0.05,
    q=0.03,        # Continuous dividend yield
    sigma=0.2,
    n_paths=100000,
)
```

---

## Working with Custom Paths

For advanced use cases (e.g., stochastic volatility), generate paths separately:

```python
import numpy as np
from src.models.numeric.monte_carlo.lsm import lsm_american_put

# Generate custom paths (e.g., Heston model)
n_paths, n_steps = 50000, 50
T, r = 1.0, 0.05
dt = T / n_steps

# Your custom path simulation here
# paths shape: (n_paths, n_steps + 1)
paths = generate_heston_paths(...)  # Example

# Price with custom paths
result = lsm_american_put(
    paths=paths,
    strike=100.0,
    r=r,
    dt=dt,
    basis_type=BasisType.LAGUERRE,
    basis_degree=3,
)
```

---

## Analyzing the Exercise Boundary

LSM estimates the early exercise boundary:

```python
import matplotlib.pyplot as plt

result = price_american_put_lsm(
    spot0=100, strike=100, maturity=1.0, r=0.05, sigma=0.2,
    n_paths=100000, n_steps=50, seed=42
)

# Plot exercise boundary
time_grid = np.linspace(0, 1.0, len(result.exercise_boundary))
valid_mask = ~np.isnan(result.exercise_boundary)

plt.figure(figsize=(10, 6))
plt.plot(time_grid[valid_mask], result.exercise_boundary[valid_mask], 'b-', linewidth=2)
plt.axhline(y=100, color='r', linestyle='--', label='Strike')
plt.xlabel('Time to Maturity')
plt.ylabel('Spot Price')
plt.title('American Put Exercise Boundary')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Comparing with European Options

American puts have an early exercise premium:

```python
from scipy.stats import norm

# Black-Scholes European put
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
european_put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# LSM American put
american_result = price_american_put_lsm(
    spot0=S, strike=K, maturity=T, r=r, sigma=sigma,
    n_paths=200000, n_steps=100, seed=42
)

premium = american_result.price - european_put
print(f"European Put: {european_put:.4f}")
print(f"American Put: {american_result.price:.4f}")
print(f"Early Exercise Premium: {premium:.4f}")
```

---

## Tips and Best Practices

### Number of Paths

- **10,000 paths**: Quick estimate, ~5% error
- **100,000 paths**: Standard accuracy, ~1-2% error
- **500,000+ paths**: High accuracy, <1% error

### Number of Time Steps

- **25 steps**: Quick estimate
- **50 steps**: Standard (weekly exercise)
- **100+ steps**: High accuracy, captures early exercise better

### Basis Functions

- **Laguerre**: Best general-purpose choice
- **Polynomial**: Good for simple payoffs
- **Chebyshev**: Good numerical stability

### Degree Selection

- **Degree 2-3**: Usually sufficient
- **Degree 4-5**: For complex payoffs
- Higher degrees may overfit with limited ITM paths

### ITM Requirement

LSM only regresses on in-the-money paths. If very few paths are ITM (deep OTM option), results may be unreliable.

---

## Common Issues

### 1. High Standard Error

**Cause**: Not enough paths.
**Solution**: Increase `n_paths`.

### 2. Unstable Price Estimates

**Cause**: Not enough ITM paths for regression.
**Solution**: The option may be very OTM. Try different strikes or increase paths.

### 3. Price Below European

**Cause**: MC noise or insufficient time steps.
**Solution**: Increase `n_steps` and `n_paths`.

---

## See Also

- [Monte Carlo Methods Reference](../reference/models/monte_carlo_methods.md) - Mathematical details
- [American Options Tutorial](../tutorials/pricing/advanced_mc_methods.ipynb) - Interactive examples
- [QMC Guide](qmc.md) - Quasi-Monte Carlo methods
- [Importance Sampling Guide](importance_sampling.md) - Variance reduction

---

*QuantStrata User Guide | January 2026*
