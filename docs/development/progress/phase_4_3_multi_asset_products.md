# Phase 4.3: Multi-Asset Products - Progress Report

**Status:** COMPLETE  
**Completed:** January 27, 2026

---

## Overview

Phase 4.3 implements multi-asset derivative products including basket options, spread options, and rainbow options (best-of/worst-of).

---

## Deliverables

### 1. Multi-Asset Simulation Infrastructure

**Location:** `src/models/multi_asset/simulation.py`

**Components:**
- `CorrelationMatrix`: Validated correlation structure with Cholesky decomposition
  - Factory methods: `from_flat()`, `from_pairs()`
  - Validation: symmetry, PSD, diagonal=1
- `MultiAssetGBM`: Multi-dimensional GBM simulator
  - Methods: `simulate()` (full paths), `simulate_terminal()` (efficient for Europeans)
  - Features: Antithetic variates, Cholesky correlation
- `MultiAssetSimulation`: Result container for simulated paths

**Tests:** 17 unit tests passing

### 2. Basket Options

**Location:** `src/models/multi_asset/basket.py`

**Components:**
- `BasketParameters`: Parameters for basket options
- `basket_call_mc()`: MC pricer for basket calls
- `basket_put_mc()`: MC pricer for basket puts
- Convenience functions: `basket_call_simple()`, `basket_put_simple()`

**Payoff:**
- Call: max(Σ wᵢSᵢ(T) - K, 0)
- Put: max(K - Σ wᵢSᵢ(T), 0)

**Tests:** 12 unit tests passing

### 3. Spread Options

**Location:** `src/models/multi_asset/spread.py`

**Components:**
- `SpreadParameters`: Parameters for spread options
- `spread_call_mc()`: MC pricer for spread calls
- `spread_put_mc()`: MC pricer for spread puts
- `kirk_spread_call()`: Kirk's approximation for calls
- `kirk_spread_put()`: Kirk's approximation for puts
- `margrabe_exchange()`: Exact formula for exchange options (K=0)

**Payoff:**
- Call: max(S₁(T) - S₂(T) - K, 0)
- Put: max(K - (S₁(T) - S₂(T)), 0)

**Features:**
- Monte Carlo pricing
- Kirk's closed-form approximation
- Margrabe's formula for exchange options

**Tests:** 17 unit tests passing

### 4. Rainbow Options (Best-of / Worst-of)

**Location:** `src/models/multi_asset/rainbow.py`

**Components:**
- `RainbowParameters`: Parameters for rainbow options
- `best_of_call_mc()`: MC pricer for best-of calls
- `best_of_put_mc()`: MC pricer for best-of puts
- `worst_of_call_mc()`: MC pricer for worst-of calls
- `worst_of_put_mc()`: MC pricer for worst-of puts
- Convenience functions for simplified interfaces

**Payoffs:**
- Best-of Call: max(max(S₁, S₂, ..., Sₙ) - K, 0)
- Worst-of Call: max(min(S₁, S₂, ..., Sₙ) - K, 0)

**Tests:** 16 unit tests passing

---

## Test Summary

**Total Tests:** 62 passing

| Component | Tests |
|-----------|-------|
| Simulation Infrastructure | 17 |
| Basket Options | 12 |
| Spread Options | 17 |
| Rainbow Options | 16 |
| **Total** | 62 |

---

## Architecture

### Directory Structure

```
src/models/multi_asset/
├── __init__.py
├── simulation.py      # Core simulation infrastructure
├── basket.py          # Basket option pricing
├── spread.py          # Spread option pricing
└── rainbow.py         # Rainbow option pricing
```

### Key Design Decisions

1. **Correlation Handling**: Centralized `CorrelationMatrix` class with validation and Cholesky decomposition for efficient correlated sampling.

2. **Efficient Terminal Simulation**: `simulate_terminal()` method for European options avoids storing full paths.

3. **Multiple Pricing Methods**: MC for all products, plus Kirk's approximation and Margrabe's formula for spread options.

4. **Flexible Weights**: Basket options support arbitrary weights (positive or negative).

---

## Usage Examples

### Basket Call

```python
from src.models.multi_asset import basket_call_simple
import numpy as np

# 3-asset basket call
corr = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0]
])

price, std = basket_call_simple(
    spots=[100.0, 100.0, 100.0],
    weights=[0.4, 0.35, 0.25],
    strike=100.0,
    maturity=1.0,
    r=0.05,
    dividends=[0.02, 0.02, 0.02],
    volatilities=[0.2, 0.25, 0.3],
    correlations=corr,
    n_paths=100000,
    seed=42,
)
print(f"Basket Call: {price:.4f} ± {std:.4f}")
```

### Spread Option with Kirk's Approximation

```python
from src.models.multi_asset.spread import SpreadParameters, spread_call_mc, kirk_spread_call

params = SpreadParameters(
    spot1=100.0, spot2=95.0, strike=5.0,
    maturity=0.5, r=0.05, q1=0.02, q2=0.01,
    sigma1=0.2, sigma2=0.25, rho=0.6
)

mc_price, std = spread_call_mc(params, n_paths=100000)
kirk_price = kirk_spread_call(params)

print(f"MC Price: {mc_price:.4f}")
print(f"Kirk's Approximation: {kirk_price:.4f}")
```

### Best-of / Worst-of Options

```python
from src.models.multi_asset import best_of_call_simple, worst_of_call_simple
import numpy as np

corr = np.array([[1.0, 0.5], [0.5, 1.0]])

best_price, _ = best_of_call_simple(
    spots=[100.0, 100.0],
    strike=100.0,
    maturity=1.0,
    r=0.05,
    dividends=[0.02, 0.02],
    volatilities=[0.25, 0.3],
    correlations=corr,
    n_paths=100000,
)

worst_price, _ = worst_of_call_simple(
    spots=[100.0, 100.0],
    strike=100.0,
    maturity=1.0,
    r=0.05,
    dividends=[0.02, 0.02],
    volatilities=[0.25, 0.3],
    correlations=corr,
    n_paths=100000,
)

print(f"Best-of Call: {best_price:.4f}")
print(f"Worst-of Call: {worst_price:.4f}")
print(f"Best-of always >= Worst-of: {best_price >= worst_price}")
```

---

## Key Relationships

### Correlation Effects

| Product | Higher Correlation Effect |
|---------|--------------------------|
| Basket Call | Lower price (less diversification) |
| Spread Option | Lower price (assets move together) |
| Best-of Call | Lower price (best less likely to excel) |
| Worst-of Call | Higher price (worst less likely to crash) |

### Price Ordering

For same parameters:
- Best-of Call ≥ Single-asset Call ≥ Worst-of Call
- Worst-of Put ≥ Single-asset Put ≥ Best-of Put

---

## Performance Notes

### Monte Carlo Paths

| Accuracy | Paths | Time (3 assets) |
|----------|-------|-----------------|
| ~5% | 10,000 | <0.1s |
| ~1-2% | 100,000 | ~0.5s |
| <1% | 500,000 | ~2s |

### Tips

- Use `simulate_terminal()` for European options (faster than full paths)
- Kirk's approximation is fast but less accurate for extreme parameters
- Antithetic variates are enabled by default for variance reduction

---

*Document Version: 1.0 | QuantStrata Phase 4.3 | January 2026*
