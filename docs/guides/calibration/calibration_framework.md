# Calibration Framework User Guide

This guide explains how to use the QuantStrata calibration framework to calibrate model parameters to market data.

## Quick Start

### Basic Calibration

```python
from src.calibration.core import CalibrationEngine, LBFGSBConfig
from src.calibration.core.objectives import WeightedLeastSquares
import numpy as np

# 1. Define your model function
def model_func(params):
    """Compute model values given parameters."""
    alpha, beta = params
    return alpha * np.array([1, 2, 3]) + beta

# 2. Create objective
objective = WeightedLeastSquares(
    model_func=model_func,
    market_values=np.array([1.5, 3.0, 4.5]),
)

# 3. Create engine and calibrate
engine = CalibrationEngine(optimizer=LBFGSBConfig())
result = engine.calibrate(
    objective=objective,
    initial_params=[0.0, 0.0],
    bounds=[(-10, 10), (-10, 10)],
)

print(f"Calibrated params: {result.params}")
print(f"Final error: {result.objective_value:.6e}")
```

## Calibrating Specific Models

### Heston Stochastic Volatility

Calibrate Heston model to an implied volatility surface:

```python
from src.calibration.stochastic_volatility import calibrate_heston_to_vols
from src.calibration.stochastic_volatility.heston import HestonCalibrationConfig
import numpy as np

# Market data
strikes = np.array([90, 95, 100, 105, 110])
expiries = np.array([0.25, 0.5, 1.0])
market_vols = np.array([
    [0.22, 0.21, 0.20, 0.21, 0.22],  # 3M
    [0.23, 0.22, 0.21, 0.22, 0.23],  # 6M
    [0.24, 0.23, 0.22, 0.23, 0.24],  # 1Y
])

# Calibrate
result = calibrate_heston_to_vols(
    market_vols=market_vols,
    strikes=strikes,
    expiries=expiries,
    spot=100.0,
    r=0.05,
    q=0.02,
    config=HestonCalibrationConfig(
        fix_v0_to_atm=True,    # Set initial variance to ATM vol²
        enforce_feller=True,   # Ensure variance stays positive
        use_global_optimizer=True,  # Use DE for robust search
    ),
)

print(result)
# Output:
# HestonCalibrationResult
#   κ (kappa) = 2.5431
#   θ (theta) = 0.0400 (long-term vol = 20.00%)
#   ξ (xi)    = 0.3521
#   V₀ (v0)   = 0.0400 (initial vol = 20.00%)
#   ρ (rho)   = -0.6234
#   Feller satisfied: True
#   RMSE: 0.0032 (0.32%)
```

### Hull-White Short Rate

Calibrate Hull-White to swaption volatilities:

```python
from src.calibration.short_rate import calibrate_hull_white_to_swaptions
from src.calibration.short_rate.hull_white import HullWhiteCalibrationConfig
import numpy as np

# Yield curve (discount factors)
def df(t):
    r = 0.03  # 3% flat rate
    return np.exp(-r * t)

# Swaption vol grid (normal vols in bp)
expiries = np.array([1.0, 2.0, 5.0, 10.0])
tenors = np.array([5.0, 10.0])
swaption_vols = np.array([
    [0.0045, 0.0050],  # 1Y expiry
    [0.0048, 0.0052],  # 2Y expiry
    [0.0052, 0.0055],  # 5Y expiry
    [0.0055, 0.0058],  # 10Y expiry
])  # ~50bp normal vols

result = calibrate_hull_white_to_swaptions(
    swaption_vols=swaption_vols,
    expiries=expiries,
    tenors=tenors,
    yield_curve_df=df,
    r0=0.03,
    config=HullWhiteCalibrationConfig(vol_type="normal"),
)

print(result)
# Output:
# HullWhiteCalibrationResult
#   a (mean reversion) = 0.0523
#   σ (volatility)     = 0.0089 (89.0 bp)
```

### SABR for FX Smile

```python
from src.calibration.volatility_surface.sabr import (
    calibrate_sabr_to_smile,
    SabrConfig,
)
import numpy as np

# FX smile data (EURUSD 1Y)
forward = 1.10
strikes = np.array([1.00, 1.05, 1.10, 1.15, 1.20])
market_vols = np.array([0.12, 0.10, 0.09, 0.10, 0.11])

params = calibrate_sabr_to_smile(
    forward=forward,
    strikes=strikes,
    market_vols=market_vols,
    expiry=1.0,
    config=SabrConfig(beta=1.0),  # Log-normal SABR for FX
)

print(f"α = {params.alpha:.4f}")
print(f"ρ = {params.rho:.4f}")
print(f"ν = {params.nu:.4f}")
```

### SABR for IR Swaption Smile

```python
from src.calibration.volatility_surface.sabr import (
    calibrate_sabr_to_swaption_smile,
    SabrConfig,
)
import numpy as np

# Swaption smile data (10Y10Y)
forward_swap_rate = 0.03  # 3%
strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
market_vols = np.array([0.0052, 0.0050, 0.0048, 0.0050, 0.0052])  # Normal vols

params = calibrate_sabr_to_swaption_smile(
    strikes=strikes,
    market_vols=market_vols,
    forward_swap_rate=forward_swap_rate,
    expiry=10.0,
    tenor=10.0,
    vol_type="normal",  # Normal SABR for rates
    config=SabrConfig(beta=0.0),  # β=0 for normal SABR
)
```

## Choosing the Right Optimizer

### Local vs Global Optimization

| Optimizer | Use When | Speed |
|-----------|----------|-------|
| `LBFGSBConfig` | Good initial guess available | Fast |
| `DifferentialEvolutionConfig` | Unknown starting region, multiple local minima | Slow |
| `LevenbergMarquardtConfig` | Least-squares problems, known residual structure | Medium |

### Global + Local Strategy

For difficult problems (like Heston), use global search followed by local refinement:

```python
from src.calibration.core.optimizers import create_global_then_local_optimizer

optimizer = create_global_then_local_optimizer(
    global_iters=500,  # DE iterations
)
# polish=True is set by default, so L-BFGS-B refines the result
```

## Adding Constraints

### Soft Constraints via Penalties

Use `PenalizedObjective` to add soft constraints:

```python
from src.calibration.core.objectives import PenalizedObjective, WeightedLeastSquares

# Base objective
base = WeightedLeastSquares(model_func=..., market_values=...)

# Penalty function (returns 0 if constraint satisfied)
def constraint_penalty(params):
    if params[0] < 0:
        return params[0]**2  # Penalize negative values
    return 0.0

# Penalized objective
objective = PenalizedObjective(
    base_objective=base,
    penalty_func=constraint_penalty,
    penalty_weight=1000.0,  # Large weight = hard constraint
)
```

### Box Constraints

Use `bounds` parameter for simple box constraints:

```python
result = engine.calibrate(
    objective=objective,
    initial_params=[1.0, 2.0, 3.0],
    bounds=[
        (0.01, 10.0),   # param 0: must be positive
        (0.001, 0.5),   # param 1: variance range
        (-0.99, 0.99),  # param 2: correlation
    ],
)
```

## Handling Calibration Results

### Checking Convergence

```python
result = engine.calibrate(...)

if result.success:
    print("Calibration converged!")
    print(f"Final objective: {result.objective_value:.6e}")
    print(f"Improvement: {result.improvement_ratio:.1f}x")
else:
    print(f"Warning: {result.message}")
    # Consider using global optimizer or different initial guess
```

### Accessing Diagnostics

```python
# Timing
print(f"Calibration took {result.elapsed_time:.2f}s")
print(f"Function evaluations: {result.n_function_evals}")

# Compare initial vs final
print(f"Initial params: {result.initial_params}")
print(f"Final params: {result.params}")
print(f"Initial error: {result.initial_objective:.6e}")
print(f"Final error: {result.objective_value:.6e}")
```

## Best Practices

### 1. Scale Your Objective

Ensure objective values are in a reasonable range (not too small or large):

```python
# Bad: Objective values ~1e-12
objective = WeightedLeastSquares(model_func, tiny_market_values)

# Good: Scale to reasonable range
scaled_market = market_values * 10000
objective = WeightedLeastSquares(model_func_scaled, scaled_market)
```

### 2. Use Good Initial Guesses

```python
# For Heston, use ATM vol for initial variance
atm_vol = market_vols[len(market_vols)//2, len(strikes)//2]
initial_v0 = atm_vol**2

# For Hull-White, use historical mean reversion estimate
initial_a = 0.1  # Typical value
```

### 3. Validate Results

```python
# After calibration, compute model values and check fit
model_vols = compute_model_vols(result.params)
rmse = np.sqrt(np.mean((model_vols - market_vols)**2))
print(f"RMSE: {rmse:.4f} ({rmse*100:.2f}%)")

# Check for parameter bounds
if result.params[0] > bounds[0][1] * 0.99:
    print("Warning: Parameter hitting upper bound")
```

### 4. Consider Stability

```python
# Use previous calibration as starting point for stability
yesterday_params = load_yesterday_params()

result = calibrate_heston_to_vols(
    ...,
    initial_guess=yesterday_params,
    config=HestonCalibrationConfig(use_global_optimizer=False),
)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Calibration not converging" | Increase `max_iter`, use global optimizer |
| "Parameters at bounds" | Check if bounds are realistic, widen if needed |
| "Large RMSE" | Check data quality, try different model |
| "Slow calibration" | Reduce grid size, use local optimizer |
| "Different results each run" | Set random seed, use deterministic optimizer |

---

*QuantStrata Calibration Guide | Phase 5.1 | January 2026*
