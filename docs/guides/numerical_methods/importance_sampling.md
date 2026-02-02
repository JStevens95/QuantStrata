# Importance Sampling User Guide

This guide explains how to use importance sampling in QuantStrata for variance reduction, particularly for deep out-of-the-money options.

---

## Overview

Importance sampling changes the sampling distribution to oversample "important" regions (where the payoff is non-zero), then corrects via likelihood ratios. This is especially useful for:

- **Deep OTM options**: Standard MC rarely samples ITM paths
- **Rare event pricing**: Tail risk, extreme scenarios
- **Barrier options**: Near knock-out levels

**Key benefit:** Dramatically reduced variance for low-probability payoffs.

---

## Quick Start

```python
from src.models.numeric.monte_carlo.importance_sampling import is_european_put

# Price a deep OTM put (standard MC struggles here)
result = is_european_put(
    spot0=100.0,       # Initial spot
    strike=70.0,       # Deep OTM (30% below spot)
    maturity=1.0,      # Time to maturity
    r=0.05,            # Risk-free rate
    q=0.02,            # Dividend yield
    sigma=0.2,         # Volatility
    n_samples=50000,   # Number of samples
    seed=42,           # Random seed
)

print(f"Price: {result.price:.6f}")
print(f"Std Error: {result.std_error:.6f}")
print(f"Variance Reduction: {result.variance_reduction:.1f}x")
print(f"Effective Sample Size: {result.effective_sample_size:.0f}")
```

---

## Core Concepts

### Mean-Shift Importance Sampling

The idea is simple: shift the sampling distribution so that the terminal spot is more likely to be near the strike.

For a put option with S₀ > K:
- Standard MC: Most paths end above K → payoff = 0
- With IS: Shift the mean down so more paths end below K

The correction factor (likelihood ratio) ensures the estimate remains unbiased.

### Optimal Drift Shift

The optimal shift θ* moves the mean of the terminal distribution to the strike:

```python
from src.models.numeric.monte_carlo.importance_sampling import (
    optimal_drift_shift_call,
    optimal_drift_shift_put,
)

# Compute optimal shift
theta_call = optimal_drift_shift_call(
    spot0=100, strike=130, maturity=1.0, r=0.05, q=0.02, sigma=0.2
)
print(f"Optimal shift for OTM call: {theta_call:.3f}")

theta_put = optimal_drift_shift_put(
    spot0=100, strike=70, maturity=1.0, r=0.05, q=0.02, sigma=0.2
)
print(f"Optimal shift for OTM put: {theta_put:.3f}")
```

---

## Result Container

```python
@dataclass
class ImportanceSamplingResult:
    price: float                   # Estimated price
    std_error: float              # Standard error
    variance_reduction: float     # Ratio of standard MC variance to IS variance
    effective_sample_size: float  # ESS (measures weight concentration)
    drift_shift: float           # The drift shift used
    n_samples: int               # Number of samples
```

### Interpreting Results

- **variance_reduction > 1**: IS is more efficient than standard MC
- **variance_reduction = 10**: IS achieves same accuracy with 10x fewer samples
- **effective_sample_size**: Higher is better; low ESS indicates weight concentration

---

## Pricing European Options

### OTM Puts

```python
from src.models.numeric.monte_carlo.importance_sampling import is_european_put

# Deep OTM put
result = is_european_put(
    spot0=100,
    strike=70,         # 30% OTM
    maturity=1.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    n_samples=100000,
    seed=42,
)

print(f"Deep OTM Put Price: {result.price:.6f}")
print(f"Variance Reduction: {result.variance_reduction:.1f}x")
```

### OTM Calls

```python
from src.models.numeric.monte_carlo.importance_sampling import is_european_call

# Deep OTM call
result = is_european_call(
    spot0=100,
    strike=150,        # 50% OTM
    maturity=1.0,
    r=0.05,
    q=0.02,
    sigma=0.25,
    n_samples=100000,
    seed=42,
)

print(f"Deep OTM Call Price: {result.price:.6f}")
print(f"Variance Reduction: {result.variance_reduction:.1f}x")
```

### Custom Drift Shift

You can override the optimal shift:

```python
result = is_european_put(
    spot0=100,
    strike=70,
    maturity=1.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    n_samples=100000,
    drift_shift=-1.5,   # Custom shift (negative = shift down)
    seed=42,
)
```

---

## Adaptive Importance Sampling

For cases where the optimal shift is uncertain:

```python
from src.models.numeric.monte_carlo.importance_sampling import adaptive_is_european_call

result = adaptive_is_european_call(
    spot0=100,
    strike=130,
    maturity=1.0,
    r=0.05,
    q=0.02,
    sigma=0.3,
    n_samples=100000,
    n_pilot=1000,      # Pilot samples for calibration
    seed=42,
)
```

---

## Comparing IS vs Standard MC

```python
from src.models.numeric.monte_carlo.importance_sampling import compare_is_standard_mc

# Compare for a deep OTM put
comparison = compare_is_standard_mc(
    spot0=100,
    strike=70,         # Deep OTM put
    maturity=1.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    n_samples=100000,
    seed=42,
)

print("=" * 50)
print(f"Black-Scholes Price:  {comparison['bs_price']:.6f}")
print("-" * 50)
print(f"Standard MC Price:    {comparison['mc_price']:.6f}")
print(f"Standard MC Error:    {comparison['mc_std_error']:.6f}")
print(f"MC Absolute Error:    {comparison['mc_error']:.6f}")
print("-" * 50)
print(f"IS Price:             {comparison['is_price']:.6f}")
print(f"IS Std Error:         {comparison['is_std_error']:.6f}")
print(f"IS Absolute Error:    {comparison['is_error']:.6f}")
print("-" * 50)
print(f"Variance Reduction:   {comparison['variance_reduction']:.1f}x")
print(f"Drift Shift Used:     {comparison['drift_shift']:.3f}")
```

---

## Visualization

### Variance Reduction by Moneyness

```python
import numpy as np
import matplotlib.pyplot as plt
from src.models.numeric.monte_carlo.importance_sampling import is_european_put

strikes = [50, 60, 70, 80, 90, 100]
variance_reductions = []

for K in strikes:
    result = is_european_put(
        spot0=100, strike=K, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
        n_samples=50000, seed=42
    )
    variance_reductions.append(result.variance_reduction)

plt.figure(figsize=(10, 6))
plt.bar(strikes, variance_reductions)
plt.xlabel('Strike Price')
plt.ylabel('Variance Reduction Factor')
plt.title('IS Variance Reduction by Strike (Put, S₀=100)')
plt.grid(axis='y')
plt.show()
```

---

## Tips and Best Practices

### When IS is Most Effective

| Scenario | Expected Variance Reduction |
|----------|---------------------------|
| ATM options | ~1x (no benefit) |
| 10-20% OTM | 2-5x |
| 30% OTM | 5-20x |
| 50%+ OTM | 10-100x |

### Effective Sample Size (ESS)

Monitor ESS to ensure weights aren't too concentrated:

```python
# If ESS is much lower than n_samples, weights are concentrated
if result.effective_sample_size < result.n_samples * 0.1:
    print("Warning: Weight concentration detected. Consider adjusting shift.")
```

### Drift Shift Tuning

- Optimal shift moves mean to strike
- Too large a shift can increase variance (overcorrection)
- For extreme OTM, adaptive IS may help

### Combining with Other Techniques

IS works well with:
- Antithetic variates (both applied)
- QMC (low-discrepancy + IS)
- Control variates

---

## Common Issues

### 1. Variance Reduction < 1

**Cause**: IS is hurting, not helping.
**Possible reasons**:
- Option is ATM or ITM (IS not beneficial)
- Drift shift is poorly chosen
**Solution**: Use standard MC for ATM/ITM options.

### 2. Very Low Effective Sample Size

**Cause**: Extreme likelihood ratios.
**Possible reasons**:
- Drift shift too aggressive
- Very OTM option with small sample size
**Solution**: Reduce drift shift or increase samples.

### 3. Price is Negative or NaN

**Cause**: Numerical instability from extreme weights.
**Solution**: Cap the drift shift (already done internally for extreme cases).

---

## Mathematical Background

### The Estimator

Under importance sampling with mean shift θ:

$$
\hat{V}_{IS} = \frac{1}{N} \sum_{i=1}^N g(S_T^{(i)}) \cdot L(Z_i)
$$

Where the likelihood ratio is:

$$
L(Z) = \exp\left(-\theta \sigma \sqrt{T} Z + \frac{1}{2}(\theta \sigma \sqrt{T})^2\right)
$$

### Optimal Shift for Options

For a call option, the optimal shift targets:

$$
\theta^* = \frac{\ln(K/S_0) - (r-q-\sigma^2/2)T}{\sigma^2 T}
$$

This makes the expected terminal spot equal to the strike.

---

## See Also

- [Monte Carlo Methods Reference](../reference/models/monte_carlo_methods.md) - Mathematical details
- [LSM Guide](lsm.md) - American option pricing
- [QMC Guide](qmc.md) - Quasi-Monte Carlo methods
- [Advanced MC Tutorial](../tutorials/pricing/advanced_mc_methods.ipynb) - Interactive examples

---

*QuantStrata User Guide | January 2026*
