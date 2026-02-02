# Variance Gamma Model - User Guide

This guide shows how to use the Variance Gamma (VG) model in QuantStrata.

## Overview

The Variance Gamma model is a pure-jump Lévy process constructed as time-changed Brownian motion:

$$X_t = \theta G_t + \sigma W_{G_t}$$

where $G_t$ is a Gamma process that acts as "business time".

**Key features:**
- **Pure jump process** (no diffusion component)
- **Controllable fat tails** via ν parameter
- **Controllable skewness** via θ parameter
- **Only 3 parameters**

## Model Parameters

```python
from src.models.levy import VarianceGammaParameters

params = VarianceGammaParameters(
    theta=-0.1,   # Drift parameter (negative = negative skew)
    sigma=0.2,    # Volatility parameter
    nu=0.2,       # Variance rate of Gamma time (controls kurtosis)
)
```

**Parameter Guide:**
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `theta` | Skewness control | -0.3 to +0.1 |
| `sigma` | Volatility | 0.15 - 0.35 |
| `nu` | Fat tails (kurtosis) | 0.1 - 0.5 |

## Basic Usage

### 1. Monte Carlo Simulation

```python
from src.models.levy import VarianceGammaDynamics, VarianceGammaParameters

# Define parameters
params = VarianceGammaParameters(
    theta=-0.1,  # Negative skew
    sigma=0.2,   # 20% vol
    nu=0.2,      # Moderate fat tails
)

# Create dynamics with drift (r - q)
dynamics = VarianceGammaDynamics(params=params, drift=0.05)

# Simulate paths
sim = dynamics.simulate(
    spot0=100.0,
    maturity=1.0,
    n_paths=10000,
    n_steps=252,
    seed=42,
)

# Access results
print(f"Terminal spots mean: {sim.terminal_spots.mean():.2f}")
print(f"Average Gamma time: {sim.average_gamma_time:.3f}")
print(f"Maturity: {sim.maturity}")
```

### 2. European Option Pricing

```python
from src.models.levy import vg_european_call, vg_european_put

# Price ATM call via Monte Carlo
call_price = vg_european_call(
    S=100,
    K=100,
    T=1.0,
    r=0.05,
    q=0.02,
    params=params,
    n_paths=100000,
    seed=42,
)

print(f"Call price: {call_price:.4f}")

# Price put
put_price = vg_european_put(
    S=100, K=100, T=1.0, r=0.05, q=0.02,
    params=params, n_paths=100000, seed=42,
)

print(f"Put price: {put_price:.4f}")
```

### 3. Exact Terminal Simulation

For efficient European option pricing:

```python
# Simulate terminal spots directly (single time step)
S_T = dynamics.simulate_terminal(
    spot0=100.0,
    maturity=1.0,
    n_paths=100000,
    seed=42,
)

# Price options
import numpy as np

call_payoffs = np.maximum(S_T - 100, 0)
put_payoffs = np.maximum(100 - S_T, 0)

call_price = np.exp(-0.05) * call_payoffs.mean()
put_price = np.exp(-0.05) * put_payoffs.mean()

print(f"Call: {call_price:.4f}, Put: {put_price:.4f}")
```

### 4. Understanding the Parameters

```python
# Check model properties
print(f"Martingale correction ω: {params.omega:.4f}")
print(f"Variance rate: {params.variance_rate:.4f}")
print(f"Equivalent BS vol: {params.equivalent_bs_vol * 100:.1f}%")
print(f"Skewness: {params.skewness:.4f}")
print(f"Excess kurtosis: {params.excess_kurtosis:.4f}")
```

## Comparing Different Parameters

### Effect of θ (Skewness)

```python
import numpy as np
import matplotlib.pyplot as plt

# Different theta values
thetas = [-0.2, 0.0, 0.2]
colors = ['red', 'black', 'green']

fig, ax = plt.subplots(figsize=(10, 6))

for theta, color in zip(thetas, colors):
    params = VarianceGammaParameters(theta=theta, sigma=0.2, nu=0.2)
    dynamics = VarianceGammaDynamics(params=params, drift=0.05)
    S_T = dynamics.simulate_terminal(spot0=100, maturity=1.0, n_paths=50000, seed=42)
    
    ax.hist(S_T, bins=50, alpha=0.5, color=color, label=f'θ={theta}', density=True)

ax.set_xlabel('Terminal Spot')
ax.set_ylabel('Density')
ax.set_title('Effect of θ on Terminal Distribution')
ax.legend()
plt.show()
```

### Effect of ν (Kurtosis)

```python
nus = [0.1, 0.2, 0.5]

for nu in nus:
    params = VarianceGammaParameters(theta=-0.1, sigma=0.2, nu=nu)
    print(f"ν={nu}: Excess kurtosis = {params.excess_kurtosis:.2f}")
```

## Practical Tips

### 1. Parameter Constraints

The model requires:
```python
1 - theta * nu - 0.5 * sigma**2 * nu > 0
```

If violated, the martingale correction ω is undefined.

### 2. When to Use VG

**Good for:**
- Pricing with fat tails
- Simple 3-parameter smile fit
- Fast terminal simulation

**Not ideal for:**
- Very short-term options (may need diffusion)
- Extreme skew (may need more parameters)

### 3. Comparison with Merton

| Aspect | Merton | Variance Gamma |
|--------|--------|----------------|
| Jumps | Poisson (discrete) | Gamma subordination |
| Diffusion | Yes | No |
| Parameters | 4 | 3 |
| Simulation | Step-by-step or exact | Subordination-based |

## See Also

- [Technical Reference: Variance Gamma](../reference/models/variance_gamma.md)
- [Merton Jump-Diffusion](merton.md)
- [SABR Model](sabr.md)
