# Merton Jump-Diffusion Model - User Guide

This guide shows how to use the Merton jump-diffusion model in QuantStrata.

## Overview

The Merton model extends Black-Scholes by adding Poisson-distributed jumps, capturing:
- **Fat tails**: Market returns have heavier tails than GBM
- **Crash risk**: Sudden large price movements
- **Volatility smile**: Natural smile generation

## Model Parameters

```python
from src.models.jump_diffusion import MertonParameters

params = MertonParameters(
    sigma=0.2,      # Diffusion volatility (20%)
    lambda_=0.5,    # Jump intensity (0.5 jumps/year expected)
    mu_j=-0.1,      # Mean of log-jump (negative = crash-like)
    sigma_j=0.2,    # Std dev of log-jump (20% jump size uncertainty)
)
```

**Parameter Guide:**
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `sigma` | Continuous volatility | 0.15 - 0.35 |
| `lambda_` | Jumps per year | 0.1 - 2.0 |
| `mu_j` | Log-jump mean | -0.2 to +0.1 |
| `sigma_j` | Log-jump std | 0.1 - 0.3 |

## Basic Usage

### 1. Monte Carlo Simulation

```python
from src.models.jump_diffusion import MertonDynamics, MertonParameters

# Define parameters
params = MertonParameters(
    sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2
)

# Create dynamics with drift (r - q)
dynamics = MertonDynamics(params=params, drift=0.05)

# Simulate paths
sim = dynamics.simulate(
    spot0=100.0,
    maturity=1.0,
    n_paths=10000,
    n_steps=252,
    seed=42,
    antithetic=True,
)

# Access results
print(f"Terminal spots mean: {sim.terminal_spots.mean():.2f}")
print(f"Average jumps per path: {sim.average_jumps_per_path:.2f}")
print(f"Paths with jumps: {sim.jump_fraction * 100:.1f}%")
```

### 2. European Option Pricing (Analytic)

```python
from src.models.jump_diffusion.merton import merton_european_call, merton_european_put

# Price ATM call
call_price = merton_european_call(
    S=100,      # Spot price
    K=100,      # Strike
    T=1.0,      # Time to maturity
    r=0.05,     # Risk-free rate
    q=0.02,     # Dividend yield
    sigma=0.2,  # Diffusion vol
    lambda_=0.5,
    mu_j=-0.1,
    sigma_j=0.2,
)

print(f"Call price: {call_price:.4f}")

# Price put
put_price = merton_european_put(
    S=100, K=100, T=1.0, r=0.05, q=0.02,
    sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2,
)

print(f"Put price: {put_price:.4f}")
```

### 3. Implied Volatility Smile

```python
import numpy as np
from src.models.jump_diffusion.merton import merton_implied_vol

strikes = np.linspace(80, 120, 9)
impl_vols = []

for K in strikes:
    iv = merton_implied_vol(
        S=100, K=K, T=0.5, r=0.05, q=0.0,
        sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2,
    )
    impl_vols.append(iv)

# Plot the smile
import matplotlib.pyplot as plt
plt.plot(strikes, np.array(impl_vols) * 100, 'o-')
plt.xlabel('Strike')
plt.ylabel('Implied Vol (%)')
plt.title('Merton Implied Volatility Smile')
plt.show()
```

### 4. Exact Terminal Simulation

For European option pricing, use exact terminal simulation (faster):

```python
# Simulate terminal spots directly (no intermediate steps)
S_T = dynamics.simulate_exact(
    spot0=100.0,
    maturity=1.0,
    n_paths=100000,
    seed=42,
)

# Price call option via MC
payoffs = np.maximum(S_T - 100, 0)
call_mc = np.exp(-0.05 * 1.0) * payoffs.mean()
print(f"MC call price: {call_mc:.4f}")
```

## Practical Tips

### 1. Parameter Interpretation

- **Negative `mu_j`**: Creates downward skew (higher OTM put vols)
- **Higher `lambda_`**: More frequent jumps, fatter tails
- **Higher `sigma_j`**: More uncertainty in jump sizes

### 2. When to Use Merton

**Good for:**
- Crash protection pricing
- Short-dated options with steep skew
- Event risk modeling

**Not ideal for:**
- Very long-dated options (Heston better)
- Path-dependent exotics (may need full simulation)

### 3. Comparison with Black-Scholes

```python
from scipy.stats import norm

# Black-Scholes price
def bs_call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r-q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

bs_price = bs_call(100, 100, 1.0, 0.05, 0.02, 0.2)
merton_price = merton_european_call(100, 100, 1.0, 0.05, 0.02, 0.2, 0.5, -0.1, 0.2)

print(f"BS price: {bs_price:.4f}")
print(f"Merton price: {merton_price:.4f}")
```

## See Also

- [Technical Reference: Merton Jump-Diffusion](../reference/models/merton_jump_diffusion.md)
- [Heston Stochastic Volatility](heston.md)
- [Variance Gamma Model](variance_gamma.md)
