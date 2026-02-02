# Quasi-Monte Carlo (QMC) User Guide

This guide explains how to use Quasi-Monte Carlo methods in QuantStrata for faster-converging option pricing.

---

## Overview

Quasi-Monte Carlo (QMC) replaces pseudo-random numbers with low-discrepancy sequences that fill the sample space more uniformly. This typically provides:

- **Faster convergence**: O(1/N) vs O(1/√N) for standard MC
- **More stable estimates**: Less variance between runs
- **Better efficiency**: Same accuracy with fewer samples

**Best for:**
- Low to moderate dimensions (d < 20)
- Smooth integrands (vanilla options)
- Terminal payoffs (European options)

---

## Quick Start

```python
from src.models.numeric.monte_carlo.qmc import qmc_european_call

# Price a European call with QMC
price, std_error = qmc_european_call(
    spot0=100.0,       # Initial spot
    strike=100.0,      # Strike price
    maturity=1.0,      # Time to maturity
    r=0.05,            # Risk-free rate
    q=0.02,            # Dividend yield
    sigma=0.2,         # Volatility
    n_samples=50000,   # Number of QMC samples
    seed=42,           # Random seed
)

print(f"QMC Price: {price:.4f} ± {std_error:.4f}")
```

---

## Sequence Generators

### Sobol Sequences (Recommended)

Sobol sequences are the most popular for finance applications:

```python
from src.models.numeric.monte_carlo.qmc import SobolRng

# Create a 5-dimensional Sobol generator
rng = SobolRng(
    d=5,              # Dimension
    seed=42,          # Random seed for scrambling
    scramble=True,    # Owen scrambling (recommended)
)

# Generate uniform samples in [0, 1]^d
uniform_samples = rng.uniform(10000)  # Shape: (10000, 5)

# Generate standard normal samples
normal_samples = rng.standard_normals(10000)  # Shape: (10000, 5)

# Generate with antithetic variates (doubles sample size)
antithetic_samples = rng.standard_normals_antithetic(5000)  # Shape: (10000, 5)
```

**Key features:**
- Good for up to ~1000 dimensions
- Owen scrambling provides unbiased error estimates
- Can be reset to regenerate the same sequence

### Halton Sequences

Better for very low dimensions:

```python
from src.models.numeric.monte_carlo.qmc import HaltonRng

rng = HaltonRng(d=3, seed=42, scramble=True)
samples = rng.standard_normals(5000)
```

**Best for:** d ≤ 5 dimensions.

---

## European Option Pricing

### Calls

```python
from src.models.numeric.monte_carlo.qmc import qmc_european_call

price, std_error = qmc_european_call(
    spot0=100,
    strike=105,        # OTM call
    maturity=0.5,
    r=0.05,
    q=0.02,
    sigma=0.25,
    n_samples=100000,
    use_antithetic=True,  # Use antithetic variates
)
```

### Puts

```python
from src.models.numeric.monte_carlo.qmc import qmc_european_put

price, std_error = qmc_european_put(
    spot0=100,
    strike=95,         # OTM put
    maturity=0.5,
    r=0.05,
    q=0.02,
    sigma=0.25,
    n_samples=100000,
)
```

---

## Path Simulation with QMC

For path-dependent options, simulate full paths:

```python
from src.models.numeric.monte_carlo.qmc import qmc_path_simulation

# Simulate GBM paths with Sobol sequences
paths = qmc_path_simulation(
    spot0=100,
    maturity=1.0,
    r=0.05,
    q=0.02,
    sigma=0.2,
    n_paths=10000,
    n_steps=50,
    seed=42,
)

# paths.shape = (10000, 51)  # Including t=0

# Use paths for Asian options, etc.
average_price = paths.mean(axis=1)
asian_call_payoff = np.maximum(average_price - 100, 0)
asian_call_price = np.exp(-0.05 * 1.0) * asian_call_payoff.mean()
```

---

## Comparing QMC vs Standard MC

```python
from src.models.numeric.monte_carlo.qmc import compare_mc_qmc_convergence
from scipy.stats import norm

# Black-Scholes reference price
S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.02, 0.2
d1 = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
bs_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Compare convergence
results = compare_mc_qmc_convergence(
    spot0=S, strike=K, maturity=T, r=r, q=q, sigma=sigma,
    true_price=bs_price,
    sample_sizes=[1000, 5000, 10000, 50000, 100000],
    n_trials=10,
    seed=42,
)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.loglog(results['sample_sizes'], results['mc_errors'], 'b-o', label='Standard MC')
plt.loglog(results['sample_sizes'], results['qmc_errors'], 'r-s', label='QMC (Sobol)')
plt.xlabel('Number of Samples')
plt.ylabel('Absolute Error')
plt.title('MC vs QMC Convergence')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Tips and Best Practices

### When to Use QMC

✅ **Good for:**
- European options (1D problem)
- Basket options (low dimension)
- Path-dependent with few monitoring dates
- Smooth payoffs

❌ **Less effective for:**
- High dimensions (d > 20)
- Very path-dependent (many time steps)
- Discontinuous payoffs (digitals)

### Sample Size Considerations

- QMC converges faster, so fewer samples needed
- For powers of 2, Sobol has best uniformity properties
- 10,000-50,000 samples often sufficient

### Scrambling

Always use scrambling (`scramble=True`) for:
- Unbiased error estimates
- Breaking correlations in higher dimensions
- Better robustness

### Dimension Considerations

| Dimension | Recommendation |
|-----------|----------------|
| 1-5 | Excellent for QMC |
| 5-20 | Good, use Sobol |
| 20-100 | Moderate benefit |
| 100+ | Limited benefit over MC |

---

## Common Issues

### 1. Warning About Powers of 2

```
UserWarning: The balance properties of Sobol' points require n to be a power of 2.
```

**Solution**: Use n = 2^k samples (1024, 4096, 16384, etc.) for optimal uniformity. Otherwise, results are still valid but slightly less optimal.

### 2. High-Dimensional Problems

**Issue**: QMC advantage diminishes.
**Solution**: Consider hybrid approaches or standard MC with variance reduction.

### 3. Path-Dependent Options

**Issue**: Each time step adds a dimension.
**Solution**: Use fewer time steps or Brownian bridge construction.

---

## Advanced: Combining with Other Methods

### QMC + Antithetic

```python
from src.models.numeric.monte_carlo.qmc import SobolRng

rng = SobolRng(d=10, seed=42)
Z = rng.standard_normals_antithetic(5000)  # Returns 10000 samples
```

### QMC for LSM Paths

```python
from src.models.numeric.monte_carlo.qmc import qmc_path_simulation
from src.models.numeric.monte_carlo.lsm import lsm_american_put

# Generate paths with QMC
paths = qmc_path_simulation(
    spot0=100, maturity=1.0, r=0.05, q=0.0, sigma=0.2,
    n_paths=50000, n_steps=50, seed=42
)

# Use LSM with QMC paths
result = lsm_american_put(paths, strike=100, r=0.05, dt=1.0/50)
```

---

## See Also

- [Monte Carlo Methods Reference](../reference/models/monte_carlo_methods.md) - Mathematical details
- [LSM Guide](lsm.md) - American option pricing
- [Importance Sampling Guide](importance_sampling.md) - Variance reduction
- [Advanced MC Tutorial](../tutorials/pricing/advanced_mc_methods.ipynb) - Interactive examples

---

*QuantStrata User Guide | January 2026*
