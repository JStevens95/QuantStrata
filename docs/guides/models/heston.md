# Heston Stochastic Volatility Model

## Overview

The Heston model is the most widely used stochastic volatility model in quantitative finance. Unlike Black-Scholes, volatility itself is a random process, which generates:

- **Volatility Smile**: Different implied vols at different strikes
- **Fat Tails**: More realistic extreme move probabilities
- **Mean Reversion**: Volatility reverts to long-term level

QuantStrata provides Heston Monte Carlo pricers for FX options.

## Mathematical Framework

### Dynamics

Under the risk-neutral measure:

$$dS_t = (r - q) S_t \, dt + \sqrt{V_t} S_t \, dW_t^S$$

$$dV_t = \kappa(\theta - V_t) \, dt + \xi \sqrt{V_t} \, dW_t^V$$

Where:
$$\text{Corr}(dW_t^S, dW_t^V) = \rho \, dt$$

### Key Properties

1. **Mean-reverting variance**: $V_t$ reverts to $\theta$ at speed $\kappa$
2. **Stochastic vol**: Variance is random, not constant
3. **Correlation**: Spot-vol correlation creates skew
4. **Semi-analytic**: Fourier pricing for Europeans

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| `kappa` | κ | Mean reversion speed | 0.5 - 5.0 |
| `theta` | θ | Long-term variance | 0.01 - 0.10 |
| `xi` | ξ | Vol of vol | 0.1 - 1.0 |
| `v0` | V₀ | Initial variance | 0.01 - 0.20 |
| `rho` | ρ | Spot-vol correlation | -0.9 to 0.0 |

### Feller Condition

To ensure variance stays positive:

$$2\kappa\theta > \xi^2$$

**Feller ratio**: $\phi = 2\kappa\theta / \xi^2$
- φ > 1: Variance never touches zero
- φ ≤ 1: Variance can touch zero (but never negative)

## Usage Examples

### FX Vanilla Options - Monte Carlo

```python
from src.pricers.fx.european_heston_mc import (
    FxHestonMcPricer,
    HestonParams,
    MCConfig,
)
from src.instruments.fx.options.vanilla import FxVanillaOption

# Create Heston parameters
heston_params = HestonParams(
    kappa=2.0,       # Mean reversion speed
    theta=0.04,      # Long-term variance (20% vol)
    xi=0.5,          # Vol of vol
    v0=0.04,         # Initial variance
    rho=-0.7,        # Negative correlation (leverage effect)
)

# Check Feller condition
feller_ratio = 2 * heston_params.kappa * heston_params.theta / heston_params.xi**2
print(f"Feller ratio: {feller_ratio:.2f} (>1 = safe)")

# Create option
option = FxVanillaOption(
    option_type="call",
    strike=1.12,
    expiry=0.5,
    notional=1_000_000,
)

# Configure Monte Carlo
mc_config = MCConfig(
    n_paths=100_000,
    n_steps=100,
    seed=42,
    scheme="full_truncation",  # Handles negative variance
)

# Create pricer
pricer = FxHestonMcPricer(
    spot=1.10,
    rate_d=0.05,
    rate_f=0.03,
    heston_params=heston_params,
    config=mc_config,
)

# Price with standard error
estimate = pricer.price_with_estimate(option)
print(f"Heston MC Price: {estimate.mean:.6f} ± {estimate.stderr:.6f}")
print(f"95% CI: [{estimate.ci_lower:.6f}, {estimate.ci_upper:.6f}]")
```

### Comparing Heston vs BSM

```python
from src.pricers.fx.european_bsm import vanilla_price as bsm_price

# BSM price with ATM vol
atm_vol = 0.20  # 20% flat vol
bsm = bsm_price("call", 1.10, 1.12, 0.5, atm_vol, 0.05, 0.03)
print(f"BSM Price (20% vol): {bsm:.6f}")

# Heston captures the smile
heston = estimate.mean
print(f"Heston Price: {heston:.6f}")
print(f"Difference: {(heston - bsm) * 10000:.1f} pips")
```

### Smile Generation

```python
import numpy as np

# Price options at multiple strikes to see the smile
strikes = np.linspace(1.00, 1.20, 11)
heston_prices = []
heston_vols = []

for K in strikes:
    option = FxVanillaOption("call", K, 0.5, 1_000_000)
    price = pricer.price(option)
    heston_prices.append(price)
    
    # Imply BSM vol from Heston price
    from src.pricers.fx.european_bsm import implied_vol
    iv = implied_vol("call", 1.10, K, 0.5, price, 0.05, 0.03)
    heston_vols.append(iv)

# Plot would show the volatility smile/skew
for K, iv in zip(strikes, heston_vols):
    print(f"K={K:.2f}: IV={iv*100:.2f}%")
```

### Greeks via Finite Difference

```python
# Greeks computed via bump-and-reprice
greeks = pricer.greeks(option)

print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Volga (vanna): {greeks.get('volga', 'N/A')}")
```

### Path Simulation

```python
from src.pricers.fx.european_heston_mc import simulate_heston_paths

# Simulate paths for visualization
paths = simulate_heston_paths(
    S0=1.10,
    V0=0.04,
    r=0.05,
    q=0.03,
    kappa=2.0,
    theta=0.04,
    xi=0.5,
    rho=-0.7,
    T=1.0,
    n_paths=10,
    n_steps=252,
    seed=42,
)

# paths['S'] contains spot paths: shape (n_paths, n_steps+1)
# paths['V'] contains variance paths
print(f"Terminal spot range: [{paths['S'][:, -1].min():.4f}, {paths['S'][:, -1].max():.4f}]")
print(f"Terminal variance range: [{paths['V'][:, -1].min():.4f}, {paths['V'][:, -1].max():.4f}]")
```

## Discretization Schemes

### Full Truncation (Default)

```python
# Handles negative variance by clamping
V_pos = np.maximum(V, 0)
V_new = V + kappa * (theta - V_pos) * dt + xi * np.sqrt(V_pos) * sqrt_dt * Z_V
V = np.maximum(V_new, 0)
```

- Most common scheme
- Simple and stable
- Small bias but robust

### Reflection

```python
V_new = V + kappa * (theta - np.abs(V)) * dt + xi * np.sqrt(np.abs(V)) * sqrt_dt * Z_V
V = np.abs(V_new)
```

- Reflects negative values to positive
- Better preserves some distributional properties

### QE (Quadratic-Exponential)

- Advanced scheme by Andersen (2008)
- Exactly matches first two moments
- Most accurate but more complex
- Best for production use

## Parameter Interpretation

### Mean Reversion (κ)

- **High κ** (>3): Fast reversion, short-lived vol spikes
- **Low κ** (<1): Slow reversion, persistent vol regimes
- **Half-life**: $t_{1/2} = \ln(2)/\kappa$

### Long-Term Variance (θ)

- Equilibrium variance level
- Long-term volatility: $\sqrt{\theta}$
- Example: θ = 0.04 → 20% long-term vol

### Vol of Vol (ξ)

- Controls variance randomness
- **High ξ**: More smile curvature, fatter tails
- **Low ξ**: Flatter smile, closer to BSM

### Correlation (ρ)

- **ρ < 0** (typical): Negative correlation
  - When spot ↓, vol ↑ (leverage effect)
  - Creates **negative skew** (downside protection expensive)
- **ρ > 0** (rare): Positive correlation
  - Creates **positive skew**

## Implied Volatility Smile

The Heston model generates smiles through:

1. **Skew** (ρ < 0): Negative correlation → higher vol at low strikes
2. **Curvature** (ξ > 0): Vol of vol → smile convexity
3. **Term Structure**: Smile flattens with maturity

### Short-Maturity Skew

$$\text{Skew} \approx \frac{\rho \xi}{2} \sqrt{T}$$

## Calibration

### Typical Approach

1. Collect vanilla option prices across strikes and expiries
2. Minimize squared IV differences:

$$\min_{\kappa, \theta, \xi, V_0, \rho} \sum_{i,j} w_{ij}(\sigma_{ij}^{model} - \sigma_{ij}^{market})^2$$

3. Use semi-analytic (Fourier) pricing for speed
4. Apply constraints: Feller condition, bounds on ρ

### Initial Guess Strategy

```python
# Reasonable starting point
initial_guess = {
    'theta': atm_var_long_expiry,  # ATM variance at longest maturity
    'v0': atm_var_short_expiry,    # ATM variance at shortest maturity
    'rho': -0.5 * np.sign(skew),   # Sign from market skew
    'kappa': 2.0,                   # Moderate mean reversion
    'xi': 0.3,                      # Start small
}
```

## Interview Key Points

1. **Two Factors**: Spot S and variance V are both stochastic
2. **Variance Process**: CIR (mean-reverting square-root)
3. **Feller Condition**: $2\kappa\theta > \xi^2$ ensures $V > 0$
4. **Correlation**: ρ < 0 creates leverage effect and skew
5. **Semi-Analytic**: Fourier-based pricing is fast and accurate
6. **Full Truncation**: Most common MC scheme

## Common Pitfalls

1. **Feller Violation**: Calibrated params may violate Feller
2. **Scheme Choice**: Euler can give negative variance
3. **Correlation**: Must use Cholesky decomposition correctly
4. **Convergence**: Need many paths for accurate pricing

## When to Use Heston

### ✅ Use Heston For:

- FX options with significant smile
- Equity index options (leverage effect)
- When vol dynamics matter
- Smile-sensitive exotics

### ❌ Don't Use Heston For:

- Simple European vanillas (BSM is faster)
- When smile is flat
- Very short maturities (calibration unstable)

## References

1. Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
2. Andersen, L. (2008). "Efficient Simulation of the Heston Model"
3. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*
