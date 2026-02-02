# Black-Karasinski Model

## Overview

The Black-Karasinski (BK) model is a one-factor short rate model where the **logarithm** of the short rate follows a mean-reverting process. This ensures that interest rates remain strictly positive, making it suitable for environments where negative rates are not desired.

## Mathematical Framework

### Dynamics

The Black-Karasinski model specifies:

$$d(\ln r(t)) = [\theta(t) - a \cdot \ln r(t)] \, dt + \sigma \, dW(t)$$

Or equivalently, defining $x(t) = \ln r(t)$:

$$dx(t) = [\theta(t) - a \cdot x(t)] \, dt + \sigma \, dW(t)$$

where:
- $r(t)$: instantaneous short rate (always positive)
- $x(t) = \ln r(t)$: log of the short rate
- $\theta(t)$: time-dependent drift (fitted to initial term structure)
- $a$: mean reversion speed ($a > 0$)
- $\sigma$: volatility of log-rate ($\sigma > 0$)

### Key Properties

1. **Log-Normal Rates**: Since $r(t) = e^{x(t)}$, rates are always positive
2. **Mean Reversion**: $\ln r(t)$ reverts to $\theta/a$ (long-term log-rate)
3. **Proportional Volatility**: Vol of $r$ is proportional to $r$ (unlike Hull-White)
4. **No Closed-Form Bond Prices**: Requires numerical methods (MC, FDE)

### Distribution

The log-rate $x(t) = \ln r(t)$ is normally distributed:

$$x(t) \sim \mathcal{N}\left(\theta + (x_0 - \theta)e^{-at}, \frac{\sigma^2}{2a}(1 - e^{-2at})\right)$$

As $t \to \infty$:
- Mean of log-rate: $E[\ln r_\infty] = \theta$
- Variance of log-rate: $\text{Var}[\ln r_\infty] = \frac{\sigma^2}{2a}$
- Long-term rate level: $r_\infty \approx e^\theta$

## Comparison with Hull-White

| Aspect | Hull-White | Black-Karasinski |
|--------|------------|------------------|
| **Model** | $dr = (\theta - ar)dt + \sigma dW$ | $d(\ln r) = (\theta - a \ln r)dt + \sigma dW$ |
| **Rate Distribution** | Gaussian (can be negative) | Log-normal (always positive) |
| **Bond Pricing** | Closed-form (affine model) | Numerical (non-affine) |
| **Volatility Structure** | Constant (additive) | Proportional to rate |
| **Mean Reversion** | In rate space | In log-rate space |
| **Industry Use** | General, negative rate environments | Positive rate environments |

## Parameters

### BlackKarasinskiParameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `a` | Mean reversion speed | 0.01 - 0.5 |
| `sigma` | Volatility of log-rate | 0.10 - 0.30 |
| `r0` | Initial short rate (must be > 0) | 0.01 - 0.10 |
| `theta` | Long-term mean for $\ln(r)$ | $\ln(r_0)$ by default |

### Derived Properties

- **Half-life**: $t_{1/2} = \frac{\ln 2}{a}$ (time for log-rate to revert halfway to mean)
- **Long-term volatility**: $\sigma_\infty = \frac{\sigma}{\sqrt{2a}}$
- **Long-term rate**: $r_\infty = e^\theta$

## Usage Examples

### Basic Parameter Setup

```python
from src.models.short_rate.black_karasinski import (
    BlackKarasinskiParameters,
    BlackKarasinskiDynamics,
)

# Create parameters
params = BlackKarasinskiParameters(
    a=0.1,       # Mean reversion speed (10% per year)
    sigma=0.15,  # 15% volatility of log-rate
    r0=0.03,     # 3% initial short rate
    theta=-3.5,  # Long-term log-rate (≈3% rate)
)

# Properties
print(f"Initial log-rate: {params.x0:.4f}")
print(f"Long-term rate: {params.long_term_rate:.4%}")
print(f"Half-life: {params.half_life:.2f} years")
print(f"Long-term vol: {params.long_term_vol:.4f}")
```

### Simulation

```python
# Create dynamics simulator
dynamics = BlackKarasinskiDynamics(params=params)

# Simulate paths
sim = dynamics.simulate(
    maturity=1.0,        # 1 year
    n_paths=10000,       # 10,000 paths
    n_steps=252,         # Daily steps
    scheme="exact",      # Exact OU transition
    seed=42,             # Reproducibility
    antithetic=True,     # Variance reduction
    compute_discount_factors=True,
)

# Access results
print(f"Mean terminal rate: {sim.mean_terminal_rate:.4%}")
print(f"All rates positive: {(sim.rate_paths > 0).all()}")
print(f"Mean discount factor: {sim.discount_factors.mean():.4f}")
```

### Monte Carlo Bond Pricing

```python
from src.models.short_rate.black_karasinski import bk_zc_bond_price_mc

# Price a 1-year zero coupon bond
price = bk_zc_bond_price_mc(
    T=1.0,
    params=params,
    n_paths=100_000,
    seed=42,
)
print(f"ZC Bond Price: {price:.4f}")
```

### Using the IR Pricers

```python
from src.pricers.ir.european_bk_mc import (
    BKMCConfig,
    IrBondZeroCouponBKMCPricerSimple,
)
from src.instruments.ir.linear.bond import IrBondZeroCouponSimple

# Create instrument
bond = IrBondZeroCouponSimple(
    maturity=1.0,
    face_value=100.0,
    discount_factor=0.97,
)

# Configure MC
config = BKMCConfig(
    n_paths=50_000,
    n_steps=100,
    seed=42,
    antithetic=True,
)

# Price with Black-Karasinski
pricer = IrBondZeroCouponBKMCPricerSimple(params=params, config=config)
price = pricer.price(bond)
print(f"Bond Price: {price:.2f}")

# Get full estimate with standard error
estimate = pricer.price_with_estimate(bond)
print(f"Price: {estimate.mean:.4f} ± {estimate.stderr:.4f}")
print(f"95% CI: [{estimate.ci_lower:.4f}, {estimate.ci_upper:.4f}]")

# Greeks
greeks = pricer.greeks(bond)
print(f"Delta: {greeks['delta']:.4f}")
print(f"DV01: {greeks['dv01']:.4f}")
```

## Simulation Schemes

### Exact Scheme (Recommended)

Uses the exact transition distribution of the OU process:

$$x(t+\Delta t) | x(t) \sim \mathcal{N}\left(\theta + (x(t) - \theta)e^{-a\Delta t}, \frac{\sigma^2}{2a}(1 - e^{-2a\Delta t})\right)$$

Then $r(t) = e^{x(t)}$.

### Euler Scheme

Simple Euler-Maruyama discretization:

$$x(t+\Delta t) = x(t) + a(\theta - x(t))\Delta t + \sigma\sqrt{\Delta t} Z$$

where $Z \sim \mathcal{N}(0,1)$.

Use the exact scheme when possible; it's more accurate and stable.

## Numerical Considerations

### No Closed-Form Bond Prices

Unlike Hull-White, Black-Karasinski is **non-affine** and does not have closed-form bond prices. All pricing must use numerical methods:

1. **Monte Carlo**: Simulate paths, compute $\text{DF} = E[e^{-\int_0^T r(s)ds}]$
2. **Finite Difference**: Solve the bond pricing PDE numerically
3. **Tree Methods**: Build a tree for $\ln(r)$ and price backward

### Monte Carlo Variance Reduction

The implementation supports:
- **Antithetic Variates**: Reduces variance by using $-Z$ alongside $Z$
- **Control Variates**: Can be added for further reduction

### Convergence

Typical MC convergence:
- Standard error: $O(1/\sqrt{N})$ where $N$ = number of paths
- For 1bp accuracy: ~10,000-100,000 paths

## When to Use Black-Karasinski

### Advantages

1. **Positive Rates**: Guaranteed positive rates (ideal for historical contexts)
2. **Realistic Volatility**: Vol proportional to rate level (matches market behavior)
3. **Flexible Calibration**: Can fit term structure exactly with time-varying $\theta(t)$

### Disadvantages

1. **No Closed-Form**: Requires numerical methods (slower than Hull-White)
2. **Negative Rate Environments**: Cannot model negative rates
3. **Complexity**: More complex than Vasicek/Hull-White

### Use Cases

- Historical rate modeling (pre-2014 when rates were positive)
- Markets where negative rates are impossible (e.g., some EM markets)
- When proportional volatility is desired
- Alternative model for benchmarking against Hull-White

## Interview Key Points

1. **Model Definition**: BK models log-rate as OU process, ensuring positive rates
2. **Key Difference from HW**: Log-normal vs Gaussian distribution for rates
3. **Non-Affine**: No closed-form bond prices (unlike Hull-White)
4. **Volatility Structure**: Vol of rate is proportional to rate level
5. **Calibration**: θ(t) function fitted to match initial yield curve
6. **Practical Use**: Simulation-based pricing required

## References

1. Black, F. & Karasinski, P. (1991). "Bond and Option Pricing when Short Rates are Lognormal." *Financial Analysts Journal*.
2. Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*. Springer.
3. Hull, J. (2018). *Options, Futures, and Other Derivatives*. Chapter 31.
