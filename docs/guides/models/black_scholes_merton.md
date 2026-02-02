# Black-Scholes-Merton Model

## Overview

The Black-Scholes-Merton (BSM) model is the foundation of modern options pricing. It models the underlying asset price as a geometric Brownian motion with constant volatility. QuantStrata provides BSM pricers for:

- **FX Options**: Vanilla calls/puts, barriers, digitals
- **Equity Options**: European and American vanilla options
- **Multiple Methods**: Analytic, Monte Carlo, and Finite Difference

## Mathematical Framework

### Dynamics

Under the risk-neutral measure, the asset price follows:

$$dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t$$

where:
- $r$: Risk-free rate (or domestic rate for FX)
- $q$: Dividend yield (or foreign rate for FX)
- $\sigma$: Constant volatility

### Key Properties

1. **Log-normal prices**: $S_T$ is log-normally distributed
2. **Closed-form solutions**: Analytic pricing for European vanillas
3. **Greeks available**: Delta, gamma, vega, theta, rho

## Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `spot` | Current asset price | > 0 |
| `strike` | Option strike price | > 0 |
| `expiry` | Time to expiration (years) | > 0 |
| `vol` | Annualized volatility | 0.05 - 1.0 |
| `rate` | Risk-free rate (domestic) | -0.01 to 0.10 |
| `div_yield` / `rate_f` | Dividend/foreign rate | -0.01 to 0.10 |

## Usage Examples

### FX Vanilla Options - Analytic Pricer

```python
from src.pricers.fx.european_bsm import (
    vanilla_price,
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
)

# EUR/USD call option
price = vanilla_price(
    option_type="call",
    spot=1.10,           # EUR/USD spot
    strike=1.12,         # Strike
    expiry=0.25,         # 3 months
    vol=0.08,            # 8% volatility
    rate_d=0.05,         # USD rate (domestic)
    rate_f=0.03,         # EUR rate (foreign)
)
print(f"Call Price: {price:.4f}")

# Greeks
delta = vanilla_delta("call", 1.10, 1.12, 0.25, 0.08, 0.05, 0.03)
gamma = vanilla_gamma(1.10, 1.12, 0.25, 0.08, 0.05, 0.03)
vega = vanilla_vega(1.10, 1.12, 0.25, 0.08, 0.05, 0.03)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega: {vega:.4f}")
```

### FX Vanilla Options - Monte Carlo Pricer

```python
from src.pricers.fx.european_bsm_mc import (
    FxEuropeanVanillaMcPricer,
    MCConfig,
)
from src.instruments.fx.options.vanilla import FxVanillaOption

# Create instrument
option = FxVanillaOption(
    option_type="call",
    strike=1.12,
    expiry=0.25,
    notional=1_000_000,
)

# Configure Monte Carlo
config = MCConfig(
    n_paths=100_000,
    n_steps=100,
    seed=42,
    antithetic=True,
)

# Create pricer
pricer = FxEuropeanVanillaMcPricer(
    spot=1.10,
    vol=0.08,
    rate_d=0.05,
    rate_f=0.03,
    config=config,
)

# Price with standard error
estimate = pricer.price_with_estimate(option)
print(f"MC Price: {estimate.mean:.6f} ± {estimate.stderr:.6f}")
print(f"95% CI: [{estimate.ci_lower:.6f}, {estimate.ci_upper:.6f}]")
```

### FX Vanilla Options - Finite Difference Pricer

```python
from src.pricers.fx.european_bsm_fde import (
    FxEuropeanVanillaFdPricer,
    FDConfig,
)

# Configure FD grid
fd_config = FDConfig(
    n_spot_steps=200,
    n_time_steps=100,
    spot_max_multiple=3.0,
    theta=0.5,  # Crank-Nicolson
)

# Create pricer
fd_pricer = FxEuropeanVanillaFdPricer(
    spot=1.10,
    vol=0.08,
    rate_d=0.05,
    rate_f=0.03,
    config=fd_config,
)

# Price
price = fd_pricer.price(option)
print(f"FD Price: {price:.6f}")

# Greeks from FD solution
greeks = fd_pricer.greeks(option)
print(f"FD Delta: {greeks['delta']:.4f}")
print(f"FD Gamma: {greeks['gamma']:.4f}")
```

### American FX Options - Finite Difference

```python
from src.pricers.fx.american_bsm_fde import FxAmericanVanillaFdPricer
from src.instruments.fx.options.vanilla import FxVanillaOption

# American put option
american_put = FxVanillaOption(
    option_type="put",
    strike=1.12,
    expiry=0.25,
    notional=1_000_000,
    exercise_type="american",
)

# Create American pricer
american_pricer = FxAmericanVanillaFdPricer(
    spot=1.10,
    vol=0.08,
    rate_d=0.05,
    rate_f=0.03,
    config=fd_config,
)

price = american_pricer.price(american_put)
print(f"American Put Price: {price:.6f}")

# Early exercise boundary
boundary = american_pricer.exercise_boundary()
```

### Equity Options - Analytic Pricer

```python
from src.pricers.equity.european_bsm import (
    vanilla_price,
    vanilla_greeks,
)

# Apple call option
price = vanilla_price(
    option_type="call",
    spot=175.0,          # AAPL stock price
    strike=180.0,        # Strike
    expiry=0.5,          # 6 months
    vol=0.25,            # 25% volatility
    rate=0.05,           # Risk-free rate
    div_yield=0.005,     # 0.5% dividend yield
)
print(f"Call Price: ${price:.2f}")

# All Greeks at once
greeks = vanilla_greeks(
    option_type="call",
    spot=175.0,
    strike=180.0,
    expiry=0.5,
    vol=0.25,
    rate=0.05,
    div_yield=0.005,
)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
```

### Equity Options - Monte Carlo

```python
from src.pricers.equity.european_bsm_mc import (
    EquityVanillaEuropeanOptionMcPricer,
    MCConfig,
)
from src.instruments.equity.options.vanilla import EquityVanillaOption

# Create option
option = EquityVanillaOption(
    option_type="call",
    strike=180.0,
    expiry=0.5,
    notional=100,
)

# MC config
config = MCConfig(n_paths=50_000, seed=42, antithetic=True)

# Price
pricer = EquityVanillaEuropeanOptionMcPricer(
    spot=175.0,
    vol=0.25,
    rate=0.05,
    div_yield=0.005,
    config=config,
)
estimate = pricer.price_with_estimate(option)
print(f"MC Price: ${estimate.mean:.2f} ± ${estimate.stderr:.2f}")
```

## Pricing Method Comparison

| Method | Speed | Accuracy | American | Path-Dependent |
|--------|-------|----------|----------|----------------|
| **Analytic** | Very Fast | Exact | No | No |
| **Monte Carlo** | Slow | Controllable | Limited | Yes |
| **Finite Difference** | Medium | High | Yes | Limited |

### When to Use Each Method

- **Analytic**: European vanillas, quick calculations, Greeks
- **Monte Carlo**: Exotic payoffs, path-dependent options, baskets
- **Finite Difference**: American options, early exercise boundaries

## Greeks Summary

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| **Delta (Δ)** | $\partial V/\partial S$ | Hedge ratio |
| **Gamma (Γ)** | $\partial^2 V/\partial S^2$ | Delta sensitivity |
| **Vega (ν)** | $\partial V/\partial \sigma$ | Volatility sensitivity |
| **Theta (Θ)** | $\partial V/\partial t$ | Time decay |
| **Rho (ρ)** | $\partial V/\partial r$ | Rate sensitivity |

## FX Garman-Kohlhagen Extension

For FX options, BSM extends to Garman-Kohlhagen with two rates:

$$C = S e^{-r_f T} N(d_1) - K e^{-r_d T} N(d_2)$$

Where $r_d$ is domestic rate and $r_f$ is foreign rate.

## Common Pitfalls

1. **Rate conventions**: Ensure consistent rate/yield conventions
2. **Day count**: Use proper year fractions for expiry
3. **Volatility**: Must be annualized
4. **Dividends**: Use continuous yield or discrete model appropriately

## Interview Key Points

1. **Model Definition**: GBM with constant volatility
2. **Key Assumption**: Log-normal returns, constant vol
3. **Limitations**: No smile, constant vol unrealistic
4. **Greeks**: Know formulas and interpretations
5. **Put-Call Parity**: $C - P = S e^{-qT} - K e^{-rT}$

## References

1. Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
2. Garman, M. & Kohlhagen, S. (1983). "Foreign Currency Option Values"
3. Hull, J.C. *Options, Futures, and Other Derivatives*
