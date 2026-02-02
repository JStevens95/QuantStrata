# Local Volatility Model

## Overview

The Local Volatility (Local Vol) model extends Black-Scholes by making volatility a deterministic function of both spot price and time: σ = σ(S, t). This allows the model to:

- **Fit all vanilla prices exactly** by construction
- **Remain arbitrage-free** for European options
- **Provide a complete market** model

QuantStrata provides Local Vol Finite Difference pricers for FX options.

## Mathematical Framework

### Dynamics

Under the risk-neutral measure:

$$dS_t = (r - q) S_t \, dt + \sigma(S_t, t) S_t \, dW_t$$

**Key Difference from BSM**: Volatility σ(S, t) is a 2D surface, not a constant.

### Dupire's Formula

Given vanilla call prices C(K, T), the unique local vol is:

$$\sigma_{LV}^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}$$

## Parameters

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `local_vol_surface` | σ(S, t) surface | Calibrated from market |
| `spot` | Current spot price | Starting point |
| `rate_d` | Domestic rate | Risk-free rate |
| `rate_f` / `div_yield` | Foreign rate / dividends | Cost of carry |

## Usage Examples

### Creating a Local Vol Surface

```python
from src.models.local_volatility import LocalVolSurface
import numpy as np

# Define a simple parameterized local vol surface
def local_vol_func(S, t, S0=1.10, atm_vol=0.10, skew=-0.05):
    """Simple local vol with linear skew."""
    moneyness = np.log(S / S0)
    vol = atm_vol + skew * moneyness
    return np.maximum(vol, 0.01)  # Floor at 1%

# Create surface
lv_surface = LocalVolSurface(
    spot_grid=np.linspace(0.80, 1.40, 61),
    time_grid=np.array([0.01, 0.1, 0.25, 0.5, 1.0, 2.0]),
    vol_func=local_vol_func,
)

# Query the surface
print(f"LV at S=1.10, t=0.5: {lv_surface(1.10, 0.5):.4f}")
print(f"LV at S=1.00, t=0.5: {lv_surface(1.00, 0.5):.4f}")
print(f"LV at S=1.20, t=0.5: {lv_surface(1.20, 0.5):.4f}")
```

### Calibrating from Market Implied Vols

```python
from src.models.local_volatility import calibrate_dupire

# Market implied vol surface (K, T) → σ_BS
market_strikes = np.array([0.95, 1.00, 1.05, 1.10, 1.15])
market_expiries = np.array([0.25, 0.5, 1.0])
market_ivs = np.array([
    [0.12, 0.11, 0.10, 0.105, 0.11],   # 3M
    [0.115, 0.105, 0.10, 0.102, 0.108], # 6M
    [0.11, 0.10, 0.10, 0.101, 0.105],   # 1Y
])

# Calibrate local vol via Dupire
lv_surface = calibrate_dupire(
    spot=1.10,
    strikes=market_strikes,
    expiries=market_expiries,
    implied_vols=market_ivs,
    rate_d=0.05,
    rate_f=0.03,
)
```

### Pricing with Local Vol FD

```python
from src.pricers.fx.european_localvol_fde import (
    FxLocalVolFdPricer,
    FDConfig,
)
from src.instruments.fx.options.vanilla import FxVanillaOption

# Create option
option = FxVanillaOption(
    option_type="call",
    strike=1.12,
    expiry=0.5,
    notional=1_000_000,
)

# Configure FD grid
fd_config = FDConfig(
    n_spot_steps=200,
    n_time_steps=100,
    spot_min_multiple=0.5,
    spot_max_multiple=2.0,
    theta=0.5,  # Crank-Nicolson
)

# Create pricer
pricer = FxLocalVolFdPricer(
    spot=1.10,
    rate_d=0.05,
    rate_f=0.03,
    local_vol_surface=lv_surface,
    config=fd_config,
)

# Price
price = pricer.price(option)
print(f"Local Vol Price: {price:.6f}")

# Greeks
greeks = pricer.greeks(option)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Vega (local): {greeks['vega']:.4f}")
```

### Comparing Local Vol vs BSM

```python
from src.pricers.fx.european_bsm import vanilla_price as bsm_price

# BSM with ATM implied vol
atm_iv = 0.10
bsm = bsm_price("call", 1.10, 1.12, 0.5, atm_iv, 0.05, 0.03)
print(f"BSM Price (ATM vol): {bsm:.6f}")

# Local Vol should match market prices exactly
lv = price
print(f"Local Vol Price: {lv:.6f}")
```

### Pricing Barrier Options

```python
from src.instruments.fx.options.barrier import FxBarrierOption

# Up-and-out call
barrier_option = FxBarrierOption(
    option_type="call",
    strike=1.12,
    barrier=1.25,
    barrier_type="up_and_out",
    expiry=0.5,
    notional=1_000_000,
)

# Local Vol captures smile effects on barriers
barrier_price = pricer.price(barrier_option)
print(f"Barrier Option (Local Vol): {barrier_price:.6f}")

# Compare with BSM (flat vol)
from src.pricers.fx.european_bsm import barrier_price as bsm_barrier
bsm_bar = bsm_barrier("call", 1.10, 1.12, 1.25, "up_and_out", 0.5, atm_iv, 0.05, 0.03)
print(f"Barrier Option (BSM): {bsm_bar:.6f}")
```

## Local Vol vs Implied Vol

| Aspect | Local Vol σ_LV(S, t) | Implied Vol σ_BS(K, T) |
|--------|---------------------|------------------------|
| **Definition** | Instantaneous vol at (S, t) | Vol matching BS to market |
| **Dependency** | Function of spot and time | Function of strike and maturity |
| **Uniqueness** | Unique from vanilla prices | Unique per option |
| **Interpretation** | What vol will be at (S, t) | Average vol to expiry |

### Key Relationship

$$\sigma_{BS}^2(K, T) \cdot T = \int_0^T \mathbb{E}[\sigma_{LV}^2(S_t, t) | S_T = K] \, dt$$

Implied variance = expected integrated local variance along paths ending at K.

### ATM Relationship

At ATM forward: $\sigma_{LV}(F, T) \approx \sigma_{BS}(F, T)$

## Smile Dynamics

### Local Vol Predicts Flattening

**Known limitation**: Local Vol predicts the smile will flatten as time passes.

This happens because:
1. Local vol encodes today's smile into σ(S, t)
2. As paths diffuse, they sample from different local vols
3. The averaging effect reduces smile steepness

### Implications for Exotics

- **Barrier options**: May misprice due to incorrect forward smile
- **Forward-starting options**: Poor performance
- **Cliquets**: Use with caution

## When to Use Local Vol

### ✅ Use Local Vol For:

1. **European exotics**: Exact vanilla fit is important
2. **Vol surface interpolation**: Smooth arbitrage-free surface
3. **Initial benchmark**: Compare against stochastic vol
4. **Risk management**: P&L explain against vanilla hedge

### ❌ Don't Use Local Vol For:

1. **Forward-starting options**: Wrong forward smile
2. **Products sensitive to smile dynamics**: Variance swaps
3. **Long-dated barrier options**: Path-dependency issues

## Calibration Considerations

### Numerical Derivatives

Dupire's formula requires derivatives of call prices:
- $\partial C/\partial T$: Time derivative
- $\partial C/\partial K$: Strike derivative (delta-like)
- $\partial^2 C/\partial K^2$: Strike convexity (gamma-like)

### Handling Edge Cases

**Negative Local Variance:**
- Indicates calendar arbitrage in input surface
- Clamp to minimum: σ_LV = max(σ_LV, σ_min)

**Numerical Instability:**
- Near ATM with short maturity
- Use implied vol formulation instead

### Arbitrage-Free Input

Ensure input surface satisfies:
1. **No calendar arbitrage**: Total variance increasing in T
2. **No butterfly arbitrage**: $\partial^2 C/\partial K^2 > 0$

## Finite Difference Implementation

### Grid Considerations

```python
# Recommended grid setup
fd_config = FDConfig(
    n_spot_steps=200,     # Spatial resolution
    n_time_steps=100,     # Temporal resolution
    spot_min_multiple=0.3,  # S_min = 0.3 * S0
    spot_max_multiple=3.0,  # S_max = 3.0 * S0
    theta=0.5,              # Crank-Nicolson (stable, 2nd order)
)
```

### Local Vol Lookup

At each grid point, query the local vol surface:
```python
for j in range(n_spot):
    for n in range(n_time):
        sigma[j, n] = local_vol_surface(S[j], t[n])
```

## Interview Key Points

1. **Definition**: σ(S, t) is deterministic, not stochastic
2. **Dupire**: Unique local vol from vanilla prices
3. **Completeness**: Local vol model is complete
4. **ATM**: Local vol ≈ implied vol at ATM forward
5. **Limitation**: Predicts flattening smile dynamics (unrealistic)
6. **Use Case**: European exotic benchmark, vol interpolation

## Common Pitfalls

1. **Calendar Arbitrage**: Input surface must be arbitrage-free
2. **Smile Dynamics**: Don't use for forward-starting products
3. **Extrapolation**: Be careful at boundaries of calibrated region
4. **Negative LV**: Indicates bad input data

## References

1. Dupire, B. (1994). "Pricing with a Smile"
2. Derman, E. and Kani, I. (1994). "The Volatility Smile and Its Implied Tree"
3. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*
