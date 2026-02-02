# Hull-White Model

## Overview

The Hull-White (HW) model is a one-factor short rate model where the short rate follows a mean-reverting Gaussian (Ornstein-Uhlenbeck) process. It is the most widely used short rate model in practice due to its:

- **Analytic tractability**: Closed-form solutions for bonds and European options
- **Term structure fitting**: Can match any initial yield curve exactly
- **Simplicity**: Single-factor with intuitive parameters

## Mathematical Framework

### Dynamics

The Hull-White model specifies:

$$dr(t) = [\theta(t) - a \cdot r(t)] \, dt + \sigma \, dW(t)$$

Or with constant θ:

$$dr(t) = a(\theta - r(t)) \, dt + \sigma \, dW(t)$$

where:
- $r(t)$: instantaneous short rate
- $\theta(t)$: time-dependent drift (fitted to initial term structure)
- $a$: mean reversion speed ($a > 0$)
- $\sigma$: short rate volatility ($\sigma > 0$)

### Key Properties

1. **Gaussian Distribution**: r(t) is normally distributed (can go negative)
2. **Mean Reversion**: Rate reverts to θ/a over time
3. **Affine Structure**: Bond prices have closed-form exponential-affine solutions
4. **Analytic Options**: European bond options have closed-form prices

## Comparison with Black-Karasinski

| Feature | Hull-White | Black-Karasinski |
|---------|------------|------------------|
| Rate distribution | Gaussian | Log-normal |
| Negative rates | Allowed | Not possible |
| Bond pricing | Closed-form | Numerical only |
| Volatility | Constant (additive) | Proportional to rate |
| Speed | Fast (analytic) | Slower (MC required) |

## Parameters

### HullWhiteParameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `a` | Mean reversion speed | 0.01 - 0.5 |
| `sigma` | Short rate volatility | 0.005 - 0.02 (50-200 bp) |
| `r0` | Initial short rate | -0.01 to 0.10 |
| `theta` | Long-term rate level | r0 by default |

### Derived Properties

- **Half-life**: $t_{1/2} = \frac{\ln 2}{a}$ (time to revert halfway)
- **Long-term volatility**: $\sigma_\infty = \frac{\sigma}{\sqrt{2a}}$

## Usage Examples

### Basic Parameter Setup

```python
from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    HullWhiteDynamics,
)

# Create parameters
params = HullWhiteParameters(
    a=0.1,       # Mean reversion speed (10% per year)
    sigma=0.01,  # 1% (100 bp) short rate volatility
    r0=0.03,     # 3% initial short rate
    theta=0.04,  # 4% long-term rate
)

# Properties
print(f"Half-life: {params.half_life:.2f} years")
print(f"Long-term vol: {params.long_term_vol:.4f}")
```

### Analytic Bond Pricing

```python
from src.pricers.ir.european_hw import IrBondZeroCouponHWPricerSimple
from src.instruments.ir.linear.bond import IrBondZeroCouponSimple

# Create bond
bond = IrBondZeroCouponSimple(
    maturity=1.0,
    face_value=100.0,
    discount_factor=0.97,
)

# Price analytically
pricer = IrBondZeroCouponHWPricerSimple(params=params)
price = pricer.price(bond)
print(f"Analytic Price: {price:.4f}")
```

### Monte Carlo Pricing

```python
from src.pricers.ir.european_hw_mc import (
    MCConfig,
    IrBondZeroCouponMCPricerSimple,
)

# Configure MC
config = MCConfig(
    n_paths=100_000,
    n_steps=252,
    seed=42,
    antithetic=True,
)

# Price with MC
mc_pricer = IrBondZeroCouponMCPricerSimple(params=params, config=config)
mc_price = mc_pricer.price(bond)

# Get estimate with standard error
estimate = mc_pricer.price_with_estimate(bond)
print(f"MC Price: {estimate.mean:.4f} ± {estimate.stderr:.4f}")
```

### Finite Difference Pricing

```python
from src.pricers.ir.european_hw_fde import (
    FDConfig,
    IrBondZeroCouponFDPricerSimple,
)

# Configure FD
fd_config = FDConfig(
    n_rate_steps=200,
    n_time_steps=100,
    rate_std_devs=5.0,
    theta=0.5,  # Crank-Nicolson
)

# Price with FD
fd_pricer = IrBondZeroCouponFDPricerSimple(params=params, config=fd_config)
fd_price = fd_pricer.price(bond)
print(f"FD Price: {fd_price:.4f}")
```

## Supported Instruments

### Analytic Pricers (`european_hw.py`)

| Pricer | Instrument |
|--------|------------|
| `IrBondZeroCouponHWPricerSimple` | Zero-coupon bonds |
| `IrBondEuropeanOptionHWPricerSimple` | Bond options |
| `IrCapletEuropeanOptionHWPricerSimple` | Caplets |
| `IrFloorletEuropeanOptionHWPricerSimple` | Floorlets |
| `IrCapEuropeanOptionHWPricerSimple` | Caps |
| `IrFloorEuropeanOptionHWPricerSimple` | Floors |
| `IrSwaptionEuropeanOptionHWPricerSimple` | Swaptions (Jamshidian) |

### Monte Carlo Pricers (`european_hw_mc.py`)

| Pricer | Instrument |
|--------|------------|
| `IrBondZeroCouponMCPricerSimple` | Zero-coupon bonds |
| `IrBondEuropeanOptionMCPricerSimple` | Bond options |
| `IrCapletEuropeanOptionMCPricerSimple` | Caplets |
| `IrFloorletEuropeanOptionMCPricerSimple` | Floorlets |
| `IrSwaptionEuropeanOptionMCPricerSimple` | Swaptions |

### Finite Difference Pricers (`european_hw_fde.py`)

| Pricer | Instrument |
|--------|------------|
| `IrBondZeroCouponFDPricerSimple` | Zero-coupon bonds |
| `IrBondEuropeanOptionFDPricerSimple` | Bond options |
| `IrCapletEuropeanOptionFDPricerSimple` | Caplets |
| `IrFloorletEuropeanOptionFDPricerSimple` | Floorlets |

## Greeks

Hull-White pricers provide:
- **Delta**: Rate sensitivity (∂P/∂r)
- **DV01**: Dollar value of 1bp move
- **Gamma**: Second-order rate sensitivity
- **Vega**: Volatility sensitivity

```python
# Get Greeks (analytic)
greeks = pricer.greeks(bond)
print(f"Delta: {greeks['delta']:.4f}")
print(f"DV01: {greeks['dv01']:.4f}")
```

## When to Use Hull-White

### Advantages

1. **Speed**: Closed-form solutions for most instruments
2. **Negative Rates**: Can model EUR/CHF/JPY negative rate environments
3. **Calibration**: Easy to fit to initial yield curve
4. **Well-Understood**: Industry standard with extensive literature

### Disadvantages

1. **Negative Rates**: May produce unrealistic negative rates
2. **Single Factor**: Cannot capture term structure dynamics
3. **Constant Vol**: Rate vol doesn't depend on rate level

### Use Cases

- General-purpose interest rate modeling
- Markets with negative rates
- When analytic speed is important
- Hedging linear rate risk

## Interview Key Points

1. **Model Definition**: HW models short rate as OU process
2. **Affine Property**: Bond prices are exponential-affine in r
3. **Gaussian Rates**: Rates are normally distributed
4. **Calibration**: θ(t) fitted to match initial yield curve
5. **Options**: Jamshidian decomposition for swaptions

## References

1. Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities." *Review of Financial Studies*.
2. Hull, J. & White, A. (1994). "Numerical Procedures for Implementing Term Structure Models I: Single-Factor Models."
3. Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*. Springer.
