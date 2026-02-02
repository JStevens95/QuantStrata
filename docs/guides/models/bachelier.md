# Bachelier (Normal) Model

## Overview

The Bachelier model assumes the underlying follows arithmetic Brownian motion (normal distribution) rather than geometric Brownian motion (log-normal). This makes it essential for:

- **Negative Rate Environments**: EUR, CHF, JPY where rates went negative
- **Spread Options**: Spreads can be positive or negative
- **Swaptions**: Market standard post-2015

QuantStrata provides Bachelier pricers across FX, Equity, and IR asset classes.

## Mathematical Framework

### Dynamics

The forward price follows arithmetic Brownian motion:

$$dF_t = \sigma \, dW_t$$

**Key Difference from Black-Scholes**: Volatility is absolute (same units as $F$), not percentage.

### Bachelier Formula

**Call:**
$$C = e^{-rT}[(F - K) N(d) + \sigma\sqrt{T} \cdot n(d)]$$

**Put:**
$$P = e^{-rT}[(K - F) N(-d) + \sigma\sqrt{T} \cdot n(d)]$$

Where:
$$d = \frac{F - K}{\sigma\sqrt{T}}$$

## Parameters

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `forward` | Forward price | Can be negative |
| `strike` | Option strike | Can be negative |
| `expiry` | Time to expiration | Years |
| `vol` | Normal (absolute) volatility | Same units as forward |
| `discount_factor` | DF to expiry | $e^{-rT}$ |

### Volatility Units

**Normal vol is in same units as the underlying:**
- For rates: basis points (e.g., 50 bp = 0.0050)
- For prices: currency units

## Usage Examples

### Interest Rate Swaptions

```python
from src.pricers.ir.european_bch import (
    IrEuropeanSwaptionBchPricerSimple,
)
from src.instruments.ir.options.swaption import SwaptionSimple

# 5Y x 5Y payer swaption (right to pay fixed)
swaption = SwaptionSimple(
    option_type="payer",
    option_expiry=5.0,       # 5Y expiry
    swap_tenor=5.0,          # 5Y underlying swap
    strike=0.025,            # 2.5% strike
    notional=100_000_000,    # $100M notional
)

# Price with Bachelier
pricer = IrEuropeanSwaptionBchPricerSimple(
    forward_swap_rate=0.0275,  # 2.75% forward rate
    vol=0.0045,                 # 45 bp normal vol
    annuity_factor=4.5,         # ~4.5 for 5Y swap
    discount_factor=0.78,       # DF to expiry
)

price = pricer.price(swaption)
print(f"Swaption Price: ${price:,.2f}")

# Greeks
greeks = pricer.greeks(swaption)
print(f"Delta: {greeks['delta']:,.2f}")
print(f"Vega (per bp): {greeks['vega'] * 0.0001:,.2f}")
```

### Negative Rate Caplets

```python
from src.pricers.ir.european_bch import IrEuropeanCapletBchPricerSimple
from src.instruments.ir.options.cap_floor import CapletSimple

# EUR caplet with negative strike
caplet = CapletSimple(
    fixing_date=0.5,
    payment_date=0.75,
    strike=-0.002,           # -0.20% strike (negative!)
    notional=10_000_000,
    day_count_fraction=0.25,
)

pricer = IrEuropeanCapletBchPricerSimple(
    forward_rate=-0.0015,    # -0.15% forward rate
    vol=0.0035,              # 35 bp normal vol
    discount_factor=1.002,   # DF > 1 with negative rates
)

price = pricer.price(caplet)
print(f"Caplet Price: ${price:,.2f}")
```

### FX Options with Bachelier

```python
from src.pricers.fx.european_bch import (
    vanilla_price,
    vanilla_delta,
)

# EUR/CHF option (can handle negative rates)
price = vanilla_price(
    option_type="call",
    forward=0.9650,          # Forward rate
    strike=0.9700,           # Strike
    expiry=0.25,             # 3 months
    discount_factor=1.001,   # DF (CHF negative rates)
    vol=0.0045,              # Normal vol
)
print(f"Option Price: {price:.6f}")

# Delta
delta = vanilla_delta("call", 0.9650, 0.9700, 0.25, 1.001, 0.0045)
print(f"Delta: {delta:.4f}")
```

### Spread Options

```python
from src.pricers.equity.european_bch import vanilla_price

# Calendar spread option (Dec - March)
# Spread can be positive or negative
spread_price = vanilla_price(
    option_type="call",
    forward=2.50,            # Current spread
    strike=3.00,             # Strike spread
    expiry=0.5,
    discount_factor=0.975,
    vol=1.20,                # Spread volatility
)
print(f"Spread Option Price: {spread_price:.4f}")
```

## Bachelier vs Black-Scholes

| Aspect | Black-Scholes | Bachelier |
|--------|---------------|-----------|
| **Dynamics** | $dF = \sigma F dW$ | $dF = \sigma dW$ |
| **Distribution** | Log-normal | Normal |
| **Volatility units** | Percentage | Absolute |
| **Negative F** | Not allowed | Allowed |
| **Skewness** | Positive | Zero |

### Volatility Conversion

**Approximate relationship at ATM:**

$$\sigma_{normal} \approx \sigma_{lognormal} \times F$$

**More precise:**
$$\sigma_{N} = \sigma_{LN} \cdot F \cdot \sqrt{2\pi/T} \cdot \frac{N(d_1) - N(d_2)}{2n(0)}$$

For ATM: $\sigma_N \approx \sigma_{LN} \cdot F$

## Greeks

| Greek | Call Formula | Notes |
|-------|--------------|-------|
| **Delta** | $e^{-rT} N(d)$ | Bounded in $[0, e^{-rT}]$ |
| **Gamma** | $\frac{e^{-rT} n(d)}{\sigma\sqrt{T}}$ | Independent of $F$ |
| **Vega** | $e^{-rT} \sqrt{T} \cdot n(d)$ | Per unit of absolute vol |
| **Theta** | Complex | Time decay |

**Key Property**: Gamma is constant across all forward levels (for fixed vol).

## When to Use Bachelier

### ✅ Use Bachelier For:

1. **Negative rates**: EUR, CHF, JPY swaptions post-2012
2. **Spread options**: Calendar spreads, basis trades
3. **Low/zero forwards**: When $F$ near zero
4. **Market convention**: Swaption markets quote normal vol

### ❌ Don't Use Bachelier For:

1. **Equity options**: Stock prices can't go negative
2. **FX spot options**: Rates typically positive
3. **Commodity options**: Prices positive (usually)

## Market Conventions

### Swaption Market

**Pre-2015**: Quoted in log-normal (Black) vol
**Post-2015**: Quoted in normal (Bachelier) vol in bp

Example quote:
- "5Y5Y ATM at 45bp" means normal vol = 0.0045

### Converting Quotes

```python
def lognormal_to_normal(sigma_ln, forward, expiry):
    """Approximate conversion (ATM)."""
    return sigma_ln * forward

def normal_to_lognormal(sigma_n, forward, expiry):
    """Approximate conversion (ATM)."""
    return sigma_n / forward

# Example
sigma_ln = 0.20  # 20% log-normal
forward = 0.025  # 2.5%
sigma_n = lognormal_to_normal(sigma_ln, forward, 1.0)
print(f"Normal vol: {sigma_n * 10000:.0f} bp")  # ~50 bp
```

## ATM Approximations

**ATM Bachelier price** ($F = K$):

$$C_{ATM} = e^{-rT} \cdot \sigma\sqrt{T} \cdot n(0) \approx 0.399 \cdot e^{-rT} \cdot \sigma\sqrt{T}$$

**Quick estimate**: ATM call ≈ 40% × normal vol × √T × DF

## Interview Key Points

1. **Model Definition**: Arithmetic BM, normal distribution
2. **Key Difference**: Absolute vol (same units as underlying)
3. **Negative Rates**: Essential for EUR/CHF/JPY post-2012
4. **Market Convention**: Swaptions now quoted in normal vol
5. **Constant Gamma**: Unlike BSM, gamma doesn't depend on $F$
6. **Conversion**: $\sigma_N \approx \sigma_{LN} \times F$ (ATM)

## Common Pitfalls

1. **Vol Units**: Normal vol is NOT percentage - it's absolute
2. **Basis Points**: 50bp = 0.0050, not 0.50
3. **Vega Reporting**: Often reported "per bp" in practice
4. **Negative Prices**: Model allows $F < 0$, which is valid for rates

## References

1. Bachelier, L. (1900). "Théorie de la Spéculation"
2. Hagan, P.S. & Woodward, D.E. (1999). "Equivalent Black Volatilities"
3. Brigo, D. & Mercurio, F. *Interest Rate Models*
