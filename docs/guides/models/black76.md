# Black76 Model

## Overview

The Black76 model (Black's Model) prices options on forwards and futures. It's the standard model for:

- **Interest Rate Derivatives**: Caps, floors, swaptions
- **Commodity Options**: Options on futures
- **FX Forward Options**: Options on forward exchange rates

QuantStrata provides Black76 pricers across FX, Equity, and IR asset classes.

## Mathematical Framework

### Dynamics

Under the T-forward measure, the forward price follows a driftless GBM:

$$dF_t = \sigma F_t \, dW_t^T$$

**Key Insight**: No drift because the forward is a martingale under the forward measure.

### Black76 Formula

**Call:**
$$C = e^{-rT}[F N(d_1) - K N(d_2)]$$

**Put:**
$$P = e^{-rT}[K N(-d_2) - F N(-d_1)]$$

Where:
$$d_1 = \frac{\ln(F/K) + \sigma^2 T/2}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

## Parameters

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `forward` | Forward price | Observable or computed |
| `strike` | Option strike | Same units as forward |
| `expiry` | Time to expiration | Years |
| `vol` | Forward volatility | Annualized |
| `discount_factor` | DF to expiry | $e^{-rT}$ |

## Usage Examples

### FX Forward Options

```python
from src.pricers.fx.european_b76 import (
    vanilla_price,
    vanilla_delta,
    vanilla_vega,
)

# EUR/USD forward option
# 6M forward at 1.1050, option expiry in 3M
price = vanilla_price(
    option_type="call",
    forward=1.1050,          # Forward price
    strike=1.1100,           # Strike
    expiry=0.25,             # 3 months
    discount_factor=0.9875,  # DF to expiry
    vol=0.09,                # 9% forward vol
)
print(f"Forward Option Price: {price:.6f}")

# Greeks
delta = vanilla_delta("call", 1.1050, 1.1100, 0.25, 0.9875, 0.09)
vega = vanilla_vega(1.1050, 1.1100, 0.25, 0.9875, 0.09)
print(f"Delta: {delta:.4f}")
print(f"Vega: {vega:.6f}")
```

### Interest Rate Caplets/Floorlets

```python
from src.pricers.ir.european_b76 import (
    IrEuropeanCapletB76PricerSimple,
    IrEuropeanFloorletB76PricerSimple,
)
from src.instruments.ir.options.cap_floor import CapletSimple, FloorletSimple

# Create caplet on 3M SOFR
# Fixing in 6M, payment in 9M
caplet = CapletSimple(
    fixing_date=0.5,         # 6 months
    payment_date=0.75,       # 9 months (3M tenor)
    strike=0.045,            # 4.5% strike
    notional=10_000_000,     # $10M notional
    day_count_fraction=0.25, # 3M accrual
)

# Price with Black76
pricer = IrEuropeanCapletB76PricerSimple(
    forward_rate=0.0475,     # 4.75% forward rate
    vol=0.35,                # 35% normal vol (or use lognormal)
    discount_factor=0.965,   # DF to payment date
)

price = pricer.price(caplet)
print(f"Caplet Price: ${price:,.2f}")

# Greeks
greeks = pricer.greeks(caplet)
print(f"Delta: {greeks['delta']:,.2f}")
print(f"Vega: {greeks['vega']:,.2f}")
```

### Interest Rate Caps

```python
from src.pricers.ir.european_b76 import IrEuropeanCapB76PricerSimple
from src.instruments.ir.options.cap_floor import CapSimple

# 2-year cap with quarterly resets
cap = CapSimple(
    start_date=0.0,
    end_date=2.0,
    strike=0.05,             # 5% cap rate
    notional=50_000_000,     # $50M
    frequency=4,             # Quarterly
)

# Cap is sum of caplets - each priced with its forward
cap_pricer = IrEuropeanCapB76PricerSimple(
    forward_curve=forward_curve,  # Forward rate curve
    vol_surface=vol_surface,      # Caplet vol surface
    discount_curve=discount_curve,
)

cap_price = cap_pricer.price(cap)
print(f"Cap Price: ${cap_price:,.2f}")
```

### Equity Index Futures Options

```python
from src.pricers.equity.european_b76 import (
    vanilla_price,
    vanilla_greeks,
)

# S&P 500 E-mini futures option
# Futures at 5000, option expiry in 1M
price = vanilla_price(
    option_type="call",
    forward=5000.0,          # Futures price
    strike=5050.0,           # Strike
    expiry=1/12,             # 1 month
    discount_factor=0.9958,  # DF
    vol=0.15,                # 15% vol
)
print(f"Futures Option Price: {price:.2f}")

greeks = vanilla_greeks(
    option_type="call",
    forward=5000.0,
    strike=5050.0,
    expiry=1/12,
    discount_factor=0.9958,
    vol=0.15,
)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
```

### Bond Options (Black76)

```python
from src.pricers.ir.european_b76 import IrBondEuropeanOptionB76PricerSimple
from src.instruments.ir.options.bond import IrBondEuropeanOptionSimple

# Option on a 10Y Treasury
bond_option = IrBondEuropeanOptionSimple(
    option_type="call",
    option_expiry=0.25,      # 3M option
    bond_maturity=10.0,      # 10Y bond
    strike=98.50,            # Strike price
    notional=1_000_000,      # Face value
)

pricer = IrBondEuropeanOptionB76PricerSimple(
    forward_bond_price=99.25,  # Forward bond price
    vol=0.05,                   # 5% price vol
    discount_factor=0.9875,
)

price = pricer.price(bond_option)
print(f"Bond Option Price: ${price:,.2f}")
```

## Black76 vs Black-Scholes

| Aspect | Black-Scholes | Black76 |
|--------|---------------|---------|
| **Underlying** | Spot price $S$ | Forward price $F$ |
| **Drift** | $(r-q)S$ | Zero (martingale) |
| **Input** | Spot + rates | Forward + DF |
| **Use case** | Spot options | Forward/futures options |

### Relationship

$$F = S \cdot e^{(r-q)T}$$

When you substitute this into BSM, you get Black76.

## Greeks

| Greek | Formula | Notes |
|-------|---------|-------|
| **Delta** | $e^{-rT} N(d_1)$ | Forward delta |
| **Gamma** | $\frac{e^{-rT} n(d_1)}{F \sigma \sqrt{T}}$ | Same for call/put |
| **Vega** | $e^{-rT} F \sqrt{T} \cdot n(d_1)$ | Same for call/put |
| **Theta** | Complex | Includes $rV$ term |

## Common Applications

### 1. Interest Rate Caps/Floors

Caps are portfolios of caplets, each priced with Black76:

$$\text{Caplet} = \tau \cdot DF_{pay} \cdot [F \cdot N(d_1) - K \cdot N(d_2)]$$

### 2. Swaptions

Receiver/payer swaptions using forward swap rate:

$$\text{Payer} = A \cdot [S \cdot N(d_1) - K \cdot N(d_2)]$$

Where $A$ is the annuity factor.

### 3. Commodity Futures Options

Standard model for NYMEX, ICE options on:
- Crude oil (WTI, Brent)
- Natural gas
- Precious metals

## Volatility Conventions

### Log-Normal Vol (Standard Black76)
- Expressed as percentage (e.g., 20%)
- Vol of $\ln(F)$

### Normal Vol (Bachelier)
- Expressed in same units as forward
- Better for low/negative rates

**Approximate conversion:**
$$\sigma_{normal} \approx \sigma_{lognormal} \times F$$

## Interview Key Points

1. **No Drift**: Forward is martingale under forward measure
2. **Relation to BSM**: $F = S e^{(r-q)T}$ links them
3. **Applications**: Caps, floors, swaptions, commodity options
4. **Forward Measure**: Numeraire is the ZC bond maturing at $T$
5. **Theta**: Requires discount rate (not just DF)

## Common Pitfalls

1. **Discount Factor**: Use DF to option expiry, not underlying expiry
2. **Vol Quote**: Know if log-normal or normal vol
3. **Forward vs Spot Delta**: Different hedge ratios
4. **Day Count**: Consistent across forward rate and accrual

## References

1. Black, F. (1976). "The Pricing of Commodity Contracts"
2. Hull, J.C. *Options, Futures, and Other Derivatives*
3. Brigo, D. & Mercurio, F. *Interest Rate Models*
