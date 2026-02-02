# Bond Options

**Options on Bond Prices**

This document covers bond options - European call and put options on bond prices, commonly priced using the Black76 model.

---

## Overview

A **bond option** gives the holder the right, but not the obligation, to buy (call) or sell (put) a bond at a specified strike price on the expiry date.

### When to Use Bond Options

| Scenario | Option Type | Benefit |
|----------|-------------|---------|
| Protect against rising rates | Buy put | Limits downside if rates rise |
| Protect against falling rates | Buy call | Captures upside if rates fall |
| Generate income | Sell covered options | Premium collection |
| Speculate on volatility | Buy straddle | Profit from large moves |

### Key Characteristics

| Feature | Description |
|---------|-------------|
| Underlying | Bond price (not yield) |
| Exercise | European (expiry only) |
| Settlement | Cash or physical delivery |
| Model | Black76 on forward bond price |

---

## Mathematical Framework

### Payoffs

**Call Option** (right to buy):
$$\text{Payoff} = \max(B_T - K, 0)$$

**Put Option** (right to sell):
$$\text{Payoff} = \max(K - B_T, 0)$$

Where:
- $B_T$ = bond price at expiry
- $K$ = strike price

### Forward Bond Price

The forward bond price accounts for coupons during the option life:

$$F = \frac{B_0 - PV(\text{coupons during option})}{DF(T)}$$

For a zero coupon bond:
$$F = B_0 \times e^{rT}$$

### Black76 Pricing

Bond options use Black76 model on the forward bond price:

**Call:**
$$C = DF(T) \times [F \cdot N(d_1) - K \cdot N(d_2)]$$

**Put:**
$$P = DF(T) \times [K \cdot N(-d_2) - F \cdot N(-d_1)]$$

Where:
$$d_1 = \frac{\ln(F/K) + \frac{\sigma^2 T}{2}}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

And:
- $F$ = forward bond price
- $K$ = strike price
- $\sigma$ = bond price volatility
- $T$ = time to expiry
- $DF(T)$ = discount factor to expiry

### Put-Call Parity

For European bond options:
$$C - P = DF(T) \times (F - K)$$

---

## Greeks

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| **Delta** | $\partial V / \partial F$ | Sensitivity to forward price |
| **Gamma** | $\partial^2 V / \partial F^2$ | Convexity of delta |
| **Vega** | $\partial V / \partial \sigma$ | Sensitivity to volatility |
| **Theta** | $\partial V / \partial t$ | Time decay |
| **Rho** | $\partial V / \partial r$ | Rate sensitivity |

### Greek Properties

| Property | Call | Put |
|----------|------|-----|
| Delta sign | Positive | Negative |
| Delta range | [0, 1] | [-1, 0] |
| Gamma | Always positive | Always positive |
| Vega | Always positive | Always positive |
| Theta | Usually negative | Usually negative |

---

## Volatility

### Bond Price Volatility

Bond options use **price volatility** (not yield volatility):

$$\sigma_{price} \approx D_{mod} \times \sigma_{yield}$$

Where:
- $D_{mod}$ = modified duration of the bond
- $\sigma_{yield}$ = yield volatility

### Volatility Term Structure

Bond option volatility typically exhibits:
- **Downward sloping**: Longer expiries have lower vol (mean reversion)
- **Maturity effect**: Vol decreases as bond approaches maturity
- **Smile**: ITM/OTM options may have higher implied vol

---

## Example

```python
from src.instruments.ir.options.bond import IrBondEuropeanOptionSimple
from src.pricers.ir.european_b76 import IrBondEuropeanOptionB76PricerSimple

# 6-month call on a 10-year bond
call = IrBondEuropeanOptionSimple(
    notional=1_000_000,         # $1M face value
    strike=102.0,               # Strike at 102%
    expiry=0.5,                 # 6 months
    forward_bond_price=103.5,   # Forward at 103.5%
    vol=0.08,                   # 8% bond price vol
    discount_factor=0.975,
    option_type="call",
)

pricer = IrBondEuropeanOptionB76PricerSimple()

# Price
pv = pricer.price(call)
print(f"Call value: ${pv:,.2f}")

# Greeks
greeks = pricer.greeks(call)
print(f"Delta: {greeks['delta']:,.0f}")
print(f"Gamma: {greeks['gamma']:,.2f}")
print(f"Vega: {greeks['vega']:,.2f}")
```

### Put Option Example

```python
# Put option with same parameters
put = IrBondEuropeanOptionSimple(
    notional=1_000_000,
    strike=102.0,
    expiry=0.5,
    forward_bond_price=103.5,
    vol=0.08,
    discount_factor=0.975,
    option_type="put",
)

put_pv = pricer.price(put)

# Verify put-call parity
df = 0.975
F = 103.5
K = 102.0
N = 1_000_000
expected_diff = N * df * (F - K)
actual_diff = pv - put_pv
print(f"Parity check: {abs(actual_diff - expected_diff) < 0.01}")  # True
```

---

## QuantStrata Implementation

### Instruments

```python
# Simple (direct parameters)
IrBondEuropeanOptionSimple(
    notional,           # Number of bonds or face value
    strike,             # Strike price
    expiry,             # Time to expiry (years)
    forward_bond_price, # Forward bond price
    vol,                # Bond price volatility
    discount_factor,    # DF to expiry
    option_type,        # "call" or "put"
)

# Full (market data lookup)
IrBondEuropeanOption(
    notional,
    strike,
    expiry,
    underlying_maturity,        # Bond maturity (from today)
    underlying_coupon_rate,     # Coupon rate (0 for ZC)
    underlying_coupon_frequency,# 1, 2, 4, or 12
    option_type,
    curve_id,                   # Discount curve
    vol_id,                     # Vol surface
)
```

### Pricers

```python
# Simple pricer
IrBondEuropeanOptionB76PricerSimple()

# Market data pricer
IrBondEuropeanOptionB76Pricer()
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `price(option)` | `float` | Option premium |
| `greeks(option)` | `dict` | Delta, gamma, vega, theta, rho |

---

## Comparison with Rate Options

| Feature | Bond Options | Caps/Floors | Swaptions |
|---------|--------------|-------------|-----------|
| Underlying | Bond price | Forward rate | Swap rate |
| Model | Black76 | Black76 | Bachelier/Black76 |
| Volatility | Price vol | Rate vol | Rate vol |
| Settlement | Bond delivery | Cash | Swap entry |
| Exercise | Single | Multiple (caplets) | Single |

---

## Market Conventions

### Strike Quotation

Bond option strikes are typically quoted as:
- **Percentage of par** (e.g., 102.00 means 102% of face)
- **Price** (e.g., $1,020 per $1,000 face)

### Volatility Quotation

- **Price volatility**: Annual standard deviation of price returns
- **Yield volatility**: Often converted using duration

### Common Maturities

| Option Expiry | Common Underlying |
|---------------|-------------------|
| 1 month | On-the-run bonds |
| 3 months | Benchmark bonds |
| 6 months | Various maturities |
| 1 year | Long-dated bonds |

---

## Related Topics

- **[Bonds](bonds.md)** - Zero coupon and fixed rate bonds
- **[Caps & Floors](caps_floors.md)** - Options on forward rates
- **[Swaptions](swaptions.md)** - Options on interest rate swaps
- **[Black76 Model](../reference/models/black76.md)** - Pricing model

---

## Interview Key Points

1. **Black76 on forward price** - Not spot price, accounts for carry
2. **Price vol vs yield vol** - Relationship through duration
3. **Put-call parity** - $C - P = DF(F - K)$
4. **Delta interpretation** - Hedge ratio in underlying bonds
5. **Gamma/Vega** - Always positive for long options
6. **Vol term structure** - Typically downward sloping for bonds
7. **Forward calculation** - Spot minus PV of coupons, divided by DF
