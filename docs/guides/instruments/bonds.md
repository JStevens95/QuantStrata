# Bonds (Zero Coupon & Fixed Rate)

**The Foundation of Fixed Income**

This document covers bond instruments - the fundamental building blocks of fixed income markets and yield curve construction.

---

## Overview

A **bond** is a debt instrument where the issuer promises to pay the holder a series of cash flows (coupons) and return the principal (face value) at maturity. Bonds are the most important fixed income instruments, serving as:

- Building blocks for yield curve construction
- Benchmarks for pricing other instruments
- Primary hedging instruments for interest rate risk

### Types of Bonds

| Type | Description | Cash Flows |
|------|-------------|------------|
| **Zero Coupon** | No interim payments | Face value at maturity only |
| **Fixed Rate** | Periodic fixed coupons | Regular coupons + face at maturity |
| **Floating Rate** | Coupons tied to index | Variable coupons + face at maturity |
| **Inflation-Linked** | Coupons adjusted for inflation | Real coupons + adjusted face |

This guide focuses on **Zero Coupon** and **Fixed Rate** bonds.

---

## Zero Coupon Bonds

### Definition

A zero coupon bond pays no coupons - only a single cash flow (face value) at maturity.

### Pricing Formula

$$PV = Face \times DF(T)$$

Where:
- $Face$ = face value (typically 100 or 1000)
- $DF(T)$ = discount factor to maturity
- $T$ = time to maturity in years

### Implied Zero Rate

From the discount factor, we can extract the continuously compounded zero rate:

$$r = -\frac{\ln(DF)}{T}$$

### Example

```python
from src.instruments.ir.linear.bond import IrBondZeroCouponSimple
from src.pricers.ir.bond import IrBondZeroCouponPricerSimple

# 5-year zero coupon bond
bond = IrBondZeroCouponSimple(
    face_value=100.0,
    maturity=5.0,
    discount_factor=0.85,  # ~3.25% zero rate
)

pricer = IrBondZeroCouponPricerSimple()
pv = pricer.price(bond)  # 85.0

# Implied zero rate
print(f"Zero rate: {bond.implied_zero_rate:.4%}")  # ~3.25%
```

### Risk Measures

For a zero coupon bond:

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| **Macaulay Duration** | $D_{mac} = T$ | Weighted avg time = maturity |
| **Modified Duration** | $D_{mod} \approx T$ | Price sensitivity |
| **DV01** | $PV \times D_{mod} \times 0.0001$ | Dollar value of 1bp |
| **Convexity** | $T^2$ | Second-order sensitivity |

---

## Fixed Rate (Coupon) Bonds

### Definition

A fixed rate bond pays periodic coupons at a fixed rate plus the face value at maturity.

### Pricing Formula

$$PV = \sum_{i=1}^{n} C_i \times DF(T_i) + Face \times DF(T_n)$$

Where:
- $C_i = Face \times \frac{coupon\_rate}{frequency}$ = coupon payment
- $DF(T_i)$ = discount factor to coupon date $i$
- $T_n$ = maturity date

### Clean vs Dirty Price

| Price Type | Definition | When Used |
|------------|------------|-----------|
| **Dirty Price** | Full price including accrued interest | Settlement |
| **Clean Price** | Price excluding accrued interest | Quotation |
| **Accrued Interest** | $Face \times rate \times \frac{days}{period}$ | Settlement adjustment |

$$Clean = Dirty - Accrued$$

### Example

```python
from src.instruments.ir.linear.bond import IrBondFixedRateSimple
from src.pricers.ir.bond import IrBondFixedRatePricerSimple

# 5-year annual coupon bond
bond = IrBondFixedRateSimple(
    face_value=100.0,
    coupon_rate=0.05,  # 5% annual coupon
    coupon_times=(1.0, 2.0, 3.0, 4.0, 5.0),
    coupon_dfs=(0.97, 0.94, 0.91, 0.88, 0.85),
    accrued_interest=2.5,  # Half-year accrued
)

pricer = IrBondFixedRatePricerSimple()
dirty_price = pricer.price(bond)
clean_price = pricer.clean_price(bond)

print(f"Dirty price: {dirty_price:.4f}")
print(f"Clean price: {clean_price:.4f}")
print(f"Accrued: {dirty_price - clean_price:.4f}")
```

### Yield to Maturity (YTM)

The **yield to maturity** is the internal rate of return - the single rate that equates PV to market price:

$$Price = \sum_{i=1}^{n} \frac{C_i}{(1+y)^{T_i}} + \frac{Face}{(1+y)^{T_n}}$$

Solving for $y$ requires numerical iteration (Newton-Raphson).

### Risk Measures

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| **Macaulay Duration** | $D_{mac} = \frac{\sum t_i \times CF_i \times DF_i}{PV}$ | Weighted avg time |
| **Modified Duration** | $D_{mod} = -\frac{1}{PV}\frac{dPV}{dy}$ | Price sensitivity |
| **DV01** | $PV \times D_{mod} \times 0.0001$ | Dollar value of 1bp |
| **Convexity** | $\frac{1}{PV}\frac{d^2PV}{dy^2}$ | Curvature |

### Duration Properties

1. **Higher coupon → Lower duration**: More weight on earlier cash flows
2. **Longer maturity → Higher duration**: More distant cash flows
3. **Zero coupon duration = maturity**: All weight on final payment
4. **Duration < maturity**: Always true for coupon bonds

---

## Day Count Conventions

Bonds use various day count conventions for accrual calculations:

| Convention | Description | Typical Use |
|------------|-------------|-------------|
| **ACT/360** | Actual days / 360 | Money markets |
| **ACT/365** | Actual days / 365 | UK gilts |
| **30/360** | 30 days/month, 360 days/year | Corporate bonds |
| **ACT/ACT** | Actual days / actual year | Government bonds |

---

## QuantStrata Implementation

### Instruments

```python
# Zero Coupon - Simple (direct parameters)
IrBondZeroCouponSimple(face_value, maturity, discount_factor)

# Zero Coupon - Full (market data lookup)
IrBondZeroCoupon(face_value, maturity, curve_id)

# Fixed Rate - Simple (direct parameters)
IrBondFixedRateSimple(
    face_value, coupon_rate, coupon_times, coupon_dfs, accrued_interest
)

# Fixed Rate - Full (market data lookup)
IrBondFixedRate(
    face_value, coupon_rate, maturity, frequency, day_count, curve_id
)
```

### Pricers

```python
# Zero Coupon Pricers
IrBondZeroCouponPricerSimple()  # Direct parameters
IrBondZeroCouponPricer()        # Market data lookup

# Fixed Rate Pricers
IrBondFixedRatePricerSimple()   # Direct parameters
IrBondFixedRatePricer()         # Market data lookup
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `price(bond)` | `float` | Dirty price (PV) |
| `clean_price(bond)` | `float` | Clean price (fixed rate only) |
| `greeks(bond)` | `dict` | DV01, duration, convexity |
| `yield_to_maturity(bond, price)` | `float` | YTM (fixed rate only) |

---

## Comparison: Zero Coupon vs Fixed Rate

| Property | Zero Coupon | Fixed Rate |
|----------|-------------|------------|
| Cash flows | Single | Multiple |
| Reinvestment risk | None | Yes |
| Duration | = Maturity | < Maturity |
| Price sensitivity | Maximum | Less than ZC |
| Typical use | Curve building | Investment, hedging |

---

## Related Topics

- **[FRAs](fra.md)** - Single-period rate agreements
- **[Interest Rate Swaps](irs.md)** - Multi-period rate exchanges
- **[Bond Options](bond_options.md)** - Options on bond prices
- **[Caps & Floors](caps_floors.md)** - Options on rates

---

## Interview Key Points

1. **Zero coupon duration equals maturity** - All weight on final payment
2. **Higher coupon → lower duration** - Cash flows weighted earlier
3. **Clean vs dirty** - Clean for quotation, dirty for settlement
4. **DV01** - Dollar value of 1 basis point shift
5. **Convexity benefit** - Bonds gain more when rates fall than they lose when rates rise
6. **YTM assumptions** - Assumes reinvestment at YTM (unrealistic)
