# Interest Rate Caps and Floors

**Protection Against Rate Movements**

This document covers Interest Rate Caps and Floors - option-based interest rate derivatives that provide protection against adverse rate movements while preserving favorable outcomes.

---

## Overview

**Caps** and **Floors** are portfolios of interest rate options (caplets/floorlets) that protect against rate movements over multiple periods.

- **Cap**: Protection against rising rates (portfolio of call options on rates)
- **Floor**: Protection against falling rates (portfolio of put options on rates)

### When to Use Caps/Floors

| Scenario | Instrument |
|----------|------------|
| Floating-rate borrower wants upside protection | Buy Cap |
| Floating-rate investor wants downside protection | Buy Floor |
| Reduce hedging cost | Sell Floor (to fund Cap purchase) |
| Range-bound rate view | Collar (buy Cap, sell Floor) |

### Key Characteristics

| Feature | Cap/Floor | FRA/Swap |
|---------|-----------|----------|
| Payoff | Non-linear (option) | Linear |
| Downside | Limited (premium) | Unlimited |
| Upside | Unlimited | Locked |
| Cost | Upfront premium | Zero at par |

---

## Mathematical Framework

### Caplet (Single Period Call on Rate)

A **caplet** pays the holder if the floating rate $L$ exceeds the strike rate $K$:

$$\text{Caplet Payoff} = N \times \tau \times \max(L - K, 0)$$

Paid at $T_{end}$, the present value is:

$$PV_{caplet} = N \times \tau \times DF(T_{end}) \times \mathbb{E}[\max(L - K, 0)]$$

### Floorlet (Single Period Put on Rate)

A **floorlet** pays if the floating rate falls below the strike:

$$\text{Floorlet Payoff} = N \times \tau \times \max(K - L, 0)$$

### Cap and Floor

A **Cap** is the sum of caplets over all reset periods:
$$\text{Cap PV} = \sum_{i} \text{Caplet}_i$$

A **Floor** is the sum of floorlets:
$$\text{Floor PV} = \sum_{i} \text{Floorlet}_i$$

### Black76 Pricing

Caplets and floorlets are priced using Black76 on forward rates:

**Caplet:**
$$PV = N \times \tau \times DF(T_{end}) \times [F \cdot N(d_1) - K \cdot N(d_2)]$$

**Floorlet:**
$$PV = N \times \tau \times DF(T_{end}) \times [K \cdot N(-d_2) - F \cdot N(-d_1)]$$

Where:
- $F$ = forward rate for the period
- $d_1 = \frac{\ln(F/K) + \sigma^2 T_{fix}/2}{\sigma \sqrt{T_{fix}}}$
- $d_2 = d_1 - \sigma \sqrt{T_{fix}}$
- $T_{fix}$ = time to fixing date

### Put-Call Parity (Cap-Floor)

$$\text{Cap} - \text{Floor} = \text{Payer Swap}$$

A long cap and short floor at the same strike equals a payer swap.

---

## Greeks (Risk Measures)

### Delta

Sensitivity to forward rate (per caplet/floorlet):
$$\Delta_{caplet} = N \times \tau \times DF \times N(d_1)$$
$$\Delta_{floorlet} = N \times \tau \times DF \times [N(d_1) - 1]$$

### Gamma

Convexity (same for call and put):
$$\Gamma = N \times \tau \times DF \times \frac{n(d_1)}{F \sigma \sqrt{T}}$$

### Vega

Volatility sensitivity:
$$\nu = N \times \tau \times DF \times F \times n(d_1) \times \sqrt{T}$$

### Theta

Time decay (includes rate effect).

### Rho

Discount rate sensitivity:
$$\rho = -T \times PV$$

---

## Volatility Types

### Lognormal (Black) Volatility

- Traditional convention
- Forward rate follows lognormal process
- Used in Black76 model
- Issues when rates are near zero/negative

### Normal (Bachelier) Volatility

- Alternative convention
- Forward rate follows normal process
- Better for low/negative rates
- Increasingly popular post-2008

QuantStrata currently implements Black76 (lognormal vol).

---

## Implementation in QuantStrata

### Instrument Classes

```python
from src.instruments.ir import Cap, CapSimple, Floor, FloorSimple
from src.instruments.ir import Caplet, CapletSimple, Floorlet, FloorletSimple

# Full cap with market data lookup
cap = Cap(
    notional=50_000_000,
    strike=0.05,              # 5% strike
    start_time=0.25,          # First reset in 3 months
    end_time=5.0,             # 5-year cap
    frequency=0.25,           # Quarterly resets
    day_count="ACT/360",
    curve_id=MarketId("IR", "CURVE", "USD.SOFR"),
    vol_id=MarketId("IR", "VOL", "USD.CAPS"),
)

# Individual caplet
caplet = Caplet(
    notional=50_000_000,
    strike=0.05,
    fixing_time=1.0,
    payment_time=1.25,
    day_count="ACT/360",
    curve_id=curve_id,
    vol_id=vol_id,
)
```

### Pricer Classes

```python
from src.pricers.ir import (
    CapBlack76Pricer, CapBlack76PricerSimple,
    FloorBlack76Pricer, FloorBlack76PricerSimple,
    CapletBlack76Pricer, FloorletBlack76Pricer,
)

# Price a cap
cap_pricer = CapBlack76Pricer()
pv = cap_pricer.price(cap, market)
greeks = cap_pricer.greeks(cap, market)

# Price individual caplet
caplet_pricer = CapletBlack76Pricer()
caplet_pv = caplet_pricer.price(caplet, market)
```

### Simple Pricer (Direct Parameters)

```python
# For testing or when forward rates are known
caplet_simple = CapletSimple(
    notional=10_000_000,
    strike=0.05,
    fixing_time=1.0,
    payment_time=1.25,
    accrual_factor=0.25,
    forward_rate=0.055,       # ITM: forward > strike
    vol=0.20,                 # 20% lognormal vol
    discount_factor=0.95,
)

pricer = CapletBlack76PricerSimple()
pv = pricer.price(caplet_simple)
greeks = pricer.greeks(caplet_simple)
```

---

## Practical Examples

### Example 1: Borrower Protection

Company has floating-rate debt. Buy 5Y cap at 5% to limit maximum rate.

```python
cap = Cap(
    notional=100_000_000,
    strike=0.05,              # Max rate = 5%
    start_time=0.25,
    end_time=5.0,
    frequency=0.25,
    curve_id=usd_curve,
    vol_id=usd_cap_vol,
)

pricer = CapBlack76Pricer()
premium = pricer.price(cap, market)
# Pay premium upfront for rate protection
```

### Example 2: Investor Floor

Money market fund wants to guarantee minimum return.

```python
floor = Floor(
    notional=500_000_000,
    strike=0.03,              # Min return = 3%
    start_time=0.25,
    end_time=2.0,
    frequency=0.25,
    curve_id=usd_curve,
    vol_id=usd_cap_vol,
)
```

### Example 3: Collar (Zero-Cost Structure)

Buy cap, sell floor to reduce premium:

```python
# Long cap (protection)
long_cap = Cap(notional=50_000_000, strike=0.06, ...)

# Short floor (finance the cap)
short_floor = Floor(notional=50_000_000, strike=0.04, ...)

cap_pricer = CapBlack76Pricer()
floor_pricer = FloorBlack76Pricer()

cap_premium = cap_pricer.price(long_cap, market)
floor_premium = floor_pricer.price(short_floor, market)

net_premium = cap_premium - floor_premium
# Adjust strikes to achieve zero-cost collar
```

---

## Relationship to Other Instruments

### Cap/Floor vs Swaption

| Feature | Cap/Floor | Swaption |
|---------|-----------|----------|
| Exercise | Multiple periods | Single exercise |
| Underlying | Forward rates | Swap rate |
| Settlement | Periodic | Swap entry |

### Cap-Floor Parity

$$\text{Cap}(K) - \text{Floor}(K) = \text{Payer Swap}(K)$$

This is the option-market equivalent of put-call parity.

### ATM Strike

The **ATM cap/floor strike** is the par swap rate, where:
$$\text{Cap PV} = \text{Floor PV}$$

---

## Day Count Conventions

| Convention | Usage |
|------------|-------|
| ACT/360 | USD caps (money market) |
| ACT/365 | GBP caps |
| 30/360 | Some EUR markets |

---

## Volatility Surface

Cap volatility varies by:
- **Expiry**: Term structure (short vs long dated)
- **Strike**: Smile/skew (ITM vs ATM vs OTM)

QuantStrata supports:
- `FlatVolSurface`: Single volatility for all strikes/expiries
- `GridVolSurface`: Full volatility surface

---

## See Also

- [Forward Rate Agreements](fra.md) - Linear single-period instrument
- [Interest Rate Swaps](irs.md) - Linear multi-period instrument
- [Black76 Model](../reference/models/black76.md) - Pricing model
- [Swaptions](swaptions.md) - Options on swaps (Phase 3.3)
