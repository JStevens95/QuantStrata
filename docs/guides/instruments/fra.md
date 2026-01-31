# Forward Rate Agreements (FRAs)

**The Simplest Interest Rate Derivative**

This document covers Forward Rate Agreements - linear interest rate derivatives that lock in a borrowing or lending rate for a future period.

---

## Overview

A **Forward Rate Agreement (FRA)** is an OTC contract between two parties to exchange a fixed interest rate for a floating rate on a notional principal for a single future period. No principal is exchanged - only the interest rate difference is settled.

### When to Use FRAs

| Scenario | Use FRA |
|----------|---------|
| Lock in borrowing costs | Corporate treasury hedging future debt |
| Lock in investment returns | Asset managers hedging floating-rate assets |
| Speculate on rate direction | Trading desk taking view on rates |
| Building block for swaps | Single-period swap component |

### Key Characteristics

| Feature | Description |
|---------|-------------|
| Linear payoff | PV is linear in forward rate |
| Single period | One fixing, one payment |
| Cash settled | No principal exchange |
| OTC traded | Customizable terms |

---

## Mathematical Framework

### FRA Payoff

At the fixing date $T_{start}$, the floating rate $L$ is observed. The settlement amount depends on settlement convention:

**Settlement at Payment Date (FRA-in-arrears):**
$$\text{Settlement} = N \times \tau \times (L - K)$$

Paid at $T_{end}$, discounted back to today:
$$PV = N \times \tau \times DF(T_{end}) \times (L - K)$$

**Settlement at Fixing Date (Standard FRA):**
$$\text{Settlement} = \frac{N \times \tau \times (L - K)}{1 + L \times \tau}$$

Where:
- $N$ = notional principal
- $\tau$ = day count fraction for period $[T_{start}, T_{end}]$
- $L$ = floating rate (LIBOR/SOFR) fixing at $T_{start}$
- $K$ = agreed FRA rate (contract rate)
- $DF(T_{end})$ = discount factor to payment date

### Forward Rate

Before fixing, the floating rate $L$ is unknown. We use the **forward rate** $F$ as the expected value:

$$F = \frac{DF(T_{start}) / DF(T_{end}) - 1}{\tau}$$

This gives the pricing formula:
$$PV = N \times \tau \times DF(T_{end}) \times (F - K)$$

### Par Rate

The **par FRA rate** is the contract rate that makes $PV = 0$:
$$K_{par} = F$$

The par rate always equals the forward rate.

---

## Direction Conventions

### Payer FRA (Pay Fixed, Receive Floating)

- **Pays** the fixed rate $K$
- **Receives** the floating rate $L$
- **Benefits** when rates rise ($F > K$)
- **PV** = $N \times \tau \times DF \times (F - K)$

### Receiver FRA (Receive Fixed, Pay Floating)

- **Receives** the fixed rate $K$
- **Pays** the floating rate $L$
- **Benefits** when rates fall ($F < K$)
- **PV** = $N \times \tau \times DF \times (K - F)$

### Sign Convention

| Scenario | Payer PV | Receiver PV |
|----------|----------|-------------|
| $F > K$ (rates up) | Positive | Negative |
| $F = K$ (ATM) | Zero | Zero |
| $F < K$ (rates down) | Negative | Positive |

Note: Payer PV = -Receiver PV (always opposite)

---

## Tenor Notation

FRAs use a standard notation: **"AxB"** where:
- A = months until fixing date
- B = months until payment date

### Examples

| Notation | Fixing | Payment | Period |
|----------|--------|---------|--------|
| 1x4 | 1 month | 4 months | 3 months |
| 3x6 | 3 months | 6 months | 3 months |
| 6x12 | 6 months | 12 months | 6 months |
| 9x12 | 9 months | 12 months | 3 months |

---

## Greeks (Risk Measures)

### Delta (Forward Sensitivity)

$$\Delta = \frac{\partial PV}{\partial F} = N \times \tau \times DF$$

For payer FRA, $\Delta > 0$ (benefit from rate increase).

### DV01 (Dollar Value of 1 Basis Point)

$$DV01 = |N \times \tau \times DF| \times 0.0001$$

Change in PV for a 1bp parallel shift in rates.

### PV01 (Present Value of 1 Basis Point)

Same as DV01 for a single-period instrument.

---

## Implementation in QuantStrata

### Instrument Classes

```python
from src.instruments.ir import ForwardRateAgreement, ForwardRateAgreementSimple

# Market data version (looks up curve)
fra = ForwardRateAgreement(
    notional=10_000_000,
    fixed_rate=0.05,          # 5% contract rate
    fixing_time=0.25,         # 3 months to fixing
    payment_time=0.5,         # 6 months to payment
    day_count="ACT/360",
    direction="payer",
    curve_id=MarketId("IR", "CURVE", "USD.SOFR"),
)

# Simple version (direct parameters)
fra_simple = ForwardRateAgreementSimple(
    notional=10_000_000,
    fixed_rate=0.05,
    fixing_time=0.25,
    payment_time=0.5,
    accrual_factor=0.25,
    forward_rate=0.052,       # Current forward rate
    discount_factor=0.975,
    direction="payer",
)
```

### Pricer Classes

```python
from src.pricers.ir import FRAPricer, FRAPricerSimple

# Price with market data
pricer = FRAPricer()
pv = pricer.price(fra, market)
greeks = pricer.greeks(fra, market)
par_rate = pricer.par_rate(fra, market)

# Price with direct parameters
simple_pricer = FRAPricerSimple()
pv = simple_pricer.price(fra_simple)
```

### Useful Properties

```python
# Tenor description
fra.tenor_description  # Returns "3x6"

# Check if in-the-money
fra_simple.is_in_the_money  # True if payer and F > K

# Par rate
fra_simple.par_rate  # Returns forward rate
```

---

## Practical Examples

### Example 1: Hedging Future Borrowing

A company needs to borrow $10M in 3 months for 3 months. Current 3x6 forward rate is 5.2%.

```python
# Lock in borrowing rate at 5%
hedge = ForwardRateAgreementSimple(
    notional=10_000_000,
    fixed_rate=0.05,
    fixing_time=0.25,
    payment_time=0.5,
    accrual_factor=0.25,
    forward_rate=0.052,
    discount_factor=0.975,
    direction="receiver",  # Receive fixed to offset borrowing
)

pricer = FRAPricerSimple()
pv = pricer.price(hedge)
# PV = 10M × 0.25 × 0.975 × (0.05 - 0.052) = -4,875
# Negative because rates are higher than locked rate
```

### Example 2: Speculating on Rate Rise

Trader believes rates will rise above 5%.

```python
spec = ForwardRateAgreementSimple(
    notional=50_000_000,
    fixed_rate=0.05,
    fixing_time=0.25,
    payment_time=0.5,
    accrual_factor=0.25,
    forward_rate=0.052,
    discount_factor=0.975,
    direction="payer",  # Pay fixed, receive floating
)

pricer = FRAPricerSimple()
pv = pricer.price(spec)
# PV = 50M × 0.25 × 0.975 × (0.052 - 0.05) = +24,375
# Positive because forward rate > fixed rate
```

---

## Relationship to Other Instruments

### FRA vs Interest Rate Swap

| Feature | FRA | IRS |
|---------|-----|-----|
| Periods | Single | Multiple |
| Complexity | Simple | Portfolio of FRAs |
| Use case | Short-term hedging | Long-term hedging |

An IRS can be decomposed into a portfolio of FRAs.

### FRA vs Caplet

| Feature | FRA | Caplet |
|---------|-----|--------|
| Payoff | Linear | Option (max) |
| Downside | Unlimited | Limited (premium) |
| Use case | Lock rate | Protection with upside |

A caplet is a call option on a FRA.

---

## Day Count Conventions

FRAs typically use money market conventions:

| Convention | Usage | Calculation |
|------------|-------|-------------|
| ACT/360 | USD, EUR | Actual days / 360 |
| ACT/365 | GBP | Actual days / 365 |

QuantStrata supports: ACT/360, ACT/365, 30/360

---

## See Also

- [Interest Rate Swaps](irs.md) - Multi-period extension of FRAs
- [Caps and Floors](caps_floors.md) - Options on FRAs
- [Black76 Model](../reference/models/black76.md) - Pricing model for rate options
