# Swaptions

**Options on Interest Rate Swaps**

This document covers swaptions - options that give the holder the right to enter into an interest rate swap at a future date.

---

## Overview

A **swaption** is an option to enter into an interest rate swap at a predetermined rate. It's one of the most actively traded interest rate derivatives.

### Swaption Types

| Type | Right to Enter | Benefits When |
|------|----------------|---------------|
| **Payer Swaption** | Payer swap (pay fixed, receive floating) | Rates rise |
| **Receiver Swaption** | Receiver swap (receive fixed, pay floating) | Rates fall |

### When to Use Swaptions

| Scenario | Instrument |
|----------|------------|
| Hedge floating-rate liability with optionality | Buy payer swaption |
| Protect investment returns with upside | Buy receiver swaption |
| Express volatility views | Straddles/strangles |
| Callable/putable bond hedging | Receiver/payer swaptions |

---

## Mathematical Framework

### Payoff at Expiry

At option expiry $T_{opt}$, the holder can exercise into a swap.

**Payer Swaption Payoff:**
$$\max(S - K, 0) \times A \times N$$

**Receiver Swaption Payoff:**
$$\max(K - S, 0) \times A \times N$$

Where:
- $S$ = swap rate at expiry
- $K$ = strike rate
- $A$ = annuity (PV01) of the underlying swap
- $N$ = notional

### Bachelier (Normal) Model

Swaptions are commonly priced using the Bachelier model, which assumes the swap rate follows normal dynamics:
$$dS = \sigma \, dW$$

**Pricing Formulas:**

$$\text{Payer} = A \times N \times [(F - K) \cdot N(d) + \sigma\sqrt{T} \cdot n(d)]$$

$$\text{Receiver} = A \times N \times [(K - F) \cdot N(-d) + \sigma\sqrt{T} \cdot n(d)]$$

Where:
- $F$ = forward swap rate
- $\sigma$ = normal volatility (in same units as rate)
- $d = (F - K) / (\sigma\sqrt{T})$
- $N(\cdot)$ = standard normal CDF
- $n(\cdot)$ = standard normal PDF

### Why Bachelier for Swaptions?

1. **Negative rates**: Natural handling of negative/zero rates
2. **Stability**: Normal vol more stable than lognormal near zero
3. **Industry standard**: EUR, JPY, CHF markets use normal vol
4. **Additive interpretation**: 50bp vol = 50bp rate uncertainty

---

## Greeks

### Delta

$$\Delta = A \times N \times N(d) \quad \text{(payer)}$$

Sensitivity to forward swap rate. For payer, positive delta (benefit from rising rates).

### Gamma

$$\Gamma = A \times N \times \frac{n(d)}{\sigma\sqrt{T}}$$

Convexity - same for payer and receiver.

### Vega

$$\nu = A \times N \times \sqrt{T} \times n(d)$$

Sensitivity to normal volatility. Often expressed "per 1bp of normal vol":
$$\nu_{bp} = \nu \times 0.0001$$

### Theta

Time decay, includes rate discounting effects.

### Rho

Discount rate sensitivity: $\rho = -T \times PV$

---

## Implementation in QuantStrata

### Instrument Classes

```python
from src.instruments.ir.options.swaption import Swaption, SwaptionSimple

# Simple swaption with direct parameters
swaption = SwaptionSimple(
    notional=10_000_000,
    strike=0.04,                # 4% strike
    option_expiry=1.0,          # 1 year to expiry
    swap_tenor=5.0,             # 5 year underlying swap
    forward_swap_rate=0.042,    # 4.2% forward swap rate
    annuity=4.5,                # PV01 of underlying swap
    vol=0.0060,                 # 60bp normal vol
    discount_factor=0.95,
    swaption_type="payer",      # or "receiver"
    settlement="cash",          # or "physical"
)

# Properties
print(swaption.tenor_description)  # "1Y5Y"
print(swaption.is_in_the_money)    # True if F > K for payer
```

### Pricer Classes

```python
from src.pricers.ir import SwaptionBachelierPricer, SwaptionBachelierPricerSimple

pricer = SwaptionBachelierPricerSimple()

# Pricing
pv = pricer.price(swaption)
greeks = pricer.greeks(swaption)

# Vega per basis point
vega_bp = pricer.vega_bp(swaption)
```

### Market Data Version

```python
# With market data lookup
swaption = Swaption(
    notional=10_000_000,
    strike=0.04,
    option_expiry=1.0,
    swap_start=1.0,
    swap_end=6.0,               # 1Y + 5Y = 6Y
    fixed_frequency=0.5,        # Semi-annual fixed
    floating_frequency=0.25,    # Quarterly floating
    swaption_type="payer",
    curve_id=MarketId("IR", "CURVE", "USD.SOFR"),
    vol_id=MarketId("IR", "VOL", "USD.SWAPTION"),
)

pricer = SwaptionBachelierPricer()
pv = pricer.price(swaption, market)
forward_rate = pricer.forward_swap_rate(swaption, market)
annuity = pricer.annuity(swaption, market)
```

---

## Practical Examples

### Example 1: Hedge Future Borrowing

Company needs to borrow $100M in 1 year at 5-year fixed. Buy payer swaption.

```python
hedge = SwaptionSimple(
    notional=100_000_000,
    strike=0.05,               # Lock in 5% max
    option_expiry=1.0,
    swap_tenor=5.0,
    forward_swap_rate=0.048,
    annuity=4.5,
    vol=0.0060,
    discount_factor=0.95,
    swaption_type="payer",
)

pricer = SwaptionBachelierPricerSimple()
premium = pricer.price(hedge)
print(f"Protection premium: ${premium:,.0f}")
# If rates rise above 5%, exercise and lock in 5%
# If rates stay below 5%, let expire and borrow at market
```

### Example 2: Volatility Trade

Trader thinks implied vol is too low relative to expected moves.

```python
# Buy ATM straddle (payer + receiver)
payer = SwaptionSimple(
    notional=50_000_000, strike=0.04, option_expiry=1.0, swap_tenor=5.0,
    forward_swap_rate=0.04, annuity=4.5, vol=0.0060,
    discount_factor=0.95, swaption_type="payer",
)
receiver = SwaptionSimple(
    notional=50_000_000, strike=0.04, option_expiry=1.0, swap_tenor=5.0,
    forward_swap_rate=0.04, annuity=4.5, vol=0.0060,
    discount_factor=0.95, swaption_type="receiver",
)

straddle_cost = pricer.price(payer) + pricer.price(receiver)
total_vega = pricer.vega_bp(payer) + pricer.vega_bp(receiver)

print(f"Straddle cost: ${straddle_cost:,.0f}")
print(f"Vega per 1bp: ${total_vega:,.0f}")
```

---

## Tenor Notation

Swaptions use standard notation: **ExTY** where:
- E = option expiry
- T = underlying swap tenor

### Examples

| Notation | Meaning |
|----------|---------|
| 1Y5Y | 1 year expiry, 5 year swap |
| 3M10Y | 3 month expiry, 10 year swap |
| 5Y5Y | 5 year expiry, 5 year swap |
| 10Y30Y | 10 year expiry, 30 year swap |

---

## Payer-Receiver Parity

$$\text{Payer} - \text{Receiver} = A \times N \times (F - K)$$

This is the option-market equivalent of put-call parity.

At-the-money forward (F = K):
$$\text{Payer ATM} = \text{Receiver ATM}$$

---

## Settlement Styles

### Cash Settlement

At expiry, the swaption settles to its intrinsic value:
$$\text{Cash Payment} = A \times N \times \max(S - K, 0)$$

where $S$ is the prevailing swap rate at expiry.

### Physical Settlement

At expiry, the holder actually enters into the swap:
- Payer swaption → Enter payer swap at strike K
- Receiver swaption → Enter receiver swap at strike K

---

## Volatility Surfaces

Swaption volatility varies across:

1. **Expiry**: Term structure (short vs long dated)
2. **Tenor**: Swap length (2Y, 5Y, 10Y, 30Y underlying)
3. **Strike**: Smile/skew (ITM vs ATM vs OTM)

Common surface format: "Expiry x Tenor" grid (e.g., 10x10 matrix).

---

## See Also

- [Interest Rate Swaps](irs.md) - The underlying instrument
- [Caps and Floors](caps_floors.md) - Options on forward rates
- [Bachelier Model](../reference/models/bachelier.md) - Pricing model
