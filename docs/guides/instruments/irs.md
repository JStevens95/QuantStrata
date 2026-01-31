# Interest Rate Swaps (IRS)

**The Most Traded Interest Rate Derivative**

This document covers vanilla Interest Rate Swaps - linear multi-period derivatives that exchange fixed rate payments for floating rate payments.

---

## Overview

An **Interest Rate Swap (IRS)** is an OTC agreement between two parties to exchange fixed and floating interest rate payments on a notional principal over multiple periods. No principal is exchanged - only the net interest payments.

### Market Significance

- **Largest derivatives market**: >$400 trillion notional outstanding globally
- **Benchmark for rates**: Swap rates are key market references
- **Foundation for swaptions**: IRS is the underlying for swaption options

### When to Use IRS

| Scenario | Use IRS |
|----------|---------|
| Convert floating debt to fixed | Corporate treasury liability management |
| Convert fixed assets to floating | Asset-liability matching |
| Express rate views | Trading desk macro positioning |
| Hedge bond portfolios | Duration management |
| Benchmark construction | Curve building instruments |

---

## Mathematical Framework

### Swap Structure

A vanilla IRS consists of two legs:

**Fixed Leg:**
- Pays/receives a fixed rate $K$ on the notional
- Typically semi-annual or annual payments
- Cashflow at time $T_i$: $N \times \tau_i^{fix} \times K$

**Floating Leg:**
- Pays/receives a floating rate (SOFR, EURIBOR, etc.)
- Typically quarterly payments
- Cashflow at time $T_j$: $N \times \tau_j^{flt} \times L_j$

Where:
- $N$ = notional principal
- $\tau$ = day count fraction
- $K$ = fixed rate
- $L_j$ = floating rate for period $j$

### Present Value

**Fixed Leg PV:**
$$PV_{fix} = N \times K \times \sum_{i} \tau_i^{fix} \times DF(T_i)$$

**Floating Leg PV:**
Before fixing, use forward rates $F_j$:
$$PV_{flt} = N \times \sum_{j} \tau_j^{flt} \times DF(T_j) \times (F_j + s)$$

Where $s$ is any spread over the floating rate.

**Total Swap PV (Receiver):**
$$PV = PV_{fix} - PV_{flt} = N \times \sum_{i} \tau_i \times DF_i \times (K - F_i)$$

### Key Quantities

#### Annuity (PV01 Factor)

$$A = \sum_{i} \tau_i^{fix} \times DF(T_i)$$

The annuity is the present value of receiving 1 unit at each fixed payment date. It's fundamental to swap valuation.

#### Par Swap Rate

The fixed rate that makes $PV = 0$:

$$K_{par} = \frac{\sum_{j} \tau_j^{flt} \times DF_j \times F_j}{A} = \frac{PV_{flt}}{N \times A}$$

For a swap starting today with no spread:
$$K_{par} = \frac{1 - DF(T_n)}{A}$$

Where $T_n$ is the final payment date.

#### DV01

Dollar value of a 1 basis point move:
$$DV01 \approx N \times A \times 0.0001$$

For a $100M 5-year swap with $A = 4.5$:
$$DV01 \approx 100M \times 4.5 \times 0.0001 = \$45,000$$

---

## Direction Conventions

### Receiver Swap (Receive Fixed, Pay Floating)

- **Receives** fixed rate payments
- **Pays** floating rate payments
- **Benefits** when rates fall
- **Negative DV01**: loses value when rates rise

**Use case**: Lock in investment yield, hedge floating-rate liabilities

### Payer Swap (Pay Fixed, Receive Floating)

- **Pays** fixed rate payments
- **Receives** floating rate payments
- **Benefits** when rates rise
- **Positive DV01**: gains value when rates rise

**Use case**: Lock in borrowing cost, speculate on rate rises

### Sign Convention

| Scenario | Receiver PV | Payer PV |
|----------|-------------|----------|
| $K > K_{par}$ | Positive | Negative |
| $K = K_{par}$ | Zero | Zero |
| $K < K_{par}$ | Negative | Positive |

---

## Leg Structures

### Standard Frequencies

| Currency | Fixed Leg | Floating Leg |
|----------|-----------|--------------|
| USD | Semi-annual | Quarterly (SOFR) |
| EUR | Annual | Semi-annual (EURIBOR) |
| GBP | Semi-annual | Semi-annual (SONIA) |

### Day Count Conventions

| Leg | Convention |
|-----|------------|
| USD Fixed | 30/360 |
| USD Floating | ACT/360 |
| EUR Fixed | 30/360 |
| EUR Floating | ACT/360 |

---

## Greeks (Risk Measures)

### Delta (Rate Sensitivity)

Aggregate sensitivity to parallel shift in forward rates:
$$\Delta = \frac{\partial PV}{\partial F} = \pm N \times A$$

Sign depends on direction (positive for payer, negative for receiver).

### DV01 (Dollar Value of 01)

$$DV01 = N \times A \times 0.0001$$

**Interpretation**: PV change for 1bp parallel shift.

For a $50M 10-year receiver swap with $A = 8.5$:
- $DV01 = 50M \times 8.5 \times 0.0001 = -\$42,500$
- Negative because receiver loses when rates rise

### PV01 / Annuity

$$PV01 = A = \sum_{i} \tau_i \times DF_i$$

Present value of 1bp paid at each fixed date.

### Gamma

For a vanilla linear swap:
$$\Gamma = 0$$

No convexity in a linear swap (unlike options).

---

## Implementation in QuantStrata

### Instrument Classes

```python
from src.instruments.ir import InterestRateSwap, InterestRateSwapSimple

# Market data version
swap = InterestRateSwap(
    notional=50_000_000,
    fixed_rate=0.045,         # 4.5% fixed rate
    start_time=0.0,           # Spot starting
    end_time=5.0,             # 5-year tenor
    fixed_frequency=0.5,      # Semi-annual fixed
    floating_frequency=0.25,  # Quarterly floating
    fixed_day_count="30/360",
    floating_day_count="ACT/360",
    direction="receiver",
    curve_id=MarketId("IR", "CURVE", "USD.SOFR"),
    spread=0.0,               # No spread
)

# Access properties
print(swap.tenor)             # 5.0
print(swap.tenor_description) # "5Y"
```

### Pricer Classes

```python
from src.pricers.ir import IRSwapPricer, IRSwapPricerSimple

pricer = IRSwapPricer()

# Full pricing
pv = pricer.price(swap, market)
greeks = pricer.greeks(swap, market)

# Key analytics
par_rate = pricer.par_rate(swap, market)
fixed_pv = pricer.fixed_leg_pv(swap, market)
floating_pv = pricer.floating_leg_pv(swap, market)
```

### Building Swap Legs Manually

```python
from src.instruments.ir.linear.swap import FixedLeg, FloatingLeg

# Create custom leg
fixed_leg = FixedLeg(
    start_time=0.0,
    end_time=1.0,
    accrual_factor=1.0,
    discount_factor=0.95,
    notional=10_000_000,
    fixed_rate=0.05,
)

floating_leg = FloatingLeg(
    start_time=0.0,
    end_time=1.0,
    accrual_factor=1.0,
    discount_factor=0.95,
    notional=10_000_000,
    forward_rate=0.048,
    spread=0.001,  # 10bp spread
)
```

---

## Practical Examples

### Example 1: Corporate Hedging

Company has $100M floating-rate debt at SOFR + 1%. They want to fix their interest cost.

```python
# Enter 5Y payer swap to convert floating to fixed
hedge = InterestRateSwap(
    notional=100_000_000,
    fixed_rate=0.045,        # 4.5% fixed
    start_time=0.0,
    end_time=5.0,
    direction="payer",       # Pay fixed, receive floating
    curve_id=usd_sofr_curve,
)

# Net position:
# - Debt: Pay SOFR + 1%
# - Swap: Receive SOFR, Pay 4.5%
# - Net: Pay 4.5% + 1% = 5.5% fixed
```

### Example 2: Trading a Rate View

Trader believes rates will fall from current par rate of 5%.

```python
# Enter receiver swap at par
trade = InterestRateSwap(
    notional=50_000_000,
    fixed_rate=0.05,         # At par (PV = 0)
    start_time=0.0,
    end_time=10.0,
    direction="receiver",    # Receive fixed
    curve_id=usd_sofr_curve,
)

# If rates fall 50bp:
# Profit ≈ DV01 × 50 = $50M × 8.0 × 0.0001 × 50 = $2M
```

### Example 3: Asset-Liability Matching

Pension fund has fixed-rate bond assets but floating-rate liabilities.

```python
# Enter receiver swap to convert assets to floating
alm_swap = InterestRateSwap(
    notional=500_000_000,
    fixed_rate=0.04,
    start_time=0.0,
    end_time=30.0,           # Match liability duration
    direction="receiver",    # Receive fixed (matches bond coupons)
    curve_id=usd_sofr_curve,
)

# Now effectively has floating-rate assets matching liabilities
```

---

## Swap Valuation at Different Points

### At Inception (Par Swap)

$$PV = 0$$
$$K = K_{par}$$

### After Rates Move

If rates rise by $\Delta r$:
$$PV_{receiver} \approx -N \times A \times \Delta r$$

Example: $100M receiver swap, $A = 4.5$, rates up 1%:
$$PV \approx -100M \times 4.5 \times 0.01 = -\$4.5M$$

### Unwind Value

The mark-to-market PV is what the counterparty would pay/receive to unwind.

---

## Decomposition and Relationships

### Swap as Portfolio of FRAs

A swap can be decomposed into a portfolio of FRAs:
$$IRS = \sum_{i} FRA_i$$

Each FRA corresponds to one reset period.

### Swap-Spot Relationship

$$\text{Spot-Starting Swap PV} = PV_{fixed} - PV_{float}$$

For a par swap starting today:
$$PV_{float} = N \times [1 - DF(T_n)]$$

(The floating leg of a par swap equals notional minus final discount factor)

---

## Advanced Topics

### Forward-Starting Swaps

Swaps that start at a future date:

```python
forward_swap = InterestRateSwap(
    notional=50_000_000,
    fixed_rate=0.05,
    start_time=1.0,          # Starts in 1 year
    end_time=6.0,            # Ends in 6 years (5Y tenor)
    direction="receiver",
    curve_id=curve_id,
)
```

### Basis Swaps

Exchange two floating rates (e.g., SOFR vs Fed Funds):
- Both legs floating
- One leg has spread

### Amortizing Swaps

Notional decreases over time:
- Each period can have different notional
- Use `InterestRateSwapSimple` with custom legs

---

## See Also

- [Forward Rate Agreements](fra.md) - Single-period building block
- [Caps and Floors](caps_floors.md) - Options on swap rates (via caplets)
- [Swaptions](swaptions.md) - Options on swaps (Phase 3.3)
- [Curve Bootstrapping](../reference/models/curve_bootstrapping.md) - Building swap curves
