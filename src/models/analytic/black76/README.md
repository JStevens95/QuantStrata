# Black76 Model (Forward-Based)

Folder: `src/models/analytic/black76/`

This package implements **Black76** closed-form pricing for options on forwards and futures
as **pure functions**.

## Design Principles

- Forward-based model (uses F directly, not spot S)
- No cost-of-carry parameter (carry is implicit in forward)
- Ideal for options on futures, forward rates, and FX forwards

---

## When to Use Black76 vs BSM

| Use Case | Model |
|----------|-------|
| Options on spot with dividends/foreign rates | BSM |
| Options on commodity futures | Black76 |
| Interest rate caps/floors (caplets/floorlets) | Black76 |
| Swaptions | Black76 |
| Options on equity index futures | Black76 |
| Options on FX forwards | Black76 |

**Key difference:** BSM requires mapping spot → forward via cost-of-carry.
Black76 uses the observable forward price directly.

---

## Module Structure

### `base.py` - All Formulas

```python
# Validation
validate_inputs(forward, strike, expiry, vol)

# Core helpers
d1_d2(forward, strike, expiry, vol) -> (d1, d2)
intrinsic(option_type, forward, strike)

# Vanilla formulas
vanilla_price(option_type, forward, strike, expiry, discount_factor, vol)
vanilla_delta(...)
vanilla_gamma(...)
vanilla_vega(...)
vanilla_theta(..., discount_rate)  # Requires rate for full theta
vanilla_rho(...)
vanilla_greeks(...) -> dict
```

---

## Quickstart

### Option on Crude Oil Futures

```python
from src.models.analytic.black76 import vanilla_price, vanilla_greeks

# WTI crude oil futures option
F = 75.0          # Forward price $/barrel
K = 80.0          # Strike
T = 0.5           # 6 months
DF = 0.975        # Discount factor exp(-rT)
r = 0.05          # For theta calculation
sigma = 0.30      # 30% vol

pv = vanilla_price(option_type="call", forward=F, strike=K, expiry=T,
                   discount_factor=DF, vol=sigma)
g = vanilla_greeks(option_type="call", forward=F, strike=K, expiry=T,
                   discount_factor=DF, discount_rate=r, vol=sigma)
```

### Interest Rate Caplet

```python
from src.models.analytic.black76 import vanilla_price

# Caplet on 3M LIBOR
F = 0.05          # Forward rate 5%
K = 0.04          # Cap strike 4%
T = 1.0           # 1 year to expiry
tau = 0.25        # 3-month accrual period
notional = 1_000_000
DF = 0.95         # Discount factor

# Black76 price is per unit notional on the rate
pv_per_unit = vanilla_price(option_type="call", forward=F, strike=K,
                            expiry=T, discount_factor=DF, vol=0.20)

# Full caplet PV = notional × tau × Black76_price
caplet_pv = notional * tau * pv_per_unit
```

---

## Mathematical Framework

### Dynamics

Black76 assumes the forward price follows log-normal dynamics:

```
dF = σ F dW
```

Note: No drift term (forward is a martingale under forward measure).

### Closed-Form Solutions

**Call:**
```
C = DF × [F N(d₁) - K N(d₂)]
```

**Put:**
```
P = DF × [K N(-d₂) - F N(-d₁)]
```

Where:
```
d₁ = [ln(F/K) + σ²T/2] / (σ√T)
d₂ = d₁ - σ√T
DF = exp(-rT)
```

### Key Difference from BSM

BSM d₁:
```
d₁_BSM = [ln(S/K) + (b + σ²/2)T] / (σ√T)
```

Black76 d₁:
```
d₁_B76 = [ln(F/K) + σ²T/2] / (σ√T)
```

The carry term `bT` in BSM is replaced by `ln(F/S)` in Black76 (already embedded in F).

---

## Greek Formulas

### Delta (w.r.t. Forward)
```
Δ_call = DF × N(d₁)
Δ_put  = DF × [N(d₁) - 1]
```

### Gamma
```
Γ = DF × n(d₁) / (F σ √T)
```

### Vega
```
ν = DF × F × n(d₁) × √T
```

### Rho
```
ρ = -T × PV
```

### Theta
```
θ = -DF × F × n(d₁) × σ / (2√T) + r × PV
```

---

## Edge Cases

### T = 0
- Returns discounted intrinsic: `DF × max(F-K, 0)` or `DF × max(K-F, 0)`

### σ = 0
- Forward is deterministic
- Returns discounted intrinsic

---

## Input Conventions

| Parameter | Type | Description |
|-----------|------|-------------|
| `forward` | float | Forward price F > 0 |
| `strike` | float | Strike K > 0 |
| `expiry` | float | Time to expiry T ≥ 0 |
| `discount_factor` | float | DF = exp(-rT) |
| `vol` | float | Log-normal volatility σ |

---

## Put-Call Parity

```
Call - Put = DF × (F - K)
```

---

## References

1. Black, F. (1976). "The Pricing of Commodity Contracts"
2. Hull, J. "Options, Futures, and Other Derivatives"
