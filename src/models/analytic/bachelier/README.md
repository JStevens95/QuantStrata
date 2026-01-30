# Bachelier (Normal) Model

Folder: `src/models/analytic/bachelier/`

This package implements the **Bachelier** (Normal) model for options where
the underlying follows normal (arithmetic) rather than log-normal dynamics.

## Design Principles

- Normal distribution (underlying can go negative)
- Absolute volatility (same units as underlying)
- Ideal for interest rates, spreads, and basis trades

---

## When to Use Bachelier vs Log-Normal Models

| Use Case | Model |
|----------|-------|
| Equity options | BSM (log-normal) |
| FX options | BSM (log-normal) |
| Interest rate options (negative rates) | Bachelier |
| Spread options | Bachelier |
| Basis trades | Bachelier |
| Options where underlying can be negative | Bachelier |

**Key difference:** Bachelier uses normal dynamics; BSM/Black76 use log-normal.

---

## Module Structure

### `base.py` - All Formulas

```python
# Validation
validate_inputs(forward, strike, expiry, vol)  # forward/strike can be negative

# Core helpers
d_moneyness(forward, strike, expiry, vol) -> d = (F-K)/(σ√T)
intrinsic(option_type, forward, strike)

# Vanilla formulas
vanilla_price(option_type, forward, strike, expiry, discount_factor, vol)
vanilla_delta(...)
vanilla_gamma(...)
vanilla_vega(...)
vanilla_theta(..., discount_rate)
vanilla_rho(...)
vanilla_greeks(...) -> dict
```

---

## Quickstart

### Swaption with Negative Rates

```python
from src.models.analytic.bachelier import vanilla_price, vanilla_greeks

# EUR swaption when forward swap rate is negative
F = -0.005        # Forward swap rate -0.5%
K = -0.003        # Strike -0.3%
T = 1.0           # 1 year to expiry
DF = 0.98         # Discount factor
sigma = 0.0050    # 50bp normal volatility

pv_per_unit = vanilla_price(option_type="call", forward=F, strike=K,
                            expiry=T, discount_factor=DF, vol=sigma)
```

### Spread Option

```python
from src.models.analytic.bachelier import vanilla_price

# Option on spread between two assets
# Spread can be positive or negative
spread_forward = 2.5   # Expected spread
spread_strike = 3.0    # Strike
T = 0.5
DF = 0.975
sigma = 1.0            # Absolute vol (same units as spread)

pv = vanilla_price(option_type="call", forward=spread_forward, strike=spread_strike,
                   expiry=T, discount_factor=DF, vol=sigma)
```

---

## Mathematical Framework

### Dynamics

Bachelier assumes the forward price follows normal (arithmetic) dynamics:

```
dF = σ dW
```

Where σ is **absolute** volatility (not percentage).

### Distribution

```
F_T = F + σ√T × Z,  where Z ~ N(0,1)
```

Unlike log-normal models, F_T can be negative.

### Closed-Form Solutions

**Call:**
```
C = DF × [(F - K) N(d) + σ√T n(d)]
```

**Put:**
```
P = DF × [(K - F) N(-d) + σ√T n(d)]
```

Where:
```
d = (F - K) / (σ√T)
N(·) = standard normal CDF
n(·) = standard normal PDF
```

### Comparison with Black76

| Aspect | Black76 | Bachelier |
|--------|---------|-----------|
| Dynamics | dF = σF dW | dF = σ dW |
| Distribution | Log-normal | Normal |
| F can be negative | No | Yes |
| Volatility | Percentage | Absolute |

---

## Greek Formulas

### Delta
```
Δ_call = DF × N(d)
Δ_put  = DF × [N(d) - 1]
```

### Gamma
```
Γ = DF × n(d) / (σ√T)
```

Note: Gamma is constant in F (unlike BSM/Black76 where it depends on S).

### Vega
```
ν = DF × √T × n(d)
```

### Rho
```
ρ = -T × PV
```

### Theta
```
θ = -DF × σ × n(d) / (2√T) + r × PV
```

---

## Volatility Conventions

Bachelier volatility is **absolute** (same units as underlying):

| Market | Typical Quote | Interpretation |
|--------|---------------|----------------|
| Interest rates | 50bp = 0.0050 | ±0.5% move in 1σ/1Y |
| Spreads | 1.0 | ±1.0 unit move in 1σ/1Y |

### Converting to Annualized Move

For a σ_N (normal vol) quoted annually:
- 1-year expected move: ±σ_N (with 68% probability)
- 1-month expected move: ±σ_N × √(1/12)

---

## Edge Cases

### T = 0
- Returns discounted intrinsic

### σ = 0
- Forward is deterministic
- Returns discounted intrinsic

### Forward = Strike (ATM)
- d = 0
- Call price = Put price = DF × σ√T × n(0) = DF × σ√T / √(2π)

---

## Input Conventions

| Parameter | Type | Description |
|-----------|------|-------------|
| `forward` | float | Forward price F (can be negative) |
| `strike` | float | Strike K (can be negative) |
| `expiry` | float | Time to expiry T ≥ 0 |
| `discount_factor` | float | DF = exp(-rT) |
| `vol` | float | Absolute normal volatility σ |

---

## Put-Call Parity

```
Call - Put = DF × (F - K)
```

Same as Black76 (and holds for any forward-based model).

---

## Normal vs Log-Normal Smile

In markets where both quotes exist:

- **ATM:** Normal and log-normal vols are approximately related by:
  ```
  σ_N ≈ σ_LN × F  (at-the-money)
  ```

- **Smile:** Normal vol produces a flatter implied vol smile because
  the model treats up and down moves symmetrically.

---

## References

1. Bachelier, L. (1900). "Théorie de la Spéculation"
2. Hunt, P. & Kennedy, J. "Financial Derivatives in Theory and Practice"
3. Various central bank publications on negative rate option pricing
