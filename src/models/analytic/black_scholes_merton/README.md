# Black–Scholes–Merton Analytics (Generic Carry Form)

Folder: `src/models/analytic/black_scholes_merton/`

This package implements **closed-form Black–Scholes–Merton (BSM)** pricing and greeks 
as **pure functions** under a **generic cost-of-carry** parameterisation.

## Design Principles

This is a **pure mathematics layer**:
- No `Market` objects, curves, vol surfaces, or instruments
- Inputs are scalars: `spot, strike, expiry, discount_rate, carry, vol`
- Outputs are floats / greek dictionaries
- All functions are stateless and composable

The job of **asset-specific pricers** (`src/pricers/fx/`, `src/pricers/equity/`) is to:
1. Map real market objects (curves, vols) into `(discount_rate, carry, vol)`
2. Apply notionals and units
3. Map generic Greeks to asset-specific conventions

---

## Module Structure

### `base.py` - All Formulas and Validation

```python
# Validation
validate_inputs(spot, strike, expiry, vol)

# Core helpers
d1_d2(spot, strike, expiry, carry, vol) -> (d1, d2)
forward_factor(carry, discount_rate, expiry) -> exp((b-r)T)
discount_factor(rate, expiry) -> exp(-rT)
intrinsic(option_type, spot, strike)

# Vanilla formulas
vanilla_price(option_type, spot, strike, expiry, discount_rate, carry, vol)
vanilla_delta(...)
vanilla_gamma(...)
vanilla_vega(...)
vanilla_theta(...)
vanilla_rho_discount(...)  # dPV/dr holding b fixed
vanilla_rho_carry(...)     # dPV/db holding r fixed
vanilla_greeks(...) -> dict  # All Greeks in one call

# Digital cash-or-nothing
digital_cash_price(..., cash)
digital_cash_delta(...)
digital_cash_gamma(...)
digital_cash_vega(...)
digital_cash_greeks(...)

# Digital asset-or-nothing
digital_asset_price(...)
digital_asset_delta(...)
digital_asset_greeks(...)
```

---

## Quickstart

### Vanilla Option (FX Example)

```python
from src.models.analytic.black_scholes_merton import vanilla_price, vanilla_greeks

# FX mapping: r = r_d, b = r_d - r_f
S = 1.25      # Spot EUR/USD
K = 1.25      # Strike
T = 1.0       # 1 year
r_d = 0.03    # Domestic rate (USD)
r_f = 0.01    # Foreign rate (EUR)
r = r_d       # Discount rate
b = r_d - r_f # Cost-of-carry
sigma = 0.20  # Volatility

pv = vanilla_price(option_type="call", spot=S, strike=K, expiry=T,
                   discount_rate=r, carry=b, vol=sigma)
g = vanilla_greeks(option_type="call", spot=S, strike=K, expiry=T,
                   discount_rate=r, carry=b, vol=sigma)

# Map generic rhos to FX-specific:
rho_domestic = g["rho_discount"] + g["rho_carry"]  # dPV/d(r_d)
rho_foreign = -g["rho_carry"]                      # dPV/d(r_f)
```

### Vanilla Option (Equity Example)

```python
from src.models.analytic.black_scholes_merton import vanilla_price, vanilla_greeks

# Equity mapping: r = risk-free rate, b = r - q (dividend yield)
S = 100       # Stock price
K = 100       # Strike
T = 0.5       # 6 months
r = 0.05      # Risk-free rate
q = 0.02      # Dividend yield
b = r - q     # Cost-of-carry
sigma = 0.25  # Volatility

pv = vanilla_price(option_type="call", spot=S, strike=K, expiry=T,
                   discount_rate=r, carry=b, vol=sigma)
g = vanilla_greeks(option_type="call", spot=S, strike=K, expiry=T,
                   discount_rate=r, carry=b, vol=sigma)

# Equity rho is total rate sensitivity:
rho = g["rho_discount"] + g["rho_carry"]
```

### Digital Cash-or-Nothing

```python
from src.models.analytic.black_scholes_merton import digital_cash_price

# Digital call paying $10 if S_T > K
pv = digital_cash_price(option_type="call", spot=100, strike=100, expiry=0.5,
                        discount_rate=0.05, carry=0.03, vol=0.20, cash=10.0)
```

---

## Cost-of-Carry Mappings

| Asset Class | discount_rate (r) | carry (b) |
|-------------|-------------------|-----------|
| **Equity (no div)** | r | r |
| **Equity (continuous div)** | r | r - q |
| **FX (Garman-Kohlhagen)** | r_d | r_d - r_f |
| **Commodity** | r | r - c + y |

Where:
- `r` = risk-free rate
- `q` = dividend yield
- `r_d` = domestic rate
- `r_f` = foreign rate
- `c` = convenience yield
- `y` = storage cost

**Note:** For futures/forwards (b=0), use the **Black76** model instead.

---

## Greek Conventions

### Generic Greeks (Model Layer)

| Greek | Definition | Description |
|-------|------------|-------------|
| `delta` | dPV/dS | Spot sensitivity |
| `gamma` | d²PV/dS² | Convexity |
| `vega` | dPV/dσ | Per 1.0 absolute vol |
| `theta` | -dPV/dT | Time decay per year |
| `rho_discount` | dPV/dr (b fixed) | Discount rate sensitivity |
| `rho_carry` | dPV/db (r fixed) | Carry rate sensitivity |

### Asset-Specific Mapping (Pricer Layer)

**FX:**
```python
rho_domestic = rho_discount + rho_carry  # Both r and b depend on r_d
rho_foreign = -rho_carry                 # Only b depends on r_f
```

**Equity:**
```python
rho = rho_discount + rho_carry  # Single rate sensitivity
```

---

# Technical Reference

## Mathematical Framework

### SDE with Generic Carry

Under risk-neutral measure:

```
dS = b S dt + σ S dW
```

### Closed-Form Solutions

**Vanilla Call:**
```
C = S exp((b-r)T) N(d₁) - K exp(-rT) N(d₂)
```

**Vanilla Put:**
```
P = K exp(-rT) N(-d₂) - S exp((b-r)T) N(-d₁)
```

Where:
```
d₁ = [ln(S/K) + (b + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### d₁ and d₂ Derivation

From SDE solution:
```
ln(S_T) = ln(S) + (b - σ²/2)T + σ√T Z,  where Z ~ N(0,1)
```

Define:
```
d₂ = [ln(S/K) + (b - σ²/2)T] / (σ√T)
d₁ = d₂ + σ√T = [ln(S/K) + (b + σ²/2)T] / (σ√T)
```

Then:
- `P(S_T > K) = N(d₂)`
- `E[S_T | S_T > K] × P(S_T > K) = S exp(bT) N(d₁)`

---

## Digital Options

### Cash-or-Nothing

Pays fixed cash `C` if ITM at expiry:
```
Call: C × exp(-rT) × N(d₂)
Put:  C × exp(-rT) × N(-d₂)
```

### Asset-or-Nothing

Pays underlying value if ITM:
```
Call: S × exp((b-r)T) × N(d₁)
Put:  S × exp((b-r)T) × N(-d₁)
```

---

## Greek Formulas

### Vanilla Delta
```
Δ_call = exp((b-r)T) N(d₁)
Δ_put  = exp((b-r)T) [N(d₁) - 1]
```

### Vanilla Gamma
```
Γ = exp((b-r)T) n(d₁) / (S σ √T)
```

### Vanilla Vega
```
ν = S exp((b-r)T) n(d₁) √T
```

### Vanilla Theta
```
θ_call = -S exp((b-r)T) n(d₁) σ/(2√T) - (r-b) S exp((b-r)T) N(d₁) - r K exp(-rT) N(d₂)
θ_put  = -S exp((b-r)T) n(d₁) σ/(2√T) + (r-b) S exp((b-r)T) N(-d₁) + r K exp(-rT) N(-d₂)
```

### Rho Discount (holding b fixed)
```
ρ_r_call = T K exp(-rT) N(d₂) - T S exp((b-r)T) N(d₁)
ρ_r_put  = -T K exp(-rT) N(-d₂) + T S exp((b-r)T) N(-d₁)
```

### Rho Carry (holding r fixed)
```
ρ_b_call = T S exp((b-r)T) N(d₁)
ρ_b_put  = -T S exp((b-r)T) N(-d₁)
```

---

## Edge Cases

### T = 0 (At Expiry)
- Price returns intrinsic value
- Greeks return 0 (payoff is discontinuous)

### σ = 0 (Zero Volatility)
- Forward is deterministic: `F = S exp(bT)`
- Price returns discounted intrinsic at forward
- Greeks return 0

---

## Put-Call Parity

```
Call - Put = S exp((b-r)T) - K exp(-rT)
```

FX form:
```
Call - Put = S exp(-r_f T) - K exp(-r_d T)
```

---

## Related Models

| Model | Use Case | Module |
|-------|----------|--------|
| **BSM** | Spot-based options | `black_scholes_merton/` |
| **Black76** | Futures/forward options | `black76/` |
| **Bachelier** | Negative rates, spreads | `bachelier/` |

---

## References

1. Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
2. Merton, R. (1973). "Theory of Rational Option Pricing"
3. Garman, M. & Kohlhagen, S. (1983). "Foreign Currency Option Values"
4. Hull, J. "Options, Futures, and Other Derivatives"
