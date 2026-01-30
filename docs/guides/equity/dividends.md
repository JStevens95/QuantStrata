# Equity Dividend Models

**Technical Specification for Dividend Handling in Equity Derivatives Pricing**

This document provides a comprehensive guide to dividend modeling in QuantStrata, including continuous dividend yields, discrete dividends, and their impact on option pricing.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Mathematical Framework](#2-mathematical-framework)
3. [Continuous Dividend Yield](#3-continuous-dividend-yield)
4. [Discrete Dividends](#4-discrete-dividends)
5. [Impact on Option Pricing](#5-impact-on-option-pricing)
6. [Implementation](#6-implementation)
7. [Best Practices](#7-best-practices)

---

## 1. Executive Summary

### 1.1 Overview

Dividends affect equity option pricing through two mechanisms:

1. **Spot adjustment** - Dividends reduce the stock price on ex-date
2. **Forward impact** - Lower forward price affects option values

QuantStrata supports:

| Model | Use Case | Implementation |
|-------|----------|----------------|
| **Continuous Yield** | Index options, long-dated options | Built into BSM cost-of-carry |
| **Discrete Dividends** | Single stock options | Spot adjustment functions |
| **Combined** | Production systems | Forward calculation with both |

### 1.2 Key Formulas

**Continuous Dividend Yield:**
$$F = S \cdot e^{(r-q)T}$$

**Discrete Dividends:**
$$F = \left(S - \sum_{i} D_i \cdot e^{-r t_i}\right) \cdot e^{rT}$$

Where:
- $F$: Forward price
- $S$: Current spot price
- $r$: Risk-free rate
- $q$: Continuous dividend yield
- $D_i$: Discrete dividend amount at time $t_i$
- $T$: Time to expiry

---

## 2. Mathematical Framework

### 2.1 Stock Price Dynamics with Dividends

**Continuous Dividend Yield Model:**

Under the risk-neutral measure, a dividend-paying stock follows:

$$dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t$$

Where:
- $r$: Risk-free rate
- $q$: Continuous dividend yield
- $\sigma$: Volatility

**Interpretation:**
- The stock "leaks" value at rate $q$
- Expected growth is $r - q$ under risk-neutral measure
- Forward price: $F = S_0 \cdot e^{(r-q)T}$

### 2.2 Discrete Dividend Model

With discrete dividends $D_i$ at times $t_i$:

$$S_{t_i^+} = S_{t_i^-} - D_i$$

**Spot Adjustment:**
$$S_{adj} = S_0 - \sum_{t_i < T} D_i \cdot e^{-r t_i}$$

**Forward Price:**
$$F = S_{adj} \cdot e^{rT}$$

### 2.3 Combined Model

For production systems combining both:

$$F = \left(S_0 - \sum_{t_i < T} D_i \cdot e^{-r t_i}\right) \cdot e^{(r-q)T}$$

This handles:
- Discrete announced dividends (known $D_i$)
- Continuous yield for unannounced future dividends

---

## 3. Continuous Dividend Yield

### 3.1 Definition

The continuous dividend yield $q$ represents:
- Annualized dividend rate as a percentage of spot
- Convenient approximation for index options
- Standard BSM extension

### 3.2 Typical Values

| Asset Type | Typical Yield | Notes |
|------------|---------------|-------|
| S&P 500 | 1.5% - 2.5% | Reinvested dividends |
| FTSE 100 | 3% - 4% | Higher UK yields |
| Growth stocks | 0% - 1% | Low/no dividends |
| Value stocks | 2% - 5% | Regular dividends |
| REITs | 4% - 8% | High distribution |

### 3.3 Impact on Forward

For $S = 100$, $r = 5\%$, $T = 1$ year:

| Div Yield $q$ | Forward $F$ | Change |
|---------------|-------------|--------|
| 0% | 105.13 | - |
| 2% | 103.05 | -2.0% |
| 4% | 101.01 | -3.9% |
| 6% | 99.00 | -5.8% |

### 3.4 Implementation

In QuantStrata, continuous dividend yield is handled via the cost-of-carry parameter:

```python
# BSM with continuous dividend yield
from src.models.analytic.black_scholes_merton import vanilla_price

price = vanilla_price(
    option_type="call",
    spot=100.0,
    strike=100.0,
    expiry=1.0,
    discount_rate=0.05,
    carry=0.03,  # b = r - q = 0.05 - 0.02 = 0.03
    vol=0.20,
)
```

---

## 4. Discrete Dividends

### 4.1 Definition

Discrete dividends are specific cash payments at known ex-dates:

$$D_i \text{ paid at } t_i$$

### 4.2 Ex-Date vs Payment Date

| Date | Description | Impact |
|------|-------------|--------|
| **Announcement** | Dividend declared | No price impact |
| **Ex-Date** | Stock trades ex-dividend | Price drops by $D$ |
| **Record Date** | Shareholder list compiled | No impact |
| **Payment Date** | Cash distributed | No spot impact |

**For pricing:** Use the ex-date, not the payment date.

### 4.3 Spot Adjustment Method

The standard approach adjusts spot by subtracting PV of future dividends:

$$S_{adj} = S_0 - \sum_{0 < t_i < T} D_i \cdot e^{-r t_i}$$

**Example:**
- $S_0 = 100$
- $r = 5\%$
- Dividends: $\$2$ at 3 months, $\$2$ at 9 months
- $T = 1$ year

$$S_{adj} = 100 - 2 \cdot e^{-0.05 \cdot 0.25} - 2 \cdot e^{-0.05 \cdot 0.75}$$
$$S_{adj} = 100 - 1.975 - 1.927 = 96.10$$

### 4.4 Implementation

```python
from src.marketdata.providers.synthetic.generators.equity import (
    adjust_spot_for_discrete_dividend,
    compute_forward_with_dividends,
)

# Adjust spot for a single dividend
adjusted = adjust_spot_for_discrete_dividend(
    spot=100.0,
    dividend_amount=2.0,
    ex_date_fraction=0.25,  # 3 months
    current_time=0.0,
)
# → 98.0 (simplified, without discounting)

# Compute forward with multiple discrete dividends
forward = compute_forward_with_dividends(
    spot=100.0,
    discount_rate=0.05,
    dividend_yield=0.0,  # No continuous yield
    expiry=1.0,
    discrete_dividends=[(0.25, 2.0), (0.75, 2.0)],
)
# → 101.01
```

---

## 5. Impact on Option Pricing

### 5.1 Call Options

Higher dividends → Lower call value

**Intuition:**
- Dividends reduce expected stock price at expiry
- Call payoff $\max(S_T - K, 0)$ is lower
- Forward $F$ is lower

### 5.2 Put Options

Higher dividends → Higher put value

**Intuition:**
- Lower expected stock price benefits put holders
- Put payoff $\max(K - S_T, 0)$ is higher

### 5.3 Put-Call Parity with Dividends

**Continuous Yield:**
$$C - P = S \cdot e^{-qT} - K \cdot e^{-rT}$$

**Discrete Dividends:**
$$C - P = S_{adj} - K \cdot e^{-rT}$$

Where $S_{adj} = S - PV(\text{dividends})$

### 5.4 American Options

Dividends create early exercise incentives:

**American Calls:**
- It may be optimal to exercise just before ex-date
- Capture dividend by owning shares
- Only if dividend > time value lost

**Decision Rule for Calls:**
Exercise if:
$$D > K \cdot (1 - e^{-r(T-t_d)})$$

Where $t_d$ is the ex-date.

**American Puts:**
- Dividends reduce early exercise incentive for puts
- Higher dividends → Lower spot → More likely ITM
- But also → More time value

---

## 6. Implementation

### 6.1 Module Location

```
src/marketdata/providers/synthetic/generators/equity.py
├── adjust_spot_for_discrete_dividend()
└── compute_forward_with_dividends()
```

### 6.2 Continuous Yield in Pricers

All equity pricers accept dividend yield via the market:

```python
# Market setup with dividend yield
div_id = MarketId("EQUITY", "DIV_YIELD", "AAPL")
market = Market(
    quotes={div_id: Quote(value=0.02)},  # 2% yield
    ...
)

# Pricer reads yield from market
pricer = EquityEuropeanVanillaBsmPricer()
price = pricer.price(option, market)
```

### 6.3 Discrete Dividends

For discrete dividends, adjust the spot before pricing:

```python
from src.marketdata.providers.synthetic.generators.equity import (
    compute_forward_with_dividends,
)

# Known dividends
dividends = [
    (0.25, 0.50),  # $0.50 at 3 months
    (0.50, 0.50),  # $0.50 at 6 months
    (0.75, 0.50),  # $0.50 at 9 months
]

# Compute dividend-adjusted forward
forward = compute_forward_with_dividends(
    spot=100.0,
    discount_rate=0.05,
    dividend_yield=0.0,
    expiry=1.0,
    discrete_dividends=dividends,
)

# Use forward in Black76 pricing (forward-based model)
from src.models.analytic.black76 import vanilla_price as black76_price

price = black76_price(
    option_type="call",
    forward=forward,
    strike=100.0,
    expiry=1.0,
    discount_rate=0.05,
    vol=0.20,
)
```

---

## 7. Best Practices

### 7.1 Model Selection

| Scenario | Recommended Model |
|----------|-------------------|
| Index options | Continuous yield |
| Single stock, short-dated | Discrete dividends |
| Single stock, long-dated | Discrete (announced) + continuous (future) |
| Backtesting | Match production model |

### 7.2 Data Quality

**For continuous yield:**
- Use trailing 12-month yield
- Or analyst estimates of forward yield
- Adjust for special dividends

**For discrete dividends:**
- Use announced ex-dates
- Include only dividends with ex-date < expiry
- Verify against company filings

### 7.3 Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Using payment date | Always use ex-date |
| Forgetting to discount | Apply $e^{-rt}$ to future dividends |
| Double counting | Don't combine yield + discrete for same period |
| Ignoring special dividends | Track non-regular dividends separately |

### 7.4 Validation

**Check put-call parity:**
```python
C - P ≈ S * exp(-q*T) - K * exp(-r*T)  # Continuous
C - P ≈ S_adj - K * exp(-r*T)          # Discrete
```

**Check forward price:**
```python
# Implied forward from options
F_implied = K + exp(r*T) * (C - P)

# Model forward
F_model = compute_forward_with_dividends(...)

# Should match
assert abs(F_implied - F_model) < tolerance
```

---

## References

- Hull, J. (2022). *Options, Futures, and Other Derivatives*. Chapter 15.
- Haug, E.G. (2007). *The Complete Guide to Option Pricing Formulas*.
- `src/marketdata/providers/synthetic/generators/equity.py`
- `src/models/analytic/black_scholes_merton/base.py`
- `src/models/analytic/black76/base.py`
