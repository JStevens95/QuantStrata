# Bachelier (Normal) Model

**Complete Mathematical Framework for Normal Distribution Option Pricing**

This document provides a rigorous mathematical treatment of the Bachelier model, including full derivations, comparisons with log-normal models, applications to negative rate environments, and practical considerations for quantitative finance.

---

## Table of Contents

1. [Historical Context and Motivation](#1-historical-context-and-motivation)
2. [Model Assumptions](#2-model-assumptions)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Derivation of the Bachelier Formula](#4-derivation-of-the-bachelier-formula)
5. [Greeks: Sensitivities Analysis](#5-greeks-sensitivities-analysis)
6. [Comparison with Log-Normal Models](#6-comparison-with-log-normal-models)
7. [Applications](#7-applications)
8. [Volatility Conventions](#8-volatility-conventions)
9. [Interview Key Points](#9-interview-key-points)

---

## 1. Historical Context and Motivation

### 1.1 The First Option Pricing Model

Louis Bachelier published "Théorie de la Spéculation" in 1900—over 70 years before Black-Scholes. His work:

1. **First applied stochastic calculus to finance**
2. **First derived an option pricing formula**
3. **Anticipated modern mathematical finance**

While Bachelier's original model had theoretical issues (allowing negative prices), the framework has become essential for modern applications where negative underlyings are possible.

### 1.2 Why Bachelier Matters Today

The Bachelier model experienced a renaissance due to:

- **Negative interest rates** (EUR, JPY, CHF) - Log-normal models fail
- **Spread options** - Spreads can naturally be negative
- **Swaption markets** - Normal volatility quotation became standard post-2015
- **Basis trades** - Price differences that fluctuate around zero

---

## 2. Model Assumptions

### 2.1 Forward Price Dynamics

**Assumption A1: Normal (Arithmetic) Brownian Motion**

The forward price $F_t$ follows:

$$
dF_t = \sigma \, dW_t^T
$$

Where:
- $\sigma$: **Absolute volatility** (in same units as $F$)
- $W_t^T$: Brownian motion under the $T$-forward measure

**Key Insight:** Unlike log-normal models, $\sigma$ is **not** a percentage—it has the same units as the underlying.

### 2.2 Market Assumptions

| Assumption | Mathematical Statement | Reality |
|------------|----------------------|---------|
| **A2: Normal distribution** | $F_T \sim \mathcal{N}(F_0, \sigma^2 T)$ | May not hold for large moves |
| **A3: Constant volatility** | $\sigma$ is fixed | Volatility smile exists |
| **A4: Negative values allowed** | $F$ can be $< 0$ | Essential for rates/spreads |
| **A5: European exercise** | Exercise only at expiry | Some products are American |

### 2.3 Key Difference from Log-Normal

| Property | Log-Normal (BSM/Black76) | Normal (Bachelier) |
|----------|--------------------------|-------------------|
| Dynamics | $dF = \sigma F \, dW$ | $dF = \sigma \, dW$ |
| Distribution | $F_T > 0$ always | $F_T$ can be negative |
| Volatility units | Percentage (%) | Absolute (same as $F$) |
| Skewness | Positive | Zero |

---

## 3. Mathematical Framework

### 3.1 Solution to the SDE

Integrating $dF_t = \sigma \, dW_t^T$ from $t=0$ to $t=T$:

$$
F_T = F_0 + \sigma W_T^T
$$

Where $W_T^T \sim \mathcal{N}(0, T)$.

Therefore:

$$
F_T \sim \mathcal{N}(F_0, \sigma^2 T)
$$

Or equivalently:

$$
F_T = F_0 + \sigma\sqrt{T} \cdot Z, \quad Z \sim \mathcal{N}(0, 1)
$$

### 3.2 Probability of Exercise

For a call option:

$$
\mathbb{P}(F_T > K) = \mathbb{P}\left(Z > \frac{K - F_0}{\sigma\sqrt{T}}\right) = N\left(\frac{F_0 - K}{\sigma\sqrt{T}}\right) = N(d)
$$

Where:

$$
d = \frac{F_0 - K}{\sigma\sqrt{T}}
$$

---

## 4. Derivation of the Bachelier Formula

### 4.1 Risk-Neutral Valuation

Under the $T$-forward measure:

$$
C = e^{-rT} \cdot \mathbb{E}^{T}\left[\max(F_T - K, 0)\right]
$$

### 4.2 Call Option Derivation

Let $d = \frac{F_0 - K}{\sigma\sqrt{T}}$ and $\sigma_T = \sigma\sqrt{T}$.

$$
\mathbb{E}[\max(F_T - K, 0)] = \int_{-\infty}^{\infty} \max(F_0 + \sigma_T z - K, 0) \cdot n(z) \, dz
$$

The option is ITM when $z > -d$:

$$
= \int_{-d}^{\infty} (F_0 - K + \sigma_T z) \cdot n(z) \, dz
$$

Splitting the integral:

$$
= (F_0 - K) \int_{-d}^{\infty} n(z) \, dz + \sigma_T \int_{-d}^{\infty} z \cdot n(z) \, dz
$$

Using:
- $\int_{-d}^{\infty} n(z) \, dz = N(d)$
- $\int_{-d}^{\infty} z \cdot n(z) \, dz = n(d)$ (see Appendix)

We get:

$$
\mathbb{E}[\max(F_T - K, 0)] = (F_0 - K) N(d) + \sigma\sqrt{T} \cdot n(d)
$$

### 4.3 The Bachelier Formulas

**European Call:**

$$
\boxed{C = e^{-rT}\left[(F_0 - K) N(d) + \sigma\sqrt{T} \cdot n(d)\right]}
$$

**European Put:**

$$
\boxed{P = e^{-rT}\left[(K - F_0) N(-d) + \sigma\sqrt{T} \cdot n(d)\right]}
$$

**Where:**

$$
d = \frac{F_0 - K}{\sigma\sqrt{T}}
$$

And:
- $N(\cdot)$: Standard normal CDF
- $n(\cdot)$: Standard normal PDF

### 4.4 Put-Call Parity

For Bachelier:

$$
C - P = e^{-rT}(F_0 - K)
$$

**Proof:** $(F_0 - K)N(d) + (K - F_0)N(-d) = (F_0 - K)[N(d) - N(-d)] = (F_0 - K)$.

---

## 5. Greeks: Sensitivities Analysis

### 5.1 First-Order Greeks

**Delta (Δ):** Sensitivity to forward price

$$
\Delta_C = e^{-rT} N(d), \quad \Delta_P = e^{-rT}(N(d) - 1) = -e^{-rT} N(-d)
$$

**Note:** Delta is bounded in $[0, e^{-rT}]$ for calls, $[-e^{-rT}, 0]$ for puts.

**Vega (ν):** Sensitivity to absolute volatility

$$
\nu = e^{-rT} \sqrt{T} \cdot n(d)
$$

Same for call and put. **Units:** Same as underlying × time.

**Theta (Θ):** Time decay (per year)

$$
\Theta = -e^{-rT} \frac{\sigma \cdot n(d)}{2\sqrt{T}} + r \cdot V
$$

**Rho (ρ):** Sensitivity to discount rate

$$
\rho = -T \cdot V
$$

### 5.2 Second-Order Greeks

**Gamma (Γ):** Convexity in forward price

$$
\Gamma = e^{-rT} \frac{n(d)}{\sigma\sqrt{T}}
$$

Same for call and put. **Key property:** Gamma is independent of $F$ (constant across strikes at fixed $\sigma$).

**Vanna:** Cross derivative $\partial^2 V / \partial F \partial \sigma$

$$
\text{Vanna} = -e^{-rT} \frac{n(d) \cdot d}{\sigma}
$$

**Volga (Vomma):** Convexity in volatility

$$
\text{Volga} = e^{-rT} \frac{\sqrt{T} \cdot n(d) \cdot d^2}{\sigma}
$$

### 5.3 Greeks Summary Table

| Greek | Call Formula | Put Formula | Units |
|-------|--------------|-------------|-------|
| Delta | $e^{-rT} N(d)$ | $-e^{-rT} N(-d)$ | Dimensionless |
| Gamma | $\frac{e^{-rT} n(d)}{\sigma\sqrt{T}}$ | Same | $1/[\text{underlying}]$ |
| Vega | $e^{-rT} \sqrt{T} \cdot n(d)$ | Same | $[\text{underlying}] \times \sqrt{\text{time}}$ |
| Theta | See above | See above | $[\text{value}]/\text{year}$ |
| Rho | $-T \cdot C$ | $-T \cdot P$ | $[\text{value}]/[\text{rate}]$ |

---

## 6. Comparison with Log-Normal Models

### 6.1 Formula Comparison

| Aspect | Black-Scholes / Black76 | Bachelier |
|--------|-------------------------|-----------|
| **Price** | $DF[F N(d_1) - K N(d_2)]$ | $DF[(F-K)N(d) + \sigma\sqrt{T}n(d)]$ |
| **$d$ formula** | $\frac{\ln(F/K) + \sigma^2 T/2}{\sigma\sqrt{T}}$ | $\frac{F - K}{\sigma\sqrt{T}}$ |
| **ATM price** | $\approx 0.4 \sigma S \sqrt{T}$ | $\approx 0.4 \sigma \sqrt{T}$ |
| **Can handle $F < 0$** | No | Yes |

### 6.2 When to Use Each Model

| Scenario | Log-Normal | Bachelier |
|----------|------------|-----------|
| Equity options | ✓ | |
| FX options | ✓ | |
| Commodity options | ✓ | |
| Positive interest rates | ✓ | ✓ |
| **Negative interest rates** | | ✓ |
| **Spread options** | | ✓ |
| **Basis trades** | | ✓ |
| Swaptions (pre-2015) | ✓ | |
| Swaptions (post-2015) | | ✓ |

### 6.3 Volatility Conversion

**Approximate relationship** for ATM options:

$$
\sigma_{normal} \approx \sigma_{lognormal} \times F
$$

More precisely (using straddle prices):

$$
\sigma_N = \sigma_{LN} \cdot F \cdot \frac{\sqrt{2\pi}}{2} \cdot \frac{N(d_1) - N(d_2)}{n(0)}
$$

For ATM: $\sigma_N \approx \sigma_{LN} \cdot F$.

---

## 7. Applications

### 7.1 Negative Rate Swaptions

Post-2012, EUR and CHF rates went negative. Standard Black76 fails because:
- $\ln(F)$ undefined for $F < 0$
- Markets adopted **normal volatility quotation**

Example (EUR 5Y × 5Y swaption):
- Forward swap rate: $F = -0.20\%$
- Strike: $K = -0.10\%$
- Normal vol: $\sigma = 45$ bp

```python
vanilla_price(
    option_type="call",
    forward=-0.0020,        # -0.20%
    strike=-0.0010,         # -0.10%
    expiry=5.0,             # 5 years
    discount_factor=1.05,   # Annuity factor
    vol=0.0045,             # 45bp normal vol
)
```

### 7.2 Spread Options

For spread $S = F_1 - F_2$:
- The spread can be positive or negative
- Normal dynamics are natural: $dS = \sigma_S dW$
- Bachelier is the standard model

Example (calendar spread):
- Spread: $S = 2.50$ (Dec vs March)
- Strike: $K = 3.00$
- Vol: $\sigma = 1.20$

### 7.3 Basis Trades

Basis = Cash Price - Futures Price

- Fluctuates around zero
- Can be positive or negative
- Perfect fit for Bachelier

---

## 8. Volatility Conventions

### 8.1 Basis Point Volatility

For interest rates, normal vol is often quoted in **basis points (bp)**:

- 50 bp vol = $\sigma = 0.0050$ in decimal
- 100 bp vol = $\sigma = 0.0100$ in decimal

**Example:**
- Forward rate: $F = 3.00\% = 0.0300$
- Normal vol: $50$ bp = $0.0050$
- This means: Rate expected to move $\pm 50$ bp over 1 year (1 std dev)

### 8.2 Absolute vs Percentage

| Quote Type | Symbol | Example | Units |
|------------|--------|---------|-------|
| Normal/Absolute | $\sigma_N$ | 50 bp | Same as underlying |
| Log-normal/Percentage | $\sigma_{LN}$ | 20% | Dimensionless |

### 8.3 Conversion Example

Given:
- $F = 5\%$
- $\sigma_{LN} = 20\%$

Approximate normal vol:
$$
\sigma_N \approx 0.20 \times 0.05 = 0.01 = 100 \text{ bp}
$$

---

## 9. Interview Key Points

### Derivation Questions

**Q: Derive the Bachelier formula.**

A: 
1. $F_T = F_0 + \sigma\sqrt{T} Z$ where $Z \sim N(0,1)$
2. $\mathbb{E}[\max(F_T - K, 0)] = \int_{-d}^{\infty}(F_0 - K + \sigma\sqrt{T}z)n(z)dz$
3. Split into $(F_0 - K)N(d) + \sigma\sqrt{T} \cdot n(d)$
4. Discount by $e^{-rT}$

**Q: What is the key difference between Bachelier and Black-Scholes?**

A:
- **BSM:** Log-normal, percentage vol, $F > 0$ required
- **Bachelier:** Normal, absolute vol, $F$ can be negative
- BSM has positive skewness, Bachelier is symmetric

**Q: Why doesn't Bachelier work for equities?**

A:
1. Equities cannot have negative prices (limited liability)
2. Percentage moves are more natural for assets
3. Log-normal captures compounding growth better

### Practical Questions

**Q: When did markets switch to normal vol?**

A: 
- Post-2012 for EUR rates (ECB negative rates)
- 2015-2016 widespread adoption for swaptions
- Now standard for negative-rate capable markets

**Q: How do you convert between normal and log-normal vol?**

A: 
- Approximate: $\sigma_N \approx \sigma_{LN} \times F$
- Exact: Match ATM straddle prices
- For negative rates: Only normal vol is defined

**Q: What are the risks of Bachelier?**

A:
1. **Negative prices possible:** $F_T < 0$ has positive probability
2. **Fat tails underestimated:** Normal distribution has thin tails
3. **Wrong for multiplicative processes:** Doesn't capture % returns

---

## Appendix: Mathematical Details

### A.1 Integral Identity

For the derivation, we use:

$$
\int_{-d}^{\infty} z \cdot n(z) \, dz = n(d)
$$

**Proof:**

$$
\int_{-d}^{\infty} z \cdot n(z) \, dz = \int_{-d}^{\infty} z \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz
$$

Let $u = z^2/2$, then $du = z \, dz$:

$$
= \frac{1}{\sqrt{2\pi}} \left[-e^{-z^2/2}\right]_{-d}^{\infty} = \frac{1}{\sqrt{2\pi}} e^{-d^2/2} = n(d)
$$

### A.2 ATM Approximation

At-the-money ($F = K$), $d = 0$:

$$
C_{ATM} = e^{-rT} \sigma\sqrt{T} \cdot n(0) = e^{-rT} \sigma\sqrt{T} \cdot \frac{1}{\sqrt{2\pi}} \approx 0.399 \cdot e^{-rT} \sigma\sqrt{T}
$$

---

## References

1. Bachelier, L. (1900). "Théorie de la Spéculation"
2. Hagan, P.S. & Woodward, D.E. (1999). "Equivalent Black Volatilities"
3. Brigo, D. & Mercurio, F. "Interest Rate Models - Theory and Practice"
4. Rebonato, R. "Volatility and Correlation"
5. Andersen, L. & Piterbarg, V. "Interest Rate Modeling"

---

*Document Version: 1.0 | QuantStrata Phase 2.3 | January 2026*
