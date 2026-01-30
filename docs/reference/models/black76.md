# Black76 Model

**Complete Mathematical Framework for Forward-Based Option Pricing**

This document provides a rigorous mathematical treatment of the Black76 model (also known as Black's Model), including full derivations, proofs, comparisons with Black-Scholes-Merton, and practical considerations for quantitative finance.

---

## Table of Contents

1. [Historical Context and Motivation](#1-historical-context-and-motivation)
2. [Model Assumptions](#2-model-assumptions)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Derivation of the Black76 Formula](#4-derivation-of-the-black76-formula)
5. [Greeks: Sensitivities Analysis](#5-greeks-sensitivities-analysis)
6. [Comparison with Black-Scholes-Merton](#6-comparison-with-black-scholes-merton)
7. [Applications](#7-applications)
8. [Implementation Notes](#8-implementation-notes)
9. [Interview Key Points](#9-interview-key-points)

---

## 1. Historical Context and Motivation

### 1.1 Background

Fischer Black published "The Pricing of Commodity Contracts" in 1976, extending the Black-Scholes framework to options on futures and forwards. The model is particularly elegant because:

1. **Eliminates cost-of-carry complexity** - Works directly with observable forward prices
2. **Unified framework** - Applies to futures options, interest rate derivatives, and forward options
3. **Simpler inputs** - No need to separately model spot price dynamics and cost-of-carry

### 1.2 Why Black76 Matters

Black76 is the standard model for:
- **Commodity options** (crude oil, natural gas, gold, agricultural)
- **Interest rate caps/floors** (caplets/floorlets on LIBOR/SOFR)
- **Swaptions** (options on interest rate swaps)
- **FX forward options** (options on forward exchange rates)
- **Equity index futures options** (S&P 500 E-mini options)

---

## 2. Model Assumptions

### 2.1 Forward Price Dynamics

**Assumption A1: Log-Normal Forward**

The forward price $F_t$ follows a driftless geometric Brownian motion under the $T$-forward measure:

$$
dF_t = \sigma F_t \, dW_t^T
$$

Where:
- $\sigma$: Volatility (constant)
- $W_t^T$: Brownian motion under the $T$-forward measure

**Key Insight:** Unlike BSM, there is no drift term because the forward price is already a martingale under the $T$-forward measure.

### 2.2 Market Assumptions

| Assumption | Mathematical Statement | Reality |
|------------|----------------------|---------|
| **A2: No arbitrage** | Forward-spot parity holds | Generally true |
| **A3: Constant interest rate** | $r$ is deterministic | Stochastic in practice |
| **A4: Constant volatility** | $\sigma$ is fixed | Volatility smile exists |
| **A5: European exercise** | Exercise only at expiry | Some products are American |
| **A6: Observable forward** | $F$ is market-quoted | Usually true for liquid markets |

---

## 3. Mathematical Framework

### 3.1 The Forward Measure

Under the risk-neutral measure $\mathbb{Q}$, a stock paying continuous dividend $q$ follows:

$$
dS_t = (r - q)S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
$$

The forward price is:

$$
F_t = S_t \cdot e^{(r-q)(T-t)}
$$

Applying Itô's lemma:

$$
dF_t = \sigma F_t \, dW_t^{\mathbb{Q}}
$$

The drift cancels! This is because we've changed to the $T$-forward measure where the forward is a martingale.

### 3.2 Solution to the SDE

Integrating the SDE from $t=0$ to $t=T$:

$$
F_T = F_0 \exp\left(-\frac{\sigma^2 T}{2} + \sigma W_T^T\right)
$$

Where $W_T^T \sim \mathcal{N}(0, T)$.

Therefore:

$$
\ln\left(\frac{F_T}{F_0}\right) \sim \mathcal{N}\left(-\frac{\sigma^2 T}{2}, \sigma^2 T\right)
$$

---

## 4. Derivation of the Black76 Formula

### 4.1 Risk-Neutral Valuation

Under the $T$-forward measure, the option price is:

$$
V_0 = P(0, T) \cdot \mathbb{E}^{T}\left[\text{Payoff}(F_T)\right]
$$

Where $P(0, T) = e^{-rT}$ is the discount factor.

### 4.2 Call Option Derivation

For a European call with payoff $\max(F_T - K, 0)$:

$$
C = e^{-rT} \cdot \mathbb{E}^{T}\left[\max(F_T - K, 0)\right]
$$

Let $Z = \frac{\ln(F_T/F_0) + \sigma^2 T/2}{\sigma\sqrt{T}}$, then $Z \sim \mathcal{N}(0, 1)$.

$$
F_T = F_0 \exp\left(\sigma\sqrt{T} \cdot Z - \frac{\sigma^2 T}{2}\right)
$$

The option is in-the-money when $F_T > K$, i.e., when:

$$
Z > \frac{\ln(K/F_0) + \sigma^2 T/2}{\sigma\sqrt{T}} = -d_2
$$

Where:

$$
d_1 = \frac{\ln(F_0/K) + \sigma^2 T/2}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

### 4.3 The Black76 Formulas

**European Call:**

$$
\boxed{C = e^{-rT}\left[F_0 N(d_1) - K N(d_2)\right]}
$$

**European Put:**

$$
\boxed{P = e^{-rT}\left[K N(-d_2) - F_0 N(-d_1)\right]}
$$

**Where:**

$$
d_1 = \frac{\ln(F_0/K) + \frac{\sigma^2 T}{2}}{\sigma\sqrt{T}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T} = \frac{\ln(F_0/K) - \frac{\sigma^2 T}{2}}{\sigma\sqrt{T}}
$$

### 4.4 Put-Call Parity

For Black76:

$$
C - P = e^{-rT}(F_0 - K)
$$

**Proof:** Direct substitution of the formulas using $N(d_1) + N(-d_1) = 1$.

---

## 5. Greeks: Sensitivities Analysis

### 5.1 First-Order Greeks

**Delta (Δ):** Sensitivity to forward price

$$
\Delta_C = e^{-rT} N(d_1), \quad \Delta_P = e^{-rT}(N(d_1) - 1) = -e^{-rT} N(-d_1)
$$

**Vega (ν):** Sensitivity to volatility

$$
\nu = e^{-rT} F_0 \sqrt{T} \cdot n(d_1)
$$

Same for call and put.

**Theta (Θ):** Time decay (per year)

$$
\Theta = -e^{-rT} F_0 n(d_1) \frac{\sigma}{2\sqrt{T}} + r \cdot V
$$

Where $V$ is the option value and $n(\cdot)$ is the standard normal PDF.

**Rho (ρ):** Sensitivity to discount rate

$$
\rho = -T \cdot V
$$

### 5.2 Second-Order Greeks

**Gamma (Γ):** Convexity in forward price

$$
\Gamma = e^{-rT} \frac{n(d_1)}{F_0 \sigma \sqrt{T}}
$$

Same for call and put.

**Vanna:** Cross derivative $\partial^2 V / \partial F \partial \sigma$

$$
\text{Vanna} = -e^{-rT} n(d_1) \frac{d_2}{\sigma}
$$

**Volga (Vomma):** Convexity in volatility

$$
\text{Volga} = \nu \cdot \frac{d_1 d_2}{\sigma}
$$

### 5.3 Greeks Summary Table

| Greek | Call Formula | Put Formula |
|-------|--------------|-------------|
| Delta | $e^{-rT} N(d_1)$ | $-e^{-rT} N(-d_1)$ |
| Gamma | $\frac{e^{-rT} n(d_1)}{F_0 \sigma \sqrt{T}}$ | Same |
| Vega | $e^{-rT} F_0 \sqrt{T} \cdot n(d_1)$ | Same |
| Theta | $-\frac{e^{-rT} F_0 n(d_1) \sigma}{2\sqrt{T}} + rC$ | $-\frac{e^{-rT} F_0 n(d_1) \sigma}{2\sqrt{T}} + rP$ |
| Rho | $-T \cdot C$ | $-T \cdot P$ |

---

## 6. Comparison with Black-Scholes-Merton

### 6.1 Mathematical Relationship

BSM with cost-of-carry $b$:

$$
C_{BSM} = S_0 e^{(b-r)T} N(d_1^{BSM}) - K e^{-rT} N(d_2^{BSM})
$$

With $F_0 = S_0 e^{bT}$, this becomes Black76:

$$
C_{Black76} = e^{-rT} [F_0 N(d_1) - K N(d_2)]
$$

**The formulas are equivalent when:**

$$
F_0 = S_0 \cdot e^{bT}
$$

### 6.2 When to Use Each Model

| Scenario | Use BSM | Use Black76 |
|----------|---------|-------------|
| Spot price observable | ✓ | |
| Forward price observable | | ✓ |
| Equity options | ✓ | |
| Commodity futures options | | ✓ |
| Interest rate caps/floors | | ✓ |
| Swaptions | | ✓ |
| FX vanilla options | ✓ | |
| FX forward options | | ✓ |

### 6.3 Key Differences

| Aspect | BSM | Black76 |
|--------|-----|---------|
| Underlying | Spot price $S$ | Forward price $F$ |
| Drift | $(r - q)S$ | Zero (martingale) |
| Cost-of-carry | Explicit parameter $b$ | Embedded in $F$ |
| $d_1$ numerator | $\ln(S/K) + (b + \sigma^2/2)T$ | $\ln(F/K) + \sigma^2 T/2$ |
| Spot delta | Computed directly | Requires $\partial F/\partial S$ |

---

## 7. Applications

### 7.1 Interest Rate Caps/Floors

A **caplet** on forward rate $F$ with strike $K$:

$$
\text{Caplet} = \tau \cdot P(0, T_{end}) \cdot \mathbb{E}^{T_{end}}\left[\max(F - K, 0)\right]
$$

Where:
- $\tau$: Day count fraction
- $F$: Forward rate
- $P(0, T_{end})$: Discount factor to payment date

This is priced with Black76 using the forward rate as the underlying.

### 7.2 Swaptions

A payer swaption (right to enter into a pay-fixed swap):

$$
\text{Swaption} = A(0) \cdot \mathbb{E}^{A}\left[\max(S - K, 0)\right]
$$

Where:
- $A(0)$: Annuity factor
- $S$: Forward swap rate

### 7.3 Commodity Options

For options on crude oil futures:

- $F$: Futures price (WTI, Brent)
- $\sigma$: Implied volatility from options market
- No need to model convenience yield explicitly

---

## 8. Implementation Notes

### 8.1 Discount Factor vs Rate

Our implementation uses `discount_factor` rather than computing $e^{-rT}$ internally:

```python
vanilla_price(
    option_type="call",
    forward=75.0,          # F
    strike=80.0,           # K
    expiry=0.5,            # T
    discount_factor=0.975, # DF = exp(-rT)
    vol=0.30,              # σ
)
```

This allows flexibility for different discounting conventions.

### 8.2 Theta Requires Discount Rate

Theta computation requires the discount rate (not just the discount factor) because:

$$
\Theta = \frac{\partial V}{\partial T} = -\frac{e^{-rT} F n(d_1) \sigma}{2\sqrt{T}} + r \cdot V
$$

The second term depends on $r$ explicitly.

### 8.3 Edge Cases

- **$T = 0$:** Return discounted intrinsic value
- **$\sigma = 0$:** Return discounted intrinsic value (forward is deterministic)
- **$F = K$ (ATM):** $d_1 = d_2 = \sigma\sqrt{T}/2$

---

## 9. Interview Key Points

### Derivation Questions

**Q: How is Black76 related to BSM?**

A: Black76 is BSM with the spot replaced by the forward. Set $F = S \cdot e^{bT}$ where $b$ is cost-of-carry, and the formulas coincide. Black76 is simpler because:
1. No drift term (forward is martingale under $T$-forward measure)
2. No cost-of-carry parameter needed
3. $d_1$ numerator is just $\ln(F/K) + \sigma^2 T/2$

**Q: Why is there no drift in Black76?**

A: Under the $T$-forward measure, the forward price is a martingale. Intuitively, the expected forward price (under this measure) equals today's forward price. The discounting effect exactly cancels any drift.

**Q: What is the $T$-forward measure?**

A: The measure under which the $T$-maturity zero-coupon bond is the numeraire. Under this measure, any asset price divided by $P(t, T)$ is a martingale. The forward price $F_t = S_t/P(t, T)$ (for zero-dividend assets) is therefore a martingale.

### Practical Questions

**Q: When would you use Black76 vs BSM?**

A: Use Black76 when:
- The forward/futures price is directly quoted (commodities, rates)
- Cost-of-carry is hard to model (convenience yield)
- The product references a forward rate (caps, floors, swaptions)

Use BSM when:
- Spot price is the natural reference (equity options)
- You need spot Greeks (delta hedge with spot)

**Q: How do you hedge a Black76 option?**

A: 
1. **Delta hedge with forwards/futures:** Buy/sell $\Delta$ units of the forward
2. **No financing cost:** Forward hedge requires no capital (unlike spot hedge)
3. **Roll risk:** May need to roll the hedge as contracts expire

---

## Appendix: Key Formulas Reference

### Black76 Formula

$$
C = e^{-rT}\left[F N(d_1) - K N(d_2)\right]
$$

$$
P = e^{-rT}\left[K N(-d_2) - F N(-d_1)\right]
$$

$$
d_1 = \frac{\ln(F/K) + \sigma^2 T/2}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

### Put-Call Parity

$$
C - P = e^{-rT}(F - K)
$$

### BSM ↔ Black76 Mapping

$$
F = S \cdot e^{bT}
$$

$$
d_1^{Black76} = d_1^{BSM} = \frac{\ln(S/K) + (b + \sigma^2/2)T}{\sigma\sqrt{T}}
$$

---

## References

1. Black, F. (1976). "The Pricing of Commodity Contracts"
2. Hull, J.C. "Options, Futures, and Other Derivatives"
3. Brigo, D. & Mercurio, F. "Interest Rate Models - Theory and Practice"
4. Rebonato, R. "Volatility and Correlation"

---

*Document Version: 1.0 | QuantStrata Phase 2.3 | January 2026*
