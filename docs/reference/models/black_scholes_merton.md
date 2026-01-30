# Black-Scholes-Merton Model

**Complete Mathematical Framework for Option Pricing**

This document provides a rigorous mathematical treatment of the Black-Scholes-Merton (BSM) model, including full derivations, proofs, assumptions, extensions, and practical considerations for quantitative finance interviews.

---

## Table of Contents

1. [Historical Context and Significance](#1-historical-context-and-significance)
2. [Model Assumptions](#2-model-assumptions)
3. [The Stochastic Framework](#3-the-stochastic-framework)
4. [Derivation of the Black-Scholes PDE](#4-derivation-of-the-black-scholes-pde)
5. [Solving the Black-Scholes PDE](#5-solving-the-black-scholes-pde)
6. [The Risk-Neutral Measure](#6-the-risk-neutral-measure)
7. [Greeks: Sensitivities Analysis](#7-greeks-sensitivities-analysis)
8. [Extensions to the Model](#8-extensions-to-the-model)
9. [Limitations and Model Risk](#9-limitations-and-model-risk)
10. [Interview Key Points](#10-interview-key-points)

---

## 1. Historical Context and Significance

### 1.1 The 1973 Breakthrough

Fischer Black and Myron Scholes published "The Pricing of Options and Corporate Liabilities" in 1973, with Robert Merton providing additional mathematical rigor. This work revolutionized finance by:

1. Providing a **closed-form solution** for European option prices
2. Introducing **dynamic hedging** and replication arguments
3. Establishing the foundation for **risk-neutral pricing**
4. Earning Scholes and Merton the 1997 Nobel Prize in Economics

### 1.2 Why BSM Matters Today

Despite its limitations, BSM remains fundamental because:
- It provides **intuition** for option pricing mechanisms
- It establishes the **Greek** sensitivities framework
- It serves as a **benchmark** for more complex models
- Its implied volatility is the **market's language** for options

---

## 2. Model Assumptions

The BSM model rests on several idealized assumptions:

### 2.1 Asset Price Dynamics

**Assumption A1: Geometric Brownian Motion**

The underlying asset price $S_t$ follows:
$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

Where:
- $\mu$: Drift (expected return)
- $\sigma$: Volatility (constant)
- $W_t$: Standard Brownian motion

**Implication:** Log-returns are normally distributed:
$$
\ln\left(\frac{S_T}{S_0}\right) \sim \mathcal{N}\left((\mu - \frac{\sigma^2}{2})T, \sigma^2 T\right)
$$

### 2.2 Market Assumptions

| Assumption | Mathematical Statement | Reality |
|------------|----------------------|---------|
| **A2: Frictionless market** | No transaction costs | Costs exist, non-trivial for high-frequency |
| **A3: Continuous trading** | Trading at any instant | Discrete trading with gaps |
| **A4: No arbitrage** | No risk-free profit | Generally holds, but not perfectly |
| **A5: Constant interest rate** | $r$ is deterministic | Rates are stochastic |
| **A6: No dividends** | Or known continuous yield $q$ | Discrete dividends exist |
| **A7: European exercise** | Exercise only at expiry | American options common |
| **A8: Constant volatility** | $\sigma$ is fixed | Volatility is stochastic (smile) |

---

## 3. The Stochastic Framework

### 3.1 Brownian Motion Properties

A standard Brownian motion $W_t$ satisfies:
1. $W_0 = 0$
2. $W_t$ has independent increments
3. $W_t - W_s \sim \mathcal{N}(0, t-s)$ for $t > s$
4. $W_t$ has continuous paths (a.s.)

### 3.2 Itô's Lemma

**Theorem (Itô's Lemma):** If $f(S_t, t)$ is twice continuously differentiable and $S_t$ satisfies an SDE, then:

$$
df = \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial S}dS + \frac{1}{2}\frac{\partial^2 f}{\partial S^2}(dS)^2
$$

Using $dS = \mu S \, dt + \sigma S \, dW$ and $(dS)^2 = \sigma^2 S^2 \, dt$ (from $dW \cdot dW = dt$):

$$
df = \left(\frac{\partial f}{\partial t} + \mu S\frac{\partial f}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 f}{\partial S^2}\right)dt + \sigma S\frac{\partial f}{\partial S}dW
$$

### 3.3 Key Stochastic Calculus Rules

| Product | Result | Comment |
|---------|--------|---------|
| $dt \cdot dt$ | $0$ | Higher order |
| $dt \cdot dW$ | $0$ | Higher order |
| $dW \cdot dW$ | $dt$ | **Itô's rule** |

---

## 4. Derivation of the Black-Scholes PDE

### 4.1 Setup: Replicating Portfolio

Let $V(S, t)$ be the price of a derivative on $S$. Consider a portfolio:
$$
\Pi = V - \Delta S
$$

Where $\Delta = \frac{\partial V}{\partial S}$ (delta hedge).

### 4.2 Portfolio Dynamics

Using Itô's lemma on $V(S_t, t)$:

$$
dV = \left(\frac{\partial V}{\partial t} + \mu S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}\right)dt + \sigma S\frac{\partial V}{\partial S}dW
$$

The stock dynamics:
$$
dS = \mu S \, dt + \sigma S \, dW
$$

Portfolio change:
$$
d\Pi = dV - \Delta \, dS
$$

### 4.3 Eliminating Randomness

Substituting with $\Delta = \frac{\partial V}{\partial S}$:

$$
d\Pi = \left(\frac{\partial V}{\partial t} + \mu S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}\right)dt + \sigma S\frac{\partial V}{\partial S}dW - \frac{\partial V}{\partial S}\left(\mu S \, dt + \sigma S \, dW\right)
$$

The $dW$ terms cancel:
$$
d\Pi = \left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}\right)dt
$$

**Key insight:** The portfolio is **instantaneously risk-free**!

### 4.4 No-Arbitrage Condition

A risk-free portfolio must earn the risk-free rate:
$$
d\Pi = r\Pi \, dt = r(V - \Delta S)dt
$$

Equating:
$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} = r\left(V - S\frac{\partial V}{\partial S}\right)
$$

### 4.5 The Black-Scholes PDE

Rearranging:

$$
\boxed{\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} = rV}
$$

**Boundary Conditions:**
- **Call:** $V(S, T) = \max(S - K, 0)$
- **Put:** $V(S, T) = \max(K - S, 0)$

**Observation:** The drift $\mu$ does **not appear** in the PDE!

---

## 5. Solving the Black-Scholes PDE

### 5.1 Transformation to Heat Equation

**Step 1: Change variables**

Let $\tau = T - t$ (time to expiry) and $x = \ln(S/K)$ (log-moneyness).

**Step 2: Transform the PDE**

After substitution, the Black-Scholes PDE becomes the heat equation:
$$
\frac{\partial u}{\partial \tau} = \frac{\sigma^2}{2}\frac{\partial^2 u}{\partial x^2}
$$

With appropriate boundary conditions.

### 5.2 The Black-Scholes Formula

**European Call:**
$$
\boxed{C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)}
$$

**European Put:**
$$
\boxed{P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)}
$$

Where:
$$
d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}
$$
$$
d_2 = d_1 - \sigma\sqrt{T} = \frac{\ln(S_0/K) + (r - q - \sigma^2/2)T}{\sigma\sqrt{T}}
$$

And $N(\cdot)$ is the standard normal CDF:
$$
N(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x} e^{-z^2/2}dz
$$

### 5.3 Proof Sketch via Feynman-Kac

**Feynman-Kac Theorem:** The solution to the PDE
$$
\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} = rV
$$
with terminal condition $V(S, T) = g(S)$ is:
$$
V(S_0, 0) = e^{-rT}\mathbb{E}^{\mathbb{Q}}[g(S_T)]
$$

Where $S_T$ follows risk-neutral dynamics:
$$
dS_t = rS_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
$$

**Computing the expectation:**
$$
S_T = S_0 \exp\left((r - \frac{\sigma^2}{2})T + \sigma\sqrt{T}Z\right), \quad Z \sim \mathcal{N}(0,1)
$$

For a call, $g(S_T) = \max(S_T - K, 0)$:
$$
C = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]
$$

Evaluating this integral yields the Black-Scholes formula.

---

## 6. The Risk-Neutral Measure

### 6.1 Girsanov's Theorem

**Theorem:** Under change of measure from $\mathbb{P}$ to $\mathbb{Q}$, if:
$$
\frac{d\mathbb{Q}}{d\mathbb{P}} = \exp\left(-\theta W_T - \frac{1}{2}\theta^2 T\right)
$$

Then $W_t^{\mathbb{Q}} = W_t + \theta t$ is a Brownian motion under $\mathbb{Q}$.

### 6.2 Market Price of Risk

Under $\mathbb{P}$: $dS = \mu S \, dt + \sigma S \, dW$

Define $\theta = \frac{\mu - r}{\sigma}$ (market price of risk/Sharpe ratio).

Under $\mathbb{Q}$: $dS = rS \, dt + \sigma S \, dW^{\mathbb{Q}}$

**Interpretation:** The drift changes from $\mu$ to $r$, but volatility is unchanged.

### 6.3 Risk-Neutral Valuation Formula

$$
V_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]
$$

**Why "risk-neutral"?**
- All assets earn the risk-free rate in expectation under $\mathbb{Q}$
- No risk premium for bearing volatility risk
- Pricing is independent of investor preferences

---

## 7. Greeks: Sensitivities Analysis

### 7.1 First-Order Greeks

**Delta (Δ):** Sensitivity to spot price
$$
\Delta = \frac{\partial V}{\partial S}
$$

For a call: $\Delta_C = e^{-qT}N(d_1)$
For a put: $\Delta_P = -e^{-qT}N(-d_1)$

**Theta (Θ):** Sensitivity to time (time decay)
$$
\Theta = \frac{\partial V}{\partial t}
$$

For a call:
$$
\Theta_C = -\frac{S_0\sigma e^{-qT}N'(d_1)}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qS_0e^{-qT}N(d_1)
$$

**Vega (ν):** Sensitivity to volatility
$$
\nu = \frac{\partial V}{\partial \sigma} = S_0 e^{-qT} \sqrt{T} N'(d_1)
$$

**Rho (ρ):** Sensitivity to interest rate
$$
\rho_C = KTe^{-rT}N(d_2)
$$

### 7.2 Second-Order Greeks

**Gamma (Γ):** Convexity in spot price
$$
\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{e^{-qT}N'(d_1)}{S_0\sigma\sqrt{T}}
$$

**Vanna:** Mixed derivative $\partial^2 V / \partial S \partial \sigma$
$$
\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = -e^{-qT}N'(d_1)\frac{d_2}{\sigma}
$$

**Volga (Vomma):** Convexity in volatility
$$
\text{Volga} = \frac{\partial^2 V}{\partial \sigma^2} = \nu \frac{d_1 d_2}{\sigma}
$$

### 7.3 Greek Relationships

**The Fundamental PDE as Greeks:**
$$
\Theta + rS\Delta + \frac{1}{2}\sigma^2 S^2\Gamma = rV
$$

**Trader's Rule of Thumb:**
- Long options: Positive gamma, negative theta (pay for convexity)
- Short options: Negative gamma, positive theta (earn premium, bear risk)

---

## 8. Extensions to the Model

### 8.1 Continuous Dividend Yield

Replace $r$ with $r - q$ in the drift:
$$
dS_t = (r - q)S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
$$

The forward price becomes: $F = S_0 e^{(r-q)T}$

### 8.2 FX Options (Garman-Kohlhagen)

For FX rate $S$ (domestic per foreign):
- $r_d$: Domestic interest rate
- $r_f$: Foreign interest rate

$$
C = S_0 e^{-r_f T} N(d_1) - K e^{-r_d T} N(d_2)
$$

**Interpretation:** Foreign currency is like a "stock" paying dividend yield $r_f$.

### 8.3 Black's Model (Forwards/Futures)

For a forward $F$ expiring at $T$:
$$
C = e^{-rT}[F N(d_1) - K N(d_2)]
$$

Where:
$$
d_1 = \frac{\ln(F/K) + \sigma^2 T/2}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

### 8.4 Implied Volatility

**Definition:** The $\sigma$ that makes BSM price equal market price.

**Newton-Raphson iteration:**
$$
\sigma_{n+1} = \sigma_n - \frac{C_{BSM}(\sigma_n) - C_{market}}{\text{Vega}(\sigma_n)}
$$

**The Volatility Smile:** Implied vol varies with strike, contradicting BSM's constant vol assumption.

---

## 9. Limitations and Model Risk

### 9.1 Volatility Issues

| Issue | BSM Assumption | Reality |
|-------|----------------|---------|
| Smile/Skew | Flat vol surface | Vol depends on K, T |
| Stochastic vol | $\sigma$ constant | Vol varies randomly |
| Vol clustering | IID returns | Periods of high/low vol |
| Fat tails | Normal returns | Excess kurtosis |

### 9.2 Market Microstructure

- **Discrete hedging:** Can't trade continuously
- **Transaction costs:** Delta hedging is expensive
- **Liquidity:** Large orders move prices
- **Jumps:** Prices can gap overnight

### 9.3 Model Risk Quantification

**Hedging error** over $[t, t+\Delta t]$:
$$
\epsilon \approx \frac{1}{2}\Gamma(\Delta S)^2 - \Theta\Delta t
$$

For perfect hedging, we need $\Gamma(\Delta S)^2 = 2\Theta\Delta t$, which holds only in expectation under BSM.

---

## 10. Interview Key Points

### Derivation Questions

**Q: Derive the Black-Scholes PDE.**

A: 
1. Apply Itô's lemma to $V(S,t)$
2. Construct delta-hedged portfolio $\Pi = V - \Delta S$
3. Show $d\Pi$ has no $dW$ term (risk-free)
4. Set $d\Pi = r\Pi \, dt$ (no arbitrage)
5. Rearrange to get PDE

**Q: Why doesn't $\mu$ appear in the PDE?**

A: The delta hedge eliminates the random component. Under no-arbitrage, the risk-free portfolio earns $r$, not $\mu$. This is the **market price of risk** argument.

**Q: What is the probabilistic interpretation?**

A: By Feynman-Kac, $V = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{Payoff}]$ where $\mathbb{Q}$ is the risk-neutral measure under which $S$ drifts at $r$.

### Greek Questions

**Q: What is gamma and why do traders care?**

A: Gamma is $\partial^2 V/\partial S^2$, the curvature of the option price. Long gamma means:
- You profit from large moves (either direction)
- You need to hedge frequently (cost)
- You pay theta (time decay)

**Q: Explain the gamma-theta trade-off.**

A: From the PDE: $\Theta = rV - rS\Delta - \frac{1}{2}\sigma^2 S^2\Gamma$

At-the-money: High $\Gamma$, negative $\Theta$ (pay time decay for convexity).

### Model Limitation Questions

**Q: What are the main limitations of BSM?**

A:
1. **Constant volatility** - Reality: vol smile/skew
2. **Continuous hedging** - Reality: discrete, costly
3. **No jumps** - Reality: gap moves exist
4. **Normal returns** - Reality: fat tails

**Q: How do you handle the volatility smile?**

A: Use:
- **Local volatility** (Dupire): $\sigma(S,t)$ function
- **Stochastic volatility** (Heston, SABR): $\sigma$ is itself random
- **Jump-diffusion** (Merton): Add Poisson jumps

---

## Appendix: Key Formulas Reference

### Black-Scholes Formula (with dividends)

$$
C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$
$$
P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)
$$
$$
d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

### Greeks Summary

| Greek | Call | Put |
|-------|------|-----|
| Delta | $e^{-qT}N(d_1)$ | $-e^{-qT}N(-d_1)$ |
| Gamma | $\frac{e^{-qT}N'(d_1)}{S\sigma\sqrt{T}}$ | Same |
| Vega | $S e^{-qT}\sqrt{T}N'(d_1)$ | Same |
| Theta | Complex expression | Different |
| Rho | $KTe^{-rT}N(d_2)$ | $-KTe^{-rT}N(-d_2)$ |

### Put-Call Parity

$$
C - P = S_0 e^{-qT} - K e^{-rT}
$$

---

## References

1. Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
2. Merton, R.C. (1973). "Theory of Rational Option Pricing"
3. Hull, J.C. "Options, Futures, and Other Derivatives"
4. Shreve, S.E. "Stochastic Calculus for Finance II"
5. Wilmott, P. "Paul Wilmott on Quantitative Finance"

---

*Document Version: 1.0 | QuantStrata Phase 1 | January 2026*
