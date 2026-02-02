# Hull-White One-Factor Short Rate Model

**Complete Mathematical Framework for Interest Rate Modeling**

This document provides a rigorous mathematical treatment of the Hull-White model, including full derivations, closed-form solutions, proofs, and practical considerations for quantitative finance.

---

## Table of Contents

1. [Historical Context and Significance](#1-historical-context-and-significance)
2. [Model Assumptions](#2-model-assumptions)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Derivation of Bond Prices](#4-derivation-of-bond-prices)
5. [Bond Option Pricing](#5-bond-option-pricing)
6. [Cap and Floor Pricing](#6-cap-and-floor-pricing)
7. [Swaption Pricing: Jamshidian Decomposition](#7-swaption-pricing-jamshidian-decomposition)
8. [Greeks and Sensitivities](#8-greeks-and-sensitivities)
9. [Calibration to Market Data](#9-calibration-to-market-data)
10. [Comparison with Other Models](#10-comparison-with-other-models)
11. [Interview Key Points](#11-interview-key-points)

---

## 1. Historical Context and Significance

### 1.1 Background

John Hull and Alan White published "Pricing Interest-Rate-Derivative Securities" in 1990, extending the Vasicek model to fit the initial term structure exactly. The Hull-White model became the industry standard because:

1. **Analytic tractability** - Closed-form solutions for bonds and many options
2. **Term structure fitting** - Matches any initial yield curve
3. **Affine structure** - Efficient numerical methods
4. **Proven track record** - Decades of practical use

### 1.2 Why Hull-White Matters

Hull-White is fundamental because:
- **Standard model** for interest rate risk management
- **Benchmark** for more complex models
- **Foundation** for understanding affine term structure models
- **Negative rates** capability (essential post-2012)

---

## 2. Model Assumptions

### 2.1 Short Rate Dynamics

**Assumption A1: Gaussian Mean-Reverting Process**

Under the risk-neutral measure $\mathbb{Q}$, the instantaneous short rate $r(t)$ follows:

$$
dr(t) = [\theta(t) - a \cdot r(t)] \, dt + \sigma \, dW_t^{\mathbb{Q}}
$$

Where:
- $r(t)$: Instantaneous short rate
- $\theta(t)$: Time-dependent drift function
- $a > 0$: Mean reversion speed (constant)
- $\sigma > 0$: Short rate volatility (constant)
- $W_t^{\mathbb{Q}}$: Standard Brownian motion under $\mathbb{Q}$

### 2.2 Model Assumptions

| Assumption | Mathematical Statement | Reality |
|------------|----------------------|---------|
| **A2: Constant parameters** | $a$, $\sigma$ constant | May vary with time |
| **A3: Gaussian rates** | $r(t)$ normally distributed | Can produce negative rates |
| **A4: No jumps** | Continuous paths | Rates can jump |
| **A5: Perfect markets** | No arbitrage, frictionless | Transaction costs exist |

### 2.3 Simplified Hull-White

With constant $\theta$, the model simplifies to:

$$
dr(t) = a(\theta - r(t)) \, dt + \sigma \, dW_t
$$

This is the **Ornstein-Uhlenbeck** process for $r(t)$.

---

## 3. Mathematical Framework

### 3.1 Solution to the SDE

**Theorem:** The solution to the Hull-White SDE is:

$$
r(t) = r(s)e^{-a(t-s)} + \int_s^t \theta(u)e^{-a(t-u)}du + \sigma\int_s^t e^{-a(t-u)}dW_u
$$

**Proof:**

Apply the integrating factor $e^{at}$ to the SDE:

$$
d(e^{at}r(t)) = e^{at}[\theta(t)dt + \sigma dW_t]
$$

Integrating from $s$ to $t$:

$$
e^{at}r(t) - e^{as}r(s) = \int_s^t e^{au}\theta(u)du + \sigma\int_s^t e^{au}dW_u
$$

Multiply both sides by $e^{-at}$ to obtain the result. $\square$

### 3.2 Distribution of r(t)

**Theorem:** Given $r(s)$, the short rate $r(t)$ for $t > s$ is normally distributed:

$$
r(t) | r(s) \sim \mathcal{N}(\mu(s,t), \, V(s,t))
$$

**Mean:**
$$
\mu(s,t) = r(s)e^{-a(t-s)} + \int_s^t \theta(u)e^{-a(t-u)}du
$$

For constant $\theta$:
$$
\mu(s,t) = \theta + (r(s) - \theta)e^{-a(t-s)}
$$

**Variance:**
$$
V(s,t) = \frac{\sigma^2}{2a}(1 - e^{-2a(t-s)})
$$

**Proof:**

The mean follows from taking expectations of the SDE solution. For the variance, use Itô isometry:

$$
\text{Var}(r(t)|r(s)) = \sigma^2\mathbb{E}\left[\left(\int_s^t e^{-a(t-u)}dW_u\right)^2\right] = \sigma^2\int_s^t e^{-2a(t-u)}du = \frac{\sigma^2}{2a}(1 - e^{-2a(t-s)})
$$
$\square$

### 3.3 Long-Term Behavior

As $t \to \infty$ (with constant $\theta$):

$$
\mathbb{E}[r(\infty)] = \theta, \quad \text{Var}(r(\infty)) = \frac{\sigma^2}{2a}
$$

**Half-life:** The time for a rate shock to decay by 50%:
$$
t_{1/2} = \frac{\ln 2}{a}
$$

### 3.4 The Affine Structure

**Definition:** A model is **affine** if bond prices have the form:

$$
P(t, T) = A(t, T)e^{-B(t, T)r(t)}
$$

**Theorem:** Hull-White is affine with:

$$
B(t, T) = \frac{1 - e^{-a(T-t)}}{a}
$$

$$
\ln A(t, T) = \int_t^T \theta(s)B(s, T)ds - \frac{\sigma^2}{2}\int_t^T B(s, T)^2 ds
$$

---

## 4. Derivation of Bond Prices

### 4.1 The Bond Pricing PDE

Zero-coupon bond price $P(r, t; T)$ satisfies:

$$
\frac{\partial P}{\partial t} + [\theta(t) - ar]\frac{\partial P}{\partial r} + \frac{\sigma^2}{2}\frac{\partial^2 P}{\partial r^2} = rP
$$

**Boundary Condition:** $P(r, T; T) = 1$

### 4.2 Risk-Neutral Valuation

**Fundamental Result:**

$$
P(t, T) = \mathbb{E}^{\mathbb{Q}}\left[\exp\left(-\int_t^T r(s)ds\right) \middle| \mathcal{F}_t\right]
$$

### 4.3 Closed-Form Bond Price

**Theorem (Hull-White Bond Price):**

$$
\boxed{P(t, T) = A(t, T) \exp(-B(t, T) \cdot r(t))}
$$

Where:

$$
B(t, T) = \frac{1 - e^{-a(T-t)}}{a}
$$

For the time-dependent $\theta(t)$ case, $A(t, T)$ is derived from fitting to the initial term structure:

$$
A(t, T) = \frac{P^M(0, T)}{P^M(0, t)} \exp\left[B(t, T)f^M(0, t) - \frac{\sigma^2}{4a}B(t, T)^2(1 - e^{-2at})\right]
$$

Where:
- $P^M(0, T)$: Market discount factor
- $f^M(0, t)$: Market instantaneous forward rate

**Proof Sketch:**

1. Guess affine form: $P = e^{A(t,T) - B(t,T)r}$
2. Substitute into the PDE
3. Match coefficients to get ODEs for $A$ and $B$
4. Solve the ODEs with boundary conditions
$\square$

### 4.4 Yield and Forward Rates

**Continuously compounded yield:**

$$
y(t, T) = -\frac{\ln P(t, T)}{T - t} = \frac{B(t, T) \cdot r(t) - \ln A(t, T)}{T - t}
$$

**Instantaneous forward rate:**

$$
f(t, T) = -\frac{\partial \ln P(t, T)}{\partial T} = \frac{\partial B(t, T)}{\partial T} \cdot r(t) - \frac{\partial \ln A(t, T)}{\partial T}
$$

---

## 5. Bond Option Pricing

### 5.1 European Bond Option Formula

**Theorem:** A European call on a zero-coupon bond with:
- Option expiry: $T_0$
- Bond maturity: $T > T_0$
- Strike: $K$

Has price:

$$
\boxed{C = P(0, T)N(d_1) - K \cdot P(0, T_0)N(d_2)}
$$

Where:

$$
d_1 = \frac{1}{\sigma_P}\ln\left(\frac{P(0, T)}{K \cdot P(0, T_0)}\right) + \frac{\sigma_P}{2}
$$

$$
d_2 = d_1 - \sigma_P
$$

**Bond Price Volatility:**

$$
\sigma_P = \sigma B(T_0, T) \sqrt{\frac{1 - e^{-2aT_0}}{2a}}
$$

**Put Option:**

$$
P_{put} = K \cdot P(0, T_0)N(-d_2) - P(0, T)N(-d_1)
$$

### 5.2 Derivation

**Proof:**

Under $\mathbb{Q}$, the bond price $P(T_0, T)$ at option expiry is log-normally distributed because $r(T_0)$ is normal and $P(T_0, T) = A(T_0, T)e^{-B(T_0, T)r(T_0)}$.

The variance of $\ln P(T_0, T)$ is:

$$
\text{Var}[\ln P(T_0, T)] = B(T_0, T)^2 \cdot \text{Var}[r(T_0)] = \sigma_P^2
$$

Apply Black-Scholes-style pricing with the forward bond price as underlying. $\square$

---

## 6. Cap and Floor Pricing

### 6.1 Caplet as Bond Option

**Key Insight:** A caplet on forward rate $L(T_{i-1}, T_i)$ with strike $K_L$ is equivalent to a put option on a zero-coupon bond.

**Caplet Payoff:**

$$
\text{Caplet} = \tau \cdot (L(T_{i-1}, T_i) - K_L)^+ \text{ at } T_i
$$

Where $\tau = T_i - T_{i-1}$ is the accrual period.

**Bond Option Equivalence:**

$$
\text{Caplet} = (1 + \tau K_L) \cdot \text{Put on } P(T_{i-1}, T_i) \text{ with strike } \frac{1}{1 + \tau K_L}
$$

### 6.2 Hull-White Caplet Formula

$$
\text{Caplet} = (1 + \tau K_L)\left[K_P \cdot P(0, T_{i-1})N(-d_2) - P(0, T_i)N(-d_1)\right]
$$

Where:

$$
K_P = \frac{1}{1 + \tau K_L}
$$

$$
\sigma_P = \sigma B(T_{i-1}, T_i)\sqrt{\frac{1 - e^{-2aT_{i-1}}}{2a}}
$$

### 6.3 Cap Pricing

A cap is a portfolio of caplets:

$$
\text{Cap} = \sum_{i=1}^{n} \text{Caplet}_i
$$

Similarly for floors using put-call parity on individual caplets.

---

## 7. Swaption Pricing: Jamshidian Decomposition

### 7.1 Swaption Payoff

A **payer swaption** gives the right to enter a swap paying fixed rate $K$ and receiving floating.

**Payoff at expiry $T_0$:**

$$
\text{Swaption} = \left(\sum_{i=1}^{n} \tau_i P(T_0, T_i)(L(T_0; T_{i-1}, T_i) - K)\right)^+
$$

This equals:

$$
\text{Swaption} = \left(1 - P(T_0, T_n) - K\sum_{i=1}^{n}\tau_i P(T_0, T_i)\right)^+
$$

### 7.2 Jamshidian Decomposition

**Theorem (Jamshidian, 1989):** Under Hull-White, a swaption can be decomposed into a portfolio of bond options.

**Key Insight:** There exists a unique rate $r^*$ such that at $r(T_0) = r^*$, the swap is at-the-money.

**Construction:**

1. Find $r^*$ solving: $1 - P(T_0, T_n; r^*) = K\sum_{i=1}^{n}\tau_i P(T_0, T_i; r^*)$

2. Define strike prices: $K_i = P(T_0, T_i; r^*)$

3. **Decomposition:**
$$
\text{Payer Swaption} = \text{Put}(K_n, T_0, T_n) + K\sum_{i=1}^{n}\tau_i \cdot \text{Put}(K_i, T_0, T_i)
$$

### 7.3 Why Jamshidian Works

**Proof Sketch:**

In Hull-White, all bond prices $P(T_0, T_i)$ are monotonic functions of $r(T_0)$. This means:

- If $r(T_0) > r^*$: All bonds have values below their strikes → all puts exercise
- If $r(T_0) < r^*$: All bonds have values above their strikes → no puts exercise

Thus the swap exercise region coincides exactly with the put exercise regions. $\square$

---

## 8. Greeks and Sensitivities

### 8.1 Bond Greeks

**Delta (rate sensitivity):**

$$
\Delta = \frac{\partial P}{\partial r} = -B(t, T) \cdot P(t, T)
$$

**Gamma:**

$$
\Gamma = \frac{\partial^2 P}{\partial r^2} = B(t, T)^2 \cdot P(t, T)
$$

**DV01 (Dollar Value of 1bp):**

$$
\text{DV01} = -\frac{\partial P}{\partial r} \times 0.0001 = B(t, T) \cdot P(t, T) \times 0.0001
$$

### 8.2 Option Greeks

**Bond Option Delta:**

$$
\Delta_C = P(0, T) \cdot N(d_1) \cdot (-B(T_0, T))
$$

**Vega (volatility sensitivity):**

$$
\nu = P(0, T) \cdot n(d_1) \cdot \frac{\partial \sigma_P}{\partial \sigma}
$$

Where $n(\cdot)$ is the standard normal PDF.

---

## 9. Calibration to Market Data

### 9.1 Fitting the Initial Term Structure

**Requirement:** Model must reproduce all market discount factors.

**Solution:** Choose $\theta(t)$ such that:

$$
P^{model}(0, T) = P^{market}(0, T) \quad \forall T
$$

**Result:**

$$
\theta(t) = f^M_t(0, t) + af^M(0, t) + \frac{\sigma^2}{2a}(1 - e^{-2at})
$$

Where $f^M(0, t)$ is the market instantaneous forward rate and $f^M_t$ is its time derivative.

### 9.2 Calibration to Swaptions

**Objective:** Find $a$ and $\sigma$ that minimize:

$$
\min_{a, \sigma} \sum_{i,j} w_{ij}\left(\sigma^{model}_{ij} - \sigma^{market}_{ij}\right)^2
$$

Where $\sigma_{ij}$ are swaption implied volatilities at expiry $i$ and tenor $j$.

### 9.3 Parameter Sensitivity

| Parameter | Effect on Smile | Effect on Term Structure |
|-----------|-----------------|--------------------------|
| $a$ ↑ | Smile steepens | Short-end vols decrease |
| $\sigma$ ↑ | All vols increase | Parallel shift |

---

## 10. Comparison with Other Models

### 10.1 Hull-White vs Vasicek

| Aspect | Vasicek | Hull-White |
|--------|---------|------------|
| Dynamics | $dr = a(\theta - r)dt + \sigma dW$ | $dr = (\theta(t) - ar)dt + \sigma dW$ |
| Term structure fit | No | Yes (exact) |
| Parameters | Constant $\theta$ | Time-dependent $\theta(t)$ |
| Use case | Academic | Practical |

### 10.2 Hull-White vs Black-Karasinski

| Aspect | Hull-White | Black-Karasinski |
|--------|------------|------------------|
| Rate distribution | Gaussian | Log-normal |
| Negative rates | Yes | No |
| Bond pricing | Closed-form | Numerical |
| Calibration | Analytic | Numerical |
| Speed | Fast | Slower |

### 10.3 Hull-White vs HJM

| Aspect | Hull-White | HJM |
|--------|------------|-----|
| State variable | Short rate | Forward curve |
| Dimensionality | 1 | Infinite |
| Implementation | Simple | Complex |
| Flexibility | Limited | High |

---

## 11. Interview Key Points

### Derivation Questions

**Q: Derive the Hull-White bond price formula.**

A:
1. Guess affine form: $P = e^{A - Br}$
2. Substitute into bond pricing PDE
3. Match terms: Get $dB/dT = 1 - aB$, $B(T,T) = 0$ → $B = (1-e^{-a\tau})/a$
4. Solve for $A$ using boundary condition and market fit

**Q: Why is Hull-White affine?**

A: Because the short rate is Gaussian (linear drift, constant volatility), leading to exponential-affine bond prices. The integral $\int_t^T r(s)ds$ is also Gaussian, and the exponential of a Gaussian is log-normal with known moments.

**Q: How does Jamshidian decomposition work?**

A: In Hull-White, all bond prices are monotonic in $r$. Find $r^*$ where swap = 0. For $r > r^*$, all bonds < strikes (all puts exercise). For $r < r^*$, all bonds > strikes (no puts exercise). This monotonicity allows decomposition into bond options.

### Practical Questions

**Q: How do you calibrate Hull-White?**

A:
1. Fit $\theta(t)$ to match initial yield curve exactly
2. Calibrate $a$ and $\sigma$ to swaption volatilities
3. Use numerical optimization to minimize vol differences

**Q: What are the limitations of Hull-White?**

A:
1. **Gaussian rates** - Can go negative (feature or bug depending on market)
2. **Single factor** - Cannot capture term structure dynamics
3. **Constant vol** - Rate volatility independent of rate level
4. **No jumps** - Cannot model sudden rate moves

**Q: When would you use Hull-White vs Heston vs Local Vol?**

A:
- **Hull-White**: Interest rate products, curves, swaptions
- **Heston**: FX/equity options requiring stochastic vol
- **Local Vol**: Exotic pricing requiring exact vanilla fit

---

## Appendix: Key Formulas

### A.1 Bond Price

$$
P(t, T) = A(t, T)e^{-B(t, T)r(t)}, \quad B(t, T) = \frac{1-e^{-a(T-t)}}{a}
$$

### A.2 Bond Option

$$
C = P(0, T)N(d_1) - KP(0, T_0)N(d_2)
$$

$$
\sigma_P = \sigma B(T_0, T)\sqrt{\frac{1-e^{-2aT_0}}{2a}}
$$

### A.3 Rate Distribution

$$
r(t)|r(s) \sim \mathcal{N}\left(\theta + (r(s)-\theta)e^{-a(t-s)}, \frac{\sigma^2}{2a}(1-e^{-2a(t-s)})\right)
$$

---

## References

1. Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities." *Review of Financial Studies*.
2. Hull, J. & White, A. (1994). "Numerical Procedures for Implementing Term Structure Models I: Single-Factor Models."
3. Jamshidian, F. (1989). "An Exact Bond Option Formula." *Journal of Finance*.
4. Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*. Springer.
5. Hull, J.C. (2022). *Options, Futures, and Other Derivatives*, 11th ed.

---

*Document Version: 1.0 | QuantStrata Phase 3 | January 2026*
