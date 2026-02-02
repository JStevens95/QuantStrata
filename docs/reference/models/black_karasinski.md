# Black-Karasinski One-Factor Short Rate Model

**Complete Mathematical Framework for Log-Normal Interest Rate Modeling**

This document provides a rigorous mathematical treatment of the Black-Karasinski model, including derivations, proofs, comparison with affine models, and numerical considerations for quantitative finance.

---

## Table of Contents

1. [Historical Context and Motivation](#1-historical-context-and-motivation)
2. [Model Assumptions](#2-model-assumptions)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Non-Affine Structure and Bond Pricing](#4-non-affine-structure-and-bond-pricing)
5. [Option Pricing](#5-option-pricing)
6. [Numerical Methods](#6-numerical-methods)
7. [Calibration Theory](#7-calibration-theory)
8. [Comparison with Affine Models](#8-comparison-with-affine-models)
9. [Advantages and Limitations](#9-advantages-and-limitations)
10. [Interview Key Points](#10-interview-key-points)

---

## 1. Historical Context and Motivation

### 1.1 Background

Fischer Black and Piotr Karasinski published "Bond and Option Pricing when Short Rates are Lognormal" in the *Financial Analysts Journal* (1991). The model addressed a key limitation of Gaussian models:

1. **Positive rates guaranteed** - Log-normal rates cannot go negative
2. **Proportional volatility** - Rate volatility scales with rate level
3. **Fat tails** - Better captures extreme rate movements

### 1.2 Why Black-Karasinski Matters

The model is important because:
- **Pre-negative rate era** - Industry standard before 2014
- **Emerging markets** - Essential where negative rates are impossible
- **Benchmark model** - Comparison with Gaussian models
- **Volatility structure** - More realistic for some markets

---

## 2. Model Assumptions

### 2.1 Short Rate Dynamics

**Assumption A1: Log-Normal Mean-Reverting Process**

Under the risk-neutral measure $\mathbb{Q}$, the logarithm of the short rate follows:

$$
d(\ln r(t)) = [\theta(t) - a \cdot \ln r(t)] \, dt + \sigma \, dW_t^{\mathbb{Q}}
$$

Equivalently, defining $x(t) = \ln r(t)$:

$$
dx(t) = [\theta(t) - a \cdot x(t)] \, dt + \sigma \, dW_t^{\mathbb{Q}}
$$

Where:
- $r(t) = e^{x(t)}$: Instantaneous short rate (always positive)
- $x(t) = \ln r(t)$: Log-rate (state variable)
- $\theta(t)$: Time-dependent drift function
- $a > 0$: Mean reversion speed (constant)
- $\sigma > 0$: Log-rate volatility (constant)
- $W_t^{\mathbb{Q}}$: Standard Brownian motion under $\mathbb{Q}$

### 2.2 Model Assumptions

| Assumption | Mathematical Statement | Reality |
|------------|----------------------|---------|
| **A2: Constant parameters** | $a$, $\sigma$ constant | May vary with time |
| **A3: Log-normal rates** | $r(t) > 0$ always | Cannot model negative rates |
| **A4: No jumps** | Continuous paths | Rates can jump |
| **A5: Perfect markets** | No arbitrage, frictionless | Transaction costs exist |

### 2.3 Key Difference from Hull-White

| Aspect | Hull-White | Black-Karasinski |
|--------|------------|------------------|
| **SDE** | $dr = (\theta - ar)dt + \sigma dW$ | $d(\ln r) = (\theta - a\ln r)dt + \sigma dW$ |
| **State** | Rate $r$ | Log-rate $x = \ln r$ |
| **Distribution** | $r$ is Gaussian | $r$ is Log-normal |
| **Volatility of $r$** | Constant $\sigma$ | Proportional: $\sigma \cdot r$ |

---

## 3. Mathematical Framework

### 3.1 Solution to the SDE

**Theorem:** The log-rate $x(t) = \ln r(t)$ has the explicit solution:

$$
x(t) = x(s)e^{-a(t-s)} + \int_s^t \theta(u)e^{-a(t-u)}du + \sigma\int_s^t e^{-a(t-u)}dW_u
$$

**Proof:**

The log-rate SDE is an Ornstein-Uhlenbeck process. Apply the integrating factor $e^{at}$:

$$
d(e^{at}x(t)) = e^{at}[\theta(t)dt + \sigma dW_t]
$$

Integrating from $s$ to $t$:

$$
e^{at}x(t) - e^{as}x(s) = \int_s^t e^{au}\theta(u)du + \sigma\int_s^t e^{au}dW_u
$$

Multiply both sides by $e^{-at}$ to obtain the result. $\square$

### 3.2 Distribution of the Log-Rate

**Theorem:** Given $x(s) = \ln r(s)$, the log-rate $x(t)$ for $t > s$ is normally distributed:

$$
x(t) | x(s) \sim \mathcal{N}(\mu_x(s,t), \, V_x(s,t))
$$

**Mean:**
$$
\mu_x(s,t) = x(s)e^{-a(t-s)} + \int_s^t \theta(u)e^{-a(t-u)}du
$$

For constant $\theta$:
$$
\mu_x(s,t) = \theta + (x(s) - \theta)e^{-a(t-s)}
$$

**Variance:**
$$
V_x(s,t) = \frac{\sigma^2}{2a}(1 - e^{-2a(t-s)})
$$

**Proof:**

Identical to Hull-White, using Itô isometry for the stochastic integral. $\square$

### 3.3 Distribution of the Short Rate

**Theorem:** Since $r(t) = e^{x(t)}$ and $x(t)$ is Gaussian, $r(t)$ is **log-normally distributed**.

$$
r(t) | r(s) \sim \text{LogNormal}(\mu_x(s,t), V_x(s,t))
$$

**Moments:**

$$
\mathbb{E}[r(t)|r(s)] = \exp\left(\mu_x(s,t) + \frac{V_x(s,t)}{2}\right)
$$

$$
\text{Var}(r(t)|r(s)) = \exp(2\mu_x(s,t) + V_x(s,t))[\exp(V_x(s,t)) - 1]
$$

### 3.4 Long-Term Behavior

As $t \to \infty$ (with constant $\theta$):

$$
x(\infty) \sim \mathcal{N}\left(\theta, \frac{\sigma^2}{2a}\right)
$$

**Long-term rate statistics:**
- Median rate: $r_\infty^{med} = e^\theta$
- Mean rate: $r_\infty^{mean} = \exp\left(\theta + \frac{\sigma^2}{4a}\right)$

**Half-life:** Time for log-rate shock to decay by 50%:
$$
t_{1/2} = \frac{\ln 2}{a}
$$

---

## 4. Non-Affine Structure and Bond Pricing

### 4.1 Why Black-Karasinski is Non-Affine

**Definition:** A model is **affine** if bond prices have the form:
$$
P(t, T) = A(t, T)e^{-B(t, T)r(t)}
$$

**Theorem:** Black-Karasinski is **not affine**.

**Proof:**

For an affine model, the bond pricing PDE must admit solutions where $\ln P$ is linear in the state variable. In Black-Karasinski, the state is $x = \ln r$, and the bond pricing equation involves $r = e^x$, which is non-linear in $x$.

The bond pricing PDE:
$$
\frac{\partial P}{\partial t} + [\theta - ax]\frac{\partial P}{\partial x} + \frac{\sigma^2}{2}\frac{\partial^2 P}{\partial x^2} = e^x P
$$

The right-hand side $e^x P$ is non-linear in $x$, preventing affine solutions. $\square$

### 4.2 Fundamental Bond Pricing Formula

**Risk-Neutral Valuation:**

$$
P(t, T) = \mathbb{E}^{\mathbb{Q}}\left[\exp\left(-\int_t^T r(s)ds\right) \middle| \mathcal{F}_t\right]
$$

**Key Challenge:** The integral $\int_t^T r(s)ds = \int_t^T e^{x(s)}ds$ involves the exponential of a Gaussian process, which has no closed-form expression.

### 4.3 No Closed-Form Bond Prices

**Theorem:** There is no closed-form solution for zero-coupon bond prices in the Black-Karasinski model.

**Implication:** All bond pricing requires numerical methods:
1. **Monte Carlo simulation**
2. **Finite difference methods**
3. **Tree-based methods**

### 4.4 Bond Pricing PDE

The zero-coupon bond price $P(x, t; T)$ satisfies:

$$
\frac{\partial P}{\partial t} + (\theta - ax)\frac{\partial P}{\partial x} + \frac{\sigma^2}{2}\frac{\partial^2 P}{\partial x^2} = e^x P
$$

**Boundary Conditions:**
- $P(x, T; T) = 1$ (terminal condition)
- $P(x, t) \to 0$ as $x \to +\infty$ (high rates kill value)
- $P(x, t) \to 1$ as $x \to -\infty$ (zero rates = no discounting)

---

## 5. Option Pricing

### 5.1 Bond Option Pricing

A European call on a zero-coupon bond with expiry $T_0$ and underlying maturity $T > T_0$:

$$
C = \mathbb{E}^{\mathbb{Q}}\left[e^{-\int_0^{T_0}r(s)ds}(P(T_0, T) - K)^+\right]
$$

**No closed-form solution exists.** Must use:
- Monte Carlo with simulated discount factors
- Finite difference on 2D grid (time, log-rate)

### 5.2 Caplet and Floorlet Pricing

A caplet on LIBOR rate $L(T_{i-1}, T_i)$:

$$
\text{Caplet} = \mathbb{E}^{\mathbb{Q}}\left[e^{-\int_0^{T_i}r(s)ds} \cdot \tau(L(T_{i-1}, T_i) - K)^+\right]
$$

Using the bond-option equivalence:
$$
\text{Caplet} = (1 + \tau K) \cdot \text{Put on } P(T_{i-1}, T_i)
$$

### 5.3 Swaption Pricing

Unlike Hull-White, **Jamshidian decomposition does not apply** directly because bond prices are not monotonic functions of the state variable in a simple closed form.

**Swaption pricing requires:**
- Full Monte Carlo simulation
- Tree-based backward induction
- Finite difference on multi-dimensional grid

---

## 6. Numerical Methods

### 6.1 Monte Carlo Simulation

**Exact Simulation Scheme (Recommended):**

The log-rate follows an OU process with exact transition:

$$
x(t+\Delta t) | x(t) \sim \mathcal{N}\left(\theta + (x(t) - \theta)e^{-a\Delta t}, \frac{\sigma^2}{2a}(1 - e^{-2a\Delta t})\right)
$$

**Algorithm:**
1. Draw $Z \sim \mathcal{N}(0, 1)$
2. Compute: $x_{n+1} = \theta + (x_n - \theta)e^{-a\Delta t} + \sigma\sqrt{\frac{1 - e^{-2a\Delta t}}{2a}} \cdot Z$
3. Set: $r_{n+1} = e^{x_{n+1}}$

**Euler-Maruyama Scheme:**

$$
x_{n+1} = x_n + a(\theta - x_n)\Delta t + \sigma\sqrt{\Delta t} \cdot Z
$$

Less accurate but simpler.

### 6.2 Discount Factor Computation

For path $\omega$ with rates $r_0, r_1, \ldots, r_N$:

$$
DF(\omega) = \exp\left(-\sum_{i=0}^{N-1} r_i \Delta t\right) \approx \exp\left(-\int_0^T r(s)ds\right)
$$

**Trapezoidal rule (more accurate):**
$$
DF(\omega) = \exp\left(-\frac{\Delta t}{2}\sum_{i=0}^{N-1}(r_i + r_{i+1})\right)
$$

### 6.3 Finite Difference Method

**Grid transformation:** Work in log-rate space $x = \ln r$.

**Discretized PDE (Crank-Nicolson):**

$$
\frac{V_j^{n+1} - V_j^n}{\Delta t} = \frac{1}{2}[\mathcal{L}^{n+1}V^{n+1} + \mathcal{L}^n V^n]
$$

Where the operator is:
$$
\mathcal{L}V = (\theta - ax)\frac{V_{j+1} - V_{j-1}}{2\Delta x} + \frac{\sigma^2}{2}\frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta x^2} - e^{x_j}V_j
$$

### 6.4 Tree Methods

**Trinomial Tree Construction:**

Build a tree for $x = \ln r$ using standard trinomial tree methodology, then convert to rates via $r = e^x$.

**Advantages:**
- Handles American exercise
- Efficient for simple products

---

## 7. Calibration Theory

### 7.1 Term Structure Fitting

**Objective:** Choose $\theta(t)$ such that:
$$
P^{model}(0, T) = P^{market}(0, T) \quad \forall T
$$

**Challenge:** No closed-form for $P$, so $\theta(t)$ must be found numerically.

**Iterative Algorithm:**
1. Start with initial guess for $\theta(t)$
2. Compute model bond prices via MC/FD
3. Adjust $\theta(t)$ to match market
4. Repeat until convergence

### 7.2 Volatility Calibration

**Objective:** Find $a$ and $\sigma$ matching cap/swaption volatilities.

$$
\min_{a, \sigma} \sum_{i} w_i \left(\sigma^{model}_i - \sigma^{market}_i\right)^2
$$

**Computational cost:** High, since each model price requires MC/FD.

### 7.3 Practical Calibration Strategy

1. **Fix $a$** from historical rate behavior or market intuition
2. **Bootstrap $\theta(t)$** to match yield curve
3. **Calibrate $\sigma$** to ATM caps/swaptions
4. **Iterate** if needed

---

## 8. Comparison with Affine Models

### 8.1 Black-Karasinski vs Hull-White

| Aspect | Hull-White | Black-Karasinski |
|--------|------------|------------------|
| **Rate distribution** | Gaussian | Log-normal |
| **Negative rates** | Yes | No (never) |
| **Bond pricing** | Closed-form | Numerical only |
| **Option pricing** | Semi-analytic | Numerical only |
| **Jamshidian** | Applies | Does not apply |
| **Calibration** | Fast (analytic) | Slow (numerical) |
| **Volatility structure** | Additive | Proportional |

### 8.2 Affine vs Non-Affine Trade-offs

**Affine Models (Hull-White, Vasicek, CIR):**
- ✅ Closed-form bond prices
- ✅ Fast calibration
- ✅ Analytic Greeks
- ❌ May not fit volatility smile well
- ❌ Hull-White: negative rates possible

**Non-Affine Models (Black-Karasinski, Black-Derman-Toy):**
- ✅ Guaranteed positive rates
- ✅ More flexible volatility structure
- ❌ No closed-form solutions
- ❌ Slow calibration
- ❌ Numerical Greeks

---

## 9. Advantages and Limitations

### 9.1 Advantages

1. **Positive Rates**: $r(t) = e^{x(t)} > 0$ always
2. **Proportional Volatility**: Rate vol scales with rate level (more realistic)
3. **Fat Tails**: Log-normal has heavier tails than Gaussian
4. **Term Structure Fit**: Can match any initial yield curve with $\theta(t)$

### 9.2 Limitations

1. **No Closed-Form**: All pricing requires numerical methods
2. **Slow Calibration**: Iterative numerical procedures
3. **No Negative Rates**: Unsuitable for EUR/CHF/JPY post-2014
4. **Complexity**: More complex than Hull-White

### 9.3 When to Use Black-Karasinski

**Appropriate:**
- Markets where negative rates are impossible
- When proportional volatility is desired
- Historical modeling (pre-2014)
- Comparison/validation against Gaussian models

**Not Appropriate:**
- Modern EUR/CHF/JPY markets
- When speed is critical
- Simple hedging applications

---

## 10. Interview Key Points

### Derivation Questions

**Q: Why is Black-Karasinski non-affine?**

A: In an affine model, bond prices are $P = A(t,T)e^{-B(t,T) \cdot \text{state}}$. In BK, the state is $x = \ln r$, but the discounting involves $r = e^x$. The bond pricing PDE has the term $e^x P$, which is non-linear in $x$, preventing affine solutions.

**Q: Derive the distribution of $r(t)$ in Black-Karasinski.**

A: 
1. $x(t) = \ln r(t)$ follows an OU process
2. OU has Gaussian distribution: $x(t) \sim \mathcal{N}(\mu_x, V_x)$
3. $r(t) = e^{x(t)}$ is log-normal by definition
4. Moments: $\mathbb{E}[r] = e^{\mu_x + V_x/2}$

**Q: Why can't you use Jamshidian decomposition for BK swaptions?**

A: Jamshidian requires that all bond prices are monotonic functions of a single state variable with known functional form. In BK, bond prices don't have closed-form expressions in terms of $x$, so we can't identify the critical rate $r^*$ analytically.

### Practical Questions

**Q: How do you price a bond in Black-Karasinski?**

A: Monte Carlo:
1. Simulate $N$ paths of $x(t)$ using exact OU transitions
2. Compute $r(t) = e^{x(t)}$ along each path
3. Compute discount factors: $DF = e^{-\int_0^T r(s)ds}$
4. Bond price = average of $DF$ across paths

**Q: What numerical method would you use for BK?**

A: 
- **Monte Carlo**: Path-dependent products, simple implementation
- **Finite Difference**: European options, need Greeks
- **Trees**: American options, moderate complexity

**Q: BK vs Hull-White - when would you choose each?**

A:
- **Hull-White**: Speed matters, negative rates possible, need analytics
- **Black-Karasinski**: Positive rates required, proportional vol wanted, accuracy over speed

---

## Appendix: Key Formulas

### A.1 Log-Rate Dynamics

$$
dx(t) = (\theta - ax)dt + \sigma dW, \quad r(t) = e^{x(t)}
$$

### A.2 Exact Transition

$$
x(t+\Delta t) | x(t) \sim \mathcal{N}\left(\theta + (x(t)-\theta)e^{-a\Delta t}, \frac{\sigma^2}{2a}(1-e^{-2a\Delta t})\right)
$$

### A.3 Asymptotic Distribution

$$
x(\infty) \sim \mathcal{N}\left(\theta, \frac{\sigma^2}{2a}\right)
$$

### A.4 Bond Pricing PDE

$$
\frac{\partial P}{\partial t} + (\theta - ax)\frac{\partial P}{\partial x} + \frac{\sigma^2}{2}\frac{\partial^2 P}{\partial x^2} = e^x P
$$

---

## References

1. Black, F. & Karasinski, P. (1991). "Bond and Option Pricing when Short Rates are Lognormal." *Financial Analysts Journal*.
2. Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*. Springer.
3. Hull, J.C. (2022). *Options, Futures, and Other Derivatives*, 11th ed.
4. Rebonato, R. (1998). *Interest-Rate Option Models*. Wiley.
5. Pelsser, A. (2000). *Efficient Methods for Valuing Interest Rate Derivatives*. Springer.

---

*Document Version: 2.0 | QuantStrata Phase 3 | January 2026*
