# Barrier Options: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Product Class:** Path-Dependent European Exotic Option  
**Pricing Methods:** Analytic (continuous), Monte Carlo (discrete), Finite Difference  
**Target Audience:** Quantitative Analysts, Financial Mathematics Graduates

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Formal Mathematical Framework](#2-formal-mathematical-framework)
3. [Product Specification](#3-product-specification)
4. [Pricing Theory](#4-pricing-theory)
5. [Greeks and Sensitivities](#5-greeks-and-sensitivities)
6. [Numerical Methods](#6-numerical-methods)
7. [Risk Management](#7-risk-management)
8. [Key Interview Points](#8-key-interview-points)
9. [References](#9-references)

---

## 1. Executive Summary

### 1.1 Product Overview

A **barrier option** is a path-dependent option that is either activated (knock-in) or extinguished (knock-out) if the underlying price crosses a predetermined barrier level during the option's life.

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Path Dependency** | Option existence depends on whether barrier is hit |
| **Barrier Types** | Up/Down (barrier above/below spot) |
| **Barrier Styles** | Knock-In/Knock-Out |
| **Monitoring** | Continuous or Discrete |
| **Closed-Form** | Yes for continuous monitoring (Rubinstein-Reiner) |

### 1.3 Barrier Option Types

| Type | Notation | Condition |
|------|----------|-----------|
| **Down-and-Out** | DO | Knocked out if S hits lower barrier |
| **Down-and-In** | DI | Knocked in if S hits lower barrier |
| **Up-and-Out** | UO | Knocked out if S hits upper barrier |
| **Up-and-In** | UI | Knocked in if S hits upper barrier |

### 1.4 Key Relationships

**In-Out Parity:**
\[
\text{Knock-In} + \text{Knock-Out} = \text{Vanilla}
\]

This fundamental relationship enables arbitrage-free pricing consistency.

---

## 2. Formal Mathematical Framework

### 2.1 Underlying Dynamics

Under the risk-neutral measure \(\mathbb{Q}\):

\[
dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
\]

### 2.2 Barrier Hitting Time

Define the first hitting time of barrier H:

\[
\tau_H = \inf\{t \geq 0 : S_t = H\}
\]

The option payoff depends on whether \(\tau_H \leq T\).

### 2.3 Reflection Principle for Barriers

For a barrier H and log-process \(X_t = \log(S_t/S_0)\):

**Down-barrier (H < S₀):**
\[
\mathbb{P}(\min_{0 \leq t \leq T} S_t \leq H, S_T > K) = \text{involves reflection at barrier}
\]

**Up-barrier (H > S₀):**
\[
\mathbb{P}(\max_{0 \leq t \leq T} S_t \geq H, S_T > K) = \text{involves reflection at barrier}
\]

### 2.4 Key Parameter

**Log-barrier ratio:**
\[
\lambda = \frac{r - q + \sigma^2/2}{\sigma^2}
\]

This parameter appears in the analytical formulas and determines the "drift-adjusted" reflection.

---

## 3. Product Specification

### 3.1 Knock-Out Options

**Down-and-Out Call (DOC):**
\[
\text{Payoff} = (S_T - K)^+ \cdot \mathbf{1}_{\min_{0 \leq t \leq T} S_t > H}
\]

Option pays vanilla call payoff if barrier H (below spot) is never touched.

**Up-and-Out Call (UOC):**
\[
\text{Payoff} = (S_T - K)^+ \cdot \mathbf{1}_{\max_{0 \leq t \leq T} S_t < H}
\]

Option pays vanilla call payoff if barrier H (above spot) is never touched.

**Down-and-Out Put (DOP):**
\[
\text{Payoff} = (K - S_T)^+ \cdot \mathbf{1}_{\min_{0 \leq t \leq T} S_t > H}
\]

**Up-and-Out Put (UOP):**
\[
\text{Payoff} = (K - S_T)^+ \cdot \mathbf{1}_{\max_{0 \leq t \leq T} S_t < H}
\]

### 3.2 Knock-In Options

**Down-and-In Call (DIC):**
\[
\text{Payoff} = (S_T - K)^+ \cdot \mathbf{1}_{\min_{0 \leq t \leq T} S_t \leq H}
\]

Option activates only if barrier H is touched.

**Up-and-In Call (UIC):**
\[
\text{Payoff} = (S_T - K)^+ \cdot \mathbf{1}_{\max_{0 \leq t \leq T} S_t \geq H}
\]

Similarly for puts.

### 3.3 Rebate

A **rebate** R may be paid:
- For knock-out: Paid at hitting time (or expiry if barrier not hit)
- For knock-in: Paid at expiry if barrier never hit

### 3.4 Contract Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Spot Price | \(S_0\) | Current underlying price |
| Strike | \(K\) | Exercise price |
| Barrier | \(H\) | Knock-in/out level |
| Rebate | \(R\) | Compensation amount |
| Maturity | \(T\) | Time to expiry |
| Monitoring | - | Continuous/Discrete |

---

## 4. Pricing Theory

### 4.1 In-Out Parity

**Fundamental Relationship:**
\[
\boxed{V_{\text{KI}} + V_{\text{KO}} = V_{\text{vanilla}}}
\]

This holds for same barrier level, strike, and maturity.

**Implication:** Once we price knock-out, we get knock-in for free (and vice versa).

### 4.2 Rubinstein-Reiner Formulas (Continuous Monitoring)

#### Building Blocks

Define:
\[
\phi = \pm 1 \text{ (call/put)}, \quad \eta = \pm 1 \text{ (down/up barrier)}
\]

Key terms:
\[
x_1 = \frac{\log(S/K)}{\sigma\sqrt{T}} + \lambda \sigma\sqrt{T}
\]
\[
x_2 = \frac{\log(S/H)}{\sigma\sqrt{T}} + \lambda \sigma\sqrt{T}
\]
\[
y_1 = \frac{\log(H^2/(SK))}{\sigma\sqrt{T}} + \lambda \sigma\sqrt{T}
\]
\[
y_2 = \frac{\log(H/S)}{\sigma\sqrt{T}} + \lambda \sigma\sqrt{T}
\]

where \(\lambda = (r - q + \sigma^2/2)/\sigma^2\).

#### Down-and-Out Call (H < S, H < K)

\[
V_{\text{DOC}} = S e^{-qT} [N(\phi x_1) - (H/S)^{2\lambda} N(\phi y_1)]
\]
\[
- K e^{-rT} [N(\phi x_1 - \phi\sigma\sqrt{T}) - (H/S)^{2\lambda-2} N(\phi y_1 - \phi\sigma\sqrt{T})]
\]

#### General Formula Structure

Each barrier option price involves:
1. Standard vanilla-like terms
2. Reflection terms with \((H/S)^{2\lambda}\) multiplier
3. Combinations based on barrier type and direction

### 4.3 Pricing Logic by Type

| Type | When Active | Price Approach |
|------|-------------|----------------|
| **Down-and-Out** | S never hits H | Vanilla minus "lost" scenarios |
| **Down-and-In** | S hits H at least once | Parity: Vanilla - DO |
| **Up-and-Out** | S never hits H | Vanilla minus scenarios above H |
| **Up-and-In** | S hits H at least once | Parity: Vanilla - UO |

### 4.4 Barrier Option Price Ordering

For down barriers (H < S₀):
\[
0 \leq V_{\text{DO}} \leq V_{\text{vanilla}}
\]
\[
0 \leq V_{\text{DI}} \leq V_{\text{vanilla}}
\]

**Intuition:** Barriers can only restrict when the option pays, so barrier options are always ≤ vanilla.

---

## 5. Greeks and Sensitivities

### 5.1 Delta

**Key Property:** Delta can be discontinuous at the barrier.

For a down-and-out call near the barrier:
- Delta → 0 as S → H from above (option becomes worthless)
- Delta can jump when crossing barrier

### 5.2 Gamma

**Key Property:** Gamma can spike dramatically near the barrier.

As S approaches H:
- Knock-out: Gamma becomes very large (option dying/surviving)
- Knock-in: Gamma behavior depends on being activated

### 5.3 Barrier Greeks Challenges

| Greek | Challenge |
|-------|-----------|
| Delta | Discontinuous at barrier |
| Gamma | Spikes near barrier |
| Vega | Can be negative near barrier |
| Theta | Complex time behavior |

### 5.4 Numerical Computation

Greeks are typically computed via:
1. **Finite differences** (bump-and-reprice)
2. **Pathwise derivatives** (for MC)
3. **Likelihood ratio method** (for MC)

---

## 6. Numerical Methods

### 6.1 Monte Carlo (Discrete Monitoring)

**Algorithm:**
```
1. Simulate N paths of S with n_steps time points
2. For each path:
   a. Track min(S) and max(S)
   b. Check barrier condition
   c. Compute payoff if condition satisfied
3. Average and discount
```

**Key Consideration:** Discrete monitoring gives different prices than continuous.

### 6.2 Discrete vs Continuous Monitoring

**Discrete < Continuous** for knock-in options
**Discrete > Continuous** for knock-out options

**Broadie-Glasserman-Kou Correction:**
\[
H_{\text{eff}} = H \cdot e^{\pm 0.5826 \sigma \sqrt{\Delta t}}
\]

where + for up-barrier, - for down-barrier.

### 6.3 Finite Difference Methods

**Challenge:** Barrier creates non-smooth boundary conditions.

**Approach:**
1. Place barrier on grid line
2. Apply Dirichlet boundary condition at barrier
3. Use fine mesh near barrier

### 6.4 Binomial/Trinomial Trees

**Challenge:** Barrier may not coincide with tree nodes.

**Solutions:**
1. **Adjust tree** to ensure barrier is on node
2. **Interpolate** between nodes
3. **Adaptive mesh** refinement

---

## 7. Risk Management

### 7.1 Hedging Challenges

**Near-Barrier Behavior:**
- Delta becomes unstable
- Need frequent rebalancing
- Gap risk if barrier is breached

**Practical Approaches:**
1. **Barrier shift:** Price with shifted barrier accounting for uncertainty
2. **Wide corridor:** Use knock-in + knock-out spread
3. **Vega hedge:** Trade other options to reduce vol exposure

### 7.2 Gap Risk

If price gaps through barrier:
- Knock-out might not be triggered exactly at H
- Knock-in might activate at price far from H
- Creates P&L slippage

### 7.3 Model Risk

| Assumption | Reality | Impact |
|------------|---------|--------|
| Continuous monitoring | Discrete | Price difference |
| No jumps | Jumps exist | Gap through barrier |
| Constant vol | Vol smile | Mispricing near barrier |

---

## 8. Key Interview Points

### 8.1 Must-Know Facts

1. **In-Out Parity:** KI + KO = Vanilla
2. **Barrier options ≤ Vanilla:** Barrier restricts payoff
3. **Discrete vs Continuous:** Discrete KO > Continuous KO
4. **Greeks discontinuity:** Delta jumps at barrier
5. **Reflection principle:** Key to analytical pricing
6. **Rubinstein-Reiner (1991):** Standard analytical formulas

### 8.2 Common Interview Questions

**Q: Explain In-Out Parity.**

A: A knock-in and knock-out option with the same parameters sum to a vanilla option. Intuitively: either the barrier is hit (KI pays) or not (KO pays), and together they cover all scenarios, exactly like a vanilla.

**Q: Why are barrier options cheaper than vanilla?**

A: The barrier can only reduce the option value—it either has no effect (vanilla payoff) or knocks out/fails to knock in (zero or reduced payoff). Therefore:
\[
V_{\text{barrier}} \leq V_{\text{vanilla}}
\]

**Q: How does discrete monitoring affect barrier option prices?**

A: Discrete monitoring reduces the chance of hitting the barrier (you might "miss" it between observations). Therefore:
- Knock-out with discrete monitoring is worth **more** (less likely to knock out)
- Knock-in with discrete monitoring is worth **less** (less likely to knock in)

**Q: Why is hedging barrier options difficult?**

A: 
1. Delta becomes discontinuous at the barrier
2. Gamma spikes dramatically near barrier
3. Gap risk: price might jump through barrier
4. Model uncertainty (discrete vs continuous, jumps)

**Q: What is the Broadie-Glasserman-Kou correction?**

A: For discrete monitoring, we can approximate continuous monitoring by shifting the barrier:
\[
H_{\text{eff}} = H \cdot e^{\pm 0.5826 \sigma \sqrt{\Delta t}}
\]
This accounts for the probability of crossing between observations.

### 8.3 Quick Formulas

**In-Out Parity:**
\[
V_{\text{KI}} + V_{\text{KO}} = V_{\text{vanilla}}
\]

**Key Parameter:**
\[
\lambda = \frac{r - q + \sigma^2/2}{\sigma^2}
\]

**Reflection Factor:**
\[
\left(\frac{H}{S}\right)^{2\lambda}
\]

**Discrete Correction:**
\[
H_{\text{eff}} = H \cdot e^{\pm 0.5826 \sigma \sqrt{\Delta t}}
\]

---

## 9. References

### Academic Papers

1. **Merton, R.C.** (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics*, 4(1), 141-183.

2. **Rubinstein, M. and Reiner, E.** (1991). "Breaking Down the Barriers." *Risk*, 4(8), 28-35.

3. **Broadie, M., Glasserman, P., and Kou, S.** (1997). "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance*, 7(4), 325-349.

4. **Carr, P. and Chou, A.** (1997). "Breaking Barriers." *Risk*, 10(9), 139-145.

### Textbooks

5. **Hull, J.C.** (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson. Chapter 26.

6. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*. Wiley. Volume 2.

7. **Haug, E.G.** (2007). *The Complete Guide to Option Pricing Formulas*. McGraw-Hill.

---

## Appendix A: In-Out Parity Proof

### A.1 Setup

Consider a knock-in and knock-out option with identical:
- Strike K
- Barrier H
- Maturity T
- Barrier direction (up or down)

### A.2 Proof

Let \(\tau_H\) be the first hitting time of barrier H.

**Knock-Out Payoff:**
\[
\text{Payoff}_{\text{KO}} = (S_T - K)^+ \cdot \mathbf{1}_{\tau_H > T}
\]

**Knock-In Payoff:**
\[
\text{Payoff}_{\text{KI}} = (S_T - K)^+ \cdot \mathbf{1}_{\tau_H \leq T}
\]

**Sum:**
\[
\text{Payoff}_{\text{KO}} + \text{Payoff}_{\text{KI}} = (S_T - K)^+ \cdot (\mathbf{1}_{\tau_H > T} + \mathbf{1}_{\tau_H \leq T})
\]

Since \(\mathbf{1}_{\tau_H > T} + \mathbf{1}_{\tau_H \leq T} = 1\):

\[
\text{Payoff}_{\text{KO}} + \text{Payoff}_{\text{KI}} = (S_T - K)^+ = \text{Payoff}_{\text{vanilla}}
\]

Taking risk-neutral expectations:
\[
\boxed{V_{\text{KI}} + V_{\text{KO}} = V_{\text{vanilla}}}
\]

---

## Appendix B: Barrier Option Classification

### B.1 Complete Classification (8 Types for Calls)

| Barrier | Direction | Call Value vs Vanilla |
|---------|-----------|----------------------|
| H < S, H < K | Down-and-Out | Less |
| H < S, H > K | Down-and-Out | Less |
| H < S, H < K | Down-and-In | Less |
| H < S, H > K | Down-and-In | Less |
| H > S, H > K | Up-and-Out | Less |
| H > S, H < K | Up-and-Out | Less |
| H > S, H > K | Up-and-In | Less |
| H > S, H < K | Up-and-In | Less |

**Key Insight:** All barrier options are worth less than (or equal to) the corresponding vanilla.

### B.2 Degenerate Cases

- **Barrier very far from spot:** Barrier option ≈ vanilla (barrier rarely hit)
- **Barrier at spot:** Option may have special behavior
- **Barrier beyond strike:** Different analytical formulas apply

---

*Document Version: 1.0*  
*Last Updated: January 27, 2026*  
*Author: QuantStrata Library*
