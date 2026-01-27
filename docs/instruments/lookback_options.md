# Lookback Options: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Product Class:** Path-Dependent Exotic Option  
**Pricing Methods:** Monte Carlo (discrete), Analytic (continuous - Goldman-Sosin-Gatto)  
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
8. [Implementation](#8-implementation)
9. [Key Interview Points](#9-key-interview-points)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Product Overview

A **lookback option** is a path-dependent derivative whose payoff depends on the **maximum or minimum** price of the underlying asset over the option's life. This provides "perfect hindsight" - the holder always captures the optimal entry or exit point.

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Path Dependency** | Payoff depends on running extremum (max or min) |
| **Perfect Hindsight** | Holder captures optimal timing retrospectively |
| **Premium Over Vanilla** | Always more expensive (captures timing value) |
| **Closed-Form (Continuous)** | Goldman-Sosin-Gatto (1979) formulas |
| **Primary Use Cases** | Timing risk hedging, speculation on price range |

### 1.3 Variants Summary

| Variant | Call Payoff | Put Payoff | Notes |
|---------|-------------|------------|-------|
| **Floating Strike** | \(S_T - m_T\) | \(M_T - S_T\) | Always ITM |
| **Fixed Strike** | \(\max(M_T - K, 0)\) | \(\max(K - m_T, 0)\) | Option on extremum |

Where:
- \(M_T = \max_{0 \leq t \leq T} S_t\) (running maximum)
- \(m_T = \min_{0 \leq t \leq T} S_t\) (running minimum)

---

## 2. Formal Mathematical Framework

### 2.1 Probability Space and Filtration

Let \((\Omega, \mathcal{F}, \mathbb{P})\) be a probability space with filtration \(\{\mathcal{F}_t\}_{t \geq 0}\) satisfying the usual conditions.

**Brownian Motion**: Let \(W = \{W_t\}_{t \geq 0}\) be a standard Brownian motion adapted to \(\{\mathcal{F}_t\}\).

### 2.2 Asset Dynamics

Under the risk-neutral measure \(\mathbb{Q}\):

\[
dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
\]

**Solution:**
\[
S_t = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)t + \sigma W_t^{\mathbb{Q}}\right]
\]

### 2.3 Running Extrema

**Running Maximum:**
\[
M_t = \max_{0 \leq s \leq t} S_s = S_0 \exp\left[\max_{0 \leq s \leq t} X_s\right]
\]

**Running Minimum:**
\[
m_t = \min_{0 \leq s \leq t} S_s = S_0 \exp\left[\min_{0 \leq s \leq t} X_s\right]
\]

where \(X_t = \log(S_t/S_0)\) is the log-return process.

### 2.4 Reflection Principle

The **reflection principle** is fundamental to lookback option pricing. For standard Brownian motion \(W_t\):

**Theorem (Reflection Principle):**
\[
\mathbb{P}\left(\max_{0 \leq s \leq t} W_s \geq a\right) = 2\mathbb{P}(W_t \geq a) = 2\Phi\left(-\frac{a}{\sqrt{t}}\right)
\]

for \(a > 0\), where \(\Phi\) is the standard normal CDF.

**Joint Density**: The joint density of \((W_T, M_T^W)\) where \(M_T^W = \max_{0 \leq t \leq T} W_t\):

\[
f_{W_T, M_T^W}(w, m) = \frac{2(2m - w)}{T} \phi\left(\frac{2m - w}{\sqrt{T}}\right), \quad m \geq \max(0, w)
\]

where \(\phi\) is the standard normal PDF.

### 2.5 Distribution of Running Maximum

For the log-price process \(X_t = \mu t + \sigma W_t\) with drift \(\mu = r - q - \sigma^2/2\):

The running maximum \(\bar{X}_T = \max_{0 \leq t \leq T} X_t\) has the density:

\[
f_{\bar{X}_T}(x) = \frac{2}{\sigma\sqrt{T}}\phi\left(\frac{x - \mu T}{\sigma\sqrt{T}}\right) - \frac{2\mu}{\sigma^2}e^{\frac{2\mu x}{\sigma^2}}\Phi\left(\frac{-x - \mu T}{\sigma\sqrt{T}}\right)
\]

for \(x \geq 0\).

---

## 3. Product Specification

### 3.1 Floating Strike Lookback

#### Call (Buy at the Low)

**Payoff:**
\[
\text{Payoff}_{\text{call}} = S_T - m_T = S_T - \min_{0 \leq t \leq T} S_t
\]

**Interpretation**: The holder effectively buys the asset at the lowest price achieved during the option's life.

**Key Property**: **Always in-the-money** since \(S_T \geq m_T\) by definition.

#### Put (Sell at the High)

**Payoff:**
\[
\text{Payoff}_{\text{put}} = M_T - S_T = \max_{0 \leq t \leq T} S_t - S_T
\]

**Interpretation**: The holder effectively sells the asset at the highest price achieved during the option's life.

**Key Property**: **Always in-the-money** since \(M_T \geq S_T\) by definition.

### 3.2 Fixed Strike Lookback

#### Call (Option on Maximum)

**Payoff:**
\[
\text{Payoff}_{\text{call}} = \max(M_T - K, 0) = (M_T - K)^+
\]

**Interpretation**: A standard call option where the underlying is the running maximum \(M_T\).

#### Put (Option on Minimum)

**Payoff:**
\[
\text{Payoff}_{\text{put}} = \max(K - m_T, 0) = (K - m_T)^+
\]

**Interpretation**: A standard put option where the underlying is the running minimum \(m_T\).

### 3.3 Contract Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Initial Spot | \(S_0\) | Current spot price |
| Strike | \(K\) | Fixed strike (fixed strike only) |
| Maturity | \(T\) | Time to expiry |
| Monitoring | - | Continuous or Discrete |
| Current Max/Min | \(M_0, m_0\) | Starting extremum (if already accumulated) |

### 3.4 Fundamental Inequalities

**Theorem (Lookback Premium):**

For any fixed strike \(K\):
\[
C_{\text{lookback}}(K) \geq C_{\text{vanilla}}(K)
\]
\[
P_{\text{lookback}}(K) \geq P_{\text{vanilla}}(K)
\]

**Proof**: Since \(M_T \geq S_T\), we have:
\[
\max(M_T - K, 0) \geq \max(S_T - K, 0) \quad \text{pathwise}
\]
Taking expectations preserves the inequality.

---

## 4. Pricing Theory

### 4.1 Risk-Neutral Pricing

The price of a lookback option is:

\[
V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\text{Payoff}]
\]

For floating strike lookback call:
\[
V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[S_T - m_T]
\]

### 4.2 Goldman-Sosin-Gatto Formula (Continuous Monitoring)

#### Floating Strike Lookback Call

For a floating strike lookback call with continuous monitoring, starting at \(m_0 = S_0\):

\[
C_{\text{float}} = S_0 e^{-qT} N(a_1) - S_0 e^{-qT} \frac{\sigma^2}{2(r-q)} \left[ \left(\frac{S_0}{m_0}\right)^{-2(r-q)/\sigma^2} N(-a_1 + 2(r-q)\sqrt{T}/\sigma) - e^{(r-q)T} N(-a_1) \right]
\]
\[
- m_0 e^{-rT} N(a_2) + m_0 e^{-rT} \frac{\sigma^2}{2(r-q)} \left[ \left(\frac{S_0}{m_0}\right)^{1-2(r-q)/\sigma^2} N(-a_2 + 2(r-q)\sqrt{T}/\sigma) - e^{(r-q)T} N(-a_2) \right]
\]

where:
\[
a_1 = \frac{\log(S_0/m_0) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad a_2 = a_1 - \sigma\sqrt{T}
\]

#### Simplified Formula at Inception

At inception where \(m_0 = S_0\):

\[
C_{\text{float}} = S_0 e^{-qT} N(d_1) - S_0 e^{-rT} N(d_2) + S_0 e^{-rT} \frac{\sigma^2}{2(r-q)} \left[ e^{(r-q)T} N(d_1) - N(-d_1) \right] - S_0 e^{-qT} \frac{\sigma^2}{2(r-q)} N(-d_1)
\]

where \(d_1 = \frac{(r-q+\sigma^2/2)T}{\sigma\sqrt{T}}\) and \(d_2 = d_1 - \sigma\sqrt{T}\).

### 4.3 Fixed Strike Lookback Call

For fixed strike \(K\) (option on maximum):

\[
C_{\text{fixed}} = (M_0 - K)^+ e^{-rT} + S_0 e^{-qT} N(b_1) - K e^{-rT} N(b_1 - \sigma\sqrt{T})
\]
\[
+ S_0 e^{-qT} \frac{\sigma^2}{2(r-q)} \left[ -\left(\frac{S_0}{M_0}\right)^{-2(r-q)/\sigma^2} N(\eta_1) + e^{(r-q)T} N(-b_1) \right]
\]

where:
\[
b_1 = \frac{\log(S_0/K) + (r-q+\sigma^2/2)T}{\sigma\sqrt{T}} \quad (\text{if } M_0 = S_0)
\]

### 4.4 Decomposition Insight

The lookback call can be decomposed as:

\[
C_{\text{float}} = \underbrace{S_0 e^{-qT}}_{\text{Forward}} - \underbrace{\mathbb{E}^{\mathbb{Q}}[e^{-rT} m_T]}_{\text{Put on minimum}}
\]

This shows that a floating strike lookback call is equivalent to:
- Long the forward
- Minus the present value of the expected minimum

---

## 5. Greeks and Sensitivities

### 5.1 Delta (Δ)

**Definition:**
\[
\Delta = \frac{\partial V}{\partial S_0}
\]

**Floating Strike Lookback Call at Inception:**
\[
\Delta_{\text{float}} \approx 2 \quad \text{(at inception)}
\]

**Interpretation**: The delta of a floating strike lookback call at inception is approximately 2, making it extremely sensitive to spot movements. This is because:
1. The option gains from \(S_T\) rising (+1 delta)
2. The option gains from \(m_T\) staying low (additional +1 delta)

### 5.2 Gamma (Γ)

**Definition:**
\[
\Gamma = \frac{\partial^2 V}{\partial S_0^2}
\]

**Characteristics:**
- Very high gamma, especially near extremum levels
- Discontinuity in gamma at new extrema

### 5.3 Vega (ν)

**Definition:**
\[
\nu = \frac{\partial V}{\partial \sigma}
\]

**Characteristics:**
- Positive vega (benefits from volatility)
- Higher than vanilla options (more opportunity to hit extrema)
- Critical sensitivity since price depends on vol through extremum distribution

### 5.4 Theta (Θ)

**Definition:**
\[
\Theta = \frac{\partial V}{\partial t}
\]

**Characteristics:**
- Complex time decay
- Floating strike options decay slowly initially (value of future extrema)
- Accelerates as maturity approaches

### 5.5 Greeks Summary

| Greek | Floating Strike Call | Fixed Strike Call | vs Vanilla |
|-------|---------------------|-------------------|------------|
| Delta | ≈ 2 at inception | > 1 | Higher |
| Gamma | High, discontinuous | High | Higher |
| Vega | Positive, large | Positive, large | Higher |
| Theta | Slow initially | Varies | Complex |

---

## 6. Numerical Methods

### 6.1 Monte Carlo for Discrete Monitoring

#### Algorithm

```
Input: S₀, K (if fixed), T, r, q, σ, N_paths, N_steps
Output: Option price V₀

1. Δt ← T / N_steps
2. For each path i = 1, ..., N_paths:
   a. S ← S₀
   b. M ← S₀  (running max)
   c. m ← S₀  (running min)
   d. For j = 1, ..., N_steps:
      - Z ~ N(0,1)
      - S ← S × exp((r - q - σ²/2)Δt + σ√Δt × Z)
      - M ← max(M, S)
      - m ← min(m, S)
   e. S_T ← S
   f. Compute payoff:
      - Floating call: S_T - m
      - Floating put: M - S_T
      - Fixed call: max(M - K, 0)
      - Fixed put: max(K - m, 0)
3. V₀ ← e^{-rT} × mean(payoffs)
```

### 6.2 Discrete vs Continuous Monitoring

**Key Insight**: Discrete monitoring **underestimates** the continuous value.

**Correction Factors (Broadie-Glasserman-Kou):**

For the running maximum with discrete monitoring at times \(0 = t_0 < t_1 < \cdots < t_n = T\):

\[
\mathbb{E}[M_T^{\text{discrete}}] \approx \mathbb{E}[M_T^{\text{continuous}}] - 0.5826 \sigma \sqrt{\Delta t}
\]

where \(\Delta t = T/n\) is the monitoring interval and 0.5826 is related to \(\zeta(1/2)/\sqrt{2\pi}\).

### 6.3 Variance Reduction

#### Antithetic Variates

For each path with increments \(Z_1, \ldots, Z_n\), generate mirror path with \(-Z_1, \ldots, -Z_n\).

**Note**: Antithetic paths have:
- Mirrored paths (high becomes low)
- But different extrema! (may not reduce variance as effectively as for Asians)

#### Control Variate

Use the geometric mean of the path as control:
\[
\hat{V}_{\text{CV}} = \hat{V} - c(\hat{V}_{\text{geom}} - V_{\text{geom}}^{\text{exact}})
\]

### 6.4 PDE Method

The lookback option satisfies a 2D PDE in \((S, M)\) or \((S, m)\):

For floating strike lookback call \(V = V(t, S, m)\):

\[
\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV
\]

with boundary condition at \(S = m\):
\[
\frac{\partial V}{\partial m}\bigg|_{S=m} = 0
\]

---

## 7. Risk Management

### 7.1 Hedging Lookback Options

**Challenges:**
1. High delta (≈2 for floating strike call)
2. Discontinuous gamma at new extrema
3. Path-dependent hedge ratios

**Strategy:**
1. Delta hedge with underlying
2. Vega hedge with vanilla options
3. Rebalance more frequently near current extrema

### 7.2 Model Risk

| Risk Factor | Impact | Mitigation |
|-------------|--------|------------|
| Vol smile | High sensitivity | Use local/stochastic vol |
| Discrete monitoring | Underestimates continuous | Use correction factors |
| Jumps | Affects extrema significantly | Jump-diffusion models |

### 7.3 P&L Attribution

\[
\Delta P\&L \approx \Delta \cdot \Delta S + \frac{1}{2}\Gamma \cdot (\Delta S)^2 + \nu \cdot \Delta\sigma + \Theta \cdot \Delta t + \epsilon_{\text{extrema}}
\]

where \(\epsilon_{\text{extrema}}\) captures P&L from new extrema being set.

---

## 8. Implementation

### 8.1 Pseudocode: Floating Strike Lookback Call

```python
def price_floating_lookback_call(S0, T, r, q, sigma, n_paths, n_steps, seed=42):
    """
    Monte Carlo pricer for floating strike lookback call.
    
    Payoff = S_T - min(S_t)
    """
    np.random.seed(seed)
    
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Initialize arrays
    S = np.full(n_paths, S0)
    running_min = np.full(n_paths, S0)
    
    # Simulate paths
    for step in range(n_steps):
        Z = np.random.standard_normal(n_paths)
        S = S * np.exp(drift + diffusion * Z)
        running_min = np.minimum(running_min, S)
    
    # Payoff (always >= 0 for floating strike)
    payoffs = S - running_min
    
    # Discount
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error
```

### 8.2 Numerical Considerations

1. **Extrema Tracking**: Update max/min at each step, not just at end
2. **Log-Space Simulation**: Use log returns for numerical stability
3. **Monitoring Frequency**: More steps = closer to continuous value
4. **Edge Cases**: At T=0, all extrema equal S₀

### 8.3 Performance Optimization

- Vectorize path simulation
- Use numba/cython for inner loops
- Consider quasi-Monte Carlo (Sobol sequences)

---

## 9. Key Interview Points

### 9.1 Must-Know Facts

1. **Floating strike lookbacks are ALWAYS ITM** (payoff ≥ 0)
2. **Lookback ≥ Vanilla** (captures optimal timing value)
3. **Delta ≈ 2** for floating strike call at inception
4. **Reflection principle** is key mathematical tool
5. **Goldman-Sosin-Gatto (1979)** derived continuous monitoring formulas
6. **Discrete monitoring underestimates continuous** value

### 9.2 Common Interview Questions

**Q: Why is the delta of a floating strike lookback call approximately 2?**

A: The option benefits in two ways:
1. When \(S_T\) increases (standard delta ≈ 1)
2. When the minimum stays low (additional sensitivity ≈ 1)

At inception, \(m_0 = S_0\), so both effects are maximized, giving total delta ≈ 2.

**Q: Why is a floating strike lookback always ITM?**

A: By definition:
- For call: \(S_T \geq m_T\), so \(S_T - m_T \geq 0\)
- For put: \(M_T \geq S_T\), so \(M_T - S_T \geq 0\)

The holder always benefits from hindsight.

**Q: How does discrete monitoring affect the price?**

A: Discrete monitoring gives a **lower** price than continuous monitoring because:
- You may "miss" the true extremum between monitoring points
- More monitoring points converge to continuous value
- Broadie-Glasserman-Kou correction: subtract \(0.5826\sigma\sqrt{\Delta t}\) from expected max

**Q: What is the reflection principle and why is it important?**

A: The reflection principle states that for Brownian motion:
\[
\mathbb{P}(\max_{0\leq s\leq t} W_s \geq a) = 2\mathbb{P}(W_t \geq a)
\]

This allows us to derive the distribution of the running maximum (and minimum), which is essential for closed-form lookback pricing.

### 9.3 Quick Formulas to Remember

| Formula | Expression |
|---------|------------|
| Floating Call Payoff | \(S_T - m_T\) |
| Floating Put Payoff | \(M_T - S_T\) |
| Fixed Call Payoff | \(\max(M_T - K, 0)\) |
| Fixed Put Payoff | \(\max(K - m_T, 0)\) |
| Delta (Float Call) | \(\approx 2\) at inception |
| Reflection Principle | \(\mathbb{P}(M_T^W \geq a) = 2\mathbb{P}(W_T \geq a)\) |
| Discrete Correction | \(-0.5826\sigma\sqrt{\Delta t}\) |

---

## 10. References

### Academic Papers

1. **Goldman, M.B., Sosin, H.B., and Gatto, M.A.** (1979). "Path Dependent Options: Buy at the Low, Sell at the High." *Journal of Finance*, 34(5), 1111-1127.

2. **Conze, A. and Viswanathan** (1991). "Path Dependent Options: The Case of Lookback Options." *Journal of Finance*, 46(5), 1893-1907.

3. **Broadie, M., Glasserman, P., and Kou, S.** (1997). "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance*, 7(4), 325-349.

4. **Kat, H.M.** (1995). "Pricing Lookback Options Using Binomial Trees: An Evaluation." *Journal of Financial Engineering*, 4(4), 375-397.

### Textbooks

5. **Hull, J.C.** (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson. Chapter 26.

6. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance* (2nd ed.). Wiley. Volume 2, Chapter 16.

7. **Shreve, S.E.** (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer. Chapter 7.

8. **Karatzas, I. and Shreve, S.** (1991). *Brownian Motion and Stochastic Calculus*. Springer. Chapter 2 (Reflection Principle).

---

## Appendix A: Derivation of Reflection Principle

### A.1 Statement

For standard Brownian motion \(W_t\) starting at 0:

\[
\mathbb{P}\left(\max_{0 \leq s \leq t} W_s \geq a, W_t \leq b\right) = \mathbb{P}(W_t \geq 2a - b)
\]

for \(a > 0\) and \(b \leq a\).

### A.2 Proof Sketch

1. **Hitting Time**: Let \(\tau_a = \inf\{t : W_t = a\}\) be the first hitting time of level \(a\).

2. **Strong Markov Property**: Given \(\tau_a < t\), the process \(\tilde{W}_s = W_{\tau_a + s} - a\) is a new Brownian motion.

3. **Reflection**: For paths that hit \(a\), reflect the path after \(\tau_a\). The reflected path ends at \(2a - W_t\) instead of \(W_t\).

4. **Bijection**: There's a one-to-one correspondence between:
   - Paths with \(\max_{s \leq t} W_s \geq a\) and \(W_t \leq b\)
   - Paths with \(W_t \geq 2a - b\)

5. **Result**: Therefore the probabilities are equal.

### A.3 Corollary

Setting \(b = -\infty\):
\[
\mathbb{P}\left(\max_{0 \leq s \leq t} W_s \geq a\right) = 2\mathbb{P}(W_t \geq a) = 2\Phi\left(-\frac{a}{\sqrt{t}}\right)
\]

---

## Appendix B: Proof that Lookback ≥ Vanilla

### B.1 Fixed Strike Call

**Claim**: \(C_{\text{lookback}}(K) \geq C_{\text{vanilla}}(K)\)

**Proof**:

For any path \(\omega\):
\[
M_T(\omega) = \max_{0 \leq t \leq T} S_t(\omega) \geq S_T(\omega)
\]

Therefore:
\[
\max(M_T(\omega) - K, 0) \geq \max(S_T(\omega) - K, 0)
\]

Taking expectations:
\[
\mathbb{E}^{\mathbb{Q}}[\max(M_T - K, 0)] \geq \mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]
\]

Discounting:
\[
C_{\text{lookback}} = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\max(M_T - K, 0)] \geq e^{-rT}\mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)] = C_{\text{vanilla}}
\]

### B.2 Floating Strike Call

**Claim**: \(C_{\text{float}} \geq C_{\text{vanilla}}(S_0)\) (comparing to ATM vanilla)

**Proof**:

\[
C_{\text{float}} = e^{-rT}\mathbb{E}^{\mathbb{Q}}[S_T - m_T] \geq e^{-rT}\mathbb{E}^{\mathbb{Q}}[(S_T - S_0)^+] \geq e^{-rT}\mathbb{E}^{\mathbb{Q}}[(S_T - S_0)^+] = C_{\text{vanilla}}(S_0)
\]

The first inequality holds because \(m_T \leq S_0\), and the payoff \(S_T - m_T\) is never negative.

---

*Document Version: 2.0*  
*Last Updated: January 27, 2026*  
*Author: QuantStrata Library*
