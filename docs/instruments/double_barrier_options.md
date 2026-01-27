# Double Barrier Options: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Product Class:** Path-Dependent Exotic Option  
**Pricing Methods:** Monte Carlo (discrete), Analytic (continuous - Ikeda-Kunitomo)  
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

A **double barrier option** is a path-dependent derivative that has both an **upper barrier** \(U\) and a **lower barrier** \(L\). The option's existence or activation depends on whether the underlying asset price stays within or exits this corridor \([L, U]\).

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Path Dependency** | Payoff depends on whether path exits corridor |
| **Two Barriers** | Upper (U) and Lower (L) barriers monitored simultaneously |
| **Corridor Constraint** | Must satisfy \(L < S_0 < U\) |
| **Cheaper than Single** | More restrictive, so lower premium |
| **Primary Use Cases** | Range-bound strategies, structured products |

### 1.3 Variants Summary

| Variant | Condition for Vanilla Payoff | Alternative |
|---------|------------------------------|-------------|
| **Knock-Out** | Path stays entirely in \([L, U]\) | Rebate |
| **Knock-In** | Path exits corridor (touches L or U) | Rebate |

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

### 2.3 Hitting Times

Define the first exit time from the corridor:

\[
\tau = \inf\{t \geq 0 : S_t \notin (L, U)\} = \inf\{t \geq 0 : S_t \leq L \text{ or } S_t \geq U\}
\]

**Components:**
- Lower barrier hitting time: \(\tau_L = \inf\{t \geq 0 : S_t \leq L\}\)
- Upper barrier hitting time: \(\tau_U = \inf\{t \geq 0 : S_t \geq U\}\)
- Exit time: \(\tau = \min(\tau_L, \tau_U)\)

### 2.4 Log-Price Transformation

Let \(X_t = \log(S_t)\). Then:
\[
dX_t = \mu \, dt + \sigma \, dW_t^{\mathbb{Q}}
\]

where \(\mu = r - q - \frac{\sigma^2}{2}\).

**Transformed barriers:**
- Lower: \(\ell = \log(L)\)
- Upper: \(u = \log(U)\)

The corridor condition becomes:
\[
\tau = \inf\{t \geq 0 : X_t \notin (\ell, u)\}
\]

### 2.5 Joint Distribution of First Passage

For Brownian motion with drift \(\mu\), the probability of staying within \((\ell, u)\) up to time \(T\) involves an infinite series:

\[
\mathbb{P}(\tau > T) = \sum_{n=-\infty}^{\infty} \left[e^{2\mu n(u-\ell)/\sigma^2} \Phi(d_{2n}^+) - e^{2\mu(x_0 - \ell + n(u-\ell))/\sigma^2} \Phi(d_{2n}^-)\right]
\]

where \(x_0 = \log(S_0)\) and \(d_{2n}^\pm\) are suitably defined.

---

## 3. Product Specification

### 3.1 Knock-Out Double Barrier

**Description:** The option survives (pays vanilla payoff) only if the spot price **never** exits the corridor during the option's life.

#### Call Payoff

\[
\text{Payoff}_{\text{KO Call}} = \begin{cases}
\max(S_T - K, 0) & \text{if } L < S_t < U \text{ for all } t \in [0, T] \\
R & \text{otherwise (barrier hit)}
\end{cases}
\]

**Mathematical Expression:**
\[
\text{Payoff} = \mathbf{1}_{\{\tau > T\}} \cdot (S_T - K)^+ + \mathbf{1}_{\{\tau \leq T\}} \cdot R
\]

#### Put Payoff

\[
\text{Payoff}_{\text{KO Put}} = \begin{cases}
\max(K - S_T, 0) & \text{if } L < S_t < U \text{ for all } t \in [0, T] \\
R & \text{otherwise}
\end{cases}
\]

### 3.2 Knock-In Double Barrier

**Description:** The option activates (pays vanilla payoff) only if the spot price **exits** the corridor at some point.

#### Call Payoff

\[
\text{Payoff}_{\text{KI Call}} = \begin{cases}
\max(S_T - K, 0) & \text{if } S_t \leq L \text{ or } S_t \geq U \text{ for some } t \in [0, T] \\
R & \text{otherwise (never exited)}
\end{cases}
\]

**Mathematical Expression:**
\[
\text{Payoff} = \mathbf{1}_{\{\tau \leq T\}} \cdot (S_T - K)^+ + \mathbf{1}_{\{\tau > T\}} \cdot R
\]

#### Put Payoff

\[
\text{Payoff}_{\text{KI Put}} = \begin{cases}
\max(K - S_T, 0) & \text{if } S_t \leq L \text{ or } S_t \geq U \text{ for some } t \\
R & \text{otherwise}
\end{cases}
\]

### 3.3 Corridor Width and Constraint

**Critical Constraint:** The initial spot must be inside the corridor:
\[
L < S_0 < U
\]

**Corridor Width:**
\[
W = U - L = S_0 \cdot (u_{\%} - \ell_{\%})
\]

where \(u_{\%} = U/S_0 - 1\) and \(\ell_{\%} = 1 - L/S_0\) are the percentage distances.

---

## 4. Pricing Theory

### 4.1 Risk-Neutral Valuation

**Knock-Out Double Barrier Call:**
\[
V_{\text{KO}} = e^{-rT} \mathbb{E}^{\mathbb{Q}}\left[\mathbf{1}_{\{\tau > T\}} (S_T - K)^+\right] + e^{-rT} R \cdot \mathbb{P}^{\mathbb{Q}}(\tau \leq T)
\]

**Knock-In Double Barrier Call:**
\[
V_{\text{KI}} = e^{-rT} \mathbb{E}^{\mathbb{Q}}\left[\mathbf{1}_{\{\tau \leq T\}} (S_T - K)^+\right] + e^{-rT} R \cdot \mathbb{P}^{\mathbb{Q}}(\tau > T)
\]

### 4.2 In-Out Parity

**Theorem (Double Barrier In-Out Parity):**

For double barrier options with the same barriers, strike, and zero rebate:
\[
V_{\text{KO}} + V_{\text{KI}} = V_{\text{Vanilla}}
\]

**Proof:**
\[
\mathbf{1}_{\{\tau > T\}} + \mathbf{1}_{\{\tau \leq T\}} = 1
\]

Therefore:
\[
V_{\text{KO}} + V_{\text{KI}} = e^{-rT} \mathbb{E}^{\mathbb{Q}}\left[(S_T - K)^+\right] = V_{\text{Vanilla}}
\]

### 4.3 Ikeda-Kunitomo Formula (Continuous Monitoring)

For continuous monitoring, the knock-out double barrier call price is given by an infinite series involving images:

\[
V_{\text{KO}} = \sum_{n=-\infty}^{\infty} \left[V_n^{(1)} - V_n^{(2)}\right]
\]

where each term involves Black-Scholes type functions with modified arguments reflecting multiple reflections off the barriers.

**Practical Truncation:** The series converges rapidly; typically \(|n| \leq 5\) is sufficient for accuracy to 6 decimal places.

### 4.4 Relationship to Single Barriers

Double barrier options can be viewed as:
\[
V_{\text{DKO}} \approx V_{\text{Down-and-Out}} \cdot \text{(survival probability for upper)} + \text{correction}
\]

However, the interaction between barriers makes this approximation imprecise.

---

## 5. Greeks and Sensitivities

### 5.1 Delta

**General Form:**
\[
\Delta = \frac{\partial V}{\partial S}
\]

**Key Properties:**
- Delta is **discontinuous** near barriers
- Near lower barrier: Delta approaches a step function
- Near upper barrier: Delta approaches a step function
- Inside corridor: Smooth but can be complex

### 5.2 Gamma

**Near-Barrier Behavior:**
\[
\Gamma \rightarrow \pm\infty \text{ as } S \rightarrow L^+ \text{ or } S \rightarrow U^-
\]

This creates significant **gamma risk** for dealers near barriers.

### 5.3 Vega

**Key Insight:** Vega can be **negative** for knock-out double barriers:
- Higher vol increases probability of hitting either barrier
- For knock-out: higher vol reduces expected survival → lower price
- For knock-in: higher vol increases activation probability → higher price

\[
\text{Vega}_{\text{KO}} < 0 \quad \text{(typically for near-barrier spots)}
\]

### 5.4 Theta

Time decay has two components:
1. Standard time decay (like vanilla)
2. Survival probability component (unique to barriers)

For knock-out:
\[
\Theta = \frac{\partial V}{\partial t} < 0 \quad \text{(accelerated near expiry)}
\]

### 5.5 Greeks Summary Table

| Greek | Knock-Out Behavior | Knock-In Behavior |
|-------|-------------------|-------------------|
| **Delta** | Discontinuous at barriers | Discontinuous at barriers |
| **Gamma** | Explodes near barriers | Explodes near barriers |
| **Vega** | Often negative | Often positive |
| **Theta** | Accelerated decay | Complex pattern |
| **Rho** | Smaller than vanilla | Smaller than vanilla |

---

## 6. Numerical Methods

### 6.1 Monte Carlo Simulation

**Algorithm:**

```
1. Generate N paths of S_t under risk-neutral measure
2. For each path i:
   a. Check if min(S_t) <= L or max(S_t) >= U
   b. If barrier hit (knock-out): payoff_i = R
   c. If survived (knock-out): payoff_i = max(S_T - K, 0)
3. Price = e^(-rT) * mean(payoffs)
```

**Variance Reduction:**
- Antithetic variates
- Control variates (use vanilla as control)
- Importance sampling (bias towards barrier regions)

### 6.2 Discrete vs Continuous Monitoring

**Discrete Monitoring Correction (Broadie-Glasserman-Kou):**

For barriers monitored at discrete intervals \(\Delta t\):
\[
L_{\text{effective}} = L \cdot e^{-\beta \sigma \sqrt{\Delta t}}
\]
\[
U_{\text{effective}} = U \cdot e^{\beta \sigma \sqrt{\Delta t}}
\]

where \(\beta \approx 0.5826\) is a correction constant.

### 6.3 Finite Difference Methods

**PDE for Double Barrier:**
\[
\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0
\]

**Boundary Conditions (Knock-Out):**
- At \(S = L\): \(V(L, t) = R \cdot e^{-r(T-t)}\) (discounted rebate)
- At \(S = U\): \(V(U, t) = R \cdot e^{-r(T-t)}\) (discounted rebate)

**Boundary Conditions (Knock-In):**
- More complex; typically solve as Vanilla minus Knock-Out

### 6.4 Grid Considerations

- Use **non-uniform grids** with concentration near barriers
- **Boundary absorption:** Implement absorbing boundaries at L and U
- **Stability:** Use implicit schemes (Crank-Nicolson) for stability

---

## 7. Risk Management

### 7.1 Hedging Challenges

**Delta Hedging Issues:**
- Delta changes rapidly near barriers
- Transaction costs can be significant
- Gap risk if spot jumps across barrier

**Gamma Hedging:**
- Requires vanilla options to neutralize
- Position becomes large near barriers
- Cost of gamma hedging increases near barriers

### 7.2 Gap Risk

If the market gaps through a barrier (e.g., overnight, after news):
- Discrete monitoring may miss the breach
- P&L impact can be significant
- Hedging becomes imperfect

**Mitigation:**
- Use wider corridor
- Include rebates
- Add gap event clauses in contracts

### 7.3 Model Risk

**Volatility Assumptions:**
- Constant vol may misprice barrier probabilities
- Stochastic vol changes barrier hitting probabilities
- Local vol captures smile but adds complexity

**Correlation with Barriers:**
- Spot-vol correlation affects barrier hitting probabilities
- Negative correlation increases down-barrier hits
- Positive correlation increases up-barrier hits

### 7.4 Vega Hedging

Since knock-out options often have **negative vega**:
- Selling vanilla options increases vega exposure
- Need to monitor vega-gamma interaction
- Vega can flip sign as spot moves

---

## 8. Implementation

### 8.1 Pseudocode: Monte Carlo Pricer

```python
def price_double_barrier_mc(
    S0, K, T, r, q, sigma,
    lower_barrier, upper_barrier,
    barrier_style,  # "knock_out" or "knock_in"
    option_type,    # "call" or "put"
    rebate=0.0,
    n_paths=200000,
    n_steps=252
):
    """Monte Carlo pricer for double barrier options."""
    
    # Validate: S0 must be inside corridor
    assert lower_barrier < S0 < upper_barrier
    
    # Generate GBM paths
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((n_paths, n_steps))
    log_returns = drift + diffusion * Z
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.column_stack([np.full(n_paths, S0), paths])
    
    # Check barrier breaches
    min_spots = np.min(paths, axis=1)
    max_spots = np.max(paths, axis=1)
    
    hit_lower = min_spots <= lower_barrier
    hit_upper = max_spots >= upper_barrier
    exited_corridor = hit_lower | hit_upper
    
    # Compute terminal payoffs
    terminal = paths[:, -1]
    if option_type == "call":
        vanilla_payoff = np.maximum(terminal - K, 0)
    else:
        vanilla_payoff = np.maximum(K - terminal, 0)
    
    # Apply barrier logic
    if barrier_style == "knock_out":
        # Pays vanilla if stayed in corridor
        payoffs = np.where(~exited_corridor, vanilla_payoff, rebate)
    else:  # knock_in
        # Pays vanilla if exited corridor
        payoffs = np.where(exited_corridor, vanilla_payoff, rebate)
    
    # Discount and return
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error
```

### 8.2 Implementation Considerations

**Numerical Precision:**
- Use log-space for path simulation
- Careful handling of boundary conditions
- Monitor for numerical instabilities near barriers

**Performance:**
- Vectorize operations over paths
- Consider GPU acceleration for large simulations
- Cache barrier checking results

---

## 9. Key Interview Points

### 9.1 Must-Know Facts

1. **Definition:** Double barrier has BOTH upper and lower barriers
2. **Constraint:** Initial spot must be inside corridor: \(L < S_0 < U\)
3. **In-Out Parity:** \(V_{\text{KO}} + V_{\text{KI}} = V_{\text{Vanilla}}\)
4. **Cheaper:** Double barriers are cheaper than single barriers (more restrictive)
5. **Greeks:** Delta discontinuous, Gamma explodes, Vega often negative (KO)

### 9.2 Common Interview Questions

**Q1: Why are double barrier options cheaper than single barriers?**

A: They are more restrictive. The option has to stay within a narrower range. Higher probability of knock-out (for KO) means lower expected payoff.

**Q2: What happens to Greeks near the barriers?**

A: Delta becomes discontinuous, Gamma explodes to infinity. This creates significant hedging challenges as the spot approaches either barrier.

**Q3: Can Vega be negative for a knock-out double barrier?**

A: Yes! Higher volatility increases the probability of hitting either barrier, which reduces the expected payoff for a knock-out option.

**Q4: How does discrete vs continuous monitoring affect pricing?**

A: Discrete monitoring is less likely to catch barrier breaches, so:
- Knock-out options are MORE expensive with discrete monitoring
- Knock-in options are LESS expensive with discrete monitoring
The Broadie-Glasserman-Kou correction adjusts effective barrier levels.

### 9.3 Key Formulas to Remember

| Formula | Expression |
|---------|------------|
| **KO Call Payoff** | \(\mathbf{1}_{\{\tau > T\}} (S_T - K)^+ + \mathbf{1}_{\{\tau \leq T\}} R\) |
| **In-Out Parity** | \(V_{\text{KO}} + V_{\text{KI}} = V_{\text{Vanilla}}\) |
| **Exit Time** | \(\tau = \inf\{t : S_t \leq L \text{ or } S_t \geq U\}\) |
| **BGK Correction** | \(L_{\text{eff}} = L e^{-\beta\sigma\sqrt{\Delta t}}\), \(\beta \approx 0.5826\) |

### 9.4 Practical Considerations

1. **Corridor Width:** Wider corridor → higher price (more likely to survive)
2. **Barrier Positioning:** Symmetric around S0 vs asymmetric affects Greeks
3. **Strike Position:** Strike relative to barriers affects payoff distribution
4. **Rebate:** Non-zero rebate provides partial recovery on knock-out

---

## 10. References

### Academic Papers

1. **Ikeda, N. and Kunitomo, N.** (1992). "Pricing Options with Curved Boundaries." Mathematical Finance.
2. **Kunitomo, N. and Ikeda, M.** (1992). "Pricing Options with Double Barriers." Asia-Pacific Financial Markets.
3. **Broadie, M., Glasserman, P., and Kou, S.** (1997). "A Continuity Correction for Discrete Barrier Options." Mathematical Finance.
4. **Geman, H. and Yor, M.** (1996). "Pricing and Hedging Double-Barrier Options: A Probabilistic Approach." Mathematical Finance.

### Textbooks

1. **Hull, J.** (2022). *Options, Futures, and Other Derivatives*, 11th ed. Chapters 26-27.
2. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed. Chapter 12.
3. **Shreve, S.** (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Chapter 7.

### Industry Resources

1. Bloomberg: Double Barrier Option Pricing (DBLB)
2. Reuters: Exotic Options Analytics
3. QuantLib: DoubleBarrierOption implementation

---

## Appendix A: Derivation of Exit Probability

### A.1 Exit Probability via Reflection

The probability of Brownian motion with drift staying within \((\ell, u)\) involves the **method of images**.

For standard Brownian motion starting at \(x_0 \in (\ell, u)\):

\[
\mathbb{P}(\tau > T) = \sum_{n=-\infty}^{\infty} \left[N(d_n^+) - N(d_n^-)\right]
\]

where:
\[
d_n^+ = \frac{2n(u-\ell) + u - x_0}{\sigma\sqrt{T}}
\]
\[
d_n^- = \frac{2n(u-\ell) - u + x_0 + 2\ell}{\sigma\sqrt{T}}
\]

### A.2 Convergence of the Series

The series converges rapidly because:
1. Each reflection adds a factor of \(e^{-c \cdot n^2}\) for some \(c > 0\)
2. Terms with \(|n| > 5\) typically contribute less than \(10^{-10}\)

---

## Appendix B: Relationship Between Single and Double Barriers

### B.1 Decomposition (Approximation)

A double knock-out can be approximated as:

\[
V_{\text{DKO}} \approx V_{\text{Down-and-Out}} - V_{\text{Double Touch Correction}}
\]

However, this decomposition is imprecise because the barriers interact.

### B.2 Exact Relationship

The exact relationship requires the full Ikeda-Kunitomo series, accounting for multiple reflections between the two barriers.

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Author: QuantStrata Development Team*
