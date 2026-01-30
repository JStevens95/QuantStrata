# Touch Options: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Product Class:** Path-Dependent Binary/Digital Exotic Option  
**Pricing Methods:** Monte Carlo (discrete), Analytic (continuous)  
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

A **touch option** (also called a **binary barrier option**) is a path-dependent derivative that pays a fixed amount based solely on whether the underlying asset price **touches** or **avoids** a specified barrier level during the option's life. Unlike standard barrier options, touch options do not have a strike price—the payoff is purely binary.

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Path Dependency** | Payoff depends on barrier touch/no-touch |
| **Binary Payout** | All-or-nothing: pays Q or 0 |
| **No Strike** | Payoff independent of terminal spot level |
| **Simple Structure** | Easy to understand payoff |
| **Primary Use Cases** | Range bets, hedging extreme moves |

### 1.3 Variants Summary

| Variant | Direction | Pays Q If | Pays 0 If |
|---------|-----------|-----------|-----------|
| **One-Touch Up** | Up | \(\max(S_t) \geq H\) | \(\max(S_t) < H\) |
| **One-Touch Down** | Down | \(\min(S_t) \leq H\) | \(\min(S_t) > H\) |
| **No-Touch Up** | Up | \(\max(S_t) < H\) | \(\max(S_t) \geq H\) |
| **No-Touch Down** | Down | \(\min(S_t) > H\) | \(\min(S_t) \leq H\) |

Where:
- \(H\) = barrier level
- \(Q\) = fixed payout amount

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

**Up-Touch:**
\[
\tau_H^{\text{up}} = \inf\{t \geq 0 : S_t \geq H\} = \inf\{t \geq 0 : \max_{0 \leq s \leq t} S_s \geq H\}
\]

**Down-Touch:**
\[
\tau_H^{\text{down}} = \inf\{t \geq 0 : S_t \leq H\} = \inf\{t \geq 0 : \min_{0 \leq s \leq t} S_s \leq H\}
\]

### 2.4 Reflection Principle Application

The probability of touching an up-barrier \(H > S_0\) before time \(T\) is related to the first passage time of Brownian motion.

**Theorem (First Passage Time):**

For Brownian motion with drift \(\mu\):
\[
\mathbb{P}(\tau_a \leq T) = \Phi\left(\frac{-a + \mu T}{\sigma\sqrt{T}}\right) + e^{2\mu a/\sigma^2} \Phi\left(\frac{-a - \mu T}{\sigma\sqrt{T}}\right)
\]

where \(a = \log(H/S_0)\) for an up-barrier.

### 2.5 Log-Price Transformation

Let \(X_t = \log(S_t/S_0)\). Then:
\[
X_t = \mu t + \sigma W_t
\]

where \(\mu = r - q - \frac{\sigma^2}{2}\).

**Running Maximum/Minimum:**
\[
\bar{X}_T = \max_{0 \leq t \leq T} X_t, \quad \underline{X}_T = \min_{0 \leq t \leq T} X_t
\]

**Touch Conditions:**
- Up-touch at \(H\): \(\bar{X}_T \geq \log(H/S_0)\)
- Down-touch at \(H\): \(\underline{X}_T \leq \log(H/S_0)\)

---

## 3. Product Specification

### 3.1 One-Touch Options

#### One-Touch Up

**Payoff:**
\[
\text{Payoff}_{\text{1T-Up}} = Q \cdot \mathbf{1}_{\{\max_{0 \leq t \leq T} S_t \geq H\}}
\]

**Interpretation:** Pays fixed amount \(Q\) if the spot price ever reaches or exceeds \(H\) during the option's life.

**Barrier Position:** \(H > S_0\) (barrier above current spot)

#### One-Touch Down

**Payoff:**
\[
\text{Payoff}_{\text{1T-Down}} = Q \cdot \mathbf{1}_{\{\min_{0 \leq t \leq T} S_t \leq H\}}
\]

**Interpretation:** Pays fixed amount \(Q\) if the spot price ever reaches or falls below \(H\).

**Barrier Position:** \(H < S_0\) (barrier below current spot)

### 3.2 No-Touch Options

#### No-Touch Up

**Payoff:**
\[
\text{Payoff}_{\text{NT-Up}} = Q \cdot \mathbf{1}_{\{\max_{0 \leq t \leq T} S_t < H\}}
\]

**Interpretation:** Pays fixed amount \(Q\) if the spot price **never** reaches \(H\) during the option's life.

#### No-Touch Down

**Payoff:**
\[
\text{Payoff}_{\text{NT-Down}} = Q \cdot \mathbf{1}_{\{\min_{0 \leq t \leq T} S_t > H\}}
\]

**Interpretation:** Pays fixed amount \(Q\) if the spot price **never** falls to \(H\).

### 3.3 Touch-NoTouch Parity

**Theorem (Touch Parity):**

For the same barrier level and payout amount:
\[
V_{\text{One-Touch}} + V_{\text{No-Touch}} = Q \cdot e^{-rT}
\]

**Proof:**
\[
\mathbf{1}_{\{\text{touched}\}} + \mathbf{1}_{\{\text{not touched}\}} = 1
\]

Therefore:
\[
V_{\text{1T}} + V_{\text{NT}} = e^{-rT} \mathbb{E}^{\mathbb{Q}}[Q \cdot 1] = Q \cdot e^{-rT}
\]

---

## 4. Pricing Theory

### 4.1 Risk-Neutral Valuation

**One-Touch Up:**
\[
V_{\text{1T-Up}} = Q \cdot e^{-rT} \cdot \mathbb{P}^{\mathbb{Q}}\left(\max_{0 \leq t \leq T} S_t \geq H\right)
\]

**No-Touch Up:**
\[
V_{\text{NT-Up}} = Q \cdot e^{-rT} \cdot \mathbb{P}^{\mathbb{Q}}\left(\max_{0 \leq t \leq T} S_t < H\right)
\]

### 4.2 Closed-Form Formulas (Continuous Monitoring)

**One-Touch Up (H > S_0):**

\[
V_{\text{1T-Up}} = Q \cdot e^{-rT} \left[\Phi(d_1) + \left(\frac{H}{S_0}\right)^{2\lambda} \Phi(d_2)\right]
\]

where:
\[
\lambda = \frac{r - q - \frac{\sigma^2}{2}}{\sigma^2}
\]
\[
d_1 = \frac{\log(H/S_0) - (r-q-\frac{\sigma^2}{2})T}{\sigma\sqrt{T}}
\]
\[
d_2 = \frac{-\log(H/S_0) - (r-q-\frac{\sigma^2}{2})T}{\sigma\sqrt{T}}
\]

**One-Touch Down (H < S_0):**

\[
V_{\text{1T-Down}} = Q \cdot e^{-rT} \left[\Phi(-d_3) + \left(\frac{H}{S_0}\right)^{2\lambda} \Phi(-d_4)\right]
\]

with appropriately defined \(d_3, d_4\).

### 4.3 Probability of Touch

**Up-Touch Probability:**
\[
p_{\text{up}} = \mathbb{P}^{\mathbb{Q}}\left(\max_{0 \leq t \leq T} S_t \geq H\right) = \Phi(d_1) + \left(\frac{H}{S_0}\right)^{2\lambda} \Phi(d_2)
\]

**Key Insight:** The touch probability has two terms:
1. Direct paths reaching \(H\) and staying above
2. Reflected paths (via reflection principle)

### 4.4 Sensitivity to Parameters

| Parameter | Effect on One-Touch Up | Effect on No-Touch Up |
|-----------|----------------------|----------------------|
| **Volatility ↑** | Price ↑ (more likely to touch) | Price ↓ |
| **Time ↑** | Price ↑ (more time to touch) | Price ↓ |
| **Barrier closer** | Price ↑ (easier to touch) | Price ↓ |
| **Drift toward barrier** | Price ↑ | Price ↓ |

---

## 5. Greeks and Sensitivities

### 5.1 Delta

**One-Touch Up Delta:**
\[
\Delta_{\text{1T-Up}} = \frac{\partial V}{\partial S} = Q \cdot e^{-rT} \cdot \frac{\partial p_{\text{up}}}{\partial S}
\]

**Key Properties:**
- Delta is positive for one-touch up (higher S → closer to barrier)
- Delta is negative for one-touch down
- Delta changes rapidly as S approaches barrier
- **Peak Delta** occurs when S is near but below the barrier

### 5.2 Gamma

\[
\Gamma = \frac{\partial^2 V}{\partial S^2}
\]

**Near-Barrier Behavior:**
- Gamma is very large near the barrier
- For one-touch: Gamma peaks just below barrier, then drops
- Creates significant hedging challenges

### 5.3 Vega

\[
\text{Vega} = \frac{\partial V}{\partial \sigma}
\]

**One-Touch:**
- Vega is **positive**: higher vol increases touch probability
- Effect is strongest when barrier is moderately far from spot

**No-Touch:**
- Vega is **negative**: higher vol increases chance of touching
- This is opposite to vanilla options!

### 5.4 Theta

\[
\Theta = \frac{\partial V}{\partial t}
\]

**One-Touch:**
- Theta can be **positive** (unusual): more time = more chance to touch
- This is opposite to most options!

**No-Touch:**
- Theta is **negative**: less time remaining = more likely to survive

### 5.5 Greeks Summary Table

| Greek | One-Touch Up | No-Touch Up |
|-------|-------------|-------------|
| **Delta** | Positive, peaks near barrier | Negative |
| **Gamma** | Large positive near barrier | Large negative near barrier |
| **Vega** | Positive | Negative |
| **Theta** | Often positive | Negative |

---

## 6. Numerical Methods

### 6.1 Monte Carlo Simulation

**Algorithm:**

```
1. Generate N paths of S_t under risk-neutral measure
2. For each path i:
   - Up touch: check if max(path) >= H
   - Down touch: check if min(path) <= H
3. One-Touch price = Q * e^(-rT) * (count_touched / N)
4. No-Touch price = Q * e^(-rT) * (count_not_touched / N)
```

**Variance Reduction:**
- Antithetic variates
- Importance sampling (shift mean toward barrier)
- Conditional Monte Carlo

### 6.2 Discrete vs Continuous Monitoring

**Discrete Monitoring Effect:**
- One-touch: Discrete monitoring **underprices** (may miss the touch)
- No-touch: Discrete monitoring **overprices** (less likely to catch touch)

**Broadie-Glasserman-Kou Correction:**
\[
H_{\text{eff}} = H \cdot e^{\pm \beta \sigma \sqrt{\Delta t}}
\]

where \(\beta \approx 0.5826\):
- For up-touch: use \(H_{\text{eff}} = H \cdot e^{-\beta \sigma \sqrt{\Delta t}}\) (lower effective barrier)
- For down-touch: use \(H_{\text{eff}} = H \cdot e^{+\beta \sigma \sqrt{\Delta t}}\) (higher effective barrier)

### 6.3 Finite Difference Methods

**PDE Approach:**

Touch options can be priced as the limit of barrier options:
\[
V_{\text{1T-Up}} = \lim_{R \to Q} V_{\text{Up-and-In Digital}}
\]

**Boundary Conditions:**
- At barrier \(S = H\): \(V = Q \cdot e^{-r(T-t)}\) for one-touch
- At barrier \(S = H\): \(V = 0\) for no-touch

---

## 7. Risk Management

### 7.1 Hedging Challenges

**Delta Hedging:**
- Near the barrier, delta changes very rapidly
- Requires frequent rebalancing
- Transaction costs can erode hedging effectiveness

**Gamma Risk:**
- Very large gamma near barrier creates P&L swings
- Difficult to hedge with vanilla options alone
- May need other barrier products

### 7.2 Gap Risk

**Definition:** Risk that the spot "gaps" through the barrier (e.g., overnight, after news).

**Impact:**
- For one-touch: may trigger payout unexpectedly
- For no-touch: may knock out unexpectedly
- Discrete monitoring doesn't capture the gap

**Mitigation:**
- Wider barriers
- Gap event clauses
- Stochastic gap modeling

### 7.3 Liquidity Considerations

**Market Making:**
- Touch options are less liquid than vanillas
- Bid-ask spreads can be wide
- Hedging requires sophisticated infrastructure

### 7.4 Model Risk

**Volatility Model:**
- Constant vol may misprice touch probabilities
- Local vol captures smile effects
- Stochastic vol changes barrier hitting dynamics

**Jump Risk:**
- GBM assumes continuous paths
- Jumps significantly affect touch probabilities
- Jump-diffusion models may be more appropriate

---

## 8. Implementation

### 8.1 Pseudocode: Monte Carlo Pricer

```python
def price_touch_mc(
    S0, T, r, q, sigma,
    barrier_level, barrier_direction,  # "up" or "down"
    touch_style,  # "one_touch" or "no_touch"
    payout_amount,
    n_paths=200000,
    n_steps=252
):
    """Monte Carlo pricer for touch options."""
    
    # Generate GBM paths
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((n_paths, n_steps))
    log_returns = drift + diffusion * Z
    paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.column_stack([np.full(n_paths, S0), paths])
    
    # Check touch status
    if barrier_direction == "up":
        max_spots = np.max(paths, axis=1)
        touched = max_spots >= barrier_level
    else:  # down
        min_spots = np.min(paths, axis=1)
        touched = min_spots <= barrier_level
    
    # Compute payoffs
    if touch_style == "one_touch":
        payoffs = np.where(touched, payout_amount, 0.0)
    else:  # no_touch
        payoffs = np.where(~touched, payout_amount, 0.0)
    
    # Discount and return
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error
```

### 8.2 Pseudocode: Analytic Pricing

```python
def price_one_touch_up_analytic(S0, H, T, r, q, sigma, Q):
    """Analytic price for one-touch up option (continuous monitoring)."""
    
    if S0 >= H:
        # Already touched
        return Q * np.exp(-r * T)
    
    mu = r - q - 0.5 * sigma**2
    lambda_ = mu / sigma**2
    
    a = np.log(H / S0)
    sqrt_T = np.sqrt(T)
    
    d1 = (a - mu * T) / (sigma * sqrt_T)
    d2 = (-a - mu * T) / (sigma * sqrt_T)
    
    touch_prob = norm.cdf(-d1) + (H / S0)**(2 * lambda_) * norm.cdf(d2)
    
    return Q * np.exp(-r * T) * touch_prob
```

### 8.3 Implementation Considerations

**Edge Cases:**
- \(S_0 = H\): Touch has already occurred (one-touch pays immediately)
- \(T = 0\): No time for touch (no-touch pays, one-touch doesn't)
- \(\sigma = 0\): Deterministic path (easy to determine touch)

**Numerical Stability:**
- Use log-space calculations
- Handle extreme barrier levels carefully
- Monitor for overflow in exponential terms

---

## 9. Key Interview Points

### 9.1 Must-Know Facts

1. **Definition:** Binary payout based on whether barrier is touched
2. **No Strike:** Payoff doesn't depend on terminal spot value
3. **Touch Parity:** \(V_{\text{1T}} + V_{\text{NT}} = Q \cdot e^{-rT}\)
4. **Vega Sign:** One-touch has positive vega, no-touch has negative vega
5. **Theta Sign:** One-touch often has positive theta (unusual!)

### 9.2 Common Interview Questions

**Q1: How does a touch option differ from a digital/binary option?**

A: A digital option pays based on whether \(S_T > K\) (terminal condition). A touch option pays based on whether \(\max S_t \geq H\) or \(\min S_t \leq H\) (path condition). Digital is path-independent; touch is path-dependent.

**Q2: Why does a one-touch option often have positive theta?**

A: More time remaining means more opportunity for the spot to touch the barrier. Unlike standard options where time decay works against the holder, one-touch benefits from more time. No-touch has negative theta for the opposite reason.

**Q3: Why is vega negative for no-touch options?**

A: Higher volatility increases the probability of the spot reaching the barrier, which means higher probability of NOT surviving (for no-touch). This reduces the expected payout.

**Q4: How would you hedge a short one-touch position?**

A: 
1. Delta hedge by buying/selling spot
2. Challenge: delta changes rapidly near barrier
3. May use other barrier options (knockouts) for gamma offset
4. Accept that perfect hedging is impossible (gap risk)

### 9.3 Key Formulas to Remember

| Formula | Expression |
|---------|------------|
| **One-Touch Payoff** | \(Q \cdot \mathbf{1}_{\{\text{touched}\}}\) |
| **No-Touch Payoff** | \(Q \cdot \mathbf{1}_{\{\text{not touched}\}}\) |
| **Touch Parity** | \(V_{\text{1T}} + V_{\text{NT}} = Q \cdot e^{-rT}\) |
| **Touch Probability** | \(\Phi(d_1) + (H/S_0)^{2\lambda} \Phi(d_2)\) |
| **BGK Correction** | \(H_{\text{eff}} = H \cdot e^{\pm\beta\sigma\sqrt{\Delta t}}\) |

### 9.4 Practical Considerations

1. **Barrier Monitoring:** Discrete vs continuous affects price significantly
2. **Payout Timing:** Usually at expiry (deferred), not at touch
3. **Liquidity:** Less liquid than vanillas, wider bid-ask
4. **Gap Risk:** Major concern, especially for overnight gaps

---

## 10. References

### Academic Papers

1. **Reiner, E. and Rubinstein, M.** (1991). "Breaking Down the Barriers." Risk Magazine.
2. **Broadie, M., Glasserman, P., and Kou, S.** (1997). "A Continuity Correction for Discrete Barrier Options."
3. **Carr, P.** (1995). "Two Extensions to Barrier Option Valuation." Applied Mathematical Finance.

### Textbooks

1. **Hull, J.** (2022). *Options, Futures, and Other Derivatives*, 11th ed. Chapter 26.
2. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed. Chapter 11.
3. **Taleb, N.** (1997). *Dynamic Hedging*. Chapter on binary options.

### Industry Resources

1. Bloomberg: Touch Option Pricing (TOCH)
2. FINCAD: Digital and Touch Option Analytics
3. QuantLib: TouchOption implementation

---

## Appendix A: Derivation of Touch Probability

### A.1 First Passage Time for Up-Barrier

For geometric Brownian motion, we transform to log-space:
\[
X_t = \log(S_t/S_0) = \mu t + \sigma W_t
\]

The hitting time for level \(a = \log(H/S_0)\):
\[
\tau_a = \inf\{t : X_t \geq a\}
\]

### A.2 Using Reflection Principle

The probability that Brownian motion with drift reaches \(a > 0\) by time \(T\):

\[
\mathbb{P}(\tau_a \leq T) = \mathbb{P}\left(\max_{0 \leq t \leq T} (\mu t + \sigma W_t) \geq a\right)
\]

Using the reflection principle and Girsanov theorem:

\[
\mathbb{P}(\tau_a \leq T) = \Phi\left(\frac{-a + \mu T}{\sigma\sqrt{T}}\right) + e^{2\mu a/\sigma^2} \Phi\left(\frac{-a - \mu T}{\sigma\sqrt{T}}\right)
\]

### A.3 Interpretation

The two terms represent:
1. Paths that reach \(a\) and stay above
2. Paths that reach \(a\), go above, and come back below (reflected paths)

---

## Appendix B: Comparison with Digital Options

| Feature | Digital Option | Touch Option |
|---------|---------------|--------------|
| **Condition** | \(S_T > K\) | \(\max S_t \geq H\) |
| **Path Dependency** | No | Yes |
| **Timing** | Only terminal | Throughout life |
| **Vega** | Can be negative near strike | Depends on style |
| **Theta** | Usually negative | Can be positive (one-touch) |
| **Hedging** | Delta spike at strike | Delta spike near barrier |

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Author: QuantStrata Development Team*
