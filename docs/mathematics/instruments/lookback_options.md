# Lookback Options: Mathematical Foundations

**Product Type:** Path-Dependent Exotic Option  
**Pricing Methods:** Monte Carlo (discrete monitoring), Analytic (continuous monitoring)  
**Key Feature:** "Perfect Hindsight" - captures optimal entry/exit timing

---

## 1. Product Definition

A **lookback option** pays based on the **maximum or minimum** price of the underlying asset over the option's life. This provides "perfect hindsight" - the holder always captures the optimal entry or exit point.

### Two Main Variants

**1. Floating Strike Lookback** (strike determined by path extremum):
- **Call**: Strike = min(S_t), Payoff = S_T - min(S_t)
- **Put**: Strike = max(S_t), Payoff = max(S_t) - S_T

**2. Fixed Strike Lookback** (payoff on path extremum vs fixed K):
- **Call**: Payoff = max(max(S_t) - K, 0)
- **Put**: Payoff = max(K - min(S_t), 0)

---

## 2. Payoff Structures

### 2.1 Floating Strike Lookback

#### Call (Buy at the Minimum)
```
Payoff = S_T - min(S_t)
       = S_T - m_T
```

Where m_T = min_{0 ≤ t ≤ T} S_t is the minimum spot over the path.

**Key Property**: This is **always ≥ 0** since S_T ≥ m_T by definition.

#### Put (Sell at the Maximum)
```
Payoff = max(S_t) - S_T
       = M_T - S_T
```

Where M_T = max_{0 ≤ t ≤ T} S_t is the maximum spot over the path.

**Key Property**: This is **always ≥ 0** since M_T ≥ S_T by definition.

### 2.2 Fixed Strike Lookback

#### Call (Option on Maximum)
```
Payoff = max(max(S_t) - K, 0)
       = max(M_T - K, 0)
```

This is equivalent to a vanilla call option where the underlying is the **running maximum**.

#### Put (Option on Minimum)
```
Payoff = max(K - min(S_t), 0)
       = max(K - m_T, 0)
```

This is equivalent to a vanilla put option where the underlying is the **running minimum**.

---

## 3. Key Properties

### 3.1 Floating Strike: Always In-The-Money

**Critical Property**: Floating strike lookbacks are **ALWAYS ITM**.

For floating strike call:
- Payoff = S_T - m_T ≥ 0 (since S_T ≥ m_T)
- Equality holds only when S_T = m_T (worst case)

For floating strike put:
- Payoff = M_T - S_T ≥ 0 (since M_T ≥ S_T)
- Equality holds only when S_T = M_T (worst case)

### 3.2 Lookback ≥ Vanilla

**Fundamental Inequality**: Lookback options are always worth at least as much as vanilla options.

For fixed strike lookback call vs vanilla call:
```
E[max(M_T - K, 0)] ≥ E[max(S_T - K, 0)]
```

**Proof**: Since M_T ≥ S_T, we have max(M_T - K, 0) ≥ max(S_T - K, 0) pathwise.

**Intuition**: Lookback captures the value of "perfect market timing".

### 3.3 Delta of Floating Strike Lookback

At inception (t=0), the delta of a floating strike lookback call is approximately **2**.

**Intuition**: The option is sensitive to:
1. The terminal spot S_T (delta ≈ 1)
2. The running minimum m_T (additional sensitivity)

This makes lookbacks extremely sensitive to spot movements early in their life.

---

## 4. Mathematical Framework

### 4.1 Underlying Dynamics (GBM)

Under risk-neutral measure Q:
```
dS = (r - q)S dt + σS dW
```

Where:
- r = domestic rate
- q = foreign rate (for FX) or dividend yield
- σ = volatility
- W = standard Brownian motion

### 4.2 Running Maximum/Minimum

Define:
```
M_t = max_{0 ≤ s ≤ t} S_s   (running maximum)
m_t = min_{0 ≤ s ≤ t} S_s   (running minimum)
```

Under GBM, the joint distribution of (S_T, M_T) and (S_T, m_T) can be derived using the **reflection principle**.

### 4.3 Log-Transform

Let X_t = log(S_t). Under GBM:
```
X_t = X_0 + μt + σW_t
```

Where μ = r - q - σ²/2 (drift in log-space).

The running extrema in log-space are:
```
Y_T = max_{0 ≤ t ≤ T} X_t = log(M_T)
y_T = min_{0 ≤ t ≤ T} X_t = log(m_T)
```

---

## 5. Continuous Monitoring: Closed-Form Solution

### 5.1 Goldman-Sosin-Gatto Formula (1979)

For **floating strike lookback call** with continuous monitoring:

```
V = S₀ e^{-qT} N(a₁) - S₀ e^{-qT} (σ²/2μ) [
    (S₀/m₀)^{-2μ/σ²} N(-a₁ + 2μ√T/σ) - e^{μT} N(-a₁)
  ] - m₀ e^{-rT} N(a₂) + m₀ e^{-rT} (σ²/2μ) [
    (S₀/m₀)^{1-2μ/σ²} N(-a₂ + 2μ√T/σ) - e^{μT} N(-a₂)
  ]
```

Where:
- μ = r - q (risk-neutral drift)
- m₀ = current running minimum
- a₁ = [log(S₀/m₀) + (μ + σ²/2)T] / (σ√T)
- a₂ = [log(S₀/m₀) + (μ - σ²/2)T] / (σ√T)

**Note**: This formula is complex and requires careful implementation. For practical purposes, MC is often preferred.

### 5.2 Key Simplification at Inception

At inception (t=0), we have m₀ = S₀, which simplifies the formula significantly.

---

## 6. Discrete Monitoring: Monte Carlo Pricing

### 6.1 Algorithm

For discrete monitoring with n steps:

```
1. Generate GBM paths: S₀ → S₁ → ... → Sₙ
2. Compute extrema for each path:
   - M_T = max(S₀, S₁, ..., Sₙ)
   - m_T = min(S₀, S₁, ..., Sₙ)
3. Compute payoff based on lookback type:
   - Floating call: S_T - m_T
   - Floating put: M_T - S_T
   - Fixed call: max(M_T - K, 0)
   - Fixed put: max(K - m_T, 0)
4. Discount and take average
```

### 6.2 Discrete vs Continuous

**Key Insight**: Discrete monitoring typically **underestimates** the continuous value.

Why?
- With discrete monitoring, we may "miss" the true extremum
- The true max/min might occur between monitoring points
- More monitoring points → closer to continuous value

**Correction Factor**: For practical purposes, use enough monitoring points (e.g., daily for a 1-year option ≈ 252 points).

### 6.3 Variance Reduction

Lookback options can have high variance due to extremum computation. Techniques:
- **Antithetic variates**: Highly effective (halves paths needed)
- **Control variate**: Use vanilla option as control
- **Stratification**: Stratify based on terminal spot

---

## 7. Reflection Principle

### 7.1 Mathematical Background

The reflection principle relates the distribution of the running maximum to the standard Brownian motion.

For Brownian motion W_t starting at 0:
```
P(max_{0≤s≤t} W_s ≥ a) = 2 P(W_t ≥ a) = 2 N(-a/√t)
```

**Intuition**: For every path that reaches level a, there's a "reflected" path that exceeds a.

### 7.2 Joint Density

The joint density of (W_T, M_T) where M_T = max_{0≤t≤T} W_t:
```
f(w, m) = (2(2m-w)/T) φ((2m-w)/√T)  for m ≥ max(0, w)
```

Where φ is the standard normal density.

This is the foundation for deriving closed-form lookback prices.

---

## 8. Key Interview Points

### 8.1 Must-Know Facts

1. **Floating strike lookbacks are ALWAYS ITM** (payoff ≥ 0)
2. **Lookback ≥ Vanilla** (captures optimal timing)
3. **Delta at inception ≈ 2** for floating strike call
4. **Discrete monitoring underestimates continuous value**
5. **Goldman-Sosin-Gatto (1979)** derived continuous monitoring formulas

### 8.2 Common Questions

**Q: Why are lookback options expensive?**
A: They capture the value of "perfect market timing" - the holder never regrets their entry/exit point.

**Q: What's the delta of a floating strike lookback call at inception?**
A: Approximately 2. It's sensitive to both terminal spot and running minimum.

**Q: How does discrete monitoring affect the price?**
A: Discrete monitoring typically gives a lower price than continuous, as you may miss the true extremum between monitoring points.

**Q: When would you use a lookback option?**
A: When the client wants protection against timing risk - e.g., a corporate wanting to convert currency at the best rate over a period.

### 8.3 Typical Hedge Ratios

At inception for floating strike lookback call:
- Delta ≈ 2 (very sensitive)
- Gamma is large (convex payoff)
- Vega is positive (benefits from volatility)

---

## 9. Important Formulas to Remember

### 9.1 Payoff Formulas

| Type | Call | Put |
|------|------|-----|
| Floating Strike | S_T - m_T | M_T - S_T |
| Fixed Strike | max(M_T - K, 0) | max(K - m_T, 0) |

### 9.2 Key Inequalities

```
Lookback Call ≥ Vanilla Call (fixed strike)
Lookback Put ≥ Vanilla Put (fixed strike)

Floating Strike Call ≥ 0 (always)
Floating Strike Put ≥ 0 (always)
```

### 9.3 Reflection Principle

```
P(max_{0≤s≤t} W_s ≥ a) = 2P(W_t ≥ a)
```

---

## 10. Implementation Notes

### 10.1 Numerical Considerations

1. **Path Extrema**: Use `np.max(path, axis=1)` and `np.min(path, axis=1)` for efficiency
2. **Monitoring Points**: For 1Y option, use ≥52 steps (weekly) or 252 (daily)
3. **Edge Case**: At expiry (T=0), path = [S₀], so M_T = m_T = S_T = S₀

### 10.2 Common Pitfalls

1. **Forgetting S₀ in extremum**: The path includes S₀, not just future points
2. **Confusing floating vs fixed**: Floating strike has NO strike parameter
3. **Discrete vs continuous**: Don't use continuous formula for discrete monitoring

### 10.3 Code Pattern

```python
# Floating strike lookback call
terminal_spots = paths[:, -1]
min_spots = np.min(paths, axis=1)
payoff = terminal_spots - min_spots  # Always >= 0

# Fixed strike lookback call
max_spots = np.max(paths, axis=1)
payoff = np.maximum(max_spots - strike, 0.0)
```

---

## 11. Related Products

- **Asian Options**: Average over path vs extremum
- **Barrier Options**: Path-dependent with barrier trigger
- **Cliquet Options**: Multiple lookback-like resets
- **Partial Lookback**: Extremum over subset of path

---

## 12. References

1. Goldman, Sosin, Gatto (1979): "Path-Dependent Options: Buy at the Low, Sell at the High"
2. Conze, Viswanathan (1991): "Path Dependent Options: The Case of Lookback Options"
3. Hull (2018): "Options, Futures, and Other Derivatives", Chapter 26
4. Wilmott (2006): "Paul Wilmott on Quantitative Finance", Volume 2, Chapter 16

---

## Summary

Lookback options provide "perfect hindsight" by capturing the optimal entry/exit point over the option's life. Key points:

1. **Floating strike** lookbacks are **always ITM** (payoff ≥ 0)
2. **Lookback ≥ Vanilla** due to optimal timing capture
3. **Monte Carlo** is standard for discrete monitoring
4. **Closed-form solutions** exist for continuous monitoring (Goldman-Sosin-Gatto)
5. **Delta ≈ 2** at inception for floating strike call (very sensitive)
6. **Discrete monitoring underestimates continuous** - use sufficient monitoring points

Understanding lookback options demonstrates knowledge of:
- Path-dependent option theory
- Extreme value statistics
- Reflection principle
- Practical pricing considerations (discrete vs continuous)
