# Asian Options: Mathematical Foundations

**Product Type:** Path-Dependent Exotic Option  
**Asset Class:** FX (also applicable to Equity, Commodities)  
**Pricing Method:** Monte Carlo (required), Finite Difference (2D PDE, advanced)

---

## 1. Product Definition

### 1.1 What is an Asian Option?

An **Asian option** (also called an **average price option** or **average rate option**) is a path-dependent derivative whose payoff depends on the **average price** of the underlying asset over a specified period, rather than the terminal price.

**Key Insight:** Averaging reduces volatility, making Asian options **cheaper** than their vanilla counterparts.

### 1.2 Payoff Structure

#### Arithmetic Average Asian Option

**Call Payoff:**
```
Payoff = max(A_arithmetic - K, 0)
```

**Put Payoff:**
```
Payoff = max(K - A_arithmetic, 0)
```

Where:
- `A_arithmetic = (S_1 + S_2 + ... + S_n) / n` (arithmetic mean)
- `S_i` = spot price at monitoring time `t_i`
- `K` = strike price
- `n` = number of monitoring points

#### Geometric Average Asian Option

**Call Payoff:**
```
Payoff = max(A_geometric - K, 0)
```

**Put Payoff:**
```
Payoff = max(K - A_geometric, 0)
```

Where:
- `A_geometric = (S_1 * S_2 * ... * S_n)^(1/n)` (geometric mean)

### 1.3 Key Properties

**Jensen's Inequality:**
```
A_geometric ≤ A_arithmetic
```

This implies:
- **Geometric Asian options are cheaper than arithmetic Asian options**
- Geometric averaging reduces volatility more than arithmetic averaging

**Volatility Reduction:**
- Asian options have **lower volatility** than vanilla options
- This makes them **cheaper** (lower premium)
- Useful for hedging when you care about average exposure, not terminal exposure

---

## 2. Mathematical Framework

### 2.1 Model Assumptions

We assume the underlying follows **Geometric Brownian Motion (GBM)**:

```
dS_t = (r_d - r_f) S_t dt + σ S_t dW_t
```

Where:
- `S_t` = spot price at time `t`
- `r_d` = domestic risk-free rate
- `r_f` = foreign risk-free rate (for FX)
- `σ` = volatility
- `dW_t` = Brownian motion increment

### 2.2 Discrete Monitoring

In practice, averaging is done over **discrete monitoring points**:

```
t_0 = 0, t_1, t_2, ..., t_n = T
```

Where:
- `t_0` = valuation time (S_0 is known)
- `t_n = T` = expiry time
- Monitoring points are typically daily, weekly, or monthly

**Arithmetic Average:**
```
A_arithmetic = (1/n) * Σ_{i=0}^{n} S_{t_i}
```

**Geometric Average:**
```
A_geometric = (Π_{i=0}^{n} S_{t_i})^(1/n)
```

**Numerical Stability Note:**
For geometric averaging, we use log-space to avoid overflow:
```
log(A_geometric) = (1/n) * Σ_{i=0}^{n} log(S_{t_i})
A_geometric = exp(log(A_geometric))
```

---

## 3. Pricing Methodology

### 3.1 Monte Carlo Method (Standard)

**Algorithm:**
1. Simulate `N` paths of the underlying asset: `S^{(j)}_t` for `j = 1, ..., N`
2. For each path `j`:
   - Compute average: `A^{(j)} = average(S^{(j)}_t)`
   - Compute payoff: `Payoff^{(j)} = max(A^{(j)} - K, 0)` (call)
3. Discount and take expectation:
   ```
   PV = df_d(T) * (1/N) * Σ_{j=1}^{N} Payoff^{(j)} * notional
   ```

**Advantages:**
- ✅ Works for both arithmetic and geometric averaging
- ✅ Handles any monitoring schedule
- ✅ Easy to implement
- ✅ Can add variance reduction (antithetic, control variates)

**Disadvantages:**
- ❌ Slow convergence (O(1/√N))
- ❌ Requires many paths for accuracy
- ❌ No closed-form solution for arithmetic averaging

### 3.2 Finite Difference Method (Advanced)

For arithmetic averaging, we can solve a **2D PDE**:

**State Variables:**
- `S` = current spot price
- `A` = running average

**PDE:**
```
V_t + (r_d - r_f) S V_S + (1/2) σ² S² V_SS + (S - A)/(t - t_0) V_A - r_d V = 0
```

**Boundary Conditions:**
- Terminal: `V(T, S, A) = max(A - K, 0)` (call)
- Spot boundaries: standard GBM boundaries

**Complexity:**
- Requires 2D grid (S × A)
- More complex than 1D PDE for vanilla options
- Not implemented in V1 (deferred to future phase)

### 3.3 Closed-Form Solutions (Geometric Only)

For **geometric averaging**, closed-form solutions exist under GBM.

**Key Insight:** The geometric average of lognormal variables is also lognormal.

**Geometric Average Distribution:**
Under GBM, `A_geometric` follows a lognormal distribution with:
- Mean: `E[log(A_geometric)] = log(S_0) + (r_d - r_f - σ²/2) * T/2`
- Variance: `Var(log(A_geometric)) = σ² * T/3`

**Closed-Form Formula (Geometric Asian Call):**
```
PV = df_d(T) * [S_0 * exp((r_d - r_f - σ²/2) * T/2) * N(d_1) - K * N(d_2)]
```

Where:
```
d_1 = [log(S_0/K) + (r_d - r_f + σ²/6) * T/2] / (σ * √(T/3))
d_2 = d_1 - σ * √(T/3)
```

**Note:** This formula assumes continuous geometric averaging. For discrete monitoring, adjustments are needed.

---

## 4. Important Formulas to Remember

### 4.1 Arithmetic Average
```
A_arithmetic = (1/n) * Σ_{i=1}^{n} S_i
```

### 4.2 Geometric Average
```
A_geometric = (Π_{i=1}^{n} S_i)^(1/n) = exp((1/n) * Σ_{i=1}^{n} log(S_i))
```

### 4.3 Jensen's Inequality
```
Geometric Mean ≤ Arithmetic Mean
```

### 4.4 Monte Carlo Pricing Formula
```
PV = df_d(T) * notional * E[max(A - K, 0)]
   ≈ df_d(T) * notional * (1/N) * Σ_{j=1}^{N} max(A^{(j)} - K, 0)
```

### 4.5 Standard Error (MC Convergence)
```
SE = σ_payoff / √N
```

Where:
- `σ_payoff` = standard deviation of payoffs
- `N` = number of paths

---

## 5. Key Interview Points

### 5.1 Why Are Asian Options Cheaper?

**Answer:**
- Averaging reduces volatility (variance reduction)
- Lower volatility → lower option premium
- Intuition: You're betting on average performance, not extreme outcomes

**Mathematical Proof:**
- Variance of average: `Var(A) = Var(S) / n` (for independent observations)
- Lower variance → lower option value

### 5.2 Geometric vs Arithmetic: Which is Cheaper?

**Answer:**
- **Geometric Asian is cheaper** than arithmetic Asian
- Reason: Jensen's inequality (geometric mean ≤ arithmetic mean)
- For call options: lower average → lower payoff → lower premium

### 5.3 When Would You Use Asian Options?

**Answer:**
- **Hedging average exposure:** When you care about average price over a period, not terminal price
- **Reducing premium:** Cheaper than vanilla options
- **Commodity hedging:** Natural fit for averaging (e.g., average oil price over quarter)
- **Employee stock options:** Often use average price to reduce volatility

### 5.4 Pricing Challenges

**Answer:**
- **Arithmetic averaging:** No closed-form solution → requires numerical methods (MC or 2D PDE)
- **Geometric averaging:** Has closed-form solution, but discrete monitoring requires adjustments
- **Monitoring frequency:** More monitoring points → better approximation to continuous averaging
- **Convergence:** MC requires many paths for accuracy (variance reduction techniques help)

### 5.5 Greeks for Asian Options

**Delta:**
- Lower than vanilla (averaging reduces sensitivity to spot)
- Depends on how far we are through the averaging period

**Gamma:**
- Lower than vanilla (less convexity due to averaging)

**Vega:**
- Lower than vanilla (averaging reduces volatility sensitivity)

**Theta:**
- More complex than vanilla (depends on both time to expiry and averaging progress)

---

## 6. Implementation Notes

### 6.1 Monitoring Points

**Discrete Monitoring:**
- In practice, averaging is done over discrete points (daily, weekly, monthly)
- More monitoring points → better approximation to continuous averaging
- Typical: 64-256 steps for 1-year option

**Implementation:**
```python
# Paths shape: (n_paths, n_steps + 1)
# Column 0: S_0
# Columns 1..n_steps: intermediate monitoring points
# Last column: S_T

# Arithmetic average
average = np.mean(paths, axis=1)  # Average over time (columns)

# Geometric average (log-space for stability)
log_average = np.mean(np.log(paths), axis=1)
geometric_average = np.exp(log_average)
```

### 6.2 Variance Reduction

**Antithetic Variates:**
- For each path `S_t`, also simulate `S'_t` using `-Z_t` (negative random numbers)
- Reduces variance by ~50% (factor of 2 improvement)

**Control Variates:**
- Use geometric Asian (has closed-form) as control variate for arithmetic Asian
- Reduces variance significantly

### 6.3 Numerical Stability

**Geometric Average:**
- Use log-space: `log(A) = mean(log(S_i))`
- Prevents overflow when multiplying many large numbers
- More numerically stable

**Arithmetic Average:**
- Standard mean calculation is stable
- No special considerations needed

---

## 7. Extensions and Variations

### 7.1 Average Strike Asian Options

**Payoff:**
```
Call: max(S_T - A, 0)
Put: max(A - S_T, 0)
```

- Strike is the average (floating strike)
- Less common than fixed strike

### 7.2 Partial Average Asian Options

**Payoff:**
- Average over a subset of the option's life
- E.g., average over last 3 months of 1-year option

### 7.3 Weighted Average Asian Options

**Payoff:**
- Average with weights: `A = Σ w_i * S_i`
- Allows emphasis on certain periods

---

## 8. References and Further Reading

1. **Hull, J.C.** "Options, Futures, and Other Derivatives" - Chapter on Exotic Options
2. **Wilmott, P.** "Paul Wilmott on Quantitative Finance" - Asian Options chapter
3. **Kemna, A.G.Z. and Vorst, A.C.F.** "A Pricing Method for Options Based on Average Asset Values" (1990)

---

## 9. Summary Checklist

**For Interviews, Remember:**
- ✅ Asian options pay based on average price, not terminal price
- ✅ Arithmetic: `A = mean(S_i)`, Geometric: `A = (Π S_i)^(1/n)`
- ✅ Asian options are cheaper than vanilla (volatility reduction)
- ✅ Geometric Asian is cheaper than arithmetic Asian (Jensen's inequality)
- ✅ Arithmetic averaging: MC or 2D PDE (no closed-form)
- ✅ Geometric averaging: Has closed-form solution
- ✅ Use cases: Hedging average exposure, reducing premium
- ✅ Greeks are lower than vanilla (less sensitivity due to averaging)

**Key Formula:**
```
PV_asian ≈ PV_vanilla * (reduction_factor)
```

Where `reduction_factor < 1` due to volatility reduction from averaging.
