# Asian Options: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Product Class:** Path-Dependent Exotic Option  
**Pricing Methods:** Monte Carlo (arithmetic), Analytic approximations (geometric)  
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

An **Asian option** (also called an **average price option**) is a path-dependent derivative whose payoff depends on the **average price** of the underlying asset over a specified period, rather than just the terminal price.

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Path Dependency** | Payoff depends on entire price path, not just terminal value |
| **Volatility Reduction** | Averaging reduces effective volatility → cheaper than vanilla |
| **No Closed-Form (Arithmetic)** | Sum of lognormals is not lognormal |
| **Closed-Form (Geometric)** | Product of lognormals is lognormal |
| **Primary Use Cases** | Commodity hedging, FX averaging, reducing manipulation risk |

### 1.3 Pricing Summary

| Averaging Type | Closed-Form | Recommended Method |
|----------------|-------------|-------------------|
| Geometric | Yes (modified BSM) | Analytic |
| Arithmetic | No | Monte Carlo, Moment Matching |

---

## 2. Formal Mathematical Framework

### 2.1 Probability Space and Filtration

Let \((\Omega, \mathcal{F}, \mathbb{P})\) be a probability space equipped with a filtration \(\{\mathcal{F}_t\}_{t \geq 0}\) satisfying the usual conditions (right-continuous, complete).

**Brownian Motion**: Let \(W = \{W_t\}_{t \geq 0}\) be a standard Brownian motion adapted to \(\{\mathcal{F}_t\}\).

### 2.2 Asset Dynamics

Under the physical measure \(\mathbb{P}\), the spot price \(S_t\) follows **Geometric Brownian Motion (GBM)**:

\[
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t^{\mathbb{P}}
\]

where:
- \(\mu\) = drift (expected return under \(\mathbb{P}\))
- \(\sigma\) = volatility (constant)

### 2.3 Risk-Neutral Measure

By the **Fundamental Theorem of Asset Pricing**, there exists an equivalent martingale measure \(\mathbb{Q}\) under which discounted asset prices are martingales.

**Girsanov's Theorem**: Define \(W_t^{\mathbb{Q}} = W_t^{\mathbb{P}} + \lambda t\) where \(\lambda = \frac{\mu - r}{\sigma}\) is the market price of risk. Then \(W^{\mathbb{Q}}\) is a Brownian motion under \(\mathbb{Q}\).

Under \(\mathbb{Q}\):

\[
dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
\]

where:
- \(r\) = domestic risk-free rate
- \(q\) = foreign rate (FX) or dividend yield (equity)

### 2.4 Solution to the SDE

The SDE has the explicit solution:

\[
S_t = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)t + \sigma W_t^{\mathbb{Q}}\right]
\]

**Log-price process**: Let \(X_t = \log(S_t)\). Then:

\[
X_t = X_0 + \left(r - q - \frac{\sigma^2}{2}\right)t + \sigma W_t^{\mathbb{Q}}
\]

This shows that \(X_t\) is normally distributed:

\[
X_t \sim \mathcal{N}\left(X_0 + \left(r - q - \frac{\sigma^2}{2}\right)t, \sigma^2 t\right)
\]

### 2.5 Risk-Neutral Pricing Formula

The price of any European-style derivative with payoff \(H\) at maturity \(T\) is:

\[
V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[H | \mathcal{F}_0]
\]

For path-dependent options, \(H\) depends on the entire path \(\{S_t\}_{0 \leq t \leq T}\).

---

## 3. Product Specification

### 3.1 Average Definition

#### Discrete Monitoring

For monitoring times \(0 = t_0 < t_1 < \cdots < t_n = T\):

**Arithmetic Average:**
\[
A_{\text{arith}} = \frac{1}{n+1} \sum_{i=0}^{n} S_{t_i}
\]

**Geometric Average:**
\[
A_{\text{geom}} = \left(\prod_{i=0}^{n} S_{t_i}\right)^{\frac{1}{n+1}} = \exp\left(\frac{1}{n+1} \sum_{i=0}^{n} \log S_{t_i}\right)
\]

#### Continuous Monitoring

**Arithmetic Average:**
\[
A_{\text{arith}} = \frac{1}{T} \int_0^T S_t \, dt
\]

**Geometric Average:**
\[
A_{\text{geom}} = \exp\left(\frac{1}{T} \int_0^T \log S_t \, dt\right)
\]

### 3.2 Payoff Structure

#### Average Price Option (Standard)

**Call:**
\[
\text{Payoff}_{\text{call}} = \max(A - K, 0) = (A - K)^+
\]

**Put:**
\[
\text{Payoff}_{\text{put}} = \max(K - A, 0) = (K - A)^+
\]

#### Average Strike Option

**Call:**
\[
\text{Payoff}_{\text{call}} = \max(S_T - A, 0) = (S_T - A)^+
\]

**Put:**
\[
\text{Payoff}_{\text{put}} = \max(A - S_T, 0) = (A - S_T)^+
\]

### 3.3 Contract Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Notional | \(N\) | Face value (foreign currency for FX) |
| Strike | \(K\) | Exercise price |
| Maturity | \(T\) | Time to expiry (year fraction) |
| Averaging Type | - | Arithmetic or Geometric |
| Monitoring | - | Discrete (daily/weekly) or Continuous |
| Option Type | - | Call or Put |

### 3.4 Pricing Formula (General)

\[
V_0 = N \cdot e^{-rT} \cdot \mathbb{E}^{\mathbb{Q}}[\text{Payoff}]
\]

---

## 4. Pricing Theory

### 4.1 Why Arithmetic Asian Options Have No Closed-Form

**Fundamental Issue**: The sum of lognormal random variables is **not** lognormal.

If \(S_{t_i} \sim \text{Lognormal}\), then \(A_{\text{arith}} = \frac{1}{n+1}\sum_i S_{t_i}\) does **not** follow a lognormal distribution.

**Mathematical Proof**:

Let \(X_i = \log S_{t_i} \sim \mathcal{N}(\mu_i, \sigma_i^2)\). Then \(S_{t_i} = e^{X_i}\).

The sum \(\sum_i S_{t_i} = \sum_i e^{X_i}\) is a sum of correlated lognormal variables.

The characteristic function of a sum of lognormals has no closed-form expression, hence the distribution of the arithmetic average cannot be expressed analytically.

### 4.2 Geometric Asian Option: Closed-Form Solution

#### Key Insight

The geometric average **is** lognormal:

\[
\log A_{\text{geom}} = \frac{1}{n+1} \sum_{i=0}^{n} \log S_{t_i} = \frac{1}{n+1} \sum_{i=0}^{n} X_{t_i}
\]

Since each \(X_{t_i}\) is normal, and a linear combination of normals is normal, \(\log A_{\text{geom}}\) is normal, hence \(A_{\text{geom}}\) is lognormal.

#### Moments of Geometric Average (Continuous)

For continuous monitoring:

\[
\log A_{\text{geom}} = \frac{1}{T} \int_0^T X_t \, dt
\]

**Mean:**
\[
\mathbb{E}^{\mathbb{Q}}[\log A_{\text{geom}}] = \log S_0 + \frac{1}{T} \int_0^T \left(r - q - \frac{\sigma^2}{2}\right)t \, dt = \log S_0 + \frac{1}{2}\left(r - q - \frac{\sigma^2}{2}\right)T
\]

**Variance:**
\[
\text{Var}(\log A_{\text{geom}}) = \frac{1}{T^2} \int_0^T \int_0^T \text{Cov}(\sigma W_s, \sigma W_t) \, ds \, dt = \frac{\sigma^2}{T^2} \int_0^T \int_0^T \min(s,t) \, ds \, dt = \frac{\sigma^2 T}{3}
\]

#### Adjusted Parameters for BSM

Define:
\[
\tilde{\sigma} = \frac{\sigma}{\sqrt{3}}, \quad \tilde{r} = \frac{1}{2}\left(r - q - \frac{\sigma^2}{6}\right) + \frac{1}{2}q
\]

Then the geometric Asian call price is:

\[
C_{\text{geom}} = e^{-rT} \left[ F_A N(d_1) - K N(d_2) \right]
\]

where:
- \(F_A = S_0 e^{(\tilde{r} - q + \frac{\tilde{\sigma}^2}{2})T}\) (forward of average)
- \(d_1 = \frac{\log(F_A/K) + \frac{\tilde{\sigma}^2 T}{2}}{\tilde{\sigma}\sqrt{T}}\)
- \(d_2 = d_1 - \tilde{\sigma}\sqrt{T}\)

### 4.3 Arithmetic Asian Option: Monte Carlo Pricing

#### Algorithm

```
Input: S₀, K, T, r, q, σ, N_paths, N_steps
Output: Option price V₀

1. Δt ← T / N_steps
2. For each path i = 1, ..., N_paths:
   a. S ← S₀
   b. sum_S ← S₀
   c. For j = 1, ..., N_steps:
      - Z ~ N(0,1)
      - S ← S × exp((r - q - σ²/2)Δt + σ√Δt × Z)
      - sum_S ← sum_S + S
   d. A_i ← sum_S / (N_steps + 1)
   e. payoff_i ← max(A_i - K, 0)  [for call]
3. V₀ ← e^{-rT} × mean(payoff)
```

#### Convergence Rate

By the **Central Limit Theorem**, the Monte Carlo estimator converges at rate \(O(1/\sqrt{N})\):

\[
\text{Standard Error} = \frac{\hat{\sigma}_{\text{payoff}}}{\sqrt{N_{\text{paths}}}}
\]

### 4.4 Jensen's Inequality and Price Ordering

**Theorem (Jensen's Inequality)**: For a convex function \(f\) and random variable \(X\):

\[
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
\]

**Application to Averages**:

Since \(\log\) is concave:
\[
\log\left(\frac{1}{n}\sum_i S_i\right) \geq \frac{1}{n}\sum_i \log S_i
\]

Therefore:
\[
A_{\text{arith}} \geq A_{\text{geom}}
\]

**Price Implication** (for calls with same strike):
\[
C_{\text{arith}} \geq C_{\text{geom}}
\]

The geometric average is always ≤ arithmetic average, so arithmetic Asian calls are worth more.

### 4.5 Turnbull-Wakeman Approximation

For arithmetic Asian options, a popular approximation matches the first two moments:

**Adjusted Volatility:**
\[
\sigma_A^2 = \frac{1}{T^2} \left[ \frac{2S_0^2 e^{2(r-q)T}}{(r-q+\sigma^2)(2(r-q)+\sigma^2)} + \frac{2S_0^2}{(r-q)(r-q+\sigma^2)} \left(\frac{1}{2(r-q)+\sigma^2} - \frac{e^{(r-q)T}}{r-q+\sigma^2}\right) \right]
\]

This gives an approximate BSM-style formula by substituting adjusted forward and volatility.

---

## 5. Greeks and Sensitivities

### 5.1 Overview

Greeks measure the sensitivity of option price to various parameters. For path-dependent options, Greeks are typically computed via:
1. **Bump-and-reprice** (finite difference)
2. **Pathwise derivatives** (for Monte Carlo)
3. **Likelihood ratio method**

### 5.2 Delta (Δ)

**Definition:**
\[
\Delta = \frac{\partial V}{\partial S_0}
\]

**Interpretation**: Change in option price per unit change in spot.

**For Asian Options:**
- Delta < 1 for calls (due to averaging)
- As \(t \to T\), delta converges to intrinsic (average becomes fixed)

**Finite Difference Approximation:**
\[
\Delta \approx \frac{V(S_0 + \epsilon) - V(S_0 - \epsilon)}{2\epsilon}
\]

### 5.3 Gamma (Γ)

**Definition:**
\[
\Gamma = \frac{\partial^2 V}{\partial S_0^2} = \frac{\partial \Delta}{\partial S_0}
\]

**Interpretation**: Convexity of option price; rate of change of delta.

**For Asian Options:**
- Lower gamma than vanilla (averaging smooths the payoff)
- Reduces hedging frequency needed

### 5.4 Vega (ν)

**Definition:**
\[
\nu = \frac{\partial V}{\partial \sigma}
\]

**Interpretation**: Sensitivity to implied volatility.

**For Asian Options:**
- Lower vega than vanilla (averaging reduces effective volatility)
- Ratio: \(\nu_{\text{Asian}} / \nu_{\text{vanilla}} \approx 1/\sqrt{3}\) for continuous geometric

### 5.5 Theta (Θ)

**Definition:**
\[
\Theta = \frac{\partial V}{\partial t}
\]

**Interpretation**: Time decay; change in value as time passes.

**For Asian Options:**
- Complex behavior as averaging window affects value
- Path dependency makes theta time-varying

### 5.6 Rho (ρ)

**Definition:**
\[
\rho_d = \frac{\partial V}{\partial r}, \quad \rho_f = \frac{\partial V}{\partial q}
\]

**Interpretation**: Sensitivity to interest rates.

### 5.7 Greeks Summary Table

| Greek | Symbol | Formula | Asian vs Vanilla |
|-------|--------|---------|------------------|
| Delta | Δ | ∂V/∂S₀ | Lower |
| Gamma | Γ | ∂²V/∂S₀² | Lower |
| Vega | ν | ∂V/∂σ | Lower (~1/√3) |
| Theta | Θ | ∂V/∂t | Complex |
| Rho | ρ | ∂V/∂r | Similar |

---

## 6. Numerical Methods

### 6.1 Monte Carlo Enhancements

#### Antithetic Variates

For each path with \(Z_1, \ldots, Z_n\), generate a mirror path with \(-Z_1, \ldots, -Z_n\).

**Variance Reduction:**
\[
\text{Var}\left(\frac{\hat{V} + \hat{V}^{\text{anti}}}{2}\right) = \frac{1}{2}\text{Var}(\hat{V})(1 + \rho)
\]

where \(\rho = \text{Corr}(\hat{V}, \hat{V}^{\text{anti}})\). If \(\rho < 0\), variance is reduced.

#### Control Variate (Geometric as Control)

Use the geometric Asian option (with known closed-form) as control:

\[
\hat{V}_{\text{CV}} = \hat{V}_{\text{arith}} + c(\hat{V}_{\text{geom}} - V_{\text{geom}}^{\text{exact}})
\]

Optimal \(c = -\text{Cov}(\hat{V}_{\text{arith}}, \hat{V}_{\text{geom}}) / \text{Var}(\hat{V}_{\text{geom}})\).

**Effectiveness**: Can reduce variance by factor of 10-100×.

### 6.2 Moment Matching (Levy Approximation)

Match the first two moments of the arithmetic average to a lognormal:

1. Compute \(M_1 = \mathbb{E}[A_{\text{arith}}]\) and \(M_2 = \mathbb{E}[A_{\text{arith}}^2]\)
2. Set \(\mu_{\text{adj}} = 2\log M_1 - \frac{1}{2}\log M_2\)
3. Set \(\sigma_{\text{adj}}^2 = \log M_2 - 2\log M_1\)
4. Price using BSM with adjusted parameters

### 6.3 PDE Methods

For continuous averaging, the PDE has an extra state variable:

Let \(I_t = \int_0^t S_u \, du\). Then \(V = V(t, S_t, I_t)\) satisfies:

\[
\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + S\frac{\partial V}{\partial I} = rV
\]

**Terminal Condition:**
\[
V(T, S, I) = \max\left(\frac{I}{T} - K, 0\right)
\]

This 2D PDE is more expensive than 1D vanilla but tractable.

---

## 7. Risk Management

### 7.1 Hedging Asian Options

**Delta Hedging:**
- Hold Δ units of underlying
- Rebalance as Δ changes (less frequent than vanilla due to lower gamma)

**Advantages for Risk Management:**
1. **Lower gamma** → less rebalancing
2. **Lower vega** → less vol exposure
3. **Path averaging** → reduces manipulation risk

### 7.2 Model Risk Considerations

| Risk Factor | Impact | Mitigation |
|-------------|--------|------------|
| Volatility smile | Asian less sensitive | Use local vol or SABR |
| Discrete vs continuous | Discrete < continuous value | Price with actual monitoring |
| Correlation (baskets) | Critical for basket Asians | Copula methods |

### 7.3 P&L Attribution

Asian option P&L can be decomposed:
\[
\Delta P\&L = \Delta \cdot \Delta S + \frac{1}{2}\Gamma \cdot (\Delta S)^2 + \nu \cdot \Delta\sigma + \Theta \cdot \Delta t + \text{residual}
\]

---

## 8. Implementation

### 8.1 Pseudocode: Monte Carlo Pricer

```python
def price_asian_mc(S0, K, T, r, q, sigma, n_paths, n_steps, avg_type="arithmetic"):
    """
    Monte Carlo pricer for Asian options.
    
    Parameters
    ----------
    S0 : float - Initial spot price
    K : float - Strike price
    T : float - Time to maturity
    r : float - Domestic risk-free rate
    q : float - Foreign rate / dividend yield
    sigma : float - Volatility
    n_paths : int - Number of MC paths
    n_steps : int - Number of time steps
    avg_type : str - "arithmetic" or "geometric"
    
    Returns
    -------
    float - Option price
    """
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * sqrt(dt)
    
    # Generate paths
    Z = np.random.standard_normal((n_paths, n_steps))
    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(np.column_stack([np.zeros(n_paths), log_paths]))
    
    # Compute averages
    if avg_type == "arithmetic":
        averages = np.mean(paths, axis=1)
    else:  # geometric
        averages = np.exp(np.mean(np.log(paths), axis=1))
    
    # Compute payoffs and discount
    payoffs = np.maximum(averages - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price
```

### 8.2 Numerical Considerations

1. **Numerical Stability**: Use log-space for geometric averages
2. **Overflow Prevention**: Scale by notional at the end
3. **Random Number Quality**: Use Mersenne Twister or Sobol sequences
4. **Parallelization**: Paths are independent → embarrassingly parallel

### 8.3 Convergence Diagnostics

Track:
- Running mean of discounted payoffs
- Standard error: \(\text{SE} = \sigma_{\text{payoff}} / \sqrt{n}\)
- 95% CI: \(\hat{V} \pm 1.96 \times \text{SE}\)

---

## 9. Key Interview Points

### 9.1 Must-Know Facts

1. **No closed-form for arithmetic**: Sum of lognormals is not lognormal
2. **Geometric has closed-form**: Product of lognormals is lognormal
3. **Arithmetic ≥ Geometric**: Jensen's inequality
4. **Asian cheaper than vanilla**: Averaging reduces effective volatility
5. **Lower Greeks**: Delta, Gamma, Vega all reduced by averaging

### 9.2 Common Interview Questions

**Q: Why is an arithmetic Asian option cheaper than a vanilla?**

A: Averaging reduces the effective volatility of the underlying. The variance of the average is less than the variance of the terminal spot:
\[
\text{Var}(A) < \text{Var}(S_T)
\]
Since option value increases with volatility, lower effective volatility means lower price.

**Q: Why can't we get a closed-form for arithmetic Asians?**

A: The arithmetic average is a sum of correlated lognormal random variables. While each \(S_{t_i}\) is lognormal, their sum \(\sum S_{t_i}\) does not follow a known distribution with a tractable characteristic function.

**Q: How does the geometric average help as a control variate?**

A: The geometric average is highly correlated with the arithmetic average (same underlying paths), but has a known closed-form price. Using it as a control variate significantly reduces Monte Carlo variance:
\[
\hat{V}_{\text{CV}} = \hat{V}_{\text{arith}} - c(\hat{V}_{\text{geom}} - V_{\text{geom}}^{\text{exact}})
\]

**Q: What happens to delta as the averaging period progresses?**

A: As more of the average becomes "locked in" (past prices are known), the option becomes less sensitive to future spot movements. Delta decreases toward zero as \(t \to T\), converging to intrinsic value behavior.

### 9.3 Quick Formulas to Remember

| Formula | Expression |
|---------|------------|
| Arithmetic Average | \(A = \frac{1}{n}\sum_{i=1}^n S_{t_i}\) |
| Geometric Average | \(A = \left(\prod_{i=1}^n S_{t_i}\right)^{1/n}\) |
| Jensen's Inequality | \(A_{\text{arith}} \geq A_{\text{geom}}\) |
| Geometric Vol Adjustment | \(\tilde{\sigma} = \sigma/\sqrt{3}\) |
| MC Convergence | \(\text{SE} = \sigma/\sqrt{n}\) |

---

## 10. References

### Academic Papers

1. **Kemna, A.G.Z. and Vorst, A.C.F.** (1990). "A Pricing Method for Options Based on Average Asset Values." *Journal of Banking & Finance*, 14(1), 113-129.

2. **Turnbull, S.M. and Wakeman, L.M.** (1991). "A Quick Algorithm for Pricing European Average Options." *Journal of Financial and Quantitative Analysis*, 26(3), 377-389.

3. **Levy, E.** (1992). "Pricing European Average Rate Currency Options." *Journal of International Money and Finance*, 11(5), 474-491.

4. **Geman, H. and Yor, M.** (1993). "Bessel Processes, Asian Options, and Perpetuities." *Mathematical Finance*, 3(4), 349-375.

### Textbooks

5. **Hull, J.C.** (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson. Chapter 26.

6. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance* (2nd ed.). Wiley. Volume 2, Chapter 25.

7. **Shreve, S.E.** (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer. Chapter 7.

8. **Glasserman, P.** (2003). *Monte Carlo Methods in Financial Engineering*. Springer. Chapter 4.

---

## Appendix A: Derivation of Geometric Asian Moments

### A.1 Mean of Log-Average (Continuous)

Let \(X_t = \log S_t\). Under \(\mathbb{Q}\):
\[
X_t = X_0 + \left(r - q - \frac{\sigma^2}{2}\right)t + \sigma W_t
\]

The continuous geometric average satisfies:
\[
\log A_{\text{geom}} = \frac{1}{T}\int_0^T X_t \, dt
\]

Computing the expectation:
\[
\mathbb{E}[\log A_{\text{geom}}] = \frac{1}{T}\int_0^T \mathbb{E}[X_t] \, dt = \frac{1}{T}\int_0^T \left[X_0 + \left(r - q - \frac{\sigma^2}{2}\right)t\right] dt
\]

\[
= X_0 + \frac{1}{T}\left(r - q - \frac{\sigma^2}{2}\right)\frac{T^2}{2} = \log S_0 + \frac{T}{2}\left(r - q - \frac{\sigma^2}{2}\right)
\]

### A.2 Variance of Log-Average (Continuous)

\[
\text{Var}(\log A_{\text{geom}}) = \text{Var}\left(\frac{1}{T}\int_0^T \sigma W_t \, dt\right) = \frac{\sigma^2}{T^2}\text{Var}\left(\int_0^T W_t \, dt\right)
\]

Using \(\text{Cov}(W_s, W_t) = \min(s,t)\):
\[
\text{Var}\left(\int_0^T W_t \, dt\right) = \int_0^T \int_0^T \min(s,t) \, ds \, dt = \frac{T^3}{3}
\]

Therefore:
\[
\text{Var}(\log A_{\text{geom}}) = \frac{\sigma^2}{T^2} \cdot \frac{T^3}{3} = \frac{\sigma^2 T}{3}
\]

---

## Appendix B: Proof of Jensen's Inequality

**Statement**: For a concave function \(\phi\) and integrable random variable \(X\):
\[
\phi(\mathbb{E}[X]) \geq \mathbb{E}[\phi(X)]
\]

**Application**: Since \(\log\) is concave:
\[
\log\left(\frac{1}{n}\sum_{i=1}^n S_i\right) \geq \frac{1}{n}\sum_{i=1}^n \log S_i = \log\left(\prod_{i=1}^n S_i\right)^{1/n}
\]

Exponentiating both sides:
\[
A_{\text{arith}} = \frac{1}{n}\sum_{i=1}^n S_i \geq \left(\prod_{i=1}^n S_i\right)^{1/n} = A_{\text{geom}}
\]

With equality if and only if all \(S_i\) are equal.

---

*Document Version: 2.0*  
*Last Updated: January 27, 2026*  
*Author: QuantStrata Library*
