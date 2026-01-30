# Heston Stochastic Volatility Model: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Model Class:** Stochastic Volatility (Two-Factor)  
**Pricing Methods:** Monte Carlo, Semi-Analytic (Fourier)  
**Target Audience:** Quantitative Analysts, Financial Mathematics Graduates

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Formal Mathematical Framework](#2-formal-mathematical-framework)
3. [Model Properties](#3-model-properties)
4. [Pricing Theory](#4-pricing-theory)
5. [Monte Carlo Simulation](#5-monte-carlo-simulation)
6. [Semi-Analytic Pricing](#6-semi-analytic-pricing)
7. [Calibration](#7-calibration)
8. [Greeks and Sensitivities](#8-greeks-and-sensitivities)
9. [Implementation](#9-implementation)
10. [Key Interview Points](#10-key-interview-points)
11. [References](#11-references)

---

## 1. Executive Summary

### 1.1 Model Overview

The **Heston Model** (1993) is the most widely used stochastic volatility model in quantitative finance. It extends Black-Scholes by modeling variance as a **mean-reverting square-root process** (CIR process) correlated with the spot.

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Factors** | Two: Spot (S) and Variance (V) |
| **Variance Process** | Mean-reverting CIR (Cox-Ingersoll-Ross) |
| **Correlation** | Spot-variance correlation ρ |
| **Volatility Smile** | Generates realistic smiles/skews |
| **Closed-Form** | Semi-analytic pricing via Fourier |

### 1.3 Model Parameters

| Parameter | Symbol | Typical Range | Interpretation |
|-----------|--------|---------------|----------------|
| Mean Reversion | κ | 0.5 - 5.0 | Speed of variance mean reversion |
| Long-Term Var | θ | 0.01 - 0.10 | Equilibrium variance level |
| Vol of Vol | ξ | 0.1 - 1.0 | Variance volatility |
| Initial Var | V₀ | 0.01 - 0.20 | Starting variance |
| Correlation | ρ | -0.9 to 0.0 | Spot-variance correlation |

---

## 2. Formal Mathematical Framework

### 2.1 Model Dynamics

Under the risk-neutral measure \(\mathbb{Q}\), the Heston model specifies:

\[
\begin{aligned}
dS_t &= (r - q) S_t \, dt + \sqrt{V_t} \, S_t \, dW_t^S \\
dV_t &= \kappa(\theta - V_t) \, dt + \xi \sqrt{V_t} \, dW_t^V
\end{aligned}
\]

where:
\[
\text{Corr}(dW_t^S, dW_t^V) = \rho \, dt
\]

### 2.2 Parameter Interpretation

**Mean Reversion (κ):**
- Controls how fast variance reverts to θ
- Half-life of variance shock: \(t_{1/2} = \ln(2)/\kappa\)

**Long-Term Variance (θ):**
- Equilibrium variance level
- Long-term volatility: \(\sqrt{\theta}\)

**Volatility of Volatility (ξ):**
- Controls variance randomness
- Higher ξ → more volatile vol → fatter tails

**Correlation (ρ):**
- Negative ρ (typical for equities): down moves → vol up
- Creates the **leverage effect** and **volatility skew**

### 2.3 Feller Condition

**Theorem:** If \(2\kappa\theta > \xi^2\), then \(V_t > 0\) almost surely for all \(t\).

**Proof Sketch:**
- The CIR process has a reflecting boundary at 0
- When Feller condition holds, the drift at V=0 is strong enough to prevent touching zero

**Feller Ratio:**
\[
\phi = \frac{2\kappa\theta}{\xi^2}
\]
- φ > 1: Variance stays positive
- φ ≤ 1: Variance can touch zero (but never negative)

### 2.4 Correlation Structure

The correlated Brownian motions can be decomposed:
\[
dW_t^V = \rho \, dW_t^S + \sqrt{1-\rho^2} \, dW_t^{\perp}
\]

where \(W^S\) and \(W^{\perp}\) are independent.

---

## 3. Model Properties

### 3.1 Moments of Variance

**Expected Variance:**
\[
\mathbb{E}[V_t] = \theta + (V_0 - \theta)e^{-\kappa t}
\]

**Variance of Variance:**
\[
\text{Var}(V_t) = \frac{V_0 \xi^2}{\kappa}(e^{-\kappa t} - e^{-2\kappa t}) + \frac{\theta \xi^2}{2\kappa}(1 - e^{-\kappa t})^2
\]

**Asymptotic Variance:**
\[
\lim_{t \to \infty} \mathbb{E}[V_t] = \theta, \quad \lim_{t \to \infty} \text{Var}(V_t) = \frac{\theta \xi^2}{2\kappa}
\]

### 3.2 Implied Volatility Smile

The Heston model generates implied volatility smiles through:

1. **Skew (ρ < 0):** Negative correlation → higher vol for low strikes
2. **Convexity (ξ > 0):** Vol of vol → smile curvature
3. **Term Structure:** Smile flattens as maturity increases

**Short-Maturity Skew Approximation:**
\[
\text{Skew} \approx \frac{\rho \xi}{2} \sqrt{T}
\]

### 3.3 Fat Tails

The Heston model produces heavier tails than Black-Scholes:
- Returns are conditionally normal given the variance path
- Unconditional returns have excess kurtosis
- Kurtosis increases with ξ

---

## 4. Pricing Theory

### 4.1 Pricing PDE

The option value \(V(S, v, t)\) satisfies the 2D PDE:

\[
\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \kappa(\theta - v)\frac{\partial V}{\partial v} + \frac{1}{2}vS^2\frac{\partial^2 V}{\partial S^2} + \rho\xi vS\frac{\partial^2 V}{\partial S\partial v} + \frac{1}{2}\xi^2 v\frac{\partial^2 V}{\partial v^2} - rV = 0
\]

### 4.2 Risk-Neutral Expectation

European call price:
\[
C(S_0, V_0, K, T) = e^{-rT}\mathbb{E}^{\mathbb{Q}}[(S_T - K)^+]
\]

European put price:
\[
P(S_0, V_0, K, T) = e^{-rT}\mathbb{E}^{\mathbb{Q}}[(K - S_T)^+]
\]

### 4.3 Characteristic Function

The **characteristic function** of \(\ln(S_T)\) is key to semi-analytic pricing:

\[
\phi(u) = \mathbb{E}^{\mathbb{Q}}[e^{iu\ln(S_T)}] = e^{A(u, T) + B(u, T)V_0 + iu\ln(S_0 e^{(r-q)T})}
\]

where \(A(u, T)\) and \(B(u, T)\) satisfy Riccati ODEs (see Section 6).

---

## 5. Monte Carlo Simulation

### 5.1 Discretization Challenges

The square-root process in Heston can produce negative variance under naive discretization:

\[
V_{n+1} = V_n + \kappa(\theta - V_n)\Delta t + \xi\sqrt{V_n}\sqrt{\Delta t}Z_n^V
\]

If \(V_n\) is small and \(Z_n^V\) is large negative, \(V_{n+1}\) can be negative.

### 5.2 Discretization Schemes

**Euler (Naive):**
\[
V_{n+1} = V_n + \kappa(\theta - V_n)\Delta t + \xi\sqrt{V_n}\sqrt{\Delta t}Z_n^V
\]
- Can give negative variance
- Simple but unreliable

**Full Truncation:**
\[
V_{n+1} = \max\left(V_n + \kappa(\theta - V_n^+)\Delta t + \xi\sqrt{V_n^+}\sqrt{\Delta t}Z_n^V, 0\right)
\]
where \(V^+ = \max(V, 0)\).

- Most common scheme
- Introduces small bias but stable

**Reflection:**
\[
V_{n+1} = |V_n + \kappa(\theta - |V_n|)\Delta t + \xi\sqrt{|V_n|}\sqrt{\Delta t}Z_n^V|
\]
- Reflects negative values to positive
- Preserves some distributional properties

**Quadratic-Exponential (QE):**
- Advanced scheme by Andersen (2008)
- Exactly matches first two moments
- Most accurate but more complex

### 5.3 Correlated Random Numbers

To generate correlated Brownian increments:

```python
Z_S = np.random.standard_normal(n_paths)
Z_ind = np.random.standard_normal(n_paths)
Z_V = rho * Z_S + np.sqrt(1 - rho**2) * Z_ind

dW_S = sqrt_dt * Z_S
dW_V = sqrt_dt * Z_V
```

### 5.4 Spot Simulation

Use **log-Euler** for spot to preserve positivity:

\[
\ln(S_{n+1}) = \ln(S_n) + \left(r - q - \frac{V_n}{2}\right)\Delta t + \sqrt{V_n}\sqrt{\Delta t}Z_n^S
\]

Or equivalently:
\[
S_{n+1} = S_n \exp\left[\left(r - q - \frac{V_n}{2}\right)\Delta t + \sqrt{V_n}\sqrt{\Delta t}Z_n^S\right]
\]

---

## 6. Semi-Analytic Pricing

### 6.1 Heston's Original Formula

The European call price is:

\[
C = S_0 e^{-qT} P_1 - K e^{-rT} P_2
\]

where:
\[
P_j = \frac{1}{2} + \frac{1}{\pi}\int_0^{\infty} \text{Re}\left[\frac{e^{-iu\ln K}\phi_j(u)}{iu}\right]du
\]

for \(j = 1, 2\) with slightly different characteristic functions.

### 6.2 Characteristic Function Details

The characteristic function has the form:

\[
\phi_j(u) = \exp\{A_j(u, T) + B_j(u, T)V_0 + iu\ln(S_0 e^{(r-q)T})\}
\]

**Riccati Solutions:**

\[
B_j(u, T) = \frac{b_j - \rho\xi u i + d_j}{\xi^2} \cdot \frac{1 - e^{d_j T}}{1 - g_j e^{d_j T}}
\]

\[
A_j(u, T) = (r-q)iuT + \frac{\kappa\theta}{\xi^2}\left[(b_j - \rho\xi u i + d_j)T - 2\ln\left(\frac{1 - g_j e^{d_j T}}{1 - g_j}\right)\right]
\]

where:
\[
d_j = \sqrt{(\rho\xi u i - b_j)^2 - \xi^2(2u_j i - u^2)}
\]
\[
g_j = \frac{b_j - \rho\xi u i + d_j}{b_j - \rho\xi u i - d_j}
\]

and \(b_1 = \kappa - \rho\xi\), \(b_2 = \kappa\), \(u_1 = 1/2\), \(u_2 = -1/2\).

### 6.3 Numerical Integration

The integrals are computed numerically using:
- **Gauss-Laguerre** quadrature
- **Adaptive Simpson's rule**
- **FFT methods** (Carr-Madan)

---

## 7. Calibration

### 7.1 Calibration Objective

Minimize the difference between model and market implied vols:

\[
\min_{\kappa, \theta, \xi, V_0, \rho} \sum_{i,j} w_{ij}\left(\sigma_{model}(K_i, T_j) - \sigma_{market}(K_i, T_j)\right)^2
\]

### 7.2 Calibration Challenges

1. **Non-convex**: Multiple local minima
2. **Parameter Correlation**: κ and ξ are correlated
3. **Feller Constraint**: Maintain 2κθ > ξ²
4. **Numerical Stability**: Characteristic function can have numerical issues

### 7.3 Calibration Strategies

**Initial Guess:**
- θ = ATM variance at longest maturity
- V₀ = ATM variance at shortest maturity
- ρ = sign of skew × 0.5
- κ = 2.0 (moderate mean reversion)
- ξ = start small, increase if needed

**Optimization:**
- Use gradient-based methods (L-BFGS-B)
- Apply parameter bounds
- Regularize to prevent extreme parameters

---

## 8. Greeks and Sensitivities

### 8.1 Standard Greeks

| Greek | Definition | Heston Behavior |
|-------|------------|-----------------|
| **Delta** | ∂V/∂S | Similar to BS but vol-dependent |
| **Gamma** | ∂²V/∂S² | Affected by correlation |
| **Vega** | ∂V/∂σ | Complex - depends on V₀, θ |
| **Theta** | ∂V/∂t | Modified by variance dynamics |
| **Rho** | ∂V/∂r | Similar to BS |

### 8.2 Model-Specific Sensitivities

**Variance Sensitivities:**
- ∂V/∂V₀: Sensitivity to initial variance
- ∂V/∂θ: Sensitivity to long-term variance
- ∂V/∂κ: Sensitivity to mean reversion
- ∂V/∂ξ: Sensitivity to vol-of-vol
- ∂V/∂ρ: Sensitivity to correlation

### 8.3 Monte Carlo Greeks

Compute via finite difference:

```python
def delta_mc(pricer, params, epsilon=0.01):
    """Compute delta via bump-and-reprice."""
    S_up = S * (1 + epsilon)
    S_dn = S * (1 - epsilon)
    V_up = pricer.price(S_up, ...)
    V_dn = pricer.price(S_dn, ...)
    return (V_up - V_dn) / (2 * epsilon * S)
```

---

## 9. Implementation

### 9.1 Pseudocode: Heston MC Pricer

```python
def price_heston_mc(
    S0, K, T, r, q,
    kappa, theta, xi, v0, rho,
    option_type, n_paths, n_steps, scheme="full_truncation"
):
    """Monte Carlo pricing under Heston model."""
    
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize paths
    S = np.full(n_paths, S0)
    V = np.full(n_paths, v0)
    
    for step in range(n_steps):
        # Generate correlated normals
        Z_S = np.random.standard_normal(n_paths)
        Z_ind = np.random.standard_normal(n_paths)
        Z_V = rho * Z_S + np.sqrt(1 - rho**2) * Z_ind
        
        # Variance step (full truncation)
        V_pos = np.maximum(V, 0)
        V_new = V + kappa * (theta - V_pos) * dt + xi * np.sqrt(V_pos) * sqrt_dt * Z_V
        V = np.maximum(V_new, 0)
        
        # Spot step (log-Euler)
        drift = (r - q - 0.5 * V_pos) * dt
        diffusion = np.sqrt(V_pos) * sqrt_dt * Z_S
        S = S * np.exp(drift + diffusion)
    
    # Compute payoffs
    if option_type == "call":
        payoffs = np.maximum(S - K, 0)
    else:
        payoffs = np.maximum(K - S, 0)
    
    # Discount and average
    price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, std_error
```

### 9.2 Pseudocode: Semi-Analytic Pricing

```python
def price_heston_analytic(S0, K, T, r, q, kappa, theta, xi, v0, rho):
    """Semi-analytic Heston pricing via characteristic function."""
    
    def characteristic_fn(u, j):
        """Heston characteristic function."""
        if j == 1:
            b = kappa - rho * xi
            u_j = 0.5
        else:
            b = kappa
            u_j = -0.5
        
        a = kappa * theta
        x = np.log(S0)
        
        d = np.sqrt((rho * xi * u * 1j - b)**2 - xi**2 * (2 * u_j * 1j * u - u**2))
        g = (b - rho * xi * u * 1j + d) / (b - rho * xi * u * 1j - d)
        
        C = (r - q) * u * 1j * T + (a / xi**2) * (
            (b - rho * xi * u * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
        )
        D = (b - rho * xi * u * 1j + d) / xi**2 * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        
        return np.exp(C + D * v0 + 1j * u * x)
    
    def integrand(u, j):
        """Integrand for P_j."""
        return np.real(np.exp(-1j * u * np.log(K)) * characteristic_fn(u, j) / (1j * u))
    
    # Numerical integration
    from scipy import integrate
    P1 = 0.5 + (1/np.pi) * integrate.quad(lambda u: integrand(u, 1), 0, np.inf)[0]
    P2 = 0.5 + (1/np.pi) * integrate.quad(lambda u: integrand(u, 2), 0, np.inf)[0]
    
    call_price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return call_price
```

---

## 10. Key Interview Points

### 10.1 Must-Know Facts

1. **Two Factors**: Spot S and variance V are both stochastic
2. **Variance Process**: CIR (mean-reverting square-root)
3. **Feller Condition**: 2κθ > ξ² ensures V > 0
4. **Correlation**: ρ < 0 creates leverage effect and skew
5. **Semi-Analytic**: Fourier-based pricing is fast and accurate

### 10.2 Common Interview Questions

**Q1: Why use Heston over Black-Scholes?**

A: Heston captures:
- Volatility smile (BS assumes flat vol)
- Stochastic volatility (vol is random, not constant)
- Leverage effect (vol increases when spot drops)
- Fatter tails in returns

**Q2: What is the Feller condition and why does it matter?**

A: Feller condition 2κθ > ξ² ensures variance stays strictly positive. When violated, variance can touch zero (though never negative). Important for:
- Model stability
- Numerical simulation (negative variance is undefined)

**Q3: How does correlation ρ affect the smile?**

A: Negative ρ (typical):
- When S falls, V tends to rise
- Higher volatility at lower strikes
- Creates **downward-sloping skew**

Positive ρ:
- When S rises, V tends to rise
- Higher volatility at higher strikes
- Creates **upward-sloping skew** (rare in equities)

**Q4: What discretization scheme would you use for Monte Carlo?**

A: Full truncation is most common:
- Replace V with max(V, 0) in diffusion term
- Clamp final V to 0 if negative
- Simple, stable, small bias

For higher accuracy: Quadratic-Exponential (QE) scheme matches first two moments exactly.

**Q5: How do you calibrate Heston to market data?**

A: 
1. Choose liquid vanilla options across strikes/maturities
2. Use semi-analytic pricing (fast) in objective function
3. Minimize squared implied vol difference
4. Apply parameter bounds and Feller constraint
5. Use global optimizer first, then local refinement

### 10.3 Key Formulas to Remember

| Formula | Expression |
|---------|------------|
| **Spot SDE** | \(dS_t = (r-q)S_t dt + \sqrt{V_t}S_t dW_t^S\) |
| **Variance SDE** | \(dV_t = \kappa(\theta-V_t)dt + \xi\sqrt{V_t}dW_t^V\) |
| **Correlation** | \(\text{Corr}(dW^S, dW^V) = \rho\) |
| **Feller Condition** | \(2\kappa\theta > \xi^2\) |
| **Expected Variance** | \(\mathbb{E}[V_t] = \theta + (V_0-\theta)e^{-\kappa t}\) |

---

## 11. References

### Academic Papers

1. **Heston, S.L.** (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." Review of Financial Studies.
2. **Andersen, L.** (2008). "Efficient Simulation of the Heston Stochastic Volatility Model." Journal of Computational Finance.
3. **Gatheral, J.** (2006). "The Volatility Surface: A Practitioner's Guide."

### Textbooks

1. **Hull, J.** (2022). *Options, Futures, and Other Derivatives*, 11th ed.
2. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed.
3. **Shreve, S.** (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*.

### Industry Resources

1. QuantLib: HestonModel implementation
2. FINCAD: Heston Stochastic Volatility
3. Bloomberg: OVME Heston calibration

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Author: QuantStrata Development Team*
