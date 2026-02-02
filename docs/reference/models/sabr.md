# SABR Stochastic Volatility Model

**Complete Mathematical Framework for Volatility Smile Modeling**

This document provides a rigorous mathematical treatment of the SABR model (Hagan et al., 2002), the industry standard for interest rate and FX volatility smile modeling.

---

## Table of Contents

1. [Historical Context and Motivation](#1-historical-context-and-motivation)
2. [Model Specification](#2-model-specification)
3. [Hagan Approximation](#3-hagan-approximation)
4. [Monte Carlo Simulation](#4-monte-carlo-simulation)
5. [Calibration](#5-calibration)
6. [Special Cases](#6-special-cases)
7. [Advantages and Limitations](#7-advantages-and-limitations)
8. [Interview Key Points](#8-interview-key-points)

---

## 1. Historical Context and Motivation

### 1.1 Background

Patrick Hagan, Deep Kumar, Andrew Lesniewski, and Diana Woodward published "Managing Smile Risk" in *Wilmott Magazine* (2002). The SABR model provides:

1. **Analytic volatility formula**: Closed-form implied vol approximation
2. **Smile dynamics**: Captures market smile behavior
3. **Backbone flexibility**: CEV exponent β for various markets
4. **Industry standard**: Widely used for rates and FX

### 1.2 Why SABR Matters

- **Caplet/swaption pricing**: Standard for IR vol surfaces
- **FX smile**: Common for FX volatility modeling
- **Risk management**: Consistent Greeks across strikes
- **Interpolation**: Smooth vol surface construction

---

## 2. Model Specification

### 2.1 Dynamics

Under the forward measure $\mathbb{Q}^T$, the SABR model specifies:

$$
\boxed{
\begin{aligned}
dF_t &= \sigma_t F_t^\beta dW_t^F \\
d\sigma_t &= \nu \sigma_t dW_t^\sigma \\
dW_t^F dW_t^\sigma &= \rho \, dt
\end{aligned}
}
$$

**Parameters:**
- $F_t$: Forward price (or rate)
- $\sigma_t$: Stochastic volatility (starts at $\sigma_0 = \alpha$)
- $\beta \in [0, 1]$: CEV exponent (backbone)
- $\nu > 0$: Vol-of-vol
- $\rho \in (-1, 1)$: Spot-vol correlation
- $\alpha > 0$: Initial volatility

### 2.2 Parameter Interpretation

| Parameter | Symbol | Effect | Typical Values |
|-----------|--------|--------|----------------|
| Initial vol | $\alpha$ | ATM vol level | 0.1 - 0.5 |
| Backbone | $\beta$ | Smile shape | 0, 0.5, or 1 |
| Correlation | $\rho$ | Skew direction | -0.7 to +0.3 |
| Vol of vol | $\nu$ | Smile curvature | 0.2 - 0.8 |

### 2.3 Martingale Property

Under $\mathbb{Q}^T$, $F_t$ is a martingale (zero drift) because it's the forward price under the $T$-forward measure.

---

## 3. Hagan Approximation

### 3.1 Implied Volatility Formula

**Theorem (Hagan et al., 2002):** The Black implied volatility is approximately:

$$
\boxed{\sigma_{impl}(K) = \frac{\alpha}{(FK)^{(1-\beta)/2}} \cdot \frac{z}{x(z)} \cdot \left[1 + \epsilon_1 T + O(T^2)\right]}
$$

**Components:**

**z (normalized moneyness):**
$$
z = \frac{\nu}{\alpha}(FK)^{(1-\beta)/2}\ln\frac{F}{K}
$$

**x(z) (correlation adjustment):**
$$
x(z) = \ln\left[\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}\right]
$$

**Time correction $\epsilon_1$:**
$$
\epsilon_1 = \frac{(1-\beta)^2 \alpha^2}{24(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{(1-\beta)/2}} + \frac{(2-3\rho^2)\nu^2}{24}
$$

### 3.2 ATM Formula (F = K)

When $F = K$:
$$
\boxed{\sigma_{ATM} = \frac{\alpha}{F^{1-\beta}}\left[1 + \left(\frac{(1-\beta)^2\alpha^2}{24F^{2-2\beta}} + \frac{\rho\beta\nu\alpha}{4F^{1-\beta}} + \frac{(2-3\rho^2)\nu^2}{24}\right)T\right]}
$$

### 3.3 Approximation Quality

The Hagan formula is accurate for:
- Small T (T < 2Y typical)
- Strikes not too far OTM
- $\nu$ not too large

For deep OTM or long maturities, use MC or alternative formulas.

---

## 4. Monte Carlo Simulation

### 4.1 Discretization Schemes

**Log-Euler (Recommended for β=1):**
$$
\ln F_{t+dt} = \ln F_t - \frac{1}{2}\sigma_t^2 dt + \sigma_t \sqrt{dt} Z_F
$$
$$
\ln \sigma_{t+dt} = \ln \sigma_t - \frac{1}{2}\nu^2 dt + \nu \sqrt{dt} Z_\sigma
$$

where $(Z_F, Z_\sigma)$ are correlated normals with correlation $\rho$.

**General CEV (β < 1):**
$$
\ln F_{t+dt} = \ln F_t - \frac{1}{2}\sigma_t^2 F_t^{2\beta-2} dt + \sigma_t F_t^{\beta-1}\sqrt{dt} Z_F
$$

### 4.2 Correlation Structure

Generate correlated normals:
$$
Z_\sigma = \rho Z_F + \sqrt{1-\rho^2} Z_\perp
$$

where $Z_F, Z_\perp$ are independent standard normals.

### 4.3 Boundary Handling

For $\beta < 1$, $F = 0$ is an absorbing boundary. Use:
- Log-Euler to preserve positivity
- Absorbing scheme: $F_{t+dt} = \max(F_{t+dt}, 0)$

---

## 5. Calibration

### 5.1 Calibration Strategy

**Standard Approach:**
1. Fix $\beta$ based on market convention
2. Calibrate $(\alpha, \rho, \nu)$ to market smile

**β Conventions:**
- $\beta = 0$: Normal SABR (negative rates, swaptions)
- $\beta = 0.5$: CIR-like
- $\beta = 1$: Log-normal SABR (FX, caps)

### 5.2 Objective Function

$$
\min_{\alpha, \rho, \nu} \sum_{i} w_i \left(\sigma_{SABR}(K_i) - \sigma_{market}(K_i)\right)^2
$$

Subject to:
- $\alpha > 0$
- $-1 < \rho < 1$
- $\nu \geq 0$

### 5.3 Initial Guess

1. From ATM vol: $\alpha \approx \sigma_{ATM} \cdot F^{1-\beta}$
2. From skew: $\rho \approx \text{sign}(\text{skew})$
3. From curvature: $\nu \approx 0.3$

---

## 6. Special Cases

### 6.1 Log-Normal SABR (β = 1)

$$
dF_t = \sigma_t F_t dW_t^F
$$

- Simplest case
- Natural for FX
- No absorption at zero

### 6.2 Normal SABR (β = 0)

$$
dF_t = \sigma_t dW_t^F
$$

- Forward can go negative
- Natural for rates in negative rate environment
- α has dimension of rate

### 6.3 CEV-SABR (0 < β < 1)

- Interpolates between normal and log-normal
- Zero is absorbing boundary
- More complex dynamics

---

## 7. Advantages and Limitations

### 7.1 Advantages

1. **Analytic formula**: Fast implied vol computation
2. **Three free parameters**: Parsimonious fit
3. **Industry standard**: Widely understood
4. **Flexible backbone**: Different β for different markets
5. **Consistent dynamics**: Vol smile moves with forward

### 7.2 Limitations

1. **Approximation breaks down**: Deep OTM, long maturities
2. **No term structure**: Parameters constant in time
3. **No mean reversion**: Vol doesn't revert
4. **Boundary issues**: Absorption for β < 1
5. **Negative vol possible**: For extreme parameters

### 7.3 Known Issues

**Negative density:** For some parameters, the Hagan formula can produce arbitrage (negative butterfly spreads). Solutions:
- Check for arbitrage
- Use alternative formulas (e.g., Obloj)
- Use MC for problematic regions

---

## 8. Interview Key Points

### Derivation Questions

**Q: Why does SABR generate a smile?**

A: Two mechanisms:
1. **Correlation ρ**: Creates skew. Negative ρ means when F drops, σ rises, increasing OTM put vol.
2. **Vol of vol ν**: Creates curvature/kurtosis. Higher ν → more smile curvature.

**Q: What is the backbone parameter β?**

A: β controls how volatility scales with the forward:
- β=1: Percentage vol (constant σ_impl at ATM as F changes)
- β=0: Absolute vol (σ_impl increases as F drops)
- 0<β<1: Intermediate

**Q: Derive the ATM SABR formula.**

A: Starting from $\sigma_{impl} = \frac{\alpha}{(FK)^{(1-\beta)/2}} \cdot \frac{z}{x(z)} \cdot (1 + \epsilon_1 T)$:
1. At ATM: $K = F$, so $\ln(F/K) = 0$
2. z = 0, and $\lim_{z→0} z/x(z) = 1$
3. Result: $\sigma_{ATM} = \frac{\alpha}{F^{1-\beta}}(1 + \epsilon_1 T)$

### Practical Questions

**Q: How do you calibrate SABR?**

A:
1. Fix β based on convention (1 for FX, 0 for rates)
2. Initialize α from ATM vol
3. Optimize (α, ρ, ν) to minimize vol error
4. Use L-BFGS-B with bounds

**Q: When does the Hagan formula fail?**

A:
- Deep OTM strikes (|ln(F/K)| large)
- Long maturities (T > 2-3Y)
- Large ν (> 0.8)
- Near-zero forwards (for β < 1)

**Q: SABR vs Heston - when to use each?**

A:
- **SABR**: Short-dated, when smile formula is needed, rates/FX
- **Heston**: Longer-dated, when path-dependent, equities

---

## Appendix: Key Formulas

### A.1 SABR Dynamics

$$
dF = \sigma F^\beta dW^F, \quad d\sigma = \nu\sigma dW^\sigma, \quad dW^F dW^\sigma = \rho dt
$$

### A.2 ATM Implied Vol

$$
\sigma_{ATM} = \frac{\alpha}{F^{1-\beta}}\left[1 + \epsilon_1 T\right]
$$

### A.3 General Implied Vol

$$
\sigma_{impl}(K) = \frac{\alpha}{(FK)^{(1-\beta)/2}} \cdot \frac{z}{x(z)} \cdot [1 + \epsilon_1 T]
$$

### A.4 Correlation Adjustment

$$
x(z) = \ln\frac{\sqrt{1-2\rho z + z^2} + z - \rho}{1-\rho}
$$

---

## References

1. Hagan, P.S. et al. (2002). "Managing Smile Risk." *Wilmott Magazine*.

2. Obloj, J. (2008). "Fine-tune your smile: Correction to Hagan et al."

3. Rebonato, R. (2004). *Volatility and Correlation*. Wiley.

4. Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modeling*, Vol. 2. Atlantic Financial Press.

---

*Document Version: 1.0 | QuantStrata Phase 4.1 | January 2026*
