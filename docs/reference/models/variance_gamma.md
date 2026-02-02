# Variance Gamma Model

**Complete Mathematical Framework for Lévy Process-Based Pricing**

This document provides a rigorous mathematical treatment of the Variance Gamma (VG) model, including derivations, proofs, and numerical considerations.

---

## Table of Contents

1. [Historical Context and Motivation](#1-historical-context-and-motivation)
2. [Model Assumptions](#2-model-assumptions)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Option Pricing](#4-option-pricing)
5. [Calibration](#5-calibration)
6. [Comparison with Other Models](#6-comparison-with-other-models)
7. [Advantages and Limitations](#7-advantages-and-limitations)
8. [Interview Key Points](#8-interview-key-points)

---

## 1. Historical Context and Motivation

### 1.1 Background

The Variance Gamma process was introduced by Madan, Carr, and Chang (1998) as a pure-jump Lévy process alternative to GBM. Key features:

1. **Pure jump process**: No diffusion component
2. **Time-changed Brownian motion**: Simple construction
3. **Fat tails**: Controllable kurtosis via ν parameter
4. **Skewness**: Controllable via θ parameter

### 1.2 Why Variance Gamma Matters

- **Three parameters**: Parsimonious fit to market smiles
- **Finite activity**: Easier to simulate than infinite activity processes
- **Closed-form characteristic function**: Enables FFT pricing
- **Analytical Greeks**: For European options

---

## 2. Model Assumptions

### 2.1 Process Definition

**Subordination Construction:**

The VG process $X_t$ is defined as:

$$
\boxed{X_t = \theta G_t + \sigma W_{G_t}}
$$

where:
- $G_t \sim \text{Gamma}(t/\nu, \nu)$: The "business time" process
- $W_t$: Standard Brownian motion
- $\theta$: Drift parameter (controls skewness)
- $\sigma$: Volatility parameter
- $\nu$: Variance rate of time (controls kurtosis)

### 2.2 Parameter Interpretation

| Parameter | Symbol | Effect | Typical Values |
|-----------|--------|--------|----------------|
| Drift | $\theta$ | Skewness direction | -0.3 to +0.1 |
| Volatility | $\sigma$ | Spread | 0.1 to 0.4 |
| Variance rate | $\nu$ | Kurtosis (fat tails) | 0.1 to 1.0 |

**Sign of θ:**
- $\theta < 0$: Negative skew (equity-like)
- $\theta > 0$: Positive skew
- $\theta = 0$: Symmetric

---

## 3. Mathematical Framework

### 3.1 Characteristic Function

**Theorem:** The characteristic function of $X_t$ is:

$$
\boxed{\phi_{X_t}(u) = \left(1 - iu\theta\nu + \frac{\sigma^2\nu u^2}{2}\right)^{-t/\nu}}
$$

**Proof:**

The VG process is constructed as $X_t = \theta G_t + \sigma W_{G_t}$.

Conditioning on $G_t = g$:
$$
X_t | G_t = g \sim N(\theta g, \sigma^2 g)
$$

Therefore:
$$
\mathbb{E}[e^{iuX_t} | G_t = g] = e^{iu\theta g - \frac{u^2\sigma^2 g}{2}}
$$

Taking expectations over $G_t \sim \text{Gamma}(t/\nu, \nu)$:
$$
\phi_{X_t}(u) = \mathbb{E}\left[e^{iu\theta G_t - \frac{u^2\sigma^2 G_t}{2}}\right] = \mathbb{E}\left[e^{G_t(iu\theta - \frac{u^2\sigma^2}{2})}\right]
$$

Using the Gamma MGF $\mathbb{E}[e^{sG}] = (1 - \nu s)^{-t/\nu}$:
$$
\phi_{X_t}(u) = \left(1 - \nu\left(iu\theta - \frac{u^2\sigma^2}{2}\right)\right)^{-t/\nu}
$$
$\square$

### 3.2 Moments of $X_t$

**Mean:**
$$
\mathbb{E}[X_t] = \theta t
$$

**Variance:**
$$
\text{Var}(X_t) = (\sigma^2 + \theta^2\nu)t
$$

**Skewness:**
$$
\text{Skew}(X_1) = \frac{\theta(3\sigma^2\nu + 2\theta^2\nu^2)}{(\sigma^2 + \theta^2\nu)^{3/2}}
$$

**Excess Kurtosis:**
$$
\text{Kurt}(X_1) - 3 = \frac{3\sigma^4\nu + 12\sigma^2\theta^2\nu^2 + 6\theta^4\nu^3}{(\sigma^2 + \theta^2\nu)^2}
$$

### 3.3 Martingale Correction

For asset price modeling under the risk-neutral measure:

$$
S_t = S_0 \exp\left[(\mu + \omega)t + X_t\right]
$$

where $\mu = r - q$ and the martingale correction is:

$$
\boxed{\omega = \frac{1}{\nu}\ln\left(1 - \theta\nu - \frac{\sigma^2\nu}{2}\right)}
$$

**Proof:**

For $\mathbb{E}[S_t] = S_0 e^{(r-q)t}$, we need $\mathbb{E}[e^{\omega t + X_t}] = 1$.

Using the characteristic function at $u = -i$:
$$
\mathbb{E}[e^{X_t}] = \phi_{X_t}(-i) = \left(1 - \theta\nu - \frac{\sigma^2\nu}{2}\right)^{-t/\nu}
$$

Setting $e^{\omega t} \cdot \phi_{X_t}(-i) = 1$ gives the result. $\square$

**Constraint:** Must have $1 - \theta\nu - \sigma^2\nu/2 > 0$.

### 3.4 Lévy Measure

The VG process has Lévy measure:

$$
\nu_{VG}(dx) = \frac{C}{|x|}\exp\left(-\frac{|x|}{G^{\pm}}\right)dx
$$

where:
- $C = 1/\nu$
- $G^+ = \sqrt{\theta^2\nu^2/4 + \sigma^2\nu/2} - \theta\nu/2$ (positive jumps)
- $G^- = \sqrt{\theta^2\nu^2/4 + \sigma^2\nu/2} + \theta\nu/2$ (negative jumps)

---

## 4. Option Pricing

### 4.1 FFT Pricing (Carr-Madan)

European options can be priced via:

$$
C(K) = \frac{e^{-\alpha \ln K}}{\pi} \int_0^\infty \text{Re}\left[e^{-iv\ln K} \psi_T(v)\right] dv
$$

where:
$$
\psi_T(v) = \frac{e^{-rT}\phi_{\ln S_T}(v - (1+\alpha)i)}{\alpha^2 + \alpha - v^2 + i(2\alpha + 1)v}
$$

### 4.2 Monte Carlo Simulation

**Exact Terminal Simulation:**

```
For each path:
    1. Draw G_T ~ Gamma(T/ν, ν)
    2. Draw Z ~ N(0, 1)
    3. X_T = θ·G_T + σ·√(G_T)·Z
    4. S_T = S_0·exp((μ + ω)T + X_T)
```

**Path Simulation:**

```
For each time step dt:
    1. Draw ΔG ~ Gamma(dt/ν, ν)
    2. Draw Z ~ N(0, 1)
    3. ΔX = θ·ΔG + σ·√(ΔG)·Z
    4. S_{t+dt} = S_t·exp((μ + ω)dt + ΔX)
```

### 4.3 Greeks

**Delta:**
$$
\Delta = e^{-qT}\mathbb{E}^{\mathbb{Q}}\left[\mathbf{1}_{S_T > K}\right] + \text{adjustment}
$$

(Computed via FFT or finite difference on MC price)

---

## 5. Calibration

### 5.1 Moment Matching

Initial guess from matching market moments:
1. ATM implied vol → $\sigma$ (approximate)
2. Skew → $\theta$ sign
3. Kurtosis → $\nu$

### 5.2 Optimization

**Objective:**
$$
\min_{\theta, \sigma, \nu} \sum_{i} w_i \left(\sigma_{model}^{impl}(K_i, T_i) - \sigma_{market}^{impl}(K_i, T_i)\right)^2
$$

**Constraints:**
- $\sigma > 0$
- $\nu > 0$
- $1 - \theta\nu - \sigma^2\nu/2 > 0$ (well-defined $\omega$)

---

## 6. Comparison with Other Models

### 6.1 VG vs Black-Scholes

| Aspect | Black-Scholes | Variance Gamma |
|--------|--------------|----------------|
| Process | Diffusion | Pure jump |
| Tails | Thin (Gaussian) | Fat (controllable) |
| Skewness | Zero | Controllable |
| Smile | Flat | Smile/skew |
| Parameters | 1 | 3 |

### 6.2 VG vs Merton Jump-Diffusion

| Aspect | Merton | Variance Gamma |
|--------|--------|----------------|
| Jumps | Compound Poisson | Infinite activity |
| Diffusion | Yes | No |
| Kurtosis control | Via jump size | Via ν |
| Complexity | 4 parameters | 3 parameters |

### 6.3 VG vs NIG/CGMY

| Aspect | VG | NIG | CGMY |
|--------|----|----|------|
| Activity | Finite | Infinite | Parameterized |
| Paths | Finite variation | Infinite variation | Depends on Y |
| Parameters | 3 | 4 | 4 |
| Simulation | Easy | Harder | Harder |

---

## 7. Advantages and Limitations

### 7.1 Advantages

1. **Parsimonious**: Only 3 parameters
2. **Fat tails and skew**: Independent control
3. **Closed-form characteristic function**: Efficient pricing
4. **Simple simulation**: Exact terminal distribution
5. **Finite variation paths**: No diffusion approximation needed

### 7.2 Limitations

1. **No diffusion component**: May underfit short-term ATM
2. **Stationary increments**: No term structure in parameters
3. **Symmetric in jump sizes**: May not fit extreme skews
4. **Finite activity**: May not capture high-frequency dynamics

---

## 8. Interview Key Points

### Derivation Questions

**Q: How is the VG process constructed?**

A: As a time-changed Brownian motion: $X_t = \theta G_t + \sigma W_{G_t}$ where $G_t$ is a Gamma process. The Gamma process acts as a "business clock" that randomly speeds up and slows down, creating fat tails.

**Q: Derive the characteristic function of VG.**

A:
1. Condition on $G_t = g$: $X_t | G_t = g \sim N(\theta g, \sigma^2 g)$
2. $\mathbb{E}[e^{iuX_t}|G_t=g] = e^{iu\theta g - u^2\sigma^2 g/2}$
3. Take expectation over $G_t \sim \text{Gamma}(t/\nu, \nu)$
4. Use Gamma MGF: result is $(1 - iu\theta\nu + \sigma^2\nu u^2/2)^{-t/\nu}$

**Q: What is the martingale correction ω?**

A: $\omega = \frac{1}{\nu}\ln(1 - \theta\nu - \sigma^2\nu/2)$ ensures $\mathbb{E}[S_T] = S_0 e^{(r-q)T}$ under the risk-neutral measure.

### Practical Questions

**Q: How do you simulate VG paths?**

A: Use subordination:
1. Simulate Gamma increments $\Delta G \sim \text{Gamma}(dt/\nu, \nu)$
2. Simulate standard normals $Z$
3. VG increment: $\Delta X = \theta \Delta G + \sigma \sqrt{\Delta G} \cdot Z$

**Q: VG vs Merton - when to use each?**

A:
- **VG**: When you want a pure-jump model with controlled kurtosis, simpler calibration
- **Merton**: When you want explicit diffusion component, modeling specific crash scenarios

---

## Appendix: Key Formulas

### A.1 Process Definition

$$
X_t = \theta G_t + \sigma W_{G_t}, \quad G_t \sim \text{Gamma}(t/\nu, \nu)
$$

### A.2 Characteristic Function

$$
\phi_{X_t}(u) = \left(1 - iu\theta\nu + \frac{\sigma^2\nu u^2}{2}\right)^{-t/\nu}
$$

### A.3 Martingale Correction

$$
\omega = \frac{1}{\nu}\ln\left(1 - \theta\nu - \frac{\sigma^2\nu}{2}\right)
$$

### A.4 Variance

$$
\text{Var}(X_t) = (\sigma^2 + \theta^2\nu)t
$$

---

## References

1. Madan, D.B., Carr, P., & Chang, E.C. (1998). "The Variance Gamma Process and Option Pricing." *European Finance Review*.

2. Carr, P. & Madan, D.B. (1999). "Option Valuation Using the Fast Fourier Transform." *Journal of Computational Finance*.

3. Madan, D.B. & Seneta, E. (1990). "The Variance Gamma (V.G.) Model for Share Market Returns." *Journal of Business*.

4. Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.

---

*Document Version: 1.0 | QuantStrata Phase 4.1 | January 2026*
