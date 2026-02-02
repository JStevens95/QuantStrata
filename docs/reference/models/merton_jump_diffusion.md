# Merton Jump-Diffusion Model

**Complete Mathematical Framework for Jump-Diffusion Pricing**

This document provides a rigorous mathematical treatment of the Merton (1976) jump-diffusion model, including derivations, proofs, pricing formulas, and numerical considerations.

---

## Table of Contents

1. [Historical Context and Motivation](#1-historical-context-and-motivation)
2. [Model Assumptions](#2-model-assumptions)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Option Pricing](#4-option-pricing)
5. [Greeks and Sensitivities](#5-greeks-and-sensitivities)
6. [Calibration](#6-calibration)
7. [Numerical Methods](#7-numerical-methods)
8. [Comparison with Other Models](#8-comparison-with-other-models)
9. [Advantages and Limitations](#9-advantages-and-limitations)
10. [Interview Key Points](#10-interview-key-points)

---

## 1. Historical Context and Motivation

### 1.1 Background

Robert Merton published "Option pricing when underlying stock returns are discontinuous" in the *Journal of Financial Economics* (1976). The model extends Black-Scholes-Merton (BSM) to include:

1. **Jump risk**: Sudden large price movements
2. **Fat tails**: Better fit to market return distributions
3. **Volatility smile**: Generates implied vol smile naturally
4. **Crash modeling**: Captures market crash scenarios

### 1.2 Why Merton Matters

The model is important because:
- **First jump-diffusion model**: Foundation for later models
- **Semi-closed form**: European options have series solution
- **Market crashes**: Can model sudden large movements
- **Smile generation**: Creates volatility smile without stochastic vol

---

## 2. Model Assumptions

### 2.1 Asset Price Dynamics

**Assumption A1: Jump-Diffusion Process**

Under the risk-neutral measure $\mathbb{Q}$, the asset price follows:

$$
\frac{dS_t}{S_t} = (r - q - \lambda\kappa) dt + \sigma dW_t + (J - 1) dN_t
$$

Where:
- $S_t$: Asset price
- $r$: Risk-free rate
- $q$: Dividend yield
- $\sigma$: Diffusion volatility
- $W_t$: Standard Brownian motion
- $N_t$: Poisson process with intensity $\lambda$
- $J$: Jump multiplier, $J = e^Y$ where $Y \sim N(\mu_J, \sigma_J^2)$
- $\kappa = \mathbb{E}[J - 1] = e^{\mu_J + \sigma_J^2/2} - 1$

### 2.2 Model Parameters

| Parameter | Symbol | Typical Range | Interpretation |
|-----------|--------|---------------|----------------|
| Diffusion vol | $\sigma$ | 10-40% | Continuous price volatility |
| Jump intensity | $\lambda$ | 0-3/year | Expected jumps per year |
| Log-jump mean | $\mu_J$ | -20% to +10% | Average log-jump size |
| Log-jump std | $\sigma_J$ | 5-30% | Jump size uncertainty |

### 2.3 Model Assumptions

| Assumption | Mathematical Statement | Reality |
|------------|----------------------|---------|
| **A2: Independent jumps** | $N_t, W_t$ independent | May be correlated |
| **A3: Log-normal jumps** | $\ln J \sim N(\mu_J, \sigma_J^2)$ | May have other distributions |
| **A4: Constant parameters** | All parameters constant | May vary with time |
| **A5: No jump clustering** | Jumps are i.i.d. | Jumps may cluster |

---

## 3. Mathematical Framework

### 3.1 Solution to the SDE

**Theorem:** The solution to the Merton SDE is:

$$
S_T = S_0 \exp\left[(r - q - \lambda\kappa - \frac{\sigma^2}{2})T + \sigma W_T + \sum_{i=1}^{N_T} Y_i\right]
$$

where $N_T \sim \text{Poisson}(\lambda T)$ and $Y_i \sim N(\mu_J, \sigma_J^2)$ i.i.d.

**Proof:**

Apply Itô's lemma to $\ln S_t$ between jumps:
$$
d(\ln S_t) = (r - q - \lambda\kappa - \frac{\sigma^2}{2})dt + \sigma dW_t
$$

At jump times, $\ln S$ jumps by $Y_i = \ln J_i$. Integrating:
$$
\ln S_T = \ln S_0 + (r - q - \lambda\kappa - \frac{\sigma^2}{2})T + \sigma W_T + \sum_{i=1}^{N_T} Y_i
$$

Exponentiating gives the result. $\square$

### 3.2 Distribution of Returns

**Theorem:** The log-return $\ln(S_T/S_0)$ has a mixture distribution:

$$
\ln\frac{S_T}{S_0} | N_T = n \sim N\left(\mu_n, \sigma_n^2\right)
$$

where:
$$
\mu_n = (r - q - \lambda\kappa - \frac{\sigma^2}{2})T + n\mu_J
$$
$$
\sigma_n^2 = \sigma^2 T + n\sigma_J^2
$$

### 3.3 Characteristic Function

**Theorem:** The characteristic function of $X_T = \ln(S_T/S_0)$ is:

$$
\phi_{X_T}(u) = \exp\left[iu\left(r - q - \lambda\kappa - \frac{\sigma^2}{2}\right)T - \frac{u^2\sigma^2 T}{2} + \lambda T\left(e^{iu\mu_J - u^2\sigma_J^2/2} - 1\right)\right]
$$

**Proof:**

Use the law of total expectation conditioning on $N_T$:
$$
\phi_{X_T}(u) = \mathbb{E}[e^{iuX_T}] = \mathbb{E}[\mathbb{E}[e^{iuX_T} | N_T]]
$$

Given $N_T = n$, $X_T$ is Gaussian with mean $\mu_n$ and variance $\sigma_n^2$:
$$
\mathbb{E}[e^{iuX_T} | N_T = n] = e^{iu\mu_n - u^2\sigma_n^2/2}
$$

Since $N_T \sim \text{Poisson}(\lambda T)$:
$$
\phi_{X_T}(u) = \sum_{n=0}^{\infty} e^{-\lambda T}\frac{(\lambda T)^n}{n!} e^{iu\mu_n - u^2\sigma_n^2/2}
$$

After algebraic simplification, this yields the result. $\square$

---

## 4. Option Pricing

### 4.1 European Call Option Formula

**Theorem (Merton, 1976):** The European call option price is:

$$
\boxed{C = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!} C_{BS}(S, K, T, r_n, q, \sigma_n)}
$$

where:
- $\lambda' = \lambda \cdot \mathbb{E}[J] = \lambda e^{\mu_J + \sigma_J^2/2}$
- $r_n = r - \lambda\kappa + n\ln(\mathbb{E}[J])/T$
- $\sigma_n^2 = \sigma^2 + n\sigma_J^2/T$
- $C_{BS}$ is the Black-Scholes call formula

**Proof:**

1. Condition on number of jumps:
$$
C = e^{-rT}\mathbb{E}^{\mathbb{Q}}[(S_T - K)^+] = e^{-rT}\sum_{n=0}^{\infty} P(N_T = n) \mathbb{E}[(S_T - K)^+ | N_T = n]
$$

2. Given $N_T = n$, $S_T$ is log-normal with modified parameters.

3. Each conditional expectation is a Black-Scholes price with adjusted parameters.

4. The sum of terms gives the Merton formula. $\square$

### 4.2 Series Convergence

**Proposition:** The series converges rapidly for typical parameters.

For $n > 20$, terms are typically $< 10^{-10}$ of the total. In practice, summing 50 terms is sufficient.

### 4.3 Put-Call Parity

Standard put-call parity holds:
$$
C - P = S e^{-qT} - K e^{-rT}
$$

### 4.4 Monte Carlo Pricing

For path-dependent options, use simulation:

1. Generate $N_T \sim \text{Poisson}(\lambda T)$
2. If $N_T > 0$: Generate $\sum_{i=1}^{N_T} Y_i \sim N(N_T \mu_J, N_T \sigma_J^2)$
3. Generate $\sigma W_T \sim N(0, \sigma^2 T)$
4. Compute $S_T = S_0 \exp[(r - q - \lambda\kappa - \sigma^2/2)T + \sigma W_T + \sum Y_i]$

---

## 5. Greeks and Sensitivities

### 5.1 Delta

$$
\Delta = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!} \Delta_{BS,n}
$$

where $\Delta_{BS,n}$ is the Black-Scholes delta with parameters $(r_n, \sigma_n)$.

### 5.2 Gamma

$$
\Gamma = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!} \Gamma_{BS,n}
$$

### 5.3 Vega (Diffusion)

$$
\mathcal{V}_\sigma = \frac{\partial C}{\partial \sigma} = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!} \frac{\sigma}{\sigma_n} \mathcal{V}_{BS,n}
$$

### 5.4 Jump Sensitivities

**Sensitivity to jump intensity:**
$$
\frac{\partial C}{\partial \lambda} \text{ (complex - requires numerical differentiation)}
$$

---

## 6. Calibration

### 6.1 Calibration Strategy

**Objective:** Fit model to market option prices across strikes and maturities.

**Parameters to calibrate:**
- $\sigma$: Diffusion volatility
- $\lambda$: Jump intensity
- $\mu_J$: Mean log-jump
- $\sigma_J$: Std dev of log-jump

**Typical approach:**
1. Fix $\sigma$ from ATM implied vol
2. Calibrate $(\lambda, \mu_J, \sigma_J)$ to smile

### 6.2 Implied Volatility Smile

The Merton model generates a volatility smile characterized by:
- **Skew**: Negative $\mu_J$ creates negative skew
- **Kurtosis**: Higher $\lambda$ or $\sigma_J$ increases kurtosis
- **Term structure**: Smile flattens with maturity

### 6.3 Moment Matching

Initial guess from matching first four moments of market-implied distribution:
- Mean: $(r - q - \sigma^2/2)T + \lambda T \kappa$
- Variance: $\sigma^2 T + \lambda T(\mu_J^2 + \sigma_J^2)$
- Skewness: From third moment
- Kurtosis: From fourth moment

---

## 7. Numerical Methods

### 7.1 Path Simulation

**Exact Terminal Simulation:**
```
For each path:
    1. Draw N ~ Poisson(λT)
    2. Draw Z ~ N(0, 1)
    3. If N > 0: Draw Y ~ N(N·μ_J, N·σ_J²)
    4. S_T = S_0 × exp((r-q-λκ-σ²/2)T + σ√T·Z + Y)
```

**Step-by-Step Simulation:**
```
For each time step dt:
    1. Draw Z ~ N(0, 1) for diffusion
    2. Draw N ~ Poisson(λdt) for jumps
    3. If N > 0: Draw Y ~ N(N·μ_J, N·σ_J²)
    4. S_{t+dt} = S_t × exp((r-q-λκ-σ²/2)dt + σ√dt·Z + Y)
```

### 7.2 FFT Pricing

For faster European pricing:
1. Compute characteristic function
2. Apply Carr-Madan or Lewis formula
3. Use FFT for efficient integration

### 7.3 Finite Difference (PIDE)

Merton satisfies a Partial Integro-Differential Equation:
$$
\frac{\partial V}{\partial t} + \frac{\sigma^2 S^2}{2}\frac{\partial^2 V}{\partial S^2} + (r - q - \lambda\kappa)S\frac{\partial V}{\partial S} - rV + \lambda \mathbb{E}[V(SJ) - V(S)] = 0
$$

The integral term requires special treatment in finite difference schemes.

---

## 8. Comparison with Other Models

### 8.1 Merton vs Black-Scholes

| Aspect | Black-Scholes | Merton |
|--------|--------------|--------|
| Dynamics | Pure diffusion | Diffusion + jumps |
| Returns | Gaussian | Mixture of Gaussians |
| Tails | Thin | Fat (controllable) |
| Smile | Flat | Smile/skew |
| Closed form | Yes | Series |
| Parameters | 1 (σ) | 4 (σ, λ, μ_J, σ_J) |

### 8.2 Merton vs Heston

| Aspect | Merton | Heston |
|--------|--------|--------|
| Mechanism | Jumps | Stochastic vol |
| Smile origin | Jump sizes | Vol correlation |
| Short-term smile | Strong | Weak |
| Long-term smile | Weak | Strong |
| Path-dependency | Harder | Easier |

---

## 9. Advantages and Limitations

### 9.1 Advantages

1. **Fat tails**: Naturally captures heavy tails
2. **Crash modeling**: Can model sudden large moves
3. **Semi-closed form**: Series solution for Europeans
4. **Intuitive parameters**: Jump frequency/size interpretable
5. **Smile generation**: Creates volatility smile

### 9.2 Limitations

1. **Independent jumps**: Cannot model jump clustering
2. **No mean reversion**: Vol doesn't revert
3. **Parameter sensitivity**: Many parameters to calibrate
4. **PIDE complexity**: FD methods harder than PDE
5. **Jump risk premium**: Requires assumption on market price of jump risk

---

## 10. Interview Key Points

### Derivation Questions

**Q: Derive the Merton call option formula.**

A:
1. Condition on $N_T = n$ jumps
2. Given $n$ jumps, $S_T$ is log-normal with mean $(r-q-\lambda\kappa-\sigma^2/2)T + n\mu_J$ and variance $\sigma^2 T + n\sigma_J^2$
3. Each conditional expectation is a Black-Scholes call
4. Sum over Poisson probabilities with modified intensity $\lambda' = \lambda e^{\mu_J + \sigma_J^2/2}$

**Q: Why does Merton generate a volatility smile?**

A: The mixture of log-normals (one for each number of jumps) creates heavier tails than a single log-normal. This maps to higher implied vols for OTM options. Negative $\mu_J$ creates left skew (higher OTM put vols).

**Q: What is the drift adjustment $\lambda\kappa$?**

A: The term $\lambda\kappa = \lambda(\mathbb{E}[J] - 1)$ compensates for the expected change due to jumps. Without it, $\mathbb{E}[S_T] \neq S_0 e^{(r-q)T}$, violating the martingale condition.

### Practical Questions

**Q: How do you calibrate Merton?**

A:
1. Set $\sigma$ from ATM implied vol (approximate)
2. Use optimization to fit $(\lambda, \mu_J, \sigma_J)$ to OTM options
3. Objective: minimize sum of squared implied vol errors
4. Constraints: $\sigma_J > 0$, $\lambda \geq 0$

**Q: Merton vs Heston - when to use each?**

A:
- **Merton**: Short-dated options, crash protection, explicit jump modeling
- **Heston**: Longer-dated options, vol surface fitting, path-dependent exotics

---

## Appendix: Key Formulas

### A.1 Asset Price Dynamics

$$
\frac{dS_t}{S_t} = (r - q - \lambda\kappa)dt + \sigma dW_t + (J-1)dN_t
$$

### A.2 Drift Correction

$$
\kappa = \mathbb{E}[J - 1] = e^{\mu_J + \sigma_J^2/2} - 1
$$

### A.3 European Call Price

$$
C = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!} C_{BS}(S, K, T, r_n, q, \sigma_n)
$$

### A.4 Adjusted Parameters

$$
\lambda' = \lambda e^{\mu_J + \sigma_J^2/2}, \quad r_n = r - \lambda\kappa + \frac{n\ln\mathbb{E}[J]}{T}, \quad \sigma_n = \sqrt{\sigma^2 + \frac{n\sigma_J^2}{T}}
$$

---

## References

1. Merton, R.C. (1976). "Option pricing when underlying stock returns are discontinuous." *Journal of Financial Economics*.

2. Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.

3. Kou, S.G. (2002). "A jump-diffusion model for option pricing." *Management Science*.

4. Andersen, L. & Andreasen, J. (2000). "Jump-Diffusion Processes: Volatility Smile Fitting and Numerical Methods for Option Pricing." *Review of Derivatives Research*.

---

*Document Version: 1.0 | QuantStrata Phase 4.1 | January 2026*
