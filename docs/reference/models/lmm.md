# LIBOR Market Model (BGM Model)

**Complete Mathematical Framework for Multi-Factor Forward Rate Modeling**

This document provides a rigorous mathematical treatment of the LIBOR Market Model, including derivations, proofs, measure theory, discretization schemes, and calibration theory for quantitative finance.

---

## Table of Contents

1. [Historical Context and Motivation](#1-historical-context-and-motivation)
2. [Model Assumptions](#2-model-assumptions)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Measure Theory and Drift Derivation](#4-measure-theory-and-drift-derivation)
5. [Correlation Structure](#5-correlation-structure)
6. [Discretization Schemes](#6-discretization-schemes)
7. [Product Pricing](#7-product-pricing)
8. [Calibration Theory](#8-calibration-theory)
9. [Numerical Considerations](#9-numerical-considerations)
10. [Comparison with Other Models](#10-comparison-with-other-models)
11. [Advantages and Limitations](#11-advantages-and-limitations)
12. [Interview Key Points](#12-interview-key-points)

---

## 1. Historical Context and Motivation

### 1.1 Background

The LIBOR Market Model was independently developed by Brace, Gatarek, and Musiela (1997) and Jamshidian (1997), earning it the name "BGM Model." It arose from the need to:

1. **Model observable rates** - Unlike short-rate models, LMM directly models forward LIBOR rates
2. **Exact cap calibration** - Reproduces Black's caplet formula by construction
3. **Multi-factor dynamics** - Captures term structure movements beyond parallel shifts
4. **Market consistency** - Forward rates are the underlying for most IR derivatives

### 1.2 Why LMM Matters

The model is the industry standard for pricing exotic interest rate derivatives because:

- **Cap/floor calibration is trivial** - Model volatilities = market Black vols
- **Correlation flexibility** - Rich correlation structure for swaption pricing
- **Forward rates are observable** - State variables are directly quoted in markets
- **Exotic pricing** - Natural framework for path-dependent IR products
- **Smile handling** - Extensions (SABR-LMM, shifted LMM) handle volatility smiles

### 1.3 Evolution of the Model

| Year | Development |
|------|------------|
| 1997 | BGM and Jamshidian publish original papers |
| 1998 | Rebonato develops swaption approximation |
| 2002 | SABR-LMM extensions for smile |
| 2008+ | Shifted LMM for negative rates |
| 2012+ | RFR transition: SOFR/SONIA adaptations |

---

## 2. Model Assumptions

### 2.1 State Variables

The model tracks $n$ forward LIBOR rates $F_1(t), F_2(t), \ldots, F_n(t)$ spanning the tenor structure:

$$
T_0 < T_1 < T_2 < \cdots < T_n
$$

**Notation:**
- $F_i(t)$: Forward rate for accrual period $[T_i, T_{i+1}]$ at time $t$
- $\tau_i = T_{i+1} - T_i$: Accrual fraction (day count)
- $P(t, T)$: Zero-coupon bond price at time $t$ for maturity $T$

### 2.2 Forward Rate Definition

**No-Arbitrage Relationship:**

$$
\boxed{F_i(t) = \frac{1}{\tau_i}\left(\frac{P(t, T_i)}{P(t, T_{i+1})} - 1\right)}
$$

Equivalently:

$$
P(t, T_{i+1}) = \frac{P(t, T_i)}{1 + \tau_i F_i(t)}
$$

### 2.3 Model Assumptions

| Assumption | Mathematical Statement | Reality |
|------------|----------------------|---------|
| **A1: Log-normal forwards** | $dF_i/F_i = \mu_i dt + \sigma_i dW_i$ | Can produce unrealistic extremes |
| **A2: Positive rates** | $F_i(t) > 0$ always | Violated in EUR/CHF/JPY post-2014 |
| **A3: Continuous trading** | No jumps in forward rates | Rates can jump |
| **A4: Perfect correlation** | Specified $\rho_{ij}$ matrix | Correlation is stochastic |
| **A5: Deterministic volatility** | $\sigma_i(t)$ known functions | Volatility is stochastic |

### 2.4 Key Difference from Short-Rate Models

| Aspect | LMM | Hull-White / BK |
|--------|-----|-----------------|
| **State variable** | Forward rates $F_i(t)$ | Short rate $r(t)$ |
| **Dimension** | $n$ forwards (high) | 1 (low) |
| **Observability** | Directly quoted | Inferred from curve |
| **Caplet pricing** | Exact (Black's formula) | Approximate |
| **Bond pricing** | Recursive from forwards | Closed-form (HW) |

---

## 3. Mathematical Framework

### 3.1 Forward Rate Dynamics (General Form)

Under any equivalent martingale measure, the forward rate $F_i(t)$ follows:

$$
\frac{dF_i(t)}{F_i(t)} = \mu_i(t) dt + \sigma_i(t) dW_i(t)
$$

where:
- $\mu_i(t)$: Drift (measure-dependent)
- $\sigma_i(t)$: Instantaneous volatility
- $dW_i(t)$: Brownian increment with correlation $d\langle W_i, W_j \rangle = \rho_{ij} dt$

### 3.2 The Log-Normal Property

**Theorem:** Under the $T_{i+1}$-forward measure $\mathbb{Q}^{i+1}$, the forward rate $F_i(t)$ is a **martingale**, hence:

$$
F_i(T_i) | F_i(0) \sim \text{LogNormal}\left(\ln F_i(0) - \frac{1}{2}\Sigma_i^2, \Sigma_i^2\right)
$$

where $\Sigma_i^2 = \int_0^{T_i} \sigma_i(s)^2 ds$.

**Proof:**

Under $\mathbb{Q}^{i+1}$ with numeraire $P(t, T_{i+1})$, the discounted forward $\tilde{F}_i(t) = F_i(t) \cdot 1$ (since forward has zero initial investment) must be a martingale by the First Fundamental Theorem.

Therefore:
$$
\frac{dF_i(t)}{F_i(t)} = \sigma_i(t) dW_i^{i+1}(t)
$$

with zero drift. Applying Itô's lemma to $\ln F_i(t)$:
$$
d\ln F_i(t) = -\frac{1}{2}\sigma_i(t)^2 dt + \sigma_i(t) dW_i^{i+1}(t)
$$

Integrating:
$$
\ln F_i(T_i) = \ln F_i(0) - \frac{1}{2}\int_0^{T_i}\sigma_i(s)^2 ds + \int_0^{T_i}\sigma_i(s)dW_i^{i+1}(s)
$$

The stochastic integral is Gaussian with variance $\Sigma_i^2$, proving log-normality. $\square$

### 3.3 Forward Price Connection

**Recursive Bond Pricing:**

$$
P(t, T_j) = P(t, T_i) \prod_{k=i}^{j-1} \frac{1}{1 + \tau_k F_k(t)}
$$

**Proof:** Apply the forward rate definition iteratively:
$$
P(t, T_{i+1}) = \frac{P(t, T_i)}{1 + \tau_i F_i(t)}, \quad P(t, T_{i+2}) = \frac{P(t, T_{i+1})}{1 + \tau_{i+1} F_{i+1}(t)}, \ldots
$$

Multiplying:
$$
P(t, T_j) = P(t, T_i) \cdot \frac{1}{1 + \tau_i F_i(t)} \cdot \frac{1}{1 + \tau_{i+1} F_{i+1}(t)} \cdots \frac{1}{1 + \tau_{j-1} F_{j-1}(t)}
$$
$\square$

---

## 4. Measure Theory and Drift Derivation

### 4.1 The Choice of Numeraire

Three common measures are used in LMM:

| Measure | Numeraire | Symbol | Use Case |
|---------|-----------|--------|----------|
| Terminal | $P(t, T_N)$ | $\mathbb{Q}^N$ | Backward induction |
| Spot | $B(t) = \prod_{j: T_j \leq t}(1 + \tau_j F_j(T_j))$ | $\mathbb{Q}^B$ | Forward simulation |
| $T_k$-forward | $P(t, T_k)$ | $\mathbb{Q}^k$ | Product fixing at $T_k$ |

### 4.2 Terminal Measure Dynamics

**Theorem:** Under the terminal measure $\mathbb{Q}^N$ with numeraire $P(t, T_N)$:

$$
\boxed{\frac{dF_i(t)}{F_i(t)} = -\sum_{j=i+1}^{N-1} \frac{\rho_{ij} \sigma_i(t) \sigma_j(t) \tau_j F_j(t)}{1 + \tau_j F_j(t)} dt + \sigma_i(t) dW_i^N(t)}
$$

**Proof:**

1. Under $\mathbb{Q}^N$, the deflated asset $\tilde{F}_i(t) = F_i(t) / P(t, T_N)$ need not be a martingale since $F_i$ is not an asset.

2. However, $F_i(t) = \frac{1}{\tau_i}(P(t, T_i)/P(t, T_{i+1}) - 1)$ involves bond ratios.

3. Under $\mathbb{Q}^N$, bonds $P(t, T_j)/P(t, T_N)$ are martingales.

4. Using Itô's product rule and the forward rate dynamics, the drift emerges from the correlation between $F_i$ and the bonds $P(t, T_j)$ for $j > i$.

5. Each bond $P(t, T_j)$ depends on forwards $F_i, F_{i+1}, \ldots, F_{j-1}$, creating the sum of drift adjustments. $\square$

### 4.3 Spot Measure Dynamics

**Theorem:** Under the spot measure $\mathbb{Q}^B$ with numeraire $B(t)$:

$$
\boxed{\frac{dF_i(t)}{F_i(t)} = \sum_{j=\beta(t)}^{i} \frac{\rho_{ij} \sigma_i(t) \sigma_j(t) \tau_j F_j(t)}{1 + \tau_j F_j(t)} dt + \sigma_i(t) dW_i^B(t)}
$$

where $\beta(t) = \min\{j : T_j > t\}$ is the index of the first unfixed forward at time $t$.

**Derivation via Girsanov:**

1. Start from the terminal measure dynamics.

2. The Radon-Nikodym derivative from $\mathbb{Q}^N$ to $\mathbb{Q}^B$ is:
$$
\frac{d\mathbb{Q}^B}{d\mathbb{Q}^N}\bigg|_t = \frac{B(t) P(0, T_N)}{B(0) P(t, T_N)}
$$

3. By Girsanov's theorem:
$$
dW_i^B(t) = dW_i^N(t) + \gamma_i(t) dt
$$

4. The Girsanov kernel $\gamma_i(t)$ accounts for the measure change:
$$
\gamma_i(t) = \sum_{j=\beta(t)}^{N-1} \frac{\rho_{ij} \sigma_j(t) \tau_j F_j(t)}{1 + \tau_j F_j(t)}
$$

5. Substituting:
$$
\frac{dF_i(t)}{F_i(t)} = \left[-\sum_{j=i+1}^{N-1}(\cdots) + \sigma_i(t)\gamma_i(t)\right]dt + \sigma_i(t)dW_i^B(t)
$$

6. After simplification (using $\rho_{ij} = \rho_{ji}$):
$$
\mu_i^B(t) = \sum_{j=\beta(t)}^{i} \frac{\rho_{ij} \sigma_i(t) \sigma_j(t) \tau_j F_j(t)}{1 + \tau_j F_j(t)}
$$
$\square$

### 4.4 Drift Structure Analysis

**Key Observations:**

1. **Terminal measure:** Drift is **negative** (sum from $i+1$ to $N-1$)
2. **Spot measure:** Drift is **positive** (sum from $\beta(t)$ to $i$)
3. **Forward-$k$ measure:** $F_k$ has zero drift; others adjust accordingly

**Why Spot Measure for Simulation:**

- Forwards fix sequentially: $F_0$ at $T_0$, $F_1$ at $T_1$, etc.
- Under spot measure, only "alive" forwards ($j \geq \beta(t)$) appear in drift
- Natural for forward-stepping Monte Carlo

---

## 5. Correlation Structure

### 5.1 Instantaneous Correlation

Forward rate Brownian motions are correlated:

$$
d\langle W_i, W_j \rangle_t = \rho_{ij} dt
$$

**Requirements:**
- $\rho_{ii} = 1$
- $\rho_{ij} = \rho_{ji}$ (symmetry)
- $\boldsymbol{\rho}$ is positive semi-definite

### 5.2 Parametric Correlation Forms

**Flat Correlation:**
$$
\rho_{ij} = \rho \quad \forall i \neq j
$$

Simple but unrealistic. PSD requires $\rho \geq -1/(n-1)$.

**Exponential Decay (Rebonato):**
$$
\boxed{\rho_{ij} = \exp(-\beta |T_i - T_j|)}
$$

- $\beta > 0$: Decorrelation speed
- Captures: Nearby forwards more correlated than distant ones
- Always positive semi-definite

**Two-Parameter Exponential:**
$$
\rho_{ij} = \rho_\infty + (1 - \rho_\infty)\exp(-\beta |T_i - T_j|)
$$

Allows long-range correlation floor $\rho_\infty$.

### 5.3 Principal Component Analysis

**Eigendecomposition:**
$$
\boldsymbol{\rho} = \sum_{k=1}^{n} \lambda_k \mathbf{v}_k \mathbf{v}_k^\top
$$

**Typical Structure:**
- PC1 ($\sim 85\%$): Parallel shift
- PC2 ($\sim 10\%$): Slope (twist)
- PC3 ($\sim 3\%$): Curvature (butterfly)

**Factor Reduction:**
$$
\boldsymbol{\rho} \approx \sum_{k=1}^{m} \lambda_k \mathbf{v}_k \mathbf{v}_k^\top, \quad m \ll n
$$

### 5.4 Cholesky Decomposition for Simulation

**Decomposition:**
$$
\boldsymbol{\rho} = \mathbf{L}\mathbf{L}^\top
$$

where $\mathbf{L}$ is lower triangular.

**Correlated Brownian Increments:**
$$
\mathbf{Z} = \mathbf{L} \cdot \boldsymbol{\xi}, \quad \boldsymbol{\xi} \sim N(\mathbf{0}, \mathbf{I}_n)
$$

Then $\mathbb{E}[\mathbf{Z}\mathbf{Z}^\top] = \mathbf{L}\mathbf{L}^\top = \boldsymbol{\rho}$.

---

## 6. Discretization Schemes

### 6.1 Log-Euler Scheme (Standard)

**Discretization:**

$$
\boxed{F_i(t_{k+1}) = F_i(t_k) \exp\left[\left(\mu_i(t_k) - \frac{1}{2}\sigma_i^2\right)\Delta t + \sigma_i \sqrt{\Delta t} \, Z_i^k\right]}
$$

**Properties:**
- Preserves positivity: $F_i(t) > 0$ always
- First-order weak convergence: $O(\Delta t)$ bias
- Simple and stable

**Algorithm (Spot Measure):**

```
For each time step k = 0, 1, ..., n_steps-1:
    1. Identify alive forwards: j ≥ β(t_k)
    2. Compute drift for each alive forward i:
       μ_i = Σ_{j=β}^{i} ρ_ij σ_i σ_j τ_j F_j / (1 + τ_j F_j)
    3. Generate correlated normals: Z = L · ξ
    4. Update: F_i ← F_i · exp[(μ_i - σ_i²/2)Δt + σ_i√Δt Z_i]
    5. Fix forwards that reach their fixing date
```

### 6.2 Predictor-Corrector Scheme

**Step 1 - Predictor:**
$$
\hat{F}_i(t_{k+1}) = F_i(t_k) \exp\left[\left(\mu_i(t_k, \mathbf{F}(t_k)) - \frac{1}{2}\sigma_i^2\right)\Delta t + \sigma_i \sqrt{\Delta t} Z_i\right]
$$

**Step 2 - Corrector:**
$$
\bar{\mu}_i = \frac{1}{2}\left[\mu_i(t_k, \mathbf{F}(t_k)) + \mu_i(t_{k+1}, \hat{\mathbf{F}}(t_{k+1}))\right]
$$

**Step 3 - Update:**
$$
F_i(t_{k+1}) = F_i(t_k) \exp\left[\left(\bar{\mu}_i - \frac{1}{2}\sigma_i^2\right)\Delta t + \sigma_i \sqrt{\Delta t} Z_i\right]
$$

**Properties:**
- Second-order weak convergence: $O(\Delta t^2)$ bias
- Better for drift-sensitive products (swaptions)
- Double the computational cost

### 6.3 Time Step Guidelines

| Product | Minimum Steps | Recommended |
|---------|---------------|-------------|
| Caplet/Floorlet | 1 per period | 1 per period |
| Cap/Floor | 1 per period | 1 per period |
| European Swaption | 5 per period | 10 per period |
| Bermudan Swaption | 10 per period | 20 per period |
| Path-dependent | 20 per period | 50+ per period |

---

## 7. Product Pricing

### 7.1 Caplet Pricing (Exact)

**Payoff at $T_{i+1}$:**
$$
\text{Caplet}_i = \tau_i \max(F_i(T_i) - K, 0)
$$

**Black's Formula:**

$$
\boxed{\text{Caplet}_i = \tau_i P(0, T_{i+1}) \left[F_i(0) N(d_1) - K N(d_2)\right]}
$$

where:
$$
d_1 = \frac{\ln(F_i(0)/K) + \frac{1}{2}\sigma_i^2 T_i}{\sigma_i\sqrt{T_i}}, \quad d_2 = d_1 - \sigma_i\sqrt{T_i}
$$

**Proof:**

Under $\mathbb{Q}^{i+1}$, $F_i(t)$ is a martingale with log-normal terminal distribution. The pricing is:
$$
\text{Caplet}_i = \tau_i P(0, T_{i+1}) \mathbb{E}^{i+1}[(F_i(T_i) - K)^+]
$$

This is exactly Black's formula for a call on $F_i(0)$ with volatility $\sigma_i\sqrt{T_i}$. $\square$

### 7.2 Cap Pricing

A cap is a portfolio of caplets:
$$
\text{Cap} = \sum_{i=1}^{n} \text{Caplet}_i
$$

Priced by summing individual caplet values.

### 7.3 Swaption Pricing (No Closed Form)

**Payer Swaption Payoff at $T_\alpha$:**

$$
\text{Swaption} = \max\left(\sum_{i=\alpha}^{\beta-1} \tau_i P(T_\alpha, T_{i+1})(F_i(T_\alpha) - K), 0\right)
$$

**Equivalently using swap rate:**
$$
\text{Swaption} = A(T_\alpha) \max(S(T_\alpha) - K, 0)
$$

where:
- $S(T_\alpha) = \frac{1 - P(T_\alpha, T_\beta)}{A(T_\alpha)}$ is the forward swap rate
- $A(T_\alpha) = \sum_{i=\alpha}^{\beta-1} \tau_i P(T_\alpha, T_{i+1})$ is the annuity

**Monte Carlo Pricing:**
$$
V_0 = \frac{1}{N}\sum_{n=1}^{N} B(0)^{-1} \cdot \max(S^{(n)}(T_\alpha) - K, 0) \cdot A^{(n)}(T_\alpha) \cdot B^{(n)}(T_\alpha)
$$

under the spot measure with numeraire $B(t)$.

### 7.4 Rebonato's Swaption Approximation

**Theorem (Rebonato):** The forward swap rate $S(t)$ is approximately log-normal with effective volatility:

$$
\boxed{\sigma_{S}^2 T_\alpha \approx \sum_{i=\alpha}^{\beta-1}\sum_{j=\alpha}^{\beta-1} w_i(0) w_j(0) \rho_{ij} \sigma_i \sigma_j T_\alpha}
$$

where the weights are:
$$
w_i(0) = \frac{\tau_i P(0, T_{i+1}) F_i(0)}{S(0) A(0)}
$$

**Approximation Quality:**
- Excellent for ATM swaptions
- Degrades for deep ITM/OTM
- Used for quick calibration checks

---

## 8. Calibration Theory

### 8.1 Calibration to Cap/Floor Market

**Objective:** Match Black implied volatilities for liquid caps/floors.

**Exact Calibration:**
$$
\sigma_i^{LMM} = \sigma_i^{Black}
$$

This is **trivially exact** - the LMM was designed to reproduce Black's formula.

### 8.2 Calibration to Swaption Market

**Objective:** Find volatilities and correlations matching swaption vols:

$$
\min_{\boldsymbol{\sigma}, \boldsymbol{\rho}} \sum_{(\alpha,\beta) \in \mathcal{S}} w_{\alpha\beta} \left(\sigma_{\alpha\beta}^{LMM} - \sigma_{\alpha\beta}^{market}\right)^2
$$

**Challenges:**

1. **Over-parameterization:** More parameters than observables
2. **Cap-swaption conflict:** Perfect cap fit may not give perfect swaption fit
3. **Computational cost:** Each evaluation requires MC or approximation

### 8.3 Joint Calibration Strategy

**Step 1: Fix Caplet Volatilities**
$$
\sigma_i = \sigma_i^{cap}, \quad i = 1, \ldots, n
$$

**Step 2: Calibrate Correlation to Swaptions**

Using Rebonato's approximation:
$$
\min_{\boldsymbol{\rho}} \sum_{(\alpha,\beta)} w_{\alpha\beta} \left(\sigma_{\alpha\beta}^{Rebonato}(\boldsymbol{\sigma}, \boldsymbol{\rho}) - \sigma_{\alpha\beta}^{market}\right)^2
$$

Subject to: $\boldsymbol{\rho}$ positive semi-definite.

**Step 3: Iterate if Necessary**

If Rebonato approximation error is too large, refine with MC validation.

### 8.4 Volatility Parameterizations

**Time-Homogeneous:**
$$
\sigma_i(t) = \sigma_i \quad \text{(constant)}
$$
Simple; $n$ parameters.

**Rebonato Parametric:**
$$
\sigma_i(t) = [a + b(T_i - t)]e^{-c(T_i - t)} + d
$$
4 parameters; captures humped vol structure.

**Piecewise Constant:**
$$
\sigma_i(t) = \sigma_{ik} \quad \text{for } t \in [T_{k-1}, T_k)
$$
Flexible; $O(n^2)$ parameters.

---

## 9. Numerical Considerations

### 9.1 Variance Reduction

**Antithetic Variates:**
$$
\hat{V} = \frac{1}{2}[V(\mathbf{Z}) + V(-\mathbf{Z})]
$$

Reduces variance for payoffs with monotonic dependence on $\mathbf{Z}$.

**Control Variates:**

Use caplet prices (known analytically) as controls:
$$
\hat{V}_{CV} = V - c(C^{MC} - C^{analytic})
$$

where $c$ is optimally chosen.

### 9.2 Path-Dependent Products

**American/Bermudan Swaptions:**

Use Longstaff-Schwartz regression:
1. Simulate paths forward
2. At each exercise date, regress continuation value on basis functions
3. Exercise when intrinsic > continuation

**Andersen's Method:**

Pre-compute exercise boundary as function of swap rate.

### 9.3 Memory Management

For large $n$ (e.g., 40+ forwards):
- Store only current and previous time step
- Use factor reduction (3-5 PCs) for correlation
- Batch paths to manage memory

---

## 10. Comparison with Other Models

### 10.1 LMM vs Short-Rate Models

| Feature | LMM | Hull-White | Black-Karasinski |
|---------|-----|------------|------------------|
| **State variable** | Forward rates $F_i$ | Short rate $r$ | Log short rate $x$ |
| **Dimension** | $n$ (multi-factor) | 1 | 1 |
| **Observability** | Direct (market quoted) | Inferred | Inferred |
| **Caplet pricing** | Exact (Black) | Approximate | Numerical |
| **Swaption pricing** | Monte Carlo | Jamshidian (analytic) | Numerical |
| **Calibration** | Cap: trivial; Swaption: hard | Both: moderate | Both: hard |
| **Exotic pricing** | Natural framework | Requires extensions | Requires extensions |

### 10.2 When to Use LMM

**Use LMM for:**
- Exotic IR derivatives (path-dependent)
- Products sensitive to correlation
- When cap calibration is paramount
- CVA/XVA calculations

**Use Short-Rate for:**
- Simple European swaptions (Jamshidian)
- Speed-critical applications
- When negative rates are important (Hull-White)

---

## 11. Advantages and Limitations

### 11.1 Advantages

1. **Observable state variables:** Forward rates are market-quoted
2. **Exact cap calibration:** Reproduces Black's formula by construction
3. **Flexible correlation:** Can match swaption matrix
4. **Multi-factor:** Captures term structure dynamics
5. **Industry standard:** Widely implemented and understood

### 11.2 Limitations

1. **No negative rates:** Standard LMM requires $F_i > 0$
2. **No closed-form swaptions:** Monte Carlo required
3. **High dimensionality:** $n$ forwards to simulate
4. **Calibration complexity:** Cap-swaption tension
5. **No stochastic volatility:** Standard model has deterministic vol

### 11.3 Model Extensions

| Extension | Purpose |
|-----------|---------|
| **Shifted LMM** | Handle negative rates: $d(F_i + s) = (F_i + s)\sigma dW$ |
| **SABR-LMM** | Stochastic volatility for smile |
| **Jump-diffusion** | Fat tails, event risk |
| **RFR-LMM** | Risk-free rate (SOFR/SONIA) adaptation |

---

## 12. Interview Key Points

### Derivation Questions

**Q: Why does LMM exactly calibrate to caps but not swaptions?**

A: Under the $T_{i+1}$-forward measure, $F_i(t)$ is a driftless martingale, so $F_i(T_i)$ is log-normal with known variance. This is exactly the Black model assumption. Swaptions involve sums of bond prices, which don't have log-normal distributions, so no closed form exists.

**Q: Derive the drift under the spot measure.**

A:
1. Under terminal measure $\mathbb{Q}^N$: $dF_i/F_i = -\sum_{j>i}(\cdots)dt + \sigma_i dW_i^N$
2. Girsanov: $dW_i^B = dW_i^N + \gamma_i dt$ with $\gamma_i = \sum_j \rho_{ij}\sigma_j\tau_j F_j/(1+\tau_j F_j)$
3. Spot drift: $\mu_i = \sum_{j=\beta}^{i} \rho_{ij}\sigma_i\sigma_j\tau_j F_j/(1+\tau_j F_j)$

**Q: Why is the sum in the spot drift from $\beta(t)$ to $i$, not to $N-1$?**

A: The Girsanov kernel involves all forwards, but when combined with the terminal drift (which sums from $i+1$ to $N-1$), the terms for $j > i$ cancel, leaving only $j \leq i$. The lower bound $\beta(t)$ arises because fixed forwards no longer contribute to the numeraire dynamics.

### Practical Questions

**Q: How do you calibrate LMM?**

A:
1. Set caplet vols = market Black vols (exact)
2. Choose correlation parameterization (e.g., exponential)
3. Calibrate correlation parameters to swaption vols using Rebonato approximation
4. Validate with Monte Carlo

**Q: What variance reduction do you use for LMM Monte Carlo?**

A: Antithetic variates (most common), control variates using caplets (if applicable), and importance sampling for rare events.

**Q: How do you handle negative rates in LMM?**

A: Use shifted LMM: replace $F_i$ with $F_i + s$ where $s > 0$ is the shift. Dynamics become $d(F_i + s) = (F_i + s)\sigma dW$, allowing $F_i > -s$.

---

## Appendix: Key Formulas

### A.1 Forward Rate Dynamics (Spot Measure)

$$
\frac{dF_i(t)}{F_i(t)} = \sum_{j=\beta(t)}^{i} \frac{\rho_{ij} \sigma_i \sigma_j \tau_j F_j(t)}{1 + \tau_j F_j(t)} dt + \sigma_i dW_i^B(t)
$$

### A.2 Log-Euler Discretization

$$
F_i(t+\Delta t) = F_i(t) \exp\left[\left(\mu_i - \frac{\sigma_i^2}{2}\right)\Delta t + \sigma_i\sqrt{\Delta t}Z_i\right]
$$

### A.3 Black's Caplet Formula

$$
\text{Caplet} = \tau P(0, T_{i+1})[F_i(0)N(d_1) - KN(d_2)]
$$

$$
d_1 = \frac{\ln(F/K) + \sigma^2 T/2}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

### A.4 Rebonato Swaption Approximation

$$
\sigma_S^2 T \approx \sum_{i,j} w_i w_j \rho_{ij} \sigma_i \sigma_j T
$$

### A.5 Exponential Correlation

$$
\rho_{ij} = \exp(-\beta|T_i - T_j|)
$$

---

## References

1. Brace, A., Gatarek, D., & Musiela, M. (1997). "The Market Model of Interest Rate Dynamics." *Mathematical Finance*, 7(2), 127-155.

2. Jamshidian, F. (1997). "LIBOR and Swap Market Models and Measures." *Finance and Stochastics*, 1(4), 293-330.

3. Rebonato, R. (2002). *Modern Pricing of Interest-Rate Derivatives: The LIBOR Market Model and Beyond*. Princeton University Press.

4. Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*, 2nd ed. Springer.

5. Andersen, L. & Piterbarg, V. (2010). *Interest Rate Modeling*, Vols. 1-3. Atlantic Financial Press.

6. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

---

*Document Version: 2.0 | QuantStrata Phase 3.8 | January 2026*
