# Monte Carlo Methods in Derivatives Pricing

**Complete Mathematical Framework for Simulation-Based Valuation**

This document provides a rigorous treatment of Monte Carlo methods for derivatives pricing, including theoretical foundations, variance reduction techniques, convergence analysis, and advanced applications.

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Basic Monte Carlo Estimator](#2-basic-monte-carlo-estimator)
3. [Path Simulation](#3-path-simulation)
4. [Variance Reduction Techniques](#4-variance-reduction-techniques)
5. [Convergence and Error Analysis](#5-convergence-and-error-analysis)
6. [Path-Dependent Options](#6-path-dependent-options)
7. [American Options: Longstaff-Schwartz](#7-american-options-longstaff-schwartz)
8. [Greeks by Monte Carlo](#8-greeks-by-monte-carlo)
9. [Advanced Topics](#9-advanced-topics)
10. [Interview Key Points](#10-interview-key-points)

---

## 1. Theoretical Foundation

### 1.1 Risk-Neutral Valuation

The fundamental theorem of asset pricing states that in an arbitrage-free market, derivative prices equal:

$$
V_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]
$$

Where:
- $\mathbb{Q}$: Risk-neutral (martingale) measure
- $r$: Risk-free rate
- $T$: Time to expiry
- $S_T$: Asset price at expiry

### 1.2 Why Monte Carlo?

Monte Carlo is essential when:
1. **No closed-form solution** exists (most exotics)
2. **High dimensionality** (baskets, multi-asset)
3. **Path-dependence** (barriers, Asians, lookbacks)
4. **Complex payoffs** (autocallables, TARFs)
5. **Stochastic models** (Heston, local vol)

### 1.3 The Law of Large Numbers

**Strong Law of Large Numbers:** If $X_1, X_2, \ldots$ are i.i.d. with $\mathbb{E}[|X|] < \infty$:

$$
\frac{1}{n}\sum_{i=1}^{n}X_i \xrightarrow{a.s.} \mathbb{E}[X] \quad \text{as } n \to \infty
$$

This guarantees Monte Carlo convergence.

---

## 2. Basic Monte Carlo Estimator

### 2.1 The Estimator

Generate $N$ independent samples $S_T^{(1)}, \ldots, S_T^{(N)}$ under $\mathbb{Q}$:

$$
\hat{V} = e^{-rT} \cdot \frac{1}{N}\sum_{i=1}^{N} g(S_T^{(i)})
$$

Where $g(\cdot)$ is the payoff function.

### 2.2 Unbiasedness

$$
\mathbb{E}[\hat{V}] = e^{-rT}\mathbb{E}^{\mathbb{Q}}[g(S_T)] = V_0
$$

The estimator is **unbiased** for any sample size $N$.

### 2.3 Variance of the Estimator

$$
\text{Var}(\hat{V}) = \frac{e^{-2rT}}{N}\text{Var}^{\mathbb{Q}}(g(S_T)) = \frac{\sigma_g^2}{N}
$$

Where $\sigma_g^2$ is the variance of the discounted payoff.

### 2.4 Standard Error

$$
\text{SE}(\hat{V}) = \frac{\sigma_g}{\sqrt{N}}
$$

**Key insight:** Error decreases as $O(1/\sqrt{N})$.

To halve the error, we need **4× more paths**.

### 2.5 Confidence Interval

By the Central Limit Theorem:

$$
\frac{\hat{V} - V_0}{\sigma_g/\sqrt{N}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

**95% Confidence Interval:**

$$
\hat{V} \pm 1.96 \cdot \frac{\hat{\sigma}_g}{\sqrt{N}}
$$

Where $\hat{\sigma}_g$ is the sample standard deviation.

---

## 3. Path Simulation

### 3.1 Geometric Brownian Motion

Under $\mathbb{Q}$:
$$
dS_t = rS_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
$$

**Exact Solution:**
$$
S_T = S_0 \exp\left((r - \frac{\sigma^2}{2})T + \sigma\sqrt{T}Z\right)
$$

Where $Z \sim \mathcal{N}(0,1)$.

### 3.2 Discretization Schemes

For multi-step simulation with $\Delta t = T/n$:

**Euler-Maruyama:**
$$
S_{t+\Delta t} = S_t + rS_t\Delta t + \sigma S_t\sqrt{\Delta t}Z
$$
- Simple but can go negative
- Weak order 1.0

**Milstein:**
$$
S_{t+\Delta t} = S_t + rS_t\Delta t + \sigma S_t\sqrt{\Delta t}Z + \frac{1}{2}\sigma^2 S_t(\Delta t Z^2 - \Delta t)
$$
- Strong order 1.0 (vs 0.5 for Euler)
- Better for path-dependent payoffs

**Exact (Log-Euler):**
$$
S_{t+\Delta t} = S_t \exp\left((r - \frac{\sigma^2}{2})\Delta t + \sigma\sqrt{\Delta t}Z\right)
$$
- Always positive
- Exact distribution at each step

### 3.3 Multi-Dimensional Simulation

For correlated assets with correlation $\rho$:

**Cholesky Decomposition:** If $\Sigma = LL^T$:
$$
\begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix} = L \begin{pmatrix} W_1 \\ W_2 \end{pmatrix}
$$

For 2D:
$$
Z_1 = W_1, \quad Z_2 = \rho W_1 + \sqrt{1-\rho^2}W_2
$$

---

## 4. Variance Reduction Techniques

### 4.1 Antithetic Variates

**Idea:** Use $-Z$ alongside $Z$ to create negatively correlated samples.

**Estimator:**
$$
\hat{V}_{AV} = \frac{1}{2N}\sum_{i=1}^{N}\left[g(S_T^{(Z_i)}) + g(S_T^{(-Z_i)})\right]
$$

**Variance Reduction:**
$$
\text{Var}(\hat{V}_{AV}) = \frac{1}{N}\left[\frac{\sigma_g^2}{2} + \frac{\text{Cov}(g(S^+), g(S^-))}{2}\right]
$$

If $g$ is monotonic in $S$, the covariance is negative → variance reduced.

**Effective for:**
- Vanilla options (highly effective)
- Monotonic payoffs

**Less effective for:**
- Straddles (symmetric payoff)
- Digital options (discontinuous)

### 4.2 Control Variates

**Idea:** Use a correlated variable with known expectation to reduce variance.

If $Y$ has known $\mathbb{E}[Y] = \mu_Y$:

$$
\hat{V}_{CV} = \hat{V} - c(\hat{Y} - \mu_Y)
$$

**Optimal $c$:**
$$
c^* = \frac{\text{Cov}(V, Y)}{\text{Var}(Y)}
$$

**Variance Reduction:**
$$
\text{Var}(\hat{V}_{CV}) = \text{Var}(\hat{V})(1 - \rho_{VY}^2)
$$

**Common control variates:**
- Stock price: $\mathbb{E}^{\mathbb{Q}}[S_T] = S_0 e^{rT}$
- Vanilla option: BSM price (for exotic options)
- Geometric Asian: Closed-form (for arithmetic Asian)

### 4.3 Importance Sampling

**Idea:** Sample from a different distribution $\mathbb{P}'$ to reduce variance.

$$
\mathbb{E}^{\mathbb{Q}}[g(S)] = \mathbb{E}^{\mathbb{P}'}\left[g(S)\frac{d\mathbb{Q}}{d\mathbb{P}'}(S)\right]
$$

**Application: OTM Options**

For deep OTM calls, shift the mean to make exercise more likely:

Sample from $\mathcal{N}(\mu', 1)$ instead of $\mathcal{N}(0, 1)$:
$$
\hat{V}_{IS} = \frac{1}{N}\sum_{i=1}^{N}g(S_T^{(i)})e^{-\mu' Z_i + \mu'^2/2}
$$

### 4.4 Stratified Sampling

**Idea:** Divide sample space into strata and sample from each.

Partition $[0,1]$ into $K$ intervals. In each interval $[(k-1)/K, k/K]$:
- Generate $U_k \sim \text{Uniform}[(k-1)/K, k/K]$
- Transform: $Z_k = \Phi^{-1}(U_k)$

Guarantees better coverage of the distribution.

### 4.5 Quasi-Monte Carlo

**Idea:** Replace pseudo-random numbers with low-discrepancy sequences.

**Sequences:**
- Halton sequence
- Sobol sequence
- Faure sequence

**Convergence:** $O((\log N)^d / N)$ vs $O(1/\sqrt{N})$ for MC

**Best for:** Low dimensions ($d < 20$), smooth payoffs.

---

## 5. Convergence and Error Analysis

### 5.1 Central Limit Theorem

For i.i.d. samples with finite variance:

$$
\sqrt{N}(\hat{V} - V_0) \xrightarrow{d} \mathcal{N}(0, \sigma_g^2)
$$

### 5.2 Berry-Esseen Bound

If $\mathbb{E}[|g|^3] < \infty$:

$$
\sup_x |P(\sqrt{N}(\hat{V} - V_0)/\sigma_g \leq x) - \Phi(x)| \leq \frac{C\mathbb{E}[|g|^3]}{\sigma_g^3\sqrt{N}}
$$

Quantifies the rate of convergence to normality.

### 5.3 Bias-Variance Decomposition

**Mean Squared Error:**
$$
\text{MSE} = \text{Bias}^2 + \text{Variance}
$$

For basic MC: Bias = 0, so MSE = Var = $\sigma_g^2/N$.

**Discretization introduces bias:**
- Euler: $O(\Delta t)$ bias
- Milstein: $O(\Delta t^2)$ bias

### 5.4 Practical Error Estimation

**Estimated Standard Error:**
$$
\hat{\text{SE}} = \frac{\hat{\sigma}_g}{\sqrt{N}} = \sqrt{\frac{1}{N(N-1)}\sum_{i=1}^{N}(g_i - \bar{g})^2}
$$

**Rule of Thumb:** For 1% relative error:
$$
N \approx \left(\frac{1.96 \cdot \text{CV}}{0.01}\right)^2
$$

Where CV = coefficient of variation = $\sigma_g / V_0$.

---

## 6. Path-Dependent Options

### 6.1 Barrier Options

**Discrete Monitoring:**
Check barrier at times $t_1, \ldots, t_n$:

```
knocked_out = any(S[t_i] >= B for i in 1..n)
payoff = max(S[T] - K, 0) * (1 - knocked_out)  # for up-and-out call
```

**Continuity Correction (Broadie-Glasserman-Kou):**

For discrete monitoring with $n$ steps:
$$
\tilde{B} = B \cdot e^{\beta\sigma\sqrt{\Delta t}}
$$

Where $\beta \approx 0.5826$ adjusts for discrete vs continuous monitoring.

### 6.2 Asian Options

**Arithmetic Average:**
$$
A = \frac{1}{n}\sum_{i=1}^{n}S_{t_i}
$$

**Payoff:** $\max(A - K, 0)$ (call)

**No closed-form** for arithmetic average → MC essential.

**Variance Reduction:** Use geometric average (closed-form) as control variate:
$$
G = \left(\prod_{i=1}^{n}S_{t_i}\right)^{1/n}
$$

### 6.3 Lookback Options

**Floating Strike:**
$$
\text{Payoff}_{\text{call}} = S_T - \min_{t \in [0,T]} S_t
$$

**Fixed Strike:**
$$
\text{Payoff}_{\text{call}} = \max\left(\max_{t \in [0,T]} S_t - K, 0\right)
$$

**Discrete vs Continuous:** Discrete monitoring introduces bias; use fine grid.

---

## 7. American Options: Longstaff-Schwartz

### 7.1 The Optimal Stopping Problem

American option value:
$$
V_0 = \sup_{\tau} \mathbb{E}^{\mathbb{Q}}[e^{-r\tau}g(S_\tau)]
$$

Where $\tau$ is a stopping time.

### 7.2 Dynamic Programming Principle

At each time $t$:
$$
V_t = \max\left(g(S_t), \mathbb{E}^{\mathbb{Q}}[e^{-r\Delta t}V_{t+\Delta t}|S_t]\right)
$$

**Problem:** $\mathbb{E}[V_{t+\Delta t}|S_t]$ is a function of $S_t$ — how to estimate?

### 7.3 Least-Squares Monte Carlo (LSM)

**Longstaff-Schwartz Algorithm:**

1. Simulate $N$ paths forward
2. Work backward from expiry:
   - At $t_n = T$: $V^{(i)}_{t_n} = g(S_{t_n}^{(i)})$
   - At $t_k < T$:
     a. Regress discounted continuation values on basis functions:
        $$
        e^{-r\Delta t}V^{(i)}_{t_{k+1}} = \sum_j \beta_j \phi_j(S_{t_k}^{(i)}) + \epsilon^{(i)}
        $$
     b. Continuation value estimate: $\hat{C}(S_{t_k}^{(i)}) = \sum_j \hat{\beta}_j \phi_j(S_{t_k}^{(i)})$
     c. Exercise decision: Exercise if $g(S_{t_k}^{(i)}) > \hat{C}(S_{t_k}^{(i)})$

**Basis Functions:**
- Polynomials: $1, S, S^2, S^3, \ldots$
- Laguerre polynomials: $L_n(S)$
- Hermite polynomials

### 7.4 Bias in LSM

**Low bias:** Uses in-the-money paths only for regression
**High bias:** Polynomial approximation error

**Remedy:** 
- Use separate "training" and "pricing" paths
- Increase basis functions (but avoid overfitting)

---

## 8. Greeks by Monte Carlo

### 8.1 Finite Difference (Bump-and-Reprice)

**Delta:**
$$
\Delta \approx \frac{V(S_0 + h) - V(S_0 - h)}{2h}
$$

**Gamma:**
$$
\Gamma \approx \frac{V(S_0 + h) - 2V(S_0) + V(S_0 - h)}{h^2}
$$

**Problem:** Each bump requires fresh simulation — expensive!

### 8.2 Pathwise Method (IPA)

**Idea:** Differentiate inside the expectation.

If payoff is Lipschitz continuous:
$$
\Delta = \mathbb{E}^{\mathbb{Q}}\left[e^{-rT}\frac{\partial g}{\partial S_0}\right]
$$

**For vanilla call:**
$$
\frac{\partial}{\partial S_0}\max(S_T - K, 0) = \mathbf{1}_{S_T > K}\frac{\partial S_T}{\partial S_0} = \mathbf{1}_{S_T > K}\frac{S_T}{S_0}
$$

So:
$$
\Delta = \mathbb{E}^{\mathbb{Q}}\left[e^{-rT}\mathbf{1}_{S_T > K}\frac{S_T}{S_0}\right]
$$

**Advantage:** Same paths, one simulation run.

**Limitation:** Doesn't work for discontinuous payoffs (digitals).

### 8.3 Likelihood Ratio Method

**Idea:** Differentiate the density, not the payoff.

$$
\frac{\partial}{\partial \theta}\mathbb{E}_\theta[g(S)] = \mathbb{E}_\theta\left[g(S)\frac{\partial \log p_\theta(S)}{\partial \theta}\right]
$$

**For delta with GBM:**
$$
\Delta = \mathbb{E}\left[e^{-rT}g(S_T)\frac{Z}{\sigma S_0 \sqrt{T}}\right]
$$

**Advantage:** Works for discontinuous payoffs.

**Disadvantage:** Higher variance than pathwise.

### 8.4 Malliavin Calculus

Advanced technique using stochastic calculus to derive:
$$
\Delta = \mathbb{E}\left[e^{-rT}g(S_T)\frac{W_T}{\sigma S_0 T}\right]
$$

Provides smooth Greeks even for discontinuous payoffs.

---

## 9. Advanced Topics

### 9.1 Multi-Level Monte Carlo (MLMC)

**Idea:** Combine coarse and fine discretizations.

$$
\mathbb{E}[P_L] = \mathbb{E}[P_0] + \sum_{\ell=1}^{L}\mathbb{E}[P_\ell - P_{\ell-1}]
$$

Use many samples for cheap coarse levels, few for expensive fine levels.

**Complexity:** $O(\epsilon^{-2})$ vs $O(\epsilon^{-3})$ for standard MC to achieve error $\epsilon$.

### 9.2 Stochastic Volatility Models

**Heston Model:**
$$
dS_t = rS_t \, dt + \sqrt{v_t}S_t \, dW_t^S
$$
$$
dv_t = \kappa(\theta - v_t)dt + \xi\sqrt{v_t}dW_t^v
$$

**Correlation:** $dW_t^S \cdot dW_t^v = \rho \, dt$

**Simulation Challenges:**
- Variance can go negative (Euler) → use QE scheme
- Correlation structure
- Higher computational cost (2D)

### 9.3 Jump-Diffusion Models

**Merton Model:**
$$
dS_t = (r - \lambda k)S_t \, dt + \sigma S_t \, dW_t + S_t \, dJ_t
$$

Where $J_t$ is a compound Poisson process.

**Simulation:**
1. Simulate number of jumps: $N \sim \text{Poisson}(\lambda T)$
2. Simulate jump times: Uniform on $[0,T]$
3. Simulate jump sizes: $Y_i \sim \text{LogNormal}(\mu_J, \sigma_J^2)$

---

## 10. Interview Key Points

### Basic Questions

**Q: Explain Monte Carlo pricing.**

A: Generate $N$ random paths under the risk-neutral measure, compute payoffs, average, and discount. By LLN, this converges to the true price.

**Q: What's the convergence rate?**

A: $O(1/\sqrt{N})$. Standard error = $\sigma/\sqrt{N}$. To halve error, need 4× more paths.

**Q: Name three variance reduction techniques.**

A:
1. **Antithetic variates**: Use $+Z$ and $-Z$ pairs
2. **Control variates**: Use correlated variable with known mean
3. **Importance sampling**: Change sampling distribution

### Advanced Questions

**Q: When is antithetic variates NOT effective?**

A: For symmetric payoffs (straddles) or discontinuous payoffs (digitals) where $f(S^+)$ and $f(S^-)$ are not negatively correlated.

**Q: Explain Longstaff-Schwartz.**

A: Backward induction with regression:
1. Simulate paths forward
2. At each exercise date, regress continuation value on basis functions
3. Exercise if intrinsic > estimated continuation
4. Average discounted payoffs

**Q: How would you compute delta by MC?**

A: Three methods:
1. **Bump-and-reprice**: Expensive (2 simulations)
2. **Pathwise**: Differentiate payoff, single simulation
3. **Likelihood ratio**: Differentiate density, works for digitals

**Q: What's the bias in discrete barrier monitoring?**

A: Discrete monitoring under-prices knock-out options (barrier not checked between dates). Use Broadie-Glasserman-Kou correction: shift barrier by $0.5826\sigma\sqrt{\Delta t}$.

### Practical Questions

**Q: How many paths do you need?**

A: For 1% relative error: $N \approx (196 \cdot \text{CV})^2$ where CV = coefficient of variation.

**Q: MC vs PDE — when to use each?**

A: 
- **MC**: High dimensions, path-dependent, stochastic vol
- **PDE**: Low dimensions, American options, require Greeks

---

## Appendix: Algorithm Pseudocode

### Basic MC Pricer

```python
def mc_price(S0, K, T, r, sigma, n_paths, payoff_func):
    # Generate terminal spots under Q
    Z = np.random.randn(n_paths)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    
    # Compute payoffs
    payoffs = payoff_func(ST, K)
    
    # Discount and average
    price = np.exp(-r*T) * np.mean(payoffs)
    stderr = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_paths)
    
    return price, stderr
```

### Antithetic Variates

```python
def mc_price_antithetic(S0, K, T, r, sigma, n_paths, payoff_func):
    Z = np.random.randn(n_paths // 2)
    
    # Positive and negative paths
    ST_plus = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    ST_minus = S0 * np.exp((r - 0.5*sigma**2)*T - sigma*np.sqrt(T)*Z)
    
    # Average paired payoffs
    payoffs = 0.5 * (payoff_func(ST_plus, K) + payoff_func(ST_minus, K))
    
    return np.exp(-r*T) * np.mean(payoffs)
```

---

## References

1. Glasserman, P. "Monte Carlo Methods in Financial Engineering"
2. Jäckel, P. "Monte Carlo Methods in Finance"
3. Longstaff, F. & Schwartz, E. (2001). "Valuing American Options by Simulation"
4. Giles, M. (2008). "Multilevel Monte Carlo Methods"

---

---

## Library Implementation

The methods described in this document are implemented in QuantStrata:

### Core Implementation

| Method | Module | Key Functions |
|--------|--------|---------------|
| Standard MC | `src.models.numeric.monte_carlo.rng` | `NormalRng` |
| Antithetic Variates | `src.models.numeric.monte_carlo.rng` | `standard_normals(antithetic=True)` |
| Control Variates | `src.models.numeric.monte_carlo.control_variates` | See module |
| Longstaff-Schwartz | `src.models.numeric.monte_carlo.lsm` | `lsm_american_put`, `price_american_put_lsm` |
| Quasi-Monte Carlo | `src.models.numeric.monte_carlo.qmc` | `SobolRng`, `qmc_european_call` |
| Importance Sampling | `src.models.numeric.monte_carlo.importance_sampling` | `is_european_put`, `is_european_call` |

### User Guides

- [LSM User Guide](../../guides/numerical_methods/lsm.md)
- [QMC User Guide](../../guides/numerical_methods/qmc.md)
- [Importance Sampling User Guide](../../guides/numerical_methods/importance_sampling.md)

### Tutorial

- [Advanced MC Methods Tutorial](../../tutorials/pricing/advanced_mc_methods.ipynb)

---

*Document Version: 2.0 | QuantStrata Phase 4.2 | January 2026*
