# Deep Hedging: A Comprehensive Technical Reference

**Status:** Phase 7.6  
**Level:** PhD / Research  
**Prerequisites:** Stochastic calculus, optimal control theory, reinforcement learning fundamentals

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Classical Optimal Hedging Theory](#3-classical-optimal-hedging-theory)
4. [The Deep Hedging Framework](#4-the-deep-hedging-framework)
5. [Reinforcement Learning Formulation](#5-reinforcement-learning-formulation)
6. [Risk Measures and Objective Functions](#6-risk-measures-and-objective-functions)
7. [Transaction Costs and Market Frictions](#7-transaction-costs-and-market-frictions)
8. [Neural Network Architectures](#8-neural-network-architectures)
9. [Training and Convergence](#9-training-and-convergence)
10. [Theoretical Analysis](#10-theoretical-analysis)
11. [Literature Review](#11-literature-review)
12. [References](#12-references)

---

## 1. Introduction and Motivation

### 1.1 The Hedging Problem

Consider a derivatives dealer who has sold a European call option with strike $K$ and maturity $T$ to a client. The dealer now faces *contingent liability*: at time $T$, they must pay $(S_T - K)^+$ to the client. The fundamental question of hedging is:

> **How should the dealer trade in the underlying asset to offset the risk of this contingent claim?**

The classical answer, derived from Black-Scholes-Merton (BSM) theory, is **delta hedging**: at each instant, hold $\Delta_t = \partial V / \partial S$ units of the underlying, where $V$ is the option value. Under the BSM assumptions, this strategy perfectly replicates the option payoff.

### 1.2 Why Classical Hedging Fails in Practice

The BSM replication argument relies on several idealised assumptions that fail in practice:

| Assumption | Reality |
|------------|---------|
| Continuous trading | Trading occurs at discrete times |
| No transaction costs | Bid-ask spreads, commissions, market impact |
| Constant volatility | Volatility is stochastic and uncertain |
| Unlimited liquidity | Large trades move the market |
| Known model | Model misspecification is ubiquitous |

When these assumptions are violated, delta hedging is no longer optimal. In fact:

1. **Discrete rehedging** introduces *hedging error* that accumulates over time
2. **Transaction costs** make frequent rehedging prohibitively expensive
3. **Stochastic volatility** means delta-neutral positions still have vega exposure
4. **Model uncertainty** means the "true" delta is unknown

### 1.3 The Deep Hedging Paradigm

**Deep hedging** (Bühler et al., 2019) proposes a fundamentally different approach:

> Instead of deriving the hedge from a model, **learn** the optimal hedging policy directly from data by minimising a risk measure over the distribution of hedging P&L.

This approach:
- Makes no assumptions about the underlying dynamics
- Naturally incorporates transaction costs
- Handles discrete rehedging
- Can adapt to any risk preference
- Leverages the universal approximation power of neural networks

---

## 2. Mathematical Foundations

### 2.1 Probability Space and Filtration

Let $(\Omega, \mathcal{F}, \mathbb{P})$ be a probability space equipped with a filtration $\{\mathcal{F}_t\}_{t \in [0,T]}$ satisfying the usual conditions.

- $\Omega$: sample space (set of all possible market scenarios)
- $\mathcal{F}$: σ-algebra of events
- $\mathbb{P}$: physical (real-world) probability measure
- $\mathcal{F}_t$: information available at time $t$

### 2.2 Asset Price Dynamics

Let $S = (S_t)_{t \in [0,T]}$ denote the price process of the underlying asset. Under the physical measure $\mathbb{P}$:

$$
dS_t = \mu(t, S_t) \, dt + \sigma(t, S_t) \, dW_t
$$

where:
- $\mu(t, S_t)$: drift (expected return)
- $\sigma(t, S_t)$: volatility
- $W_t$: standard Brownian motion under $\mathbb{P}$

**Important:** Deep hedging operates under the *physical measure* $\mathbb{P}$, not the risk-neutral measure $\mathbb{Q}$. This is because we care about the actual distribution of P&L, not the pricing measure.

### 2.3 Trading Strategies

A **trading strategy** is an $\mathcal{F}_t$-adapted process $\delta = (\delta_t)_{t \in [0,T]}$ where $\delta_t$ represents the number of units of the underlying held at time $t$.

For discrete-time trading at times $0 = t_0 < t_1 < \cdots < t_N = T$:
- $\delta_n := \delta_{t_n}$ is $\mathcal{F}_{t_n}$-measurable (depends only on information up to $t_n$)
- The position $\delta_n$ is held over the interval $[t_n, t_{n+1})$

### 2.4 Self-Financing Condition

A trading strategy is **self-financing** if the only source of funds is the initial endowment and trading gains/losses. With transaction costs, the wealth process $X$ evolves as:

$$
X_{t_{n+1}} = X_{t_n} + \delta_n (S_{t_{n+1}} - S_{t_n}) - C(\delta_n - \delta_{n-1}, S_{t_n})
$$

where $C(\cdot, \cdot)$ is the transaction cost function.

---

## 3. Classical Optimal Hedging Theory

### 3.1 Complete Markets and Perfect Replication

In the BSM model with continuous trading and no frictions, markets are **complete**: every contingent claim can be perfectly replicated.

**Theorem (BSM Replication):** Under the BSM assumptions, the value $V(t, S)$ of a European option with payoff $\Phi(S_T)$ satisfies the Black-Scholes PDE:

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$

with terminal condition $V(T, S) = \Phi(S)$.

The replicating strategy is $\delta_t = \frac{\partial V}{\partial S}(t, S_t)$, and the hedging error is exactly zero.

### 3.2 Incomplete Markets

When markets are incomplete (e.g., due to transaction costs, discrete trading, or unhedgeable risk factors), perfect replication is impossible. The hedger faces a trade-off:

- **Underhedging**: Accept residual risk, save on transaction costs
- **Overhedging**: Reduce risk, pay more transaction costs

This trade-off is formalised through **risk-based optimisation**.

### 3.3 Quadratic Hedging (Föllmer-Schweizer)

The **mean-variance hedging** approach minimises the expected squared hedging error:

$$
\min_{\delta} \mathbb{E}\left[ \left( \Phi(S_T) - X_0 - \int_0^T \delta_t \, dS_t \right)^2 \right]
$$

**Theorem (Föllmer-Schweizer Decomposition):** Under suitable conditions, the optimal hedge is given by:

$$
\delta_t^* = \frac{\text{Cov}(dS_t, d\Phi_t | \mathcal{F}_t)}{\text{Var}(dS_t | \mathcal{F}_t)}
$$

where $\Phi_t = \mathbb{E}[\Phi(S_T) | \mathcal{F}_t]$ is the intrinsic value process.

### 3.4 Utility-Based Hedging

An alternative approach uses **expected utility maximisation**:

$$
\max_{\delta} \mathbb{E}[U(X_T - \Phi(S_T))]
$$

where $U$ is a utility function (e.g., exponential: $U(x) = -e^{-\gamma x}$).

**Exponential Utility and Indifference Pricing:**

For exponential utility with risk aversion $\gamma$, the optimal hedge satisfies a nonlinear PDE (Hamilton-Jacobi-Bellman equation). The **indifference price** is the amount that makes the hedger indifferent between accepting and rejecting the contingent claim.

### 3.5 Optimal Hedging with Transaction Costs

With proportional transaction costs, the hedging problem becomes a **singular stochastic control** problem. The optimal strategy involves a **no-trade region**: the hedger only trades when the current position deviates sufficiently from the target.

**Theorem (Hodges-Neuberger, Whalley-Wilmott):** With proportional transaction cost $\kappa$, the optimal strategy for a delta-hedged position involves:

1. A **hedging bandwidth** around the BSM delta
2. Trade only when position exits the bandwidth
3. Bandwidth width scales as $\kappa^{1/3}$

This leads to the famous "Leland volatility adjustment":

$$
\tilde{\sigma}^2 = \sigma^2 \left(1 + \sqrt{\frac{8}{\pi}} \cdot \frac{\kappa}{\sigma \sqrt{\Delta t}} \right)
$$

---

## 4. The Deep Hedging Framework

### 4.1 Problem Formulation

Deep hedging reframes the hedging problem as a **learning problem**. Instead of deriving the optimal hedge from a model, we:

1. **Define a space of admissible strategies** (parameterised by neural networks)
2. **Define an objective function** (risk measure over P&L distribution)
3. **Optimise** using stochastic gradient descent

### 4.2 Discrete-Time Setup

Consider hedging over $N$ periods: $0 = t_0 < t_1 < \cdots < t_N = T$.

**State at time $t_n$:**
$$
\mathbf{x}_n = (S_{t_n}, t_n, \delta_{n-1}, V_n, \text{features}_n) \in \mathcal{X}
$$

where:
- $S_{t_n}$: spot price
- $t_n$: time
- $\delta_{n-1}$: current position
- $V_n$: current P&L or hedge value
- $\text{features}_n$: additional features (Greeks, implied vol, etc.)

**Action at time $t_n$:**
$$
\delta_n = \pi_\theta(\mathbf{x}_n) \in \mathcal{A}
$$

where $\pi_\theta$ is a neural network policy with parameters $\theta$.

**P&L Evolution:**
$$
\text{P\&L}_{n+1} = \text{P\&L}_n + \delta_n (S_{t_{n+1}} - S_{t_n}) - C(\delta_n - \delta_{n-1}, S_{t_n})
$$

**Terminal P&L:**
$$
\text{P\&L}_T = V_0 + \sum_{n=0}^{N-1} \delta_n (S_{t_{n+1}} - S_{t_n}) - \sum_{n=0}^{N-1} C(\Delta\delta_n, S_{t_n}) - \Phi(S_T)
$$

where $V_0$ is the initial option premium received.

### 4.3 Objective Function

The deep hedging objective is to minimise a **risk measure** $\rho$ applied to the terminal P&L distribution:

$$
\min_\theta \rho\left( -\text{P\&L}_T^\theta \right)
$$

The negative sign is because risk measures are typically defined on losses.

Common choices for $\rho$:
- **Mean-variance**: $\rho(L) = \mathbb{E}[L] + \lambda \cdot \text{Var}(L)$
- **CVaR (Expected Shortfall)**: $\rho(L) = \mathbb{E}[L | L \geq \text{VaR}_\alpha(L)]$
- **Exponential utility**: $\rho(L) = \frac{1}{\gamma} \log \mathbb{E}[e^{\gamma L}]$

### 4.4 Why Deep Learning?

The optimal hedging policy $\pi^*$ is a function from states to actions. Neural networks provide:

1. **Universal approximation**: Can approximate any continuous function
2. **Automatic feature extraction**: Learn relevant features from raw inputs
3. **Scalability**: Handle high-dimensional state spaces
4. **Flexibility**: No need to specify functional form

---

## 5. Reinforcement Learning Formulation

### 5.1 Markov Decision Process (MDP)

Deep hedging can be formulated as an MDP:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

where:
- $\mathcal{S}$: state space
- $\mathcal{A}$: action space
- $P(s' | s, a)$: transition probability
- $R(s, a, s')$: reward function
- $\gamma$: discount factor (typically 1 for episodic hedging)

**State:** $s_n = (S_n, t_n, \delta_{n-1}, \text{PnL}_n, \text{Greeks}_n, \ldots)$

**Action:** $a_n = \delta_n$ (new position) or $a_n = \Delta\delta_n$ (trade size)

**Reward:** There are two natural choices:

1. **Terminal reward only:**
   $$
   R_n = \begin{cases}
   -\rho(\text{P\&L}_T) & \text{if } n = N \\
   0 & \text{otherwise}
   \end{cases}
   $$

2. **Step-by-step reward:**
   $$
   R_n = \delta_{n-1}(S_n - S_{n-1}) - C(\Delta\delta_{n-1}, S_{n-1})
   $$

### 5.2 Policy Gradient Methods

The policy $\pi_\theta(a | s)$ is parameterised by a neural network. We optimise using policy gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_{n=0}^{N-1} \nabla_\theta \log \pi_\theta(a_n | s_n) \cdot A_n \right]
$$

where $A_n$ is the **advantage function**.

**REINFORCE Algorithm:**
```
For each episode:
    Sample trajectory τ = (s_0, a_0, r_0, ..., s_N)
    Compute return G = Σ r_n
    Update θ ← θ + α · ∇_θ log π_θ(τ) · G
```

### 5.3 Actor-Critic Methods

For lower variance, use an actor-critic architecture:

- **Actor** $\pi_\theta(a | s)$: outputs actions
- **Critic** $V_\phi(s)$: estimates value function

**Advantage estimate:**
$$
A_n = R_n + V_\phi(s_{n+1}) - V_\phi(s_n)
$$

### 5.4 Deterministic Policy Gradient (DPG)

For continuous action spaces (hedge ratios), deterministic policies are often preferred:

$$
\pi_\theta: \mathcal{S} \to \mathcal{A}
$$

**DPG Theorem (Silver et al.):**
$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \pi_\theta(s) \cdot \nabla_a Q(s, a)|_{a=\pi_\theta(s)} \right]
$$

### 5.5 Direct Risk Minimisation

For deep hedging, we can bypass RL machinery and directly minimise the risk measure using backpropagation through the hedging simulation:

$$
\nabla_\theta \rho(-\text{P\&L}_T^\theta) = \nabla_\theta \rho\left( -V_0 - \sum_{n} \pi_\theta(s_n) \cdot \Delta S_n + \sum_n C(\cdot) + \Phi(S_T) \right)
$$

This is the approach taken in Bühler et al. (2019) and is often more efficient than RL for this specific problem.

---

## 6. Risk Measures and Objective Functions

### 6.1 Coherent Risk Measures

A risk measure $\rho: L^p(\Omega) \to \mathbb{R}$ is **coherent** if it satisfies:

1. **Monotonicity:** $X \leq Y \Rightarrow \rho(X) \leq \rho(Y)$
2. **Translation invariance:** $\rho(X + c) = \rho(X) + c$
3. **Positive homogeneity:** $\rho(\lambda X) = \lambda \rho(X)$ for $\lambda > 0$
4. **Subadditivity:** $\rho(X + Y) \leq \rho(X) + \rho(Y)$

### 6.2 Variance-Based Objectives

**Mean-Variance:**
$$
\rho_{\text{MV}}(L) = \mathbb{E}[L] + \lambda \cdot \text{Var}(L)
$$

**Pros:** Simple, differentiable, interpretable  
**Cons:** Penalises upside and downside equally, not coherent

**Quadratic Utility:**
$$
U(x) = x - \frac{\lambda}{2} x^2 \quad \Rightarrow \quad \max_\theta \mathbb{E}[U(\text{P\&L})]
$$

### 6.3 Conditional Value-at-Risk (CVaR)

**Value-at-Risk (VaR):**
$$
\text{VaR}_\alpha(L) = \inf\{l : \mathbb{P}(L \leq l) \geq \alpha\}
$$

**CVaR (Expected Shortfall):**
$$
\text{CVaR}_\alpha(L) = \mathbb{E}[L | L \geq \text{VaR}_\alpha(L)]
$$

**Rockafellar-Uryasev Representation:**
$$
\text{CVaR}_\alpha(L) = \min_{\nu} \left\{ \nu + \frac{1}{1-\alpha} \mathbb{E}[(L - \nu)^+] \right\}
$$

This representation is crucial for optimisation: it converts CVaR into a smooth objective.

**Gradient for CVaR:**
$$
\nabla_\theta \text{CVaR}_\alpha(L^\theta) = \frac{1}{1-\alpha} \mathbb{E}\left[ \nabla_\theta L^\theta \cdot \mathbf{1}_{L^\theta \geq \text{VaR}_\alpha(L^\theta)} \right]
$$

### 6.4 Exponential Utility (Entropic Risk Measure)

**Exponential Utility:**
$$
U_\gamma(x) = -\exp(-\gamma x)
$$

**Certainty Equivalent (Entropic Risk Measure):**
$$
\rho_\gamma(L) = \frac{1}{\gamma} \log \mathbb{E}[e^{\gamma L}]
$$

**Properties:**
- Coherent (subadditive and monotone)
- Penalises tails more heavily than variance
- Risk aversion controlled by $\gamma$

**Gradient:**
$$
\nabla_\theta \rho_\gamma(L^\theta) = \frac{\mathbb{E}[e^{\gamma L^\theta} \cdot \nabla_\theta L^\theta]}{\mathbb{E}[e^{\gamma L^\theta}]}
$$

### 6.5 Comparison of Risk Measures

| Risk Measure | Coherent | Tail Sensitivity | Computation |
|--------------|----------|------------------|-------------|
| Variance | No | Low (quadratic) | Easy |
| CVaR | Yes | High | Medium |
| Entropic | Yes | Very high | Easy |

---

## 7. Transaction Costs and Market Frictions

### 7.1 Types of Transaction Costs

**Proportional Costs (Bid-Ask Spread):**
$$
C_{\text{prop}}(\Delta\delta, S) = \kappa \cdot S \cdot |\Delta\delta|
$$

where $\kappa$ is the half-spread (e.g., 0.01% = 1 bp).

**Fixed Costs:**
$$
C_{\text{fixed}}(\Delta\delta) = c \cdot \mathbf{1}_{\Delta\delta \neq 0}
$$

**Market Impact (Temporary):**
$$
C_{\text{impact}}(\Delta\delta, S) = \lambda \cdot S \cdot |\Delta\delta|^\alpha
$$

Typical values: $\alpha \in [1, 2]$, with $\alpha = 1.5$ (square-root impact) common.

**Market Impact (Permanent):**
$$
S^{\text{after}} = S^{\text{before}} + \eta \cdot \Delta\delta
$$

### 7.2 Combined Cost Model

A realistic cost model combines multiple components:

$$
C(\Delta\delta, S, \sigma, V) = \underbrace{\kappa S |\Delta\delta|}_{\text{spread}} + \underbrace{\lambda S |\Delta\delta|^{1.5}}_{\text{impact}} + \underbrace{c \cdot \mathbf{1}_{\Delta\delta \neq 0}}_{\text{fixed}}
$$

where costs may depend on:
- Asset price $S$
- Volatility $\sigma$
- Volume $V$
- Time of day

### 7.3 Optimal Trade Size

With quadratic impact, the optimal trade size to move from $\delta_{\text{curr}}$ to $\delta_{\text{target}}$ is not necessarily $\delta_{\text{target}} - \delta_{\text{curr}}$.

**Patient trading:** Split large orders over time to reduce impact.

**Urgency parameter:** Trade-off between tracking error and impact.

### 7.4 Discrete Rehedging

With discrete rehedging at intervals $\Delta t$, the hedging error accumulates:

$$
\text{Error} = \sum_{n} \int_{t_n}^{t_{n+1}} (\delta^* - \delta_n) \, dS_u
$$

where $\delta^*$ is the continuous-time optimal hedge.

**Variance of Hedging Error (BSM, delta hedge):**
$$
\text{Var}(\text{Error}) \approx \frac{1}{2} \Gamma^2 S^4 \sigma^4 T \cdot \Delta t
$$

This scales linearly with $\Delta t$, making frequent rehedging desirable in the absence of costs.

---

## 8. Neural Network Architectures

### 8.1 Feedforward Networks (MLP)

The simplest architecture for hedging policies:

$$
\pi_\theta(s) = W_L \cdot \phi(W_{L-1} \cdot \phi(\cdots \phi(W_1 \cdot s + b_1) \cdots) + b_{L-1}) + b_L
$$

where $\phi$ is an activation function (ReLU, tanh, etc.).

**Input features:**
- Spot price $S_t$ (or log-moneyness $\log(S_t/K)$)
- Time to maturity $\tau = T - t$
- Current position $\delta_{t-1}$
- Greeks: $\Delta_t^{\text{BS}}, \Gamma_t^{\text{BS}}, \mathcal{V}_t^{\text{BS}}$
- Running P&L
- Implied volatility

**Output:**
- New position $\delta_t$ (continuous)
- Or action clipped to $[-\delta_{\max}, \delta_{\max}]$

### 8.2 Recurrent Networks (LSTM/GRU)

For path-dependent features or when the optimal hedge depends on the history:

$$
h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1})
$$
$$
\delta_t = W \cdot h_t + b
$$

**Advantages:**
- Can learn complex path-dependent strategies
- Natural for sequential decision-making
- Can capture regime changes

**Disadvantages:**
- Harder to train
- Slower inference
- May overfit to spurious patterns

### 8.3 Attention Mechanisms

Attention allows the network to focus on relevant parts of the history:

$$
\alpha_n = \text{softmax}\left( \frac{Q \cdot K_n^T}{\sqrt{d}} \right)
$$
$$
\text{context} = \sum_n \alpha_n \cdot V_n
$$

### 8.4 Architecture Recommendations

| Setting | Recommended Architecture |
|---------|-------------------------|
| Simple GBM, short maturity | MLP (2-3 layers, 64-128 units) |
| Stochastic vol, path-dependent | LSTM (1-2 layers, 64 units) |
| Multi-asset portfolio | MLP with attention or GNN |

---

## 9. Training and Convergence

### 9.1 Training Procedure

**Algorithm: Deep Hedging Training**

```
Input: Initial policy parameters θ, risk measure ρ, market simulator
Output: Trained policy parameters θ*

1. For epoch = 1 to max_epochs:
   a. Sample M market paths: {S^(m)}_{m=1}^M
   b. For each path m:
      i.   Initialise state s_0 = (S_0, T, 0, 0, ...)
      ii.  For n = 0 to N-1:
           - Compute action δ_n = π_θ(s_n)
           - Simulate next state s_{n+1}
           - Accumulate P&L
      iii. Compute terminal P&L^(m)
   c. Compute risk: L = ρ({-P&L^(m)})
   d. Compute gradient: g = ∇_θ L
   e. Update parameters: θ ← θ - α · g

2. Return θ
```

### 9.2 Gradient Computation

For direct risk minimisation, gradients flow through:

1. Policy network $\pi_\theta$
2. Position updates
3. P&L computation
4. Risk measure

**Key insight:** The market simulation (path generation) does not depend on $\theta$, so we only backpropagate through the policy and P&L.

### 9.3 Variance Reduction

**Antithetic variates:** For each path $S^{(m)}$ generated from normals $Z$, also use $-Z$ to generate an antithetic path.

**Control variates:** Use the BSM delta-hedge P&L as a control:
$$
\text{P\&L}^{\text{CV}} = \text{P\&L}^{\pi_\theta} - \beta(\text{P\&L}^{\Delta} - \mathbb{E}[\text{P\&L}^{\Delta}])
$$

**Importance sampling:** Emphasise tail scenarios for CVaR training.

### 9.4 Convergence Considerations

**Theorem (Universal Approximation):** For any continuous hedging policy $\pi^*$ and $\epsilon > 0$, there exists a neural network $\pi_\theta$ such that $\|\pi_\theta - \pi^*\|_\infty < \epsilon$.

**In practice:**
- Convergence is not guaranteed to global optimum
- Local minima may still be "good enough"
- Initialisation matters: start near delta hedge
- Learning rate scheduling helps

### 9.5 Hyperparameter Selection

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| Learning rate | 1e-4 to 1e-2 | Decay over training |
| Batch size | 256 to 4096 | Larger = lower variance |
| Hidden layers | 2-4 | More for complex problems |
| Hidden units | 32-256 | Start small |
| Epochs | 100-1000 | Monitor convergence |

---

## 10. Theoretical Analysis

### 10.1 Optimality Conditions

**Theorem:** Under suitable regularity conditions, the optimal deep hedging policy $\pi^*$ satisfies:

$$
\nabla_\delta \rho\left( -\text{P\&L}(\pi^*) \right) = 0
$$

For variance-based risk:
$$
\mathbb{E}[\Delta S_n] = \lambda \cdot \mathbb{E}\left[ \frac{\partial \text{P\&L}}{\partial \delta_n} \cdot (\text{P\&L} - \mathbb{E}[\text{P\&L}]) \right]
$$

### 10.2 Comparison to Delta Hedging

**Theorem (Bühler et al.):** In the BSM model with no transaction costs and continuous trading, the deep hedging policy converges to the BSM delta as the network capacity increases.

**With transaction costs:** Deep hedging learns a policy that:
1. Has a no-trade region (like optimal control theory predicts)
2. Width of no-trade region adapts to local conditions
3. Trades more aggressively near expiry (gamma increases)

### 10.3 Model-Free Properties

**Key advantage:** Deep hedging does not require knowledge of the true data-generating process. It optimises over the empirical distribution of paths.

**Robustness:** If trained on a distribution that includes model uncertainty, the learned policy is robust to model misspecification.

### 10.4 Generalisation

**In-sample vs out-of-sample:**
- Training: Minimise risk on simulated paths
- Testing: Evaluate on held-out paths (or real data)

**Regime changes:** If the test regime differs significantly from training, performance may degrade. Solutions:
- Train on diverse scenarios
- Include regime as a state variable
- Ensemble of policies for different regimes

---

## 11. Literature Review

### 11.1 Foundational Papers

**Bühler, H., Gonon, L., Teichmann, J., & Wood, B. (2019).** *Deep hedging.* Quantitative Finance, 19(8), 1271-1291.

- Introduced the deep hedging framework
- Demonstrated superiority over delta hedging with transaction costs
- Used feedforward networks and variance-based risk

**Horvath, B., Muguruza, A., & Tomas, M. (2021).** *Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models.* Quantitative Finance, 21(1), 11-27.

- Extended deep hedging to rough volatility models
- Showed neural networks can capture complex volatility dynamics

### 11.2 Extensions

**Buehler, H., Gonon, L., Teichmann, J., Wood, B., Mohan, B., & Kochems, J. (2019).** *Deep hedging: Hedging derivatives under generic market frictions using reinforcement learning.*

- RL formulation of deep hedging
- Comparison of different RL algorithms

**Cao, J., Chen, J., Hull, J., & Poulos, Z. (2021).** *Deep hedging of derivatives using reinforcement learning.* Journal of Financial Data Science.

- Practical implementation considerations
- Comparison with analytical solutions

### 11.3 Related Work

**Kolm, P. N., & Ritter, G. (2019).** *Dynamic replication and hedging: A reinforcement learning approach.* Journal of Financial Data Science.

- Focus on execution and replication
- Continuous action spaces

**Carbonneau, A. (2021).** *Deep hedging of long-term financial derivatives.* Insurance: Mathematics and Economics.

- Extension to long-dated products
- Handling of interest rate dynamics

---

## 12. References

1. Bühler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance*, 19(8), 1271-1291.

2. Horvath, B., Muguruza, A., & Tomas, M. (2021). Deep learning volatility. *Quantitative Finance*, 21(1), 11-27.

3. Föllmer, H., & Schweizer, M. (1991). Hedging of contingent claims under incomplete information. *Applied Stochastic Analysis*, 5, 389-414.

4. Hodges, S. D., & Neuberger, A. (1989). Optimal replication of contingent claims under transaction costs. *Review of Futures Markets*, 8(2), 222-239.

5. Whalley, A. E., & Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option pricing with transaction costs. *Mathematical Finance*, 7(3), 307-324.

6. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2, 21-42.

7. Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic policy gradient algorithms. *ICML*.

8. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

9. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.

10. Shreve, S. E. (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer.

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $S_t$ | Spot price at time $t$ |
| $K$ | Strike price |
| $T$ | Maturity |
| $\sigma$ | Volatility |
| $r$ | Risk-free rate |
| $\delta_t$ | Hedge position at time $t$ |
| $\pi_\theta$ | Policy parameterised by $\theta$ |
| $\rho$ | Risk measure |
| $C(\cdot)$ | Transaction cost function |
| $\Phi(S_T)$ | Option payoff |
| $\Delta, \Gamma, \mathcal{V}$ | Greeks (delta, gamma, vega) |
| $\mathbb{P}$ | Physical probability measure |
| $\mathbb{Q}$ | Risk-neutral measure |
| $\mathcal{F}_t$ | Information filtration at time $t$ |

---

## Appendix B: Implementation Checklist

- [ ] Market simulator (GBM, Heston, historical)
- [ ] Transaction cost model (proportional, impact)
- [ ] Risk measure implementation (variance, CVaR, entropic)
- [ ] Policy network (MLP, optional LSTM)
- [ ] Training loop with gradient computation
- [ ] Delta hedging benchmark
- [ ] Evaluation metrics (P&L distribution, Sharpe, cost breakdown)
- [ ] Visualisation (P&L histogram, hedge position over time)

---

*Document prepared for QuantStrata Phase 7.6: Deep Hedging & Neural Optimal Control*
