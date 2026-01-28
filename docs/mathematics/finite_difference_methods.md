# Finite Difference Methods in Derivatives Pricing

**Complete Mathematical Framework for PDE-Based Valuation**

This document provides a rigorous treatment of finite difference methods for solving the Black-Scholes PDE and its extensions, including stability analysis, boundary conditions, and American option handling.

---

## Table of Contents

1. [Introduction to PDE Methods](#1-introduction-to-pde-methods)
2. [Discretization Fundamentals](#2-discretization-fundamentals)
3. [Finite Difference Schemes](#3-finite-difference-schemes)
4. [Stability Analysis](#4-stability-analysis)
5. [Boundary Conditions](#5-boundary-conditions)
6. [The Tridiagonal System](#6-the-tridiagonal-system)
7. [American Options: Free Boundary Problem](#7-american-options-free-boundary-problem)
8. [Greeks from FD Solutions](#8-greeks-from-fd-solutions)
9. [Advanced Topics](#9-advanced-topics)
10. [Interview Key Points](#10-interview-key-points)

---

## 1. Introduction to PDE Methods

### 1.1 The Black-Scholes PDE

The price $V(S, t)$ of a European derivative satisfies:

$$
\frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2} = rV
$$

**Terminal Condition:** $V(S, T) = g(S)$ (payoff)

**Boundary Conditions:** Depend on option type (discussed in Section 5).

### 1.2 Change of Variables

Transform to remove variable coefficients:

**Let:**
- $\tau = T - t$ (time to maturity)
- $x = \ln(S)$ (log-price)
- $u(x, \tau) = V(e^x, T - \tau)$

**Transformed PDE:**
$$
\frac{\partial u}{\partial \tau} = \frac{\sigma^2}{2}\frac{\partial^2 u}{\partial x^2} + \left(r - \frac{\sigma^2}{2}\right)\frac{\partial u}{\partial x} - ru
$$

This is a **convection-diffusion-reaction** equation.

### 1.3 Why Finite Differences?

**Advantages:**
- Handles American options naturally
- Greeks computed directly from solution
- Efficient for low dimensions (1-2D)
- Deterministic (no sampling error)

**Disadvantages:**
- Curse of dimensionality (>3D impractical)
- Boundary condition sensitivity
- Stability constraints on grid

---

## 2. Discretization Fundamentals

### 2.1 Grid Construction

**Time Grid:** $0 = \tau_0 < \tau_1 < \cdots < \tau_M = T$
- $\Delta \tau = T/M$

**Space Grid:** $S_{\min} = S_0 < S_1 < \cdots < S_N = S_{\max}$
- Uniform: $\Delta S = (S_{\max} - S_{\min})/N$
- Non-uniform: Concentrate around strike

**Notation:**
$$
V_j^m \approx V(S_j, t_m)
$$

### 2.2 Taylor Series Approximations

**First Derivative (Central):**
$$
\frac{\partial V}{\partial S}\bigg|_{S_j} \approx \frac{V_{j+1} - V_{j-1}}{2\Delta S} + O(\Delta S^2)
$$

**First Derivative (Forward):**
$$
\frac{\partial V}{\partial S}\bigg|_{S_j} \approx \frac{V_{j+1} - V_j}{\Delta S} + O(\Delta S)
$$

**First Derivative (Backward):**
$$
\frac{\partial V}{\partial S}\bigg|_{S_j} \approx \frac{V_j - V_{j-1}}{\Delta S} + O(\Delta S)
$$

**Second Derivative:**
$$
\frac{\partial^2 V}{\partial S^2}\bigg|_{S_j} \approx \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2} + O(\Delta S^2)
$$

### 2.3 Truncation Error

The **local truncation error** is:
$$
\text{LTE} = O(\Delta \tau^p) + O(\Delta S^q)
$$

Where $p$ is temporal order and $q$ is spatial order.

---

## 3. Finite Difference Schemes

### 3.1 Explicit Scheme (FTCS)

**Forward Time, Central Space:**

$$
\frac{V_j^{m+1} - V_j^m}{\Delta \tau} = \frac{\sigma^2 S_j^2}{2}\frac{V_{j+1}^m - 2V_j^m + V_{j-1}^m}{\Delta S^2} + rS_j\frac{V_{j+1}^m - V_{j-1}^m}{2\Delta S} - rV_j^m
$$

**Rearranging:**
$$
V_j^{m+1} = a_j V_{j-1}^m + b_j V_j^m + c_j V_{j+1}^m
$$

Where:
$$
a_j = \frac{\Delta \tau}{2}\left(\frac{\sigma^2 S_j^2}{\Delta S^2} - \frac{rS_j}{\Delta S}\right)
$$
$$
b_j = 1 - \Delta \tau\left(\frac{\sigma^2 S_j^2}{\Delta S^2} + r\right)
$$
$$
c_j = \frac{\Delta \tau}{2}\left(\frac{\sigma^2 S_j^2}{\Delta S^2} + \frac{rS_j}{\Delta S}\right)
$$

**Properties:**
- **Order:** $O(\Delta \tau) + O(\Delta S^2)$
- **Stability:** Conditional (requires small $\Delta \tau$)
- **Matrix-free:** Direct computation

### 3.2 Implicit Scheme (BTCS)

**Backward Time, Central Space:**

$$
\frac{V_j^{m+1} - V_j^m}{\Delta \tau} = \frac{\sigma^2 S_j^2}{2}\frac{V_{j+1}^{m+1} - 2V_j^{m+1} + V_{j-1}^{m+1}}{\Delta S^2} + rS_j\frac{V_{j+1}^{m+1} - V_{j-1}^{m+1}}{2\Delta S} - rV_j^{m+1}
$$

**Matrix Form:**
$$
A\mathbf{V}^{m+1} = \mathbf{V}^m + \mathbf{b}
$$

Where $A$ is tridiagonal.

**Properties:**
- **Order:** $O(\Delta \tau) + O(\Delta S^2)$
- **Stability:** Unconditionally stable
- **Cost:** Tridiagonal solve per time step

### 3.3 Crank-Nicolson Scheme

**Average of Explicit and Implicit:**

$$
\frac{V_j^{m+1} - V_j^m}{\Delta \tau} = \frac{1}{2}\left[\mathcal{L}V^{m+1} + \mathcal{L}V^m\right]
$$

Where $\mathcal{L}$ is the spatial differential operator.

**Properties:**
- **Order:** $O(\Delta \tau^2) + O(\Delta S^2)$
- **Stability:** Unconditionally stable
- **Cost:** Tridiagonal solve per time step
- **Industry standard** for most applications

### 3.4 Scheme Comparison

| Scheme | Order | Stability | Cost per Step |
|--------|-------|-----------|---------------|
| Explicit | $O(\Delta \tau, \Delta S^2)$ | Conditional | $O(N)$ |
| Implicit | $O(\Delta \tau, \Delta S^2)$ | Unconditional | $O(N)$ solve |
| Crank-Nicolson | $O(\Delta \tau^2, \Delta S^2)$ | Unconditional | $O(N)$ solve |

---

## 4. Stability Analysis

### 4.1 Von Neumann Analysis

Assume solution of form:
$$
V_j^m = \xi^m e^{ikj\Delta S}
$$

**Amplification Factor:** $\xi = V_j^{m+1}/V_j^m$

**Stability Condition:** $|\xi| \leq 1$ for all $k$.

### 4.2 Explicit Scheme Stability

For the heat equation $\partial u/\partial t = D\partial^2 u/\partial x^2$:

$$
\xi = 1 - 4\lambda\sin^2\left(\frac{k\Delta x}{2}\right)
$$

Where $\lambda = D\Delta t/\Delta x^2$.

**Stability requires:** $\lambda \leq 1/2$

For Black-Scholes with $D = \sigma^2 S^2/2$:
$$
\Delta \tau \leq \frac{\Delta S^2}{\sigma^2 S_{\max}^2}
$$

This is the **CFL condition**.

### 4.3 Implicit Scheme Stability

$$
\xi = \frac{1}{1 + 4\lambda\sin^2(k\Delta x/2)}
$$

Since denominator $> 1$, we have $|\xi| < 1$ always.

**Unconditionally stable.**

### 4.4 Crank-Nicolson Stability

$$
\xi = \frac{1 - 2\lambda\sin^2(k\Delta x/2)}{1 + 2\lambda\sin^2(k\Delta x/2)}
$$

$|\xi| \leq 1$ for all $\lambda > 0$.

**Unconditionally stable** but may oscillate for large $\lambda$.

---

## 5. Boundary Conditions

### 5.1 Types of Boundary Conditions

**Dirichlet:** $V(S, t) = f(t)$ at boundary
**Neumann:** $\partial V/\partial S = g(t)$ at boundary
**Robin:** $\alpha V + \beta \partial V/\partial S = \gamma(t)$

### 5.2 European Call Boundaries

**At $S = 0$:**
$$
V(0, t) = 0
$$
(Zero value if asset worthless)

**At $S = S_{\max}$:**
$$
V(S_{\max}, t) \approx S_{\max} - Ke^{-r(T-t)}
$$
(Deep ITM call ≈ forward)

Or Neumann: $\frac{\partial^2 V}{\partial S^2} = 0$ (linear extrapolation)

### 5.3 European Put Boundaries

**At $S = 0$:**
$$
V(0, t) = Ke^{-r(T-t)}
$$

**At $S = S_{\max}$:**
$$
V(S_{\max}, t) = 0
$$

### 5.4 Implementation Details

**Ghost Nodes:** Extend grid by one node at each boundary for Neumann conditions.

**Linearity Condition:**
$$
\frac{V_N - V_{N-1}}{\Delta S} = \frac{V_{N-1} - V_{N-2}}{\Delta S}
$$

Gives $V_N = 2V_{N-1} - V_{N-2}$.

---

## 6. The Tridiagonal System

### 6.1 Matrix Structure

Implicit and Crank-Nicolson schemes lead to:

$$
\begin{pmatrix}
b_1 & c_1 & 0 & \cdots & 0 \\
a_2 & b_2 & c_2 & \cdots & 0 \\
0 & a_3 & b_3 & \ddots & \vdots \\
\vdots & \ddots & \ddots & \ddots & c_{N-1} \\
0 & \cdots & 0 & a_N & b_N
\end{pmatrix}
\begin{pmatrix}
V_1 \\
V_2 \\
V_3 \\
\vdots \\
V_N
\end{pmatrix}
=
\begin{pmatrix}
d_1 \\
d_2 \\
d_3 \\
\vdots \\
d_N
\end{pmatrix}
$$

### 6.2 Thomas Algorithm

**Forward Elimination:**
$$
c'_1 = \frac{c_1}{b_1}, \quad d'_1 = \frac{d_1}{b_1}
$$

For $i = 2, \ldots, N$:
$$
w = b_i - a_i c'_{i-1}
$$
$$
c'_i = \frac{c_i}{w}, \quad d'_i = \frac{d_i - a_i d'_{i-1}}{w}
$$

**Back Substitution:**
$$
V_N = d'_N
$$

For $i = N-1, \ldots, 1$:
$$
V_i = d'_i - c'_i V_{i+1}
$$

**Complexity:** $O(N)$ — linear in grid size.

### 6.3 Numerical Stability

**Diagonal Dominance:** If $|b_i| \geq |a_i| + |c_i|$, the algorithm is stable.

Implicit schemes typically satisfy this condition.

---

## 7. American Options: Free Boundary Problem

### 7.1 The Early Exercise Constraint

American option price satisfies:
$$
V(S, t) \geq g(S) \quad \text{(intrinsic value)}
$$

This is a **Linear Complementarity Problem (LCP):**
$$
\mathcal{L}V \leq 0, \quad V \geq g, \quad (V - g)\mathcal{L}V = 0
$$

### 7.2 The Free Boundary

**Exercise Region:** Where $V(S,t) = g(S)$
**Continuation Region:** Where $V(S,t) > g(S)$

The boundary between regions, $S^*(t)$, is unknown and must be found as part of the solution.

### 7.3 Penalty Method

Add penalty term to enforce constraint:
$$
\mathcal{L}V + \lambda\max(g - V, 0) = 0
$$

As $\lambda \to \infty$, solution approaches LCP solution.

### 7.4 Projected SOR (PSOR)

**Algorithm:** At each time step, solve:
$$
A\mathbf{V}^{new} = \mathbf{b}
$$
subject to $V_j^{new} \geq g(S_j)$.

**SOR Iteration:**
$$
V_j^{(k+1)} = \max\left(g_j, (1-\omega)V_j^{(k)} + \omega\cdot\text{GS update}\right)
$$

Where:
- $\omega$: Relaxation parameter (typically 1.0-1.5)
- GS update: Gauss-Seidel update

**Convergence:**
- Linear for $\omega = 1$
- Optimal $\omega$ depends on matrix structure
- Typically converges in 10-50 iterations

### 7.5 Brennan-Schwartz Algorithm

Special algorithm for American options:
1. Solve as if European
2. Check early exercise constraint
3. If violated, set $V_j = g_j$ and re-solve

More efficient than PSOR for simple payoffs.

---

## 8. Greeks from FD Solutions

### 8.1 Direct Extraction

Once $V_{j}^{m}$ is computed on the grid:

**Delta:**
$$
\Delta_j = \frac{V_{j+1} - V_{j-1}}{2\Delta S}
$$

**Gamma:**
$$
\Gamma_j = \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}
$$

**Theta:**
$$
\Theta_j = \frac{V_j^{m+1} - V_j^m}{\Delta \tau}
$$

### 8.2 Interpolation to Spot

Solution is on grid points. For actual spot $S_0$:

**Linear Interpolation:**
$$
V(S_0) = V_j + \frac{S_0 - S_j}{S_{j+1} - S_j}(V_{j+1} - V_j)
$$

**Cubic Spline:** Better accuracy, especially for gamma.

### 8.3 Vega and Rho

Requires solving with perturbed parameters:

**Vega:** $\nu = (V(\sigma + h) - V(\sigma - h))/(2h)$

**Rho:** $\rho = (V(r + h) - V(r - h))/(2h)$

---

## 9. Advanced Topics

### 9.1 Non-Uniform Grids

**Motivation:** Concentrate grid points near strike where gamma is highest.

**Transformation:** $\xi = \sinh^{-1}(\alpha(S - K))$

Maps uniform grid in $\xi$ to non-uniform grid in $S$.

**Benefits:**
- Better accuracy near strike
- Same computational cost
- Reduced grid size

### 9.2 Higher-Order Schemes

**Fourth-Order Spatial:**
$$
\frac{\partial^2 V}{\partial S^2} \approx \frac{-V_{j+2} + 16V_{j+1} - 30V_j + 16V_{j-1} - V_{j-2}}{12\Delta S^2}
$$

**Richardson Extrapolation:** Combine solutions at different $\Delta S$ to cancel leading error terms.

### 9.3 Multi-Dimensional Problems

**2D PDE (basket, stoch vol):**
$$
\frac{\partial V}{\partial t} + \mathcal{L}_1 V + \mathcal{L}_2 V + \mathcal{L}_{12}V = rV
$$

**ADI (Alternating Direction Implicit):**
Split into 1D problems:
1. Implicit in $S$, explicit in $v$
2. Implicit in $v$, explicit in $S$

**Complexity:** $O(N^2)$ per step (vs $O(N^4)$ for full implicit)

### 9.4 Barrier Options

**Approach 1: Adjust Grid**
Place grid point exactly on barrier.

**Approach 2: Absorbing Boundary**
Set $V = 0$ (knock-out) at barrier.

**Approach 3: Rebate**
Set $V = R \cdot e^{-r\tau}$ at barrier (with rebate $R$).

**Smoothing:** Barrier payoffs are discontinuous → oscillations. Use:
- Smoothed payoff near barrier
- Higher-order schemes
- Adaptive refinement

---

## 10. Interview Key Points

### Basic Questions

**Q: Explain explicit vs implicit schemes.**

A:
- **Explicit:** Known values at time $m$ → single value at $m+1$. No linear system to solve but conditionally stable.
- **Implicit:** Unknown values at time $m+1$ coupled → tridiagonal system. Unconditionally stable.

**Q: What is the CFL condition?**

A: For stability of explicit schemes:
$$
\Delta t \leq \frac{\Delta x^2}{2D}
$$
This limits time step size relative to grid spacing.

**Q: Why use Crank-Nicolson?**

A: Second-order in both time and space, unconditionally stable, industry standard for Black-Scholes.

### Advanced Questions

**Q: How do you handle American options with FD?**

A: At each time step, solve the LCP:
$$
\max(V - g, \mathcal{L}V) = 0
$$
using PSOR or penalty methods. Check exercise constraint after each implicit solve.

**Q: What's the free boundary?**

A: The curve $S^*(t)$ separating the exercise region ($V = g$) from the continuation region ($V > g$). It's part of the solution, not given a priori.

**Q: How do you get Greeks from FD?**

A: Direct finite differences on the solution grid:
- $\Delta = (V_{j+1} - V_{j-1})/(2\Delta S)$
- $\Gamma = (V_{j+1} - 2V_j + V_{j-1})/\Delta S^2$
- $\Theta = (V^{m+1} - V^m)/\Delta t$

**Q: Pros/cons vs Monte Carlo?**

A:
- **FD Pro:** Deterministic, direct Greeks, efficient for American
- **FD Con:** Curse of dimensionality, boundary sensitivity
- **MC Pro:** Any dimension, any payoff, parallelizable
- **MC Con:** Slow convergence, American is hard, noisy Greeks

---

## Appendix: Algorithm Pseudocode

### Crank-Nicolson for European Option

```python
def cn_european(S0, K, T, r, sigma, N, M, option_type):
    # Grid setup
    S_max = 3 * K
    dS = S_max / N
    dt = T / M
    S = np.linspace(0, S_max, N+1)
    
    # Terminal condition
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)
    
    # Coefficients
    j = np.arange(1, N)
    a = 0.25 * dt * (sigma**2 * j**2 - r * j)
    b = -0.5 * dt * (sigma**2 * j**2 + r)
    c = 0.25 * dt * (sigma**2 * j**2 + r * j)
    
    # Tridiagonal matrices
    A = np.diag(1 - b) + np.diag(-a[1:], -1) + np.diag(-c[:-1], 1)
    B = np.diag(1 + b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    
    # Time stepping
    for m in range(M):
        rhs = B @ V[1:N]
        # Apply boundary conditions
        rhs[0] += a[0] * (V_lower[m] + V_lower[m+1])
        rhs[-1] += c[-1] * (V_upper[m] + V_upper[m+1])
        # Solve
        V[1:N] = thomas_solve(A, rhs)
    
    return np.interp(S0, S, V)
```

### PSOR for American Option

```python
def psor_american(A, b, payoff, omega=1.2, tol=1e-10, max_iter=10000):
    n = len(b)
    V = np.maximum(payoff, b)  # Initial guess
    
    for iteration in range(max_iter):
        V_old = V.copy()
        
        for i in range(n):
            # Gauss-Seidel update
            sigma = b[i]
            if i > 0:
                sigma -= A[i, i-1] * V[i-1]
            if i < n-1:
                sigma -= A[i, i+1] * V_old[i+1]
            sigma /= A[i, i]
            
            # SOR relaxation
            V_gs = (1 - omega) * V_old[i] + omega * sigma
            
            # Projection (early exercise)
            V[i] = max(V_gs, payoff[i])
        
        # Convergence check
        if np.max(np.abs(V - V_old)) < tol:
            break
    
    return V
```

---

## References

1. Wilmott, P. "Paul Wilmott on Quantitative Finance"
2. Duffy, D.J. "Finite Difference Methods in Financial Engineering"
3. Tavella, D. & Randall, C. "Pricing Financial Instruments: The Finite Difference Method"
4. Strikwerda, J.C. "Finite Difference Schemes and PDEs"

---

*Document Version: 1.0 | QuantStrata Phase 1 | January 2026*
