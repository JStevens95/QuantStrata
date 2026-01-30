# Local Volatility Model: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Model Class:** Deterministic Volatility Model  
**Pricing Methods:** Finite Difference (PDE), Monte Carlo  
**Target Audience:** Quantitative Analysts, Financial Mathematics Graduates

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Formal Mathematical Framework](#2-formal-mathematical-framework)
3. [Dupire's Formula](#3-dupires-formula)
4. [Relationship to Implied Volatility](#4-relationship-to-implied-volatility)
5. [Numerical Implementation](#5-numerical-implementation)
6. [Calibration](#6-calibration)
7. [Greeks and Sensitivities](#7-greeks-and-sensitivities)
8. [Advantages and Limitations](#8-advantages-and-limitations)
9. [Implementation](#9-implementation)
10. [Key Interview Points](#10-key-interview-points)
11. [References](#11-references)

---

## 1. Executive Summary

### 1.1 Model Overview

The **Local Volatility Model** extends Black-Scholes by allowing volatility to be a deterministic function of both spot price and time: σ = σ(S, t). This single modification creates a powerful framework that:

- Fits all vanilla option prices exactly
- Remains arbitrage-free by construction
- Provides a complete market model

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Volatility** | Deterministic function σ(S, t) |
| **Market Completeness** | Complete (perfect replication) |
| **Calibration** | Exact fit to vanilla surface |
| **Complexity** | Single-factor diffusion |
| **Primary Use Cases** | Exotic pricing, vol surface interpolation |

### 1.3 Model Comparison

| Model | Vol Structure | Market Fit | Exotic Pricing |
|-------|--------------|------------|----------------|
| Black-Scholes | Constant σ | Poor | Baseline |
| Local Vol | σ(S, t) | Exact vanillas | Good for Europeans |
| Stochastic Vol | σ_t random | Calibrated | Smile dynamics |

---

## 2. Formal Mathematical Framework

### 2.1 Probability Space and Filtration

Let \((\Omega, \mathcal{F}, \mathbb{P})\) be a probability space with filtration \(\{\mathcal{F}_t\}_{t \geq 0}\) satisfying the usual conditions.

### 2.2 Asset Dynamics

Under the risk-neutral measure \(\mathbb{Q}\):

\[
dS_t = (r - q) S_t \, dt + \sigma(S_t, t) S_t \, dW_t^{\mathbb{Q}}
\]

where:
- \(S_t\): Spot price at time \(t\)
- \(r\): Domestic risk-free rate (constant)
- \(q\): Foreign rate / dividend yield (constant)
- \(\sigma(S, t)\): **Local volatility function**
- \(W_t^{\mathbb{Q}}\): Standard Brownian motion under \(\mathbb{Q}\)

### 2.3 Key Properties

**1. Diffusion Coefficient:**
\[
a(S, t) = \sigma(S, t) \cdot S
\]

**2. Infinitesimal Generator:**
\[
\mathcal{L} = \frac{\partial}{\partial t} + (r-q)S\frac{\partial}{\partial S} + \frac{1}{2}\sigma^2(S,t)S^2\frac{\partial^2}{\partial S^2}
\]

**3. Fokker-Planck Equation:**

The transition density \(p(S, t; S_0, t_0)\) satisfies:
\[
\frac{\partial p}{\partial t} = -\frac{\partial}{\partial S}[(r-q)Sp] + \frac{1}{2}\frac{\partial^2}{\partial S^2}[\sigma^2(S,t)S^2 p]
\]

### 2.4 Pricing PDE

The value \(V(S, t)\) of a derivative satisfies the **Generalized Black-Scholes PDE**:

\[
\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2(S,t)S^2\frac{\partial^2 V}{\partial S^2} - rV = 0
\]

with terminal condition \(V(S, T) = \text{payoff}(S)\).

---

## 3. Dupire's Formula

### 3.1 The Fundamental Result

**Theorem (Dupire, 1994):**

Given a continuum of European call prices \(C(K, T)\) for all strikes \(K > 0\) and maturities \(T > 0\), there exists a unique local volatility function \(\sigma_{LV}(K, T)\) such that:

\[
\sigma_{LV}^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}
\]

### 3.2 Derivation

**Step 1: Forward PDE**

European call prices satisfy the **Dupire PDE** (forward in strike and maturity):

\[
\frac{\partial C}{\partial T} = \frac{1}{2}\sigma_{LV}^2(K,T)K^2\frac{\partial^2 C}{\partial K^2} - (r-q)K\frac{\partial C}{\partial K} - qC
\]

**Step 2: Rearranging**

Solving for \(\sigma_{LV}^2\):

\[
\sigma_{LV}^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}
\]

### 3.3 Alternative Formulations

**In terms of implied volatility \(\sigma_{BS}(K, T)\):**

\[
\sigma_{LV}^2 = \frac{\frac{\partial w}{\partial T}}{1 - \frac{y}{w}\frac{\partial w}{\partial y} + \frac{1}{4}\left(-\frac{1}{4} - \frac{1}{w} + \frac{y^2}{w^2}\right)\left(\frac{\partial w}{\partial y}\right)^2 + \frac{1}{2}\frac{\partial^2 w}{\partial y^2}}
\]

where:
- \(w = \sigma_{BS}^2 T\) (total variance)
- \(y = \ln(K/F)\) (log-moneyness)
- \(F = S_0 e^{(r-q)T}\) (forward price)

### 3.4 Intuition

- **Numerator**: Measures how call prices change with maturity
- **Denominator**: Related to the probability density (Breeden-Litzenberger)
- **At ATM**: \(\sigma_{LV} \approx \sigma_{BS}\) when smile is flat

---

## 4. Relationship to Implied Volatility

### 4.1 Local Vol vs Implied Vol

| Aspect | Local Vol σ_LV(K, T) | Implied Vol σ_BS(K, T) |
|--------|---------------------|----------------------|
| **Definition** | Instantaneous vol at (S=K, t=T) | Vol that matches BS to market price |
| **Dependency** | Function of spot and time | Function of strike and maturity |
| **Uniqueness** | Unique from market prices | Unique per option |
| **Smile Interpretation** | Determines smile dynamics | Describes current smile |

### 4.2 Key Relationship

For European options, local vol and implied vol are connected via:

\[
\sigma_{BS}^2(K, T) \cdot T = \int_0^T \mathbb{E}^{\mathbb{Q}}[\sigma_{LV}^2(S_t, t) \mid S_T = K] \, dt
\]

**Interpretation**: Implied variance is the expected integrated local variance along paths ending at \(K\).

### 4.3 ATM Relationship

At the ATM forward strike \(K = F\):

\[
\sigma_{LV}(F, T) \approx \sigma_{BS}(F, T) + O(\sigma^2 T)
\]

For short maturities or flat smile, local vol ≈ implied vol at ATM.

---

## 5. Numerical Implementation

### 5.1 Finite Difference Method

**Grid Setup:**

1. **Spatial grid**: \(S_i\) for \(i = 0, ..., N_S\)
2. **Time grid**: \(t_n\) for \(n = 0, ..., N_T\)
3. **Local vol evaluation**: \(\sigma_{i,n} = \sigma_{LV}(S_i, t_n)\)

**Discretized PDE (Crank-Nicolson):**

\[
\frac{V_i^{n} - V_i^{n+1}}{\Delta t} = \frac{1}{2}(\mathcal{L}^n V^n + \mathcal{L}^{n+1} V^{n+1})
\]

where the operator \(\mathcal{L}^n\) uses \(\sigma_{i,n}\).

**Key Difference from Constant Vol:**

At each time step, recompute the tridiagonal coefficients using the **time-dependent local volatility** at that time level.

### 5.2 Monte Carlo Simulation

**Euler Scheme:**

\[
S_{n+1} = S_n \exp\left[\left(r - q - \frac{1}{2}\sigma_{LV}^2(S_n, t_n)\right)\Delta t + \sigma_{LV}(S_n, t_n)\sqrt{\Delta t} \, Z_n\right]
\]

**Implementation Notes:**

1. At each step, query \(\sigma_{LV}(S_n, t_n)\) from the surface
2. Use log-space for positivity preservation
3. Interpolate if \(S_n\) is off the calibrated grid

### 5.3 Grid Construction for Local Vol Surface

**Recommended Grids:**

```python
# Time grid (denser at short end)
times = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

# Spot grid (centered around ATM, wider for tails)
spot_range = spot * np.array([0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5])
```

---

## 6. Calibration

### 6.1 From Market Data

**Input**: Market implied volatilities \(\sigma_{BS}(K_i, T_j)\)

**Process:**
1. Convert implied vols to call prices using Black-Scholes
2. Compute numerical derivatives \(\partial C/\partial T\), \(\partial C/\partial K\), \(\partial^2 C/\partial K^2\)
3. Apply Dupire's formula at each grid point
4. Handle edge cases (arbitrage, numerical instability)

### 6.2 Numerical Derivatives

**Time derivative (forward difference):**
\[
\frac{\partial C}{\partial T} \approx \frac{C(K, T + \Delta T) - C(K, T)}{\Delta T}
\]

**Strike derivatives (central difference):**
\[
\frac{\partial C}{\partial K} \approx \frac{C(K + \Delta K, T) - C(K - \Delta K, T)}{2\Delta K}
\]
\[
\frac{\partial^2 C}{\partial K^2} \approx \frac{C(K + \Delta K, T) - 2C(K, T) + C(K - \Delta K, T)}{\Delta K^2}
\]

### 6.3 Handling Edge Cases

**Negative Local Variance:**

If \(\sigma_{LV}^2 < 0\), this indicates **calendar arbitrage** in the input surface.

**Solution:**
1. Clamp to minimum: \(\sigma_{LV} = \max(\sigma_{LV}, \sigma_{min})\)
2. Or interpolate from neighboring valid points
3. Or re-arbitrage the input surface

**Numerical Instability:**

Near ATM with short maturity, the denominator \(\partial^2 C/\partial K^2\) can be very small.

**Solution:**
1. Use implied vol formulation instead
2. Add regularization
3. Fall back to implied vol at very short maturities

---

## 7. Greeks and Sensitivities

### 7.1 Delta

\[
\Delta = \frac{\partial V}{\partial S}
\]

Under local vol, delta differs from Black-Scholes because the volatility depends on spot.

### 7.2 Gamma

\[
\Gamma = \frac{\partial^2 V}{\partial S^2}
\]

Gamma captures the curvature and is affected by the local vol surface shape.

### 7.3 Vega (Local Vega)

Traditional vega assumes a parallel shift in implied vol. Under local vol:

**Local Vega:** Sensitivity to shifting the entire local vol surface.

\[
\text{Vega}_{LV} = \frac{\partial V}{\partial \sigma_{LV}}
\]

This is computed via bump-and-reprice on the local vol surface.

### 7.4 Vanna and Volga

These higher-order Greeks are more complex under local vol due to the spot-dependent volatility structure.

---

## 8. Advantages and Limitations

### 8.1 Advantages

1. **Exact Vanilla Fit**: Prices all vanillas correctly by construction
2. **Arbitrage-Free**: Guaranteed no-arbitrage for vanillas
3. **Single Factor**: Computationally tractable
4. **Market Standard**: Widely used benchmark

### 8.2 Limitations

1. **Smile Dynamics**: Predicts smile flattening with time (unrealistic)
2. **Forward Smile**: May give poor forward-starting option prices
3. **Barrier Options**: Can misprice due to incorrect smile dynamics
4. **Calibration Sensitivity**: Sensitive to input surface quality

### 8.3 When to Use Local Vol

**Good For:**
- European exotic pricing benchmark
- Vol surface interpolation
- Initial calibration for more complex models

**Less Suitable For:**
- Products sensitive to smile dynamics
- Forward-starting options
- Products with long forward dates

---

## 9. Implementation

### 9.1 Pseudocode: Dupire Calibration

```python
def calibrate_dupire(implied_surface, spot, r, q, times, strikes):
    """Calibrate local vol surface from implied vol surface."""
    
    local_vols = np.zeros((len(times), len(strikes)))
    dT = 1/252  # Time bump
    dK_pct = 0.01  # Strike bump percentage
    
    for i, T in enumerate(times):
        for j, K in enumerate(strikes):
            dK = K * dK_pct
            
            # Get call prices via BS
            C = bs_call(spot, K, T, r, q, implied_surface(K, T))
            C_T_up = bs_call(spot, K, T + dT, r, q, implied_surface(K, T + dT))
            C_K_up = bs_call(spot, K + dK, T, r, q, implied_surface(K + dK, T))
            C_K_dn = bs_call(spot, K - dK, T, r, q, implied_surface(K - dK, T))
            
            # Numerical derivatives
            dC_dT = (C_T_up - C) / dT
            dC_dK = (C_K_up - C_K_dn) / (2 * dK)
            d2C_dK2 = (C_K_up - 2*C + C_K_dn) / dK**2
            
            # Dupire formula
            numerator = dC_dT + (r - q) * K * dC_dK + q * C
            denominator = 0.5 * K**2 * d2C_dK2
            
            if denominator > 0:
                local_vols[i, j] = np.sqrt(numerator / denominator)
            else:
                local_vols[i, j] = implied_surface(K, T)  # Fallback
    
    return LocalVolSurface(times, strikes, local_vols)
```

### 9.2 Pseudocode: Local Vol FD Pricer

```python
def price_local_vol_fd(payoff, local_vol_surface, spot, K, T, r, q, n_space, n_time):
    """Price option using FD with local volatility."""
    
    # Setup grids
    x = np.linspace(log_S_min, log_S_max, n_space)  # Log-space
    dt = T / n_time
    
    # Terminal condition
    S = np.exp(x)
    V = payoff(S)
    
    # Backward time-stepping
    for n in range(n_time, 0, -1):
        t = n * dt
        
        # Get local vol at each grid point
        sigma = np.array([local_vol_surface(s, t) for s in S])
        
        # Build tridiagonal system with space-dependent sigma
        # ... (standard FD scheme with sigma[i] at each point)
        
        V = solve_tridiagonal(A, V)
    
    return np.interp(np.log(spot), x, V)
```

---

## 10. Key Interview Points

### 10.1 Must-Know Facts

1. **Definition**: Local vol σ(S, t) is deterministic, not stochastic
2. **Dupire**: Unique local vol from vanilla prices via Dupire's formula
3. **Completeness**: Local vol model is complete (unique replication)
4. **ATM**: At ATM forward, local vol ≈ implied vol
5. **Limitation**: Predicts flattening smile dynamics (unrealistic)

### 10.2 Common Interview Questions

**Q1: What's the difference between local vol and implied vol?**

A: Implied vol is the constant vol that prices a single option via BS. Local vol is the instantaneous vol at each (spot, time) point that prices ALL vanillas consistently. Local vol is a 2D surface; implied vol is just a market-derived number per option.

**Q2: How does Dupire's formula work?**

A: Dupire showed that given all vanilla call prices C(K,T), the unique local vol is:
\[
\sigma_{LV}^2 = \frac{\partial C/\partial T + (r-q)K \partial C/\partial K + qC}{\frac{1}{2}K^2 \partial^2 C/\partial K^2}
\]
The numerator measures time value decay; the denominator is related to the risk-neutral density.

**Q3: Why can local vol give wrong prices for barriers?**

A: Local vol predicts that the smile flattens as time passes (all vols converge to ATM). But market smiles don't actually flatten this way. For barrier options sensitive to smile dynamics at future dates, local vol can be systematically wrong.

**Q4: What does it mean if Dupire gives negative local variance?**

A: It indicates calendar arbitrage in the input implied vol surface (the term structure violates no-arbitrage). You should either re-arbitrage the input surface or clamp the local vol to a minimum positive value.

### 10.3 Key Formulas to Remember

| Formula | Expression |
|---------|------------|
| **SDE** | \(dS_t = (r-q)S_t dt + \sigma(S_t, t)S_t dW_t\) |
| **Dupire Formula** | \(\sigma_{LV}^2 = \frac{\partial_T C + (r-q)K\partial_K C + qC}{\frac{1}{2}K^2\partial_{KK}C}\) |
| **Pricing PDE** | \(\partial_t V + (r-q)S\partial_S V + \frac{1}{2}\sigma^2 S^2 \partial_{SS} V - rV = 0\) |

---

## 11. References

### Academic Papers

1. **Dupire, B.** (1994). "Pricing with a Smile." Risk, 7(1), 18-20.
2. **Derman, E. and Kani, I.** (1994). "The Volatility Smile and Its Implied Tree." Risk, 7(2), 139-145.
3. **Gatheral, J.** (2006). "The Volatility Surface: A Practitioner's Guide."

### Textbooks

1. **Hull, J.** (2022). *Options, Futures, and Other Derivatives*, 11th ed.
2. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed.
3. **Rebonato, R.** (2004). *Volatility and Correlation*, 2nd ed.

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Author: QuantStrata Development Team*
