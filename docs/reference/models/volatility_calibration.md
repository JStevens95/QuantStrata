# Volatility Surface Calibration

**Mathematical Theory and Implementation**

This document provides comprehensive mathematical background for volatility surface calibration methods implemented in QuantStrata.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [SABR Model](#2-sabr-model)
3. [Dupire Local Volatility](#3-dupire-local-volatility)
4. [Arbitrage Conditions](#4-arbitrage-conditions)
5. [Numerical Methods](#5-numerical-methods)
6. [Interview Key Points](#6-interview-key-points)

---

## 1. Introduction

### Why Volatility Calibration?

In options markets, we observe **implied volatilities** from traded option prices. However, for pricing exotic derivatives, we need consistent **volatility models** that:

1. **Reproduce market prices** - Calibrate to observed vanilla prices
2. **Extrapolate sensibly** - Provide vols for unquoted strikes/expiries
3. **Avoid arbitrage** - Ensure no free-lunch opportunities
4. **Support pricing** - Feed into Monte Carlo or PDE pricers

### Two Main Approaches

| Approach | Model | Use Case |
|----------|-------|----------|
| **Parametric** | SABR, SVI | Smile interpolation, fast recalibration |
| **Non-parametric** | Dupire Local Vol | Full surface, exotic pricing |

---

## 2. SABR Model

### 2.1 Model Dynamics

The SABR model (Hagan et al., 2002) describes stochastic volatility dynamics:

$$
\begin{aligned}
dF_t &= \sigma_t F_t^\beta \, dW_t^{(1)} \\
d\sigma_t &= \nu \sigma_t \, dW_t^{(2)} \\
dW_t^{(1)} \cdot dW_t^{(2)} &= \rho \, dt
\end{aligned}
$$

**Parameters:**
- $\alpha$ (α): Initial volatility level $\sigma_0$
- $\beta$ (β): CEV exponent, $\beta \in [0, 1]$
- $\rho$ (ρ): Correlation, $\rho \in (-1, 1)$
- $\nu$ (ν): Vol-of-vol, $\nu \geq 0$

### 2.2 Hagan's Implied Volatility Formula

For strike $K$ and expiry $T$, the Black-Scholes implied volatility is approximately:

$$
\sigma_{BS}(K) = \frac{\alpha}{(FK)^{(1-\beta)/2}} \cdot \frac{z}{x(z)} \cdot \left(1 + \varepsilon T\right) \cdot \frac{1}{D}
$$

Where:

**Log-moneyness term:**
$$
z = \frac{\nu}{\alpha} (FK)^{(1-\beta)/2} \ln\left(\frac{F}{K}\right)
$$

**Correction function:**
$$
x(z) = \ln\left[\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}\right]
$$

**Denominator correction:**
$$
D = 1 + \frac{(1-\beta)^2}{24} \ln^2\left(\frac{F}{K}\right) + \frac{(1-\beta)^4}{1920} \ln^4\left(\frac{F}{K}\right)
$$

**Time correction:**
$$
\varepsilon = \frac{(1-\beta)^2 \alpha^2}{24(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{(1-\beta)/2}} + \frac{(2-3\rho^2)\nu^2}{24}
$$

### 2.3 ATM Formula (K = F)

At-the-money, the formula simplifies to:

$$
\sigma_{ATM} = \frac{\alpha}{F^{1-\beta}} \left[1 + \left(\frac{(1-\beta)^2\alpha^2}{24 F^{2-2\beta}} + \frac{\rho\beta\nu\alpha}{4 F^{1-\beta}} + \frac{(2-3\rho^2)\nu^2}{24}\right) T\right]
$$

### 2.4 Parameter Interpretation

| Parameter | Effect on Smile |
|-----------|-----------------|
| $\alpha$ | Level of ATM vol |
| $\beta$ | Backbone of smile (vol vs spot relationship) |
| $\rho$ | Skew direction (ρ < 0 → left skew for equity) |
| $\nu$ | Smile curvature (convexity) |

### 2.5 Special Cases

**β = 1 (Log-normal):**
- Standard case for FX markets
- Forward is a geometric Brownian motion (with stochastic vol)

**β = 0 (Normal):**
- Useful for interest rates (can go negative)
- Forward is arithmetic Brownian motion

**β = 0.5 (CIR-like):**
- Square-root process for forward
- Sometimes used for commodities

### 2.6 Calibration Procedure

**Objective Function:**
$$
\min_{\alpha, \rho, \nu} \sum_{i} w_i \left(\sigma_{SABR}(K_i) - \sigma_{market}(K_i)\right)^2
$$

**Typical constraints:**
- $\alpha > 0$
- $-1 < \rho < 1$
- $\nu \geq 0$
- $\beta$ is usually **fixed** (not calibrated)

**Why fix β?**
1. Avoids over-parameterization
2. β and α are correlated in calibration
3. Different β values can give similar fits
4. Market convention: β = 1 for FX, β often lower for rates

---

## 3. Dupire Local Volatility

### 3.1 The Local Volatility Model

The local volatility model assumes:

$$
dS_t = (r - q) S_t \, dt + \sigma(S_t, t) S_t \, dW_t
$$

Where $\sigma(S, t)$ is a **deterministic function** of spot and time.

### 3.2 Dupire's Formula

Dupire (1994) showed that given European call prices $C(K, T)$, the unique local volatility is:

$$
\sigma_{LV}^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}
$$

### 3.3 Derivation Sketch

**Starting point:** The Fokker-Planck (forward Kolmogorov) equation for the risk-neutral density $p(S, t; S_0, 0)$:

$$
\frac{\partial p}{\partial t} = -\frac{\partial}{\partial S}\left[(r-q)Sp\right] + \frac{1}{2}\frac{\partial^2}{\partial S^2}\left[\sigma^2(S,t)S^2 p\right]
$$

**Connection to call prices:**
$$
C(K, T) = e^{-rT} \int_K^\infty (S - K) p(S, T) \, dS
$$

**Differentiating twice with respect to K:**
$$
\frac{\partial^2 C}{\partial K^2} = e^{-rT} p(K, T)
$$

This is the famous **Breeden-Litzenberger formula**: the second derivative of call prices gives the risk-neutral density.

**Using Fokker-Planck on calls:**

After substitution and manipulation:
$$
\frac{\partial C}{\partial T} = -qC + (r-q)K\frac{\partial C}{\partial K} + \frac{1}{2}\sigma^2(K,T)K^2\frac{\partial^2 C}{\partial K^2}
$$

**Solving for local variance:**
$$
\sigma_{LV}^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}
$$

### 3.4 Formula in Terms of Implied Volatility

Let $w(y, T) = \sigma_{BS}^2(K, T) \cdot T$ be the **total variance** where $y = \ln(K/F)$ is log-moneyness.

$$
\sigma_{LV}^2 = \frac{\frac{\partial w}{\partial T}}{1 - \frac{y}{w}\frac{\partial w}{\partial y} + \frac{1}{4}\left(-\frac{1}{4} - \frac{1}{w} + \frac{y^2}{w^2}\right)\left(\frac{\partial w}{\partial y}\right)^2 + \frac{1}{2}\frac{\partial^2 w}{\partial y^2}}
$$

### 3.5 Numerical Implementation

**Finite Difference Approximations:**

Time derivative (forward difference):
$$
\frac{\partial C}{\partial T} \approx \frac{C(K, T + \Delta T) - C(K, T)}{\Delta T}
$$

Strike derivatives (central differences):
$$
\frac{\partial C}{\partial K} \approx \frac{C(K + \Delta K, T) - C(K - \Delta K, T)}{2\Delta K}
$$

$$
\frac{\partial^2 C}{\partial K^2} \approx \frac{C(K + \Delta K, T) - 2C(K, T) + C(K - \Delta K, T)}{(\Delta K)^2}
$$

**Common Issues:**
1. **Denominator → 0**: At extreme strikes, $\partial^2 C/\partial K^2 \to 0$
2. **Negative variance**: Arbitrage in input surface
3. **Short expiry**: Time derivatives are unstable

**Solutions:**
- Clamp local vol to reasonable bounds
- Use smooth interpolation of implied vols
- Validate input surface for arbitrage

---

## 4. Arbitrage Conditions

### 4.1 Calendar Arbitrage

**Condition:** Total variance must be non-decreasing in time for any fixed strike.

$$
\sigma^2(K, T_1) \cdot T_1 \leq \sigma^2(K, T_2) \cdot T_2 \quad \text{for } T_1 < T_2
$$

**Violation means:** You can sell a longer-dated option and buy a shorter-dated one for a riskless profit.

### 4.2 Butterfly Arbitrage

**Condition:** Call prices must be convex in strike.

$$
\frac{\partial^2 C}{\partial K^2} \geq 0
$$

**Violation means:** A butterfly spread (buy K₁, sell 2×K₂, buy K₃) has negative cost but non-negative payoff.

### 4.3 Implications for Dupire

If **either** arbitrage condition is violated:
- Dupire's formula gives **negative local variance**
- The local vol model cannot reproduce the input prices
- Must smooth/adjust the input surface

---

## 5. Numerical Methods

### 5.1 SABR Calibration

**Algorithm (L-BFGS-B):**

```python
def calibrate_sabr(market_vols, strikes, forward, expiry, beta=1.0):
    # Initial guess from ATM vol
    atm_vol = interpolate_atm(market_vols, strikes, forward)
    alpha0 = atm_vol * forward**(1 - beta)
    rho0, nu0 = 0.0, 0.5
    
    # Objective: sum of squared errors
    def objective(params):
        alpha, rho, nu = params
        model_vols = sabr_vol(strikes, forward, expiry, alpha, beta, rho, nu)
        return sum((model_vols - market_vols)**2)
    
    # Bounds: alpha > 0, |rho| < 1, nu >= 0
    bounds = [(1e-6, 10), (-0.999, 0.999), (0, 5)]
    
    result = minimize(objective, [alpha0, rho0, nu0], 
                     method='L-BFGS-B', bounds=bounds)
    return result.x
```

### 5.2 Dupire Implementation

**Key steps:**

1. **Build implied vol surface** from market quotes
2. **Convert to call prices** via Black-Scholes formula
3. **Compute derivatives** using finite differences
4. **Apply Dupire formula** at each grid point
5. **Clamp results** to avoid extreme values

### 5.3 Grid Construction

**Recommendations:**
- Dense grid near ATM (highest gamma)
- Logarithmic spacing in strike
- Finer time grid for short expiries

---

## 6. Interview Key Points

### SABR Questions

**Q: What are the SABR parameters and their effects?**

A: 
- **α (alpha)**: Controls ATM vol level
- **β (beta)**: CEV exponent, affects backbone shape
- **ρ (rho)**: Controls skew direction (neg = left skew)
- **ν (nu)**: Controls smile curvature/convexity

**Q: Why is β typically fixed?**

A: To avoid over-parameterization. Different (α, β) combinations can produce similar smiles. Market convention fixes β (1 for FX, lower for rates).

**Q: What's the limitation of Hagan's formula?**

A: It's an **asymptotic expansion** valid for:
- Small T
- K near F
- Moderate parameters

For extreme strikes or long expiries, use numerical methods.

### Dupire Questions

**Q: What's the key insight of Dupire's formula?**

A: Given vanilla option prices, there exists a **unique** local volatility function that reproduces them exactly. This is a model-free result.

**Q: What's the relationship between implied and local vol?**

A: Local vol is generally **higher than** implied vol at wings:
$$
\sigma_{LV}(K,T) \approx \sigma_{BS}(K,T) \cdot \sqrt{1 + \text{correction terms}}
$$

**Q: What happens if Dupire gives negative variance?**

A: The input surface contains **arbitrage**. Either:
1. Fix the input (smooth, interpolate properly)
2. Use a different calibration approach

### General Questions

**Q: When would you use SABR vs Dupire?**

| SABR | Dupire |
|------|--------|
| Smile interpolation | Full surface calibration |
| Fast recalibration | Exotic pricing (barriers, etc.) |
| Simple structure | Handles any smile shape |
| Parameterized | Non-parametric |

**Q: How do you check for arbitrage?**

1. **Calendar**: Total variance must increase with T
2. **Butterfly**: Call prices must be convex in K
3. **Call spread**: Call prices must decrease in K

---

## References

1. Hagan, P.S., et al. (2002). "Managing Smile Risk." Wilmott Magazine.
2. Dupire, B. (1994). "Pricing with a Smile." Risk Magazine.
3. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide."
4. Rebonato, R. (2004). "Volatility and Correlation."

---

*Document Version: 1.0 | Last Updated: January 2026*
