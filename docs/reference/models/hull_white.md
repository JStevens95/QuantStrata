# Hull-White One-Factor Short Rate Model

**The Industry Standard for Interest Rate Modeling**

This document covers the Hull-White model - the most widely used short rate model in practice, providing analytic tractability while fitting the initial term structure.

---

## Overview

The **Hull-White model** is a one-factor short rate model where the instantaneous interest rate follows a mean-reverting Gaussian (Ornstein-Uhlenbeck) process. It extends the Vasicek model by adding time-dependent drift to match the initial term structure exactly.

### Key Features

| Feature | Description |
|---------|-------------|
| **Mean Reversion** | Rates revert to long-term level θ at speed a |
| **Gaussian Distribution** | Rates can go negative (suitable for modern markets) |
| **Affine Structure** | Bond prices have exponential-affine form |
| **Analytic Tractability** | Closed-form solutions for bonds and European options |
| **Term Structure Fitting** | Matches initial yield curve exactly |

---

## Mathematical Framework

### Short Rate Dynamics

Under the risk-neutral measure Q, the short rate follows:

$$dr(t) = [\theta(t) - a \cdot r(t)] \, dt + \sigma \, dW(t)$$

Where:
- **r(t)**: Instantaneous short rate
- **θ(t)**: Time-dependent mean reversion level (fitted to initial curve)
- **a**: Mean reversion speed (a > 0)
- **σ**: Short rate volatility (σ > 0)
- **W(t)**: Standard Brownian motion under Q

For constant θ, this simplifies to:

$$dr(t) = a(\theta - r(t)) \, dt + \sigma \, dW(t)$$

### Distribution of r(t)

Given r(s) at time s < t, the short rate r(t) is normally distributed:

$$r(t) | r(s) \sim N(\mu(s,t), \, \nu^2(s,t))$$

Where:
- **Mean**: $\mu(s,t) = \theta + (r(s) - \theta) e^{-a(t-s)}$
- **Variance**: $\nu^2(s,t) = \frac{\sigma^2}{2a}(1 - e^{-2a(t-s)})$

### Long-Term Properties

| Property | Formula | Interpretation |
|----------|---------|----------------|
| Half-life | $\ln(2)/a$ | Time to revert halfway to mean |
| Long-term mean | $\theta$ | Rate converges to θ as t → ∞ |
| Long-term variance | $\sigma^2/(2a)$ | Asymptotic variance |
| Long-term vol | $\sigma/\sqrt{2a}$ | Asymptotic standard deviation |

---

## Zero-Coupon Bond Pricing

### Affine Bond Price

The Hull-White model produces affine bond prices:

$$P(t,T) = A(t,T) \cdot e^{-B(t,T) \cdot r(t)}$$

Where:
$$B(t,T) = \frac{1 - e^{-a(T-t)}}{a}$$

$$\ln A(t,T) = \ln\frac{P(0,T)}{P(0,t)} + B(t,T) f(0,t) - \frac{\sigma^2}{4a} B(t,T)^2 (1 - e^{-2at})$$

- **P(0,T)**: Initial zero-coupon bond price from market
- **f(0,t)**: Initial instantaneous forward rate

### B Factor Properties

The B(t,T) factor has intuitive interpretations:

| Limit | B(t,T) Value | Interpretation |
|-------|--------------|----------------|
| τ → 0 | B → τ | Short maturity: sensitivity ≈ duration |
| a → 0 | B → τ | No mean reversion: like Vasicek limit |
| a → ∞ | B → 0 | Fast mean reversion: low duration |

---

## European Option Pricing

### Bond Option Formula

For a European option on a zero-coupon bond:

**Call Option** (right to buy bond at K):
$$C = P(0,T_{bond}) N(h) - K \cdot P(0,T_{opt}) N(h - \sigma_p)$$

**Put Option** (right to sell bond at K):
$$P = K \cdot P(0,T_{opt}) N(-h + \sigma_p) - P(0,T_{bond}) N(-h)$$

Where:
$$\sigma_p = \sigma \sqrt{\frac{1 - e^{-2aT_{opt}}}{2a}} \cdot B(T_{opt}, T_{bond})$$

$$h = \frac{1}{\sigma_p} \ln\frac{P(0,T_{bond})}{K \cdot P(0,T_{opt})} + \frac{\sigma_p}{2}$$

### Caplet/Floorlet Pricing

Interest rate caps and floors are portfolios of caplets/floorlets. Under Hull-White:

**Caplet** (option to receive L - K if L > K):
$$\text{Caplet} = (1 + \tau K) \times \text{Put on ZC bond with strike } \frac{1}{1+\tau K}$$

**Floorlet** (option to receive K - L if K > L):
$$\text{Floorlet} = (1 + \tau K) \times \text{Call on ZC bond with strike } \frac{1}{1+\tau K}$$

Where τ is the accrual factor and L is the LIBOR rate.

### Swaption Pricing (Jamshidian Decomposition)

Swaptions can be priced using Jamshidian's trick, which decomposes a swaption into a portfolio of bond options:

1. Find r* such that the swap has zero value at option expiry
2. Compute bond option strikes K_i = P(S, T_i; r*)
3. Sum up bond options weighted by cash flows

---

## Calibration

### Parameters to Calibrate

| Parameter | Typical Range | Source |
|-----------|---------------|--------|
| **a** (mean reversion) | 0.01 - 0.5 | Cap/swaption volatilities |
| **σ** (volatility) | 0.005 - 0.03 | Cap/swaption ATM prices |
| **θ(t)** (drift) | From curve | Initial yield curve (exact fit) |

### Calibration Approach

1. **Fit θ(t)** to match initial term structure exactly
2. **Calibrate (a, σ)** to cap/swaption market prices:
   - Use ATM caps for overall vol level
   - Use swaption matrix for mean reversion

### Volatility Surface Fitting

Hull-White produces a specific vol smile/surface shape:
- **Short-term options**: Higher vol (less time for mean reversion)
- **Long-term options**: Lower vol (mean reversion dominates)

---

## QuantStrata Implementation

### Model Classes

```python
from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    HullWhiteDynamics,
    HullWhiteSimulation,
)

# Create parameters
params = HullWhiteParameters(
    a=0.1,       # Mean reversion speed
    sigma=0.01,  # Short rate volatility (1%)
    r0=0.03,     # Initial short rate (3%)
    theta=0.04,  # Long-term mean (4%)
)

# Simulate paths
dynamics = HullWhiteDynamics(params=params)
sim = dynamics.simulate(
    maturity=5.0,
    n_paths=100000,
    n_steps=500,
    scheme="exact",
    antithetic=True,
)
```

### Available Pricers

| Pricer | Method | Use Case |
|--------|--------|----------|
| `IrBondZeroCouponHWPricerSimple` | Analytic | ZC bond pricing |
| `IrBondEuropeanOptionHWPricerSimple` | Analytic | Bond options |
| `IrCapletEuropeanOptionHWPricerSimple` | Analytic | Caplets |
| `IrFloorletEuropeanOptionHWPricerSimple` | Analytic | Floorlets |
| `IrSwaptionEuropeanOptionHWPricerSimple` | Jamshidian | Swaptions |
| `IrBondZeroCouponMCPricerSimple` | Monte Carlo | Exotic payoffs |
| `IrBondEuropeanOptionMCPricerSimple` | Monte Carlo | Path-dependent |
| `IrBondZeroCouponFDPricerSimple` | Finite Diff | American options |
| `IrBondEuropeanOptionFDPricerSimple` | Finite Diff | Bermudan options |

### Example: Bond Option Pricing

```python
from src.models.short_rate.hull_white import HullWhiteParameters
from src.pricers.ir.european_hw import IrBondEuropeanOptionHWPricerSimple
from src.instruments.ir.options.bond import IrBondEuropeanOptionSimple
import math

# Setup
params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
pricer = IrBondEuropeanOptionHWPricerSimple(params=params)

# Create bond option
option = IrBondEuropeanOptionSimple(
    notional=1_000_000,
    strike=97.0,
    expiry=1.0,
    forward_bond_price=97.5,
    vol=params.sigma,
    discount_factor=math.exp(-0.03),
    option_type="call",
)

# Price and Greeks
price = pricer.price(option)
greeks = pricer.greeks(option)

print(f"Price: {price:,.2f}")
print(f"Delta: {greeks['delta']:,.2f}")
print(f"Vega: {greeks['vega']:,.2f}")
```

---

## Simulation Schemes

### Exact Scheme (Recommended)

Uses the exact transition distribution of the OU process:

$$r(t+\Delta t) = \theta + (r(t) - \theta) e^{-a\Delta t} + \sqrt{\frac{\sigma^2}{2a}(1-e^{-2a\Delta t})} \cdot Z$$

**Advantages:**
- No discretization error
- Preserves exact distribution
- Efficient for any time step

### Euler Scheme

Standard Euler-Maruyama discretization:

$$r(t+\Delta t) = r(t) + a(\theta - r(t))\Delta t + \sigma\sqrt{\Delta t} \cdot Z$$

**Use when:**
- Comparing to other implementations
- Educational purposes

---

## Interview Key Points

### Conceptual Questions

1. **Why use Hull-White over Vasicek?**
   - Hull-White fits the initial term structure exactly
   - Vasicek has constant parameters, creating arbitrage with market

2. **What does mean reversion represent?**
   - Central bank policy pulling rates back to equilibrium
   - Higher a = faster convergence = lower long-term rate volatility

3. **Why can rates go negative in Hull-White?**
   - Gaussian distribution has no lower bound
   - This is realistic post-2008 (negative rates in EUR, JPY, CHF)

4. **What is the Jamshidian decomposition?**
   - Technique to price swaptions as portfolios of bond options
   - Works because swap value is monotonic in short rate

### Quantitative Questions

1. **Derive the bond price formula**
   - Start from Feynman-Kac: $P(t,T) = E_t[\exp(-\int_t^T r(s)ds)]$
   - Use affine structure of Gaussian processes

2. **What is σ_p in the option formula?**
   - Volatility of the forward bond price P(S,T)/P(S,S)
   - Combines rate volatility with duration effect

3. **How does mean reversion affect option prices?**
   - Higher a → lower long-dated option prices
   - Reduces effective volatility over long horizons

---

## Comparison with Other Models

| Model | Distribution | Rates | Tractability | Term Structure |
|-------|--------------|-------|--------------|----------------|
| **Hull-White** | Gaussian | Can be negative | Very high | Exact fit |
| Vasicek | Gaussian | Can be negative | Very high | No fit |
| CIR | Non-central χ² | Always positive | High | Partial fit |
| Black-Karasinski | Log-normal | Always positive | Low (MC only) | Exact fit |
| LMM/BGM | Log-normal | Always positive | Low (MC only) | Exact fit |

---

## References

- Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities."
- Hull, J. & White, A. (1994). "Numerical Procedures for Implementing Term Structure Models."
- Brigo, D. & Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
- Jamshidian, F. (1989). "An Exact Bond Option Formula."
