# Curve Bootstrapping

**Mathematical Theory and Implementation**

This document provides comprehensive mathematical background for interest rate curve bootstrapping implemented in QuantStrata.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Discount Factors and Zero Rates](#2-discount-factors-and-zero-rates)
3. [Bootstrapping from Deposits](#3-bootstrapping-from-deposits)
4. [Bootstrapping from Swaps](#4-bootstrapping-from-swaps)
5. [Forward Rate Agreements](#5-forward-rate-agreements)
6. [Interpolation Methods](#6-interpolation-methods)
7. [Arbitrage Conditions](#7-arbitrage-conditions)
8. [Interview Key Points](#8-interview-key-points)

---

## 1. Introduction

### What is Curve Bootstrapping?

**Curve bootstrapping** is the process of constructing a discount curve from market-quoted instruments (deposits, FRAs, swaps). The goal is to find discount factors $DF(T)$ such that:

1. All input instruments are priced at par (zero NPV)
2. The curve is internally consistent
3. No arbitrage opportunities exist

### Why Curves Matter

Discount curves are fundamental to:
- **Pricing**: PV = Σ(cashflow × DF)
- **Risk**: Duration, DV01, curve risk
- **Hedging**: Determining hedge ratios
- **Valuation**: Mark-to-market of portfolios

---

## 2. Discount Factors and Zero Rates

### 2.1 Definitions

**Discount Factor** $DF(T)$: Present value of $1 received at time $T$
$$
DF(T) = e^{-r(T) \cdot T}
$$

**Continuous Zero Rate** $r(T)$: The constant rate that gives the discount factor
$$
r(T) = -\frac{\ln(DF(T))}{T}
$$

**Forward Rate** $f(T_1, T_2)$: The implied rate between two dates
$$
f(T_1, T_2) = -\frac{\ln(DF(T_2)/DF(T_1))}{T_2 - T_1}
$$

**Simple Forward Rate** (LIBOR-style):
$$
F(T_1, T_2) = \frac{1}{T_2 - T_1}\left(\frac{DF(T_1)}{DF(T_2)} - 1\right)
$$

### 2.2 Key Relationships

| Relationship | Formula |
|--------------|---------|
| DF from zero rate | $DF(T) = e^{-r(T) \cdot T}$ |
| Zero rate from DF | $r(T) = -\ln(DF(T))/T$ |
| Forward DF | $DF(T_1, T_2) = DF(T_2)/DF(T_1)$ |
| Continuously compounded forward | $f = -\ln(DF(T_1,T_2))/(T_2-T_1)$ |
| Simple forward | $F = (DF(T_1)/DF(T_2) - 1)/(T_2-T_1)$ |

---

## 3. Bootstrapping from Deposits

### 3.1 Deposit Rate Definition

A deposit pays simple interest. For rate $r_{dep}$ and maturity $T$:

**Simple Compounding:**
$$
DF(T) = \frac{1}{1 + r_{dep} \cdot T}
$$

**Continuous Compounding:**
$$
DF(T) = e^{-r_{dep} \cdot T}
$$

### 3.2 Example: 6-Month Deposit

Given: 6M deposit rate = 5% (simple, ACT/360)

$$
T = \frac{180}{360} = 0.5 \text{ years}
$$

$$
DF(0.5) = \frac{1}{1 + 0.05 \times 0.5} = \frac{1}{1.025} = 0.9756
$$

### 3.3 Day Count Conventions

| Convention | Description | Use |
|------------|-------------|-----|
| ACT/360 | Actual days / 360 | Money markets, LIBOR |
| ACT/365 | Actual days / 365 | UK markets |
| 30/360 | 30 days/month, 360/year | Bonds, swaps |
| ACT/ACT | Actual / actual | Government bonds |

---

## 4. Bootstrapping from Swaps

### 4.1 Par Swap Definition

A **par swap** has NPV = 0 at inception. The fixed rate $R$ is chosen such that:

$$
\text{PV(Fixed Leg)} = \text{PV(Float Leg)}
$$

### 4.2 Swap Valuation

**Fixed Leg PV:**
$$
PV_{fixed} = R \sum_{i=1}^{n} \alpha_i \cdot DF(T_i)
$$

**Float Leg PV** (at par):
$$
PV_{float} = DF(T_0) - DF(T_n) = 1 - DF(T_n)
$$

Where:
- $\alpha_i$ = accrual fraction for period $i$
- $T_i$ = payment date $i$
- $DF(T_i)$ = discount factor to $T_i$

### 4.3 Bootstrapping Algorithm

For a par swap with rate $R$ and maturity $T_n$:

$$
R \sum_{i=1}^{n} \alpha_i \cdot DF(T_i) = 1 - DF(T_n)
$$

**Solving for $DF(T_n)$:**

$$
DF(T_n) = \frac{1 - R \sum_{i=1}^{n-1} \alpha_i \cdot DF(T_i)}{1 + R \cdot \alpha_n}
$$

### 4.4 Iterative Bootstrapping

**Algorithm:**

```
Input: Sorted instruments (deposits, swaps) by maturity
Output: DF(T) for all maturities

1. For each deposit (short end):
   DF(T) = 1 / (1 + rate × T)

2. For each swap (in order of maturity):
   a. Build payment schedule
   b. Interpolate DF for intermediate dates (if needed)
   c. Solve for DF(maturity) using par swap equation
```

### 4.5 Example: 2-Year Swap

Given:
- 1Y swap rate: 3.0% (annual)
- 2Y swap rate: 3.5% (annual)
- We have: $DF(1) = 0.9709$ from 1Y deposit

**Step 1:** From 1Y swap (verify):
$$
DF(1) = \frac{1}{1 + 0.03 \times 1} = 0.9709 \quad \checkmark
$$

**Step 2:** From 2Y swap:
$$
0.035 \times (1 \times DF(1) + 1 \times DF(2)) = 1 - DF(2)
$$

$$
0.035 \times 0.9709 + 0.035 \times DF(2) = 1 - DF(2)
$$

$$
0.03398 + 1.035 \times DF(2) = 1
$$

$$
DF(2) = \frac{1 - 0.03398}{1.035} = 0.9332
$$

---

## 5. Forward Rate Agreements

### 5.1 FRA Definition

A **Forward Rate Agreement (FRA)** is a contract to exchange:
- Fixed rate payment
- Floating rate payment (determined at settlement)

FRA notation: "3x6" means starts in 3 months, ends in 6 months.

### 5.2 FRA Pricing

The FRA rate $F_{t_1,t_2}$ for period $(t_1, t_2)$ is:

$$
F_{t_1,t_2} = \frac{1}{t_2 - t_1}\left(\frac{DF(t_1)}{DF(t_2)} - 1\right)
$$

### 5.3 Bootstrapping with FRAs

Given a FRA rate $F$ for period $(t_1, t_2)$, we can find $DF(t_2)$:

$$
DF(t_2) = \frac{DF(t_1)}{1 + F \times (t_2 - t_1)}
$$

### 5.4 Example: 3x6 FRA

Given:
- 3M deposit: 2.5% → $DF(0.25) = 0.9938$
- 3x6 FRA rate: 2.8%

$$
DF(0.5) = \frac{0.9938}{1 + 0.028 \times 0.25} = \frac{0.9938}{1.007} = 0.9869
$$

---

## 6. Interpolation Methods

### 6.1 Linear in Discount Factors

$$
DF(t) = DF(T_1) + \frac{t - T_1}{T_2 - T_1}(DF(T_2) - DF(T_1))
$$

**Pros:** Simple
**Cons:** Can produce non-smooth forward rates

### 6.2 Log-Linear in Discount Factors (Industry Standard)

$$
\ln(DF(t)) = \ln(DF(T_1)) + \frac{t - T_1}{T_2 - T_1}(\ln(DF(T_2)) - \ln(DF(T_1)))
$$

Equivalently:
$$
DF(t) = DF(T_1) \times \left(\frac{DF(T_2)}{DF(T_1)}\right)^{\frac{t-T_1}{T_2-T_1}}
$$

**Key Property:** **Constant forward rates** between nodes

$$
f(T_1, t) = f(T_1, T_2) = -\frac{\ln(DF(T_2)/DF(T_1))}{T_2 - T_1}
$$

**Pros:** Industry standard, no discontinuous forwards
**Cons:** Forward rates jump at nodes

### 6.3 Linear in Zero Rates

$$
r(t) = r(T_1) + \frac{t - T_1}{T_2 - T_1}(r(T_2) - r(T_1))
$$

**Pros:** Intuitive
**Cons:** Can produce non-monotonic DFs in some cases

### 6.4 Cubic Spline in Zero Rates

Fit a cubic spline $r(t)$ through zero rate nodes with natural boundary conditions.

**Pros:** Very smooth curves
**Cons:** May introduce negative forward rates (arbitrage!)

### 6.5 Comparison

| Method | Forward Rates | Smoothness | Arbitrage Risk |
|--------|--------------|------------|----------------|
| Linear DF | Discontinuous | Low | Medium |
| Log-linear DF | Piecewise constant | Medium | Low |
| Linear zero | Continuous | Medium | Low |
| Cubic spline | Very smooth | High | High |

---

## 7. Arbitrage Conditions

### 7.1 Monotonicity

**Discount factors must decrease with time:**
$$
DF(T_2) < DF(T_1) \quad \text{for } T_2 > T_1
$$

**Violation implies:** Negative forward rates (free money)

### 7.2 Convexity

**Zero rates should not produce negative forwards:**
$$
f(T_1, T_2) = \frac{r(T_2) \cdot T_2 - r(T_1) \cdot T_1}{T_2 - T_1} \geq 0
$$

### 7.3 Validation Checks

```python
def validate_curve(tenors, dfs):
    # Check DF(0) = 1
    assert abs(dfs[0] - 1.0) < 1e-10 if tenors[0] == 0 else True
    
    # Check DFs are positive
    assert all(df > 0 for df in dfs)
    
    # Check DFs are decreasing
    assert all(dfs[i] > dfs[i+1] for i in range(len(dfs)-1))
    
    # Check no extreme values
    assert all(0.01 < df < 2.0 for df in dfs)
```

---

## 8. Interview Key Points

### Bootstrapping Questions

**Q: What is curve bootstrapping?**

A: The process of extracting discount factors from market instrument prices (deposits, swaps, FRAs) such that all instruments price at par.

**Q: Why use log-linear interpolation?**

A: It produces **constant forward rates** between nodes, avoiding discontinuities. This is the industry standard for rate curves.

**Q: What's the difference between zero rate and forward rate?**

A:
- **Zero rate** $r(T)$: Average rate from today to $T$
- **Forward rate** $f(T_1, T_2)$: Rate implied for future period $(T_1, T_2)$

**Q: How do you check for arbitrage in a curve?**

A:
1. DFs must be positive
2. DFs must be decreasing
3. Forward rates must be positive
4. No jumps that would allow arbitrage

### Swap Questions

**Q: Why does PV(float) = 1 - DF(Tn)?**

A: At inception, the floating leg is worth par because each future floating payment exactly equals the forward rate. The PV is:

$$
PV_{float} = \sum_{i=1}^{n} F_i \cdot \alpha_i \cdot DF(T_i)
$$

Using forward rate definition and telescoping:
$$
= DF(T_0) - DF(T_n) = 1 - DF(T_n)
$$

**Q: How do you bootstrap a swap with semi-annual payments?**

A: 
1. Build payment schedule (every 6 months)
2. For intermediate payments, interpolate DFs
3. Solve the par swap equation for the final DF

### Technical Questions

**Q: What day count convention is used for LIBOR?**

A: ACT/360 (actual days over 360)

**Q: What's the problem with cubic spline interpolation?**

A: It can introduce **negative forward rates** (arbitrage). The smoothness comes at the cost of potentially non-physical behavior.

---

## Worked Example: Full Bootstrap

**Given Market Data:**
| Instrument | Rate | Maturity |
|------------|------|----------|
| 3M Deposit | 2.0% | 0.25Y |
| 6M Deposit | 2.2% | 0.50Y |
| 1Y Swap | 2.5% | 1.00Y |
| 2Y Swap | 3.0% | 2.00Y |

**Step 1: 3M Deposit**
$$
DF(0.25) = \frac{1}{1 + 0.02 \times 0.25} = 0.9950
$$

**Step 2: 6M Deposit**
$$
DF(0.50) = \frac{1}{1 + 0.022 \times 0.50} = 0.9891
$$

**Step 3: 1Y Swap (annual)**
$$
0.025 \times DF(1) = 1 - DF(1)
$$
$$
DF(1) = \frac{1}{1.025} = 0.9756
$$

**Step 4: 2Y Swap (annual)**
$$
0.03 \times (DF(1) + DF(2)) = 1 - DF(2)
$$
$$
0.03 \times 0.9756 + 0.03 \times DF(2) = 1 - DF(2)
$$
$$
0.02927 + 1.03 \times DF(2) = 1
$$
$$
DF(2) = \frac{0.97073}{1.03} = 0.9424
$$

**Result:**

| Tenor | DF | Zero Rate |
|-------|-----|-----------|
| 0.25Y | 0.9950 | 2.01% |
| 0.50Y | 0.9891 | 2.20% |
| 1.00Y | 0.9756 | 2.47% |
| 2.00Y | 0.9424 | 2.97% |

---

## References

1. Hull, J.C. "Options, Futures, and Other Derivatives"
2. Brigo, D. & Mercurio, F. "Interest Rate Models - Theory and Practice"
3. Andersen, L. & Piterbarg, V. "Interest Rate Modeling"

---

*Document Version: 1.0 | Last Updated: January 2026*
