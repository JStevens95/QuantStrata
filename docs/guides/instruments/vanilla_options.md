# Vanilla Options: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Product Classes:** European Vanilla Options | American Vanilla Options  
**Pricing Methods:** Black-Scholes-Merton (European), Finite Difference/Binomial (American)  
**Target Audience:** Quantitative Analysts, Financial Mathematics Graduates

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Formal Mathematical Framework](#2-formal-mathematical-framework)
3. [Product Specification](#3-product-specification)
4. [European Options: Black-Scholes-Merton](#4-european-options-black-scholes-merton)
5. [American Options: Early Exercise](#5-american-options-early-exercise)
6. [Greeks and Sensitivities](#6-greeks-and-sensitivities)
7. [Put-Call Parity and Relationships](#7-put-call-parity-and-relationships)
8. [Numerical Methods](#8-numerical-methods)
9. [Risk Management](#9-risk-management)
10. [Asset Class Specifics: FX vs Equity](#10-asset-class-specifics-fx-vs-equity)
11. [Model Extensions](#11-model-extensions)
12. [Key Interview Points](#12-key-interview-points)
13. [References](#13-references)

---

## 1. Executive Summary

### 1.1 Product Overview

**Vanilla options** are the fundamental derivative contracts giving the holder the right (but not the obligation) to buy (call) or sell (put) an underlying asset at a predetermined strike price.

| Feature | European | American |
|---------|----------|----------|
| **Exercise** | Only at maturity T | Any time t ≤ T |
| **Closed-Form** | Yes (BSM) | No (except special cases) |
| **Early Exercise Premium** | N/A | Can be significant |
| **Pricing Methods** | Analytic, MC, FD | Binomial, FD (PSOR), LSM |

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Payoff Type** | Terminal for European; path-independent but exercise-dependent for American |
| **Underlying** | Equity, FX, commodity, index |
| **Historical Significance** | BSM (1973) - Nobel Prize in Economics (1997) |

### 1.3 When to Use Each Type

- **European**: Standard hedging, vol trading, simpler risk management
- **American**: Dividend capture, early profit taking, physical delivery needs

---

## 2. Formal Mathematical Framework

### 2.1 Probability Space and Filtration

Let \((\Omega, \mathcal{F}, \mathbb{P})\) be a probability space with filtration \(\{\mathcal{F}_t\}_{t \geq 0}\) satisfying the usual conditions.

**Brownian Motion** \(W = \{W_t\}_{t \geq 0}\) has properties:
1. \(W_0 = 0\) a.s.
2. Independent increments: \(W_t - W_s \perp \mathcal{F}_s\) for \(t > s\)
3. Stationary increments: \(W_t - W_s \sim \mathcal{N}(0, t-s)\)
4. Continuous paths a.s.

### 2.2 Asset Dynamics (Geometric Brownian Motion)

Under the physical measure \(\mathbb{P}\):

\[
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t^{\mathbb{P}}
\]

Under the risk-neutral measure \(\mathbb{Q}\):

\[
dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
\]

where:
- \(r\) = risk-free rate (domestic rate for FX)
- \(q\) = dividend yield (foreign rate for FX)
- \(\sigma\) = volatility

### 2.3 Itô's Lemma

For a function \(f(t, S_t)\):

\[
df = \left(\frac{\partial f}{\partial t} + \mu S \frac{\partial f}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 f}{\partial S^2}\right) dt + \sigma S \frac{\partial f}{\partial S} dW
\]

### 2.4 Risk-Neutral Measure (Girsanov's Theorem)

**Market Price of Risk:**
\[
\lambda = \frac{\mu - r}{\sigma}
\]

**Girsanov Transform:**
\[
W_t^{\mathbb{Q}} = W_t^{\mathbb{P}} + \lambda t
\]

### 2.5 GBM Solution

\[
\boxed{S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma W_T^{\mathbb{Q}}\right]}
\]

**Distribution:** \(\log S_T \sim \mathcal{N}\left(\log S_0 + (r - q - \frac{\sigma^2}{2})T, \sigma^2 T\right)\)

---

## 3. Product Specification

### 3.1 Payoff Structure

**Call Option:**
\[
\text{Payoff}_{\text{call}} = \max(S_\tau - K, 0) = (S_\tau - K)^+
\]

**Put Option:**
\[
\text{Payoff}_{\text{put}} = \max(K - S_\tau, 0) = (K - S_\tau)^+
\]

where \(\tau = T\) for European and \(\tau \in [0, T]\) (optimal stopping time) for American.

### 3.2 Contract Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Spot Price | \(S_0\) | Current price of underlying |
| Strike Price | \(K\) | Exercise price |
| Time to Maturity | \(T\) | Years until expiration |
| Risk-Free Rate | \(r\) | Domestic interest rate |
| Dividend/Foreign Rate | \(q\) | Continuous yield |
| Volatility | \(\sigma\) | Annualized standard deviation |
| Exercise Style | - | European or American |

### 3.3 Moneyness

| Term | Call | Put |
|------|------|-----|
| **ITM** | \(S > K\) | \(S < K\) |
| **ATM** | \(S \approx K\) | \(S \approx K\) |
| **OTM** | \(S < K\) | \(S > K\) |

**Forward Moneyness:** \(m = F/K = S_0 e^{(r-q)T}/K\)

---

## 4. European Options: Black-Scholes-Merton

### 4.1 Pricing Formula

**Theorem (Black-Scholes-Merton, 1973):**

**European Call:**
\[
\boxed{C_E = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)}
\]

**European Put:**
\[
\boxed{P_E = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)}
\]

where:
\[
d_1 = \frac{\log(S_0/K) + (r - q + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}
\]
\[
d_2 = d_1 - \sigma\sqrt{T} = \frac{\log(S_0/K) + (r - q - \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}
\]

and \(N(\cdot)\) is the standard normal CDF.

### 4.2 Derivation Outline

**Step 1:** Risk-neutral pricing formula
\[
C_E = e^{-rT} \mathbb{E}^{\mathbb{Q}}[(S_T - K)^+]
\]

**Step 2:** Substitute lognormal distribution of \(S_T\)
\[
C_E = e^{-rT} \int_K^{\infty} (s - K) f_{S_T}(s) \, ds
\]

**Step 3:** Change of variables and complete the square
\[
C_E = S_0 e^{-qT} \int_{-d_1}^{\infty} \phi(z) dz - K e^{-rT} \int_{-d_2}^{\infty} \phi(z) dz
\]

**Step 4:** Recognize standard normal integrals
\[
C_E = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
\]

### 4.3 Interpretation of Terms

| Term | Meaning |
|------|---------|
| \(N(d_2)\) | Risk-neutral probability option expires ITM |
| \(N(d_1)\) | Delta-adjusted probability (share measure) |
| \(S_0 e^{-qT} N(d_1)\) | PV of receiving stock if ITM |
| \(K e^{-rT} N(d_2)\) | PV of paying strike if ITM |

### 4.4 Black-Scholes PDE

\[
\boxed{\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV}
\]

**Boundary Conditions (Call):**
- Terminal: \(V(T, S) = (S - K)^+\)
- Lower: \(V(t, 0) = 0\)
- Upper: \(V(t, S) \to S e^{-q(T-t)}\) as \(S \to \infty\)

---

## 5. American Options: Early Exercise

### 5.1 Optimal Stopping Problem

An American option is an **optimal stopping problem**:

\[
\boxed{V_A(t, S) = \sup_{\tau \in [t, T]} \mathbb{E}^{\mathbb{Q}}\left[e^{-r(\tau - t)} \text{Payoff}(S_\tau) \mid S_t = S\right]}
\]

where \(\tau\) is a stopping time adapted to the filtration.

### 5.2 Free Boundary Problem

The American option satisfies the **free boundary problem**:

**In the Continuation Region** (\(V_A > \text{Payoff}\)):
\[
\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV
\]

**In the Exercise Region** (\(V_A = \text{Payoff}\)):
\[
V_A(t, S) = (S - K)^+ \quad \text{(call)} \quad \text{or} \quad (K - S)^+ \quad \text{(put)}
\]

**Free Boundary** \(S^*(t)\) separates the regions.

### 5.3 Early Exercise Premium

\[
\boxed{V_A = V_E + \text{Early Exercise Premium}}
\]

The early exercise premium is always non-negative: \(V_A \geq V_E\).

### 5.4 When is Early Exercise Optimal?

#### American Call on Non-Dividend Paying Stock

**Theorem:** It is **never** optimal to exercise an American call on a non-dividend paying stock early.

**Proof:**

For \(S > K\) (ITM):
\[
C_A(S) \geq S - Ke^{-r(T-t)} > S - K = \text{Intrinsic Value}
\]

The inequality is strict, so holding is always worth more than exercising.

**Intuition:** You keep the time value and delay paying the strike.

#### American Call with Dividends

Early exercise may be optimal **just before a dividend** if:
\[
\text{Dividend} > K(1 - e^{-r\Delta t}) \approx Kr\Delta t
\]

where \(\Delta t\) is time until next dividend.

#### American Put

Early exercise is often optimal for deep ITM puts because:
1. You receive cash K immediately (can invest at rate r)
2. Time value of receiving K now vs later
3. Limited upside from waiting (S can't go below 0)

**Critical Stock Price** \(S^*(t)\): Exercise immediately if \(S \leq S^*(t)\).

### 5.5 Bounds on American Options

**Lower Bound (European):**
\[
V_A \geq V_E
\]

**Upper Bound (Early Exercise Benefit):**
\[
C_A \leq S \quad \text{(call always worth less than stock)}
\]
\[
P_A \leq K \quad \text{(put always worth less than strike)}
\]

### 5.6 American vs European Premium

| Option | Early Exercise Premium |
|--------|----------------------|
| Call (no dividends) | 0 (never exercise early) |
| Call (with dividends) | Small (only before dividends) |
| Put | Significant (especially deep ITM) |

---

## 6. Greeks and Sensitivities

### 6.1 First-Order Greeks

#### Delta (Δ)

\[
\Delta = \frac{\partial V}{\partial S}
\]

**European Call:** \(\Delta_C = e^{-qT} N(d_1)\)

**European Put:** \(\Delta_P = -e^{-qT} N(-d_1)\)

**American:** Computed numerically; higher for ITM puts (due to early exercise)

| Property | Call | Put |
|----------|------|-----|
| Range | \([0, 1]\) | \([-1, 0]\) |
| ATM approx | ~0.5 | ~-0.5 |
| Deep ITM | →1 | →-1 |
| Deep OTM | →0 | →0 |

#### Gamma (Γ)

\[
\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\partial \Delta}{\partial S}
\]

**Formula:**
\[
\Gamma = \frac{e^{-qT} \phi(d_1)}{S \sigma \sqrt{T}}
\]

- Always positive (convexity)
- Maximum at ATM
- Increases as expiry approaches

#### Vega (ν)

\[
\nu = \frac{\partial V}{\partial \sigma}
\]

**Formula:**
\[
\nu = S e^{-qT} \sqrt{T} \phi(d_1)
\]

- Always positive for long options
- Maximum at ATM
- Increases with time to maturity

#### Theta (Θ)

\[
\Theta = \frac{\partial V}{\partial t}
\]

**European Call:**
\[
\Theta_C = -\frac{S e^{-qT} \phi(d_1) \sigma}{2\sqrt{T}} + qS e^{-qT} N(d_1) - rK e^{-rT} N(d_2)
\]

- Usually negative (time decay)
- Maximum magnitude at ATM

#### Rho (ρ)

\[
\rho = \frac{\partial V}{\partial r}
\]

**European Call:** \(\rho_C = KT e^{-rT} N(d_2)\)

**European Put:** \(\rho_P = -KT e^{-rT} N(-d_2)\)

### 6.2 Second-Order Greeks

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| **Vanna** | \(\frac{\partial^2 V}{\partial S \partial \sigma}\) | Delta sensitivity to vol |
| **Volga** | \(\frac{\partial^2 V}{\partial \sigma^2}\) | Vega convexity |
| **Charm** | \(\frac{\partial^2 V}{\partial S \partial t}\) | Delta decay |
| **Speed** | \(\frac{\partial^3 V}{\partial S^3}\) | Gamma sensitivity to spot |

### 6.3 Greeks Summary Table

| Greek | European Call | European Put | American |
|-------|---------------|--------------|----------|
| Δ | \(e^{-qT}N(d_1)\) | \(-e^{-qT}N(-d_1)\) | Numerical |
| Γ | \(\frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}\) | Same | Numerical |
| ν | \(Se^{-qT}\sqrt{T}\phi(d_1)\) | Same | Numerical |
| Θ | Complex | Complex | Numerical |
| ρ | \(KTe^{-rT}N(d_2)\) | \(-KTe^{-rT}N(-d_2)\) | Numerical |

---

## 7. Put-Call Parity and Relationships

### 7.1 European Put-Call Parity

\[
\boxed{C_E - P_E = S_0 e^{-qT} - K e^{-rT}}
\]

**Derivation:** Two portfolios with identical payoffs:
- Portfolio A: Long call + cash \(Ke^{-rT}\)
- Portfolio B: Long put + stock \(S_0 e^{-qT}\)

At maturity, both equal \(\max(S_T, K)\).

### 7.2 American Put-Call Inequality

For American options, we only have **inequalities**:

\[
\boxed{S_0 - K \leq C_A - P_A \leq S_0 e^{-qT} - K e^{-rT}}
\]

**Lower bound:** Exercise both immediately
**Upper bound:** Hold both to maturity

### 7.3 Early Exercise Premium Relation

\[
C_A - C_E = \text{Call EEP} \geq 0
\]
\[
P_A - P_E = \text{Put EEP} \geq 0
\]

---

## 8. Numerical Methods

### 8.1 European Options

#### Analytic (BSM)

- Exact closed-form solution
- Fastest computation
- Used for benchmarking

#### Monte Carlo

```python
def european_mc(S0, K, T, r, q, sigma, n_paths, option_type='call'):
    Z = np.random.standard_normal(n_paths)
    S_T = S0 * np.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    return np.exp(-r*T) * np.mean(payoffs)
```

### 8.2 American Options

#### Binomial Tree (CRR)

**Parameters:**
\[
u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}
\]

**Algorithm:**
1. Build forward tree: \(S_{i,j} = S_0 u^j d^{i-j}\)
2. Terminal payoffs: \(V_{N,j} = \text{Payoff}(S_{N,j})\)
3. Backward induction with early exercise check:
   \[
   V_{i,j} = \max\left(\text{Payoff}(S_{i,j}), e^{-r\Delta t}[p V_{i+1,j+1} + (1-p)V_{i+1,j}]\right)
   \]

#### Finite Difference (PSOR)

For American options, the PDE becomes a **linear complementarity problem**:

\[
\min\left(-\mathcal{L}V, V - \text{Payoff}\right) = 0
\]

where \(\mathcal{L}\) is the BS differential operator.

**PSOR (Projected Successive Over-Relaxation):**
1. Solve implicit FD system
2. Project to satisfy \(V \geq \text{Payoff}\)
3. Iterate until convergence

```python
# PSOR iteration
for iteration in range(max_iter):
    for j in range(1, N_S - 1):
        # Compute continuation value
        V_cont = A @ V_old
        # Apply early exercise constraint
        V_new[j] = max(payoff[j], V_cont[j])
```

#### Longstaff-Schwartz (LSM)

Monte Carlo for American options using regression:

1. Simulate paths forward
2. At each exercise date, regress continuation value on basis functions
3. Compare with immediate exercise value
4. Work backward to determine optimal exercise strategy

---

## 9. Risk Management

### 9.1 Delta Hedging

**Strategy:** Hold Δ shares against each option.

**Hedge Portfolio:**
\[
\Pi = V - \Delta S
\]

**P&L (first order):**
\[
d\Pi \approx \frac{1}{2}\Gamma(dS)^2 + \Theta \, dt + \nu \, d\sigma
\]

### 9.2 Gamma Hedging

**Problem:** Delta changes with S → frequent rebalancing.

**Solution:** Add another option to neutralize gamma.

**Gamma P&L:**
\[
\text{Realized Gamma P\&L} = \frac{1}{2}\Gamma \sum_i (\Delta S_i)^2
\]

### 9.3 American Option Hedging Considerations

- **Exercise boundary:** Greeks can jump at the free boundary
- **Early exercise risk:** May receive/deliver earlier than expected
- **Pin risk:** Near expiry, delta can oscillate rapidly

### 9.4 Greeks-Based P&L Attribution

\[
\Delta V \approx \Delta \cdot \Delta S + \frac{1}{2}\Gamma \cdot (\Delta S)^2 + \nu \cdot \Delta\sigma + \Theta \cdot \Delta t + \rho \cdot \Delta r
\]

---

## 10. Asset Class Specifics: FX vs Equity

The BSM framework applies to both FX and Equity options through the **cost-of-carry** parameter \(b\). This section details the key differences in interpretation, implementation, and risk management.

### 10.1 Parameter Mapping

| Parameter | FX Interpretation | Equity Interpretation |
|-----------|-------------------|----------------------|
| \(S\) | Spot exchange rate (DOM/FOR) | Stock price |
| \(r\) | Domestic risk-free rate \(r_d\) | Risk-free rate |
| \(q\) | Foreign risk-free rate \(r_f\) | Continuous dividend yield |
| \(b = r - q\) | Interest rate differential \(r_d - r_f\) | Cost-of-carry \(r - q\) |

### 10.2 FX Options: Two-Currency Framework

#### Market Convention

FX options are quoted in the **DOM/FOR** convention:
- **Domestic (DOM)**: Currency in which the option premium is paid
- **Foreign (FOR)**: Currency being bought/sold (the underlying)

**Example:** EURUSD = 1.0850 means 1 EUR = 1.0850 USD
- DOM = USD, FOR = EUR
- A call option gives the right to **buy EUR** (pay USD)

#### FX Forward Price

\[
F_{FX} = S_0 \cdot \exp\left[(r_d - r_f) \cdot T\right]
\]

#### FX Put-Call Parity

\[
\boxed{C - P = S_0 e^{-r_f T} - K e^{-r_d T}}
\]

#### FX Greeks: Dual Rho Decomposition

For FX options, interest rate sensitivity splits into two components:

**Rho Domestic** (sensitivity to \(r_d\)):
\[
\rho_d = \frac{\partial V}{\partial r_d} = KT e^{-r_d T} N(\pm d_2) \quad \text{(call: +, put: -)}
\]

**Rho Foreign** (sensitivity to \(r_f\)):
\[
\rho_f = \frac{\partial V}{\partial r_f} = -S_0 T e^{-r_f T} N(\pm d_1) \quad \text{(call: +, put: -)}
\]

**Key Insight:** \(\rho_f\) has the opposite sign to \(\rho_d\) because:
- Higher \(r_d\) → Lower PV of strike → Higher call value
- Higher \(r_f\) → Lower PV of spot (via forward) → Lower call value

### 10.3 Equity Options: Single-Curve with Dividends

#### Continuous Dividend Yield

For stocks paying continuous dividends at yield \(q\):

\[
F_{EQ} = S_0 \cdot \exp\left[(r - q) \cdot T\right]
\]

#### Equity Put-Call Parity

\[
\boxed{C - P = S_0 e^{-qT} - K e^{-rT}}
\]

#### Equity Greeks: Single Rho

For equity options with a single rate \(r\):

\[
\rho_{EQ} = \rho_{discount} + \rho_{carry}
\]

Where:
- \(\rho_{discount}\): Sensitivity to discounting of strike
- \(\rho_{carry}\): Sensitivity to drift (affects forward price)

**Combined effect:** Higher \(r\) generally increases call values (lower PV of strike, higher forward price).

### 10.4 Dividend Impact on Equity Options

#### Effect on Option Values

| Option | Dividend Effect | Reason |
|--------|-----------------|--------|
| Call | **Decreases** value | Stock drops on ex-dividend, forward price lower |
| Put | **Increases** value | Lower forward price benefits put holder |

#### Quantitative Impact

For a small change in dividend yield \(\Delta q\):

\[
\Delta C \approx -S_0 T e^{-qT} N(d_1) \cdot \Delta q
\]

\[
\Delta P \approx S_0 T e^{-qT} N(-d_1) \cdot \Delta q
\]

### 10.5 Early Exercise: FX vs Equity

#### FX American Options

| Scenario | Early Exercise? | Reason |
|----------|-----------------|--------|
| Call, \(r_d > r_f\) | Rarely | Time value from discounting dominates |
| Call, \(r_d < r_f\) | Possible | May capture interest rate advantage |
| Put | Possible | Receive domestic currency earlier |

**FX Intuition:** Early exercise trades optionality for immediate currency exchange and interest accrual in the stronger-rate currency.

#### Equity American Options

| Scenario | Early Exercise? | Reason |
|----------|-----------------|--------|
| Call, no dividends | **Never** | Time value always positive |
| Call, with dividends | **Just before ex-div** | Capture dividend by owning stock |
| Put | **Deep ITM** | Receive \(K\) now, invest at rate \(r\) |

**Equity Dividend Rule:** Exercise an American call just before an ex-dividend date if:

\[
\text{Dividend} > K \left(1 - e^{-r \cdot \Delta t}\right) \approx K \cdot r \cdot \Delta t
\]

where \(\Delta t\) is the time until next opportunity to exercise (or expiry).

### 10.6 Implementation in QuantStrata

#### FX Vanilla Pricing

```python
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption

# FX option uses TWO curves (domestic and foreign)
option = EuropeanFxVanillaOption(
    option_type="call",
    strike=1.10,
    expiry=1.0,
    notional=1_000_000,
    spot_id=EURUSD_SPOT,
    vol_id=EURUSD_VOL,
    domestic_curve_id=USD_CURVE,  # r_d
    foreign_curve_id=EUR_CURVE,   # r_f (acts like dividend yield)
)

pricer = FxEuropeanVanillaBsmPricer()
greeks = pricer.greeks(option, market)
# Returns: delta, gamma, vega, theta, rho_domestic, rho_foreign
```

#### Equity Vanilla Pricing

```python
from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer
from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption

# Equity option uses ONE curve plus dividend yield
option = EuropeanEquityVanillaOption(
    ticker="AAPL",
    option_type="call",
    strike=150.0,
    expiry=1.0,
    notional=100,
    dividend_yield=0.02,  # 2% continuous yield
    spot_id=AAPL_SPOT,
    vol_id=AAPL_VOL,
    curve_id=USD_CURVE,   # Single rate r
)

pricer = EquityEuropeanVanillaBsmPricer()
greeks = pricer.greeks(option, market)
# Returns: delta, gamma, vega, theta, rho (combined)
```

### 10.7 Summary: Key Differences

| Aspect | FX | Equity |
|--------|-----|--------|
| **Curves** | Two (\(r_d\), \(r_f\)) | One (\(r\)) + dividend \(q\) |
| **Rho** | Split: \(\rho_d\), \(\rho_f\) | Combined: single \(\rho\) |
| **Carry** | \(b = r_d - r_f\) | \(b = r - q\) |
| **Early exercise driver** | Interest rate differential | Dividends |
| **Put-call parity** | \(C - P = Se^{-r_f T} - Ke^{-r_d T}\) | \(C - P = Se^{-qT} - Ke^{-rT}\) |

---

## 11. Model Extensions

### 11.1 Beyond Black-Scholes

| Assumption | Reality | Extension |
|------------|---------|-----------|
| Constant vol | Smile/skew | Local vol, Stochastic vol |
| Continuous trading | Discrete | Transaction costs |
| No jumps | Crashes | Jump-diffusion |
| Lognormal | Fat tails | Levy processes |
| Constant rates | Term structure | Stochastic rates |

### 11.2 Local Volatility (Dupire)

\[
\sigma_{\text{loc}}(K, T) = \sqrt{\frac{\frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2 \frac{\partial^2 C}{\partial K^2}}}
\]

### 11.3 Stochastic Volatility (Heston)

\[
dS_t = (r-q)S_t dt + \sqrt{v_t} S_t dW_t^S
\]
\[
dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v
\]

### 11.4 Jump-Diffusion (Merton)

\[
dS_t = (r - q - \lambda \bar{k}) S_t dt + \sigma S_t dW_t + (J - 1) S_{t-} dN_t
\]

---

## 12. Key Interview Points

### 12.1 Must-Know Facts

1. **BSM assumptions:** GBM, constant vol, no arbitrage, continuous trading
2. **\(N(d_2)\):** Risk-neutral probability of expiring ITM
3. **\(N(d_1)\):** Delta of the option
4. **Put-Call Parity:** \(C - P = Se^{-qT} - Ke^{-rT}\) (European only)
5. **American call (no div):** Never exercise early
6. **American put:** May exercise early (deep ITM)
7. **Gamma risk:** Maximum at ATM, increases near expiry

### 12.2 Common Interview Questions

**Q: Why is an American call on a non-dividend stock never exercised early?**

A: The call is always worth more alive than dead because:
\[
C(S) \geq S - Ke^{-r(T-t)} > S - K
\]
You keep the time value and delay paying the strike.

**Q: When would you exercise an American put early?**

A: When the interest earned on receiving K now exceeds the time value lost. This happens when the put is deep ITM (S << K).

**Q: Derive the Black-Scholes PDE.**

A: 
1. Delta-hedge portfolio: \(\Pi = V - \frac{\partial V}{\partial S} S\)
2. Apply Itô's lemma to \(V\)
3. Stochastic terms cancel (delta hedge)
4. No-arbitrage: \(d\Pi = r\Pi \, dt\)
5. Rearrange to get PDE

**Q: What is the difference between \(d_1\) and \(d_2\)?**

A: 
- \(d_2\): Standardized log-moneyness under the forward measure
- \(d_1 = d_2 + \sigma\sqrt{T}\): Accounts for the drift adjustment when computing the expected stock price conditional on exercise

**Q: How does gamma behave as expiry approaches?**

A: 
- ATM: Gamma → ∞ (pin risk)
- ITM/OTM: Gamma → 0
- This creates "gamma scalping" opportunities

### 12.3 Quick Formulas

| Formula | Expression |
|---------|------------|
| BSM Call | \(Se^{-qT}N(d_1) - Ke^{-rT}N(d_2)\) |
| BSM Put | \(Ke^{-rT}N(-d_2) - Se^{-qT}N(-d_1)\) |
| \(d_1\) | \(\frac{\log(S/K) + (r-q+\sigma^2/2)T}{\sigma\sqrt{T}}\) |
| \(d_2\) | \(d_1 - \sigma\sqrt{T}\) |
| Put-Call Parity | \(C - P = Se^{-qT} - Ke^{-rT}\) |
| Delta (Call) | \(e^{-qT}N(d_1)\) |
| Gamma | \(\frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}\) |
| CRR: \(u\) | \(e^{\sigma\sqrt{\Delta t}}\) |
| CRR: \(p\) | \(\frac{e^{(r-q)\Delta t} - d}{u - d}\) |

---

## 13. References

### Original Papers

1. **Black, F. and Scholes, M.** (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.

2. **Merton, R.C.** (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics*, 4(1), 141-183.

3. **Cox, J.C., Ross, S.A., and Rubinstein, M.** (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics*, 7(3), 229-263.

4. **Brennan, M.J. and Schwartz, E.S.** (1977). "The Valuation of American Put Options." *Journal of Finance*, 32(2), 449-462.

5. **Longstaff, F.A. and Schwartz, E.S.** (2001). "Valuing American Options by Simulation." *Review of Financial Studies*, 14(1), 113-147.

### Textbooks

6. **Hull, J.C.** (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

7. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*. Wiley.

8. **Shreve, S.E.** (2004). *Stochastic Calculus for Finance II*. Springer.

---

## Appendix A: Full BSM Derivation

### A.1 Setup

Under \(\mathbb{Q}\):
\[
S_T = S_0 e^{(r-q-\sigma^2/2)T + \sigma W_T}
\]

where \(W_T \sim \mathcal{N}(0, T)\).

### A.2 Call Price Integral

\[
C = e^{-rT} \mathbb{E}[(S_T - K)^+] = e^{-rT} \int_K^\infty (s - K) f_{S_T}(s) ds
\]

### A.3 Change of Variables

Let \(z = \frac{\log(s/S_0) - (r-q-\sigma^2/2)T}{\sigma\sqrt{T}}\), so \(s = S_0 e^{(r-q-\sigma^2/2)T + \sigma\sqrt{T}z}\).

The condition \(s > K\) becomes:
\[
z > \frac{\log(K/S_0) - (r-q-\sigma^2/2)T}{\sigma\sqrt{T}} = -d_2
\]

### A.4 Split the Integral

\[
C = e^{-rT} \int_{-d_2}^\infty S_0 e^{(r-q-\sigma^2/2)T + \sigma\sqrt{T}z} \phi(z) dz - e^{-rT} K \int_{-d_2}^\infty \phi(z) dz
\]

The second integral is \(N(d_2)\).

For the first integral, complete the square:
\[
(r-q-\sigma^2/2)T + \sigma\sqrt{T}z - z^2/2 = (r-q)T - (z - \sigma\sqrt{T})^2/2
\]

Substituting \(u = z - \sigma\sqrt{T}\):
\[
\int_{-d_2}^\infty e^{(r-q)T} \phi(u + \sigma\sqrt{T}) du = e^{(r-q)T} \int_{-d_1}^\infty \phi(u) du = e^{(r-q)T} N(d_1)
\]

### A.5 Final Result

\[
\boxed{C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)}
\]

---

## Appendix B: American Option Free Boundary

### B.1 Optimal Stopping Formulation

The American put value satisfies:
\[
P_A(t, S) = \sup_{\tau \geq t} \mathbb{E}^{\mathbb{Q}}[e^{-r(\tau - t)}(K - S_\tau)^+ | S_t = S]
\]

### B.2 Free Boundary Conditions

**Continuation region** \(\mathcal{C} = \{(t, S) : P_A(t, S) > (K - S)^+\}\):
\[
\frac{\partial P}{\partial t} + (r-q)S\frac{\partial P}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 P}{\partial S^2} = rP
\]

**Exercise region** \(\mathcal{E} = \{(t, S) : P_A(t, S) = (K - S)^+\}\):
\[
P_A(t, S) = K - S
\]

### B.3 Smooth Pasting Conditions

At the free boundary \(S = S^*(t)\):
\[
P_A(t, S^*(t)) = K - S^*(t) \quad \text{(value matching)}
\]
\[
\frac{\partial P_A}{\partial S}(t, S^*(t)) = -1 \quad \text{(smooth pasting)}
\]

---

*Document Version: 1.0*  
*Last Updated: January 27, 2026*  
*Author: QuantStrata Library*
