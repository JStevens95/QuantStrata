# Digital Options: Complete Technical Specification

**Document Type:** Technical Specification & Mathematical Derivation  
**Product Class:** European Digital (Binary) Option  
**Pricing Methods:** Black-Scholes Analytic, Monte Carlo, Finite Difference  
**Target Audience:** Quantitative Analysts, Financial Mathematics Graduates

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Formal Mathematical Framework](#2-formal-mathematical-framework)
3. [Product Specification](#3-product-specification)
4. [Pricing Theory](#4-pricing-theory)
5. [Greeks and Sensitivities](#5-greeks-and-sensitivities)
6. [Replication and Hedging](#6-replication-and-hedging)
7. [Numerical Methods](#7-numerical-methods)
8. [Risk Management](#8-risk-management)
9. [Key Interview Points](#9-key-interview-points)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Product Overview

A **digital option** (also called **binary option**) is an option that pays a fixed amount (cash or asset) if the underlying crosses a certain threshold, and nothing otherwise. The payoff is discontinuous ("all-or-nothing").

### 1.2 Key Characteristics

| Feature | Description |
|---------|-------------|
| **Payoff Type** | Discontinuous (step function) |
| **Settlement** | Cash-or-nothing, Asset-or-nothing |
| **Exercise Style** | European (at expiry only) |
| **Closed-Form** | Yes (using BSM N(d₂) terms) |
| **Primary Use Cases** | Structured products, betting on levels, hedging |

### 1.3 Digital Option Variants

| Variant | Call Payoff | Put Payoff |
|---------|-------------|------------|
| **Cash-or-Nothing** | Q if S_T > K, else 0 | Q if S_T < K, else 0 |
| **Asset-or-Nothing** | S_T if S_T > K, else 0 | S_T if S_T < K, else 0 |

Where Q is the fixed cash amount.

---

## 2. Formal Mathematical Framework

### 2.1 Underlying Dynamics

Under the risk-neutral measure \(\mathbb{Q}\):

\[
dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t^{\mathbb{Q}}
\]

Solution:
\[
S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma W_T^{\mathbb{Q}}\right]
\]

### 2.2 Risk-Neutral Probability

The key quantity is the **risk-neutral probability of finishing ITM**:

\[
\mathbb{P}^{\mathbb{Q}}(S_T > K) = N(d_2)
\]

where:
\[
d_2 = \frac{\log(S_0/K) + (r - q - \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}
\]

Similarly:
\[
\mathbb{P}^{\mathbb{Q}}(S_T < K) = N(-d_2) = 1 - N(d_2)
\]

---

## 3. Product Specification

### 3.1 Cash-or-Nothing Digital

#### Call (pays Q if S_T > K)

**Payoff:**
\[
\text{Payoff}_{\text{call}} = Q \cdot \mathbf{1}_{S_T > K}
\]

where \(\mathbf{1}_{S_T > K}\) is the indicator function.

#### Put (pays Q if S_T < K)

**Payoff:**
\[
\text{Payoff}_{\text{put}} = Q \cdot \mathbf{1}_{S_T < K}
\]

### 3.2 Asset-or-Nothing Digital

#### Call (pays S_T if S_T > K)

**Payoff:**
\[
\text{Payoff}_{\text{call}} = S_T \cdot \mathbf{1}_{S_T > K}
\]

#### Put (pays S_T if S_T < K)

**Payoff:**
\[
\text{Payoff}_{\text{put}} = S_T \cdot \mathbf{1}_{S_T < K}
\]

### 3.3 Contract Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Spot Price | \(S_0\) | Current price of underlying |
| Strike (Barrier) | \(K\) | Trigger level |
| Payout Amount | \(Q\) | Fixed cash payout |
| Time to Maturity | \(T\) | Years until expiration |
| Risk-Free Rate | \(r\) | Domestic interest rate |
| Dividend Yield | \(q\) | Continuous dividend rate |
| Volatility | \(\sigma\) | Annualized standard deviation |

---

## 4. Pricing Theory

### 4.1 Cash-or-Nothing Pricing

#### Call Price

\[
\boxed{V_{\text{cash-call}} = Q \cdot e^{-rT} \cdot N(d_2)}
\]

**Derivation:**
\[
V = e^{-rT} \mathbb{E}^{\mathbb{Q}}[Q \cdot \mathbf{1}_{S_T > K}] = Q \cdot e^{-rT} \cdot \mathbb{P}^{\mathbb{Q}}(S_T > K) = Q \cdot e^{-rT} \cdot N(d_2)
\]

#### Put Price

\[
\boxed{V_{\text{cash-put}} = Q \cdot e^{-rT} \cdot N(-d_2)}
\]

### 4.2 Asset-or-Nothing Pricing

#### Call Price

\[
\boxed{V_{\text{asset-call}} = S_0 \cdot e^{-qT} \cdot N(d_1)}
\]

**Derivation:**

Using the "share measure" (numeraire = stock):
\[
V = e^{-rT} \mathbb{E}^{\mathbb{Q}}[S_T \cdot \mathbf{1}_{S_T > K}] = S_0 e^{-qT} \cdot \mathbb{E}^{\mathbb{Q}^S}[\mathbf{1}_{S_T > K}] = S_0 e^{-qT} \cdot N(d_1)
\]

where \(\mathbb{Q}^S\) is the stock measure and \(d_1\) accounts for the change of measure.

#### Put Price

\[
\boxed{V_{\text{asset-put}} = S_0 \cdot e^{-qT} \cdot N(-d_1)}
\]

### 4.3 Relationship to Vanilla Options

The vanilla option can be decomposed:

**Vanilla Call = Asset-or-Nothing Call - K × Cash-or-Nothing Call**

\[
C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2) = V_{\text{asset-call}} - K \cdot V_{\text{cash-call}}/Q
\]

This shows that vanilla options are portfolios of digital options.

### 4.4 Digital Put-Call Parity

For cash-or-nothing digitals with payout Q:

\[
V_{\text{cash-call}} + V_{\text{cash-put}} = Q \cdot e^{-rT}
\]

**Interpretation:** Buying both a digital call and put with the same strike guarantees receiving Q at expiry.

For asset-or-nothing digitals:

\[
V_{\text{asset-call}} + V_{\text{asset-put}} = S_0 \cdot e^{-qT}
\]

---

## 5. Greeks and Sensitivities

### 5.1 Cash-or-Nothing Call Greeks

#### Delta

\[
\Delta = \frac{\partial V}{\partial S} = \frac{Q \cdot e^{-rT} \cdot \phi(d_2)}{S_0 \sigma \sqrt{T}}
\]

**Key Property:** Delta can be very large near ATM/expiry (discontinuous payoff).

#### Gamma

\[
\Gamma = \frac{\partial^2 V}{\partial S^2} = -\frac{Q \cdot e^{-rT} \cdot \phi(d_2) \cdot d_1}{S_0^2 \sigma^2 T}
\]

**Key Property:** Gamma can be extremely large and switches sign near the strike.

#### Vega

\[
\nu = \frac{\partial V}{\partial \sigma} = -Q \cdot e^{-rT} \cdot \phi(d_2) \cdot \sqrt{T} \cdot \frac{d_1}{\sigma}
\]

**Key Property:** Vega can be negative (unusual for options).

#### Theta

\[
\Theta = -rV + Q \cdot e^{-rT} \cdot \phi(d_2) \cdot \frac{d_1}{2T}
\]

### 5.2 Asset-or-Nothing Call Greeks

#### Delta

\[
\Delta = e^{-qT} N(d_1) + S_0 e^{-qT} \phi(d_1) \frac{1}{S_0 \sigma \sqrt{T}}
\]

\[
= e^{-qT} \left(N(d_1) + \frac{\phi(d_1)}{\sigma \sqrt{T}}\right)
\]

### 5.3 Greeks Summary

| Greek | Cash-or-Nothing | Characteristic |
|-------|-----------------|----------------|
| Delta | \(\propto \phi(d_2)/(S\sigma\sqrt{T})\) | Large near ATM/expiry |
| Gamma | Sign-switching | Extremely large |
| Vega | Can be negative | Counterintuitive |
| Theta | Complex | Time-dependent sign |

### 5.4 The "Greeks Explosion" Problem

Near expiry with S ≈ K:
- Delta → ∞ (impossible to hedge perfectly)
- Gamma → ±∞ (extreme convexity/concavity)

This is why digital options are challenging to hedge and often require spread approximations.

---

## 6. Replication and Hedging

### 6.1 Call Spread Approximation

A digital call can be approximated by a tight call spread:

\[
\text{Digital Call} \approx \frac{1}{\epsilon}\left[C(K - \epsilon/2) - C(K + \epsilon/2)\right]
\]

As \(\epsilon \to 0\), this converges to the digital.

**Practical Use:** Trade a call spread instead of the pure digital to:
1. Reduce Greeks explosion
2. Make hedging feasible
3. Account for bid-ask on the barrier

### 6.2 Hedging Strategy

**Delta Hedging Issues:**
- Delta becomes infinite at expiry if S = K
- Requires infinite rebalancing

**Practical Approaches:**
1. **Spread approximation:** Trade call spread instead
2. **Wide barrier:** Price with barrier shift accounting for bid-ask
3. **Static hedge:** Use portfolio of vanilla options
4. **Accept residual risk:** Cannot perfectly hedge

### 6.3 Overhedge/Underhedge

**Bid:** Assume spot finishes just below strike → price lower
**Offer:** Assume spot finishes just above strike → price higher

The bid-offer spread incorporates hedging uncertainty.

---

## 7. Numerical Methods

### 7.1 Analytic Pricing (BSM)

```python
def digital_cash_call(S, K, T, r, q, sigma, Q):
    d2 = (np.log(S/K) + (r - q - 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return Q * np.exp(-r*T) * norm.cdf(d2)

def digital_asset_call(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return S * np.exp(-q*T) * norm.cdf(d1)
```

### 7.2 Monte Carlo

```python
def digital_mc(S0, K, T, r, q, sigma, Q, n_paths, digital_type='cash', option_type='call'):
    Z = np.random.standard_normal(n_paths)
    S_T = S0 * np.exp((r - q - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    
    if option_type == 'call':
        indicator = (S_T > K).astype(float)
    else:
        indicator = (S_T < K).astype(float)
    
    if digital_type == 'cash':
        payoffs = Q * indicator
    else:  # asset
        payoffs = S_T * indicator
    
    return np.exp(-r*T) * np.mean(payoffs)
```

### 7.3 Finite Difference Challenges

- Discontinuous payoff causes oscillations
- Need smoothing or adaptive mesh near strike
- Use fine grid around K

---

## 8. Risk Management

### 8.1 Key Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Pin Risk** | S ≈ K at expiry | Spread approximation |
| **Model Risk** | BSM assumptions | Stochastic vol models |
| **Gap Risk** | Jumps over barrier | Jump-diffusion pricing |
| **Vol Risk** | Vega can be negative | Understand sign dynamics |

### 8.2 Structured Products Applications

Digitals are commonly embedded in:
- **Range Accruals:** Pay coupon if S stays in range
- **Autocallables:** Early redemption if S > barrier
- **Bonus Certificates:** Extra payout if condition met

### 8.3 Regulatory Considerations

- Digital options can resemble gambling products
- Subject to stricter regulation in retail markets
- Often restricted to professional investors

---

## 9. Key Interview Points

### 9.1 Must-Know Facts

1. **Digital pays all-or-nothing:** Discontinuous payoff
2. **Price = discounted probability:** \(V = Qe^{-rT}N(d_2)\) for cash call
3. **Vanilla = Asset digital - K × Cash digital**
4. **Hedging nightmare:** Delta → ∞ at ATM/expiry
5. **Call spread approximation:** Practical hedging approach
6. **Vega can be negative:** Counterintuitive behavior

### 9.2 Common Interview Questions

**Q: How do you hedge a digital option?**

A: Perfect hedging is impossible due to the discontinuous payoff. Practical approaches:
1. Trade a tight call spread approximation
2. Widen the barrier to account for uncertainty
3. Accept residual risk and charge for it

**Q: Why can digital vega be negative?**

A: For an ITM digital call (S > K), higher volatility increases the chance of finishing OTM. The option already benefits from the current situation, and more volatility introduces risk of crossing below K.

**Q: What is the delta of a digital option at expiry?**

A: Technically infinite if S = K (Dirac delta). In practice, this represents the hedging impossibility near the barrier at expiry.

**Q: How does a vanilla option relate to digital options?**

A: A vanilla call is long an asset-or-nothing call and short K units of cash-or-nothing calls:
\[
C = \text{Asset-Call}(K) - K \times \text{Cash-Call}(K)
\]

### 9.3 Quick Formulas

| Digital Type | Call Price | Put Price |
|--------------|------------|-----------|
| Cash-or-Nothing | \(Qe^{-rT}N(d_2)\) | \(Qe^{-rT}N(-d_2)\) |
| Asset-or-Nothing | \(Se^{-qT}N(d_1)\) | \(Se^{-qT}N(-d_1)\) |

**Digital Parity:**
\[
V_{\text{call}} + V_{\text{put}} = Qe^{-rT} \text{ (cash)}
\]
\[
V_{\text{call}} + V_{\text{put}} = Se^{-qT} \text{ (asset)}
\]

---

## 10. References

### Academic Papers

1. **Rubinstein, M. and Reiner, E.** (1991). "Unscrambling the Binary Code." *Risk*, 4(9), 75-83.

2. **Carr, P. and Chou, A.** (1997). "Breaking Barriers." *Risk*, 10(9), 139-145.

### Textbooks

3. **Hull, J.C.** (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson. Chapter 26.

4. **Wilmott, P.** (2006). *Paul Wilmott on Quantitative Finance*. Wiley. Volume 1, Chapter 10.

5. **Haug, E.G.** (2007). *The Complete Guide to Option Pricing Formulas* (2nd ed.). McGraw-Hill.

---

## Appendix A: Derivation of Cash-or-Nothing Price

### A.1 Setup

\[
V = e^{-rT} \mathbb{E}^{\mathbb{Q}}[Q \cdot \mathbf{1}_{S_T > K}]
\]

### A.2 Indicator Expectation

\[
\mathbb{E}^{\mathbb{Q}}[\mathbf{1}_{S_T > K}] = \mathbb{P}^{\mathbb{Q}}(S_T > K)
\]

### A.3 Probability Calculation

Since \(\log(S_T/S_0) \sim \mathcal{N}((r-q-\sigma^2/2)T, \sigma^2 T)\):

\[
\mathbb{P}^{\mathbb{Q}}(S_T > K) = \mathbb{P}\left(\frac{\log(S_T/S_0) - (r-q-\sigma^2/2)T}{\sigma\sqrt{T}} > \frac{\log(K/S_0) - (r-q-\sigma^2/2)T}{\sigma\sqrt{T}}\right)
\]

\[
= \mathbb{P}(Z > -d_2) = N(d_2)
\]

### A.4 Final Result

\[
\boxed{V_{\text{cash-call}} = Q \cdot e^{-rT} \cdot N(d_2)}
\]

---

## Appendix B: Derivation of Asset-or-Nothing Price

### B.1 Change of Numeraire

Under the stock measure \(\mathbb{Q}^S\) with \(S_t\) as numeraire:

\[
V = S_0 e^{-qT} \mathbb{E}^{\mathbb{Q}^S}[\mathbf{1}_{S_T > K}]
\]

### B.2 Radon-Nikodym Derivative

\[
\frac{d\mathbb{Q}^S}{d\mathbb{Q}} = \frac{S_T e^{-qT}}{S_0} \cdot e^{-rT + qT} = \frac{S_T}{S_0 e^{(r-q)T}}
\]

### B.3 Distribution Under Stock Measure

Under \(\mathbb{Q}^S\), the log-return has drift \(r - q + \sigma^2/2\) (extra \(\sigma^2\) term):

\[
\log(S_T/S_0) \sim \mathcal{N}\left((r-q+\sigma^2/2)T, \sigma^2 T\right) \text{ under } \mathbb{Q}^S
\]

### B.4 Final Result

\[
\mathbb{P}^{\mathbb{Q}^S}(S_T > K) = N(d_1)
\]

\[
\boxed{V_{\text{asset-call}} = S_0 \cdot e^{-qT} \cdot N(d_1)}
\]

---

*Document Version: 1.0*  
*Last Updated: January 27, 2026*  
*Author: QuantStrata Library*
