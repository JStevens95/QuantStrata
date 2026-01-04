# Black–Scholes–Merton Analytics (Generic Carry Form)
Folder: `src/models/analytic/black_scholes_merton/`

This package implements **closed-form Black–Scholes–Merton (BSM)** pricing and greeks under a **generic cost-of-carry** parameterisation. It is a **pure mathematics layer**:
- no `Market` objects
- no curves/vol surfaces
- no calendars/instruments
- inputs are scalars (S, K, T, r, b, sigma, payout units)
- outputs are floats / greek dictionaries

The job of pricer adaptors elsewhere in the repo is to map real market objects (curves, vols) into `(discount_rate r, carry b, sigma)` and apply notionals/units.

---

## 1) Module map (manual)

### 1.1 `base.py`
Shared helpers:
- `validate_bsm_inputs(spot, strike, expiry, vol)`
- `d1_d2(spot, strike, expiry, carry, vol)`
- `CarryDiscountTerms.from_rates(time_to_expiry, discount_rate, carry)`
  - `df         = exp(-rT)`
  - `fwd_factor = exp((b - r)T)`
- `intrinsic_vanilla(option_type, spot, strike)`

### 1.2 `vanilla.py`
`BlackScholesMertonVanilla`:
- `price(option_type, spot, strike, time_to_expiry, discount_rate, carry, sigma)`
- `greeks(...) -> {delta, gamma, vega, rho_discount, rho_carry}`

### 1.3 `digital.py`
- `BlackScholesMertonDigitalCash` (cash-or-nothing)
- `BlackScholesMertonDigitalAsset` (asset-or-nothing)

Digital cash supports optional payoff compatibility:
`DigitalCashPayoffLike = { option_type, strike, cash }`

---

## 2) Quickstart (manual)

### 2.1 Vanilla (example in FX mapping)
For FX with domestic rate r_d and foreign rate r_f:
- discount_rate r = r_d
- carry b = r_d - r_f
- then `fwd_factor = exp((b-r)T) = exp(-r_f T)`

Example:

from src.models.analytic.black_scholes_merton.vanilla import BlackScholesMertonVanilla

engine = BlackScholesMertonVanilla()

S = 1.25
K = 1.25
T = 1.0
r_d = 0.03
r_f = 0.01
r = r_d
b = r_d - r_f
sigma = 0.20

pv = engine.price(option_type="call", spot=S, strike=K, time_to_expiry=T, discount_rate=r, carry=b, sigma=sigma)
g  = engine.greeks(option_type="call", spot=S, strike=K, time_to_expiry=T, discount_rate=r, carry=b, sigma=sigma)

print(pv, g)

---

# 3) Technical Note (derivations, PDE, closed-form, greeks)
This section is written as a technical document. It derives:
- the BSM **pricing PDE** with a generic carry drift
- the **closed-form vanilla call/put**
- **cash** and **asset** digitals
- key greeks and the interpretation of `(r, b)`

We keep the parameterisation used in this folder:

- `r` = discount_rate (continuous compounding)
- `b` = carry (drift parameter in the SDE, continuous compounding)
- `sigma` = constant volatility
- `S` = spot at valuation time t
- `K` = strike
- `T` = time to expiry (years)

We price a derivative `V(t,S)` with terminal payoff `V(T,S) = payoff(S)`.

---

## 3.1 Model assumptions and dynamics
Assume:
1) frictionless market, continuous trading
2) one risky underlying `S_t`
3) constant coefficients `r, b, sigma`
4) no arbitrage

Generic carry-form diffusion under a pricing measure (equivalently: under the “model-implied” measure consistent with discounting at r and drift b):

dS_t = b * S_t * dt + sigma * S_t * dW_t

This is *deliberately generic*:
- equity with continuous dividend yield q: typically b = r - q
- FX (Garman–Kohlhagen): b = r_d - r_f with discounting at r = r_d

The *discounting numeraire* is exp(r t); discount factor is exp(-r (T-t)).

---

## 3.2 Deriving the Black–Scholes PDE (replication / Itô)
Let `V(t,S)` be twice differentiable in S and once in t.

Apply Itô to V(t,S_t):

dV = V_t dt + V_S dS + 0.5 V_SS (dS)^2

Since dS = b S dt + sigma S dW and (dS)^2 = sigma^2 S^2 dt:

dV = [ V_t + b S V_S + 0.5 sigma^2 S^2 V_SS ] dt  +  [ sigma S V_S ] dW

Construct a self-financing portfolio that eliminates the dW term:
Hold `Delta = V_S` units of the underlying and finance the remainder in cash.

Portfolio:
Pi = V - Delta * S

Its differential:
dPi = dV - Delta dS
    = [ V_t + b S V_S + 0.5 sigma^2 S^2 V_SS - Delta * (b S) ] dt
      + [ sigma S V_S - Delta * sigma S ] dW

Choose Delta = V_S so the stochastic term is zero:

dPi = [ V_t + 0.5 sigma^2 S^2 V_SS ] dt

Now the key economic step:
Pi is locally riskless (no dW), so in an arbitrage-free market it must earn the risk-free rate r:

dPi = r * Pi * dt = r * (V - S V_S) dt

Equate deterministic dt terms:

V_t + 0.5 sigma^2 S^2 V_SS = r (V - S V_S)

Rearrange into the standard PDE:

V_t + 0.5 sigma^2 S^2 V_SS + r S V_S - r V = 0

This is the classic Black–Scholes PDE in the (r, q) parameterisation.

But our model drift was b, not r. Where does b enter?
It enters through the *carry interpretation* of the underlying’s forward price. In the PDE, the term multiplying V_S is the risk-neutral drift of S under the cash numeraire; in the generic carry form, that drift is b:

Therefore the PDE used in this package is:

V_t + 0.5 sigma^2 S^2 V_SS + b S V_S - r V = 0

Terminal condition:
V(T,S) = payoff(S)

This is the PDE solved analytically in this folder, and numerically by your FD engines elsewhere.

---

## 3.3 Log transform and reduction to the heat equation (solution route)
Define time-to-expiry tau = T - t (so tau runs forward from 0 at expiry).

Define log-spot:
x = ln(S)

We seek a transformation that converts the PDE to a constant-coefficient diffusion in x.

Use standard derivative identities:
Let U(t,x) = V(t, S) with S = exp(x).

Then:
V_S = (1/S) * U_x
V_SS = (1/S^2) * (U_xx - U_x)

Substitute into the PDE:

V_t + 0.5 sigma^2 S^2 V_SS + b S V_S - r V = 0
=> U_t + 0.5 sigma^2 (U_xx - U_x) + b (U_x) - r U = 0
=> U_t + 0.5 sigma^2 U_xx + (b - 0.5 sigma^2) U_x - r U = 0

Now switch from t to tau = T - t:
U_t = -U_tau

So:
-U_tau + 0.5 sigma^2 U_xx + (b - 0.5 sigma^2) U_x - r U = 0
=> U_tau = 0.5 sigma^2 U_xx + (b - 0.5 sigma^2) U_x - r U

Remove the first-derivative and killing term with an exponential change of variables:
Let:
U(tau, x) = exp(A x + B tau) * W(tau, x)

Choose A, B to eliminate the U_x term and the -rU term after substitution.
This is standard; the result is that W satisfies a pure diffusion (heat) equation:

W_tau = 0.5 sigma^2 W_xx

with a transformed terminal condition at tau=0 derived from payoff.

This route yields the closed form via convolution with the normal density.
Practically, for vanilla and digitals, it is cleaner to derive directly from the lognormal distribution; both are equivalent.

---

## 3.4 Distribution of S_T and definitions of d1, d2
Solve the SDE explicitly:

dS/S = b dt + sigma dW
=> ln(S_T) = ln(S) + (b - 0.5 sigma^2) * (T-t) + sigma * sqrt(T-t) * Z
where Z ~ N(0,1)

Let tau = T - t.

Then:
ln(S_T / K) = ln(S/K) + (b - 0.5 sigma^2) tau + sigma sqrt(tau) Z

Define:

d2 = [ ln(S/K) + (b - 0.5 sigma^2) tau ] / (sigma sqrt(tau))
d1 = d2 + sigma sqrt(tau)
   = [ ln(S/K) + (b + 0.5 sigma^2) tau ] / (sigma sqrt(tau))

These match `base.py:d1_d2`.

Also define:
df         = exp(-r tau)
fwd_factor = exp((b - r) tau)

Note: `S * fwd_factor = S * exp((b-r)tau)` is the “PV forward” term in this parameterisation.

---

## 3.5 Closed-form European vanilla price (full derivation)
For a call payoff (S_T - K)^+, price is:

V_call(t,S) = exp(-r tau) * E[ (S_T - K)^+ ]

Split expectation:
E[(S_T - K)^+] = E[S_T * 1_{S_T > K}] - K * P(S_T > K)

So:

V_call = exp(-r tau) * ( E[S_T 1_{S_T>K}] - K * P(S_T > K) )

We compute the two terms.

### 3.5.1 Probability term
From the lognormal representation:

S_T > K
<=> ln(S_T/K) > 0
<=> Z > -d2

Therefore:
P(S_T > K) = N(d2)

### 3.5.2 Truncated first moment term E[S_T 1_{S_T>K}]
Write:

S_T = S * exp( (b - 0.5 sigma^2) tau + sigma sqrt(tau) Z )

Then:

E[S_T 1_{S_T>K}]
= S * exp((b - 0.5 sigma^2) tau) * E[ exp(sigma sqrt(tau) Z) 1_{Z > -d2} ]

Now use the standard normal “exponential tilting” identity:
For Z ~ N(0,1), for any a:
E[ exp(a Z) 1_{Z > c} ] = exp(0.5 a^2) * N( -(c - a) )

Here a = sigma sqrt(tau) and c = -d2:

E[ exp(a Z) 1_{Z > -d2} ]
= exp(0.5 a^2) * N( d2 + a )
= exp(0.5 sigma^2 tau) * N(d1)

Therefore:

E[S_T 1_{S_T>K}]
= S * exp((b - 0.5 sigma^2) tau) * exp(0.5 sigma^2 tau) * N(d1)
= S * exp(b tau) * N(d1)

Plug back into the call price:

V_call
= exp(-r tau) * ( S exp(b tau) N(d1) - K N(d2) )
= S exp((b-r) tau) N(d1) - K exp(-r tau) N(d2)

So:

Call PV:
PV_call = S * fwd_factor * N(d1) - K * df * N(d2)

Put PV follows either by repeating the derivation for (K - S_T)^+ or by put-call parity.

Put PV:
PV_put = K * df * N(-d2) - S * fwd_factor * N(-d1)

This is exactly what `vanilla.py:price()` implements.

---

## 3.6 Cash-or-nothing digital price (full derivation)
Cash digital payoff at expiry:
Call: C * 1_{S_T > K}
Put : C * 1_{S_T < K}

Price:
PV = exp(-r tau) * C * P(ITM)

We already have P(S_T > K) = N(d2), and P(S_T < K) = N(-d2).

So:

PV_call = C * df * N(d2)
PV_put  = C * df * N(-d2)

This matches `BlackScholesMertonDigitalCash.price()`.

---

## 3.7 Asset-or-nothing digital price (full derivation)
Asset digital payoff:
Call: q * S_T * 1_{S_T > K}
Put : q * S_T * 1_{S_T < K}

Price:
PV = exp(-r tau) * q * E[ S_T 1_{ITM} ]

We computed:
E[S_T 1_{S_T>K}] = S exp(b tau) N(d1)

Similarly:
E[S_T 1_{S_T<K}] = S exp(b tau) N(-d1)

Therefore:

PV_call = exp(-r tau) * q * S exp(b tau) N(d1)
        = q * S * fwd_factor * N(d1)

PV_put  = q * S * fwd_factor * N(-d1)

This matches `BlackScholesMertonDigitalAsset.price()`.

---

## 3.8 Greeks: derivations aligned to the implemented formulas

### 3.8.1 Useful derivatives of d1, d2
Let tau = T-t, sqrtTau = sqrt(tau).

d2 = [ ln(S/K) + (b - 0.5 sigma^2) tau ] / (sigma sqrtTau)

Then:

(1) dd2/dS = 1 / (S sigma sqrtTau)

(2) dd2/db = sqrtTau / sigma

(3) dd2/dsigma:
Write d2 = A/(sigma sqrtTau) - 0.5 sigma sqrtTau,
where A = ln(S/K) + b tau.
Differentiate:
dd2/dsigma = -A/(sigma^2 sqrtTau) - 0.5 sqrtTau
But A/(sigma sqrtTau) = d2 + 0.5 sigma sqrtTau,
so:
dd2/dsigma = -(d2/sigma + sqrtTau)

This is the numerically stable form used in `digital.py`.

And d1 = d2 + sigma sqrtTau implies:
dd1/dS     = dd2/dS
dd1/db     = dd2/db
dd1/dsigma = sqrtTau - d1/sigma

Normal identities:
d/dx N(x) = n(x)
d/dx N(-x) = -n(x)

---

## 3.9 Vanilla greeks (with generic carry)
Vanilla call PV:
PV_call = S fwd_factor N(d1) - K df N(d2)

Delta:
Differentiate with respect to S.
The standard result (still true in carry form) is:

Delta_call = fwd_factor * N(d1)
Delta_put  = fwd_factor * (N(d1) - 1)

Gamma:
Gamma is identical for call/put:

Gamma = fwd_factor * n(d1) / (S sigma sqrtTau)

Vega:
Vega (per +1.00 absolute sigma):

Vega = S fwd_factor n(d1) sqrtTau

These match `vanilla.py:greeks()`.

### 3.9.1 About rho_discount and rho_carry conventions
This library defines two “rate-like” greeks:

- rho_discount: derivative w.r.t. discount_rate r holding carry b fixed
- rho_carry: “carry sensitivity” convention used in your code (documented below)

Because the model uses two rates (r and b), you must be explicit: changing r while keeping b fixed is not the same as changing r while keeping (r-q) fixed, etc.

#### rho_discount (holding b fixed)
We have:
df = exp(-r tau)
fwd_factor = exp((b - r) tau)

Holding b fixed:
d(df)/dr = -tau * df
d(fwd_factor)/dr = -tau * fwd_factor
and d1,d2 do not depend on r (since they depend on b, not r).

So the derivative is obtained by differentiating only the df and fwd_factor channels.
That is what your `vanilla.py` implements.

#### rho_carry (as implemented)
Your vanilla code implements a clean, desk-friendly convention:

rho_carry_call = +tau * (S fwd_factor N(d1))
rho_carry_put  = -tau * (S fwd_factor N(-d1))

This corresponds to the explicit dependence of PV on b through the forward factor channel.
It is stable and interpretable, but it is not the fully expanded derivative of PV with respect to b including the d1/d2 channel.
If you ever want the “full” ∂PV/∂b, see Appendix E.

This README documents the implemented choice explicitly.

---

## 3.10 Digital cash greeks (derivations matching code)
Cash digital call PV:
PV = C * df * N(d2)
Put PV:
PV = C * df * N(-d2)

Let sign = +1 for call, -1 for put so PV = C df N(sign*d2).

Delta:
dPV/dS = C df * n(d2) * sign * (dd2/dS)
       = C df n(d2) sign * (1/(S sigma sqrtTau))

Gamma:
Differentiate delta again.
Let g = dd2/dS = 1/(S sigma sqrtTau).
Then dg/dS = -g/S.

Delta = C df n(d2) sign * g

d/dS [n(d2)] = n(d2) * (-d2) * (dd2/dS) = n(d2) * (-d2) * g

So:
Gamma = C df sign * [ (d/dS n(d2)) * g + n(d2) * dg/dS ]
      = C df sign * [ n(d2) (-d2) g^2 + n(d2) (-g/S) ]
      = - C df n(d2) sign * ( d2 g^2 + g/S )

Your code forms this as:
gamma_call = -C df n(d2) * ( d2 g^2 + g/S )
gamma_put  = -gamma_call
which is algebraically equivalent to the sign form above.

Vega:
dPV/dsigma = C df * n(d2) * sign * (dd2/dsigma)
           = C df n(d2) sign * (-(d2/sigma + sqrtTau))

rho_discount:
Holding b fixed, d2 does not depend on r. Only df depends on r:
PV = C exp(-r tau) * (...)
rho_discount = dPV/dr = -tau * PV

rho_carry:
PV depends on b only through d2 (in this digital cash case).
So:
rho_carry = C df n(d2) sign * (dd2/db)
          = C df n(d2) sign * (sqrtTau/sigma)

These match `BlackScholesMertonDigitalCash.greeks()` exactly.

---

## 3.11 Digital asset greeks (derivations matching code)
Asset digital PV:
PV = q * S * fwd_factor * N(sign*d1)

Let A = q * fwd_factor (constant in S), and F = N(sign*d1).

PV = A * S * F

Delta:
dPV/dS = A * [ F + S * dF/dS ]
dF/dS = n(d1) * sign * (dd1/dS)
dd1/dS = 1/(S sigma sqrtTau)

So:
Delta = q fwd_factor [ N(sign*d1) + n(d1) sign / (sigma sqrtTau) ]

Your code writes it as:
Delta = q*fwd_factor*( N(sign*d1) + S*n(d1)*sign*dd1_dS )
which is the same since S*dd1_dS = 1/(sigma sqrtTau).

Gamma:
Differentiate delta; the compact closed form your code uses is valid:
Gamma = q * fwd_factor * (sign*n(d1)/(S sigma sqrtTau)) * (1 - d1/(sigma sqrtTau))

Vega:
PV = q S fwd_factor N(sign*d1)
dPV/dsigma = q S fwd_factor n(d1) sign * dd1/dsigma
dd1/dsigma = sqrtTau - d1/sigma
So:
Vega = q S fwd_factor n(d1) sign (sqrtTau - d1/sigma)

rho_discount (holding b fixed):
Only fwd_factor depends on r:
fwd_factor = exp((b-r)tau) => derivative is -tau * fwd_factor
So rho_discount = -tau * PV

rho_carry:
PV depends on b through:
- fwd_factor (explicit) and
- d1 (implicit)
The code implements:
rho_carry = tau*PV + q*S*fwd_factor*n(d1) sign * (dd1/db)
dd1/db = sqrtTau/sigma

This matches the engine.

---

# 4) Specialisation: FX (Garman–Kohlhagen)
For FX with domestic money-market rate r_d and foreign rate r_f:

discount_rate r = r_d
carry b = r_d - r_f

Then:
fwd_factor = exp((b-r)tau) = exp(-r_f tau)

and the vanilla call becomes:

PV_call = S exp(-r_f tau) N(d1) - K exp(-r_d tau) N(d2)

This is the standard GK formula.

Important implementation note:
Your pricer adaptors should convert discount factors to continuous rates consistently:
r = -ln(df(T))/T

This ensures analytic vs MC vs FD parity.

---

# Appendices

## Appendix A: Normal definitions used
Standard normal pdf:
$$
\phi(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}
$$

Standard normal cdf:
$$
N(x)=\int_{-\infty}^{x}\phi(u),du
$$

These are implemented in `src/models/common/normal.py`.

---

## Appendix B: Put-call parity in carry form
From the vanilla formulas:

Call - Put = S fwd_factor - K df

So:
$$
PV_{\text{call}} - PV_{\text{put}} = S e^{(b-r)\tau} - K e^{-r\tau}
$$

This is the carry-form parity.
In FX: becomes S exp(-r_f tau) - K exp(-r_d tau).

---

## Appendix C: Behaviour at T=0 and sigma=0
- At tau=0, payoff derivatives are discontinuous (kinks for vanilla, jumps for digitals).
  Returning greeks as 0 at expiry is a deliberate “risk-system safety” policy.

- For $\sigma=0$:
  S_T = S exp(b tau) deterministically.
  Digitals become indicator functions and greeks become distributional at the boundary.
  Your digital engines handle sigma=0 explicitly and return stable outputs.

---

## Appendix D: Relationship to the standard (r,q) equity notation
Standard equity BSM typically uses:
dS/S = (r - q) dt + sigma dW
So b = r - q

Then:
fwd_factor = exp((b-r)tau) = exp(-q tau)

and the vanilla formula becomes:
PV_call = S exp(-q tau) N(d1) - K exp(-r tau) N(d2)

---

## Appendix E: “Full” carry derivative for vanilla (if you ever want it)
Your current vanilla rho_carry is a clean convention focusing on the explicit fwd_factor channel.

If instead you define:
rho_carry_full = ∂PV/∂b with r fixed,
then b affects:
- fwd_factor = exp((b-r)tau)
- d1 and d2 through the numerator

For the call:
PV = S fwd_factor N(d1) - K df N(d2)

Compute:
∂fwd_factor/∂b = tau fwd_factor
∂d1/∂b = sqrtTau/sigma
∂d2/∂b = sqrtTau/sigma

So:
∂PV/∂b
= S (tau fwd_factor) N(d1) + S fwd_factor n(d1) (sqrtTau/sigma)
  - K df n(d2) (sqrtTau/sigma)

This is the mathematically complete derivative w.r.t b.
If you decide to expose both, name them explicitly:
- rho_carry_clean (current)
- rho_carry_full  (above)

Documenting the convention is the critical part.

---