# DESIGN — quantitative foundations of `rade_static_replication`

> A quant-heavy design document. The emphasis is the mathematics: the static-replication
> learning problem, the market-data objects and their interpolation, the shock algebra,
> and the front-office pricing derivations that produce the elementary-basis PnL tensors
> consumed by the `rade_ml` hybrid GNN-RNN. Engineering rationale is kept deliberately
> light; see `README.md` for the module map.

---

## 1. The learning problem

Let a portfolio contain target trades indexed by $j$ and let the bank produce a set of
$S$ historical (or hypothetical) market scenarios. For trade $j$ under scenario $s$ we
observe a realised PnL $\Pi^{\text{tgt}}_{j,s}$. We construct, per risk factor, a basis
of *elementary* instruments indexed by $i$ and compute their scenario PnL
$\Pi^{\text{el}}_{i,s}$ analytically.

The downstream model learns a map

$$
\Pi^{\text{tgt}}_{j,\cdot} \;\approx\; \mathcal{F}_\theta\!\left(\Pi^{\text{el}}_{\cdot,\cdot},\, a_j\right),
$$

where $a_j$ are trade attributes and $\mathcal{F}_\theta$ is the GNN-RNN. The theoretical
justification that such a map exists and generalises across scenarios is **static
replication**: the value of a derivative is, to leading order, a *fixed* (scenario-independent)
combination of the values of elementary instruments. This document's job is to make the
$\Pi^{\text{el}}$ correct, consistent, and auditable.

---

## 2. Static replication

### 2.1 The Carr–Madan spanning identity

For a payoff $f(S_T)$ that is twice differentiable, and any expansion point $\kappa$,

$$
f(S_T) = f(\kappa) + f'(\kappa)\,(S_T-\kappa)
+ \int_0^{\kappa} f''(K)\,(K-S_T)^+\,\mathrm{d}K
+ \int_{\kappa}^{\infty} f''(K)\,(S_T-K)^+\,\mathrm{d}K .
$$

Taking the discounted risk-neutral expectation $V_0(f)=e^{-r_dT}\mathbb{E}^{\mathbb{Q}}[f(S_T)]$
and using $\mathbb{E}^{\mathbb{Q}}[S_T]=F$ gives the price as a **static portfolio** of a
bond, a forward, and a continuum of out-of-the-money puts and calls:

$$
V_0(f) = e^{-r_dT}f(\kappa) + f'(\kappa)\,\big(\text{fwd}\big)
+ \int_0^{\kappa} f''(K)\,P(K)\,\mathrm{d}K
+ \int_{\kappa}^{\infty} f''(K)\,C(K)\,\mathrm{d}K .
$$

The weights $\{f(\kappa), f'(\kappa), f''(K)\}$ depend only on the payoff, **not** on the
market state. That scenario-independence is precisely the inductive bias the model
exploits.

### 2.2 Discretisation into an elementary basis

We replace the strike continuum by a finite grid $\{K_m\}$ at expiries $\{T_n\}$ and a
finite instrument set (vanilla calls/puts, digitals, forwards for FX; payer/receiver
swaptions for rates). For a single risk factor with elementary PVs $V_i(\mathcal{M})$
under market state $\mathcal{M}$, the replication hypothesis becomes

$$
V^{\text{tgt}}(\mathcal{M}) \;\approx\; \sum_i w_i\, V_i(\mathcal{M}),
\qquad w_i \text{ independent of } \mathcal{M},
$$

and differencing across a scenario $s$ versus the base state $0$ gives the **PnL form**
we actually emit:

$$
\boxed{\;\Pi^{\text{tgt}}_s \approx \sum_i w_i\,\Pi^{\text{el}}_{i,s},\qquad
\Pi^{\text{el}}_{i,s} \equiv V_i(\mathcal{M}_s) - V_i(\mathcal{M}_0).\;}
$$

The model relaxes the *linear* $\sum_i w_i$ to a learned nonlinear $\mathcal{F}_\theta$ and
couples factors via the graph, but §2.1 guarantees the **basis spans** the leading-order
payoff behaviour, so the residual is controlled by grid density (§10).

---

## 3. Risk factors and the market state

A risk factor $\phi$ is the smallest market object against which an elementary instrument
can be priced. The portfolio induces a **dependency graph**: an FX factor
`FX.SPOT.USD.EUR` requires its two discounting curves `IR_CURVE_SWAP.EUR`,
`IR_CURVE_SWAP.USD`. We resolve the unique factor set, topologically sort it
(`RiskFactorUniverse.build_order`), and build each factor once.

The market state for a factor is a tuple

$$
\mathcal{M} = \big(\,\text{spot/forwards},\ \text{discount curve(s)},\ \text{vol surface/cube}\,\big),
$$

represented by a typed snapshot ($\mathcal{M}_0$, COB) and a scenario set
($\mathcal{M}_s,\ s=1,\dots,S$, resolved to **absolute** levels — §5).

---

## 4. Market-data objects and interpolation

### 4.1 Discount curve

We store continuously-compounded zero rates $z(t)$ on year-fraction pillars $t_k$, with
discount factor $P(0,t)=e^{-z(t)\,t}$ and instantaneous forward
$f(t) = -\partial_t \ln P(0,t)$.

**Two interpolation policies** (`Interpolation`):

- `LINEAR_ZERO`: linear in $z$,
  $z(t) = z_k + (z_{k+1}-z_k)\tfrac{t-t_k}{t_{k+1}-t_k}$.
- `LOG_LINEAR_DF`: linear in $\ln P$, i.e. $\ln P(0,t)$ piecewise linear. Since
  $f(t) = -\partial_t \ln P$, this is equivalent to a **piecewise-constant (flat)
  forward** between pillars — the desk-standard choice that keeps implied forwards stable:

$$
\ln P(0,t) = \ln P(0,t_k) + \frac{t-t_k}{t_{k+1}-t_k}\big(\ln P(0,t_{k+1})-\ln P(0,t_k)\big).
$$

The continuously-compounded forward over $[t_1,t_2]$ used throughout pricing is

$$
f(t_1,t_2) = \frac{\ln P(0,t_1) - \ln P(0,t_2)}{t_2 - t_1}.
$$

### 4.2 Volatility surface

A surface carries an explicit `StrikeConvention` (absolute / moneyness $K/F$ / delta) and
`VolType` (lognormal/normal) so a pricer can never misread the axis. We interpolate
**linearly in strike** and **linearly in total variance** along expiry. Define total
variance

$$
w(K,T) \equiv \sigma_{\text{imp}}^2(K,T)\,T .
$$

For $T\in[T_0,T_1]$ with $\theta=\frac{T-T_0}{T_1-T_0}$,

$$
w(K,T) = (1-\theta)\,w(K,T_0) + \theta\,w(K,T_1),
\qquad
\sigma_{\text{imp}}(K,T) = \sqrt{w(K,T)/T}.
$$

**Calendar-arbitrage** is excluded iff $w(K,\cdot)$ is non-decreasing in $T$ at fixed
moneyness; total-variance interpolation preserves this between pillars, whereas naive
linear-in-$\sigma$ interpolation does not. This is the reason for the choice and the seam
(`base.total_variance_interp`) where a SABR/SVI slice plugs in.

### 4.3 Swaption cube

A 3-D cube in $(\text{expiry}, \text{swap tenor}, \text{strike})$ holding **normal** (bp)
vols for Bachelier pricing, trilinearly interpolated. Normal vols are used because rates
can be negative and the Bachelier model (§7) is the market convention for swaptions.

---

## 5. Shock algebra

How a raw shock $\delta_s$ maps onto the COB level $x_0$ is a property of the *data
source*, not the model. `marketdata/shocks.py` makes this a first-class enum
(`ShockMode`) and resolves everything to **absolute** levels at the client/builder
boundary:

$$
x_s =
\begin{cases}
\delta_s & \text{absolute},\\
x_0 + \delta_s & \text{additive (e.g. rates, in bp)},\\
x_0\,(1+\delta_s) & \text{relative},\\
x_0\,e^{\delta_s} & \text{log-relative (e.g. spot, vols)}.
\end{cases}
$$

Broadcasting follows NumPy rules, so a base curve $x_0\in\mathbb{R}^{P}$ and a shock block
$\delta\in\mathbb{R}^{S\times P}$ yield scenario curves $x\in\mathbb{R}^{S\times P}$.
Downstream, `ScenarioSet.validate_against` asserts the resolved grids match the snapshot
pillars exactly — the check that prevents silent axis-misalignment PnL bugs. Because only
absolute states cross the boundary, pricers never need to know the original convention.

---

## 6. FX pricing — Garman–Kohlhagen

Under the domestic risk-neutral measure $\mathbb{Q}^d$, spot (domestic per foreign)
follows

$$
\mathrm{d}S_t = (r_d - r_f)\,S_t\,\mathrm{d}t + \sigma S_t\,\mathrm{d}W_t^{\,\mathbb{Q}^d},
$$

so the foreign currency behaves like a dividend-yielding asset with yield $r_f$. The
outright forward is $F = S_0\,e^{(r_d-r_f)T}$.

**Vanilla** (notional 1):

$$
C = S_0 e^{-r_fT}N(d_1) - K e^{-r_dT}N(d_2),\qquad
P = K e^{-r_dT}N(-d_2) - S_0 e^{-r_fT}N(-d_1),
$$

$$
d_{1} = \frac{\ln(S_0/K) + (r_d - r_f + \tfrac12\sigma^2)T}{\sigma\sqrt{T}},\qquad
d_2 = d_1 - \sigma\sqrt{T}.
$$

**Cash-or-nothing digital** paying 1 domestic if $S_T \gtrless K$:

$$
D_{\text{call}} = e^{-r_dT}N(d_2),\qquad D_{\text{put}} = e^{-r_dT}N(-d_2).
$$

**Forward** PV: $\;e^{-r_dT}(F-K) = S_0 e^{-r_fT} - K e^{-r_dT}.$

**Put–call parity** (a unit test asserts this to machine precision):

$$
C - P = S_0 e^{-r_fT} - K e^{-r_dT}.
$$

In the kernels $N(\cdot)$ uses $\tfrac12\,\mathrm{erfc}(-x/\sqrt2)$ so the entire pricing
path stays inside the compiled boundary. At $T\to 0$ or $\sigma\to 0$ we return discounted
intrinsic. The elementary grid places these on $(T_n, m K=mF)$ for moneyness $m$, one
ATM-forward per expiry.

---

## 7. Rates pricing — Bachelier swaptions

### 7.1 Forward swap rate and annuity

For a swap starting at $T_0$ with payment dates $T_1<\dots<T_N$ and accruals $\tau_i$, the
present value of a basis point (annuity) and the forward par swap rate are

$$
A(0) = \sum_{i=1}^{N}\tau_i\,P(0,T_i),\qquad
S(0) = \frac{P(0,T_0) - P(0,T_N)}{A(0)} .
$$

### 7.2 Bachelier (normal) swaption

Under the annuity (swap) measure $\mathbb{Q}^A$ the forward swap rate is a martingale; the
Bachelier model takes $\mathrm{d}S_t = \sigma_N\,\mathrm{d}W_t^{\mathbb{Q}^A}$. With
$d = \dfrac{S(0)-K}{\sigma_N\sqrt{T}}$, the payer/receiver swaption PVs are

$$
V_{\text{pay}} = A(0)\Big[(S(0)-K)\,N(d) + \sigma_N\sqrt{T}\,\varphi(d)\Big],
$$
$$
V_{\text{rec}} = A(0)\Big[(K-S(0))\,N(-d) + \sigma_N\sqrt{T}\,\varphi(d)\Big],
$$

with $\varphi$ the standard normal PDF. Payer$-$receiver $= A(0)\,(S(0)-K)$ recovers the
forward swap (parity). As $\sigma_N\to0$ we return $A(0)\max(\pm(S(0)-K),0)$.

Normal vol (rather than Black) is used because swap rates can be negative; the cube stores
$\sigma_N$ directly. The annuity and forward rate are computed from the (possibly shocked)
curve, so scenario PnL captures both rate-level and curve-shape moves.

---

## 8. PnL tensor construction

For each primary factor $\phi$ with elementary set $I_\phi$:

1. **Base PVs** $V_i(\mathcal{M}_0)$, $i\in I_\phi$ (`Pricer.base_prices`).
2. **Scenario PVs** $V_i(\mathcal{M}_s)$ vectorised over $s=1,\dots,S$
   (`Pricer.scenario_pnl`, one trade across the whole window in a `*_vec` kernel).
3. **Elementary PnL** $\Pi^{\text{el}}_{i,s} = V_i(\mathcal{M}_s) - V_i(\mathcal{M}_0)$,
   assembled into $\Pi^{\text{el}}_\phi \in \mathbb{R}^{|I_\phi|\times S}$.

On write, matrices are transposed to the $[\text{scenarios}\times\text{trade-ids}]$
convention `rade_ml` expects and the **market scenario axis is intersected with the
portfolio target scenario axis** (preserving market order) so elementary and target PnL
are perfectly aligned per cluster.

---

## 9. Clustering and the consumption contract

Clusters are single-asset-class by construction (`AssetClass` is always the first
grouping key) because the model assumes intra-cluster homogeneity and prices no
cross-asset replication. Each cluster $c$ emits four artifacts:

$$
\Pi^{\text{el}}_c\in\mathbb{R}^{S\times E_c},\quad
\Pi^{\text{tgt}}_c\in\mathbb{R}^{S\times J_c},\quad
a^{\text{el}}_c,\ a^{\text{tgt}}_c\ \text{(attribute dicts)},
$$

where $E_c=\sum_{\phi\in c}|I_\phi|$ over the cluster's **primary** factors (dependency-only
factors remain market data, not a replication basis). The `jobs.pkl` manifest points
`rade_ml` at every cluster.

---

## 10. Approximation and error budget

The end-to-end error decomposes as

$$
\underbrace{\varepsilon_{\text{rep}}}_{\text{finite basis (§2.2)}}
\;+\;
\underbrace{\varepsilon_{\text{model}}}_{\text{calibration / data}}
\;+\;
\underbrace{\varepsilon_{\text{interp}}}_{\text{curve/surface (§4)}}
\;+\;
\underbrace{\varepsilon_{\text{disc}}}_{\text{annuity/quadrature}} .
$$

- $\varepsilon_{\text{rep}}$ shrinks as the strike/expiry grid densifies; from §2.1 the
  leading term is $O(\Delta K^2)$ for the put/call continuum quadrature.
- $\varepsilon_{\text{interp}}$ is controlled by total-variance interpolation (calendar-arb
  safe) and flat-forward DF interpolation (stable forwards).
- $\varepsilon_{\text{disc}}$ enters the annuity sum at frequency `freq`; halving it
  roughly quarters the annuity quadrature error.

These are knobs in the config (`elementary` grids, `freq`) and the interpolation policy in
`marketdata/base.py`.

---

## 11. Numerical methods and performance (brief)

- **Compiled kernels** (`pricing/kernels`, `@njit`) keep the inner pricing loop in machine
  code with no scipy dependency ($N(\cdot)$ via `erfc`); a shim degrades to pure Python
  when Numba is absent.
- **Vectorisation**: each elementary trade is priced across all $S$ scenarios in one
  `*_vec` call; per-expiry market data (rates, vol planes) is cached.
- **Threading**: per-factor PnL is independent and fanned out over a `ThreadPoolExecutor`
  (`engine.max_workers`); compiled kernels release the GIL. Threads, not processes, avoid
  pickling large market objects.

---

## 12. Where the maths lives (extension map)

| Quant change | File |
|--------------|------|
| New payoff in the basis | `assets/<class>/instruments.py` + `generator.py` |
| New/closed-form model | `assets/<class>/pricer.py`, kernels in `pricing/kernels/` |
| Curve interpolation policy | `marketdata/common/curves.py` + `marketdata/base.py` |
| Vol interpolation (SABR/SVI) | `marketdata/base.py::total_variance_interp` + `marketdata/fx/instruments.py` |
| Shock convention | `marketdata/shocks.py` |
| New asset's market objects | `marketdata/<class>/instruments.py` + `snapshot.py` + `scenarios.py` |

---

### Appendix A — notation

| Symbol | Meaning |
|--------|---------|
| $S_0, F$ | spot, outright forward |
| $r_d, r_f$ | domestic/foreign continuously-compounded rates |
| $P(0,t)$ | discount factor to $t$ |
| $\sigma, \sigma_N$ | lognormal vol, normal (bp) vol |
| $N, \varphi$ | standard normal CDF, PDF |
| $A(0), S(0)$ | swap annuity, forward par swap rate |
| $\mathcal{M}_0,\mathcal{M}_s$ | base (COB) and scenario market states |
| $\Pi^{\text{el}}, \Pi^{\text{tgt}}$ | elementary, target scenario PnL |
