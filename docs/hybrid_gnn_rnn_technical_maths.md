# The Hybrid GNN–RNN Model — A Technical & Mathematical Reference

*Part of a personal quant/ML document library. Written both as a study text and
as interview preparation (TradingHub Senior Quant Developer, Risk & Pricing).*

This document has three jobs:

1. **Teach the general ML theory** you need to reason about *any* deep model
   (function approximation, risk minimisation, representation learning, sequence
   and graph models, attention).
2. **Explain the financial theory** the model is built on — **static replication /
   spanning** — rigorously.
3. **Dissect this specific model** (`HybridGnnRnn`, as implemented in
   `src/rade_ml_pt/models/hybrid_gnn_rnn/`) layer by layer, with the maths
   *between* layers, why each component works, and why the whole is well-suited to
   the target problem.

> **Reading convention.** Every technical block is followed by a
> **▶ In plain English** commentary that restates the idea without jargon. If you
> only read those, you still get the whole story.

**Contents**

- Part I — Machine-learning foundations
- Part II — The financial problem: static replication theory
- Part III — The Hybrid GNN–RNN model, layer by layer
- Part IV — Why this architecture fits this problem
- Part V — Limitations, failure modes, and fixes
- Part VI — Interview questions & graded follow-ups
- Part VII — References & further reading

---

# Part I — Machine-learning foundations

## I.1 What "a model" actually is

A supervised ML model is a **parametric function** $f_\theta : \mathcal X \to
\mathcal Y$ chosen from a family $\{f_\theta : \theta \in \Theta\}$ to approximate
an unknown relationship $y \approx f^\star(x)$ from data
$\mathcal D = \{(x_i, y_i)\}_{i=1}^N$. "Training" = searching $\Theta$ for the
$\theta$ that makes $f_\theta$ predict well on data it has **not** seen.

We formalise "predict well" with a **loss** $L(\hat y, y)\ge 0$ and minimise the
**empirical risk** (with regularisation $R$):

$$
\hat\theta \;=\; \arg\min_{\theta}\; \frac1N \sum_{i=1}^N L\big(f_\theta(x_i), y_i\big) \;+\; \lambda R(\theta).
$$

This is **Empirical Risk Minimisation (ERM)**. It is a *proxy* for what we really
want — low **population risk** $\mathbb E_{(x,y)}[L(f_\theta(x),y)]$ — and the gap
between the two is the **generalisation gap**.

> **▶ In plain English.** A model is an adjustable formula. Training turns the
> knobs so the formula reproduces the examples we have, *hoping* it then works on
> new examples. The loss is the scorecard; regularisation is a penalty that stops
> the formula from over-fitting the specific examples.

## I.2 Why neural networks can represent almost anything

A single-hidden-layer network $f_\theta(x) = \sum_k a_k\,\sigma(w_k^\top x + b_k)$
is, by the **Universal Approximation Theorem** (Cybenko 1989; Hornik 1991), dense
in the space of continuous functions on compact sets: for any continuous $g$ and
tolerance $\varepsilon$ there exist parameters making $\|f_\theta - g\|_\infty <
\varepsilon$. **Depth** buys this *efficiently* — deep compositions represent
certain functions with exponentially fewer units than shallow ones (Telgarsky
2016; Montúfar et al. 2014).

A deep network is a composition of simple layers:

$$
f_\theta = \ell_L \circ \ell_{L-1} \circ \cdots \circ \ell_1,
\qquad
\ell_k(h) = \sigma\big(W_k h + b_k\big).
$$

Each layer is an **affine map then a pointwise non-linearity**. Without the
non-linearity the whole stack collapses to a single affine map — non-linearity is
what gives expressive power.

> **▶ In plain English.** Stacked simple transformations can bend input space into
> almost any shape. Making the network *deep* rather than *wide* is usually a far
> cheaper way to get that flexibility.

## I.3 How training actually happens: gradients & backprop

We minimise the loss by **gradient descent**:

$$
\theta \leftarrow \theta - \eta \,\nabla_\theta \mathcal L(\theta),
$$

with learning rate $\eta$. The gradient of a deep composition is computed by the
**chain rule**, organised as **backpropagation**: a forward pass caches
activations; a backward pass propagates $\partial \mathcal L / \partial(\cdot)$
from output to input, reusing intermediate results. For a layer $h_{k} =
\sigma(W_k h_{k-1})$,

$$
\frac{\partial \mathcal L}{\partial W_k}
= \delta_k\, h_{k-1}^\top,
\qquad
\delta_{k-1} = \big(W_k^\top \delta_k\big)\odot \sigma'(\cdot),
\qquad
\delta_k \equiv \frac{\partial \mathcal L}{\partial (W_k h_{k-1})}.
$$

In practice we use **stochastic** gradient descent (mini-batches) and adaptive
optimisers (Adam), which rescale each coordinate by running estimates of the
gradient's first and second moments.

> **▶ In plain English.** The network learns by nudging every weight a little in
> the direction that reduces the error, and backprop is just the bookkeeping that
> works out each weight's share of the blame efficiently using the chain rule.

## I.4 The central tension: bias, variance, generalisation

For squared loss the expected test error decomposes as

$$
\mathbb E\big[(y - \hat f(x))^2\big]
= \underbrace{\big(\mathbb E[\hat f(x)] - f^\star(x)\big)^2}_{\text{bias}^2}
+ \underbrace{\mathrm{Var}[\hat f(x)]}_{\text{variance}}
+ \underbrace{\sigma^2_{\text{noise}}}_{\text{irreducible}}.
$$

- **Under-fitting** (high bias): model too simple / too constrained.
- **Over-fitting** (high variance): model memorises noise; large train–test gap.

We control this with **regularisation** (weight decay $\lambda\|\theta\|^2$,
dropout, early stopping), **inductive bias** (architecture that encodes problem
structure), and **more/better data**. The single most powerful lever is
**inductive bias**: choosing an architecture whose built-in assumptions match the
problem (convolutions for images, recurrence for sequences, message passing for
graphs) drastically cuts the data needed to generalise.

> **▶ In plain English.** Too rigid and you miss the pattern; too flexible and you
> memorise the noise. The art is to bake the *right assumptions* into the model so
> it needs less data to find the true pattern. Most of this document's model is
> about baking in the *right* assumptions for trade P&L.

## I.5 Representation learning & embeddings

Deep models work because intermediate layers learn **representations**
(embeddings): vectors $z = g_\phi(x) \in \mathbb R^{d}$ in which the downstream
task becomes easy (often linear). Similar inputs map to nearby vectors; useful
directions become linear. An embedding is *learned* geometry.

> **▶ In plain English.** Instead of hand-crafting features, the network invents
> its own coordinate system in which the answer is simple. "Similar trades end up
> near each other" is a learned embedding.

## I.6 Sequence models (the RNN family)

To model ordered data $x_{1:S}$ a **recurrent neural network** carries a hidden
state:

$$
h_t = \phi\big(W_x x_t + W_h h_{t-1} + b\big), \qquad t = 1,\dots,S.
$$

The final $h_S$ summarises the sequence. Vanilla RNNs suffer **vanishing/exploding
gradients** over long horizons (the Jacobian product $\prod_t W_h\,\mathrm{diag}(\phi')$
shrinks/blows up). **LSTM** fixes this with a gated cell:

$$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1},x_t]), &\quad i_t &= \sigma(W_i[h_{t-1},x_t]),\\
o_t &= \sigma(W_o[h_{t-1},x_t]), &\quad \tilde c_t &= \tanh(W_c[h_{t-1},x_t]),\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t, &\quad h_t &= o_t \odot \tanh(c_t).
\end{aligned}
$$

The **cell state** $c_t$ is an additive memory highway; the gates $f,i,o$ learn to
keep/forget/expose information, so gradients survive long horizons. **GRU** is a
lighter 2-gate variant. **TCN** (temporal convolutional network) drops recurrence
for **dilated causal convolutions**: output at $t$ depends only on inputs $\le t$
(causality), and dilation $2^i$ per layer gives an exponentially growing receptive
field $1+(k-1)(2^L-1)$ with fully parallel training and stable gradients.

> **▶ In plain English.** An RNN reads a sequence one step at a time, keeping a
> running summary. LSTMs add gated "memory cells" so they don't forget the distant
> past. A TCN gets the same long memory with a stack of look-back filters that
> train faster because there's no step-by-step loop.

## I.7 Graph models (the GNN family)

When data are **nodes connected by edges** (not a grid or a sequence), we use
**message passing**:

$$
h_i^{(l+1)} = \mathrm{UPDATE}\Big(h_i^{(l)},\; \mathrm{AGG}\big(\{h_j^{(l)} : j \in \mathcal N(i)\}\big)\Big).
$$

Each node updates itself from an **aggregate of its neighbours**. The aggregator
must be **permutation-invariant** (sum/mean/max), so the network is invariant to
node ordering — the correct inductive bias for sets/graphs. Stacking $L$ layers
lets information travel $L$ hops. Expressive power is bounded by the
**Weisfeiler–Lehman** graph-isomorphism test (Xu et al. 2019); GraphSAGE
(Hamilton et al. 2017) is the inductive mean/max-aggregation variant used here.

> **▶ In plain English.** On a network of related things, each item repeatedly
> "asks its neighbours how they're doing" and updates itself. Do this a few times
> and information spreads across the network. Because it only cares about *who* the
> neighbours are, not their order, it's the natural model for relationships.

## I.8 Attention & transformers

**Attention** computes a data-dependent weighted average. Given queries $Q$, keys
$K$, values $V$:

$$
\mathrm{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V.
$$

Row $i$ of the output is $\sum_j \alpha_{ij} v_j$ with weights
$\alpha_{ij} \propto \exp(q_i^\top k_j / \sqrt d)$ — "how relevant is $j$ to $i$."
**Multi-head** attention runs several such maps in parallel subspaces and
concatenates. The $\sqrt d$ scaling keeps logits from saturating softmax.
**Masked** attention forbids certain $j$ (here: non-neighbours in the graph) by
setting their logits to $-\infty$.

> **▶ In plain English.** Attention lets each item look over a set of others and
> decide, on the fly, whom to listen to and how much. "Masking" just means it's
> only allowed to listen to a permitted subset (its economic neighbours).

---

# Part II — The financial problem: static replication theory

## II.1 Replication, spanning, and the law of one price

The bedrock: if two portfolios have **identical payoffs in every state of the
world**, they must have the **same price** (no-arbitrage / law of one price). If a
set of **elementary instruments spans** the payoff of a target instrument, then
holding the spanning combination *replicates* the target — its price and its P&L
under any scenario equal the target's.

Let a target payoff be $g(S_T)$ in terminal underlying $S_T$. **Static
replication** seeks fixed weights $\{w_k\}$ over elementary payoffs
$\{e_k(S_T)\}$ (bonds, forwards, options) with

$$
g(S_T) \;=\; \sum_k w_k\, e_k(S_T) \quad \text{for all } S_T,
$$

held **without rebalancing**. Contrast **dynamic replication** (Black–Scholes
delta-hedging), which needs continuous trading in the underlying.

> **▶ In plain English.** If you can build a fixed basket of simple, liquid trades
> that pays exactly what a complicated trade pays in every scenario, then the
> complicated trade is "just" that basket — same value, same P&L. Static means you
> set the basket once and don't touch it.

## II.2 The Carr–Madan spanning formula (the theoretical core)

Any twice-differentiable European payoff $g(S_T)$ can be **exactly** decomposed
around any anchor $\kappa$:

$$
g(S_T) = \underbrace{g(\kappa)}_{\text{bond}}
+ \underbrace{g'(\kappa)\,(S_T-\kappa)}_{\text{forward}}
+ \underbrace{\int_0^{\kappa} g''(K)\,(K-S_T)^+\,dK}_{\text{puts}}
+ \underbrace{\int_{\kappa}^{\infty} g''(K)\,(S_T-K)^+\,dK}_{\text{calls}}.
$$

**This is a replication recipe:** hold $g(\kappa)$ in bonds, $g'(\kappa)$
forwards, and a *continuum* of options with density $g''(K)$ (the local
convexity). Taking expectations under the risk-neutral measure prices *any*
European claim from the option surface. The dual result, **Breeden–Litzenberger**,
recovers the risk-neutral density as $\phi(K) = e^{rT}\,\partial^2 C/\partial K^2$.

**Discretisation.** With only finitely many strikes $\{K_m\}$ we replace the
integrals by a quadrature:

$$
g(S_T) \approx g(\kappa) + g'(\kappa)(S_T-\kappa) + \sum_m w_m\,(S_T-K_m)^{+} + \sum_m w'_m\,(K_m-S_T)^{+},
$$

with weights $w_m$ from $g''$ and the strike spacing. **The replication is exact
in the continuum and approximate on a finite grid** — the error is a quadrature
error concentrated where $g''$ is large (near kinks / discontinuities).

> **▶ In plain English.** There's a precise mathematical recipe: any European
> payoff equals some bonds + forwards + a spread of options, where you hold *more*
> options where the payoff bends more. With infinitely many strikes it's exact;
> with a finite set of real options it's a very good approximation, worst near
> sharp kinks. This is *why* a model built on elementary trades can reproduce
> target P&L — the theory guarantees a combination exists.

## II.3 Linear (delta-1) products: the easy case

An FX forward's value is $V = e^{-r_f T} S - K e^{-r_d T}$ (domestic per unit
foreign). Its scenario P&L is **linear** in spot and the two discount factors,
with a small **bilinear** cross term $\Delta S \cdot \Delta \mathrm{df}$. So a
forward is *spanned exactly* by {spot/forward, domestic DF, foreign DF} building
blocks — no options needed. This is why linear products *should* be the easiest to
replicate; when they are not, the culprit is almost always **data/scaling**, not
theory (see Part V).

> **▶ In plain English.** Simple forwards and cash flows are straight-line
> functions of a few rates, so a handful of building blocks reproduces them
> perfectly. If a model struggles here, suspect the plumbing (units, scaling),
> not the maths.

## II.4 From replication theory to the ML model

The link that motivates the whole architecture:

| Replication theory | Model component |
|---|---|
| Elementary instruments $e_k$ | Elementary trades / their scenario P&L $p$ |
| Target payoff $g$ | Target trade P&L $y$ |
| Replicating weights $w_k$ (from $g',g''$) | Learned baseline read-out $\langle z_i,k_i\rangle$ |
| Weights vary smoothly across strike/tenor | GNN + k-NN interpolation over attribute space |
| Curvature $g''$ / higher-order Greeks | Non-linear residual MLP + RNN temporal terms |
| Continuum → finite-grid error | Residual correction absorbs quadrature gap |

The static-replication ideal is the **linear map** $y \approx W p$. The hybrid
model learns a **structured, non-linear generalisation** of $W$: it shares
replication structure across similar trades (graph), conditions on book dynamics
(RNN), and corrects the finite-grid/curvature gap (residual MLP).

> **▶ In plain English.** The model is not fighting the theory — it *is* the
> theory, made learnable. The "linear basket of building blocks" is the backbone;
> the neural parts learn how that basket changes smoothly from trade to trade and
> patch up the bits a finite set of building blocks can't cover exactly.

---

# Part III — The Hybrid GNN–RNN model, layer by layer

## III.0 Objects and the end-to-end map

| Symbol | Meaning | Code shape |
|---|---|---|
| $N,\,T_e,\,n$ | total / elementary / target trades | — |
| $B,\,S,\,p$ | batch, history length, attribute dim | — |
| $X\in\mathbb R^{N\times p}$ | static trade features | `trade_features` |
| $A\in\mathbb R^{N\times N}$ | sparse row-normalised adjacency | `adjacency` |
| $P\in\mathbb R^{B\times S\times T_e}$ | elementary P&L history | `pnl_history` |
| $H\in\mathbb R^{N\times d_g}$ | GNN node embeddings | `gnn_features` |
| $r\in\mathbb R^{B\times d_r}$ | RNN temporal embedding | `rnn_features` |
| $F\in\mathbb R^{B\times N\times d_f}$ | fused features | `fused_features` |
| $Z\in\mathbb R^{B\times n\times d_a}$ | target-attended features | `attended_features` |
| $\hat y\in\mathbb R^{B\times n}$ | predicted target P&L (scaled) | `predictions` |

$$
X,A \xrightarrow{\text{GNN}} H \;\big\|\; P \xrightarrow{\text{RNN}} r
\xrightarrow{\text{Fusion}} F \xrightarrow{\text{TargetAttn}} Z \xrightarrow{\text{Projection}} \hat y.
$$

> **▶ In plain English.** Two parallel "readers" — one reads the *structure* of the
> book (who resembles whom), one reads the *time series* of building-block P&L —
> are merged, focused on the target trades, and turned into a P&L number per
> target.

## III.1 Graph construction (`utilities/graph_builder.py`)

Attributes are encoded, then a k-NN graph is built with an $\alpha$-weighted
distance and a Gaussian RBF kernel with **adaptive bandwidth**:

$$
d_{ij}^2 = \sum_f \alpha_f (x_{if}-x_{jf})^2,
\qquad
w_{ij} = \exp\!\Big(-\tfrac{d_{ij}^2}{2\sigma_i^2}\Big),
\qquad
\sigma_i = \mathrm{median}(d_{i,1},\dots,d_{i,k}),
$$

then row-normalise $A \leftarrow D^{-1}A$ so each row sums to 1.

**Why it works.** The $\alpha$ weights encode economic priors (moneyness/tenor
matter more than, say, book id). The adaptive $\sigma_i$ makes the kernel
scale-free across dense/sparse regions. Row normalisation turns $Ax$ into a convex
combination — a *bounded, stable* neighbour average and the exact operator
GraphSAGE-mean needs.

> **▶ In plain English.** We connect each trade to its most economically similar
> trades, weighting the links by similarity, and normalise so "listening to
> neighbours" is a proper weighted average. This is the learned analogue of an
> interpolation grid over the strike/tenor surface.

## III.2 GNN stream — structural embeddings

`GnnBlock` stacks $L$ **linear** message-passing sublayers; the block applies
LayerNorm, activation, dropout between them and a projected **residual**:

$$
H = \sigma\Big( Z^{(L)} + W_{\text{proj}} X \Big),\quad
Z^{(l)} = \text{GNN}_l\big(\sigma(\mathrm{LN}(Z^{(l-1)}))\big).
$$

**GraphSAGE:** $h_i' = W_s h_i + W_n\,\mathrm{AGG}_{j\in\mathcal N(i)} h_j$
(mean $=Ah$, or per-feature max).

**Mixed GraphSAGE (default):**
$h_i' = W\,[\,h_i \,\|\, \mathrm{mean}_{\mathcal N(i)}h \,\|\, \max_{\mathcal N(i)}h\,]$
— captures **smooth** (mean → interpolation) and **salient** (max → tail/dominant
neighbour) structure together.

**Graphormer:** sparse multi-head attention with zero-initialised **centrality
encodings** $z_q,z_k(\deg_{\text{in}})$, so it begins as vanilla attention and
learns to weight structurally important trades.

**Why it works / why it fits.** Replicating weights vary *smoothly* across the
trade surface — neighbouring strikes have similar decompositions. Message passing
is exactly a learned smoother over that surface, giving the model the correct
inductive bias and enabling **generalisation to unseen trades** by interpolation.
The stream is **batch-invariant** (depends only on $X,A$) and is therefore
**cached** in eval and recomputed in training.

> **▶ In plain English.** The GNN builds a "fingerprint" for each trade that
> blends in its economic neighbours. Because similar trades genuinely have similar
> replication recipes, this sharing makes the model data-efficient and able to
> handle trades it has never seen by leaning on similar ones.

## III.3 RNN stream — temporal embeddings

`RnnBlock` maps $P\in\mathbb R^{B\times S\times T_e}$ to $r\in\mathbb R^{B\times
d_r}$ via LSTM / BiLSTM / GRU / TCN / Dense (last-step MLP). The recurrent final
state (or TCN last step) is the embedding; BiLSTM concatenates both directions
($2d_r$).

**Why it works / why it fits.** For **linear** products the contemporaneous
elementary P&L slice is a *sufficient statistic* — the `dense` backend is the
theoretically minimal input. Recurrence/TCN adds value for **path- and
regime-dependence and curvature** (vega/theta/barriers): the state summarises how
the building blocks have been evolving, information a single snapshot lacks.

> **▶ In plain English.** This reader watches how the simple building-block P&Ls
> have moved over recent scenarios. For straight-line products the latest snapshot
> is enough; for options and path-dependent trades, the recent *history* carries
> the extra information the model needs.

> **Magnitude caveat.** LSTM/GRU states are $\tanh$/$\sigma$-bounded to
> $[-1,1]$; the *magnitude* of predictions must come from the downstream **linear
> read-outs**, not the recurrent state (see Part V).

## III.4 Fusion — cross-attention between structure and time

`FusionLayer` broadcasts both streams to $[B,N,\cdot]$ and forms an
**asymmetric, adjacency-masked** attention:

$$
Q = W_q^{\text{rnn}}\tilde r + W_q^{\text{gnn}}\tilde H,\qquad
K = W_k \tilde H,\qquad V = W_v \tilde H,
$$
$$
\alpha_{ij} = \operatorname{softmax}_{j\in\mathcal N(i)}\!\Big(\tfrac{Q_i^\top K_j}{\sqrt{d_h}}\Big),\qquad
\mathrm{attn}_i = \sum_j \alpha_{ij} V_j,
$$
then a **gated residual**:
$g=\sigma(W_g[\mathrm{attn}\,\|\,\tilde r])$,
$F=\mathrm{LN}(g\odot\mathrm{attn}+(1-g)\odot\tilde r)$.

**Why K,V are structural but Q is joint.** Keys/values define *what content is
available to attend over* — here the **trade-identity/structural** content. The
query defines *what each trade is looking for*, and it should reflect both what
the trade **is** (structure) and how the book **has moved** (time). This keeps
attention anchored to economically meaningful neighbours while the temporal signal
modulates *how much* to draw from them. The gate then decides, per feature,
whether to trust the structural mix or the raw temporal stream.

> **▶ In plain English.** Each trade asks a question shaped by both its identity
> and recent market moves, then gathers answers from its structural neighbours. A
> learned "trust dial" blends that structural answer with the plain time-series
> signal.

## III.5 Target self-attention (`TargetAttentionLayer`)

Gather the $n$ target rows of $F$ and run **masked multi-head self-attention**
over the target–target adjacency submatrix, then a Transformer FFN with residual
+ LayerNorm:

$$
Z = \mathrm{LN}\!\Big(\mathrm{FFN}\big(\mathrm{LN}(\mathrm{MHA}(F_{\mathcal T})+F_{\mathcal T})\big) + (\cdot)\Big).
$$

**Why it works.** Correlated targets (same underlying, adjacent strikes) should
share information before the read-out — this is a learned cross-sectional
smoother restricted to genuine economic links.

> **▶ In plain English.** Before making final calls, related target trades compare
> notes with each other so their predictions stay mutually consistent.

## III.6 Projection to P&L (`TargetPnlOutput`) — the crucial layer

Prediction = **baseline + residual**.

**Baseline (train targets):** each of $n_0$ targets has a learned kernel $k_i$ and
bias $b_i$,

$$
\text{base}_i = \langle z_i, k_i\rangle + b_i,
$$

optionally weight-normed $k_i = \mathrm{softplus}(g_i)\,\hat k_i$ (decoupled
direction/amplitude). **This dot-product read-out is precisely the learned
analogue of the linear replication weights $W$** — $z_i$ is the trade's fused
representation, $k_i$ its replicating direction.

**Baseline (unseen targets):** k-NN blend of train baselines in attribute space,

$$
w_{ij} = \operatorname{softmax}_j\big(\tau\cos(a_i,a_j)\big),\qquad
\text{base}_i = \sum_{j\in\mathrm{kNN}(i)} w_{ij}\,\text{base}_j.
$$

**Residual:** shared MLP on $[z_i \,\|\, a_i]$ adds a non-linear correction
(curvature / finite-grid gap), damped for new targets. Optional attention-conditioned
positive scale/bias applies to new targets only.

$$
\boxed{\;\hat y_i = \underbrace{\langle z_i,k_i\rangle + b_i}_{\text{linear replication read-out}} \;+\; \underbrace{\mathrm{MLP}([z_i\,\|\,a_i])}_{\text{curvature / correction}}\;}
$$

**Why it works / why it fits.** The decomposition mirrors the theory: a linear
spanning term plus a curvature correction. It also gives a clean
**generalisation** mechanism (k-NN baseline) and a place for the deep stack to
add only what linearity misses.

> **▶ In plain English.** The final step says "your P&L is mostly a weighted sum
> of building-block P&Ls (the replication), plus a small learned adjustment for
> the non-linear bits." Unseen trades borrow their baseline from similar known
> trades.

## III.7 Losses & standardisation

**Loss in use — Huber + quantile:**

$$
L = \mathrm{Huber}_\delta(y,\hat y) + \alpha\,L_\tau(y,\hat y),\qquad
L_\tau = \max(\tau e,(\tau-1)e),\ e=y-\hat y,
$$

defaults $\delta=2,\ \tau=0.95,\ \alpha=0.3$. Huber is quadratic within $\delta$
and **linear beyond** (robust to P&L outliers); the $\tau=0.95$ pinball term adds
**asymmetric upper-tail pressure** so the model respects large-move magnitude that
MSE would smooth away.

**Standardisation** (fit on train scenarios only): `standard` (per-column),
`signed_log`, and the added `global` / `signed_log_global` (single pooled
mean/std). Because targets are already per-unit-notional (comparable across
columns), per-column scaling **re-inflates the tails of low-variance convex
targets** (a legitimate ITM value can become $>20\sigma$), dominating the loss and
exceeding what the bounded stack can emit. A **global** scaler weights each target
by its genuine per-unit variance and removes that artefact. (See Part V.)

> **▶ In plain English.** The loss is a robust error measure with an extra nudge to
> get big moves right. The scaling choice matters a lot: normalising each trade
> separately can turn a normal large-but-real value into a monster the model
> can't reproduce — a shared scale fixes that.

## III.8 Systems details (worth knowing cold)

- **Lazy init:** `LazyLinear`/`UninitializedParameter` resolve dims on first
  forward (e.g. `attn_dim`, `baseline_trade_count` from input shapes).
- **GNN caching:** cached in eval, recomputed in train (gradient flow); cleared on
  `train()/eval()` switch.
- **Sparse attention:** padded neighbour gathers → $O(Nk)$, not $O(N^2)$; a dense
  path exists for small graphs.
- **Inference:** predictions are in scaled space; `post_infer` inverse-transforms
  with the saved scaler, then $\times(\text{notional}\cdot\text{sign})$ to recover
  notional P&L.

---

# Part IV — Why this architecture fits this problem

1. **Right backbone.** The linear baseline read-out is the static-replication map
   $W p$ the theory guarantees exists — the model starts from the correct
   hypothesis class.
2. **Right inductive bias for the surface.** Replication weights vary smoothly in
   strike/tenor/underlying; the GNN + k-NN interpolation encode exactly that
   smoothness, giving data efficiency and unseen-trade generalisation.
3. **Right handling of non-linearity.** Curvature ($g''$, vega/gamma) and the
   finite-grid quadrature gap are exactly what the residual MLP and temporal
   stream are there to absorb.
4. **Right treatment of cross-sectional structure.** Correlated targets share
   information through masked attention rather than being predicted in isolation.
5. **Right tail behaviour (when scaled correctly).** Huber + 0.95-quantile with a
   pooled per-unit scale targets robust central fit *and* large-move magnitude.

> **▶ In plain English.** Every piece maps to something real: the backbone is the
> replication basket, the graph is the "similar trades" surface, the deep bits are
> the non-linear corrections, and the loss/scaling are tuned so big P&L moves
> aren't ignored. That alignment is why it's a good fit, not just a generic net.

---

# Part V — Limitations, failure modes & fixes

**1. Magnitude bounding.** Softmax convex-combinations, LayerNorm, and
$\tanh/\sigma$ gates all bound internal magnitudes; prediction *scale* rests on the
linear read-outs. Very convex targets with rare large moves get under-shot under
MSE/Huber. *Fixes:* quantile tail term; weight-norm; an **unbounded linear head
warm-started from OLS/Ridge**; correct target scaling.

**2. Per-column scaling artefact.** On per-unit-notional targets, per-column
standardisation inflates low-variance convex columns to $>20\sigma$, so a single
legitimate ITM scenario dominates the loss and is unreachable. *Fix:* `global` /
`signed_log_global` pooled scaling; evaluate in original/notional units.

**3. Loss weights by per-unit variance, not economics.** Because notional was
divided out, a global scale de-emphasises low-per-unit-variance trades that may
carry huge notional. *Fixes:* **notional-weighted loss/eval**; **per-group**
scaling for heterogeneous multi-asset books.

**4. Collinear elementary sets.** Options at nearby strikes are highly collinear →
unstable/exploding OLS weights and an ill-conditioned baseline. *Fix:* Ridge
regularisation for the linear floor; dimensionality reduction of the elementary
grid.

**5. Discontinuous / hard-to-span payoffs.** Digitals need a *co-strike* digital
elementary (vanillas only approximate them, with quadrature error at the jump).
*Fix:* ensure the elementary set truly spans the target family.

**6. No hard no-arbitrage constraints.** It's statistical replication, not a
pricer. *Optional fixes:* monotonicity/convexity penalties, put–call-parity
residual penalties, or a non-negative replicating combination.

> **▶ In plain English.** The main ways it breaks are (a) it structurally can't
> shout loud enough for rare big moves, and (b) the data plumbing (scaling, units,
> collinearity) can manufacture problems that look like model failures. Most fixes
> are about scaling, an unbounded linear head, and making sure the building blocks
> genuinely span the targets.

---

# Part VI — Interview questions & graded follow-ups

Rehearse out loud. Each item: **core answer** then likely **follow-ups**.

### VI.1 Conceptual

**Q. What problem does this model solve and why is ML appropriate?**
It predicts target-trade scenario P&L from elementary-trade P&L via a learned,
graph-aware generalisation of static replication — cheaper than full reval,
generalises across the trade surface. ML is appropriate because the
replication weights vary non-linearly across trades and we want to learn/interpolate
them from data.
- *Follow-up: When would you NOT use ML here?* If the book is small and linear,
  a regularised linear regression (Ridge) per target is simpler, auditable and
  usually sufficient — ML earns its keep on large, heterogeneous, non-linear books
  and unseen-trade generalisation.
- *Follow-up: How do you validate it's not just overfitting?* Train-only scaler
  fit; out-of-time validation split; compare to an OLS/Ridge floor; evaluate in
  notional units per desk; check residuals vs Greeks.

**Q. Explain static replication and how the model embodies it.**
Carr–Madan spanning: any European payoff = bonds + forwards + a $g''$-weighted
strip of options; finite grid → approximate. The model's linear baseline read-out
is $Wp$; the GNN interpolates weights across the surface; the residual absorbs
curvature/grid error.
- *Follow-up: Derive the spanning formula.* Twice-integrate $g(S_T)$ by parts
  around $\kappa$ using $(S_T-K)^+$ / $(K-S_T)^+$ as the Green's functions of
  $\partial_K^2$.
- *Follow-up: What's Breeden–Litzenberger?* $\phi(K)=e^{rT}\partial^2_K C$ — the
  risk-neutral density is the option-price convexity in strike.

### VI.2 Architecture

**Q. Why a GNN rather than plain regression?**
Shares replication structure across similar trades; sparse economic
neighbourhoods; generalises to unseen trades by interpolation.
- *Follow-up: What limits a GNN's expressiveness?* The Weisfeiler–Lehman bound;
  over-smoothing with depth (embeddings converge as $L$ grows) — mitigated by
  residuals and modest depth.
- *Follow-up: Why row-normalise the adjacency?* Makes aggregation a convex
  combination → bounded, stable, and equals GraphSAGE-mean.

**Q. Why keys/values from the GNN but query from both streams (fusion)?**
K/V = available content (structure); Q = what each trade seeks (structure +
recent dynamics). Anchors attention to economic neighbours while time modulates
intensity.
- *Follow-up: Why the $\sqrt{d}$ scaling in attention?* Keeps dot-product logits
  $O(1)$ so softmax doesn't saturate and gradients don't vanish.
- *Follow-up: What does the gate add over a plain residual?* A per-feature,
  data-dependent trust dial between structural and temporal signal.

**Q. Walk me through the projection layer.**
Baseline (learned kernel·attn + bias) = linear replication read-out; k-NN blend
for unseen trades; residual MLP for curvature; optional new-target scale/bias.
- *Follow-up: Why weight-norm the kernel?* Decouples direction from amplitude so
  magnitude can be learned independently and stably.
- *Follow-up: How exactly does an unseen trade get a prediction?* Cosine-softmax
  k-NN over attributes blends nearby train baselines, plus the shared residual.

### VI.3 Training & data

**Q. Where does prediction magnitude come from, and why under-shoot big moves?**
From the linear read-outs; bounded internals can't produce large scaled outputs,
and MSE/Huber regress to the mean on rare tails.
- *Follow-up: Concretely fix it?* 0.95-quantile term; weight-norm; unbounded
  OLS/Ridge-warm-started linear head; correct (global) scaling.

**Q. Explain the per-column vs global scaling issue.**
Per-unit targets are already comparable; per-column scaling inflates low-variance
convex tails to tens of sigma → loss domination + unreachable targets. Global
pooled scaling weights by true per-unit variance.
- *Follow-up: Downside of global scaling on a 4k book?* De-emphasises
  low-per-unit-variance but high-notional trades; over-compresses heterogeneous
  sub-books → use notional-weighted loss and per-group scaling.
- *Follow-up: Why is scaling invertible and safe?* It's an affine map with saved
  parameters; `inverse_transform` recovers raw P&L exactly.

**Q. Why Huber + quantile rather than MSE?**
Huber is robust to P&L outliers (linear tails); the pinball term restores
asymmetric large-move sensitivity.
- *Follow-up: What does $\tau=0.95$ do to the optimum?* Shifts the conditional
  target toward the 95th percentile — penalises under-prediction ~19:1 vs
  over-prediction.

### VI.4 Systems / production

**Q. Complexity and scaling to thousands of trades?**
Sparse k-NN attention $O(Nk)$; GNN cached across the epoch; RNN $O(BST_ed_r)$.
- *Follow-up: Biggest scaling risks?* Scaler heterogeneity and loss weighting
  (fix: per-group scaling, notional weighting); graph build cost (fix: approximate
  k-NN).
- *Follow-up: How do you serve it?* Cache GNN/adjacency; batch scenarios;
  inverse-transform then rescale by notional; monitor per-trade residuals.

**Q. How would you unit-test a model like this?**
Shape/serialisation round-trips; a linear-only synthetic where OLS is optimal and
the model must match; gradient-flow checks; scaler invertibility; permutation
tests on the graph.
- *Follow-up: A financial sanity test?* Put–call parity: predicted call − put must
  track forward within tolerance.

### VI.5 "Senior" curveballs

- *If the model can't beat Ridge on a linear book, what's your triage order?*
  Check units/sign of targets → scaling (global) → collinearity (Ridge floor) →
  elementary spanning → only then architecture.
- *Would you deploy a black box in risk/pricing?* Only with a linear-explainable
  backbone, per-trade attribution (baseline vs residual), benchmark vs Ridge,
  and monitoring — which this design supports because the baseline *is* an
  interpretable replication read-out.
- *How would you make it arbitrage-aware?* Soft penalties (convexity/monotonicity,
  parity residuals) or constrain the baseline to a non-negative elementary
  combination.

> **▶ In plain English (how to perform in the room).** Lead with the *economics*
> (replication), then the *architecture choice that encodes it*, then the *maths*,
> then the *failure modes and how you'd debug them*. Interviewers in risk/pricing
> care most that you can connect the ML to the finance and that you know how it
> breaks.

---

# Part VII — References & further reading

**Finance / replication**
- Carr & Madan (1998), *Towards a Theory of Volatility Trading* — spanning formula.
- Breeden & Litzenberger (1978) — risk-neutral density from option prices.
- Derman, Ergener & Kani (1995) — static replication of exotic options.
- Hull, *Options, Futures, and Other Derivatives* — Greeks, replication basics.

**Machine learning**
- Cybenko (1989); Hornik (1991) — universal approximation.
- Goodfellow, Bengio & Courville, *Deep Learning* — backprop, regularisation.
- Hochreiter & Schmidhuber (1997) — LSTM. Bai, Kolter & Koltun (2018) — TCN.
- Hamilton, Ying & Leskovec (2017) — GraphSAGE. Xu et al. (2019) — GIN / WL bound.
- Ying et al. (2021) — Graphormer / centrality encoding.
- Vaswani et al. (2017) — attention / transformers.

**In this repo**
- `src/rade_ml_pt/models/hybrid_gnn_rnn/` — model, layers, config.
- `src/rade_ml_pt/utilities/graph_builder.py` — k-NN + RBF adjacency.
- `src/rade_ml_pt/features/transforms/standardiser.py` — scalers (incl. global).
- `src/rade_ml_pt/training/losses.py` — Huber/quantile losses.

---

*Template note (for the library): reuse this skeleton per project — Foundations →
Domain theory → Model layer-by-layer (with ▶ plain-English) → Why it fits →
Limitations → Interview Q&A → References.*
