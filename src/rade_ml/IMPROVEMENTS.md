# HybridGnnRnn — Planned Improvements

> **Status**: Design specification — no code changes yet.
> **Scope**: Six architectural enhancements to the `hybrid_gnn_rnn` model targeting
> scalability (5k+ trade portfolios) and accuracy (exotic / structured products).
>
> This document is the implementation reference.  Each section is self-contained
> and includes the motivation, mathematical formulation, affected files, and
> interaction notes.

---

## Table of Contents

1. [Sparse Target Attention](#1-sparse-target-attention)
2. [Trade-Type Conditional Layers (FiLM)](#2-trade-type-conditional-layers-film)
3. [Quantile / Distributional Output](#3-quantile--distributional-output)
4. [Multi-Task Learning Heads](#4-multi-task-learning-heads)
5. [Learnable Graph Structure](#5-learnable-graph-structure)
6. [Position-Aware Encoding for P&L Curves](#6-position-aware-encoding-for-pl-curves)
7. [Cross-Cutting Interactions](#7-cross-cutting-interactions)
8. [Implementation Priority](#8-implementation-priority)

---

## 1. Sparse Target Attention

### 1.1 Problem

`TargetAttentionLayer._core_calc()` computes full dense self-attention over all
target trades.  After `_extract_target_submatrix()` slices the `[n_tgt, n_tgt]`
adjacency block, the attention score matrix is `[B, h, n_tgt, n_tgt]`.

For 5,000 targets with 4 heads at batch size 32:

$$\text{Memory} = B \times h \times n_{\text{tgt}}^2 \times 4\text{B} = 32 \times 4 \times 25\text{M} \times 4 \approx 12.8\text{ GB}$$

This is the single largest scalability blocker for large target portfolios.

### 1.2 Solution

Mirror the pattern already implemented in `FusionLayer._sparse_nbr_attention()`
(lines 249–306 of `fusion_layer.py`).  Each target trade only attends to its $k$
nearest target-trade neighbors from the adjacency submatrix.

Complexity drops from $O(B \cdot h \cdot n_{\text{tgt}}^2 \cdot d_h)$ to
$O(B \cdot h \cdot n_{\text{tgt}} \cdot k \cdot d_h)$.

At $n_{\text{tgt}} = 5000$, $k = 50$: **100x reduction** in compute and memory.

### 1.3 What It Improves

| Metric | Before | After |
|---|---|---|
| Score tensor memory | $O(n_{\text{tgt}}^2)$ | $O(n_{\text{tgt}} \cdot k)$ |
| Forward pass compute | $O(n_{\text{tgt}}^2 \cdot d_h)$ | $O(n_{\text{tgt}} \cdot k \cdot d_h)$ |
| Gradient memory | $O(n_{\text{tgt}}^2)$ | $O(n_{\text{tgt}} \cdot k)$ |
| Max trainable targets (single GPU, 16GB) | ~2,000 | ~50,000+ |

The model's learned attention pattern is unchanged because the adjacency mask
already pushes non-neighbor scores to $-10^9$ before softmax — the model is
effectively doing sparse attention but paying the full dense cost.

### 1.4 Files to Change

**`attention_layer.py` — `TargetAttentionLayer`**

- `_extract_target_submatrix()`: Return a `tf.SparseTensor` directly instead of
  calling `tf.sparse.to_dense()` at the end.  The submatrix is already built as
  a SparseTensor internally.

- New method `_sparse_target_attention()`: Following the
  `FusionLayer._sparse_nbr_attention()` pattern:
  1. Extract rows/cols from the sparse submatrix indices.
  2. Build `[n_tgt, k]` padded neighbor index array via
     `tf.RaggedTensor.from_value_rowids`.
  3. Gather neighbor keys/values: `tf.gather(k_proj, nbr_idx, axis=2)` →
     `[B, h, n_tgt, k, d_h]`.
  4. Per-neighbor scores via element-wise dot product (not full matmul).
  5. `tf.sequence_mask` for padding, softmax over the $k$ dimension.
  6. Weighted sum over neighbors for context.

- `_core_calc()`: Route to `_sparse_target_attention()` when the adjacency is
  sparse; fall back to the current dense path for dense inputs.

- `call()`: Pass the sparse submatrix (instead of dense) to `_core_calc()`.

**`config.py`**: The `k_nbrs: 50` setting already exists in
`attention_layer.general` but is currently unused.  It becomes the sparsity
parameter.

**No changes** to the data pipeline, `model.py`, or any other layer.

---

## 2. Trade-Type Conditional Layers (FiLM)

### 2.1 Problem

Every trade — vanilla call, barrier option, autocallable, interest rate swap —
flows through identical GNN, RNN, and fusion pathways with shared weights.  The
model has no mechanism to specialise representations based on trade type.

A vanilla European option and a path-dependent barrier option receive the same
transformations, forcing the model to find a single weight set that works for all
trade types simultaneously.

### 2.2 Solution

Feature-wise Linear Modulation (FiLM) conditioning.  Given a trade's type
embedding $\mathbf{e}_{\text{type}}$, the network learns per-type scale and shift
parameters that modulate hidden representations:

$$\gamma = W_\gamma \cdot \mathbf{e}_{\text{type}} + b_\gamma$$

$$\beta = W_\beta \cdot \mathbf{e}_{\text{type}} + b_\beta$$

$$\hat{\mathbf{h}} = \gamma \odot \mathbf{h} + \beta$$

where $\mathbf{h}$ is the hidden representation at a given layer and the
modulation is element-wise.

This is lightweight — two small Dense layers per conditioning point — but allows
entirely different feature importance patterns per trade type without duplicating
the full network.

### 2.3 What It Improves

- **Accuracy on mixed portfolios**: A barrier option can emphasise the
  barrier-level feature while a vanilla focuses on moneyness/vol.
- **Exotic trade learning**: Path-dependent and structured products have
  fundamentally different P&L dynamics.  Conditioning lets the shared backbone
  adapt its internal representations per type.
- **Convergence speed**: The model doesn't need to implicitly learn
  type-dependent behavior through attention alone.
- **Fusion gating**: The relative importance of structural (GNN) vs temporal
  (RNN) information varies by trade type.  FiLM on the gate logit makes this
  trade-off type-specific.

### 2.4 Files to Change

**New class `FiLMConditioner`** (new file `layers/film_layer.py` or added to
existing layer file):
- Takes a trade-type embedding `[T, d_type]` (already available from
  `TradeAttributeEncoder` as one-hot `product_type_embedding`).
- Produces `(gamma, beta)` each of shape `[T, d_hidden]` via two Dense layers.

**`gnn_layers.py` — `GnnBlock.call()`**:
- After each GNN sub-layer output (line 150), apply FiLM modulation before
  LayerNorm / activation.
- Trade-type embedding passed as additional input to `GnnBlock`.

**`fusion_layer.py` — `FusionLayer.call()`**:
- Apply FiLM after the cross-attention output (line 159) and before the gating
  mechanism (line 163).
- Alternatively, condition the gate logit itself so the GNN/RNN mixing ratio is
  type-dependent.

**`model.py` — `HybridGnnRnn`**:
- Extract the trade-type embedding from `trade_features` or pass it as a separate
  input key (`trade_type_embedding`).
- Instantiate `FiLMConditioner` layers (one per conditioning point).
- Thread the type embedding through to `GnnBlock` and `FusionLayer`.

**`config.py`**:
- New `film_conditioning` section under `general`:

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch for backward compatibility |
| `condition_gnn` | `bool` | `True` | Apply FiLM after GNN sub-layers |
| `condition_fusion` | `bool` | `True` | Apply FiLM on fusion gate |
| `condition_attention` | `bool` | `False` | Apply FiLM after attention |
| `type_embedding_dim` | `int` | inferred | Dimension of type embedding |

**Data pipeline (`build.py`)**:
- Pass `product_type_embedding` as a separate tensor in `static_inputs` (not
  merged into `trade_features`) so the model can distinguish type dimensions.

---

## 3. Quantile / Distributional Output

### 3.1 Problem

`TargetPnlOutput` produces a single scalar per target trade per scenario — a
point estimate of P&L.  The MSE loss optimises for the conditional mean.  This
gives no information about the shape, spread, or tail behavior of the P&L
distribution — information that is critical for risk management.

### 3.2 Solution

Replace the single scalar output with multiple quantile predictions.  For a set
of quantile levels $\boldsymbol{\tau} = \{\tau_1, \ldots, \tau_Q\}$, the model
predicts $\hat{q}_{\tau_j}$ for each target trade at each quantile level.

The training loss is the pinball (quantile) loss:

$$\mathcal{L}_\tau(y, \hat{q}_\tau) = \begin{cases} \tau \cdot (y - \hat{q}_\tau) & \text{if } y \geq \hat{q}_\tau \\ (1 - \tau) \cdot (\hat{q}_\tau - y) & \text{if } y < \hat{q}_\tau \end{cases}$$

Total loss summed over all quantile levels and target trades:

$$\mathcal{L} = \frac{1}{B \cdot n_{\text{tgt}} \cdot Q} \sum_{b,i,j} \mathcal{L}_{\tau_j}\!\left(y_{b,i},\; \hat{q}_{\tau_j, b, i}\right)$$

Default quantile levels: $\tau \in \{0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99\}$.

### 3.3 What It Improves

- **Risk metrics from one forward pass**: 1% quantile = 99% VaR, average of tail
  quantiles = Expected Shortfall, spread between 5%–95% = confidence interval.
  No Monte Carlo needed at inference.
- **Tail sensitivity**: MSE treats all errors equally.  Quantile loss at extreme
  levels (1%, 99%) forces the model to capture tail behavior where P&L risk
  actually matters.
- **Uncertainty quantification**: Wide quantile spread = high uncertainty = model
  is less confident about that trade.
- **Asymmetric P&L profiles**: Options have asymmetric payoffs.  A point estimate
  loses this asymmetry.  Quantile outputs preserve it.
- **Encoder regularisation**: Gradients from the full distribution shape
  (not just the mean) enrich shared representations.

### 3.4 Files to Change

**`projection_layer.py` — `TargetPnlOutput`**:
- Output changes from `[B, n_tgt]` to `[B, n_tgt, Q]`.
- `_residual_fc_2`: Output units from `1` to `Q`.
- `_baseline_kernels`: Shape from `[n0, attn_dim]` to `[n0, attn_dim, Q]`.
- `_baseline_biases`: Shape from `[n0]` to `[n0, Q]`.
- kNN output mixing: neighbor selection shared across quantiles; mixing applied
  per quantile.
- New attribute `quantile_levels` storing the $\tau$ values.

**New loss class `QuantileLoss`** (in `training/` or
`models/hybrid_gnn_rnn/losses.py`):
- Implements the pinball loss summed over quantile levels and targets.
- Optional composite: MSE on median ($\tau=0.5$) + quantile loss on tails,
  with configurable weighting.
- Optional quantile crossing penalty:
  $\lambda \sum_{j} \max(0, \hat{q}_{\tau_j} - \hat{q}_{\tau_{j+1}})$ to
  enforce monotonicity.

**`model.py` — `HybridGnnRnn.call()`**:
- Output shape changes from `[B, n_tgt]` to `[B, n_tgt, Q]`.
- Backward compatible via config flag `output_mode`.

**`config.py`**:
- Add to `projection_layer`:

| Key | Type | Default | Description |
|---|---|---|---|
| `output_mode` | `str` | `"point"` | `"point"` or `"quantile"` |
| `quantile_levels` | `List[float]` | `[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]` | Quantile levels |
| `quantile_crossing_penalty` | `float` | `0.0` | Monotonicity regulariser weight |

**`evaluation/metrics.py`**:
- Quantile calibration: is the 5% quantile exceeded 5% of the time?
- Pinball loss metric.
- VaR backtesting: Kupiec test, Christoffersen test.

**Training pipeline**:
- When `output_mode == "quantile"`, use `QuantileLoss` instead of MSE.
- Target tensor `y` is unchanged (`[B, n_tgt]`) — loss compares each quantile
  prediction against the single observed value.

**Evaluation / plotting**:
- Fan charts showing quantile predictions as colored bands around the median.
- PIT (Probability Integral Transform) histogram for calibration assessment.

---

## 4. Multi-Task Learning Heads

### 4.1 Problem

The model has a single objective: predict target P&L.  The shared encoder
(GNN + RNN + Fusion + Attention) receives gradient signal only through the P&L
loss.  This means:

- The GNN only gets indirect gradients, propagated back through fusion and
  attention — limiting embedding quality for elementary trades.
- The encoder can learn shortcuts that predict mean P&L without capturing
  economically meaningful structure (e.g. Greeks, correlations).
- With a small number of target trades relative to model parameters, the model
  is prone to overfitting.

### 4.2 Solution

Add auxiliary prediction heads that branch off the shared encoder at various
points.  Each predicts a different quantity.  Auxiliary losses are summed with
task-specific weights into the total loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pnl}} + \sum_{t} w_t \cdot \mathcal{L}_t$$

### 4.3 Candidate Auxiliary Tasks

**Greeks prediction** (branches from attention output):
- Predict delta, gamma, vega per target trade: `[B, n_tgt, n_greeks]`.
- Greeks are first-order sensitivities of P&L — if the model understands
  Greeks, its P&L predictions must be internally consistent.
- Loss: MSE on normalised Greeks.

**Elementary P&L reconstruction** (branches from fused features):
- Reconstruct per-elementary-trade mean P&L from fused features: `[B, T, 1]`.
- Autoencoder-style regularisation that forces the encoder to retain
  information about the full trade universe, not just targets.
- Provides direct, dense gradient signal to the GNN for every elementary
  trade (not just targets).
- Loss: MSE.

**Trade-type classification** (branches from fused features):
- Predict product type from fused embedding: `[B, T, n_types]`.
- Ensures the encoder doesn't discard type information during fusion.
- Loss: categorical cross-entropy.

**Pairwise correlation prediction** (branches from GNN output):
- Given two trades' embeddings, predict their P&L correlation.
- Forces the GNN to learn meaningful similarity structure.
- Loss: MSE on Fisher-transformed correlations.

### 4.4 What It Improves

- **Representation quality**: Each task forces the encoder to preserve different
  input aspects.
- **Regularisation**: Additional loss terms reduce overfitting.
- **Data efficiency**: More learning signal from the same training data.
- **Interpretability**: Accurate Greek predictions validate that internal
  representations capture economically meaningful quantities.

### 4.5 Files to Change

**New file `layers/auxiliary_heads.py`**:
- `GreeksPredictionHead`: Dense from `[B, n_tgt, d_a]` → `[B, n_tgt, n_greeks]`.
- `ReconstructionHead`: Dense from `[B, T, d_f]` → `[B, T, 1]`.
- `TradeTypeClassificationHead`: Dense + softmax from `[B, T, d_f]` →
  `[B, T, n_types]`.

**`model.py` — `HybridGnnRnn.run_model()`**:
- After step 3 (fusion), branch off the reconstruction head.
- After step 4 (attention), branch off the Greeks head.
- Return a dictionary when multi-task is enabled:
  `{"pnl": pnl_pred, "greeks": greeks_pred, "reconstruction": recon_pred}`.

**`config.py`**:
- New `auxiliary_tasks` section:

| Key | Type | Default | Description |
|---|---|---|---|
| `greeks_prediction` | `bool` | `False` | Enable Greeks auxiliary head |
| `greeks_weight` | `float` | `0.1` | Loss weight for Greeks task |
| `reconstruction` | `bool` | `False` | Enable elementary P&L reconstruction |
| `reconstruction_weight` | `float` | `0.05` | Loss weight for reconstruction |
| `trade_type_classification` | `bool` | `False` | Enable type classification head |
| `trade_type_weight` | `float` | `0.02` | Loss weight for classification |

**Training pipeline**:
- Custom composite loss combining primary + weighted auxiliary losses.
- Data pipeline needs to supply Greeks labels if available (added as optional
  field in `static_inputs`).
- For reconstruction, labels are the elementary P&L itself (already in dataset).

**Evaluation**:
- Separate metrics per task head.
- Validation monitors auxiliary losses independently.

---

## 5. Learnable Graph Structure

### 5.1 Problem

The adjacency matrix is fixed at data build time by `TradeGraphBuilder`.  It uses
sklearn's `NearestNeighbors` with handcrafted alpha-weighted Euclidean distance
over encoded trade attributes.  The alpha weights are hyperparameters, and the
graph never changes during training.

Limitations:

- Economically meaningful relationships may not follow attribute similarity
  (e.g. a delta hedge pair — long call + short underlying — has very different
  attributes but a strong P&L relationship).
- The graph is optimised for none of the model's objectives.
- Quality depends heavily on alpha weights, $k$ value, and distance metric.

### 5.2 Solution

A differentiable graph learning module that produces a soft, trainable adjacency
end-to-end.

**Edge scoring function** (bilinear):

$$s_{ij} = \mathbf{h}_i^{\top} \, W_{\text{edge}} \, \mathbf{h}_j + b_{\text{edge}}$$

where $W_{\text{edge}} \in \mathbb{R}^{d \times d}$ is learnable (or low-rank
$W = U V^{\top}$, $U, V \in \mathbb{R}^{d \times r}$ to reduce parameters).

**Differentiable top-$k$ sparsification**:
For each node $i$, keep only the $k$ highest-scoring edges:

$$\mathcal{N}(i) = \text{top-}k_j\{s_{ij}\}$$

$$\alpha_{ij} = \frac{\exp(s_{ij} / \tau)}{\sum_{j' \in \mathcal{N}(i)} \exp(s_{ij'} / \tau)} \quad \text{for } j \in \mathcal{N}(i)$$

The straight-through estimator (STE) allows gradients to flow through the
discrete top-$k$ during backprop.

**Temperature annealing**: Start with high $\tau$ (soft attention over many
neighbors) and anneal to low $\tau$ (hard top-$k$) during training for stable
convergence.

### 5.3 What It Improves

- **Discovers hidden relationships**: Finds connectivity that attribute similarity
  misses (hedge pairs, correlation-linked trades, netting set members).
- **Task-adapted structure**: Optimised jointly with P&L prediction, so it
  discovers the connectivity that actually helps accuracy.
- **Removes hyperparameter sensitivity**: Eliminates dependence on alpha weights,
  $k$, and distance metric.
- **Inductive at inference**: The scoring function can compute edge scores for
  any trade pair given feature embeddings — new trades are automatically connected.

### 5.4 Files to Change

**New file `layers/graph_learner.py` — `LearnableGraphLayer`**:
- Bilinear scoring: `W_edge` of shape `[d_feat, d_feat]` (or low-rank
  `[d_feat, r]` × `[r, d_feat]`).
- Top-$k$ sparsification: `tf.math.top_k` on score rows → sparse index/value
  arrays.
- Straight-through estimator for backward pass.
- Temperature parameter with linear or cosine annealing schedule.
- Output: `tf.SparseTensor` of shape `[T, T]`.

**`model.py` — `HybridGnnRnn.run_model()`**:
- Before step 1 (GNN), run `LearnableGraphLayer(trade_features)` to produce a
  learned adjacency.
- Two strategies:
  - **Replace**: Use only the learned adjacency.
  - **Combine** (safer): $A_{\text{combined}} = \alpha \cdot A_{\text{fixed}} + (1 - \alpha) \cdot A_{\text{learned}}$
    where $\alpha$ can also be learned.

**`config.py`**:

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch |
| `mode` | `str` | `"combine"` | `"combine"` or `"replace"` |
| `scoring` | `str` | `"bilinear"` | `"bilinear"`, `"mlp"`, or `"cosine"` |
| `top_k` | `int` | `50` | Edges per node |
| `temperature_init` | `float` | `1.0` | Initial softmax temperature |
| `temperature_min` | `float` | `0.1` | Final temperature after annealing |
| `rank` | `int` | `32` | Low-rank factorisation rank |
| `candidate_limit` | `int` | `0` | 0 = all-pairs; >0 = approx top-k |

**Scalability concern**: All-pairs scoring is $O(T^2)$.  For 5,000+ trades,
support a `candidate_limit` that narrows candidates via locality-sensitive hashing
or random subsets before scoring.

**No changes** to `graph_builder.py` — the fixed graph is still built and used as
the initial adjacency.

---

## 6. Position-Aware Encoding for P&L Curves

### 6.1 Problem

`RnnBlock` receives `pnl_history` of shape `[B, S, T_e]` and relies entirely on
its recurrent state to encode positional / temporal information.  There is no
explicit encoding of *when* each time step occurs.

P&L curves have strong calendar effects: month-end rebalancing, quarterly rolls,
option expiry clusters, central bank meeting days, holiday effects.  An LSTM can
theoretically learn these from sequence order, but in practice it:

- Struggles with long-range positional dependencies.
- Cannot generalise to sequences of different lengths or sampling frequencies.
- Burns recurrent capacity on learning "what position am I at" instead of
  temporal dynamics.

### 6.2 Solution

Add explicit temporal information to each time step before it enters the RNN:

$$\tilde{\mathbf{x}}_t = \mathbf{x}_t + \text{PE}(t)$$

Three modes:

**Sinusoidal** (Vaswani et al., 2017):
$$\text{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

- Fixed, deterministic, generalises to unseen sequence lengths.
- Requires projection if $d_{\text{pe}} \neq T_e$.

**Learned**:
- Trainable embedding: `Embedding(max_seq_len, d_pe)` indexed by position.
- More flexible but cannot generalise beyond training sequence length.

**Calendar-aware**:
- Encode actual date features: day-of-week, day-of-month, month-of-year,
  days-to-next-expiry, is-month-end.
- Small Dense layer maps `[S, n_date_features]` → `[S, d_pe]`.
- Most powerful for financial time series; requires date metadata in the dataset.

### 6.3 What It Improves

- **Calendar sensitivity**: Distinguishes "expiry day P&L move" (gamma spike,
  delta jump) from "mid-month" (normal vol).
- **Long-range temporal structure**: Sinusoidal encodings give explicit distance
  information between time steps — quarterly effects learnable directly.
- **Sequence length generalisation**: With sinusoidal encodings, a model trained
  on 20-step sequences can be applied to 50-step sequences without retraining.
- **RNN convergence**: The LSTM focuses on dynamics instead of position tracking.
- **Richer fusion**: If the RNN hidden state retains better temporal structure,
  the fused representation carries more informative temporal features per trade.

### 6.4 Files to Change

**New class `PositionalEncoding`** (in `layers/rnn_layers.py` or new
`layers/positional.py`):
- Sinusoidal mode: compute $\text{PE}(t, 2i)$, $\text{PE}(t, 2i+1)$ for all
  positions and dimensions.  Output `[S, d_pe]`.
- Learned mode: `tf.keras.layers.Embedding(max_seq_len, d_pe)`.
- Calendar mode: Dense layer from `[S, n_date_features]` → `[S, d_pe]`.

**`rnn_layers.py` — `RnnBlock`**:
- In `call()`, before the LSTM stack (line 173), apply positional encoding:
  - **Additive** ($d_{\text{pe}} = T_e$ or via projection):
    `pnl_history = pnl_history + PE`
  - **Concatenative** (first LSTM input dim increases by $d_{\text{pe}}$):
    `pnl_history = concat([pnl_history, PE], axis=-1)`

**`model.py` — `HybridGnnRnn`**:
- If calendar mode, `date_features` added as an optional input key.
- For sinusoidal / learned modes, no new inputs needed.

**`config.py`**:
- Add to `rnn_layer`:

| Key | Type | Default | Description |
|---|---|---|---|
| `positional_encoding` | `str` | `"none"` | `"none"`, `"sinusoidal"`, `"learned"`, `"calendar"` |
| `pe_dim` | `int` | `32` | Positional embedding dimension |
| `pe_mode` | `str` | `"add"` | `"add"` or `"concat"` |
| `max_seq_len` | `int` | `256` | For learned mode only |

**Data pipeline (`build.py`)**:
- Sinusoidal / learned: no changes needed.
- Calendar mode: sliding window construction must also produce date feature
  arrays `[n_windows, seq_len, n_date_features]`.  Date features computed from
  scenario date index.  Added to `_make_ds()` as an additional dataset element.

---

## 7. Cross-Cutting Interactions

These improvements are not independent — they reinforce each other:

| Pair | Interaction |
|---|---|
| **Sparse Target Attn + Learnable Graph** | Learned graph produces sparse edges that feed directly into sparse attention — both must agree on $k$. |
| **FiLM + Multi-Task** | Type conditioning makes auxiliary heads (especially Greeks) more accurate because each type has different Greek profiles. |
| **Quantile Output + Multi-Task** | Greeks prediction constrains quantile predictions to be physically consistent (quantile spread should correlate with gamma). |
| **Positional Encoding + Quantile Output** | Calendar-aware encoding helps predict wider quantile spreads on high-vol days (expiry, meetings). |
| **FiLM + Learnable Graph** | Edge scoring can be type-conditioned: "barrier options should connect to their underlying, not to other barriers". |
| **Sparse Target Attn + Quantile Output** | $Q$ quantile outputs multiply attention memory by $Q$, making sparse attention even more critical. |
| **Learnable Graph + Multi-Task** | Correlation prediction auxiliary task directly trains the graph scoring function toward economically meaningful connectivity. |

---

## 8. Implementation Priority

Recommended order balancing impact, effort, and dependencies:

| Priority | Improvement | Impact | Effort | Dependencies |
|---|---|---|---|---|
| 1 | Sparse Target Attention | Unlocks 5k+ targets | Low | Pattern exists in FusionLayer |
| 2 | FiLM Conditioning | Major accuracy gain for mixed portfolios | Medium | Needs type embedding split in data |
| 3 | Quantile / Distributional Output | Directly useful for risk | Medium | New loss class needed |
| 4 | Position-Aware Encoding | Better temporal learning | Low-Medium | Calendar mode needs date metadata |
| 5 | Multi-Task Learning Heads | Better representations + regularisation | Medium | Optional Greeks labels |
| 6 | Learnable Graph Structure | Advanced; highest ceiling | High | Needs careful annealing + STE |

**Suggested implementation phases**:

- **Phase 1** (scalability): Items 1 + 4 — unblock large portfolios and improve
  temporal learning with minimal architecture change.
- **Phase 2** (accuracy): Items 2 + 3 — add type conditioning and distributional
  output for better predictions and risk metrics.
- **Phase 3** (advanced): Items 5 + 6 — multi-task regularisation and learned
  graph structure for maximum model quality.
