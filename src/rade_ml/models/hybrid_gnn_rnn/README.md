# HybridGnnRnn — Technical Model Documentation

> **Model class**: `src.rade_ml.models.hybrid_gnn_rnn.model.HybridGnnRnn`
> **Framework**: TensorFlow / Keras 3
> **Version**: 1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation and Objectives](#2-motivation-and-objectives)
3. [Industry Use Cases](#3-industry-use-cases)
4. [Architecture Overview](#4-architecture-overview)
5. [Data Pipeline](#5-data-pipeline)
6. [Layer-by-Layer Technical Detail](#6-layer-by-layer-technical-detail)
   - 6.1 [GnnBlock — Structural Embedding](#61-gnnblock--structural-embedding)
   - 6.2 [RnnBlock — Temporal Embedding](#62-rnnblock--temporal-embedding)
   - 6.3 [FusionLayer — Cross-Modal Attention with Gating](#63-fusionlayer--cross-modal-attention-with-gating)
   - 6.4 [TargetAttentionLayer — Inter-Trade Self-Attention](#64-targetattentionlayer--inter-trade-self-attention)
   - 6.5 [TargetPnlOutput — Dynamic Projection with kNN Transfer](#65-targetpnloutput--dynamic-projection-with-knn-transfer)
7. [Sparse Tensor Design and Scalability](#7-sparse-tensor-design-and-scalability)
8. [Loss Function and Training Objective](#8-loss-function-and-training-objective)
9. [Inference and Generalisation to New Trades](#9-inference-and-generalisation-to-new-trades)
10. [Hyperparameter Reference](#10-hyperparameter-reference)
11. [Numerical Stability and Reproducibility](#11-numerical-stability-and-reproducibility)
12. [Computational Complexity](#12-computational-complexity)
13. [Limitations and Future Work](#13-limitations-and-future-work)

---

## 1. Executive Summary

**HybridGnnRnn** is a graph-temporal deep learning model designed for portfolio-level Profit & Loss (P&L) simulation. It jointly learns:

- **Structural relationships** between financial instruments (trades) via a Graph Neural Network (GNN) operating over a k-nearest-neighbor trade similarity graph.
- **Temporal dynamics** of portfolio P&L via a Recurrent Neural Network (RNN) operating over historical scenario time series.

The two representation streams are fused via a gated cross-attention mechanism, then projected to per-target-trade P&L predictions. The architecture is **inductive**: it generalises to unseen trades at inference time without retraining, using attribute-space nearest-neighbor transfer.

---

## 2. Motivation and Objectives

### 2.1 The Problem

Traditional approaches to portfolio P&L simulation fall into two categories:

1. **Full revaluation**: Re-price every instrument under each scenario using analytic or Monte Carlo engines. Accurate but computationally prohibitive for large portfolios (O(T x S) pricing calls, where T = trades, S = scenarios).
2. **Taylor approximation (Greeks-based)**: Fast but inaccurate for large market moves, path-dependent products, or non-linear payoffs.

Neither approach captures the **cross-instrument correlation structure** inherent in a real portfolio, nor do they generalise to newly onboarded trades without re-running the full pipeline.

### 2.2 The Objective

HybridGnnRnn aims to:

1. **Learn a compressed, graph-aware representation** of the portfolio that captures how each trade's P&L co-moves with its structural neighbors (similar delta, vega, maturity, underlying, etc.).
2. **Model temporal regime dynamics** via recurrent processing of historical P&L scenarios, capturing autocorrelation, volatility clustering, and regime shifts.
3. **Fuse both streams** in a principled attention-based framework where the graph structure constrains which trades can influence each other, preventing attention weight dilution across unrelated instruments.
4. **Project to target-trade P&L** using a dual baseline + residual architecture that separates the learned amplitude (scale) from the learned shape (direction), enabling stable transfer to unseen trades via kNN output-space mixing.
5. **Scale to production portfolios** of 10,000+ trades with sub-quadratic memory via sparse neighborhood attention.

### 2.3 Design Principles

| Principle | Implementation |
|---|---|
| **Inductive** | GNN uses sampling-based aggregation (GraphSAGE), not spectral filters tied to a fixed graph |
| **Sparse-first** | Adjacency stored as `tf.SparseTensor`; attention is O(T x k) not O(T^2) |
| **Modular** | Each layer is independently configurable, serializable, and testable |
| **Transfer-capable** | Projection layer uses kNN output-space mixing for zero-shot new trade P&L |
| **Deterministic** | Full seed control (`tf.keras.utils.set_random_seed`, `enable_op_determinism`) for regulatory reproducibility |

---

## 3. Industry Use Cases

### 3.1 Front Office — Real-Time P&L Prediction

Desk-level P&L prediction for intra-day risk monitoring. The model replaces full revaluation for peripheral trades while the pricing engine focuses on material positions.

### 3.2 Risk Management — Scenario Analysis and Stress Testing

Generate portfolio P&L distributions under historical or hypothetical stress scenarios (e.g., 2008 credit crisis, 2020 COVID drawdown, bespoke regulatory scenarios). The graph structure ensures correlated instruments move together under stress.

### 3.3 xVA / Counterparty Credit Risk

Approximate CVA/DVA exposure profiles by simulating portfolio-level P&L paths. The temporal component captures wrong-way risk dynamics; the graph component models netting set dependencies.

### 3.4 Portfolio Construction and Optimisation

Evaluate the marginal impact of adding or removing trades from a portfolio. The inductive architecture allows scoring candidate trades without retraining.

### 3.5 Model Risk Management

Serve as a challenger model to production pricing engines, flagging instruments or scenarios where the pricing engine and the learned model diverge beyond tolerance.

### 3.6 Trade Lifecycle — New Trade Onboarding

When a new instrument is onboarded, extend the trade graph with its attribute-derived edges and generate P&L predictions immediately using kNN transfer from similar calibrated trades.

---

## 4. Architecture Overview

### 4.1 High-Level Data Flow

```
                    ┌──────────────────────┐
                    │    Input Dictionary   │
                    │                      │
                    │  trade_features      │  [T, p]        encoded trade attributes
                    │  pnl_history         │  [B, S, T_e]   elementary P&L time series
                    │  adjacency (sparse)  │  [T, T]        k-NN trade similarity graph
                    │  target_indices      │  [n_tgt]       indices of target trades
                    │  elementary_indices  │  [T_e]         indices of elementary trades
                    └─────────┬────────────┘
                              │
                 ┌────────────┴─────────────┐
                 │                          │
                 ▼                          ▼
    ┌────────────────────┐     ┌────────────────────┐
    │     GNN Block      │     │     RNN Block      │
    │                    │     │                    │
    │  trade_features    │     │  pnl_history       │
    │  + adjacency       │     │  [B, S, T_e]       │
    │  [T, p] → [T, d_g]│     │  → [B, d_r]        │
    └────────┬───────────┘     └────────┬───────────┘
             │                          │
             │   LayerNorm              │   LayerNorm
             │                          │
             ▼                          ▼
    ┌────────────────────────────────────────────────┐
    │              Fusion Layer                       │
    │                                                │
    │  Cross-attention (RNN queries GNN keys)        │
    │  + Sparse neighborhood masking [T, k]          │
    │  + Sigmoid gating                              │
    │  → [B, T, d_f]                                 │
    └────────────────────┬───────────────────────────┘
                         │   LayerNorm
                         ▼
    ┌────────────────────────────────────────────────┐
    │         Target Attention Layer                  │
    │                                                │
    │  Gather target-trade features                  │
    │  Self-attention over [n_tgt] target trades     │
    │  + Adjacency-masked softmax                    │
    │  + FFN sublayer                                │
    │  → [B, n_tgt, d_a]                             │
    └────────────────────┬───────────────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────────┐
    │        Target PnL Output (Projection)          │
    │                                                │
    │  Baseline: per-target learned kernel + bias    │
    │  Residual: MLP(attention ‖ attributes)         │
    │  New trades: kNN output-space mixing           │
    │  → [B, n_tgt]  (predicted P&L per target)     │
    └────────────────────────────────────────────────┘
```

### 4.2 Tensor Dimensions Reference

| Symbol | Meaning | Typical Range |
|---|---|---|
| T | Total trades (elementary + target) | 100 — 50,000+ |
| T_e | Elementary trades (basis trades after dimensionality reduction) | 10 — 5,000 |
| n_tgt | Target trades to predict | 1 — 500 |
| B | Batch size (number of scenario windows) | 16 — 128 |
| S | Sequence length (lookback window) | 1 — 252 |
| p | Encoded trade attribute dimension | 10 — 50 |
| k | k-NN graph degree | 4 — 100 |
| d_g | GNN output dimension | 64 — 256 |
| d_r | RNN output dimension | 64 — 256 |
| d_f | Fusion output dimension | 32 — 128 |
| d_a | Attention output dimension | 16 — 64 |
| h | Number of attention heads | 1 — 8 |

---

## 5. Data Pipeline

The data pipeline (`src.rade_ml.data.hybrid_gnn_rnn.build`) transforms raw portfolio data into model-ready `tf.data.Dataset` objects through the following stages:

### 5.1 P&L Loading and Standardisation

Elementary and target P&L matrices are loaded and standardised (zero mean, unit variance) using a fitted `StandardScaler`. The scaler parameters are stored for inverse-transforming predictions back to P&L space.

### 5.2 Dimensionality Reduction (Basis Selection)

For large portfolios, a pivoted QR decomposition on the SVD-compressed P&L covariance matrix selects a representative subset of elementary trades (the "basis"). This reduces the RNN input dimension from T_e to a manageable rank while preserving the dominant modes of P&L variation:

\[
\mathbf{P} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top, \quad \mathbf{Q}\mathbf{R}\boldsymbol{\Pi}^\top = \mathbf{V}_{:k}^\top \implies \text{basis} = \boldsymbol{\Pi}_{:k}
\]

### 5.3 Trade Attribute Encoding

Raw trade attributes (moneyness, maturity, Greeks, product type, etc.) are encoded into a fixed-length numeric vector per trade:
- **Numeric fields**: standard-scaled.
- **Categorical fields**: one-hot or label encoded.
- **Time-to-maturity**: exponential decay features at multiple lambda scales.
- **Multi-label fields**: multi-hot binary encoding.

### 5.4 Trade Graph Construction (k-NN)

A k-nearest-neighbor graph is built over the encoded attribute space using Euclidean distance. The resulting adjacency matrix is stored as a `tf.SparseTensor` with row-normalized weights. The graph captures structural similarity: trades with similar Greeks, maturity profiles, and underlying risk factors are connected.

### 5.5 Dataset Assembly

The static tensors (trade features, adjacency components, index arrays) and per-scenario tensors (P&L windows) are combined into a `tf.data.Dataset`. The adjacency SparseTensor is decomposed into three dense component tensors (`indices`, `values`, `dense_shape`) for serialization compatibility and reassembled inside the model's `call()` method.

---

## 6. Layer-by-Layer Technical Detail

### 6.1 GnnBlock — Structural Embedding

**File**: `layers/gnn_layers.py`
**Input**: `(trade_features [T, p], adjacency [T, T])`
**Output**: `[T, d_g]`

#### Purpose

Learns a d_g-dimensional embedding for every trade that encodes both its own attributes and its structural position in the portfolio graph. Trades with similar neighbors receive similar embeddings.

#### Architecture

The GnnBlock stacks L GNN sub-layers (configurable as `GraphSage` or `MixedGraphSage`) with:
- Inter-layer normalization (LayerNorm)
- Dropout between layers
- A residual skip connection from input to output (with a learned linear projection when dimensions differ)

#### 6.1.1 GraphSAGE Sub-Layer

The standard inductive GraphSAGE update rule:

\[
\mathbf{h}_v^{(l+1)} = \sigma\!\left(\mathbf{W}_{\text{self}}^{(l)} \, \mathbf{h}_v^{(l)} + \mathbf{W}_{\text{neigh}}^{(l)} \, \text{AGG}\!\left(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)
\]

Where:
- \(\mathbf{h}_v^{(l)} \in \mathbb{R}^{d_l}\) is the embedding of trade v at layer l.
- \(\mathcal{N}(v)\) is the neighbor set of v from the k-NN graph.
- \(\text{AGG}\) is mean or max aggregation.
- \(\mathbf{W}_{\text{self}}^{(l)}, \mathbf{W}_{\text{neigh}}^{(l)} \in \mathbb{R}^{d_{l+1} \times d_l}\) are learned weight matrices.

**Mean aggregation** (default) uses sparse matrix-vector multiplication:

\[
\text{AGG}_{\text{mean}}(\mathcal{N}(v)) = \mathbf{A}_{v,:} \, \mathbf{H}^{(l)} = \sum_{u \in \mathcal{N}(v)} a_{vu} \, \mathbf{h}_u^{(l)}
\]

Since the adjacency is row-normalized (\(\sum_u a_{vu} = 1\)), this is implemented as a single `tf.sparse.sparse_dense_matmul(A, H)` call — O(nnz) where nnz = T x k.

**Max aggregation** uses `tf.math.unsorted_segment_max` over the sparse edge list — also O(nnz).

#### 6.1.2 MixedGraphSage Sub-Layer

Concatenates self-features, mean-aggregated neighbors, and max-aggregated neighbors before a single linear projection:

\[
\mathbf{h}_v^{(l+1)} = \sigma\!\left(\mathbf{W}_{\text{fuse}}^{(l)} \left[\mathbf{h}_v^{(l)} \;\|\; \text{AGG}_{\text{mean}} \;\|\; \text{AGG}_{\text{max}}\right]\right)
\]

This captures both the average neighborhood signal and outlier/dominant neighbor features, providing a richer representation at the cost of a 3x wider intermediate dimension.

#### 6.1.3 Residual Connection

When `use_residual=True`, the block adds a skip connection:

\[
\mathbf{H}^{\text{out}} = \sigma\!\left(\mathbf{H}^{(L)} + \mathbf{W}_{\text{proj}} \mathbf{H}^{(0)}\right)
\]

The projection matrix \(\mathbf{W}_{\text{proj}}\) aligns the input dimension to the GNN output dimension.

#### Design Rationale

GraphSAGE is chosen over spectral methods (e.g., ChebNet, GCN) because:
1. **Inductive**: Works on unseen graph topologies at inference (new trades modify the graph).
2. **Scalable**: Sparse aggregation is O(T x k) per layer.
3. **Flexible**: Supports multiple aggregators without eigendecomposition.

---

### 6.2 RnnBlock — Temporal Embedding

**File**: `layers/rnn_layers.py`
**Input**: `pnl_history [B, S, T_e]`
**Output**: `[B, d_r]`

#### Purpose

Compresses the historical P&L scenario window into a fixed-length temporal embedding per sample. This captures autocorrelation, volatility clustering, regime dynamics, and cross-trade temporal co-movements.

#### Architecture

A stack of L recurrent layers (LSTM, BiLSTM, or GRU) arranged as a `tf.keras.Sequential`. All intermediate layers return sequences; the final layer returns only the last hidden state.

#### 6.2.1 LSTM Mathematics (Default)

For input \(\mathbf{x}_t \in \mathbb{R}^{T_e}\) at time step t:

\[
\begin{aligned}
\mathbf{f}_t &= \sigma\!\left(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f\right) & \text{(forget gate)} \\
\mathbf{i}_t &= \sigma\!\left(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i\right) & \text{(input gate)} \\
\tilde{\mathbf{c}}_t &= \tanh\!\left(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c\right) & \text{(candidate cell)} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t & \text{(cell state)} \\
\mathbf{o}_t &= \sigma\!\left(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o\right) & \text{(output gate)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) & \text{(hidden state)}
\end{aligned}
\]

The final hidden state \(\mathbf{h}_S \in \mathbb{R}^{d_r}\) becomes the temporal embedding for the sample.

#### Design Rationale

- **LSTM over Transformer**: For the typical sequence lengths in scenario analysis (S = 1 to 252 business days), LSTM's inductive bias toward sequential processing and its gated memory are well-suited. The O(S) sequential cost is acceptable and provides stronger extrapolation to unseen sequence lengths than position-encoded self-attention.
- **Bidirectional option**: BiLSTM captures both past-to-future and future-to-past temporal context, useful when the full scenario window is available (not for online/streaming inference).
- **Dropout**: Applied at the recurrent level to regularize temporal feature extraction.

---

### 6.3 FusionLayer — Cross-Modal Attention with Gating

**File**: `layers/fusion_layer.py`
**Input**: `(gnn_features [T, d_g], rnn_features [B, d_r], adjacency [T, T])`
**Output**: `[B, T, d_f]`

#### Purpose

Merges the structural (GNN) and temporal (RNN) information streams into a single per-trade, per-scenario representation. This is the critical junction where "what a trade is" meets "what the market has done."

#### Architecture

1. **Broadcast and project**: GNN features (shared across the batch) are broadcast to `[B, T, d_g]` and projected to d_f. RNN features (shared across trades) are broadcast to `[B, T, d_r]` and projected to d_f.

2. **Joint query formation**: The query is formed as a sum of two projections, encoding both the temporal context and the structural identity of each trade:

\[
\mathbf{Q} = \mathbf{W}_Q^{\text{rnn}} \, \mathbf{E}^{\text{rnn}} + \mathbf{W}_Q^{\text{gnn}} \, \mathbf{E}^{\text{gnn}} \in \mathbb{R}^{B \times T \times d_f}
\]

3. **Keys and values** are derived from the GNN embedding alone (the structural signal):

\[
\mathbf{K} = \mathbf{W}_K \, \mathbf{E}^{\text{gnn}}, \quad \mathbf{V} = \mathbf{W}_V \, \mathbf{E}^{\text{gnn}}
\]

4. **Sparse neighborhood attention**: Rather than computing the full \(T \times T\) attention matrix, each trade attends only to its k neighbors from the adjacency graph. For trade i with neighbor set \(\mathcal{N}(i)\):

\[
\alpha_{ij} = \frac{\exp\!\left(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_h}\right)}{\sum_{j' \in \mathcal{N}(i)} \exp\!\left(\mathbf{q}_i^\top \mathbf{k}_{j'} / \sqrt{d_h}\right)}, \quad j \in \mathcal{N}(i)
\]

\[
\mathbf{c}_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \, \mathbf{v}_j
\]

Memory: O(B x h x T x k) instead of O(B x h x T^2). For T=10,000, k=50: ~32 MB vs 6.4 GB.

5. **Multi-head attention**: Q, K, V are split into h heads of dimension d_h = d_f / h, processed independently, then concatenated:

\[
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}_O
\]

6. **Gated fusion** (default mode `gate`): A learned sigmoid gate controls the mixing ratio between the attended structural signal and the temporal signal:

\[
g = \sigma\!\left(\mathbf{W}_g [\mathbf{c} \;\|\; \mathbf{e}^{\text{rnn}}] + \mathbf{b}_g\right) \in [0, 1]
\]

\[
\mathbf{f} = g \odot \mathbf{c} + (1 - g) \odot \mathbf{e}^{\text{rnn}}
\]

The gate is initialized near 0.5, allowing the model to learn during training whether to rely more on the structural or temporal signal for each trade.

#### Design Rationale

- **Cross-attention (not self-attention)**: The RNN stream provides the "question" (what temporal regime are we in?), and the GNN stream provides the "answer" (which structural relationships matter in this regime?). This asymmetry is by design.
- **Sparse masking**: The k-NN graph constrains each trade to attend only to structurally similar instruments, preventing the attention weights from diluting across unrelated trades. This is both a regularization mechanism and a scalability enabler.
- **Gating over addition**: The sigmoid gate provides a per-element, learned interpolation that can suppress uninformative structural signals for trades where the temporal signal alone is sufficient, and vice versa.

---

### 6.4 TargetAttentionLayer — Inter-Trade Self-Attention

**File**: `layers/attention_layer.py`
**Input**: `(fused_features [B, T, d_f], adjacency [T, T], target_indices [n_tgt])`
**Output**: `[B, n_tgt, d_a]`

#### Purpose

Refines the fused representations of the target trades by allowing them to attend to each other, capturing inter-target dependencies (e.g., two target trades on the same underlying with different strikes should have correlated P&L).

#### Architecture

1. **Target extraction**: Fused features are gathered to `[B, n_tgt, d_f]` using `target_indices`. The adjacency submatrix is extracted directly from the sparse structure (O(nnz) scan) to produce a small `[n_tgt, n_tgt]` dense binary mask.

2. **Transformer-style self-attention block**:
   - Linear projection to d_a
   - Multi-head self-attention with adjacency masking
   - Residual connection + LayerNorm
   - Position-wise feed-forward network (expansion factor 4x)
   - Residual connection + LayerNorm

\[
\mathbf{Z} = \text{LayerNorm}\!\left(\mathbf{F}_{\text{tgt}} + \text{MHA}(\mathbf{F}_{\text{tgt}}, \mathbf{F}_{\text{tgt}}, \mathbf{F}_{\text{tgt}}; \mathbf{M}_{\text{tgt}})\right)
\]
\[
\mathbf{O} = \text{LayerNorm}\!\left(\mathbf{Z} + \text{FFN}(\mathbf{Z})\right)
\]

Where \(\mathbf{M}_{\text{tgt}}\) is the binary target submatrix from the adjacency graph.

#### Sparse Submatrix Extraction

The `_extract_target_submatrix` method avoids materializing the full `[T, T]` dense matrix:

1. Build a lookup table: global trade id -> local target id (or -1).
2. For every sparse edge (i, j), look up both endpoints.
3. Keep only edges where both i and j are targets.
4. Remap to local coordinates `[0, n_tgt)`.
5. Build a tiny `[n_tgt, n_tgt]` SparseTensor, then convert to dense.

Cost: O(nnz) + O(n_tgt^2), trivial compared to O(T^2).

#### Design Rationale

- **Self-attention (not cross-attention)**: Unlike the fusion layer, targets attend to each other to capture co-dependencies. A butterfly spread's long and short legs, for example, should be processed jointly.
- **Adjacency masking**: Prevents attention leakage between structurally unrelated targets.
- **FFN sublayer**: Adds per-trade non-linearity for richer transformations.
- **Small matrix**: Since n_tgt << T, this layer's O(n_tgt^2) attention is cheap even for large portfolios.

---

### 6.5 TargetPnlOutput — Dynamic Projection with kNN Transfer

**File**: `layers/projection_layer.py`
**Input**: `(trade_features [T, p], attended_features [B, n_tgt, d_a], target_indices [n_tgt])`
**Output**: `[B, n_tgt]` (predicted P&L per target trade)

#### Purpose

Projects the attended feature representation to a scalar P&L prediction for each target trade. Critically, this layer separates **trained targets** (seen during calibration) from **new targets** (unseen at training time), using different strategies for each.

#### Architecture

The output for each target is composed of two additive terms:

\[
\hat{y}_i = \underbrace{\text{baseline}_i}_{\text{amplitude}} + \underbrace{\text{residual}_i}_{\text{correction}}
\]

##### Baseline (Trained Targets)

Each trained target has a dedicated learned kernel \(\mathbf{w}_i \in \mathbb{R}^{d_a}\) and bias \(b_i\):

\[
\text{baseline}_i^{\text{train}} = g_i \cdot \frac{\mathbf{w}_i^\top}{\|\mathbf{w}_i\|} \mathbf{a}_i + b_i
\]

Where \(g_i = \text{softplus}(\tilde{g}_i)\) is a learned positive gain per target. The weight normalization separates the direction of the kernel (learned shape) from its magnitude (learned amplitude), stabilizing training.

##### Baseline (New Targets — kNN Output-Space Mixing)

New targets inherit their baseline from the k nearest trained targets in attribute space:

\[
\text{baseline}_i^{\text{new}} = \sum_{j \in \text{kNN}(i)} w_{ij} \cdot \text{baseline}_j^{\text{train}}
\]

Weights \(w_{ij}\) are computed via cosine similarity with temperature-scaled softmax:

\[
w_{ij} = \frac{\exp(\tau \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}{\sum_{j'} \exp(\tau \cdot \cos(\mathbf{x}_i, \mathbf{x}_{j'}))}
\]

Or inverse distance weighting (IDW):

\[
w_{ij} = \frac{\|\mathbf{x}_i - \mathbf{x}_j\|^{-p}}{\sum_{j'} \|\mathbf{x}_i - \mathbf{x}_{j'}\|^{-p}}
\]

This is performed in **output space** (on the scalar P&L predictions), not in embedding space, ensuring the magnitude of the transferred baseline is consistent with the learned scale.

##### Residual (All Targets)

A shared 2-layer MLP processes the concatenation of attended features and trade attributes:

\[
\text{residual}_i = \mathbf{w}_2^\top \, \sigma\!\left(\mathbf{W}_1 [\mathbf{a}_i \;\|\; \mathbf{x}_i] + \mathbf{b}_1\right)
\]

The residual captures target-specific corrections that the baseline cannot represent.

##### Optional Attention-Conditioned Modulation

For new targets only, optional learned scale and bias heads can post-multiply/add to the prediction, conditioned on the attention features:

\[
\hat{y}_i^{\text{new}} = \text{softplus}(\mathbf{w}_s^\top \mathbf{a}_i) \cdot \hat{y}_i + \mathbf{w}_b^\top \mathbf{a}_i
\]

#### Design Rationale

- **Dual baseline + residual**: Separating the learned amplitude from the correction term prevents the MLP residual from dominating and destabilizing training. The baseline provides a stable, interpretable linear component.
- **kNN in output space**: Performing interpolation on scalar predictions (not embeddings) avoids the curse of dimensionality and ensures the transferred signal has physically meaningful units (P&L).
- **Weight normalization**: Decoupling direction and magnitude in the baseline kernel improves optimization stability, especially for trades with different P&L scales.
- **Damped residual for new trades**: The `residual_new_damp` parameter (default 1.0) allows suppressing the MLP correction for unseen trades where the residual has not been calibrated.

---

## 7. Sparse Tensor Design and Scalability

### 7.1 Adjacency Representation

The trade adjacency matrix is stored as a `tf.SparseTensor` with three component tensors:

| Component | Shape | Dtype | Description |
|---|---|---|---|
| `adjacency_indices` | `[nnz, 2]` | int64 | Row-column pairs of non-zero entries |
| `adjacency_values` | `[nnz]` | float32 | Edge weights (row-normalized) |
| `adjacency_dense_shape` | `[2]` | int64 | `[T, T]` |

For a k-NN graph: nnz = T x k. At T=10,000 and k=50, storage is 500,000 entries (~6 MB) vs 100M entries (~400 MB) for the dense matrix.

### 7.2 SparseTensor in `tf.data.Dataset`

SparseTensors cannot be directly serialized in `tf.data.Dataset` pipelines. The three component tensors are passed as separate dense tensors and reconstructed inside the model's `call()` method:

```python
adjacency = tf.sparse.reorder(tf.SparseTensor(
    indices=inputs["adjacency_indices"],
    values=inputs["adjacency_values"],
    dense_shape=inputs["adjacency_dense_shape"],
))
```

`tf.sparse.reorder()` ensures row-major canonical ordering, required by downstream sparse operations.

### 7.3 Memory Budget Comparison (T = 10,000, k = 50, B = 16, h = 1, d_h = 64)

| Component | Dense O(T^2) | Sparse O(T x k) | Ratio |
|---|---|---|---|
| Adjacency storage | 400 MB | 6 MB | 67x |
| Fusion attention scores | 6.4 GB | 32 MB | 200x |
| Fusion attention weights | 6.4 GB | 32 MB | 200x |
| Target attention (n_tgt=50) | 10 KB | 10 KB | 1x |
| GNN aggregation | N/A (always sparse) | 25 MB | — |

---

## 8. Loss Function and Training Objective

### 8.1 Primary Loss

Mean Squared Error (MSE) over target trade P&L predictions in standardised (z-score) space:

\[
\mathcal{L} = \frac{1}{B \cdot n_{\text{tgt}}} \sum_{b=1}^{B} \sum_{i=1}^{n_{\text{tgt}}} \left(\hat{y}_{b,i} - y_{b,i}\right)^2
\]

Where \(\hat{y}_{b,i}\) is the model prediction and \(y_{b,i}\) is the true standardised P&L for target trade i in scenario b.

### 8.2 Why z-Space

Training in standardised P&L space (zero mean, unit variance per trade) ensures that:
1. All target trades contribute equally to the loss regardless of their natural P&L scale.
2. The learning rate is effective across trades with different notional values.
3. Predictions are inverse-transformed to P&L space for evaluation and reporting.

### 8.3 Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| RMSE | \(\sqrt{\text{MSE}}\) | Average prediction error magnitude |
| MAE | \(\frac{1}{n}\sum\|y - \hat{y}\|\) | Median-robust error magnitude |
| R-squared | \(1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}\) | Fraction of variance explained |
| MAPE | \(\frac{100}{n}\sum\|\frac{y - \hat{y}}{y}\|\) | Percentage error (sensitive to near-zero P&L) |
| Residual P95/P99 | Percentiles of \(\|y - \hat{y}\|\) | Tail risk of prediction error |

---

## 9. Inference and Generalisation to New Trades

### 9.1 Inference Pipeline

At inference, the pipeline:
1. Extends the trade attribute matrix with new target trades.
2. Re-encodes attributes using the fitted `TradeAttributeEncoder`.
3. Rebuilds the k-NN graph to include edges to/from new trades.
4. Passes the extended inputs through the model's `call()` method.

### 9.2 New Trade Generalisation

The model generalises to unseen trades through three mechanisms:

1. **GNN (inductive)**: GraphSAGE aggregates neighbor features using learned weight matrices that are independent of the specific graph topology. A new trade's embedding is computed by aggregating its k nearest neighbors' features — no retraining required.

2. **Fusion + Attention**: The attention layers process arbitrary-length trade sequences; new trades participate in the attention mechanism via their position in the extended graph.

3. **Projection (kNN transfer)**: The `TargetPnlOutput` layer identifies that the new trade has no calibrated baseline kernel and falls back to kNN output-space mixing from the k nearest trained targets.

### 9.3 Cold-Start Behavior

For a completely novel trade (no similar trades in the training set):
- GNN embedding is a zero-information aggregation (distant neighbors).
- kNN weights are approximately uniform over the k nearest (dissimilar) trains.
- The model's prediction will be a weighted average of the k most similar trained targets' P&L, plus a residual correction.

This provides a conservative, regression-to-the-mean estimate that degrades gracefully rather than failing.

---

## 10. Hyperparameter Reference

### 10.1 GNN Block

| Parameter | Default | Description |
|---|---|---|
| `layers` | 2 | Number of stacked GNN sub-layers |
| `layer_type` | `mixed_graph_sage` | Sub-layer class (`graph_sage` or `mixed_graph_sage`) |
| `units` | 128 | Output dimension per sub-layer |
| `activation` | `relu` | Non-linearity applied at block output |
| `dropout_rate` | 0.1 | Dropout between sub-layers |
| `use_residual` | True | Skip connection from block input to output |
| `batch_norm` | True | LayerNorm between sub-layers |
| `aggregator_op` | `mean` | Neighbor aggregation function (`mean` or `max`) |

### 10.2 RNN Block

| Parameter | Default | Description |
|---|---|---|
| `layers` | 2 | Number of stacked RNN layers |
| `layer_type` | `lstm` | RNN cell type (`lstm`, `bilstm`, `gru`) |
| `units` | 128 | Hidden state dimension |
| `activation` | `relu` | Cell activation |
| `recurrent_activation` | `sigmoid` | Gate activation |
| `dropout_rate` | 0.1 | Recurrent dropout rate |

### 10.3 Fusion Layer

| Parameter | Default | Description |
|---|---|---|
| `units` | 64 | Attention/projection dimension |
| `num_heads` | 1 | Number of attention heads |
| `k_nbrs` | 50 | Max neighbors per trade in sparse attention |
| `fusion_mode` | `gate` | Mixing strategy (`gate` or `add`) |
| `dropout_rate` | 0.1 | Attention dropout |

### 10.4 Target Attention Layer

| Parameter | Default | Description |
|---|---|---|
| `units` | 32 | Attention dimension |
| `num_heads` | 1 | Number of attention heads |
| `dropout_rate` | 0.1 | Attention and FFN dropout |
| `k_nbrs` | 50 | Config-carried (not used directly; submatrix is small) |

### 10.5 Projection Layer

| Parameter | Default | Description |
|---|---|---|
| `units` | 32 | Residual MLP hidden dimension |
| `activation` | `gelu` | Residual MLP activation |
| `baseline_new_mode` | `output_mix` | Strategy for new target baselines |
| `use_baseline_weight_norm` | True | Separate kernel direction from gain |
| `knn_k` | 5 | Number of nearest trained targets for transfer |
| `knn_mode` | `cosine_softmax` | kNN weight scheme (`cosine_softmax` or `idw`) |
| `knn_temperature` | 5.0 | Softmax temperature for cosine kNN |
| `knn_power` | 2.0 | IDW distance exponent |
| `residual_new_damp` | 1.0 | Damping factor for new-trade residuals |

---

## 11. Numerical Stability and Reproducibility

### 11.1 Seed Control

Full deterministic execution is achieved via:

```python
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(seed)
tf.keras.utils.set_random_seed(seed)           # seeds Python, NumPy, TF
tf.config.experimental.enable_op_determinism()  # forces deterministic kernels
```

The `tf.data.Dataset.shuffle()` call also receives an explicit seed.

### 11.2 Numerical Guards

- **Sparse reorder**: `tf.sparse.reorder()` is applied once at model entry to ensure row-major index ordering, required by `tf.sparse.to_dense` and segment operations.
- **Masked softmax**: Large negative sentinel (-1e9) drives masked positions to ~0 after softmax without requiring post-hoc masking.
- **Softplus for gains**: Baseline gain parameters use softplus to ensure strict positivity.
- **L2 normalization epsilon**: All divisions in kNN weight computation include epsilon (1e-8) to prevent division by zero.
- **Debug numerics**: `tf.debugging.check_numerics` runs on the RNN output during training only, catching NaN/Inf from exploding gradients.

---

## 12. Computational Complexity

### 12.1 Per-Layer Complexity

| Layer | Time | Memory | Dominant Term |
|---|---|---|---|
| GnnBlock (L layers) | O(L x nnz x d_g) | O(T x d_g) | Sparse matmul |
| RnnBlock (L layers) | O(L x S x B x d_r^2) | O(B x d_r) | LSTM gates |
| FusionLayer (sparse) | O(B x h x T x k x d_h) | O(B x h x T x k) | Neighbor gather + scores |
| FusionLayer (dense fallback) | O(B x h x T^2 x d_h) | O(B x h x T^2) | Full attention |
| TargetAttentionLayer | O(B x h x n_tgt^2 x d_a) | O(B x h x n_tgt^2) | Full attention (n_tgt << T) |
| TargetPnlOutput | O(B x n_tgt x d_a) | O(n_tgt x d_a) | Baseline + residual |
| **Total (sparse path)** | **O(B x T x k x d)** | **O(B x T x k)** | **FusionLayer dominates** |

### 12.2 Scaling Characteristics

| Portfolio Size (T) | k | nnz | Fusion Attention Memory | Training Time (relative) |
|---|---|---|---|---|
| 100 | 10 | 1,000 | ~100 KB | 1x |
| 1,000 | 30 | 30,000 | ~4 MB | ~10x |
| 10,000 | 50 | 500,000 | ~32 MB | ~100x |
| 50,000 | 50 | 2,500,000 | ~160 MB | ~500x |

The dominant cost scales **linearly** with T (at fixed k), not quadratically.

---

## 13. Limitations and Future Work

### 13.1 Current Limitations

1. **Static graph within a training run**: The k-NN graph is built once during data preparation. Trades that become more/less similar during different market regimes are not dynamically re-connected.
2. **Homogeneous aggregation**: All GNN layers use the same aggregation scheme. Heterogeneous aggregation (different schemes for different edge types) could capture richer structural patterns.
3. **Single-step P&L**: The current setup predicts P&L at a single horizon. Multi-step forecasting would require autoregressive decoding or a sequence-to-sequence head.
4. **No uncertainty quantification**: Point predictions only. Conformal prediction or distributional outputs (e.g., mixture density networks) would provide prediction intervals.

### 13.2 Future Directions

- **Dynamic graph attention**: Re-compute k-NN edges per scenario based on market-conditioned attributes.
- **Temporal attention**: Replace or augment the LSTM with a lightweight temporal attention mechanism for long-horizon lookback.
- **Distributional output**: Replace MSE loss with a distributional loss (e.g., quantile regression, CRPS) for risk-aware predictions.
- **Hierarchical graph**: Multi-level graph (trade -> desk -> portfolio -> entity) for enterprise-wide P&L simulation.
- **Explanability**: Attention weight extraction for trade-level feature attribution (which neighbors most influenced a target's prediction?).

---

*Document version: 1.0 — Generated 2026-02-22*
