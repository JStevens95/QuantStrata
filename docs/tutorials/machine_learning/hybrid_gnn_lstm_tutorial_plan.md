# Hybrid GNN–LSTM Tutorial Notebook — Plan

**Objective:** A showcase tutorial for the end-to-end hybrid GNN–LSTM (GNN–RNN) model, structured and formatted like front-office ML quant documentation: clear process descriptions, rigorous model and mathematical (PhD-level) description, and strong visualisation.

**Target file:** `docs/tutorials/machine_learning/hybrid_gnn_lstm_tutorial.ipynb`

**Reference structure:** `docs/tutorials/machine_learning/ml_model_lifecycle.ipynb` (Section 0–9 flow, config → data → model → training → evaluation → tuning → deployment → monitoring → advanced).

---

## 0. One-Page Notebook Outline

| Section | Title | Main content |
|--------|--------|--------------|
| 0 | Architecture Overview | Problem, package layout, high-level flow diagram |
| 1 | Configuration and Reproducibility | `default_hybrid_model_config`, `TrainingConfiguration`, paths |
| 2 | Data Engineering | `build_gnn_data`, GnnDataResult, synthetic/FX, shapes, graph + PnL viz |
| 3 | Model Architecture (Math) | GNN (GraphSAGE), RNN (LSTM), Fusion (cross-attn + gate), Target attention, Projection (baseline + k-NN) |
| 4 | Training Pipeline | TrainingManager, stages, callbacks, loss curves |
| 5 | Evaluation and Benchmarking | Metrics (MSE, MAE, R², etc.), pred vs actual, residuals, hybrid vs rnn_only |
| 6 | Hyperparameter Sensitivity | Optional: validation loss vs gnn_units / rnn_units / dropout |
| 7 | Model Deployment | Save/load weights, inference contract |
| 8 | Production Monitoring | Optional: latency, throughput, drift |
| 9 | Advanced Topics | New-trade k-NN, FX data, sparse adjacency, architecture variants |

---

## 1. Notebook-Level Structure

- **Title:** *Hybrid GNN–LSTM Model: Portfolio PnL Simulation*
- **Subtitle:** End-to-end process from graph construction and temporal data to trained model and evaluation.
- **Sections:** Numbered 0–9, with subsections (###) where needed. Each section has:
  - **Process description** (what and why)
  - **Mathematical / conceptual detail** where relevant (PhD-level)
  - **Code** (minimal, runnable, using `src.machine_learning`)
  - **Visualisations** (architecture, data, training, evaluation)

**Output standardisation (as in lifecycle notebook):** Define at top a single `NOTEBOOK_SUBFOLDER` (e.g. `hybrid_gnn_lstm`) and derive `CHECKPOINTS_DIR`, `LOGS_DIR`, `MODEL_ARTIFACTS_DIR` so all checkpoints, TensorBoard/CSV logs, and saved models go under that subfolder.

---

## 2. Section-by-Section Plan

### Section 0: Architecture Overview

- **Purpose:** Where the hybrid GNN–LSTM sits in the library and what problem it solves.
- **Process:**
  - **Problem:** Portfolio-level PnL simulation with (i) trade-level structure and relationships (graph), (ii) temporal PnL evolution (elementary trades over time), (iii) target trades whose PnL we predict. Need to generalise to new trades and new scenarios.
  - **Package layout:** `machine_learning/models/gnn_rnn_hybrid/` (model, config, layers), `machine_learning/data/gnn_rnn_hybrid/` (build, synthetic, dataset_utils, portfolio_builder), `machine_learning/calibration/training_manager.py`.
- **Math (optional here):** One equation for the overall pipeline:  
  **Input:** trade features \(\mathbf{X} \in \mathbb{R}^{T \times P}\), adjacency \(\mathbf{A}\), PnL history \(\mathbf{H} \in \mathbb{R}^{B \times S \times N_e}\), target indices.  
  **Output:** \(\hat{\mathbf{y}} \in \mathbb{R}^{B \times N_{\mathrm{tgt}}}\) (PnL prediction per target per sample).
- **Visualisation:** High-level flow diagram (text/diagram): Graph + PnL history → GNN block → RNN block → Fusion → Target attention → PnL projection → Predictions. Same style as lifecycle “ML Model Lifecycle Flow Diagram”.
- **Code:** Imports (TensorFlow, NumPy, matplotlib, pathlib), path setup, optional `sys.path.insert`, and listing of key modules/classes (`HybridGnnRnn`, `build_gnn_data`, `TrainingManager`, `default_hybrid_model_config`).

---

### Section 1: Configuration and Reproducibility

- **Purpose:** Single place to define model and training config so runs are reproducible.
- **Process:**
  - Use `default_hybrid_model_config()` for model (GNN, RNN, fusion, attention, projection).
  - Use `TrainingConfiguration` (training_manager) for training (epochs, batch_size, learning_rate, callbacks, paths).
  - Document meaning of key keys: `general.architecture` (`default` vs `rnn_only`), GNN layer type (`graph_sage`, `mixed_graph_sage`), RNN layer type (`lstm`, `bilstm`, `gru`), fusion mode (`gate`, `add`), projection options (`baseline_new_mode`, `knn_*`).
- **Math:** Not required; keep as “configuration schema” narrative.
- **Visualisation:** Optional: small table or tree of config sections (general, gnn_model, rnn_model, fusion_model, attention_model, projection_model).
- **Code:** Build `model_config` with `default_hybrid_model_config(...)`, build `TrainingConfiguration` with `model_dir=CHECKPOINTS_DIR` (or similar), print or display key values.

---

### Section 2: Data Engineering for the Hybrid Model

- **Purpose:** Contract of inputs/outputs and how train/val/projection splits are produced.
- **Process:**
  - **Contract:** `build_gnn_data()` returns `GnnDataResult(train_ds, val_ds, proj_ds, metadata)`. Each dataset yields `(inputs_dict, targets)` where `inputs_dict` contains `trade_features`, `pnl_history`, `adjacency_matrix`, `elementary_indices`, `target_indices`.
  - **Synthetic path:** `generate_synthetic_gnn_data()` → trade features (e.g. moneyness, TTM, delta, vega, product type), k-NN adjacency from features, elementary/target index split, PnL history (e.g. random walk), targets (e.g. linear combination of final elementary PnL + noise).
  - **FX path (optional mention):** `build_fx_gnn_data()` for FX portfolio (vanilla, digital, barrier, etc.) when `use_synthetic=False`.
- **Math (PhD-level):**
  - **Trade features:** \(\mathbf{X} \in \mathbb{R}^{T \times P}\) (e.g. \(P=7\): moneyness, TTM, delta, vega, one-hot product).
  - **Adjacency:** \(\mathbf{A} \in \mathbb{R}^{T \times T}\), row-normalised from k-NN graph: \(A_{ij} = \mathbb{1}_{j \in \mathcal{N}_k(i)} / |\mathcal{N}_k(i)|\) (or with self-loops).
  - **PnL history:** \(\mathbf{H} \in \mathbb{R}^{N \times S \times N_e}\), \(N\) samples, \(S\) timesteps, \(N_e\) elementary trades.
  - **Targets:** \(\mathbf{Y} \in \mathbb{R}^{N \times N_{\mathrm{tgt}}}\); synthetic: \(\mathbf{Y} = \mathbf{H}_{:,-1,:} \mathbf{W} + \boldsymbol{\epsilon}\), \(\mathbf{W} \in \mathbb{R}^{N_e \times N_{\mathrm{tgt}}}\), \(\boldsymbol{\epsilon}\) noise.
- **Visualisation:**
  - **Graph:** Plot adjacency (or small subgraph) as network: nodes = trades, edges = k-NN; colour by product type or delta.
  - **Data shapes:** Bar or table of shapes (n_trades, n_features, n_elementary, n_targets, n_samples, n_timesteps).
  - **Sample PnL paths:** 2–3 samples of PnL history (e.g. one row per elementary, time on x-axis).
  - **Target vs elementary:** Scatter or histogram of target PnL vs sum of elementary (or first principal component).
- **Code:** Call `build_gnn_data(use_synthetic=True, ...)`, inspect `metadata`, take one batch from `train_ds`, show `train_ds.element_spec` and shapes.

---

### Section 3: Model Architecture — Mathematical Description

- **Purpose:** PhD-level formulation of each block (GNN, RNN, Fusion, Attention, Projection) and how they connect.
- **Process:** Walk through the forward pass in order; for each layer give equation and short intuition.
- **Math:**

  1. **GNN Block (GraphSAGE-style)**  
     - Aggregation: \(\mathbf{h}_{\mathcal{N}(v)} = \mathrm{Agg}(\{\mathbf{x}_u : u \in \mathcal{N}(v)\})\) (mean or max).  
     - Update: \(\mathbf{h}_v^{(l+1)} = \sigma\big(\mathbf{W}_{\mathrm{self}}^{(l)} \mathbf{h}_v^{(l)} + \mathbf{W}_{\mathrm{neigh}}^{(l)} \mathbf{h}_{\mathcal{N}(v)}^{(l)}\big)\).  
     - With residual: \(\mathbf{h}_v^{(l+1)} \leftarrow \mathbf{h}_v^{(l+1)} + \mathbf{P} \mathbf{x}_v\) (input projection \(\mathbf{P}\)).  
     - Output: \(\mathbf{H}_{\mathrm{gnn}} \in \mathbb{R}^{T \times d_g}\).

  2. **RNN Block (LSTM)**  
     - Standard LSTM equations (input, forget, cell, output gates; optional reference to Hochreiter & Schmidhuber).  
     - Input: \(\mathbf{H}_{\mathrm{pnl}} \in \mathbb{R}^{B \times S \times N_e}\); output: \(\mathbf{h}_{\mathrm{rnn}} \in \mathbb{R}^{B \times d_r}\).

  3. **Fusion Layer (cross-attention + gating)**  
     - Broadcast: \(\mathbf{H}_{\mathrm{gnn}} \to [B,T,d_g]\), \(\mathbf{h}_{\mathrm{rnn}} \to [B,T,d_r]\) (after projection to \(d_f\)).  
     - Query from RNN+GNN, Key/Value from GNN: \(\mathbf{Q} = \mathbf{W}_q^r \mathbf{R} + \mathbf{W}_q^g \mathbf{G}\), \(\mathbf{K}=\mathbf{W}_k \mathbf{G}\), \(\mathbf{V}=\mathbf{W}_v \mathbf{G}\).  
     - Scaled dot-product attention with **adjacency mask**: \(\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V};\mathbf{A}) = \mathrm{softmax}\big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}} + (1-\mathbf{A}) \cdot (-\infty)\big)\mathbf{V}\).  
     - Gating: \(\mathbf{F} = \sigma(\mathbf{W}_g [\mathbf{F}_{\mathrm{attn}}; \mathbf{R}])\), output \(\mathbf{F} \odot \mathbf{F}_{\mathrm{attn}} + (1-\mathbf{F})\odot \mathbf{R}\).  
     - Output: \(\mathbf{F}_{\mathrm{fused}} \in \mathbb{R}^{B \times T \times d_f}\).

  4. **Target Attention (self-attention over targets)**  
     - Restrict to target indices: \(\mathbf{F}_{\mathrm{tgt}} = \mathbf{F}_{\mathrm{fused}}[:, \mathcal{I}_{\mathrm{tgt}}, :]\), adjacency to \(\mathbf{A}_{\mathrm{tgt}}\).  
     - Multi-head self-attention with same adjacency masking; then FFN + residual + LayerNorm (Transformer-style).  
     - Output: \(\mathbf{Z} \in \mathbb{R}^{B \times N_{\mathrm{tgt}} \times d_a}\).

  5. **Projection (TargetPnlOutput)**  
     - **Baseline (train targets):** \(b_i = \langle \mathbf{z}_i, \mathbf{k}_i \rangle + c_i\) (per-target kernel \(\mathbf{k}_i\), bias \(c_i\); optional unit-norm + gain).  
     - **Residual:** \(r_i = \mathrm{MLP}([\mathbf{z}_i; \mathbf{x}_i])\) (attributes \(\mathbf{x}_i\) for target \(i\)).  
     - **New targets (zero-shot):** baseline from k-NN in attribute space: \(b_j^{\mathrm{new}} = \sum_{i \in \mathcal{N}_k(j)} w_{ji} b_i\), \(w_{ji} = \mathrm{softmax}(\tau \cdot \mathrm{sim}(\mathbf{x}_j, \mathbf{x}_i))\) (cosine or IDW).  
     - Final: \(\hat{y}_i = b_i + r_i\) (train), \(\hat{y}_j = b_j^{\mathrm{new}} + \lambda r_j^{\mathrm{new}}\) (new).

- **Visualisation:**
  - **Block diagram:** One box per block with input/output shapes (e.g. [B,T,P] → GNN → [T,d_g]; [B,S,N_e] → RNN → [B,d_r]; then Fusion [B,T,d_f] → Attention [B,N_tgt,d_a] → Projection [B,N_tgt]).
  - **Attention mask:** Small heatmap of adjacency (or masked attention weights) for a toy graph.
- **Code:** Instantiate `HybridGnnRnn(model_config)`, build with one batch from `train_ds`, call `model(batch_inputs)` and show output shape; optionally `model.summary()` if available.

---

### Section 4: Training Pipeline

- **Purpose:** How to train with `TrainingManager`, including callbacks and logging.
- **Process:**
  - `TrainingManager(training_ds, model_config, validation_ds=val_ds, custom_callbacks=[...])`.
  - Build `TrainingConfiguration` (epochs, batch_size, learning_rate, model_dir, early_stopping, reduce_lr_on_plateau, etc.).
  - `manager.run(stages=[stage])` runs one or more stages; callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger) are built inside `build_callbacks(stage)` and extended with `custom_callbacks`.
  - Mention that this is the **model-specific** trainer for HybridGnnRnn (vs generic `Trainer` in lifecycle notebook).
- **Math:** Loss: typically MSE or MAE over \(\hat{\mathbf{Y}}\) vs \(\mathbf{Y}\) (optional: \(\mathcal{L} = \frac{1}{BN_{\mathrm{tgt}}}\sum_{b,i}(y_{bi} - \hat{y}_{bi})^2\)).
- **Visualisation:**
  - Training/validation loss (and metrics) over epochs; optional generalization gap (fill_between).
  - Optional: learning rate schedule over epochs if using ReduceLROnPlateau.
- **Code:** Create `TrainingManager`, define one or two stages, run training, store history; plot loss curves from `manager.training_history`.

---

### Section 5: Evaluation and Benchmarking

- **Purpose:** Assess performance on train/val/projection splits; compare with RNN-only baseline if desired.
- **Process:**
  - Evaluate on `train_ds`, `val_ds`, `proj_ds` (or test set): aggregate MSE, MAE, RMSE, R², MAPE.
  - Per-target or per-sample analysis optional.
  - Compare **default** (full hybrid) vs **rnn_only** (same data) to show benefit of GNN + fusion.
- **Math:** Standard regression metrics; R² = \(1 - \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}}\); residual normality (optional Q–Q).
- **Visualisation:**
  - **Metrics bar chart:** MSE, MAE, RMSE, R², MAPE for train/val/proj (and optionally rnn_only vs default).
  - **Predicted vs actual:** Scatter with regression line and equation (e.g. \(\hat{y} = \beta_1 x + \beta_0\), R²) for each split.
  - **Residuals:** Histogram, residual vs predicted, Q–Q plot (as in lifecycle notebook).
  - **Per-target error:** Box plot or bar of MAE/MSE per target index (optional).
- **Code:** Loop over splits, run `model.predict(ds)` (or batched loop), compute metrics, build plots.

---

### Section 6: Hyperparameter Sensitivity (Optional)

- **Purpose:** Show how validation loss or R² changes with key knobs.
- **Process:** Vary one or two parameters (e.g. gnn_units, rnn_units, fusion_units; or dropout, learning_rate) and record validation metric.
- **Visualisation:** Line plot or heatmap: e.g. validation loss vs gnn_units and rnn_units; or bar chart of best validation loss per config.
- **Code:** Small grid or list of configs, train short runs, collect validation loss, plot.

---

### Section 7: Model Deployment and Inference

- **Purpose:** Save/load model or weights; run inference on new batches.
- **Process:**
  - Save: `model.save_weights(...)` or `model.save(...)` under `MODEL_ARTIFACTS_DIR`.
  - Load: rebuild model with same `model_config`, then `model.load_weights(...)`.
  - Inference: same input dict format; batched or single sample.
- **Visualisation:** Optional: latency vs batch size (bar or line).
- **Code:** Save after training; reload in new “section”; run one batch and compare to pre-save prediction.

---

### Section 8: Production Monitoring (Optional)

- **Purpose:** Track performance over time (e.g. drift, latency).
- **Process:** Reuse ideas from lifecycle: latency (ms per batch), throughput; optional drift (e.g. mean prediction or z-score over time on a replay dataset).
- **Visualisation:** Latency and throughput by batch size; time series of mean prediction or z-score (simulated batches).
- **Code:** Time `model.predict()` for several batch sizes; optional loop over “days” of data and plot metric.

---

### Section 9: Advanced Topics and Extensions

- **Purpose:** Point to extensions and further reading.
- **Process:**
  - **New-trade generalisation:** Projection layer’s k-NN in attribute space for unseen target trades; mention `baseline_new_mode`, `knn_mode`, `knn_k`, `knn_temperature`.
  - **FX portfolio data:** `build_fx_gnn_data()` and `train_val_projection_split()` for real portfolio graphs.
  - **Sparse adjacency:** Model accepts `tf.SparseTensor` for large graphs; mention when to use.
  - **Architecture variants:** `rnn_only` baseline; MixedGraphSage vs GraphSage; BiLSTM/GRU; fusion `add` vs `gate`.
- **Math:** Optional: k-NN weight formula (cosine softmax, IDW) as in projection layer docstring.
- **Visualisation:** Optional: small schematic of “new target” baseline from k-NN of trained targets.
- **Code:** Optional: run with `architecture: "rnn_only"` and compare metrics; or build data with `use_synthetic=False` if FX path is available.

---

## 3. Source Files to Reference

| Component            | Path |
|---------------------|------|
| Model                | `src/machine_learning/models/gnn_rnn_hybrid/model.py` |
| Config               | `src/machine_learning/models/gnn_rnn_hybrid/config.py` |
| GNN layers           | `src/machine_learning/models/gnn_rnn_hybrid/layers/gnn_layers.py` |
| RNN layers           | `src/machine_learning/models/gnn_rnn_hybrid/layers/rnn_layers.py` |
| Fusion               | `src/machine_learning/models/gnn_rnn_hybrid/layers/fusion_layer.py` |
| Attention            | `src/machine_learning/models/gnn_rnn_hybrid/layers/attention_layer.py` |
| Projection           | `src/machine_learning/models/gnn_rnn_hybrid/layers/projection_layer.py` |
| Data build           | `src/machine_learning/data/gnn_rnn_hybrid/build.py` |
| Synthetic data       | `src/machine_learning/data/gnn_rnn_hybrid/synthetic.py` |
| Dataset utils        | `src/machine_learning/data/gnn_rnn_hybrid/dataset_utils.py` |
| Training manager     | `src/machine_learning/calibration/training_manager.py` |

---

## 4. Visualisation Checklist

- [ ] High-level architecture flow (Section 0).
- [ ] Config tree/table (Section 1, optional).
- [ ] Graph (adjacency) + data shapes + sample PnL paths + target vs elementary (Section 2).
- [ ] Block diagram with shapes + attention mask example (Section 3).
- [ ] Training/validation loss and metrics (Section 4).
- [ ] Metrics bar, predicted vs actual (with regression line), residuals (histogram, vs pred, Q–Q) (Section 5).
- [ ] Hyperparameter sensitivity plot (Section 6, optional).
- [ ] Latency/throughput (Section 7/8, optional).
- [ ] New-target k-NN schematic (Section 9, optional).

---

## 5. Tone and Formatting

- **Process:** Short paragraphs; bullet lists for contracts and steps.
- **Math:** LaTeX in markdown where supported (\(\mathbf{X}\), \(\mathbb{R}^{B \times T}\), etc.); key equations in display form.
- **Code:** Commented, minimal; favour library API over one-off scripts.
- **Figures:** Clear titles, axis labels, legends; consistent style (e.g. `seaborn-v0_8-whitegrid` as in lifecycle).

This plan is the single source of truth for implementing `hybrid_gnn_lstm_tutorial.ipynb` so the notebook reads like front-office ML quant documentation with strong structure, math, and visuals.
