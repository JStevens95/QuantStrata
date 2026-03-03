# rade_ml — Changes Summary & Production UI Architecture

> **Date:** 2026-02-22  
> **Purpose:** Summary of changes made to QuantStrata `rade_ml` for the new artifact structure, plus architecture notes for the production Dash UI. Review this first thing Monday and replicate on work env.

---

## 1. Source Changes Made

### 1.1 `src/rade_ml/pipelines/base.py`

| Change | Detail |
|--------|--------|
| `TrainPipeline.__init__` | Added `self._registered_entry: Optional[Any] = None` |
| `TrainPipeline.post_train` | After `registry.register()`, stores the `RegistryEntry` in `self._registered_entry` so subclasses can access the version directory |
| `_generate_training_report` | Report path changed from `artifacts_dir/training_reports/{run_name}_{timestamp}` → `artifacts_dir/training/{version}` (falls back to timestamp if no registry) |
| `EvalPipeline.__init__` | Added `self._loaded_entry: Optional[Any] = None` |
| `EvalPipeline.run` | Stores `self._loaded_entry = entry` after loading model; calls `get_target_scaler()` and passes it to `Evaluator.run()`; passes `data_result` to `post_eval()` |
| `EvalPipeline.post_eval` | Signature updated: added optional `data_result` kwarg |
| `EvalPipeline.get_target_scaler` | New hook (returns `None` by default); override in model pipelines |

### 1.2 `src/rade_ml/evaluation/evaluator.py`

| Change | Detail |
|--------|--------|
| `Evaluator.run()` | Added `target_scaler: Optional[Any] = None` parameter |
| Inverse transform | When `target_scaler` is provided, calls `self._inverse_transform()` before computing residuals so all downstream metrics are in original PnL units |
| `dataset_info` | Adds `"inverse_transformed": True` flag when scaler was applied |

### 1.3 `src/rade_ml/pipelines/hybrid_gnn_rnn/train.py`

| Change | Detail |
|--------|--------|
| `post_train` | After `super().post_train()` (which registers model), calls `self._save_inference_artifacts()` |
| `_save_inference_artifacts()` | **New method.** Saves into the registry version directory: `graph_builder.pkl`, `encoder.pkl`, `target_scaler.pkl`, `elementary_scaler.pkl`, `data_config.json`, `trade_universe.json` |

### 1.4 `src/rade_ml/pipelines/hybrid_gnn_rnn/eval.py`

| Change | Detail |
|--------|--------|
| `get_target_scaler` | Overrides base hook; returns `data_result.metadata["target_pnl_transformer"]` |
| `post_eval` | Calls `self._save_evaluation_data()` when `artifacts_dir` is set |
| `_save_evaluation_data()` | **New method.** Saves `eval_results.json`, `predictions.npz`, `residuals.npz`, `target_pnl.parquet`, `elementary_pnl.parquet` into `artifacts_dir/evaluation/{version}/` |

---

## 2. Resulting Folder Structure

After a full train → eval cycle, the output directory looks like:

```
rade_ml_output/
│
├── registry/                                    ← registry_dir
│   └── 20260222_143052_a1b2c3/                  # One version = one self-contained unit
│       ├── model.keras                           # Trained model weights
│       ├── metadata.json                         # RegistryEntry (metrics, config, tags)
│       ├── graph_builder.pkl                     # Fitted TradeGraphBuilder
│       ├── encoder.pkl                           # Fitted TradeAttributeEncoder
│       ├── target_scaler.pkl                     # Fitted StandardScaler (target PnL)
│       ├── elementary_scaler.pkl                 # Fitted StandardScaler (elementary PnL)
│       ├── data_config.json                      # HybridGnnRnnDataConfig snapshot
│       └── trade_universe.json                   # Trade IDs, indices, selected/removed
│
├── experiments/                                  ← tracking_dir
│   └── {run_id}/run.json
│
└── artifacts/                                    ← artifacts_dir
    ├── training/
    │   └── 20260222_143052_a1b2c3/               # Keyed by registry version
    │       ├── training_report.md
    │       ├── loss_curve.png
    │       └── trade_graph.png
    │
    └── evaluation/
        └── 20260222_143052_a1b2c3/               # Keyed by registry version
            ├── eval_results.json                  # Metrics, loss, dataset_info
            ├── predictions.npz                    # predictions + targets arrays
            ├── residuals.npz                      # residual arrays
            ├── target_pnl.parquet                 # Original target PnL DataFrame
            └── elementary_pnl.parquet             # Original elementary PnL DataFrame
```

**Key principle:** `registry/{version}/` is self-contained for cold-start inference. `artifacts/` holds analytics that the UI reads.

---

## 3. How Registry → Eval → UI Data Flow Works

### 3.1 Training Pipeline (produces registry + training artifacts)

```
PipelineConfig(registry_dir, artifacts_dir, ...)
        │
        ▼
  TrainPipeline.run()
        │
        ├── build_data()           → HybridGnnRnnResult
        ├── build_model()          → tf.keras.Model
        ├── Trainer.fit()          → TrainingResult
        ├── post_train()
        │     ├── registry.register(model)     → saves model.keras + metadata.json
        │     └── _save_inference_artifacts()   → saves scalers, encoder, graph builder
        │
        └── _generate_training_report()        → artifacts/training/{version}/
```

### 3.2 Evaluation Pipeline (loads from registry → produces analytics)

```
PipelineConfig(registry_dir, artifacts_dir, version_or_tag="latest")
        │
        ▼
  EvalPipeline.run()
        │
        ├── registry.load(version_or_tag)     → model, RegistryEntry
        ├── build_data()                       → HybridGnnRnnResult (with test_ds)
        ├── get_target_scaler()                → StandardScaler from data_result
        ├── Evaluator.run(test_ds, target_scaler)
        │     ├── model.evaluate()             → compiled metrics
        │     ├── _collect_predictions()        → preds, targets (scaled)
        │     ├── _inverse_transform()          → preds, targets (original units)
        │     └── _aggregate_stats()            → residual MAE, p95, etc.
        │
        └── post_eval()
              └── _save_evaluation_data()       → artifacts/evaluation/{version}/
```

### 3.3 What the UI Reads

The UI never re-runs training or evaluation. It reads pre-computed files:

| UI Need | Source File | Load Method |
|---------|-------------|-------------|
| Model metadata & metrics | `registry/{v}/metadata.json` | `RegistryEntry.from_json()` |
| Training loss curve | `artifacts/training/{v}/loss_curve.png` | Direct image load |
| Training report | `artifacts/training/{v}/training_report.md` | Read text |
| Evaluation metrics | `artifacts/evaluation/{v}/eval_results.json` | `EvaluationResult.from_json()` |
| Predictions vs actuals | `artifacts/evaluation/{v}/predictions.npz` | `np.load()` |
| Residual analysis | `artifacts/evaluation/{v}/residuals.npz` | `np.load()` |
| Target PnL history | `artifacts/evaluation/{v}/target_pnl.parquet` | `pd.read_parquet()` |
| Elementary PnL history | `artifacts/evaluation/{v}/elementary_pnl.parquet` | `pd.read_parquet()` |
| Trade universe | `registry/{v}/trade_universe.json` | `json.load()` |

---

## 4. Production UI Architecture

### 4.1 Layout: Three Pages, Each With Sub-Tabs

```
┌──────────────────────────────────────────────────────────────────────┐
│  RADE ML Dashboard                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │  Model Review     │  │  Live Inference   │  │  Risk & Analytics  │  │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

Three pages, each with internal sub-tabs for progressive depth. Summary cards sit at the top of every page (always visible regardless of sub-tab), with detailed analytics below in the active sub-tab. This lets a trader glance at the headline numbers and move on, while a quant drills into the specifics.

---

### 4.2 Page 1: Model Review (Quant Validation / Front Office)

**Primary users:** Quants, model validation, risk methodology  
**Purpose:** "Is this model good enough to use?"  
**Data source:** All read-only from `artifacts/` and `registry/` — zero model calls.

**Persistent header (always visible):**

```
┌─────────────────────────────────────────────────────────────────┐
│ Model Review                              [Version ▼] [Load]    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─── Model Summary Card ──────────────────────────────────┐    │
│  │ Version: 20260222_143052_a1b2c3                         │    │
│  │ Tags: [production] [gnn-rnn-v3]                         │    │
│  │ Best Val Loss: 0.00142  |  Epoch: 47/100                │    │
│  │ Training Time: 14m 32s  |  Parameters: 1.2M             │    │
│  │ Eval MAE: 0.0034 (original units)  |  P95: 0.0089       │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌───────────────┐   │
│  │ Overview │ │ Training │ │  Evaluation  │ │ Trade Universe│   │
│  └──────────┘ └──────────┘ └──────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Sub-Tab 1: Overview

Quick health-check landing tab. Green/amber/red status badges based on metric
thresholds. Key numbers from both training and evaluation at a glance. Rendered
markdown training report. This is what senior management sees when they open the
page — everything they need in 5 seconds.

| Component | Data Source |
|-----------|-------------|
| Status badges | `eval_results.json` metrics vs configured thresholds |
| Training report | `artifacts/training/{v}/training_report.md` |
| Config snapshot | `registry/{v}/metadata.json` |

#### Sub-Tab 2: Training

Deep dive into the training process.

| Component | Data Source |
|-----------|-------------|
| Loss curves (train vs val, interactive Plotly) | `artifacts/training/{v}/loss_curve.png` or loss data from `TrainingResult` |
| Learning rate schedule | `TrainingResult` history |
| Epoch-by-epoch metrics table | `TrainingResult` history |
| Convergence diagnostics | Derived from loss curve (gradient, plateau detection) |

#### Sub-Tab 3: Evaluation

**The core analytics tab.** This is where most of the model validation time is spent.

```
┌─── Evaluation Sub-Tab ──────────────────────────────────────┐
│                                                              │
│  ┌─── PnL Time Series ─────────────────────────────────┐    │
│  │ Predicted PnL vs Actual PnL (overlay line chart)     │    │
│  │ X-axis: scenario/time index                          │    │
│  │ Per-trade toggle: [All ▼] or select individual trade │    │
│  │ [Interactive zoom, hover shows trade ID + values]    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── Scatter ──────────────┐  ┌─── Residual Dist ──────┐   │
│  │ Pred vs Actual scatter   │  │ Histogram + KDE         │   │
│  │ 45° reference line       │  │ MAE: 0.0034             │   │
│  │ Color by trade type      │  │ P95: 0.0089             │   │
│  │ Size by abs(error)       │  │ P99: 0.0134             │   │
│  └──────────────────────────┘  └─────────────────────────┘   │
│                                                              │
│  ┌─── Per-Trade Metrics ────────────────────────────────┐    │
│  │ Sortable DataTable:                                   │    │
│  │ Trade ID | MAE | Max Error | Directional Acc | R²     │    │
│  │ [Filter by product type] [Sort by worst]              │    │
│  │ Highlight: worst trade, best trade                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── PnL Heatmap ─────────────────────────────────────┐    │
│  │ Residual heatmap: X=scenario, Y=trade                │    │
│  │ Color intensity = abs(residual)                       │    │
│  │ Hover shows exact values                              │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

| Component | Data Source |
|-----------|-------------|
| PnL time series overlay | `predictions.npz` (predictions + targets arrays) |
| Scatter plot | `predictions.npz` |
| Residual distribution | `residuals.npz` |
| Per-trade metrics table | Derived from `predictions.npz` + `trade_universe.json` |
| Residual heatmap | `residuals.npz` + `trade_universe.json` for labels |

#### Sub-Tab 4: Trade Universe

The data the model was trained on.

| Component | Data Source |
|-----------|-------------|
| Interactive graph visualisation (Plotly network/Cytoscape) | `registry/{v}/graph_builder.pkl` |
| Trade attribute summary table | `registry/{v}/trade_universe.json` |
| Selected vs removed trades (dimensionality reduction) | `trade_universe.json` selected/removed fields |
| Elementary → target mapping | `trade_universe.json` indices |

---

### 4.3 Page 2: Live Inference (Traders / Front Office)

**Primary users:** Traders, structurers, desk quants  
**Purpose:** "What is the predicted PnL for these trades/scenarios?"  
**Data flow:** Interactive — triggers model inference via cached `InferencePipeline`.

**Persistent header (always visible):**

```
┌─────────────────────────────────────────────────────────────────┐
│ Live Inference                         [Model: latest ▼]        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─── Input Panel ─────────────────────────────────────────┐    │
│  │  Upload trades CSV  [Browse...] [Upload]                 │    │
│  │  — OR —                                                  │    │
│  │  Scenario builder:                                       │    │
│  │    Shock Type: [Parallel ▼]  Magnitude: [+50bps]         │    │
│  │    Risk Factors: [IR_USD ☑] [FX_EUR ☑] [EQ_SPX ☐]       │    │
│  │                                       [Run Inference]    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─── Summary Cards (appear after inference) ──────────────┐    │
│  │  Total PnL: $1.24M | Mean: $12.4K | Max: $89.2K        │    │
│  │  Latency: 0.34s    | Trades: 104   | Model: v3-prod    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────┐ ┌────────────┐ ┌──────────────┐ ┌─────────────┐   │
│  │ Results  │ │ Trade Graph│ │ Elem. Trades │ │ Market Data │   │
│  └──────────┘ └────────────┘ └──────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Sub-Tab 1: Results

The primary output tab.

```
┌─── Results Sub-Tab ─────────────────────────────────────────┐
│                                                              │
│  ┌─── Per-Trade Predictions ────────────────────────────┐    │
│  │ DataTable:                                            │    │
│  │ Trade ID | Product | Predicted PnL | Rank | Export    │    │
│  │ [Sort] [Filter by product type]                       │    │
│  │ [Export CSV] [Export Excel]                            │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── PnL Distribution ─────┐  ┌─── Waterfall ──────────┐   │
│  │ Histogram of predicted    │  │ Top contributors to    │   │
│  │ PnL across trades         │  │ total PnL (waterfall)  │   │
│  └───────────────────────────┘  └────────────────────────┘   │
│                                                              │
│  ┌─── Inference History ────────────────────────────────┐    │
│  │ Recent runs (last 20):                                │    │
│  │ [timestamp | #trades | total PnL | scenario | ...]    │    │
│  │ Click to reload results from any past run             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### Sub-Tab 2: Trade Graph

After inference, shows the trade relationship graph with **new trades highlighted**.

```
┌─── Trade Graph Sub-Tab ─────────────────────────────────────┐
│                                                              │
│  ┌─── Interactive Network Graph ────────────────────────┐    │
│  │                                                       │    │
│  │  Training universe nodes: grey                        │    │
│  │  New inference trades: highlighted (blue/orange)      │    │
│  │  Target trades: square nodes                          │    │
│  │  Elementary trades: circle nodes                      │    │
│  │  Edge thickness: adjacency weight (RBF kernel)        │    │
│  │                                                       │    │
│  │  [Interactive: zoom, pan, hover for trade details]    │    │
│  │  [Toggle: show all / show new only / show neighbors]  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── Proximity Warning ────────────────────────────────┐    │
│  │ Trades far from training cluster (low confidence):    │    │
│  │ [TRADE_XYZ: nearest neighbor distance = 2.34]         │    │
│  │ [TRADE_ABC: nearest neighbor distance = 1.87]         │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

This tab uses the saved `graph_builder.pkl` from the registry to embed new
trades into the existing graph topology. The proximity warning is a useful
production feature — if a new trade is an outlier relative to the training
universe, the model's prediction should be treated with lower confidence.

#### Sub-Tab 3: Elementary Trades

Decomposition of the predicted PnL by elementary trade legs.

```
┌─── Elementary Trades Sub-Tab ───────────────────────────────┐
│                                                              │
│  ┌─── PnL Decomposition Waterfall ─────────────────────┐    │
│  │ Per-elementary-trade contribution to target PnL       │    │
│  │ [Waterfall chart: each bar = one leg's contribution]  │    │
│  │ Positive legs (green) | Negative legs (red)           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── Elementary PnL Table ─────────────────────────────┐    │
│  │ Elem ID | Product | PnL | Weight | Risk Factors      │    │
│  │ [Sortable, filterable by product type]                │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── Elementary PnL History ───────────────────────────┐    │
│  │ Time series of elementary PnL for selected trades     │    │
│  │ [Multi-select trades, overlay line chart]             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### Sub-Tab 4: Market Data

The underlying risk factor data that drove the prediction — the "why" behind
the numbers.

```
┌─── Market Data Sub-Tab ─────────────────────────────────────┐
│                                                              │
│  ┌─── Volatility Surface ───────────────────────────────┐    │
│  │ 3D surface or heatmap (strike × tenor × vol)         │    │
│  │ Base surface vs shocked surface overlay               │    │
│  │ [Select underlying: EUR/USD ▼]                        │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── Yield / Spot Curves ──────┐  ┌─── Curve Shifts ───┐   │
│  │ IR curves by tenor            │  │ Delta between base │   │
│  │ Base vs scenario overlay      │  │ and shocked curves │   │
│  │ [Multi-currency select]       │  │ [bar chart by tenor│   │
│  └───────────────────────────────┘  └────────────────────┘   │
│                                                              │
│  ┌─── Risk Factor Importance (if attention available) ──┐    │
│  │ Which risk factors did the model weight most heavily  │    │
│  │ for this particular inference run?                    │    │
│  │ [Horizontal bar chart: RF name → attention weight]    │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

The market data tab bridges the gap between "here's the number" and "here's why".
If a trader shocked IR by +50bps and sees a large negative PnL, they can visually
trace it back to the curve shift. The risk factor importance panel (available if
the model's attention layer exposes weights) shows what the model focused on for
this specific prediction.

**Implementation notes:**
- Model and artifacts cached in a module-level dict (keyed by version) so repeated inferences don't reload
- CSV parsing validates column schema against `data_config.json`
- Scenario builder applies shocks to the base risk factor data and re-runs through the pipeline
- Results table supports export to CSV/Excel for trader downstream use
- Inference history stored in a lightweight SQLite or JSON log
- Graph builder extended with new trade embeddings at inference time (no retrain needed)
- Vol surface data comes from the job/scenario inputs, rendered via Plotly Surface or Heatmap

---

### 4.4 Page 3: Risk & Analytics (Senior Management / Risk)

**Primary users:** Head of desk, CRO, risk committee  
**Purpose:** Portfolio-level model risk overview and scenario analysis.  
**Data source:** Aggregates from multiple evaluation runs, inference history, and registry metadata across versions.

**Persistent header:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Risk & Portfolio Analytics                 [Date Range ▼]       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─── Portfolio Summary Cards ─────────────────────────────┐    │
│  │  Total Trades: 1,247  |  Predicted PnL: $4.2M           │    │
│  │  Active Model: v3-prod  |  Active Alerts: 2              │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌───────────┐ ┌─────────────────┐ ┌───────────┐ ┌───────────┐  │
│  │ Dashboard │ │ Model Comparison│ │ Scenarios │ │ Drift     │  │
│  └───────────┘ └─────────────────┘ └───────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

#### Sub-Tab 1: Dashboard

The executive summary. Concentration risk treemap (PnL by product type), top 10
worst trades by prediction error, and the alert panel with actionable items.

#### Sub-Tab 2: Model Comparison

Side-by-side metrics across registered model versions. Rolling MAE/RMSE chart
(version on x-axis), evaluation results diff table, and a recommendation badge
for "should we promote the new model?"

#### Sub-Tab 3: Scenario Analysis

Compare PnL outcomes across saved scenario runs (base, stress, historical).
Grouped bar charts by product type per scenario, with a scenario selector
checklist. This is the risk committee's primary view.

#### Sub-Tab 4: Drift Monitoring

Residual distribution shift detection — KDE overlay of training-time residuals
vs recent inference residuals. Feature drift statistics, and automated alerts
when distributions diverge beyond configurable thresholds.

---

### 4.5 Page 4: Hyperparameter Tuning (Quants / ML Engineers)

**Primary users:** Quants, ML engineers  
**Purpose:** "What are the best hyperparameters for this model?"  
**Data source:** Tuning results from `artifacts/tuning/{study_name}/`.

**Persistent header:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Hyperparameter Tuning                  [Study ▼] [Load]        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─── Tuning Summary Card ─────────────────────────────────┐    │
│  │ Study: gnn_rnn_tune_v3  |  Direction: minimize           │    │
│  │ Trials: 87/100 complete  |  12 pruned  |  1 failed       │    │
│  │ Best: 0.00089 (trial #62)  |  Elapsed: 4h 12m            │    │
│  │ Best LR: 2.3e-4  |  GNN: 128x2  |  RNN: 96x2            │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌───────────┐ ┌───────────┐ ┌──────────────┐ ┌─────────────┐   │
│  │ Overview  │ │ Params    │ │ Landscape    │ │ Run Tuning  │   │
│  └───────────┘ └───────────┘ └──────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Sub-Tab 1: Overview

Optimisation history (objective value over trials with running-best line),
trial state breakdown (completed vs pruned vs failed pie chart), and the
best trial's full hyperparameter configuration.

| Component | Data Source |
|-----------|-------------|
| Optimisation history chart | `tuning_result.json` → `all_trials` |
| Running-best line | Derived from `all_trials` values |
| Trial state breakdown | `n_completed`, `n_pruned`, `n_trials` |
| Best params table | `best_params` |

#### Sub-Tab 2: Parameter Analysis

Deep dive into which hyperparameters matter most and how they interact.

```
┌─── Parameter Analysis Sub-Tab ──────────────────────────────┐
│                                                              │
│  ┌─── Importance ───────────┐  ┌─── Parallel Coordinates ┐  │
│  │ Horizontal bar chart:     │  │ Multi-dimensional view  │  │
│  │ fANOVA importance scores  │  │ Params colored by       │  │
│  │ [learning_rate: 0.42]     │  │ objective value          │  │
│  │ [gnn_units:     0.23]     │  │ [interactive filtering]  │  │
│  │ [rnn_dropout:   0.11]     │  │                          │  │
│  └───────────────────────────┘  └──────────────────────────┘  │
│                                                              │
│  ┌─── Per-Parameter Slice Plots ────────────────────────┐    │
│  │ Marginal objective value vs each parameter             │    │
│  │ [Select params: ☑ lr ☑ gnn_units ☑ rnn_dropout ...]    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── Contour Plots ───────────────────────────────────┐    │
│  │ 2D landscape: [param_x ▼] vs [param_y ▼]            │    │
│  │ Color = objective value                               │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

| Component | Data Source |
|-----------|-------------|
| Importance bar chart | `param_importances.png` or live Optuna study |
| Parallel coordinates | Live Optuna study (if SQLite storage used) or `all_trials` |
| Slice plots | `all_trials` |
| Contour plots | `all_trials` |

#### Sub-Tab 3: Landscape

Exploration of the hyperparameter landscape for understanding sensitivity.
Contour plots for any pair of parameters, and a sortable trials table showing
all trials with their params, values, states, and durations.

#### Sub-Tab 4: Run Tuning

Interactive tuning launch panel. Configure n_trials, pruner, search space
bounds, and start a tuning run from the UI. Progress bar updates as trials
complete. This is the only sub-tab that triggers compute.

```
┌─── Run Tuning Sub-Tab ──────────────────────────────────────┐
│                                                              │
│  ┌─── Configuration ────────────────────────────────────┐    │
│  │  Trials: [100]  Pruner: [hyperband ▼]  Seed: [42]    │    │
│  │                                                       │    │
│  │  Search Space Overrides:                              │    │
│  │  GNN Units:  min [32]  max [256]  step [32]           │    │
│  │  RNN Units:  min [32]  max [256]  step [32]           │    │
│  │  LR:         min [1e-5] max [1e-2] (log scale)       │    │
│  │                                                       │    │
│  │                         [Start Tuning]  [Cancel]      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─── Progress (live) ──────────────────────────────────┐    │
│  │  Trial 47/100  ████████████████░░░░░  47%             │    │
│  │  Current best: 0.00112 (trial #31)                    │    │
│  │  ETA: ~1h 45m                                         │    │
│  │                                                       │    │
│  │  Live optimisation history (updates per trial):       │    │
│  │  [streaming chart]                                    │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

### 4.6 Ensemble (Future)

The ensemble layer sits **behind** the inference page, not as a separate UI concern. The architecture:

```
UI "Run Inference" button
        │
        ▼
  ClusterRouter.key_for_trade(trade) → cluster_id
        │
        ▼
  KeyedModelEnsemble
        ├── registry.load("cluster_A") → model_A
        ├── registry.load("cluster_B") → model_B
        └── registry.load("cluster_C") → model_C
        │
        ▼
  Merge predictions by sample order
        │
        ▼
  InferenceResult (returned to UI)
```

The ensemble is invisible to the user — they still see one "Run Inference" button. The routing and merging happen inside the pipeline. Each cluster model has its own registry entry, and the ensemble config maps cluster IDs to versions.

This means the current single-model `InferencePipeline` is the building block. Once it works end-to-end for one model, the ensemble wraps multiple instances of it.

---

## 5. UI Folder Structure

### 5.1 Design Principle

The UI framework is split into two layers:

- **Core framework** (`src/rade_ml/ui/`) — model-independent. Theming, reusable
  components, data loading utilities, layout scaffolding. This never imports from
  `models/hybrid_gnn_rnn/` or any model-specific module.
- **Model apps** (`src/rade_ml/ui/apps/`) — model-dependent pages. Each model
  gets its own sub-package containing the three pages and their callbacks.

This means adding a new model to the UI is: create a new folder under `apps/`,
implement the page layouts and callbacks, register it in the app factory. The
core framework doesn't change.

### 5.2 Full Folder Structure

```
src/rade_ml/ui/
│
├── __init__.py
├── app.py                              # Dash app factory: create_app()
├── server.py                           # Entry point: python -m rade_ml.ui.server
├── config.py                           # UI configuration (paths, defaults, themes)
│
├── core/                               # ── Model-Independent Framework ──
│   ├── __init__.py
│   ├── theme.py                        # Bloomberg-style dark theme definition
│   ├── styles.py                       # CSS-in-Python style constants
│   ├── colors.py                       # Color palette (brand, status, chart series)
│   │
│   ├── components/                     # Reusable UI building blocks
│   │   ├── __init__.py
│   │   ├── summary_card.py             # Metric card component (value + label + delta)
│   │   ├── status_badge.py             # Green/amber/red status indicator
│   │   ├── version_selector.py         # Registry version dropdown (scans registry_dir)
│   │   ├── data_table.py               # Styled DataTable wrapper with export buttons
│   │   ├── tab_container.py            # Sub-tab container with consistent styling
│   │   ├── header.py                   # Page header with navigation
│   │   ├── sidebar.py                  # Optional sidebar for filters/controls
│   │   └── loading.py                  # Loading spinner overlay component
│   │
│   ├── data/                           # Data loading utilities
│   │   ├── __init__.py
│   │   ├── registry_loader.py          # Load RegistryEntry, list versions
│   │   ├── artifact_loader.py          # Load eval results, predictions, parquets
│   │   ├── inference_cache.py          # Model + artifact cache (keyed by version)
│   │   └── history_store.py            # Inference history read/write (SQLite or JSON)
│   │
│   └── plots/                          # Reusable Plotly figure builders
│       ├── __init__.py
│       ├── loss_curves.py              # Training loss curve (train vs val)
│       ├── scatter.py                  # Pred vs actual scatter with 45° line
│       ├── residual.py                 # Residual histogram + KDE
│       ├── heatmap.py                  # Residual heatmap (trade × scenario)
│       ├── waterfall.py                # PnL decomposition waterfall
│       ├── time_series.py              # Overlay time series (pred vs actual)
│       └── network_graph.py            # Trade relationship graph (Plotly/Cytoscape)
│
├── apps/                               # ── Model-Dependent Applications ──
│   ├── __init__.py
│   │
│   └── hybrid_gnn_rnn/                 # Hybrid GNN-RNN model app
│       ├── __init__.py
│       ├── register.py                 # Register pages with the Dash app
│       │
│       ├── pages/                      # Page layouts (pure layout, no logic)
│       │   ├── __init__.py
│       │   ├── model_review.py         # Page 1: Model Review layout
│       │   ├── live_inference.py        # Page 2: Live Inference layout
│       │   └── risk_analytics.py       # Page 3: Risk & Analytics layout
│       │
│       ├── callbacks/                  # Callback logic (reactive behaviour)
│       │   ├── __init__.py
│       │   ├── model_review_cb.py      # Page 1 callbacks (version load, tab switch)
│       │   ├── live_inference_cb.py    # Page 2 callbacks (upload, run, tab switch)
│       │   └── risk_analytics_cb.py    # Page 3 callbacks (filters, comparisons)
│       │
│       ├── data/                       # Model-specific data transforms
│       │   ├── __init__.py
│       │   ├── eval_loader.py          # Load GNN-RNN-specific eval artifacts
│       │   ├── inference_prep.py       # CSV parsing, scenario building → model inputs
│       │   └── market_data.py          # Vol surface / curve extraction from job data
│       │
│       └── plots/                      # Model-specific plot builders
│           ├── __init__.py
│           ├── trade_graph.py          # GNN-RNN trade graph with new trade overlay
│           ├── elementary_decomp.py    # Elementary trade PnL decomposition
│           └── vol_surface.py          # 3D vol surface / yield curve rendering
│
└── assets/                             # Static assets (Dash auto-serves this folder)
    ├── styles.css                      # Global CSS overrides
    ├── favicon.ico                     # App icon
    └── fonts/                          # Custom fonts (monospace for numbers)
```

### 5.3 How It Scales

Adding a second model (e.g. deep hedging):

```
src/rade_ml/ui/apps/
├── hybrid_gnn_rnn/          # existing
└── deep_hedging/            # new — same structure
    ├── register.py
    ├── pages/
    ├── callbacks/
    ├── data/
    └── plots/
```

The `app.py` factory discovers and registers all model apps automatically
(or via explicit config). The navigation bar updates to show a model selector.

### 5.4 Data Loading Strategy for Page 1

Page 1 (Model Review) is **entirely read-only**. No model loading, no pipeline
execution. When the user selects a version from the dropdown:

```
User selects version "20260222_143052_a1b2c3"
        │
        ▼
  registry_loader.load_entry(version)
        → RegistryEntry from registry/{v}/metadata.json
        │
  artifact_loader.load_training(version)
        → loss curve, training report from artifacts/training/{v}/
        │
  artifact_loader.load_evaluation(version)
        → eval_results.json, predictions.npz, residuals.npz,
          target_pnl.parquet, elementary_pnl.parquet
          from artifacts/evaluation/{v}/
        │
  registry_loader.load_trade_universe(version)
        → trade_universe.json from registry/{v}/
        │
        ▼
  All sub-tabs populated instantly (< 1 second)
```

The evaluation plots (pred vs actual, residuals, heatmaps) are generated from
the saved numpy arrays — no model inference needed. The graph visualisation loads
the saved `graph_builder.pkl` from the registry.

**Advanced: Re-run Evaluation button** — In the Evaluation sub-tab corner, a
small "Re-evaluate" button (disabled by default, with a confirmation dialog)
allows quants to re-run the eval pipeline with different parameters. This is
an edge case, not the default flow.

---

## 6. Visual Design: Bloomberg-Style Professional Theme

### 6.1 Design System

The goal is a dark, information-dense UI that feels like a professional terminal
— clean, fast, authoritative. Not a consumer app. Think Bloomberg Terminal meets
modern web.

### 6.2 Color Palette

```python
# src/rade_ml/ui/core/colors.py

PALETTE = {
    # ── Backgrounds ──
    "bg_primary":      "#0A0E17",     # Near-black (main background)
    "bg_secondary":    "#111827",     # Slightly lighter (cards, panels)
    "bg_tertiary":     "#1F2937",     # Elevated surfaces (hover, active tab)
    "bg_input":        "#162032",     # Input fields, dropdowns

    # ── Text ──
    "text_primary":    "#E5E7EB",     # Primary text (light grey)
    "text_secondary":  "#9CA3AF",     # Secondary text (muted)
    "text_muted":      "#6B7280",     # Labels, captions
    "text_heading":    "#F9FAFB",     # Headings (near-white)

    # ── Accent / Brand ──
    "accent_blue":     "#3B82F6",     # Primary action, links
    "accent_cyan":     "#06B6D4",     # Secondary accent, highlights

    # ── Status ──
    "status_green":    "#10B981",     # Positive PnL, success, pass
    "status_red":      "#EF4444",     # Negative PnL, error, fail
    "status_amber":    "#F59E0B",     # Warning, caution
    "status_blue":     "#3B82F6",     # Informational

    # ── Chart Series (8 distinguishable colors for multi-line) ──
    "series": [
        "#3B82F6",    # blue
        "#10B981",    # green
        "#F59E0B",    # amber
        "#EF4444",    # red
        "#8B5CF6",    # purple
        "#EC4899",    # pink
        "#06B6D4",    # cyan
        "#F97316",    # orange
    ],

    # ── Borders & Dividers ──
    "border":          "#1F2937",
    "border_active":   "#3B82F6",
    "divider":         "#1F2937",
}
```

### 6.3 Typography

```python
# src/rade_ml/ui/core/styles.py

FONT_STACK = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO  = "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace"

# All numeric values (PnL, metrics, trade IDs) use FONT_MONO
# All labels, headings, body text use FONT_STACK
```

**Key rules:**
- Numbers always in monospace — aligns columns, looks professional
- Font size: 13px base (dense but readable), 11px for table cells
- No rounded corners > 4px — sharp, terminal aesthetic
- Cards have 1px border in `border` color, no shadows (flat design)

### 6.4 Component Styling

```python
# Summary Card style
CARD_STYLE = {
    "backgroundColor": PALETTE["bg_secondary"],
    "border": f"1px solid {PALETTE['border']}",
    "borderRadius": "4px",
    "padding": "16px 20px",
    "fontFamily": FONT_STACK,
}

# Metric value inside card
METRIC_VALUE_STYLE = {
    "fontSize": "24px",
    "fontWeight": "600",
    "fontFamily": FONT_MONO,
    "color": PALETTE["text_heading"],
    "letterSpacing": "-0.5px",
}

# Metric label
METRIC_LABEL_STYLE = {
    "fontSize": "11px",
    "fontWeight": "500",
    "fontFamily": FONT_STACK,
    "color": PALETTE["text_muted"],
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
}

# PnL value — green/red based on sign
def pnl_style(value: float) -> dict:
    color = PALETTE["status_green"] if value >= 0 else PALETTE["status_red"]
    return {**METRIC_VALUE_STYLE, "color": color}

# DataTable
TABLE_STYLE = {
    "style_header": {
        "backgroundColor": PALETTE["bg_tertiary"],
        "color": PALETTE["text_secondary"],
        "fontWeight": "600",
        "fontSize": "11px",
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
        "borderBottom": f"2px solid {PALETTE['border_active']}",
        "fontFamily": FONT_STACK,
    },
    "style_cell": {
        "backgroundColor": PALETTE["bg_secondary"],
        "color": PALETTE["text_primary"],
        "fontSize": "12px",
        "fontFamily": FONT_MONO,
        "borderBottom": f"1px solid {PALETTE['border']}",
        "padding": "8px 12px",
        "textAlign": "right",
    },
    "style_data_conditional": [
        {"if": {"row_index": "odd"},
         "backgroundColor": PALETTE["bg_primary"]},
    ],
}

# Tab styling
TAB_STYLE = {
    "backgroundColor": PALETTE["bg_primary"],
    "color": PALETTE["text_muted"],
    "borderBottom": f"2px solid transparent",
    "padding": "8px 16px",
    "fontSize": "12px",
    "fontWeight": "500",
    "fontFamily": FONT_STACK,
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "color": PALETTE["text_heading"],
    "borderBottom": f"2px solid {PALETTE['accent_blue']}",
    "backgroundColor": PALETTE["bg_secondary"],
}
```

### 6.5 Plotly Chart Template

```python
# src/rade_ml/ui/core/theme.py

import plotly.graph_objects as go
import plotly.io as pio

rade_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PALETTE["bg_secondary"],
        plot_bgcolor=PALETTE["bg_primary"],
        font=dict(
            family=FONT_STACK,
            size=12,
            color=PALETTE["text_secondary"],
        ),
        title=dict(
            font=dict(size=14, color=PALETTE["text_heading"]),
            x=0, xanchor="left",
        ),
        xaxis=dict(
            gridcolor=PALETTE["border"],
            zerolinecolor=PALETTE["border"],
            tickfont=dict(family=FONT_MONO, size=11),
        ),
        yaxis=dict(
            gridcolor=PALETTE["border"],
            zerolinecolor=PALETTE["border"],
            tickfont=dict(family=FONT_MONO, size=11),
        ),
        colorway=PALETTE["series"],
        margin=dict(l=48, r=16, t=40, b=36),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=PALETTE["text_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor=PALETTE["bg_tertiary"],
            font=dict(family=FONT_MONO, size=12, color=PALETTE["text_primary"]),
            bordercolor=PALETTE["border_active"],
        ),
    ),
)

pio.templates["rade"] = rade_template
pio.templates.default = "rade"
```

### 6.6 Dash App Setup

```python
# src/rade_ml/ui/app.py

import dash
import dash_bootstrap_components as dbc

def create_app(
    output_dir: str,
    model: str = "hybrid_gnn_rnn",
    debug: bool = False,
) -> dash.Dash:
    app = dash.Dash(
        __name__,
        use_pages=True,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title="RADE ML",
        update_title="Loading...",
    )

    # Override DARKLY with our custom theme via assets/styles.css
    # Register model-specific pages
    if model == "hybrid_gnn_rnn":
        from src.rade_ml.ui.apps.hybrid_gnn_rnn.register import register_pages
        register_pages(app, output_dir)

    return app
```

### 6.7 Visual Reference

The overall aesthetic:

```
┌──────────────────────────────────────────────────────────────────┐
│  ┌─ RADE ML ──────────────────────────────── Model: GNN-RNN ─┐  │
│  │                                                            │  │
│  │  MODEL REVIEW    LIVE INFERENCE    RISK & ANALYTICS        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐    │
│  │ BEST LOSS  │ │   EPOCH    │ │  EVAL MAE  │ │  EVAL P95  │    │
│  │            │ │            │ │            │ │            │    │
│  │  0.00142   │ │   47/100   │ │   0.0034   │ │   0.0089   │    │
│  │ ▼ -12.3%   │ │            │ │ ▲ +2.1%    │ │ ▼ -5.4%    │    │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘    │
│                                                                  │
│  ╔══════════╗ ┌──────────┐ ┌────────────┐ ┌───────────────┐     │
│  ║ OVERVIEW ║ │ TRAINING │ │ EVALUATION │ │ TRADE UNIVERSE│     │
│  ╚══════════╝ └──────────┘ └────────────┘ └───────────────┘     │
│  ─────────────────────── blue underline ─────────────────────    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                                                          │    │
│  │                    (sub-tab content)                      │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Dark background (#0A0E17) everywhere                            │
│  Cards on #111827 with 1px #1F2937 border                        │
│  Numbers in monospace, green/red for PnL                         │
│  Blue accent for active tab, links, focus states                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Artifact & Registry Structure — What Changed and Target State

### 7.1 What Changed (Summary)

| File | Change | Why |
|------|--------|-----|
| `base.py` `TrainPipeline.post_train` | Stores `self._registered_entry` after `registry.register()` | Subclasses need the version to save artifacts alongside the model |
| `base.py` `_generate_training_report` | Path: `artifacts_dir/training/{version}/` (was `training_reports/{run_name}_{timestamp}`) | Version-keyed paths for consistency with eval and registry |
| `base.py` `EvalPipeline.__init__` | Added `self._loaded_entry` | Subclasses need the version to key evaluation artifacts |
| `base.py` `EvalPipeline.run` | Stores `_loaded_entry`, calls `get_target_scaler()`, passes `data_result` to `post_eval()` | Enables inverse transform and model-specific post-eval saving |
| `base.py` `EvalPipeline.get_target_scaler` | New hook (returns `None`) | Model pipelines override to provide the fitted scaler |
| `evaluator.py` `run()` | Added `target_scaler` parameter, wires `_inverse_transform()` | Metrics in original PnL units, not z-scores |
| `train.py` `post_train` | Calls `_save_inference_artifacts()` | Persists everything needed for cold-start inference into registry dir |
| `train.py` `_save_inference_artifacts` | **New.** Saves 6 files into `registry/{version}/` | Self-contained registry entry for inference without re-running data pipeline |
| `eval.py` `get_target_scaler` | Overrides base hook | Returns `data_result.metadata["target_pnl_transformer"]` |
| `eval.py` `post_eval` | Calls `_save_evaluation_data()` | Persists analytics for the UI |
| `eval.py` `_save_evaluation_data` | **New.** Saves 5 files into `artifacts/evaluation/{version}/` | UI reads pre-computed data |

### 7.2 Before (Old Structure)

```
rade_ml_output/
├── registry/
│   └── {version}/
│       ├── model.keras
│       └── metadata.json
│
├── experiments/
│   └── {run_id}/run.json
│
└── artifacts/
    └── training_reports/
        └── {run_name}_{timestamp}/
            ├── training_report.md
            └── loss_curve.png
```

Problems:
- Registry only had model weights + metadata — nothing for cold-start inference
- No evaluation artifacts saved anywhere
- Training reports keyed by run_name+timestamp, not linked to registry version
- Inference required re-running the full data pipeline to get graph builder, encoder, scalers

### 7.3 After (Target Structure)

```
rade_ml_output/
│
├── registry/                                    ← registry_dir
│   └── 20260222_143052_a1b2c3/                  # Self-contained inference + eval unit
│       ├── model.keras                           # Trained model weights
│       ├── metadata.json                         # RegistryEntry (metrics, config, tags)
│       ├── graph_builder.pkl                     # [NEW] Fitted TradeGraphBuilder
│       ├── encoder.pkl                           # [NEW] Fitted TradeAttributeEncoder
│       ├── target_scaler.pkl                     # [NEW] Fitted StandardScaler (target PnL)
│       ├── elementary_scaler.pkl                 # [NEW] Fitted StandardScaler (elementary)
│       ├── data_config.json                      # [NEW] HybridGnnRnnDataConfig snapshot
│       ├── trade_universe.json                   # [NEW] Trade IDs, indices, selected/removed
│       ├── target_pnl.parquet                    # [NEW] Original target PnL DataFrame
│       ├── elementary_pnl.parquet                # [NEW] Original elementary PnL DataFrame
│       ├── target_attributes.json                # [NEW] Target trade attributes (moneyness, delta, etc.)
│       ├── elementary_attributes.json            # [NEW] Elementary trade attributes
│       └── datasets/                             # [NEW] Cached tf.data.Datasets
│           ├── train/                            # tf.data.Dataset.save() output
│           ├── val/                              # tf.data.Dataset.save() output
│           └── test/                             # tf.data.Dataset.save() output
│
├── experiments/                                  ← tracking_dir (unchanged)
│   └── {run_id}/run.json
│
└── artifacts/                                    ← artifacts_dir
    ├── training/                                 # [RENAMED] was training_reports/
    │   └── 20260222_143052_a1b2c3/               # [CHANGED] keyed by version
    │       ├── training_report.md
    │       ├── loss_curve.png
    │       └── trade_graph.png
    │
    ├── evaluation/                               # [NEW] entire directory
    │   └── 20260222_143052_a1b2c3/               # Keyed by version
    │       ├── eval_results.json                  # Metrics, loss, dataset_info
    │       ├── predictions.npz                    # Test-set predictions + targets
    │       ├── residuals.npz                      # Test-set residual arrays
    │       ├── target_pnl.parquet                 # Original target PnL DataFrame
    │       ├── elementary_pnl.parquet             # Original elementary PnL DataFrame
    │       └── splits/                            # [NEW] Per-split pred vs actual
    │           ├── train.npz                      # Train predictions + targets
    │           ├── val.npz                        # Val predictions + targets
    │           └── test.npz                       # Test predictions + targets
    │
    └── tuning/                                   # [NEW] entire directory
        └── gnn_rnn_tune_v3/                      # Keyed by study_name
            ├── tuning_result.json                 # TuningResult (all trials, best params)
            ├── optimization_history.png           # Objective value over trials
            └── param_importances.png              # fANOVA importance bar chart
```

### 7.4 Key Properties of the New Structure

| Property | Detail |
|----------|--------|
| **Cold-start inference** | `registry/{version}/` has everything: model, graph builder, encoder, scalers, config, trade universe. One directory = one self-contained unit. |
| **Version-keyed everywhere** | Training artifacts, evaluation artifacts, and registry all use the same version string. Jump from one to another trivially. |
| **UI reads, never computes** | Page 1 loads files from `artifacts/` and `registry/`. Zero model calls. |
| **Inference uses registry only** | Page 2 loads from `registry/{version}/`. The `artifacts/` directory is irrelevant for inference. |
| **Backward compatible** | Old registry entries (model.keras + metadata.json only) still load fine. Missing `.pkl`/`.json` files are simply absent. |

---

## 8. Recommended Implementation Order for Work Env

| # | Task | Dependencies | Effort |
|---|------|-------------|--------|
| 1 | Copy the four changed source files to work env | None | 10 min |
| 2 | Run full train pipeline → verify registry + artifacts structure | (1) | 30 min |
| 3 | Run eval pipeline against registered model → verify eval artifacts | (2) | 20 min |
| 4 | Build `HybridGnnRnnInferencePipeline` (single model) | (2) | 2-3 hrs |
| 5 | Build UI core framework (`ui/core/`, theme, components) | None | 3-4 hrs |
| 6 | Build UI Page 1: Model Review (read-only analytics) | (3), (5) | 3-4 hrs |
| 7 | Build UI Page 2: Live Inference (upload + run) | (4), (5) | 4-5 hrs |
| 8 | Build UI Page 3: Risk & Analytics | (3), (7) | 3-4 hrs |
| 9 | Ensemble framework (`ClusterRouter`, `KeyedModelEnsemble`) | (4) | 4-5 hrs |
| 10 | Wire ensemble into inference page | (7), (9) | 2 hrs |
| 11 | Unit tests for all new components | Ongoing | 3-4 hrs |

**Priority:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

The UI core framework (step 5) and inference pipeline (step 4) are independent
and can be built in parallel. Page 1 (step 6) can also be built in parallel
with step 4 since it only reads files.

---

## 9. File Diff Summary

For quick reference, here are the exact files changed/created in this session:

```
MODIFIED:
  src/rade_ml/pipelines/base.py                       # _registered_entry, _loaded_entry, get_target_scaler, report path, TunePipeline
  src/rade_ml/pipelines/__init__.py                   # Export TunePipeline
  src/rade_ml/pipelines/hybrid_gnn_rnn/__init__.py    # Export HybridGnnRnnTunePipeline
  src/rade_ml/pipelines/hybrid_gnn_rnn/train.py       # _save_inference_artifacts
  src/rade_ml/pipelines/hybrid_gnn_rnn/eval.py        # get_target_scaler, _save_evaluation_data
  src/rade_ml/evaluation/evaluator.py                 # target_scaler param + inverse transform wiring

CREATED:
  src/rade_ml/pipelines/hybrid_gnn_rnn/tune.py        # HybridGnnRnnTunePipeline (search space, build_trial_model)
  docs/rade_ml/WORK_ENV_CHANGES_SUMMARY.md             # This document
```

All changes are additive — no existing behaviour was removed. The training report path format changed (`training_reports/` → `training/`) but this is cosmetic and only affects new runs.
