# Ensemble Model — Implementation Guide

## 1. Overview

The ensemble model combines N members (each trained on a separate cluster of trades) into
a single prediction surface covering the full trade universe.

**Core idea:** Trades are clustered by similarity (e.g. product type, underlying, risk
profile). Each cluster gets its own model with a focused trade graph / feature set. At
prediction time a `TradeRouter` directs each trade to its cluster's model and results are
concatenated into a unified output.

**Model-agnostic:** The ensemble layer does not depend on any specific model architecture.
Members can be Hybrid GNN-RNN, GNN-only, RNN-only, or any `nn.Module` that conforms to
the `BaseModel` interface. Different clusters can even use different architectures (e.g.
Hybrid GNN-RNN for exotic clusters, RNN-only for vanilla clusters) — each cluster's
`PipelineConfig` specifies its own `model_type` and corresponding pipeline class.

**Design principles:**
- The ensemble is the "product"; clusters are the implementation detail.
- Model architecture is a per-cluster configuration choice, not a framework constraint.
- Offline: quants train, evaluate, register. Online: traders/FO/Risk consume via UI.
- The UI runs only inference; everything else is read-only display of saved artifacts.

---

## 2. Folder Structure

```
src/rade_ml_pt/
├── ensemble/
│   ├── __init__.py
│   │
│   ├── config.py                   # EnsembleConfig
│   │                               #   - member_configs: Dict[str, PipelineConfig]
│   │                               #   - pipeline_class: Dict[str, str]
│   │                               #     (cluster_id -> pipeline dotpath, e.g.
│   │                               #      "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train.HybridGnnRnnTrainPipeline"
│   │                               #      "src.rade_ml_pt.pipelines.rnn.train.RnnTrainPipeline")
│   │                               #     defaults to HybridGnnRnnTrainPipeline for all if omitted
│   │                               #   - cluster_mapping: Dict[str, List[str]]
│   │                               #     (cluster_id -> list of trade IDs)
│   │                               #   - aggregation: str ("concat", "weighted_mean")
│   │                               #   - weights: Optional[Dict[str, float]]
│   │                               #   - registry_dir, artifacts_dir
│   │                               #   - to_dict() / from_dict() / from_yaml()
│   │
│   ├── builder.py                  # EnsembleBuilder
│   │                               #   - build(config) -> EnsembleModel
│   │                               #   - _validate_coverage(cluster_mapping, trade_universe)
│   │                               #   - _load_members(member_versions, registry)
│   │
│   ├── model.py                    # EnsembleModel
│   │                               #   - members: Dict[str, nn.Module]
│   │                               #   - router: TradeRouter
│   │                               #   - predict(inputs) -> combined predictions
│   │                               #   - predict_member(cluster_id, inputs) -> member preds
│   │                               #   - get_member_metadata() -> per-member info for UI
│   │
│   ├── router.py                   # TradeRouter
│   │                               #   - cluster_mapping: Dict[str, List[str]]
│   │                               #   - route(trade_ids) -> Dict[str, List[str]]
│   │                               #   - assign_new_trade(attribs) -> cluster_id
│   │                               #   - get_cluster_for_trade(trade_id) -> cluster_id
│   │
│   ├── aggregation.py              # Aggregation strategies
│   │                               #   - concat_aggregate()  (disjoint clusters)
│   │                               #   - weighted_mean_aggregate()  (overlapping)
│   │                               #   - stacking_aggregate()  (meta-learner, future)
│   │
│   ├── registry.py                 # EnsembleRegistry
│   │                               #   - register(ensemble_model, metadata) -> version
│   │                               #   - load(version) -> EnsembleModel
│   │                               #   - tag(version, label)  e.g. "production"
│   │                               #   - list_versions() -> for UI version selector
│   │                               #   - get_metadata(version) -> for UI overview
│   │
│   ├── metrics.py                  # Ensemble-level metric aggregation
│   │                               #   - aggregate_member_metrics(per_member_results)
│   │                               #   - compute_ensemble_metrics(combined_preds, targets)
│   │                               #   - build_comparison(version_a, version_b)
│   │                               #   - build_trade_to_cluster_mapping() -> for UI drill-down
│   │
│   └── plots.py                    # Ensemble-specific visualisations
│                                   #   - plot_member_comparison(member_metrics)
│                                   #   - plot_cluster_performance_heatmap(member_metrics)
│                                   #   - plot_trade_cluster_assignment(router, features)
│                                   #   - plot_version_comparison(metrics_a, metrics_b)
│                                   #   - plot_ensemble_vs_members(ensemble_preds, member_preds)
│
├── pipelines/
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── train.py                # EnsembleTrainPipeline
│   │   ├── eval.py                 # EnsembleEvalPipeline
│   │   └── infer.py                # EnsembleInferencePipeline
│   │
│   └── hybrid_gnn_rnn/             # EXISTING (unchanged)
│       ├── train.py
│       ├── eval.py
│       ├── infer.py
│       └── tune.py
```

---

## 3. Pipelines

### 3.1 EnsembleTrainPipeline

```
EnsembleTrainPipeline.run()
    │
    ├── For each cluster in config.cluster_mapping:
    │       │
    │       ├── Resolve pipeline class for this cluster:
    │       │     config.pipeline_class.get(cluster_id, default=HybridGnnRnnTrainPipeline)
    │       │     → dynamically import via importlib (dotpath string)
    │       │
    │       └── pipeline_cls(member_config).run()
    │           → build data (this cluster's trades only)
    │           → train model (architecture determined by pipeline_cls)
    │           → register member as "{cluster_id}_{version}" in ModelRegistry
    │           → save artifacts (training plots, scalers, graph_builder, trade_graph)
    │
    ├── EnsembleBuilder.build()
    │       → load all N members from registry
    │       → validate trade coverage (every target assigned to exactly one member)
    │       → create EnsembleModel with TradeRouter
    │
    └── EnsembleRegistry.register()
            → save: member_versions, cluster_mapping, weights, aggregation config
            → save: trade_cluster_map.json (trade_id -> cluster_id)
            → tag as new ensemble version
```

The pipeline class resolution uses `importlib.import_module` so adding a new model type
(e.g. `RnnTrainPipeline`) requires zero changes to the ensemble code — just provide the
dotpath in config.

### 3.2 EnsembleEvalPipeline

```
EnsembleEvalPipeline.run()
    │
    ├── Load ensemble from EnsembleRegistry (all N members + router)
    │
    ├── Build/load test data (full trade universe, all scenarios)
    │
    ├── For each member:
    │       → route member's trades
    │       → member.predict() on those trades
    │       → compute per-member metrics (MAE, MSE, residual stats)
    │       → save per-member evaluation artifacts
    │
    ├── Aggregate predictions across members (concat for disjoint)
    │
    ├── Compute ensemble-level metrics
    │
    ├── Save ensemble evaluation artifacts:
    │       → ensemble plots (predicted vs actual, residuals, etc.)
    │       → member_summary.json (per-cluster metrics table)
    │       → per-split predictions (train/val/test .npz)
    │
    └── post_eval()
```

### 3.3 EnsembleInferencePipeline

```
EnsembleInferencePipeline.run()
    │
    ├── Load ensemble from EnsembleRegistry
    │
    ├── Resolve input_mode (new_trades | new_scenarios)
    │
    ├── If new_trades:
    │       → TradeRouter.assign_new_trade() for each new trade
    │       → For affected members: extend graph (build_graph_projection)
    │       → Each member predicts its trades
    │       → Aggregate
    │
    ├── If new_scenarios:
    │       → Load job from registry (portfolio, elementary trade objects)
    │       → Build new elementary PnL from scenario CSVs
    │       → Each member: same graph, new pnl_history, predict
    │       → Aggregate
    │
    └── post_infer()
            → save predictions CSV
            → compute summary stats
            → return combined InferenceResult
```

---

## 4. Artifacts Layout

The artifacts directory is the "API" between offline pipelines and the UI dashboard.

```
artifacts_dir/
├── ensemble/
│   └── ensemble_v3/
│       ├── config.json                     # EnsembleConfig (cluster_mapping, member versions)
│       ├── metrics.json                    # Ensemble-level metrics
│       ├── trade_cluster_map.json          # { "USDHKD_Call_1Y": "cluster_0", ... }
│       ├── member_summary.json             # { "cluster_0": {trades: 412, mae: 0.041}, ... }
│       ├── version_history.json            # Previous versions for comparison
│       │
│       ├── plots/                          # Ensemble-level plots
│       │   ├── member_comparison.png
│       │   ├── cluster_heatmap.png
│       │   ├── ensemble_predicted_vs_actual.png
│       │   ├── ensemble_residual_distribution.png
│       │   └── ensemble_residual_by_target.png
│       │
│       ├── predictions/                    # Per-split combined predictions
│       │   ├── train.npz
│       │   ├── val.npz
│       │   └── test.npz
│       │
│       └── members/                        # Per-member artifacts
│           ├── cluster_0/
│           │   ├── training_plots.png
│           │   ├── trade_graph.png
│           │   ├── eval_metrics.json
│           │   ├── predicted_vs_actual.png
│           │   ├── residual_distribution.png
│           │   ├── residual_by_target.png
│           │   └── predictions/
│           │       ├── train.npz
│           │       ├── val.npz
│           │       └── test.npz
│           ├── cluster_1/
│           │   └── ...
│           └── cluster_2/
│               └── ...
```

### Key artifact: `trade_cluster_map.json`

This file enables all cluster-based filtering in the UI. Every trade ID maps to its
cluster. The UI uses this to:
- Filter predictions tables by cluster.
- Navigate from a specific trade to its cluster's training/eval artifacts.
- Show cluster assignment for new trades after inference.

---

## 5. UI Dashboard

### 5.1 Scope

The dashboard is a **read-only** consumer of artifacts. Only inference runs live.

| Action | In the UI? |
|--------|-----------|
| Train models | No (offline: CLI, scripts, cloud) |
| Run evaluation | No (offline: CLI, scripts) |
| View training results | Yes (read saved artifacts) |
| View evaluation results | Yes (read saved artifacts) |
| Compare versions | Yes (read saved artifacts) |
| Run inference | Yes (EnsembleInferencePipeline.run()) |

### 5.2 Pages

#### Overview

Landing page. One glance tells FO/Risk: "is the model healthy?"

- Active ensemble version (tagged "production"), trained/evaluated dates.
- Key metric cards: Val MAE, Val MSE, Worst Trade (with directional arrows vs previous version).
- Loss curve + pred vs actual thumbnails (link to Model Performance).
- Version history with "Compare versions" button.

#### Model Performance

Training and evaluation results. Ensemble first, cluster drill-down second.

**Ensemble section:**
- Training plots (loss curves, train-val gap, other metrics).
- Evaluation plots (predicted vs actual, residual distribution, residual by target).
- Metrics table.
- Member comparison bar chart (MAE per cluster).
- Cluster performance heatmap.

**Per-cluster breakdown table:**

| Cluster | Trades | MAE | MSE | Status |
|---------|--------|-----|-----|--------|
| cluster_0 | 412 | 0.041 | 0.006 | ok |
| cluster_1 | 387 | 0.052 | 0.009 | ok |

Click a row to drill into that cluster's detail.

**Cluster detail (drill-down):**
- Training loss curves for this member.
- Trade graph visualisation (network + adjacency + PCA).
- Evaluation plots for this member's trades only.
- Residual by target with the selected trade highlighted (if navigated from trade click).
- Back to ensemble button.

#### Inference

The action page for traders.

- Model version selector (defaults to "production" tag).
- Input mode toggle: New Scenarios / New Trades.
- File upload (CSV or folder).
- Run button.
- Results: summary stats (mean PnL, std, VaR), predictions table (sortable, filterable by cluster), PnL distribution histogram, download CSV.
- If new trades: cluster assignment table (which cluster each new trade was routed to).

#### Risk & Analytics

Deeper analysis for Risk teams.

- **PnL Timeline:** Predicted vs actual across train/val/test (per-split toggle), filterable by cluster.
- **Tail Risk:** P95/P99 residual over time, worst N scenarios table, worst N trades table.
- **Trade Drilldown:** Select a trade → prediction timeseries, residual scatter, cluster info, link to cluster detail.
- **Version Comparison:** Side-by-side metric deltas, improved/degraded trades list.

### 5.3 Visualization drill-down flow

The key user journey: trader sees worst trade → understands why → reviews the responsible model.

```
Overview: "Worst trade: USDHKD_EUR_Call_1Y (MAE: 0.21)"
    │ [click trade]
    ▼
Risk & Analytics → Trade Drilldown
    Trade: USDHKD_EUR_Call_1Y
    Cluster: cluster_0  (from trade_cluster_map.json)
    [Prediction timeseries]  [Residual scatter]
    │ [click "View cluster_0 model details"]
    ▼
Model Performance → Cluster Detail: cluster_0
    [Training loss curves]  [Trade graph]
    [Eval predicted vs actual]  [Residual by target — trade highlighted]
    [← Back to ensemble]
```

### 5.4 Dashboard folder structure

```
dash_app/
├── app.py                      # Main Dash app, layout, routing
├── pages/
│   ├── overview.py             # Landing: active version, key metrics
│   ├── model_performance.py    # Training/eval plots, cluster breakdown + drill-down
│   ├── inference.py            # Upload, predict, view results
│   └── risk_analytics.py       # PnL timeline, tail risk, trade drill-down, version compare
├── callbacks/
│   ├── infer_callbacks.py      # Only pipeline the UI runs
│   └── artifact_callbacks.py   # Load and display artifacts from registry
├── components/
│   ├── plot_viewer.py          # Renders saved PNGs or plotly figures
│   ├── metrics_table.py        # Styled metrics table
│   ├── upload.py               # File upload component
│   └── version_selector.py     # Version dropdown (reads EnsembleRegistry)
└── utils/
    ├── artifact_loader.py      # Reads from artifacts_dir (or GCS via ArtifactStore)
    └── config_builder.py       # Builds PipelineConfig from UI form inputs
```

### 5.5 Design guidelines

1. **Dark theme** with clean typography (Inter or similar). FO tools are always dark.
2. **Metrics as big numbers** with directional arrows (up/down vs previous version).
3. **Minimal clicks**: Overview → one click to any detail. No nested menus.
4. **Tables are interactive**: sortable, filterable, searchable.
5. **Plots are secondary**: thumbnails first, expand on click.
6. **Version selector** always visible (top bar or sidebar).
7. **No ML jargon on the surface**: "Model accuracy" not "val_mae", "Prediction error" not "residual_p95". Tooltips for technical definitions.

---

## 6. Implementation Order

| Step | What | Depends on |
|------|------|-----------|
| 1 | `ensemble/config.py` + `router.py` | Nothing (foundation) |
| 2 | `ensemble/model.py` + `aggregation.py` | Step 1 |
| 3 | `ensemble/builder.py` + `registry.py` | Steps 1–2 |
| 4 | `pipelines/ensemble/train.py` | Steps 1–3 + existing HybridGnnRnn pipeline |
| 5 | `pipelines/ensemble/eval.py` | Steps 1–3 |
| 6 | `ensemble/metrics.py` + `plots.py` | Steps 4–5 (needs saved artifacts) |
| 7 | `pipelines/ensemble/infer.py` | Steps 1–3 + HybridGnnRnn infer pipeline |
| 8 | Dash UI dashboard | Steps 4–7 (needs artifacts to display) |

---

## 7. Key Design Decisions

### Disjoint vs overlapping clusters

If clusters are **disjoint** (each trade belongs to exactly one cluster), aggregation is
simple concatenation and there's no ambiguity. If clusters **overlap** (a trade appears in
multiple clusters), you need `weighted_mean_aggregate` or a meta-learner. Start with
disjoint; add overlap support later if needed.

### Ensemble versioning

An ensemble version is a **bundle** of member versions plus routing config. Changing any
member (re-training one cluster) creates a new ensemble version. The registry stores the
full mapping so any past ensemble can be exactly reproduced.

### Pipeline reuse

`EnsembleTrainPipeline` dynamically resolves the pipeline class per cluster from
`config.pipeline_class`. No training logic is duplicated. The ensemble layer is purely
orchestration + routing + aggregation, completely decoupled from model architecture.

### Model-agnostic design

The ensemble layer has exactly **one** contract with the underlying model: it must be an
`nn.Module` subclass that accepts a dict of inputs and returns a prediction tensor. This
is already satisfied by `BaseModel`. The decoupling points are:

| Layer | Coupled to model? | How |
|-------|-------------------|-----|
| `EnsembleModel` | No | Stores `Dict[str, nn.Module]`, calls `forward()` |
| `TradeRouter` | No | Works on trade IDs, no model knowledge |
| `aggregation.py` | No | Works on output tensors |
| `EnsembleRegistry` | No | Stores `nn.Module` state dicts + metadata JSON |
| `EnsembleTrainPipeline` | No | Resolves pipeline class from config string |
| `EnsembleEvalPipeline` | No | Loads members from registry, calls `predict()` |
| `EnsembleInferencePipeline` | Partially | Input prep (graph projection etc.) is delegated to member's own inference pipeline |

To add a new model type (e.g. RNN-only), you need:
1. Implement `RnnTrainPipeline`, `RnnEvalPipeline`, `RnnInferencePipeline` under
   `pipelines/rnn/` (following the same `TrainPipeline` / `EvalPipeline` base classes).
2. Set `pipeline_class: {"cluster_0": "src.rade_ml_pt.pipelines.rnn.train.RnnTrainPipeline"}`
   in `EnsembleConfig` for that cluster.
3. No changes to any ensemble code.

You can even **mix architectures** within a single ensemble — e.g. Hybrid GNN-RNN for
exotic clusters where graph structure matters, RNN-only for vanilla clusters where the
graph adds no value.

### UI as artifact consumer

The Dash app never imports `torch` or runs training/evaluation. It reads JSON, PNG, NPZ,
and CSV files from the artifacts directory. This means:
- The UI can run on a lightweight server (no GPU, no PyTorch install).
- Artifacts are the contract between ML code and the UI.
- You can swap the UI framework without touching any ML code.
