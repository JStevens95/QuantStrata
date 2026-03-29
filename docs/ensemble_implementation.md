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
│   │                               #   - cluster_key: optional ["ccy", "desk", "product"]
│   │                               #   - cluster_key_values: optional {cluster_id: [val_ccy, val_desk, val_product]}
│   │                               #   - get_cluster_keys_for_router() -> {cluster_id: {attr: value}}
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
│   │                               #   - cluster_keys: optional {cluster_id: {ccy, product, desk, ...}}
│   │                               #     for attribute-based routing of new trades (no centroids needed)
│   │                               #   - route(trade_ids) -> Dict[str, List[str]]
│   │                               #   - assign_new_trade(attribs) -> cluster_id (keys or centroids)
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

### 5.1a Who uses the ensemble pipelines (offline vs UI)

The three ensemble pipelines have different users and run locations:

| Pipeline | Who uses it | Where it runs | Purpose |
|----------|-------------|---------------|---------|
| **EnsembleTrainPipeline** | Quants | Offline (CLI, scripts, cloud) | Train one model per cluster, register ensemble. Never runs in the UI. |
| **EnsembleEvalPipeline** | Quants | Offline (CLI, scripts) | Evaluate the ensemble on test data, write metrics and plots to `artifacts_dir`. Never runs in the UI. |
| **EnsembleInferencePipeline** | Traders / FO / Risk (via UI) | In the UI session (or a backend the UI calls) | Load ensemble (or use session cache), apply user’s new scenarios/trades to copies, run predict, return results. This is the only pipeline the UI runs. |

So: **train and eval are purely for Quant offline use**. They produce the registry and artifact files (models, config, trade_cluster_map, training/eval plots, metrics). The **UI only runs inference** (and only reads the rest). The UI may use a session that pre-loads the ensemble once; when the user clicks “Run inference”, the UI (or backend) then uses that in-memory ensemble and the inference pipeline logic (prepare inputs from user upload → predict → display), without re-running the train or eval pipelines.

### 5.1b Detailed UX steps: how the ensemble flows in the UI

Step-by-step flow for the **user** and what the **system** does under the hood:

**Phase 1: Offline (Quants, no UI)**

1. **Train** — Quant runs `EnsembleTrainPipeline.run()` (CLI/script). For each cluster, the pipeline trains a member model, saves it to the model registry, then registers the ensemble (config, member_versions, trade_cluster_map) in the ensemble registry. Training plots and metadata are written to `artifacts_dir` and registry.
2. **Evaluate** — Quant runs `EnsembleEvalPipeline.run()` (CLI/script). Pipeline loads the ensemble from registry, runs each member on its test data, computes ensemble and per-member metrics, saves evaluation artifacts (JSON, plots) under `artifacts_dir/ensemble/{version}/evaluation/`. Nothing runs in the UI.

**Phase 2: UI — Landing (light load)**

3. **User opens the dashboard** — UI loads **light ensemble metadata only**: read from ensemble registry (config, member_versions, trade_cluster_map) and optionally pre-saved summary files (e.g. member_summary.json). No member models or heavy artifacts are loaded.
4. **Landing page renders** — Tables and light plots: cluster list, number of trades per cluster, version selector, high-level metric cards (from saved eval metrics if available). User sees “e.g. 5 clusters, 412 + 387 + … trades” and can switch version or click “Load full ensemble”.

**Phase 3: UI — Full load (optional, heavy)**

5. **User triggers full load** — e.g. clicks “Load full ensemble” or opens “Model Performance” or “Inference”. Backend/session loads: ensemble config + all member models from registry + train result metadata + eval result (or paths to saved plots) + inference baseline artifacts (e.g. graph_builder, encoder per member if needed). Everything is kept in the session (in-memory).
6. **Session is “warm”** — From here on, ensemble overview, cluster-filtered training/eval plots, and inference can use this in-memory state; no need to hit the registry again for the same version.

**Phase 4: UI — Viewing (read-only)**

7. **User browses Model Performance** — UI reads **saved artifacts** (training curves, eval plots, metrics) from `artifacts_dir` and, if full load was done, can also use in-memory data. Cluster breakdown table and drill-down (cluster detail, trade graph, residual by target) come from these artifacts and/or session. No pipeline run; just read + display.
8. **User browses Risk & Analytics** — Same idea: read saved artifacts (PnL timeline, tail risk, version comparison). Optional: if re-computation is ever needed, it could use the session’s in-memory ensemble and test data, but typically the UI just displays pre-saved results.

**Phase 5: UI — Inference (only live pipeline)**

9. **User goes to Inference tab** — Chooses version (if not already loaded), selects “New Scenarios” or “New Trades”, uploads file(s). Clicks “Run”.
10. **Backend runs inference** — If the session has the ensemble loaded, it uses that. Otherwise it loads the ensemble (or does a one-off `EnsembleInferencePipeline.run()`). Inference pipeline logic: **copy** baseline inputs from session (or registry), apply user’s new scenarios/trades to the copies, run `ensemble.predict()` on the copies, return predictions. Stored registry/artifacts are not modified.
11. **UI shows results** — Summary stats, predictions table (filterable by cluster), download CSV, and if new trades: cluster assignment table.

**Summary**

- **Offline:** `EnsembleTrainPipeline` and `EnsembleEvalPipeline` are for Quants only; they run outside the UI and produce registry + artifacts.
- **UI:** The UI **reads** registry metadata and artifacts (landing + model performance + risk). The only pipeline the UI **runs** is inference (`EnsembleInferencePipeline`), using a session-held ensemble and working on **copies** of inputs so stored data is never changed. Full load is one explicit or on-demand step; after that, viewing and inference are fast from memory.

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

### 5.5 UI runtime: load once, reuse for eval and inference

**What “loading an ensemble” gives you**

When the UI (or any client) loads an ensemble version from the registry, it gets:

- **Per cluster:** trained model (weights), training result metadata (e.g. best_val_loss, history), and optionally a cached **test dataset** (`registry/{member_version}/datasets/test.pt`) if it was saved at train time.
- **Ensemble-level:** config, member_versions, trade_cluster_map, and any saved **evaluation results** (metrics, plots) under `artifacts_dir/ensemble/{version}/evaluation/`.

You do **not** get raw input data (scenario CSVs, full trade table, etc.). Those stay outside the registry. So: per cluster we have training result, eval result (if run), and model + optional test.pt; we do not have the original training/eval input data in the registry.

**Why load once**

For UI runtime you want to load registry artifacts **once per session** and reuse them for:

1. **Evaluation (view)** — read saved eval metrics/plots from artifacts; optionally re-run eval using the same in-memory ensemble + cached test data.
2. **Inference** — run predictions by only updating the variable parts (new scenarios or new trades); no need to reload the ensemble or registry a second time.

So the same in-memory ensemble (and optionally cached test loaders for eval) is used for both “show eval” and “run inference,” and only the inference inputs change.

**Can the ensemble component handle this?**

Yes. This should be supported **in the ensemble component**, not only in the UI:

- **Eval:** The UI can load the ensemble (and optionally per-member test `DataLoader`s) once. The component can support an optional **pre-loaded ensemble** (and optionally pre-loaded test data) so `EnsembleEvalPipeline` does not reload from disk when the UI has already loaded.
- **Inference:** Similarly, the UI can pass a **pre-loaded ensemble** into the inference path so `EnsembleInferencePipeline` skips `_load_ensemble()` and only runs `prepare_inputs()` (with the new scenarios/trades) and `predict()`.

Concretely:

- Add optional constructor/call arguments to **EnsembleEvalPipeline** and **EnsembleInferencePipeline**, e.g. `ensemble`, `ens_config`, `member_versions` (and for eval optionally `member_test_loaders`), so that when provided, the pipeline uses them instead of loading from registry.
- Alternatively (or in addition), introduce an **EnsembleSession** (or similar) that:
  - Loads once: `EnsembleRegistry.load(version)` + `EnsembleBuilder.build(...)` + optionally load test data per member.
  - Exposes `run_eval()` and `run_inference(infer_meta)` that use the in-memory ensemble and only touch disk for writing results (e.g. saving inference CSV).

Then the UI simply: (1) on session start or version change, call the single “load” path and cache the result; (2) for eval, use cached eval artifacts or call `run_eval()` with the cached session; (3) for inference, call `run_inference(infer_meta)` with the cached session and the new inputs. No double load of registry data.

**Summary**

| Question | Answer |
|----------|--------|
| Do we have input data when we load an ensemble? | No. We have models, config, optional test.pt and saved eval artifacts; not raw scenario/trade inputs. |
| Load registry once and reuse for eval + inference? | Yes. Supported by allowing pipelines (or an EnsembleSession) to accept pre-loaded ensemble (and optional test data) so the UI loads once and reuses. |
| Where to implement? | In the **ensemble component** (optional args or session object). The UI then just passes the cached load; no need to load registry twice. |

**Copy-based inference: never mutate stored data**

Once the ensemble is loaded, the UI has access to all cluster results and the baseline inputs needed for inference (e.g. graph, encoder state, reference pnl). When the user runs inference (new scenarios or new trades), the pipeline should work on **copies** of those baseline inputs: apply the user’s specifications (uploaded files, new trade rows) to the copies, run the model, and return results. The registry and any stored artifact remain **read-only**; nothing is written back. So: load → read-only baseline; inference → copy baseline → apply user specs → predict → display. Stored inputs and results are never changed; all ad hoc manipulation and updates happen on in-memory copies.

**Load time and runtime**

The first load can be large and slow because it typically involves:

- Ensemble metadata (config, member_versions, trade_cluster_map) — small, fast.
- Per cluster: loading each member **model** from disk (state dict or full module). With many clusters or large models this dominates.
- Optionally: per-member **test data** (`test.pt`) if the UI re-runs eval; more I/O and memory.
- Optionally: per-member **inference artifacts** (e.g. graph_builder, encoder, scalers for Hybrid GNN–RNN) when the user first runs inference — can be sizeable (adjacency, feature matrices).

To keep load time and memory acceptable:

1. **Lazy load:** Load ensemble config and metadata first so the UI can render (version selector, cluster list). Load member models in the background or on-demand when the user opens a cluster or clicks “Run inference”.
2. **Cache in session:** Once a member (or inference artifacts) is loaded, keep it in memory; don’t reload on tab switch or repeat inference.
3. **Avoid loading test data for eval view** if the UI only shows saved eval metrics and plots (read JSON/PNG from `artifacts_dir`). Only load `test.pt` if the UI must re-run evaluation.
4. **Defer inference artifacts:** Load graph_builder / encoder only when the user first runs inference (or when they open the inference tab), not during the initial ensemble load.

So: yes, the first full load can be heavy; design the session so that the “critical path” is minimal (metadata first, then lazy or on-demand load of models and inference artifacts), and keep everything in memory for the rest of the session so inference and navigation are fast.

**Recommended UX: light landing, then full load**

A practical flow that keeps the landing page fast and defers heavy work until the user needs it:

1. **Landing / summary (light)**  
   Load only **light ensemble metadata**: config, member_versions, trade_cluster_map (and any pre-saved summary stats if available). No member models or large artifacts. From this you can:
   - Show **light plots and tables**: what each cluster contains, number of trades per cluster, cluster list, version selector.
   - Display a high-level summary (e.g. "5 clusters, 412 + 387 + … trades") and links to drill down.

2. **Full ensemble load (heavy, explicit)**  
   When the user chooses to "Load full ensemble" (or the first time they open a cluster, run inference, or view training/eval plots), load **all cluster results and inputs**: member models, train result, eval result, and inference baseline artifacts. Store everything in the session.

3. **After full load**  
   The user can then:
   - View **ensemble overview** and **cluster-filtered training / eval plots** from the in-memory session (no further disk load).
   - Navigate between clusters and tabs without reloading.

4. **Inference**  
   **Quick**: the session already holds models and baseline inputs; the UI only applies the user's new scenarios or trades to **copies**, runs predict, and shows results. No second load.

So: **landing = metadata + light summary only**; **full load = one explicit or on-demand step**; **then everything in memory** for fast model overview, cluster-filtered plots, and inference.

### 5.6 Design guidelines

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

### New-trade routing: cluster keys vs centroids

For **known** trades, routing is a lookup: `cluster_mapping` (and `trade_cluster_map.json`)
give trade_id → cluster_id. For **new** trades at inference time you need a rule to assign
them. Two options:

- **Cluster keys (attribute-based):** You can define routing by a shared list of attribute
  names and per-cluster value lists. Set **cluster_key** = e.g. `["ccy", "desk", "product"]`
  and **cluster_key_values** = e.g. `{"cluster_0": ["GBP", "FLOW_RATES", "EUROPEAN"], "cluster_1": ["USD", "FLOW_RATES", "AMERICAN"]}`.
  The config builds the key dict per cluster (same order as `cluster_key`), and the router
  assigns a new trade to the cluster whose key matches the trade’s attributes. Alternatively
  set **cluster_keys** directly: `{cluster_id: {attr: value, ...}}`. No centroid data needed.
- **Centroids:** You have a feature vector per cluster (e.g. from training). The router
  assigns the new trade to the cluster whose centroid is nearest to the trade’s feature
  vector (e.g. `trade_attribs["features"]`). Requires pre-computed centroids.

`assign_new_trade` tries **keys first**, then **centroids**, then **default_cluster**. So
you can use cluster keys only (no centroids), and route by ccy/product/desk (or any
attribute set you define per cluster).

**Example: config, router, and ensemble with attribute-based routing**

Using the shared-attribute-names + per-cluster-values format:

```python
from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.router import TradeRouter

# 1. Config: cluster_key = list of attribute names; each cluster has a list of values in that order
config = EnsembleConfig(
    cluster_mapping={
        "cluster_0": ["trade_001", "trade_002"],   # known trades in this cluster
        "cluster_1": ["trade_003", "trade_004"],
    },
    cluster_key=["ccy", "desk", "product"],
    cluster_key_values={
        "cluster_0": ["GBP", "FLOW_RATES", "EUROPEAN"],
        "cluster_1": ["USD", "FLOW_RATES", "AMERICAN"],
    },
    aggregation="concat",
    registry_dir="/path/to/registry",
    artifacts_dir="/path/to/artifacts",
)

# 2. Build ensemble (loads members from registry; router gets keys from config)
from src.rade_ml_pt.registry.store import ModelRegistry
registry = ModelRegistry(config.registry_dir)
builder = EnsembleBuilder(registry)
member_versions = {"cluster_0": "v_abc", "cluster_1": "v_def"}  # from EnsembleRegistry
ensemble = builder.build(config, member_versions)

# 3. Router is inside the ensemble; it has cluster_keys = {"cluster_0": {"ccy": "GBP", ...}, ...}
router = ensemble.router

# --- Known trades: lookup by trade_id ---
router.get_cluster_for_trade("trade_001")   # -> "cluster_0"
router.route(["trade_001", "trade_003"])   # -> {"cluster_0": ["trade_001"], "cluster_1": ["trade_003"]}

# --- New trades: assign by attributes (no centroids) ---
new_trade_attribs = {"ccy": "USD", "desk": "FLOW_RATES", "product": "AMERICAN"}
cluster_id = router.assign_new_trade(new_trade_attribs)   # -> "cluster_1"

# 4. Inference: pipeline builds member_inputs per cluster (e.g. from new scenario/trade data),
#    then ensemble.predict(member_inputs) returns combined predictions
member_inputs = {
    "cluster_0": {"trade_features": ..., "pnl_history": ..., ...},
    "cluster_1": {"trade_features": ..., "pnl_history": ..., ...},
}
predictions = ensemble.predict(member_inputs)   # shape (n_scenarios, n_total_targets)
```

So: **config** holds `cluster_key` + `cluster_key_values` (and `cluster_mapping` for known trades). **Builder** turns that into a **TradeRouter** with derived `cluster_keys` and an **EnsembleModel**. **Known trades** use `get_cluster_for_trade` / `route`; **new trades** use `assign_new_trade(trade_attribs)` with the same attribute names. **Inference** uses the same ensemble and router; the inference pipeline is responsible for building `member_inputs` (e.g. routing new trades to clusters via `assign_new_trade`, then preparing each cluster’s model inputs).

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

---

## 8. Step-by-Step Workflow (Hybrid GNN-RNN)

This section is the hands-on execution guide for building, evaluating, and running inference
with an ensemble of Hybrid GNN-RNN models. Follow these steps in order.

### 8.1 Prerequisites

Before starting you need:

1. **Data caches per cluster.** Each cluster needs its own set of cached data files:
   - `elementary_pnl.pkl` — elementary PnL time series for this cluster's trades.
   - `target_pnl.pkl` — target PnL time series.
   - `elementary_attribs.pkl` — trade attributes for elementary trades.
   - `target_attribs.pkl` — trade attributes for target trades.

   These are typically produced by a clustering / data preparation step that splits
   the full trade universe into groups (by ccy, desk, product type, etc.) and saves
   each group's data to a separate directory.

2. **A list of job dictionaries.** Each job has a `cluster_info` key containing the
   four file paths for that cluster:

   ```python
   jobs = [
       {
           "cluster_info": {
               "elementary_pnl_path": "/data/cluster_0/elementary_pnl.pkl",
               "target_pnl_path": "/data/cluster_0/target_pnl.pkl",
               "elementary_attribs_path": "/data/cluster_0/elementary_attribs.pkl",
               "target_attribs_path": "/data/cluster_0/target_attribs.pkl",
           }
       },
       {
           "cluster_info": {
               "elementary_pnl_path": "/data/cluster_1/elementary_pnl.pkl",
               "target_pnl_path": "/data/cluster_1/target_pnl.pkl",
               "elementary_attribs_path": "/data/cluster_1/elementary_attribs.pkl",
               "target_attribs_path": "/data/cluster_1/target_attribs.pkl",
           }
       },
       # ... one per cluster
   ]
   ```

3. **Directories for outputs:**
   - `registry_dir` — where member models and ensemble registrations are saved.
   - `artifacts_dir` — where plots, metrics, and evaluation results are written.

4. **Trade-to-cluster assignment.** You need to know which target trade IDs belong
   to each cluster. This comes from your clustering step (e.g. a `trade_cluster_map`
   CSV or dict produced by your data prep pipeline).

### 8.2 Prepare Jobs and Cluster Mapping

The goal is to go from your list of job dicts to the three inputs `EnsembleConfig` needs:

```
jobs: List[Dict]
  │
  ├─► cluster_mapping:    {cluster_id: [trade_ids]}
  ├─► member_configs:     {cluster_id: {data_config, training_config, metadata}}
  └─► cluster_key_values: {cluster_id: [routing_attribute_values]}
  │
  ▼
EnsembleConfig(...)
```

**Step 2a: Build `cluster_mapping`**

Load the target trade IDs from each cluster's data and assign them a cluster ID:

```python
from src.rade_ml_pt.data.io import CacheLoader

cluster_mapping = {}
for i, job in enumerate(jobs):
    cluster_id = f"cluster_{i}"
    target_attribs = CacheLoader.load(job["cluster_info"]["target_attribs_path"])
    # target_attribs is a dict with a "trade_id" key containing the list of IDs
    trade_ids = target_attribs["trade_id"]
    cluster_mapping[cluster_id] = trade_ids
```

**Step 2b: Build `member_configs`**

Each cluster needs its own `PipelineConfig` dict with the data paths, training settings,
and model config. The ensemble train pipeline will turn each into a full `PipelineConfig`
via `EnsembleConfig.get_member_pipeline_config()`:

```python
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import (
    HybridGnnRnnDataConfig,
    FolderEnvironmentConfig,
    DimensionalityConfig,
    BasisSelectionConfig,
    GraphBuilderConfig,
    AttributeEncoderConfig,
)
from src.rade_ml_pt.core.config import TrainingConfig, OptimizerConfig, EarlyStoppingConfig

member_configs = {}
for i, job in enumerate(jobs):
    cluster_id = f"cluster_{i}"

    data_config = HybridGnnRnnDataConfig(
        folders=FolderEnvironmentConfig(root_folder=f"/data/cluster_{i}"),
        validation_split=0.10,
        test_split=0.05,
        seq_length=1,
        batch_size=32,
        shuffle=True,
        transform_type="standard",
        dimensionality=DimensionalityConfig(
            reduction_mode="basis_selection",
            basis_selection=BasisSelectionConfig(var_threshold=0.999999, method="pca", max_components=10),
        ),
        graph_builder=GraphBuilderConfig(k=3, distance_metric="euclidean"),
        attribute_encoder=AttributeEncoderConfig(
            numeric_keys=["moneyness", "yrs_to_maturity", "delta", "vega"],
            categorical_keys=["product_type", "product_subtype", "trade_type"],
            multi_label_keys=["underlying_risk_factors"],
        ),
        seed=42,
    )

    training_config = TrainingConfig(
        epochs=100,
        loss="mae",
        metrics=["mae"],
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
        early_stopping=EarlyStoppingConfig(patience=10, monitor="val_loss", mode="min", restore_best_weights=True),
        strategy="auto",
        verbose=False,
    )

    member_configs[cluster_id] = {
        "data_config": data_config,
        "training_config": training_config.to_dict(),
        "model_config": None,  # uses HybridGnnRnn defaults; override per cluster if needed
        "metadata": {"job": job, "run_name": f"ensemble_{cluster_id}"},
    }
```

**Step 2c: Build routing keys (for new-trade assignment)**

If your clusters are defined by business attributes (e.g. ccy, desk, product), provide
them so `TradeRouter` can assign unseen trades at inference time:

```python
cluster_key = ["ccy", "desk"]

cluster_key_values = {
    "cluster_0": ["GBP", "FLOW_RATES"],
    "cluster_1": ["USD", "FLOW_RATES"],
    "cluster_2": ["EUR", "EXOTICS"],
}
```

If clusters are defined by a more complex rule (e.g. ML-based clustering), skip
`cluster_key` / `cluster_key_values` and rely on centroid-based routing or a
`default_cluster` instead.

### 8.3 Configure EnsembleConfig

Bring it all together:

```python
from src.rade_ml_pt.ensemble.config import EnsembleConfig

REGISTRY_DIR = "/path/to/registry"
ARTIFACTS_DIR = "/path/to/artifacts"

ensemble_config = EnsembleConfig(
    member_configs=member_configs,
    cluster_mapping=cluster_mapping,
    cluster_key=cluster_key,
    cluster_key_values=cluster_key_values,
    aggregation="concat",              # disjoint clusters
    execution_strategy="sequential",   # one member at a time
    registry_dir=REGISTRY_DIR,
    artifacts_dir=ARTIFACTS_DIR,
    metadata={"run_name": "ensemble_v1"},
)
```

Verify before proceeding:

```python
print(f"Members: {ensemble_config.n_members}")
print(f"Total trades: {len(ensemble_config.all_trade_ids)}")
print(f"Cluster IDs: {ensemble_config.cluster_ids}")

# Check routing keys resolve correctly
router_keys = ensemble_config.get_cluster_keys_for_router()
for cid, keys in router_keys.items():
    print(f"  {cid}: {keys}")
```

### 8.4 Run EnsembleTrainPipeline

Train all member models sequentially and register the ensemble:

```python
from src.rade_ml_pt.pipelines.ensemble.train import EnsembleTrainPipeline

train_pipeline = EnsembleTrainPipeline(
    config=ensemble_config,
    tags=["production"],  # tag the ensemble for easy retrieval
)

train_output = train_pipeline.run()

print(f"Ensemble version: {train_output['ensemble_version']}")
print(f"Member versions:")
for cid, ver in train_output["member_versions"].items():
    result = train_output["member_results"][cid]
    print(f"  {cid}: {ver} (val_loss={result.best_val_loss:.6f}, epoch={result.best_epoch})")
```

**What gets saved:**

For each member (in `registry_dir/{member_version}/`):
- Model weights (`model.pt`)
- Training result (`training_result.json`)
- Data artifacts: `graph_results.joblib`, `encoder_results.joblib`, scalers, trade_universe
- Cached datasets: `datasets/train.pt`, `datasets/val.pt`, `datasets/test.pt`
- Job metadata: `cluster_info.joblib`

For the ensemble (in `registry_dir/ensemble/{ensemble_version}/`):
- `ensemble_config.json` — full `EnsembleConfig`
- `member_versions.json` — `{cluster_id: member_version}`
- `trade_cluster_map.json` — `{trade_id: cluster_id}` for every trade
- `member_summary.json` — per-member training stats

### 8.5 Run EnsembleEvalPipeline

Evaluate the ensemble on each member's cached test data:

```python
from src.rade_ml_pt.pipelines.ensemble.eval import EnsembleEvalPipeline

eval_pipeline = EnsembleEvalPipeline(
    ensemble_config=ensemble_config,
    ensemble_version=train_output["ensemble_version"],
)

eval_output = eval_pipeline.run()

print(f"Ensemble metrics: {eval_output['ensemble_metrics']}")
print(f"Per-member metrics:")
for cid, metrics in eval_output["per_member_metrics"].items():
    print(f"  {cid}: MAE={metrics.get('mae', 'N/A'):.6f}, MSE={metrics.get('mse', 'N/A'):.6f}")
```

**What gets saved** (in `artifacts_dir/ensemble/{version}/evaluation/`):
- `ensemble_metrics.json` — combined MAE, MSE, R2, etc.
- `per_member_metrics.json` — per-cluster metrics
- `member_rollup.json` — aggregated summary
- `plots/member_comparison_mae.png` — bar chart comparing members
- `plots/cluster_performance_heatmap.png` — heatmap of all metrics x clusters

**Prerequisite:** Each member must have cached test data (`datasets/test.pt`) saved
during training. The `HybridGnnRnnTrainPipeline` saves these automatically when
`artifacts_dir` is set on the member's `PipelineConfig`.

### 8.6 Run EnsembleInferencePipeline

Two inference modes are supported:

**Mode 1: New Scenarios (same trades, new risk-factor data)**

```python
from src.rade_ml_pt.pipelines.ensemble.infer import EnsembleInferencePipeline
import numpy as np

# Prepare PnL history per cluster (new scenario data)
cluster_pnl_histories = {
    "cluster_0": np.load("/data/new_scenarios/cluster_0_pnl.npy"),
    "cluster_1": np.load("/data/new_scenarios/cluster_1_pnl.npy"),
}

infer_config = EnsembleConfig(
    **ensemble_config.to_dict(),
    metadata={
        **ensemble_config.metadata,
        "inference": {
            "input_mode": "new_scenarios",
            "cluster_pnl_histories": cluster_pnl_histories,
        },
    },
)

infer_pipeline = EnsembleInferencePipeline(
    ensemble_config=infer_config,
    ensemble_version=train_output["ensemble_version"],
)
result = infer_pipeline.run()

print(f"Predictions shape: {result.predictions.shape}")
print(f"Latency: {result.latency_seconds:.3f}s")
```

**Mode 2: New Trades (unseen trades routed via cluster keys)**

```python
# Trade attributes for the new trade
new_trade = {
    "ccy": "GBP",
    "desk": "FLOW_RATES",
    "product_type": "vanilla_option",
    # ... other attributes
}

infer_config = EnsembleConfig(
    **ensemble_config.to_dict(),
    metadata={
        **ensemble_config.metadata,
        "inference": {
            "input_mode": "new_trades",
            "new_trade_attribs": new_trade,
            "pnl_history": pnl_array,  # global fallback PnL
        },
    },
)

infer_pipeline = EnsembleInferencePipeline(
    ensemble_config=infer_config,
    ensemble_version=train_output["ensemble_version"],
)
result = infer_pipeline.run()
```

The router assigns the new trade to its cluster using the `cluster_key` / `cluster_key_values`
rules defined in the config. In this example, `ccy=GBP, desk=FLOW_RATES` matches `cluster_0`.

**Pre-built member inputs (model-agnostic fallback):**

If you prepare member inputs yourself (e.g. for a non-hybrid model), pass them directly
via `metadata["inference"]["member_inputs"]` and the pipeline skips all model-specific
context loading:

```python
metadata={
    "inference": {
        "input_mode": "new_scenarios",
        "member_inputs": {
            "cluster_0": {"pnl_history": ..., "edge_index": ..., ...},
            "cluster_1": {"pnl_history": ..., "edge_index": ..., ...},
        },
    },
}
```

### 8.7 Load via EnsembleSession (UI / Interactive Use)

For repeated inference (e.g. in a dashboard), load the ensemble once and reuse:

```python
from src.rade_ml_pt.ensemble.session import EnsembleSession

session = EnsembleSession(
    registry_dir=REGISTRY_DIR,
    artifacts_dir=ARTIFACTS_DIR,
)

# Phase 1: Light metadata (fast — config, member versions, trade map)
session.load_metadata(version_or_tag="production")

print(f"Version: {session.ensemble_version}")
print(f"Members: {session.config.n_members}")

# Phase 2: Display artifacts (training/eval plots, metrics — for UI rendering)
session.load_display_artifacts()

# Phase 3: Full inference state (models + inference contexts — heavy, do once)
session.load_inference_state()

# Now run inference as many times as needed without reloading
result = session.run_inference(
    input_mode="new_scenarios",
    cluster_pnl_histories=cluster_pnl_histories,
)

# Or via the pipeline with session warm-start
infer_pipeline = EnsembleInferencePipeline(
    ensemble_config=ensemble_config,
    ensemble_version="production",
    session=session,
)
result = infer_pipeline.run()
```

**Session phases:**

| Phase | What loads | Memory | Time |
|-------|-----------|--------|------|
| 1. `load_metadata` | Config, member_versions, trade_cluster_map | Small | Fast |
| 2. `load_display_artifacts` | Eval metrics, plot paths, member summaries | Small | Fast |
| 3. `load_inference_state` | Member models, graph builders, encoders, scalers | Large | Slow |

Phase 3 is the heavy step. After that, `run_inference()` only builds inputs from the
user's new data and runs the forward pass — no disk I/O.

### 8.8 Artifact Inventory

Complete checklist of what each pipeline produces and where:

**EnsembleTrainPipeline**

```
registry_dir/
├── {member_version}/                    # one per cluster (via ModelRegistry)
│   ├── model.pt
│   ├── training_result.json
│   ├── graph_results.joblib
│   ├── encoder_results.joblib
│   ├── elementary_scaler.pkl
│   ├── target_scaler.pkl
│   ├── trade_universe.json
│   ├── cluster_info.joblib
│   ├── data_config.json
│   └── datasets/
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
└── ensemble/
    └── {ensemble_version}/
        ├── ensemble_config.json
        ├── member_versions.json
        ├── trade_cluster_map.json
        └── member_summary.json

artifacts_dir/
└── ensemble/
    └── {ensemble_version}/
        └── member_summary.json
```

**EnsembleEvalPipeline**

```
artifacts_dir/
└── ensemble/
    └── {ensemble_version}/
        └── evaluation/
            ├── ensemble_metrics.json
            ├── per_member_metrics.json
            ├── member_rollup.json
            └── plots/
                ├── member_comparison_mae.png
                └── cluster_performance_heatmap.png
```

**EnsembleInferencePipeline**

```
artifacts_dir/
└── inference/
    ├── predictions.csv
    └── inference_result.json
```

### 8.9 Quick-Reference: Minimal End-to-End Script

```python
"""Minimal ensemble: train, evaluate, infer."""
from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.pipelines.ensemble.train import EnsembleTrainPipeline
from src.rade_ml_pt.pipelines.ensemble.eval import EnsembleEvalPipeline
from src.rade_ml_pt.pipelines.ensemble.infer import EnsembleInferencePipeline

# 1. Config (see sections 8.2–8.3 for building these)
config = EnsembleConfig(
    member_configs=member_configs,
    cluster_mapping=cluster_mapping,
    cluster_key=["ccy", "desk"],
    cluster_key_values=cluster_key_values,
    aggregation="concat",
    execution_strategy="sequential",
    registry_dir="/path/to/registry",
    artifacts_dir="/path/to/artifacts",
)

# 2. Train
train_out = EnsembleTrainPipeline(config, tags=["production"]).run()

# 3. Evaluate
eval_out = EnsembleEvalPipeline(config, train_out["ensemble_version"]).run()

# 4. Infer (new scenarios)
config.metadata["inference"] = {
    "input_mode": "new_scenarios",
    "cluster_pnl_histories": cluster_pnl_histories,
}
result = EnsembleInferencePipeline(config, train_out["ensemble_version"]).run()
```

---

## 9. Pipeline Walkthrough: Inputs, Internal Flow, and Outputs

This section provides a concrete trace of what each pipeline requires, what happens
internally at each step, and what comes out. Use this as a reference when debugging or
when constructing inputs for the first time.

### 9.1 Ensemble Training

**Inputs required:**

| Input | Type | Where it comes from |
|-------|------|---------------------|
| `member_configs` | `{cluster_id: {data_config, training_config, model_config}}` | Built from your job dicts (Section 8.2b) |
| `cluster_mapping` | `{cluster_id: [trade_id, ...]}` | Built from target attribs per cluster (Section 8.2a) |
| `registry_dir` | `str` | Output directory for models and ensemble registration |
| `artifacts_dir` | `str` | Output directory for plots and metrics |
| `cluster_key` / `cluster_key_values` | `list` / `dict` | Optional — only needed if you want routing for new trades later |
| `tags` | `list[str]` | Optional — e.g. `["production"]` for easy retrieval |

**Internal flow:**

```
EnsembleTrainPipeline.run()
│
├── For each cluster_id (sorted):
│   │
│   ├── train_single_member(config, cluster_id)
│   │   │
│   │   ├── Resolve pipeline class:
│   │   │     config.pipeline_class.get(cluster_id)
│   │   │     or default: HybridGnnRnnTrainPipeline
│   │   │
│   │   ├── Build PipelineConfig from member_configs[cluster_id]:
│   │   │     PipelineConfig(
│   │   │       training_config = member_configs[cid]["training_config"],
│   │   │       data_config     = member_configs[cid]["data_config"],
│   │   │       model_config    = member_configs[cid]["model_config"],
│   │   │       registry_dir    = config.registry_dir,
│   │   │       metadata        = {cluster_id, trade_ids, tags}
│   │   │     )
│   │   │
│   │   ├── pipeline_cls(member_config).run()
│   │   │     → Reads data from paths in data_config
│   │   │     → Builds graph, encodes features, creates train/val/test datasets
│   │   │     → Trains model (epochs, early stopping, etc.)
│   │   │     → Registers model in ModelRegistry → saves model.pt, graph_builder.pkl,
│   │   │       encoder.pkl, datasets/test.pt, etc. to registry_dir/{version}/
│   │   │     → Returns TrainingResult
│   │   │
│   │   └── Returns {"result": TrainingResult, "version": "v_20260324_..."}
│   │
│   └── Logs: member trained, version, best_val_loss
│
├── EnsembleBuilder.build(config, member_versions)
│     → Loads all member nn.Modules from ModelRegistry
│     → Validates trade coverage (no overlaps for concat, no missing trades)
│     → Creates EnsembleModel + TradeRouter (validates only, not persisted)
│
└── EnsembleRegistry.register(config, member_versions, member_summary)
      → Generates ensemble version string (e.g. "ens_20260324_143022_a1b2c3")
      → Saves to registry_dir/ensemble/{version}/:
          ensemble_config.json
          member_versions.json     {cluster_id: member_version}
          trade_cluster_map.json   {trade_id: cluster_id}
          member_summary.json      {cluster_id: {n_trades, best_val_loss, ...}}
      → Tags as "latest" + any custom tags
```

**Output:**

```python
{
    "ensemble_version": "ens_20260324_143022_a1b2c3",
    "member_versions": {"cluster_0": "v_20260324_...", "cluster_1": "v_20260324_...", ...},
    "member_results": {"cluster_0": TrainingResult(...), "cluster_1": TrainingResult(...), ...},
}
```

### 9.2 Ensemble Evaluation

**Inputs required:**

| Input | Type | Where it comes from |
|-------|------|---------------------|
| `ensemble_config` | `EnsembleConfig` | Same config used for training (needs `registry_dir`, `artifacts_dir`) |
| `ensemble_version` | `str` | From training output, or `"latest"`, or a tag like `"production"` |

Everything else is loaded from the registry. No raw data files needed — evaluation
uses the cached `test.pt` datasets saved during training.

**Internal flow:**

```
EnsembleEvalPipeline.run()
│
├── EnsembleRegistry.load(ensemble_version)
│     → Returns (config, member_versions, resolved_version)
│
├── EnsembleBuilder.build(config, member_versions)
│     → Loads all member nn.Modules from ModelRegistry
│     → Creates EnsembleModel with TradeRouter + cluster_trade_indices
│
├── For each cluster_id:
│   │
│   ├── evaluate_single_member(cid, model, registry_dir, member_version)
│   │   │
│   │   ├── Load test dataset:
│   │   │     registry_dir/{member_version}/datasets/test.pt
│   │   │     → torch.load → DataLoader (batch_size=32)
│   │   │
│   │   ├── Evaluator(model).run(test_dataloader)
│   │   │     → Forward pass on test data
│   │   │     → Computes predictions, targets, metrics (MAE, MSE, etc.)
│   │   │
│   │   └── Returns {"predictions": ndarray, "targets": ndarray, "metrics": dict}
│   │
│   └── Stores per_member_preds[cid] and per_member_targets[cid]
│
├── compute_per_member_metrics(per_member_preds, per_member_targets)
│     → {cluster_id: {mae, mse, rmse, n_targets, n_scenarios}}
│
├── Combine predictions using ensemble._combine():
│     → concat_aggregate: scatters each cluster's predictions into correct
│       columns of a full [n_scenarios, n_total_targets] array
│     → Does the same for targets
│
├── compute_ensemble_metrics(combined_preds, combined_targets)
│     → {mae, mse, rmse, max_ae, p95_ae, p99_ae}
│
└── Save artifacts to artifacts_dir/ensemble/{version}/evaluation/:
      ensemble_metrics.json
      per_member_metrics.json
      member_rollup.json
      plots/member_comparison_mae.png
      plots/cluster_performance_heatmap.png
```

**Key dependency:** Each member must have `datasets/test.pt` saved during training.
If a member is missing its test data, that member is skipped with a warning (its
predictions won't be included in the ensemble metrics).

**Output:**

```python
{
    "ensemble_version": "ens_20260324_...",
    "ensemble_metrics": {"mae": 0.0012, "mse": 0.000002, "rmse": 0.0014, ...},
    "per_member_metrics": {"cluster_0": {"mae": ..., "mse": ...}, ...},
    "member_summary": {"mean_mae": ..., "std_mae": ..., "per_member": {...}},
}
```

### 9.3 Ensemble Inference — New Scenarios

This is the "same trades, different market shocks" use case. You have new scenario
PnL data and want to re-predict with the existing models.

**Inputs required:**

| Input | Type | Where it comes from |
|-------|------|---------------------|
| `cluster_pnl_histories` | `{cluster_id: ndarray [n_scenarios, seq_len, n_elementary]}` | Your new scenario data, sliced per cluster |
| `ensemble_config` | `EnsembleConfig` | Same config (needs `registry_dir`) |
| `ensemble_version` | `str` | Which ensemble to use |

Each cluster has **different trades with different elementary PnLs**, so you must
provide separate PnL arrays per cluster. There is no valid "global" PnL for disjoint
clusters.

**Via Session (UI flow):**

```python
session = EnsembleSession(registry_dir, artifacts_dir)
session.load_metadata("production")       # Phase 1: config, member versions
session.load_display_artifacts()           # Phase 2: eval metrics, plot paths
session.load_inference_state()             # Phase 3: models, graph_builders, encoders

result = session.run_inference(
    mode="new_scenarios",
    cluster_pnl_histories=cluster_pnl_histories,
)
```

**Via Pipeline (standalone):**

```python
config.metadata["inference"] = {
    "input_mode": "new_scenarios",
    "cluster_pnl_histories": cluster_pnl_histories,
}
result = EnsembleInferencePipeline(config, ensemble_version="latest").run()
```

**Internal flow (both paths):**

```
run_inference(mode="new_scenarios")
│
├── For each cluster_id:
│   │
│   ├── Load inference context (if not already cached in session):
│   │     registry_dir/{member_version}/graph_builder.pkl
│   │     registry_dir/{member_version}/encoder.pkl
│   │
│   ├── build_static_dict(inference_context, None, data_config)
│   │     → Builds trade feature matrix from cached graph_builder + encoder
│   │     → Builds adjacency / edge_index from cached graph
│   │     → No new trade attributes passed (None) — uses existing trades only
│   │     → Returns: {trade_features, edge_index, node_encodings, ...}
│   │
│   ├── build_model_input_dict(static_dict, cluster_pnl_histories[cid])
│   │     → Combines static features with the new PnL scenario data
│   │     → Returns the 7-key model input dict ready for forward pass
│   │
│   ├── model.forward(inputs)
│   │     → nn.Module forward pass (no grad)
│   │     → Returns predictions: [n_scenarios, n_cluster_targets]
│   │
│   └── Store member_preds[cid]
│
└── concat_aggregate(member_preds, cluster_trade_indices, n_total_targets)
      → Scatters each cluster's predictions into correct columns
      → Returns combined: [n_scenarios, n_total_targets]
```

**No routing involved.** Every cluster runs on its known trades with the new PnL data.
The TradeRouter exists in memory but is never called for this mode.

**Output:**

```python
InferenceResult(
    predictions=ndarray [n_scenarios, n_total_targets],
    n_samples=n_scenarios,
    model_version="ens_20260324_...",
    latency_seconds=2.14,
    metadata={"input_mode": "new_scenarios", ...},
)
```

### 9.4 Ensemble Inference — New Trades

This is the "previously unseen trade needs a prediction" use case. A new trade that
wasn't in any cluster during training needs to be assigned to a cluster and predicted.

**Inputs required:**

| Input | Type | Where it comes from |
|-------|------|---------------------|
| `new_trade_attribs` | `dict` | New trade's attributes: `{ccy, desk, product, ...}` |
| `cluster_pnl_histories` | `{cluster_id: ndarray}` | PnL data per cluster (still needed — models need PnL input) |
| Loaded ensemble (session or pipeline) | | Same as new scenarios |

**Internal flow:**

```
run_inference(mode="new_trades", new_trade_attribs={ccy: "EUR", ...})
│
├── _route_new_trades(new_trade_attribs)
│   │
│   ├── Check if attribs are already keyed by cluster_id
│   │     (i.e. caller already routed) → use as-is
│   │
│   └── Otherwise: TradeRouter.assign_new_trade(trade_attribs)
│         │
│         ├── 1. Key match: check cluster_keys
│         │     e.g. cluster_0 = {ccy: "GBP", desk: "FLOW_RATES"}
│         │     Does trade match? All key fields must match.
│         │
│         ├── 2. Centroid match (if cluster_centroids provided):
│         │     Nearest centroid in feature space.
│         │
│         └── 3. Default cluster (if configured), else raise error
│
│   → Returns {cluster_id: trade_attribs or None} for each cluster
│     (only the assigned cluster gets the attribs; others get None)
│
├── For each cluster_id:
│   │
│   ├── If this is the assigned cluster:
│   │     build_static_dict(context, trade_attribs, data_config)
│   │       → Injects new trade features into this cluster's graph
│   │
│   ├── Otherwise:
│   │     build_static_dict(context, None, data_config)
│   │       → Normal features, no new trade
│   │
│   ├── build_model_input_dict(static_dict, pnl)
│   │
│   └── model.forward(inputs) → predictions
│
└── concat_aggregate → combined predictions
```

**Key difference from new scenarios:** `build_static_dict` receives the `trade_attribs`
for the assigned cluster, which injects the new trade's features into that cluster's
input. All other clusters run normally without modification.

### 9.5 Summary: What Each Stage Requires

| Stage | Required Inputs | Loaded from Registry | Routing Used? |
|-------|----------------|---------------------|---------------|
| **Training** | `member_configs` (data paths, training params, model config per cluster), `cluster_mapping`, `registry_dir` | Nothing (creates registry) | No |
| **Evaluation** | `registry_dir`, `artifacts_dir`, `ensemble_version` | Config, member models, test.pt datasets | No |
| **Inference (new scenarios)** | Per-cluster PnL arrays (`cluster_pnl_histories`) | Config, member models, inference contexts (graph_builder, encoder) | No |
| **Inference (new trades)** | New trade attributes + per-cluster PnL arrays | Same as new scenarios | Yes — `TradeRouter.assign_new_trade()` |

**Training and evaluation never touch routing.** They are orchestrated for-loops over
existing single-cluster pipelines. Routing only activates for `new_trades` inference.

**Session phases map to UI stages:**

| Session Phase | What loads | UI Stage |
|--------------|-----------|----------|
| Phase 1: `load_metadata()` | Config, member_versions, trade_cluster_map, member_summary | Landing page — cluster list, trade counts, version selector |
| Phase 2: `load_display_artifacts()` | Eval metrics JSONs, plot PNG paths, trade universe | Analytics — performance charts, cluster drill-down |
| Phase 3: `load_inference_state()` | Member nn.Modules, graph_builders, encoders, baseline_pnl | Inference — run predictions on new data |
