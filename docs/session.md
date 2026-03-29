# EnsembleSession — Complete Reference

> **Module:** `src.rade_ml_pt.ensemble.session`
>
> The `EnsembleSession` is the single point of contact between the ensemble
> model infrastructure and the Dash UI (or any programmatic consumer).
> It manages a **three-phase lifecycle** that separates cheap file reads from
> expensive model deserialization, enabling fast dashboard startup with
> on-demand inference loading.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Three-Phase Lifecycle](#2-three-phase-lifecycle)
3. [Dataclasses](#3-dataclasses)
4. [Class: EnsembleSession](#4-class-ensemblesession)
   - 4.1 [Constructor](#41-constructor)
   - 4.2 [Phase 1 — Metadata](#42-phase-1--metadata)
   - 4.3 [Phase 2 — Display Artifacts](#43-phase-2--display-artifacts)
   - 4.4 [Phase 3 — Inference State](#44-phase-3--inference-state)
   - 4.5 [Inference Execution](#45-inference-execution)
   - 4.6 [Properties (UI Helpers)](#46-properties-ui-helpers)
   - 4.7 [Data Helpers (Dashboard-Oriented)](#47-data-helpers-dashboard-oriented)
   - 4.8 [Guards](#48-guards)
5. [Dependency: EnsembleConfig](#5-dependency-ensembleconfig)
6. [Dependency: EnsembleRegistry](#6-dependency-ensembleregistry)
7. [Filesystem Layout](#7-filesystem-layout)
8. [Integration with the Dash UI](#8-integration-with-the-dash-ui)
9. [Complete Source Code](#9-complete-source-code)
   - 9.1 [session.py](#91-sessionpy)
   - 9.2 [config.py](#92-configpy)
   - 9.3 [registry.py](#93-registrypy)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EnsembleSession                             │
│                                                                     │
│  Phase 1 (instant)         Phase 2 (fast I/O)    Phase 3 (heavy)   │
│  ┌──────────────────┐     ┌──────────────────┐   ┌──────────────┐  │
│  │ EnsembleConfig   │     │ ClusterDisplay   │   │ nn.Module(s) │  │
│  │ member_versions  │     │ EnsembleDisplay  │   │ InfContext(s) │  │
│  │ trade_cluster_map│     │ PredictionStore  │   │ EnsembleModel │  │
│  │ member_summary   │     │ market_data      │   │              │  │
│  └──────────────────┘     │ graph_data       │   └──────────────┘  │
│         ▲                 └──────────────────┘          ▲           │
│         │                          ▲                    │           │
│    load_metadata()          load_display_artifacts() load_inference_state()
│                                                                     │
│  Properties: config, member_versions, ensemble_version,             │
│    trade_cluster_map, cluster_attributes, ensemble_display, ...     │
│                                                                     │
│  Data Helpers: build_global_trade_catalogue(),                      │
│    get_prediction_store(split), load_cluster_market_data(cid),      │
│    load_cluster_graph_data(cid)                                     │
│                                                                     │
│  Inference: run_inference(mode, ...)                                │
└─────────────────────────────────────────────────────────────────────┘
```

**Design principles:**

- The session **never mutates** registry or artifacts on disk — read-only consumer.
- Display artifacts are loaded **eagerly** (Phase 2) because they are small JSON/PNG/NPZ files.
- Inference state is loaded **lazily** (Phase 3) because `model.pt` + pickle files are large.
- Inference runs on **copies** of baseline data; cached state stays read-only.
- Parallel model loading via `ThreadPoolExecutor` keeps wall-clock time proportional to the
  slowest cluster, not the sum.

---

## 2. Three-Phase Lifecycle

| Phase | Method | What It Loads | Cost | When Used |
|-------|--------|---------------|------|-----------|
| **1 — Metadata** | `load_metadata(version)` | `EnsembleConfig`, `member_versions`, `trade_cluster_map`, `member_summary` | Instant (~ms) | App startup / landing page |
| **2 — Display** | `load_display_artifacts()` | Per-cluster eval metrics, plot paths, trade universe, target attributes, ensemble-level metrics, manifest, prediction `.npz` files | Fast (~1-5s for 100 clusters) | Full analytics drill-down |
| **3 — Inference** | `load_inference_state()` | Per-cluster `nn.Module`, inference context (graph builder, encoder, scalers), data config, baseline PnL | Heavy (~10-60s) | Only when user clicks "Load Models" in Inference tab |

### Sequence Diagram

```
App Startup
    │
    ├── initialise(registry_dir, artifacts_dir, version)
    │       │
    │       ├── EnsembleSession(registry_dir, artifacts_dir)
    │       ├── session.load_metadata(version)          ← Phase 1
    │       └── session.load_display_artifacts()         ← Phase 2
    │
    ├── Dashboard renders: Overview, Evaluation, Cluster Deep Dive,
    │   Market Data, Trade Graph, Governance tabs
    │   └── All served from Phase 1 + 2 cached state
    │
    └── User clicks "Load Models" on Inference tab
            │
            ├── session.load_inference_state(parallel=True)  ← Phase 3
            └── session.run_inference(mode="new_scenarios")
```

---

## 3. Dataclasses

### 3.1 `ClusterDisplayState`

Per-cluster display artifacts loaded in Phase 2. Contains no model objects.

| Field | Type | Description |
|-------|------|-------------|
| `cluster_id` | `str` | Cluster identifier |
| `version` | `str` | Member registry version string |
| `version_dir` | `str` | Absolute path to the member's registry directory |
| `eval_metrics` | `Dict[str, Any]` | `{split: {metric_name: value}}` — per-split evaluation metrics |
| `plot_paths` | `Dict[str, str]` | `{"split/plot_name": "/abs/path.png"}` — saved evaluation plots |
| `trade_universe` | `Dict[str, Any]` | Contents of `trade_universe.json` (elementary_ids, target_ids) |
| `target_attributes` | `Dict[str, Any]` | Contents of `target_attributes.json` (per-trade attribute arrays) |
| `predictions` | `Dict[str, Optional[np.ndarray]]` | Lazily loaded prediction arrays keyed by `"{split}_predictions"` |

### 3.2 `EnsembleDisplayState`

Ensemble-wide (portfolio-level) evaluation artifacts loaded in Phase 2.

| Field | Type | Description |
|-------|------|-------------|
| `ensemble_metrics` | `Dict[str, Dict[str, Any]]` | `{split: {mae, rmse, ...}}` |
| `member_rollup` | `Dict[str, Dict[str, Any]]` | `{split: rollup_dict}` — aggregated member statistics |
| `per_member_metrics` | `Dict[str, Dict[str, Any]]` | `{split: {cluster_id: {metric: val}}}` |
| `manifest` | `Dict[str, Any]` | `manifest.json` contents: `trade_ids`, `cluster_ids`, `cluster_trade_indices`, `splits_available` |

### 3.3 `GlobalPredictionStore`

Unified prediction/target arrays aligned by global trade order. Built lazily
from per-member `.npz` files via `get_prediction_store(split)`.

| Field | Type | Shape/Description |
|-------|------|-------------------|
| `predictions` | `np.ndarray` | `[n_scenarios, n_total_targets]` |
| `targets` | `np.ndarray` | `[n_scenarios, n_total_targets]` |
| `trade_ids` | `List[str]` | Length = `n_total_targets`, global trade order |
| `cluster_ids` | `List[str]` | Per-target cluster membership |
| `split` | `str` | `"train"`, `"val"`, or `"test"` |

The store enables arbitrary cross-cluster slicing: filter by desk, ccy, product,
or any attribute from the trade catalogue, then index into `store.predictions[:, mask]`
without per-cluster bookkeeping.

### 3.4 `ClusterInferenceState`

Per-cluster model + inference context loaded in Phase 3.

| Field | Type | Description |
|-------|------|-------------|
| `cluster_id` | `str` | Cluster identifier |
| `model` | `Any` (nn.Module) | Loaded model in `eval()` mode |
| `inference_context` | `Dict[str, Any]` | Graph builder, encoder, scalers, etc. |
| `data_config` | `Any` | `HybridGnnRnnDataConfig` (or dict) |
| `baseline_pnl` | `Optional[np.ndarray]` | Elementary PnL from `elementary_pnl.parquet` (for default new_scenarios) |

---

## 4. Class: EnsembleSession

### 4.1 Constructor

```python
session = EnsembleSession(
    registry_dir="/path/to/registry",
    artifacts_dir="/path/to/artifacts",   # optional
    max_workers=4,                         # thread pool size for Phase 3
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registry_dir` | `str \| Path` | — | Root directory for model and ensemble registries |
| `artifacts_dir` | `str \| Path \| None` | `None` | Root for evaluation artifacts. If `None`, Phase 2 display loading skips artifact-dependent data |
| `max_workers` | `int` | `4` | Thread pool size for parallel Phase 3 model loading |

**Internal state initialised at construction:**

| Attribute | Phase | Description |
|-----------|-------|-------------|
| `_ens_registry` | — | `EnsembleRegistry` instance for version resolution |
| `_config` | 1 | `EnsembleConfig` (set by `load_metadata`) |
| `_member_versions` | 1 | `{cluster_id: version_string}` |
| `_ensemble_version` | 1 | Resolved ensemble version string |
| `_trade_cluster_map` | 1 | `{trade_id: cluster_id}` |
| `_member_summary` | 1 | `{cluster_id: {n_trades, metrics...}}` |
| `_display` | 2 | `{cluster_id: ClusterDisplayState}` |
| `_ensemble_display` | 2 | `EnsembleDisplayState` |
| `_prediction_stores` | 2b | `{split: GlobalPredictionStore}` (lazy) |
| `_market_data_cache` | 2c | `{cluster_id: {asset: {rf: ndarray}}}` (lazy) |
| `_graph_data_cache` | 2c | `{cluster_id: graph_data_dict}` (lazy) |
| `_inference` | 3 | `{cluster_id: ClusterInferenceState}` |
| `_ensemble_model` | 3 | Assembled `EnsembleModel` |

---

### 4.2 Phase 1 — Metadata

#### `load_metadata(version_or_tag: str = "latest") -> None`

Loads ensemble metadata from the registry. After this call the session knows:
cluster IDs, member versions, trade-cluster map, and per-cluster summary stats.

**Flow:**

```
load_metadata("production")
    │
    ├── EnsembleRegistry.load("production")
    │     └── Resolves tag → version via index.json
    │     └── Returns (EnsembleConfig, member_versions, version)
    │
    ├── Sets: _config, _member_versions, _ensemble_version
    │
    ├── Loads trade_cluster_map.json (if exists)
    │     └── {trade_id: cluster_id}
    │
    └── Loads member_summary.json (if exists)
          └── {cluster_id: {n_trades, metrics...}}
```

**What the dashboard uses from Phase 1:**

- `config.n_members`, `config.all_trade_ids` → navbar stats
- `config.cluster_ids` → all dropdown selectors
- `ensemble_version` → version dropdown
- `cluster_attributes` → dropdown labels (ccy, desk, product)

---

### 4.3 Phase 2 — Display Artifacts

#### `load_display_artifacts() -> None`

Iterates over all cluster IDs and loads pre-saved evaluation artifacts.
Also loads ensemble-level display state (portfolio metrics, manifest).

**Per-cluster loading (`_load_cluster_display`):**

```
_load_cluster_display(cluster_id)
    │
    ├── Resolve version_dir from member_versions[cluster_id]
    │
    ├── eval_metrics["summary"] ← member_summary[cluster_id]  (Phase 1 fallback)
    │
    ├── For each split in (train, val, test):
    │     └── eval_metrics[split] ← per_member_metrics{suffix}.json → [cluster_id]
    │
    ├── trade_universe ← version_dir / trade_universe.json
    │
    ├── target_attributes ← version_dir / target_attributes.json
    │
    ├── plot_paths ← scan artifacts_dir/ensemble/{version}/evaluation/plots/{split}/*.png
    │
    └── eval_metrics["data_config"] ← version_dir / data_config.json
```

**Ensemble-level loading (`_load_ensemble_display`):**

```
_load_ensemble_display()
    │
    ├── manifest ← evaluation/manifest.json
    │     └── {trade_ids, cluster_ids, cluster_trade_indices, splits_available}
    │
    └── For each split in manifest.splits_available:
          ├── ensemble_metrics[split] ← ensemble_metrics{suffix}.json
          ├── member_rollup[split]    ← member_rollup{suffix}.json
          └── per_member_metrics[split] ← per_member_metrics{suffix}.json
```

#### `load_cluster_display(cluster_id: str) -> ClusterDisplayState`

Public lazy accessor for single-cluster display state. Useful for on-demand
drill-down (e.g., Cluster Deep Dive tab). Caches after first load.

#### `load_cluster_predictions(cluster_id, split="test") -> Optional[np.ndarray]`

Loads saved prediction arrays for one cluster + split from `.npz` files.
Checks `artifacts_dir/ensemble/{version}/members/{cid}/predictions/{split}.npz`
first, falls back to `version_dir/datasets/{split}.pt`. Caches per-key.

---

### 4.4 Phase 3 — Inference State

#### `load_inference_state(cluster_ids=None, parallel=True) -> None`

Loads models + inference contexts for the specified clusters (or all clusters
if `None`). Skips clusters already loaded. Uses `ThreadPoolExecutor` for
parallel loading.

**Per-cluster loading (`_load_cluster_inference`):**

```
_load_cluster_inference(cluster_id)
    │
    ├── ModelRegistry(registry_dir).load(version)
    │     └── Returns nn.Module → model.eval()
    │
    ├── load_inference_context_from_dir(version_dir)  [try/except]
    │     └── Loads graph_builder, encoder, scalers, data_config, etc.
    │     └── Falls back gracefully if not a hybrid GNN-RNN member
    │
    ├── HybridGnnRnnDataConfig.from_json(data_config.json)  [if exists]
    │
    └── pd.read_parquet(elementary_pnl.parquet)  [if exists]
          └── Baseline PnL for default new_scenarios inference
```

**After all clusters are loaded:**

```
_build_ensemble_model()
    │
    ├── Collects {cid: model} from all ClusterInferenceState
    │
    ├── Builds TradeRouter from config.cluster_mapping + cluster_keys
    │
    ├── Builds cluster_trade_indices via EnsembleBuilder
    │
    └── Assembles EnsembleModel(members, router, aggregation, weights, ...)
```

---

### 4.5 Inference Execution

#### `run_inference(mode, cluster_pnl_histories=None, new_trade_attribs=None, member_inputs=None) -> Dict`

Runs inference using the cached Phase 3 state.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | `str` | `"new_scenarios"` or `"new_trades"` |
| `cluster_pnl_histories` | `Dict[str, np.ndarray] \| None` | `{cid: pnl_array}` — overrides baseline PnL |
| `new_trade_attribs` | `Dict[str, Any] \| None` | For `new_trades` mode (not yet implemented) |
| `member_inputs` | `Dict[str, Any] \| None` | Pre-built model inputs (bypasses all data preparation) |

**Returns:**

```python
{
    "predictions": np.ndarray,        # [n_scenarios, n_targets]
    "per_member": {                   # per-cluster metadata
        "cluster_id": {
            "n_trades": int,
            "n_scenarios": int,
            "has_new_trades": bool,
        }
    },
    "metadata": {
        "ensemble_version": str,
        "mode": str,
        "n_scenarios": int,
        "n_targets": int,
    },
}
```

**`new_scenarios` flow (the primary path):**

```
run_inference(mode="new_scenarios")
    │
    ├── _require_inference()  ← guards: metadata + all models loaded
    │
    └── For each cluster_id in config.cluster_ids:
          │
          ├── Get ClusterInferenceState.inference_context
          │     └── Convert to InferenceContext if raw dict
          │
          ├── HybridGnnRnnInferencePipeline._inject_unchanged_inputs(ctx)
          │     └── Returns InferenceInputData with static graph, encoder, etc.
          │
          ├── Resolve elementary PnL:
          │     └── caller-provided > baseline_pnl from parquet
          │     └── Wrap np.ndarray → pd.DataFrame using ctx column names
          │
          ├── _standardise_pnl(pnl_df, scaler)
          │     └── Apply saved scaler transform (same as training)
          │
          ├── build_new_pnl_sequences(pnl, seq_length, n_targets)
          │
          ├── build_model_inputs(elem_seq, inputs, seq_length)
          │     └── Returns {"inputs": model-ready tensors}
          │
          └── Collect into member_inputs_built[cid]
    │
    └── combined = ensemble_model.predict(member_inputs_built)
```

**`member_inputs` shortcut:** If `member_inputs` is provided, all data
preparation is bypassed and the dict is passed directly to
`ensemble_model.predict()`. This is the model-agnostic path for callers
who prepare their own inputs.

---

### 4.6 Properties (UI Helpers)

| Property | Return Type | Phase Required | Description |
|----------|-------------|----------------|-------------|
| `is_metadata_loaded` | `bool` | — | True after `load_metadata()` |
| `is_display_loaded` | `bool` | — | True after `load_display_artifacts()` for all clusters |
| `inference_ready_clusters` | `List[str]` | — | Sorted list of clusters with loaded inference state |
| `all_inference_ready` | `bool` | — | True when all clusters have inference state |
| `config` | `Optional[EnsembleConfig]` | 1 | The loaded ensemble configuration |
| `member_versions` | `Optional[Dict[str, str]]` | 1 | `{cluster_id: version_string}` |
| `ensemble_version` | `Optional[str]` | 1 | Resolved ensemble version string |
| `trade_cluster_map` | `Optional[Dict[str, str]]` | 1 | `{trade_id: cluster_id}` |
| `member_summary` | `Optional[Dict[str, Dict]]` | 1 | `{cluster_id: summary_dict}` |
| `display` | `Dict[str, ClusterDisplayState]` | 2 | Copy of all per-cluster display states |
| `ensemble_display` | `Optional[EnsembleDisplayState]` | 2 | Ensemble-level display state |
| `ensemble_model` | `Any` (EnsembleModel) | 3 | The assembled ensemble model |
| `cluster_attributes` | `Dict[str, Dict]` | 1 | `{cid: {ccy, desk, product}}` from config |

---

### 4.7 Data Helpers (Dashboard-Oriented)

These methods are specifically designed for the Dash UI data layer.

#### `build_global_trade_catalogue() -> pd.DataFrame`

Builds a flat DataFrame of every target trade across all clusters,
joining per-trade attributes (from `target_attributes.json`) with
cluster-level attributes (ccy, desk, product from config).

**Columns:** `trade_id`, `cluster_id`, `ccy`, `desk`, `product_type`, plus
any per-trade attributes from the member's `target_attributes.json`.

**Used by:** `data/trade_catalogue.py` singleton cache → all Evaluation sub-tab
filter bars, overview table, cluster deep dive catalogue filtering.

#### `get_prediction_store(split: str = "test") -> Optional[GlobalPredictionStore]`

Returns a unified prediction/target store for the given split. Built lazily
from per-member `.npz` files saved by the eval pipeline. The store slots
each member's columns into the global trade order from `manifest.json`.

**Used by:** `data/prediction_store.py` → Evaluation callbacks, Overview scatter,
Cluster Deep Dive scatter/residuals, Node Analytics degree-vs-MAE.

#### `load_cluster_market_data(cluster_id: str) -> Dict[str, Any]`

Loads asset portfolio / risk-factor shock data from `cluster_assets.joblib`.
Returns `{asset_name: {rf_name: np.ndarray}}`. Cached after first load.

**Used by:** `data/market_data_loader.py` → Market Data tab (RF Summary,
Shock Explorer, Scenario Heatmap, Distribution).

#### `load_cluster_graph_data(cluster_id: str) -> Dict[str, Any]`

Loads graph adjacency and encoder feature data from `graph_results.joblib`
and `encoder_results.joblib`. Also includes `trade_universe` from display state.

**Used by:** `data/graph_data_loader.py` → Trade Graph tab (Graph View,
Adjacency Analysis, Node Analytics, Cross-Cluster).

---

### 4.8 Guards

| Method | Raises | Condition |
|--------|--------|-----------|
| `_require_metadata()` | `RuntimeError` | `_config is None` |
| `_require_inference()` | `RuntimeError` | metadata not loaded, or not all clusters have inference state, or ensemble model not assembled |

---

## 5. Dependency: EnsembleConfig

`EnsembleConfig` is the single configuration object consumed by all ensemble
pipelines and the session. Key attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `member_configs` | `Dict[str, Dict]` | `{cluster_id: PipelineConfig_dict}` |
| `cluster_mapping` | `Dict[str, List[str]]` | `{cluster_id: [trade_id, ...]}` |
| `cluster_keys` | `Optional[Dict]` | Direct `{cid: {attr: val}}` for routing |
| `cluster_key` / `cluster_key_values` | `Optional` | Alternative routing spec (shared keys + per-cluster values) |
| `aggregation` | `str` | `"concat"` or `"weighted_mean"` |
| `weights` | `Optional[Dict]` | Member weights (for `weighted_mean`) |
| `execution_strategy` | `str` | `"sequential"` (default) |
| `registry_dir` / `artifacts_dir` | `Optional[str]` | Infrastructure paths |
| `metadata` | `Dict` | Arbitrary key-value pairs |

**Derived properties:**

| Property | Description |
|----------|-------------|
| `cluster_ids` | `sorted(cluster_mapping.keys())` |
| `n_members` | `len(cluster_mapping)` |
| `all_trade_ids` | Flat list of all trade IDs across clusters |
| `get_cluster_keys_for_router()` | Builds `{cid: {attr: val}}` from either `cluster_keys` or `cluster_key` + `cluster_key_values` |

**Serialisation:** `to_dict()`, `from_dict()`, `to_json()`, `from_json()`, `from_yaml()`.

---

## 6. Dependency: EnsembleRegistry

`EnsembleRegistry` manages versioned ensemble bundles on the local filesystem.

### Storage Layout

```
registry_dir/
  ensemble/
    ens_20260324_143055_d4e5f6/
      ensemble_config.json          # Full EnsembleConfig
      member_versions.json          # {cluster_id: member_version}
      trade_cluster_map.json        # {trade_id: cluster_id}
      member_summary.json           # {cluster_id: {n_trades, metrics...}}
    index.json                      # {tag: version} mapping
```

### Key Methods

| Method | Description |
|--------|-------------|
| `register(config, member_versions, member_summary, tags)` | Register a new ensemble version. Generates timestamped version ID, saves all JSON files, updates tags + "latest" in index. |
| `load(version_or_tag)` | Load ensemble config + member versions. Resolves tags via `index.json`. Returns `(EnsembleConfig, Dict[str, str], str)`. |
| `get_metadata(version_or_tag)` | Load member_summary and trade_cluster_map for a version. |
| `tag(version, tag)` | Add a tag to an existing version. |
| `list_versions()` | List all registered versions with basic metadata (n_members, n_trades, aggregation). |

### Version ID Format

```
ens_{YYYYMMDD}_{HHMMSS}_{6-char-md5-hash}
```

Example: `ens_20260324_143055_d4e5f6`

### Tag Resolution

Tags are stored in `index.json` as `{tag_name: version_string}`. The `"latest"`
tag is always updated on registration. Custom tags (e.g., `"production"`,
`"ensemble"`, `"hybrid_gnn_rnn"`) are set via the `tags` parameter during
registration or the `tag()` method after the fact.

---

## 7. Filesystem Layout

The session reads from two directory trees: **registry** and **artifacts**.

### Registry Directory (read by Phase 1 + Phase 3)

```
registry_dir/
  ensemble/                                    ← EnsembleRegistry root
    index.json                                 ← tag → version mapping
    ens_20260324_143055_d4e5f6/                ← ensemble version dir
      ensemble_config.json
      member_versions.json
      trade_cluster_map.json
      member_summary.json
  {member_version}/                            ← ModelRegistry member dirs
    model.pt                                   ← Phase 3: nn.Module
    data_config.json                           ← Phase 2/3: HybridGnnRnnDataConfig
    trade_universe.json                        ← Phase 2: elementary_ids, target_ids
    target_attributes.json                     ← Phase 2: per-trade attributes
    cluster_assets.joblib                      ← Phase 2c: market data
    graph_results.joblib                       ← Phase 2c: graph adjacency
    encoder_results.joblib                     ← Phase 2c: encoder features
    elementary_pnl.parquet                     ← Phase 3: baseline PnL
    inference_context.joblib                   ← Phase 3: scalers, graph_builder, etc.
```

### Artifacts Directory (read by Phase 2)

```
artifacts_dir/
  ensemble/
    ens_20260324_143055_d4e5f6/                ← ensemble version
      evaluation/
        manifest.json                          ← global trade order, splits_available
        ensemble_metrics.json                  ← portfolio-level test metrics
        ensemble_metrics_train.json
        ensemble_metrics_val.json
        member_rollup.json
        member_rollup_train.json
        member_rollup_val.json
        per_member_metrics.json                ← {cid: {metric: val}} for test
        per_member_metrics_train.json
        per_member_metrics_val.json
        plots/
          test/*.png
          val/*.png
          train/*.png
        members/
          cluster_0/
            predictions/
              test.npz                         ← {predictions, targets} arrays
              val.npz
              train.npz
          cluster_1/
            predictions/
              test.npz
              ...
        combined/
          test.npz
          val.npz
          train.npz
```

### File Naming Convention for Splits

Test split files use **no suffix** (e.g., `ensemble_metrics.json`).
Train and val splits use `_{split}` suffix (e.g., `ensemble_metrics_train.json`).

---

## 8. Integration with the Dash UI

The Dash dashboard (`src/ui/apps/ensemble_analytics/`) accesses the session
through a singleton data layer:

```
session_manager.py          ─── initialise() / get_session() / reload()
    │
    ├── trade_catalogue.py  ─── get_trade_catalogue()  → session.build_global_trade_catalogue()
    ├── prediction_store.py ─── get_prediction_store()  → session.get_prediction_store(split)
    ├── market_data_loader.py── get_market_data(cid)   → session.load_cluster_market_data(cid)
    └── graph_data_loader.py ── get_graph_data(cid)    → session.load_cluster_graph_data(cid)
```

### Session API → Dashboard Tab Mapping

| Dashboard Tab | Session API Used |
|---------------|-----------------|
| **Overview** | `ensemble_display.ensemble_metrics[split]`, `ensemble_display.per_member_metrics[split]`, `get_prediction_store(split)`, `cluster_attributes`, `config.cluster_ids`, `config.cluster_mapping` |
| **Evaluation** (all sub-tabs) | `get_prediction_store(split)`, `build_global_trade_catalogue()` (for filtering by desk/product/ccy) |
| **Cluster Deep Dive** | `load_cluster_display(cid)`, `ensemble_display.per_member_metrics`, `get_prediction_store(split)`, `build_global_trade_catalogue()`, `cluster_attributes` |
| **Market Data** | `load_cluster_market_data(cid)`, `config.cluster_ids`, `cluster_attributes` |
| **Trade Graph** | `load_cluster_graph_data(cid)`, `get_prediction_store("test")`, `build_global_trade_catalogue()`, `config.cluster_ids` |
| **Inference** | `all_inference_ready`, `inference_ready_clusters`, `load_inference_state()`, `run_inference()`, `get_prediction_store("test")` (baseline comparison) |
| **Governance** | `ensemble_version`, `config`, `ensemble_display.manifest`, `member_versions`, `trade_cluster_map`, `_ens_registry.list_versions()`, `artifacts_dir` |

### Caching Strategy

| Cache | Lifetime | Invalidation |
|-------|----------|--------------|
| `_display` (per-cluster) | Session | `reload()` → `initialise()` |
| `_ensemble_display` | Session | `reload()` → `initialise()` |
| `_prediction_stores` (per-split) | Session | `reload()` → `initialise()` |
| `_market_data_cache` (per-cluster) | Session | `reload()` → `initialise()` |
| `_graph_data_cache` (per-cluster) | Session | `reload()` → `initialise()` |
| `_inference` (per-cluster) | Session | Not invalidated (reload creates new session) |
| Trade catalogue (module-level) | Session | `trade_catalogue.invalidate()` on reload |

---

## 9. Complete Source Code

### 9.1 `session.py`

`src/rade_ml_pt/ensemble/session.py`

```python
"""
Ensemble session: three-phase lifecycle for UI and programmatic use.

Separates **display state** (metrics, plots, predictions — fast file reads)
from **inference state** (nn.Modules, graph builders, encoders — heavy
deserialization, loaded on demand).

Phases
------
1. **Metadata** (instant) — ensemble config, member versions, trade-cluster
   map, member summary.  Enough for landing page rendering.
2. **Display artifacts** (fast) — per-cluster eval metrics, plot file paths,
   prediction arrays.  Enough for full analytics drill-down.  No model
   loading; just reading JSON/PNG/NPZ from the artifacts directory.
3. **Inference state** (on demand) — per-cluster nn.Module + inference
   context (graph_builder, encoder, scalers).  Loaded lazily when the user
   first runs inference, with optional parallel loading across clusters.

Design
------
- The session never mutates registry or artifacts on disk.
- Inference runs on **copies** of baseline data; cached state stays read-only.
- Display artifacts are loaded eagerly (Phase 2) because they're small files.
- Inference state is loaded lazily (Phase 3) because model.pt + pickle files
  are large; parallel loading with ``ThreadPoolExecutor`` keeps wall-clock
  time proportional to the slowest cluster, not the sum.

Usage
-----
::

    session = EnsembleSession(registry_dir, artifacts_dir)
    session.load_metadata("production")          # Phase 1: instant
    session.load_display_artifacts()             # Phase 2: fast file reads
    session.load_inference_state()               # Phase 3: parallel model load
    result = session.run_inference(mode, ...)    # Uses cached state
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Per-cluster display state (Phase 2) — lightweight file reads
# ------------------------------------------------------------------

@dataclass
class ClusterDisplayState:
    """Pre-saved evaluation artifacts for one cluster (no model objects)."""

    cluster_id: str
    version: str
    version_dir: str

    # {split: {metric_name: value}} — per-split eval metrics
    eval_metrics: Dict[str, Any] = field(default_factory=dict)

    # {"split/plot_name": "/abs/path.png"}
    plot_paths: Dict[str, str] = field(default_factory=dict)

    # from trade_universe.json
    trade_universe: Dict[str, Any] = field(default_factory=dict)

    # from target_attributes.json — per-trade attribute arrays
    target_attributes: Dict[str, Any] = field(default_factory=dict)

    # predictions arrays (loaded from .npz on demand)
    predictions: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)


@dataclass
class EnsembleDisplayState:
    """Ensemble-wide (portfolio-level) evaluation artifacts."""

    # {split: {mae, rmse, ...}}
    ensemble_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # {split: rollup dict}
    member_rollup: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # {split: {cluster_id: metrics}}
    per_member_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # manifest.json contents (trade_ids, cluster_ids, cluster_trade_indices, splits_available)
    manifest: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Global prediction store — unified arrays for cross-cluster slicing
# ------------------------------------------------------------------

@dataclass
class GlobalPredictionStore:
    """Unified prediction/target arrays aligned by global trade order.

    Built lazily from per-member ``.npz`` files.  All arrays share the
    same ``[n_scenarios, n_total_targets]`` shape so callers can slice
    by arbitrary trade subsets without per-cluster bookkeeping.
    """

    predictions: np.ndarray       # [n_scenarios, n_total_targets]
    targets: np.ndarray           # [n_scenarios, n_total_targets]
    trade_ids: List[str]          # length = n_total_targets
    cluster_ids: List[str]        # per-target cluster membership
    split: str                    # "train", "val", or "test"


# ------------------------------------------------------------------
# Per-cluster inference state (Phase 3) — heavy model objects
# ------------------------------------------------------------------

@dataclass
class ClusterInferenceState:
    """Loaded model + inference context for one cluster."""

    cluster_id: str
    model: Any = None                        # nn.Module (eval mode)
    inference_context: Dict[str, Any] = field(default_factory=dict)
    data_config: Any = None                  # HybridGnnRnnDataConfig (or dict)
    baseline_pnl: Optional[np.ndarray] = None


# ------------------------------------------------------------------
# Session
# ------------------------------------------------------------------

class EnsembleSession:
    """
    Manages the full lifecycle of an ensemble for the UI.

    Parameters
    ----------
    registry_dir : str or Path
        Root directory for model and ensemble registries.
    artifacts_dir : str or Path or None
        Root directory for evaluation artifacts (plots, metrics, predictions).
        If None, display artifact loading is skipped.
    max_workers : int
        Thread pool size for parallel model loading in Phase 3.
    """

    def __init__(
        self,
        registry_dir: Union[str, Path],
        artifacts_dir: Optional[Union[str, Path]] = None,
        max_workers: int = 4,
    ) -> None:
        self.registry_dir = Path(registry_dir)
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.max_workers = max_workers

        self._ens_registry = EnsembleRegistry(self.registry_dir)

        # Phase 1: metadata
        self._config: Optional[EnsembleConfig] = None
        self._member_versions: Optional[Dict[str, str]] = None
        self._ensemble_version: Optional[str] = None
        self._trade_cluster_map: Optional[Dict[str, str]] = None
        self._member_summary: Optional[Dict[str, Dict[str, Any]]] = None

        # Phase 2: display artifacts per cluster + ensemble-level
        self._display: Dict[str, ClusterDisplayState] = {}
        self._ensemble_display: Optional[EnsembleDisplayState] = None

        # Phase 2b: cached global prediction stores per split
        self._prediction_stores: Dict[str, GlobalPredictionStore] = {}

        # Phase 2c: cached market / graph data per cluster
        self._market_data_cache: Dict[str, Dict[str, Any]] = {}
        self._graph_data_cache: Dict[str, Dict[str, Any]] = {}

        # Phase 3: inference state per cluster
        self._inference: Dict[str, ClusterInferenceState] = {}

        # Assembled ensemble (built when all inference states are loaded)
        self._ensemble_model: Any = None  # EnsembleModel

    # ==================================================================
    # Phase 1 — Metadata (instant)
    # ==================================================================

    def load_metadata(self, version_or_tag: str = "latest") -> None:
        """
        Load ensemble metadata from the registry.

        After this call the session knows: cluster IDs, member versions,
        trade-cluster map, and per-cluster summary stats.  Enough for
        landing page rendering.
        """
        config, member_versions, version = self._ens_registry.load(version_or_tag)
        self._config = config
        self._member_versions = member_versions
        self._ensemble_version = version

        # trade_cluster_map and member_summary from the ensemble version dir
        ens_version_dir = self._ens_registry.root_dir / version

        tcm_path = ens_version_dir / "trade_cluster_map.json"
        if tcm_path.exists():
            with open(tcm_path, "r") as f:
                self._trade_cluster_map = json.load(f)

        summary_path = ens_version_dir / "member_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                self._member_summary = json.load(f)

        logger.info(
            "Session Phase 1: loaded metadata for ensemble '%s' (%d clusters)",
            version, config.n_members,
        )

    # ==================================================================
    # Phase 2 — Display artifacts (fast file reads)
    # ==================================================================

    def load_display_artifacts(self) -> None:
        """
        Load pre-saved evaluation artifacts for all clusters.

        Reads JSON metrics, plot file paths, trade universe, target
        attributes, and optionally prediction arrays.  Also loads
        ensemble-level eval metrics and the manifest.  No model
        objects are loaded.
        """
        self._require_metadata()
        for cid in self._config.cluster_ids:
            self._display[cid] = self._load_cluster_display(cid)

        self._ensemble_display = self._load_ensemble_display()

        logger.info(
            "Session Phase 2: loaded display artifacts for %d clusters",
            len(self._display),
        )

    def load_cluster_display(self, cluster_id: str) -> ClusterDisplayState:
        """Load display artifacts for a single cluster (for lazy/on-demand drill-down)."""
        self._require_metadata()
        if cluster_id not in self._display:
            self._display[cluster_id] = self._load_cluster_display(cluster_id)
        return self._display[cluster_id]

    def _load_cluster_display(self, cluster_id: str) -> ClusterDisplayState:
        version = self._member_versions[cluster_id]
        version_dir = self.registry_dir / version

        state = ClusterDisplayState(
            cluster_id=cluster_id,
            version=version,
            version_dir=str(version_dir),
        )

        # Baseline eval metrics from member_summary (Phase 1 fallback)
        if self._member_summary and cluster_id in self._member_summary:
            state.eval_metrics["summary"] = dict(self._member_summary[cluster_id])

        # Per-split eval metrics from the evaluation directory
        if self.artifacts_dir is not None:
            eval_dir = (
                self.artifacts_dir / "ensemble" / self._ensemble_version / "evaluation"
            )
            for split in ("train", "val", "test"):
                suffix = "" if split == "test" else f"_{split}"
                pm_path = eval_dir / f"per_member_metrics{suffix}.json"
                if pm_path.exists():
                    try:
                        with open(pm_path, "r") as f:
                            all_pm = json.load(f)
                        if cluster_id in all_pm:
                            state.eval_metrics[split] = all_pm[cluster_id]
                    except Exception:
                        pass

        # trade universe
        universe_path = version_dir / "trade_universe.json"
        if universe_path.exists():
            with open(universe_path, "r") as f:
                state.trade_universe = json.load(f)

        # target attributes (for global trade catalogue)
        ta_path = version_dir / "target_attributes.json"
        if ta_path.exists():
            with open(ta_path, "r") as f:
                state.target_attributes = json.load(f)

        # plot file paths: scan per-split plot directories written by eval pipeline
        if self.artifacts_dir is not None:
            for split in ("train", "val", "test"):
                plots_dir = (
                    self.artifacts_dir / "ensemble" / self._ensemble_version
                    / "evaluation" / "plots" / split
                )
                if plots_dir.exists():
                    for p in plots_dir.glob("*.png"):
                        state.plot_paths[f"{split}/{p.stem}"] = str(p)

        # data_config.json (lightweight, useful for display)
        dc_path = version_dir / "data_config.json"
        if dc_path.exists():
            with open(dc_path, "r") as f:
                state.eval_metrics["data_config"] = json.load(f)

        return state

    def _load_ensemble_display(self) -> Optional[EnsembleDisplayState]:
        """Load ensemble-wide (portfolio-level) evaluation artifacts."""
        if self.artifacts_dir is None:
            return None

        eval_dir = (
            self.artifacts_dir / "ensemble" / self._ensemble_version / "evaluation"
        )
        if not eval_dir.exists():
            return None

        state = EnsembleDisplayState()

        # Manifest
        manifest_path = eval_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                state.manifest = json.load(f)

        # Per-split: ensemble metrics, rollup, per-member metrics
        splits = state.manifest.get("splits_available", ["test"])
        for split in splits:
            suffix = "" if split == "test" else f"_{split}"

            em_path = eval_dir / f"ensemble_metrics{suffix}.json"
            if em_path.exists():
                with open(em_path, "r") as f:
                    state.ensemble_metrics[split] = json.load(f)

            rollup_path = eval_dir / f"member_rollup{suffix}.json"
            if rollup_path.exists():
                with open(rollup_path, "r") as f:
                    state.member_rollup[split] = json.load(f)

            pm_path = eval_dir / f"per_member_metrics{suffix}.json"
            if pm_path.exists():
                with open(pm_path, "r") as f:
                    state.per_member_metrics[split] = json.load(f)

        return state

    def load_cluster_predictions(
        self, cluster_id: str, split: str = "test",
    ) -> Optional[np.ndarray]:
        """
        Load saved prediction arrays for one cluster + split.

        Only loads the .npz from disk when first requested; caches for
        subsequent calls.
        """
        display = self.load_cluster_display(cluster_id)
        cache_key = f"{split}_predictions"
        if cache_key in display.predictions:
            return display.predictions[cache_key]

        version_dir = Path(display.version_dir)

        # try artifacts_dir first (from eval pipeline)
        if self.artifacts_dir is not None:
            npz_path = (
                self.artifacts_dir / "ensemble" / self._ensemble_version
                / "members" / cluster_id / "predictions" / f"{split}.npz"
            )
            if npz_path.exists():
                data = np.load(str(npz_path))
                arr = data.get("predictions", data.get("arr_0"))
                display.predictions[cache_key] = arr
                return arr

        # fallback: from registry version_dir/datasets
        ds_path = version_dir / "datasets" / f"{split}.pt"
        if ds_path.exists():
            logger.debug("Predictions .npz not found; raw dataset at %s", ds_path)

        display.predictions[cache_key] = None
        return None

    # ==================================================================
    # Phase 3 — Inference state (on demand, parallel)
    # ==================================================================

    def load_inference_state(
        self,
        cluster_ids: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> None:
        """
        Load models + inference contexts for the specified clusters.

        Parameters
        ----------
        cluster_ids : list or None
            Clusters to load.  None = all clusters.
        parallel : bool
            If True, load clusters concurrently via ThreadPoolExecutor.
        """
        self._require_metadata()
        targets = cluster_ids or list(self._config.cluster_ids)
        targets = [cid for cid in targets if cid not in self._inference]

        if not targets:
            logger.info("Session Phase 3: all requested clusters already loaded")
            return

        logger.info(
            "Session Phase 3: loading inference state for %d clusters%s",
            len(targets), " (parallel)" if parallel else "",
        )

        if parallel and len(targets) > 1:
            self._load_parallel(targets)
        else:
            for cid in targets:
                self._inference[cid] = self._load_cluster_inference(cid)

        # Build the assembled EnsembleModel if all clusters are ready
        if set(self._config.cluster_ids) <= set(self._inference.keys()):
            self._build_ensemble_model()

        logger.info(
            "Session Phase 3: inference ready for %d / %d clusters",
            len(self._inference), self._config.n_members,
        )

    def _load_parallel(self, cluster_ids: List[str]) -> None:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._load_cluster_inference, cid): cid
                for cid in cluster_ids
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    self._inference[cid] = future.result()
                    logger.info("  loaded inference state: %s", cid)
                except Exception:
                    logger.exception("  FAILED to load inference state: %s", cid)

    def _load_cluster_inference(self, cluster_id: str) -> ClusterInferenceState:
        from src.rade_ml_pt.registry.store import ModelRegistry

        version = self._member_versions[cluster_id]
        version_dir = self.registry_dir / version

        # Load model
        model_registry = ModelRegistry(self.registry_dir)
        model, _entry = model_registry.load(version)
        model.eval()

        # Load inference context (graph_builder, encoder, scalers) when present.
        # Keep this optional so non-hybrid members can still be loaded and used
        # with externally prepared member inputs.
        context: Dict[str, Any] = {}
        try:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                load_inference_context_from_dir,
            )
            context = load_inference_context_from_dir(version_dir)
        except Exception:
            logger.debug(
                "No model-specific inference context loaded for cluster '%s' "
                "(version '%s'); external/prebuilt member inputs will be required.",
                cluster_id, version,
            )

        # Data config
        data_config = None
        dc_path = version_dir / "data_config.json"
        if dc_path.exists():
            from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
            data_config = HybridGnnRnnDataConfig.from_json(dc_path)

        # Baseline PnL (optional, for new_scenarios default)
        baseline_pnl = None
        pnl_path = version_dir / "elementary_pnl.parquet"
        if pnl_path.exists():
            import pandas as pd
            baseline_pnl = pd.read_parquet(pnl_path).to_numpy().astype(np.float32)

        return ClusterInferenceState(
            cluster_id=cluster_id,
            model=model,
            inference_context=context,
            data_config=data_config,
            baseline_pnl=baseline_pnl,
        )

    def _build_ensemble_model(self) -> None:
        from src.rade_ml_pt.ensemble.model import EnsembleModel
        from src.rade_ml_pt.ensemble.router import TradeRouter
        from src.rade_ml_pt.ensemble.builder import EnsembleBuilder

        members = {cid: state.model for cid, state in self._inference.items()}
        router = TradeRouter(
            self._config.cluster_mapping,
            cluster_keys=self._config.get_cluster_keys_for_router(),
        )
        cluster_trade_indices = EnsembleBuilder._build_cluster_trade_indices(self._config)

        self._ensemble_model = EnsembleModel(
            members=members,
            router=router,
            aggregation=self._config.aggregation,
            weights=self._config.weights,
            cluster_trade_indices=cluster_trade_indices,
            n_total_targets=len(self._config.all_trade_ids),
        )
        logger.info("Assembled EnsembleModel (%d members)", len(members))

    # ==================================================================
    # Inference
    # ==================================================================

    def run_inference(
        self,
        mode: str = "new_scenarios",
        cluster_pnl_histories: Optional[Dict[str, np.ndarray]] = None,
        new_trade_attribs: Optional[Dict[str, Any]] = None,
        member_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference using cached state.

        Parameters
        ----------
        mode : str
            ``"new_scenarios"`` or ``"new_trades"``.
        cluster_pnl_histories : dict or None
            ``{cluster_id: pnl_array [n_scenarios, seq_len, n_elementary]}``.
            Required for new_scenarios; optional for new_trades (uses baseline).
        new_trade_attribs : dict or None
            For new_trades mode: ``{cluster_id: trade_attribs_dict}`` or a single
            dict that will be routed via the TradeRouter.

        Returns
        -------
        dict
            ``predictions`` (np.ndarray), ``per_member`` (dict), ``metadata`` (dict).
        """
        self._require_inference()

        if mode not in {"new_scenarios", "new_trades"}:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported modes: 'new_scenarios', 'new_trades'."
            )

        # Model-agnostic path: caller provides fully prepared per-member inputs.
        if member_inputs:
            combined = self._ensemble_model.predict(member_inputs)
            return {
                "predictions": combined,
                "per_member": {
                    cid: {"n_trades": len(self._config.cluster_mapping.get(cid, []))}
                    for cid in member_inputs.keys()
                },
                "metadata": {
                    "ensemble_version": self._ensemble_version,
                    "mode": mode,
                    "n_scenarios": combined.shape[0],
                    "n_targets": combined.shape[1] if combined.ndim > 1 else 1,
                },
            }

        if mode == "new_trades":
            raise NotImplementedError(
                "new_trades inference is not yet supported. The underlying "
                "HybridGnnRnnInferencePipeline does not implement "
                "_prepare_new_trade_inputs yet."
            )

        if mode != "new_scenarios":
            raise ValueError(f"Unknown mode: {mode}")

        from src.rade_ml_pt.pipelines.ensemble.infer import _dict_to_inference_context
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
            InferenceContext,
        )
        import pandas as pd

        member_inputs_built: Dict[str, Any] = {}
        per_member_meta: Dict[str, Dict[str, Any]] = {}

        for cid in self._config.cluster_ids:
            state = self._inference[cid]

            if not state.inference_context:
                raise ValueError(
                    f"No inference context available for cluster '{cid}'. "
                    f"Provide prebuilt member_inputs to run model-agnostic "
                    f"ensemble inference."
                )

            ctx = state.inference_context
            if not isinstance(ctx, InferenceContext):
                ctx = _dict_to_inference_context(
                    ctx, data_config_override=state.data_config,
                )

            inputs = HybridGnnRnnInferencePipeline._inject_unchanged_inputs(
                ctx, mode="new_scenarios",
            )

            # Resolve elementary PnL: caller-provided > baseline from registry.
            # Expects a pd.DataFrame with columns matching the training order
            # (ctx.elementary_pnl.columns). If a raw np.ndarray is provided,
            # wrap it using the stored column names.
            raw_pnl = None
            if cluster_pnl_histories and cid in cluster_pnl_histories:
                raw_pnl = cluster_pnl_histories[cid]
            elif state.baseline_pnl is not None:
                raw_pnl = state.baseline_pnl
            else:
                raise ValueError(
                    f"No pnl_history provided for cluster '{cid}' and no "
                    f"baseline PnL available in the session."
                )

            if isinstance(raw_pnl, pd.DataFrame):
                pnl_df = raw_pnl
            elif ctx.elementary_pnl is not None:
                pnl_df = pd.DataFrame(
                    raw_pnl,
                    columns=ctx.elementary_pnl.columns.tolist(),
                )
            else:
                raise ValueError(
                    f"Cannot wrap raw PnL array for cluster '{cid}': no "
                    f"elementary_pnl reference available for column names."
                )

            # Standardise using the saved scaler (same as training).
            if ctx.elementary_scaler is not None:
                pnl_scaled = HybridGnnRnnInferencePipeline._standardise_pnl(
                    pnl_unscaled=pnl_df, scaler=ctx.elementary_scaler,
                )
                inputs.elementary_pnl = pd.DataFrame(
                    pnl_scaled,
                    columns=pnl_df.columns.tolist(),
                    index=pnl_df.index.tolist(),
                )
            else:
                inputs.elementary_pnl = pnl_df

            elem_seq = HybridGnnRnnInferencePipeline.build_new_pnl_sequences(
                elementary_pnl=inputs.elementary_pnl,
                seq_length=ctx.data_config.seq_length,
                n_targets=len(inputs.target_indices),
            )

            result = HybridGnnRnnInferencePipeline.build_model_inputs(
                elem_seq=elem_seq,
                inputs=inputs,
                seq_length=ctx.data_config.seq_length,
            )

            member_inputs_built[cid] = result["inputs"]

            per_member_meta[cid] = {
                "n_trades": len(inputs.target_indices),
                "n_scenarios": elem_seq.shape[0],
                "has_new_trades": False,
            }

        combined = self._ensemble_model.predict(member_inputs_built)

        return {
            "predictions": combined,
            "per_member": per_member_meta,
            "metadata": {
                "ensemble_version": self._ensemble_version,
                "mode": mode,
                "n_scenarios": combined.shape[0],
                "n_targets": combined.shape[1] if combined.ndim > 1 else 1,
            },
        }

    def _route_new_trades(
        self, new_trade_attribs: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Route new trade attributes to clusters.

        If *new_trade_attribs* is already keyed by cluster_id, use as-is.
        Otherwise treat as a single trade and route via the TradeRouter.
        """
        result: Dict[str, Optional[Dict[str, Any]]] = {
            cid: None for cid in self._config.cluster_ids
        }

        if not new_trade_attribs:
            return result

        first_key = next(iter(new_trade_attribs))
        if first_key in self._config.cluster_ids:
            for cid, attribs in new_trade_attribs.items():
                result[cid] = attribs
            return result

        cid = self._ensemble_model.router.assign_new_trade(new_trade_attribs)
        result[cid] = new_trade_attribs
        return result

    # ==================================================================
    # Properties (UI helpers)
    # ==================================================================

    @property
    def is_metadata_loaded(self) -> bool:
        return self._config is not None

    @property
    def is_display_loaded(self) -> bool:
        if not self._config:
            return False
        return set(self._config.cluster_ids) <= set(self._display.keys())

    @property
    def inference_ready_clusters(self) -> List[str]:
        return sorted(self._inference.keys())

    @property
    def all_inference_ready(self) -> bool:
        if not self._config:
            return False
        return set(self._config.cluster_ids) <= set(self._inference.keys())

    @property
    def config(self) -> Optional[EnsembleConfig]:
        return self._config

    @property
    def member_versions(self) -> Optional[Dict[str, str]]:
        return self._member_versions

    @property
    def ensemble_version(self) -> Optional[str]:
        return self._ensemble_version

    @property
    def trade_cluster_map(self) -> Optional[Dict[str, str]]:
        return self._trade_cluster_map

    @property
    def member_summary(self) -> Optional[Dict[str, Dict[str, Any]]]:
        return self._member_summary

    @property
    def display(self) -> Dict[str, ClusterDisplayState]:
        return dict(self._display)

    @property
    def ensemble_display(self) -> Optional[EnsembleDisplayState]:
        return self._ensemble_display

    @property
    def ensemble_model(self) -> Any:
        return self._ensemble_model

    @property
    def cluster_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{cluster_id: {ccy: ..., desk: ..., product: ...}}``."""
        if self._config is None:
            return {}
        return self._config.get_cluster_keys_for_router() or {}

    # ==================================================================
    # Data helpers (dashboard-oriented)
    # ==================================================================

    def build_global_trade_catalogue(self) -> "pd.DataFrame":
        """Build a DataFrame of all target trades with attributes and cluster membership.

        Joins per-cluster ``target_attributes.json`` with cluster-level
        attributes so the dashboard can filter/aggregate by desk, product,
        ccy, or any other attribute across the entire portfolio.
        """
        import pandas as pd

        self._require_metadata()
        rows: List[Dict[str, Any]] = []
        cluster_attrs = self.cluster_attributes

        for cid in self._config.cluster_ids:
            display = self.load_cluster_display(cid)
            attribs = display.target_attributes

            if not attribs:
                for tid in self._config.cluster_mapping.get(cid, []):
                    row: Dict[str, Any] = {"trade_id": tid, "cluster_id": cid}
                    row.update(cluster_attrs.get(cid, {}))
                    rows.append(row)
                continue

            n = len(next(iter(attribs.values()), []))
            for i in range(n):
                row = {"cluster_id": cid}
                for key, values in attribs.items():
                    row[key] = values[i] if i < len(values) else None
                row.update(cluster_attrs.get(cid, {}))
                rows.append(row)

        return pd.DataFrame(rows)

    def get_prediction_store(self, split: str = "test") -> Optional[GlobalPredictionStore]:
        """Return a unified prediction/target store for *split*, building lazily.

        Reads per-member ``.npz`` files saved by the eval pipeline and
        slots each member's columns into the global trade order defined
        by ``manifest.json``.  Returns ``None`` if the manifest or
        prediction files are not available.
        """
        if split in self._prediction_stores:
            return self._prediction_stores[split]

        store = self._build_prediction_store(split)
        if store is not None:
            self._prediction_stores[split] = store
        return store

    def _build_prediction_store(self, split: str) -> Optional[GlobalPredictionStore]:
        if self._ensemble_display is None or not self._ensemble_display.manifest:
            return None

        manifest = self._ensemble_display.manifest
        trade_ids: List[str] = manifest.get("trade_ids", [])
        cluster_ids_list: List[str] = manifest.get("cluster_ids", [])
        cluster_trade_indices = manifest.get("cluster_trade_indices", {})

        if not trade_ids or self.artifacts_dir is None:
            return None

        eval_dir = (
            self.artifacts_dir / "ensemble" / self._ensemble_version / "evaluation"
        )
        n_total = len(trade_ids)
        n_scenarios: Optional[int] = None
        preds_global: Optional[np.ndarray] = None
        targets_global: Optional[np.ndarray] = None
        cluster_membership: List[str] = [""] * n_total

        for cid in cluster_ids_list:
            npz_path = eval_dir / "members" / cid / "predictions" / f"{split}.npz"
            if not npz_path.exists():
                continue

            data = np.load(str(npz_path))
            member_preds = data.get("predictions")
            member_targets = data.get("targets")
            if member_preds is None:
                continue

            if n_scenarios is None:
                n_scenarios = member_preds.shape[0]
                preds_global = np.zeros((n_scenarios, n_total), dtype=np.float32)
                targets_global = np.zeros((n_scenarios, n_total), dtype=np.float32)

            indices = cluster_trade_indices.get(cid, [])
            for col_local, col_global in enumerate(indices):
                if col_local < member_preds.shape[1]:
                    preds_global[:, col_global] = member_preds[:, col_local]
                    targets_global[:, col_global] = member_targets[:, col_local]
                    cluster_membership[col_global] = cid

        if preds_global is None:
            return None

        return GlobalPredictionStore(
            predictions=preds_global,
            targets=targets_global,
            trade_ids=trade_ids,
            cluster_ids=cluster_membership,
            split=split,
        )

    def load_cluster_market_data(self, cluster_id: str) -> Dict[str, Any]:
        """Load cluster asset portfolio / risk-factor shock data.

        Returns ``{asset_name: {rf_name: np.ndarray}}``.  Cached after
        first load per cluster.
        """
        if cluster_id in self._market_data_cache:
            return self._market_data_cache[cluster_id]

        self._require_metadata()
        version = self._member_versions[cluster_id]
        version_dir = self.registry_dir / version
        assets_path = version_dir / "cluster_assets.joblib"

        if not assets_path.exists():
            self._market_data_cache[cluster_id] = {}
            return {}

        import joblib
        portfolio = joblib.load(str(assets_path))

        result: Dict[str, Any] = {}
        for asset_name, asset in portfolio.items():
            rf_shocks = getattr(asset, "risk_factor_shocks", None)
            if rf_shocks is None:
                continue
            result[asset_name] = {}
            for rf_name, shocks in rf_shocks.items():
                if isinstance(shocks, dict):
                    result[asset_name][rf_name] = np.array(
                        list(shocks.values()), dtype=np.float64,
                    )
                else:
                    result[asset_name][rf_name] = np.asarray(shocks)

        self._market_data_cache[cluster_id] = result
        return result

    def load_cluster_graph_data(self, cluster_id: str) -> Dict[str, Any]:
        """Load graph adjacency and encoder feature data for one cluster.

        Returns dict with ``graph_results``, ``encoder_results``, and
        ``trade_universe`` keys.  Cached after first load per cluster.
        """
        if cluster_id in self._graph_data_cache:
            return self._graph_data_cache[cluster_id]

        self._require_metadata()
        import joblib

        version = self._member_versions[cluster_id]
        version_dir = self.registry_dir / version

        data: Dict[str, Any] = {}

        graph_path = version_dir / "graph_results.joblib"
        if graph_path.exists():
            data["graph_results"] = joblib.load(str(graph_path))

        encoder_path = version_dir / "encoder_results.joblib"
        if encoder_path.exists():
            data["encoder_results"] = joblib.load(str(encoder_path))

        display = self.load_cluster_display(cluster_id)
        data["trade_universe"] = display.trade_universe

        self._graph_data_cache[cluster_id] = data
        return data

    # ==================================================================
    # Guards
    # ==================================================================

    def _require_metadata(self) -> None:
        if self._config is None:
            raise RuntimeError(
                "Session metadata not loaded. Call load_metadata() first."
            )

    def _require_inference(self) -> None:
        self._require_metadata()
        if not self.all_inference_ready:
            missing = set(self._config.cluster_ids) - set(self._inference.keys())
            raise RuntimeError(
                f"Inference state not loaded for clusters: {sorted(missing)}. "
                f"Call load_inference_state() first."
            )
        if self._ensemble_model is None:
            raise RuntimeError("EnsembleModel not assembled. This is a bug.")
```

---

### 9.2 `config.py`

`src/rade_ml_pt/ensemble/config.py`

```python
"""
Ensemble configuration dataclasses.

``EnsembleConfig`` aggregates per-cluster member configs, trade-to-cluster
mapping, aggregation strategy, and infrastructure paths.  It is the single
config object consumed by all ensemble pipelines and the EnsembleBuilder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.rade_ml_pt.pipelines.config import PipelineConfig


@dataclass
class EnsembleConfig:
    """
    Top-level configuration for an ensemble of model members.

    Attributes
    ----------
    member_configs : dict
        ``{cluster_id: PipelineConfig_dict}`` — per-cluster pipeline config.
    pipeline_class : dict
        ``{cluster_id: dotpath_str}`` — pipeline class for each cluster.
        Omit or set to ``None`` for a cluster to use the default
        (``HybridGnnRnnTrainPipeline``).
    cluster_mapping : dict
        ``{cluster_id: [trade_id, ...]}`` — assigns every target trade to
        exactly one cluster.
    cluster_keys : dict or None
        ``{cluster_id: {attr_name: value, ...}}`` — optional attribute-based
        routing for new trades. Can be set directly or derived from
        ``cluster_key`` + ``cluster_key_values`` via ``get_cluster_keys_for_router()``.
    cluster_key : list or None
        Shared list of attribute names that define routing, e.g. ``["ccy", "desk", "product"]``.
        Used with ``cluster_key_values`` to build the key dict per cluster.
    cluster_key_values : dict or None
        ``{cluster_id: [value_ccy, value_desk, value_product, ...]}`` — per-cluster values
        in the same order as ``cluster_key``. E.g. cluster_0 = ``["GBP", "FLOW_RATES", "EUROPEAN"]``
        means cluster_0 has ccy=GBP, desk=FLOW_RATES, product=EUROPEAN.
    aggregation : str
        Aggregation strategy: ``"concat"`` (disjoint clusters) or
        ``"weighted_mean"`` (overlapping clusters).
    weights : dict or None
        ``{cluster_id: float}`` — member weights for weighted-mean
        aggregation.  Ignored when *aggregation* is ``"concat"``.
    execution_strategy : str
        How to execute per-member pipelines.  ``"sequential"`` runs one at
        a time (default).  Future values: ``"process_pool"`` (multi-CPU),
        ``"gpu_parallel"`` (multi-GPU), ``"distributed"`` (Ray / cloud).
    max_workers : int or None
        Maximum number of parallel workers for ``process_pool`` and
        ``gpu_parallel``.  ``None`` means one worker per cluster.
    gpu_device_ids : list of int or None
        Explicit GPU device IDs for ``gpu_parallel`` strategy.  ``None``
        means use all available GPUs via ``torch.cuda.device_count()``.
        Ignored by other strategies.
    registry_dir : str or None
        Root directory for ensemble and member registries.
    artifacts_dir : str or None
        Root directory for ensemble artifacts (plots, metrics, predictions).
    metadata : dict
        Arbitrary key-value pairs forwarded into run records.
    """

    member_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_class: Dict[str, Optional[str]] = field(default_factory=dict)
    cluster_mapping: Dict[str, List[str]] = field(default_factory=dict)
    cluster_keys: Optional[Dict[str, Dict[str, Any]]] = None
    cluster_key: Optional[List[str]] = None
    cluster_key_values: Optional[Dict[str, List[Any]]] = None
    aggregation: str = "concat"
    weights: Optional[Dict[str, float]] = None
    execution_strategy: str = "sequential"
    max_workers: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    registry_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def cluster_ids(self) -> List[str]:
        """Ordered list of cluster identifiers."""
        return sorted(self.cluster_mapping.keys())

    @property
    def n_members(self) -> int:
        return len(self.cluster_mapping)

    @property
    def all_trade_ids(self) -> List[str]:
        """Flat list of every trade ID across all clusters."""
        ids: List[str] = []
        for cid in self.cluster_ids:
            ids.extend(self.cluster_mapping[cid])
        return ids

    def get_cluster_keys_for_router(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Return ``{cluster_id: {attr_name: value, ...}}`` for TradeRouter.

        If ``cluster_keys`` is set, return it. Otherwise if ``cluster_key`` and
        ``cluster_key_values`` are set, build the dict from the shared attribute
        names and per-cluster value lists (same order). E.g. cluster_key =
        ["ccy", "desk", "product"], cluster_0_keys = ["GBP", "FLOW_RATES", "EUROPEAN"]
        -> cluster_0 key = {"ccy": "GBP", "desk": "FLOW_RATES", "product": "EUROPEAN"}.
        """
        if self.cluster_keys is not None:
            return self.cluster_keys
        if self.cluster_key is not None and self.cluster_key_values is not None:
            return {
                cid: dict(zip(self.cluster_key, values))
                for cid, values in self.cluster_key_values.items()
            }
        return None

    def get_member_pipeline_config(self, cluster_id: str) -> PipelineConfig:
        """Build a ``PipelineConfig`` for one member from its dict representation."""
        raw = self.member_configs.get(cluster_id, {})
        return PipelineConfig(
            training_config=raw.get("training_config"),
            data_config=raw.get("data_config"),
            model_config=raw.get("model_config"),
            registry_dir=self.registry_dir,
            tracking_dir=raw.get("tracking_dir"),
            artifacts_dir=self.artifacts_dir,
            version_or_tag=raw.get("version_or_tag", "latest"),
            metadata={
                **raw.get("metadata", {}),
                "cluster_id": cluster_id,
                "trade_ids": self.cluster_mapping.get(cluster_id, []),
            },
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnsembleConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EnsembleConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EnsembleConfig":
        """Load from a YAML file (requires ``pyyaml``)."""
        import yaml
        from src.rade_ml_pt.core.config import sanitize_yaml_values
        with open(path, "r") as f:
            return cls.from_dict(sanitize_yaml_values(yaml.safe_load(f)))
```

---

### 9.3 `registry.py`

`src/rade_ml_pt/ensemble/registry.py`

```python
"""
Ensemble-level registry.

Stores ensemble metadata (member versions, cluster mapping, aggregation
config, trade-to-cluster map) as a versioned bundle.  Uses the underlying
``ModelRegistry`` for member model storage and adds ensemble-specific
index / metadata files.

Storage layout::

    root_dir/
      ensemble/
        {ensemble_version}/
          ensemble_config.json       # Full EnsembleConfig
          member_versions.json       # {cluster_id: member_version}
          trade_cluster_map.json     # {trade_id: cluster_id}
          member_summary.json        # {cluster_id: {n_trades, metrics...}}
        index.json                   # tag -> ensemble_version
"""
from __future__ import annotations

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.rade_ml_pt.ensemble.config import EnsembleConfig

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "index.json"


class EnsembleRegistry:
    """
    Local filesystem registry for ensemble model bundles.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for ensemble registrations.  The ``ensemble/``
        subdirectory is created automatically.
    """

    def __init__(self, root_dir: Union[str, Path]) -> None:
        self.root_dir = Path(root_dir) / "ensemble"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root_dir / _INDEX_FILENAME
        self._index: Dict[str, str] = self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        config: EnsembleConfig,
        member_versions: Dict[str, str],
        member_summary: Optional[Dict[str, Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register an ensemble version.

        Parameters
        ----------
        config : EnsembleConfig
        member_versions : dict
            ``{cluster_id: registry_version_string}``
        member_summary : dict or None
            ``{cluster_id: {n_trades, mae, mse, ...}}`` for the UI.
        tags : list of str or None
            Labels for retrieval (e.g. ``["production"]``).

        Returns
        -------
        str
            The generated ensemble version identifier.
        """
        tags = tags or []
        version = self._generate_version()
        version_dir = self.root_dir / version
        version_dir.mkdir(parents=True, exist_ok=False)

        config.to_json(version_dir / "ensemble_config.json")

        with open(version_dir / "member_versions.json", "w") as f:
            json.dump(member_versions, f, indent=2)

        trade_cluster_map = {}
        for cid, tids in config.cluster_mapping.items():
            for tid in tids:
                trade_cluster_map[tid] = cid
        with open(version_dir / "trade_cluster_map.json", "w") as f:
            json.dump(trade_cluster_map, f, indent=2)

        if member_summary is not None:
            with open(version_dir / "member_summary.json", "w") as f:
                json.dump(member_summary, f, indent=2)

        for t in tags:
            self._index[t] = version
        self._index["latest"] = version
        self._save_index()

        logger.info("Registered ensemble version '%s' with tags %s", version, tags)
        return version

    def load(
        self,
        version_or_tag: str = "latest",
    ) -> tuple:
        """
        Load ensemble config and member versions.

        Returns
        -------
        tuple of (EnsembleConfig, Dict[str, str], str)
            (config, member_versions, resolved_version)
        """
        version = self._resolve_version(version_or_tag)
        version_dir = self.root_dir / version

        config = EnsembleConfig.from_json(version_dir / "ensemble_config.json")

        with open(version_dir / "member_versions.json", "r") as f:
            member_versions = json.load(f)

        logger.info("Loaded ensemble version '%s' (from '%s')", version, version_or_tag)
        return config, member_versions, version

    def get_metadata(self, version_or_tag: str = "latest") -> Dict[str, Any]:
        """Load the member summary and trade-cluster map for a version."""
        version = self._resolve_version(version_or_tag)
        version_dir = self.root_dir / version

        meta: Dict[str, Any] = {"version": version}

        summary_path = version_dir / "member_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                meta["member_summary"] = json.load(f)

        tcm_path = version_dir / "trade_cluster_map.json"
        if tcm_path.exists():
            with open(tcm_path, "r") as f:
                meta["trade_cluster_map"] = json.load(f)

        return meta

    def tag(self, version: str, tag: str) -> None:
        """Add a tag to an existing ensemble version."""
        version_dir = self.root_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Ensemble version '{version}' not found.")
        self._index[tag] = version
        self._save_index()
        logger.info("Tagged ensemble version '%s' with '%s'", version, tag)

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all registered ensemble versions with basic metadata."""
        versions = []
        for d in sorted(self.root_dir.iterdir()):
            config_path = d / "ensemble_config.json"
            if not config_path.exists():
                continue
            config = EnsembleConfig.from_json(config_path)
            versions.append({
                "version": d.name,
                "n_members": config.n_members,
                "n_trades": len(config.all_trade_ids),
                "aggregation": config.aggregation,
            })
        return versions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_version(self, version_or_tag: str) -> str:
        if version_or_tag in self._index:
            return self._index[version_or_tag]
        version_dir = self.root_dir / version_or_tag
        if version_dir.exists():
            return version_or_tag
        raise KeyError(
            f"'{version_or_tag}' is not a known tag or ensemble version. "
            f"Available tags: {list(self._index.keys())}"
        )

    def _load_index(self) -> Dict[str, str]:
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    @staticmethod
    def _generate_version() -> str:
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        short_hash = hashlib.md5(now.isoformat().encode()).hexdigest()[:6]
        return f"ens_{ts}_{short_hash}"
```
