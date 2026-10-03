# Ensemble Dashboard — Complete Design & Data Workflow

> **Target audience**: FX Derivatives traders, Front Office structurers, Risk managers.
> **Objective**: Build confidence in model outputs across the full trade portfolio by
> providing transparent, drill-down evaluation analytics and production-grade
> scenario inference — all through a single professional dashboard.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Flow: EnsembleSession ↔ Dash](#2-data-flow-ensemblesession--dash)
3. [Trade Attribute Resolution (Cross-Cluster Filtering)](#3-trade-attribute-resolution-cross-cluster-filtering)
4. [Global Prediction Store (Cross-Cluster PnL Slicing)](#4-global-prediction-store-cross-cluster-pnl-slicing)
5. [Global Controls](#5-global-controls)
6. [Tab 1 — Overview](#6-tab-1--overview)
7. [Tab 2 — Evaluation](#7-tab-2--evaluation)
8. [Tab 3 — Cluster Deep Dive](#8-tab-3--cluster-deep-dive)
9. [Tab 4 — Market Data](#9-tab-4--market-data)
10. [Tab 5 — Trade Graph Explorer](#10-tab-5--trade-graph-explorer)
11. [Tab 6 — Inference](#11-tab-6--inference)
12. [Tab 7 — Model Governance](#12-tab-7--model-governance)
13. [Session Gaps & Required Changes](#13-session-gaps--required-changes)
14. [Performance Budget](#14-performance-budget)
15. [Visual Style Guide](#15-visual-style-guide)

---

## 1. Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                          Dash App                                 │
│                                                                   │
│  ┌─────────┬──────────┬─────────┬─────────┬───────┬──────────┬──────────┐│
│  │Overview │Evaluation│ Cluster │ Market  │ Trade │Inference │   Model  ││
│  │         │          │Deep Dive│  Data   │ Graph │          │Governance││
│  └─────────┴──────────┴─────────┴─────────┴───────┴──────────┴──────────┘│
│        │             │            │           │           │        │
│        └─────────────┴────────────┴───────────┴───────────┘        │
│                              │                                     │
│                      EnsembleSession                               │
│                  ┌───────────┬────────────┐                        │
│                  │  Phase 1  │  Phase 2   │  Phase 3 (on demand)  │
│                  │ Metadata  │  Display   │  Inference state      │
│                  └───────────┴────────────┘                        │
│                        │            │                              │
│              ┌─────────┴──────┐  ┌──┴────────────────────┐        │
│              │ EnsembleRegistry│  │ Artifacts directory   │        │
│              │ (JSON metadata)│  │ (JSON, NPZ, PNG)      │        │
│              └────────────────┘  └───────────────────────┘        │
└───────────────────────────────────────────────────────────────────┘
```

### Key principle

The Dash app **never** instantiates models, loads `.pt` files, or calls
`torch` for **evaluation display**. All analytics tabs read **pre-computed
artifacts** via `EnsembleSession` Phase 1 + Phase 2. Only the Inference
tab triggers Phase 3 (model loading), and even then only for the clusters
the user selects.

---

## 2. Data Flow: EnsembleSession ↔ Dash

### 2.1 Phase 1 — Metadata (on app start, < 500ms)

```python
session = EnsembleSession(registry_dir, artifacts_dir)
session.load_metadata("production")
```

**Loaded into memory:**

| Object | Source file | Size |
|--------|-----------|------|
| `EnsembleConfig` | `ensemble_config.json` | ~2 KB |
| `member_versions` | `member_versions.json` | ~1 KB |
| `trade_cluster_map` | `trade_cluster_map.json` | ~5 KB (varies with n_trades) |
| `member_summary` | `member_summary.json` | ~2 KB per cluster |

**What the Dash app gets from Phase 1:**

- List of `cluster_ids` → populates all dropdowns.
- `cluster_key` + `cluster_key_values` → builds the **cluster attribute
  lookup table** (see §3).
- `trade_cluster_map` → `{trade_id: cluster_id}` for every known trade.
- `member_summary` → per-cluster headline metrics (enough for landing page
  KPI cards).
- `all_trade_ids` → ordered global trade list for column alignment.

### 2.2 Phase 2 — Display Artifacts (on app start, 1–5s)

```python
session.load_display_artifacts()
```

**Per-cluster, loaded into `ClusterDisplayState`:**

| Object | Source file | Notes |
|--------|-----------|-------|
| `eval_metrics` | `per_member_metrics_{split}.json` | Per-split per-cluster metrics |
| `plot_paths` | `evaluation/plots/{split}/*.png` | File paths only, not images |
| `trade_universe` | `trade_universe.json` | Split indices, trade IDs, scenario metadata |
| `data_config` snippet | `data_config.json` | seq_length, transform_type, etc. |

**Ensemble-level (new — not yet in session, see §10):**

| Object | Source file | Notes |
|--------|-----------|-------|
| `ensemble_metrics` per split | `ensemble_metrics_{split}.json` | Portfolio-level MAE, RMSE, etc. |
| `member_rollup` per split | `member_rollup_{split}.json` | Cross-member statistics |

**Per-cluster, loaded lazily on drill-down:**

| Object | Source file | Notes |
|--------|-----------|-------|
| `predictions_{split}` | `predictions/{split}.npz` | `{predictions, targets}` arrays |
| `target_attributes` | `target_attributes.json` | Per-trade metadata: `trade_id`, `product_type`, `product_subtype`, `underlying_risk_factors`, etc. |
| `elementary_attributes` | `elementary_attributes.json` | Same structure for elementary trades |

### 2.3 Phase 3 — Inference State (on user action only)

```python
# User selects clusters to run inference on
session.load_inference_state(cluster_ids=["cluster_0", "cluster_1"])
```

Only triggered from the **Inference** tab. Loads `nn.Module` + inference
context per cluster. Progress displayed via Dash interval polling.

---

## 3. Trade Attribute Resolution (Cross-Cluster Filtering)

### The problem

The user has clustered by `["ccy", "product"]` (via `cluster_key`), so
each `cluster_id` maps to exactly one (ccy, product) pair. But a trader
asks: *"Show me all FLOW_RATES desk trades."* The desk attribute is **not**
the cluster key — it is a per-trade field inside `target_attributes.json`
within each cluster's version directory.

### The solution: Global Trade Catalogue

At Phase 2 load time (or first access), the session builds a single
**global trade catalogue** DataFrame by merging per-cluster trade attributes:

```
global_trade_catalogue: pd.DataFrame
    Columns: trade_id | cluster_id | ccy | product_type | product_subtype |
             desk | underlying | moneyness | delta | vega | yrs_to_maturity | ...
```

#### How it is built

```python
def build_global_trade_catalogue(session) -> pd.DataFrame:
    """
    Merge target_attributes.json from every cluster into one
    DataFrame, enriched with cluster_id and cluster-level attributes.
    """
    rows = []
    cluster_attrs = session.config.get_cluster_keys_for_router() or {}

    for cid in session.config.cluster_ids:
        version = session.member_versions[cid]
        version_dir = Path(session.registry_dir) / version

        # Load per-trade attributes
        attribs_path = version_dir / "target_attributes.json"
        if not attribs_path.exists():
            continue
        with open(attribs_path) as f:
            attribs = json.load(f)

        n_trades = len(attribs.get("trade_id", []))
        for i in range(n_trades):
            row = {"cluster_id": cid}
            for key, values in attribs.items():
                row[key] = values[i] if i < len(values) else None
            # Enrich with cluster-level attributes
            for attr_name, attr_val in cluster_attrs.get(cid, {}).items():
                row.setdefault(attr_name, attr_val)
            rows.append(row)

    return pd.DataFrame(rows)
```

#### How it enables filtering

Trade IDs follow the convention `underlying|product_type|strike|maturity|...`,
and `target_attributes.json` stores per-trade fields including `product_type`,
`product_subtype`, `underlying_risk_factors`, etc. The desk is typically
derivable from the cluster attributes or from the trade ID convention.

For any user filter (desk, product, ccy, underlying, maturity bucket):

1. Filter `global_trade_catalogue` by the requested attribute(s).
2. Get the resulting `trade_id` list.
3. Group by `cluster_id` to know which clusters contribute.
4. For each cluster, map `trade_id` → column index in that cluster's
   prediction array (using the cluster's `target_ids` list from
   `trade_universe.json`).
5. Slice the prediction/target arrays by those column indices.
6. Aggregate (sum for PnL, mean for normalised metrics).

**This approach is fully general**: it works regardless of what the
`cluster_key` is. Filtering by desk when clusters are defined by
(ccy, product) simply selects trades from multiple clusters.

#### Performance

The catalogue is built **once** and cached in `dcc.Store` or a module-level
variable. For 100 clusters × ~50 target trades each = ~5,000 rows — trivial
to filter in real time.

---

## 4. Global Prediction Store (Cross-Cluster PnL Slicing)

### The problem

Every evaluation and filtering view needs to slice prediction/target
arrays by arbitrary trade groupings (desk, product, ccy, maturity bucket,
individual trade). Today each cluster's predictions live in separate
`.npz` files with different shapes. Every Dash callback would need to:
load N files, map trade IDs to column indices, slice, concatenate. This
is repetitive, error-prone, and slow when many clusters are involved.

### The solution: Global Prediction Store

At Phase 2 load time (lazy, per-split), build a **unified numpy structure**
that aligns all member predictions into a single global coordinate system.

```
GlobalPredictionStore (per split):
    predictions : np.ndarray  [n_scenarios, n_total_targets]   (float32)
    targets     : np.ndarray  [n_scenarios, n_total_targets]   (float32)
    trade_ids   : np.ndarray  [n_total_targets]                (str / U64)
    scenario_ids: np.ndarray  [n_scenarios]                    (str / U32)
    cluster_ids : np.ndarray  [n_total_targets]                (str / U32)
```

All arrays share the **same column order** as `EnsembleConfig.all_trade_ids`
(sorted by cluster, then by order within cluster). The `cluster_ids` array
gives each column's owning cluster for fast group-by.

#### How it is built

```python
@dataclass
class GlobalPredictionStore:
    """Aligned prediction + target arrays across all clusters for one split."""
    predictions: np.ndarray     # [n_scenarios, n_total_targets]
    targets: np.ndarray         # [n_scenarios, n_total_targets]
    trade_ids: np.ndarray       # [n_total_targets]
    scenario_ids: np.ndarray    # [n_scenarios]
    cluster_ids: np.ndarray     # [n_total_targets] — cluster owning each column

def build_global_prediction_store(
    session, split: str, manifest: dict,
) -> GlobalPredictionStore:
    """
    Assemble a unified prediction store from per-member .npz files.

    Reads each member's predictions_{split}.npz and slots columns into
    the global trade order defined by the manifest.
    """
    all_trade_ids = manifest["trade_ids"]
    cluster_trade_indices = manifest["cluster_trade_indices"]
    n_total = len(all_trade_ids)

    # Determine n_scenarios from the first available member
    first_cid = next(iter(cluster_trade_indices))
    first_npz = _load_member_npz(session, first_cid, split)
    n_scenarios = first_npz["predictions"].shape[0]

    preds = np.zeros((n_scenarios, n_total), dtype=np.float32)
    tgts  = np.zeros((n_scenarios, n_total), dtype=np.float32)
    col_cluster = np.empty(n_total, dtype="<U64")

    for cid, col_indices in cluster_trade_indices.items():
        npz = _load_member_npz(session, cid, split)
        if npz is None:
            continue
        for local_j, global_j in enumerate(col_indices):
            preds[:, global_j] = npz["predictions"][:, local_j]
            tgts[:, global_j]  = npz["targets"][:, local_j]
            col_cluster[global_j] = cid

    # Scenario IDs: from trade_universe or simple range
    scenario_ids = np.arange(n_scenarios).astype(str)

    return GlobalPredictionStore(
        predictions=preds,
        targets=tgts,
        trade_ids=np.array(all_trade_ids),
        scenario_ids=scenario_ids,
        cluster_ids=col_cluster,
    )
```

#### How it enables any aggregation

With the store loaded, **every** eval view becomes a simple numpy operation:

```python
store = prediction_stores[split]  # cached per split
catalogue = global_trade_catalogue

# --- Portfolio PnL ---
portfolio_pred   = store.predictions.sum(axis=1)
portfolio_target = store.targets.sum(axis=1)

# --- By Desk ---
desk_trades = catalogue[catalogue["desk"] == "FLOW_RATES"]["trade_id"].values
col_mask = np.isin(store.trade_ids, desk_trades)
desk_pred   = store.predictions[:, col_mask].sum(axis=1)
desk_target = store.targets[:, col_mask].sum(axis=1)

# --- By CCY ---
ccy_mask = np.isin(store.trade_ids,
    catalogue[catalogue["ccy"] == "GBP"]["trade_id"].values)
ccy_pred = store.predictions[:, ccy_mask].sum(axis=1)

# --- Single trade ---
trade_idx = np.where(store.trade_ids == "GBP|EUROPEAN|1.30|0.5Y|CALL")[0]
trade_pred = store.predictions[:, trade_idx].squeeze()

# --- Arbitrary multi-attribute filter ---
mask = (catalogue["ccy"] == "GBP") & (catalogue["product_type"] == "BARRIER")
filtered_ids = catalogue.loc[mask, "trade_id"].values
col_mask = np.isin(store.trade_ids, filtered_ids)
filtered_pred = store.predictions[:, col_mask].sum(axis=1)

# --- Per-trade metrics (vectorised) ---
residuals = store.predictions - store.targets
mae_per_trade = np.mean(np.abs(residuals), axis=0)  # [n_total_targets]

# --- Per-cluster aggregation ---
for cid in np.unique(store.cluster_ids):
    cid_mask = store.cluster_ids == cid
    cluster_pred = store.predictions[:, cid_mask].sum(axis=1)
```

#### Why numpy, not DataFrame

| Concern | NumPy arrays | DataFrame |
|---------|-------------|-----------|
| **Memory** | ~40 bytes per float32 cell | ~80+ bytes per cell (index overhead, object dtype risk) |
| **Speed** | Vectorised C-level slicing, `np.isin` mask in microseconds | Comparable for simple ops, slower for column mask + sum patterns |
| **Serialisation** | `.npz` compressed on disk, instant `np.load` | Parquet/Feather add overhead |
| **Integration** | Direct feed to Plotly `go.Scatter(y=array)` | `.values` conversion needed |

For a 5,000-scenario × 500-target store: ~10 MB per split (preds + targets).
Three splits = ~30 MB. Trivially fits in memory even for 100 clusters.

#### Lifecycle

```
Phase 2 (lazy, per split):
  1. First callback that needs predictions for a split triggers build.
  2. Store is cached in a module-level dict: {split: GlobalPredictionStore}.
  3. Subsequent callbacks for same split get instant access.
  4. On ensemble version change: invalidate cache, rebuild on next access.
```

---

## 5. Global Controls

Always visible in the top header bar.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  QuantStrata Ensemble Analytics                                             │
│                                                                             │
│  Ensemble: [production ▼]                                    As of: 2026   │
│                                                                             │
│  ┌─────────┬──────────┬─────────┬─────────┬───────┬──────────┬──────────┐   │
│  │Overview │Evaluation│ Cluster │ Market  │ Trade │Inference │   Model  │   │
│  │         │          │Deep Dive│  Data   │ Graph │          │Governance│   │
│  └─────────┴──────────┴─────────┴─────────┴───────┴──────────┴──────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Control | Type | Behaviour |
|---------|------|-----------|
| **Ensemble selector** | Dropdown | Lists all registered ensemble versions/tags. On change: reload Phase 1 + 2. |

The **split selector** (`Train` / `Val` / `Test`) is **not** a global
control. It appears as a local toggle **inside** the tabs and sub-tabs
that need it — specifically:

| Tab / Sub-tab | Split toggle? | Notes |
|---------------|:------------:|-------|
| Overview | Yes | Radio buttons above the KPI cards |
| Evaluation (all sub-tabs) | Yes | Radio buttons in the sub-tab header |
| Cluster Deep Dive | Yes (plus "all" for split comparison) | Radio buttons; split comparison table always shows all three |
| Market Data | No | Market data is scenario-level, not split-specific |
| Trade Graph Explorer | No | Graph structure is static (training-time artefact) |
| Inference | No | Inference produces new predictions, not tied to a training split |
| Model Governance | No | Metadata / config, not split-specific |

This keeps the header clean and avoids confusion on tabs where splits
are irrelevant (market data, graph, inference, governance).

---

## 6. Tab 1 — Overview

**Purpose:** 30-second confidence check for a senior trader, head of desk,
or CRO. Answer: *"Is this model production-ready?"*

**Data needed:** Phase 1 + Phase 2 (ensemble-level JSONs only, no arrays).

### Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PORTFOLIO KPIs                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │   MAE    │ │   RMSE   │ │  Max AE  │ │  P95 AE  │ │  P99 AE  │     │
│  │  0.0023  │ │  0.0041  │ │  0.0312  │ │  0.0089  │ │  0.0187  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│                                                                         │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────┐  │
│  │  Portfolio Pred vs Target       │  │  Member Comparison (MAE)     │  │
│  │  [scatter plot]                 │  │  [horizontal bar chart]      │  │
│  │  45° reference line             │  │  sorted worst → best         │  │
│  │  colour: density / split        │  │  colour: cluster palette     │  │
│  └─────────────────────────────────┘  └──────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Member KPI Table                                    [sortable] │   │
│  │  ┌───────────┬──────┬─────────┬──────┬──────┬───────┬──────┐   │   │
│  │  │ Cluster   │ CCY  │ Product │ Desk │ MAE  │ RMSE  │ #Trd │   │   │
│  │  ├───────────┼──────┼─────────┼──────┼──────┼───────┼──────┤   │   │
│  │  │ cluster_0 │ GBP  │ EUROPEAN│ FLOW │0.0018│0.0032 │  47  │   │   │
│  │  │ cluster_1 │ USD  │ BARRIER │ EXOT │0.0045│0.0078 │  23  │   │   │
│  │  │    ...    │      │         │      │      │       │      │   │   │
│  │  └───────────┴──────┴─────────┴──────┴──────┴───────┴──────┘   │   │
│  │  Conditional formatting: green < P25, amber P25-P75, red > P75 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Cluster Performance Heatmap                                     │   │
│  │  Rows: clusters │ Cols: MAE, RMSE, MaxAE, P95, P99              │   │
│  │  [colour intensity heatmap with values annotated]                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data workflow

```
Session Phase 1:  config.get_cluster_keys_for_router()  → cluster attr table
Session Phase 2:  ensemble_metrics_{split}.json          → KPI cards
                  per_member_metrics_{split}.json         → table + heatmap + bar chart
                  member_rollup_{split}.json              → cross-member stats in cards
```

**No prediction arrays loaded.** Pure JSON reads.

---

## 7. Tab 2 — Evaluation

**Purpose:** Detailed PnL comparison (predictions vs targets) at every
level of aggregation. The core confidence-builder for traders and Risk.

### Sub-tab structure

```
  ┌───────────┬──────────┬────────────┬──────────┬────────────┐
  │ Portfolio │ By Desk  │ By Product │ By CCY   │ By Cluster │
  └───────────┴──────────┴────────────┴──────────┴────────────┘
```

### 7.1 Sub-tab: Portfolio

**Shows the full-book aggregated view.** Every target trade's PnL is
summed per scenario to give a single portfolio-level predicted vs
actual PnL series.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO EVALUATION — [Test ▼]                                        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PnL Time-Series (predicted vs target)                           │   │
│  │                                                                  │   │
│  │  X: scenario index (or date if available)                        │   │
│  │  Y: summed portfolio PnL                                         │   │
│  │  Blue line: model prediction                                     │   │
│  │  Grey line: realised/target                                      │   │
│  │  Shaded band: residual (pred − target)                           │   │
│  │                                                                  │   │
│  │  [Interactive: hover shows scenario details, zoom, pan]          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Pred vs Target Scatter   │  │  Residual Distribution             │  │
│  │                           │  │                                    │  │
│  │  Each dot = 1 scenario    │  │  Histogram of per-scenario         │  │
│  │  X: target PnL            │  │  residuals (pred − target)         │  │
│  │  Y: predicted PnL         │  │                                    │  │
│  │  45° reference line       │  │  Annotation box:                   │  │
│  │  Colour: density          │  │    mean, std, skew, kurtosis       │  │
│  │  R² annotation            │  │    % within ±1σ, ±2σ               │  │
│  └───────────────────────────┘  └────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Distribution Comparison                                         │   │
│  │  ┌─────────┬───────────┬───────────┬──────┐                      │   │
│  │  │ Pctile  │ Predicted │  Target   │ Diff │                      │   │
│  │  ├─────────┼───────────┼───────────┼──────┤                      │   │
│  │  │ P1      │ -4,230    │ -4,180    │ -50  │                      │   │
│  │  │ P5      │ -2,100    │ -2,050    │ -50  │                      │   │
│  │  │ P25     │   -340    │   -310    │ -30  │                      │   │
│  │  │ P50     │    120    │    135    │ -15  │                      │   │
│  │  │ P75     │    580    │    600    │ -20  │                      │   │
│  │  │ P95     │  2,400    │  2,380    │  20  │                      │   │
│  │  │ P99     │  4,100    │  4,050    │  50  │                      │   │
│  │  └─────────┴───────────┴───────────┴──────┘                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Worst Scenarios                                     [Top 10]   │   │
│  │  ┌────────────┬───────────┬──────────┬──────────┬───────────┐   │   │
│  │  │ Scenario   │ Predicted │  Target  │ Residual │ %Err      │   │   │
│  │  ├────────────┼───────────┼──────────┼──────────┼───────────┤   │   │
│  │  │ scen_1042  │  -5,200   │ -3,900   │  -1,300  │  33.3%    │   │   │
│  │  │ scen_0887  │   4,800   │  3,700   │   1,100  │  29.7%    │   │   │
│  │  │    ...     │           │          │          │           │   │   │
│  │  └────────────┴───────────┴──────────┴──────────┴───────────┘   │   │
│  │  Click row → jumps to Cluster Deep Dive for that scenario       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data workflow — Portfolio sub-tab

```
1. store = session.get_prediction_store(split)   # cached GlobalPredictionStore

2. Sum across columns (all trades):
     portfolio_pred   = store.predictions.sum(axis=1)   → [n_scenarios]
     portfolio_target = store.targets.sum(axis=1)       → [n_scenarios]

3. Time-series: plot portfolio_pred & portfolio_target vs store.scenario_ids
4. Scatter: portfolio_pred vs portfolio_target
5. Residuals: portfolio_pred - portfolio_target → histogram + stats
6. Percentiles: np.percentile on both series
7. Worst: sort by |residual| descending → top 10 table
```

### 7.2 Sub-tab: By Desk

**Aggregates trades by desk.** Users see predicted vs target PnL
**for each desk**, enabling desk heads to judge fit quality for their
specific book.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EVALUATION BY DESK — [Test ▼]                                          │
│                                                                         │
│  Desk Filter: [■ FLOW_RATES  ■ EXOTICS  ■ EM_RATES  □ CREDIT]  ▼      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PnL Comparison by Desk                                          │   │
│  │                                                                  │   │
│  │  [Overlaid line charts — one pair (pred/target) per desk]        │   │
│  │  X: scenario index                                               │   │
│  │  Y: desk-aggregated PnL                                          │   │
│  │  Solid: predicted │ Dashed: target │ Colour: desk                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Residual Box Plot        │  │  Per-Desk Metrics Table            │  │
│  │                           │  │                                    │  │
│  │  One box per desk         │  │  ┌──────┬──────┬──────┬──────┐    │  │
│  │  Y: per-scenario residual │  │  │ Desk │ MAE  │ RMSE │MaxAE │    │  │
│  │  Identifies which desk    │  │  ├──────┼──────┼──────┼──────┤    │  │
│  │  has worst model fit      │  │  │ FLOW │0.0012│0.0028│0.018 │    │  │
│  │                           │  │  │ EXOT │0.0067│0.0091│0.043 │    │  │
│  │  Outliers highlighted     │  │  │  ... │      │      │      │    │  │
│  └───────────────────────────┘  │  └──────┴──────┴──────┴──────┘    │  │
│                                  │  Sortable, conditionally coloured │  │
│                                  └────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Scatter Grid (small multiples)                                  │   │
│  │                                                                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                         │   │
│  │  │FLOW_RATES│ │ EXOTICS  │ │ EM_RATES │  ...                    │   │
│  │  │  pred vs │ │  pred vs │ │  pred vs │                         │   │
│  │  │  target  │ │  target  │ │  target  │                         │   │
│  │  │ (scatter)│ │ (scatter)│ │ (scatter)│                         │   │
│  │  └──────────┘ └──────────┘ └──────────┘                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data workflow — By Desk (same pattern for By Product, By CCY)

```
1. store = session.get_prediction_store(split)   # cached
   catalogue = global_trade_catalogue             # cached

2. For each selected desk:
     desk_trades = catalogue[catalogue["desk"] == desk_name]["trade_id"].values
     col_mask = np.isin(store.trade_ids, desk_trades)
     desk_pred   = store.predictions[:, col_mask].sum(axis=1)
     desk_target = store.targets[:, col_mask].sum(axis=1)

3. Plot overlaid time-series, box plots, scatter grid
4. Compute per-desk metrics from the sliced arrays
```

**Key insight**: Filtering by desk when `cluster_key = ["ccy", "product"]`
works because the global trade catalogue contains per-trade `desk` (or
it is parseable from the `trade_id` convention). Trades from **multiple
clusters** can belong to the same desk, and the aggregation handles
this transparently.

### 7.3 Sub-tab: By Product

Identical layout to By Desk, but grouped by `product_type` (or
`product_subtype`).

```
Product Filter: [■ EUROPEAN  ■ BARRIER  ■ DIGITAL  □ ASIAN  □ CLIQUET]

Same charts: overlaid time-series, box plot, scatter grid, metrics table.
```

#### Extra for Product view

- **Product complexity indicator**: badge showing L1/L2/L3 next to each
  product name (from the complexity classification).
- **Expected vs actual accuracy**: L1 products should have lower MAE than
  L3; flag anomalies where exotics outperform vanillas (possible data issue).

### 7.4 Sub-tab: By CCY

Identical layout, grouped by currency.

```
CCY Filter: [■ GBP  ■ USD  ■ EUR  □ JPY  □ CHF]
```

#### Extra for CCY view

- **Cross-currency basis**: if model errors correlate across CCY pairs,
  surface this with a small correlation heatmap of per-CCY residuals.

### 7.5 Sub-tab: By Cluster

Direct per-cluster comparison with all target trades visible.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EVALUATION BY CLUSTER — [Test ▼]                                       │
│                                                                         │
│  Cluster: [cluster_0 (GBP / EUROPEAN / FLOW_RATES) ▼]                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Per-Trade PnL Heatmap                                           │   │
│  │                                                                  │   │
│  │  Rows: scenarios (sampled if > 500)                              │   │
│  │  Columns: target trades within cluster                           │   │
│  │  Colour: residual (pred − target)                                │   │
│  │  Blue: under-prediction │ Red: over-prediction                   │   │
│  │                                                                  │   │
│  │  Identifies which specific trades the model struggles with       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Cluster Pred vs Target   │  │  Trade-Level Metrics Table         │  │
│  │  (scatter, summed)        │  │                                    │  │
│  │                           │  │  ┌─────────┬──────┬──────┬──────┐  │  │
│  │  One point = 1 scenario   │  │  │ Trade   │ MAE  │ RMSE │MaxAE │  │  │
│  │  Summed across cluster    │  │  ├─────────┼──────┼──────┼──────┤  │  │
│  │  trades                   │  │  │ GBP|EUR │0.0008│0.0015│0.012 │  │  │
│  │                           │  │  │ GBP|EUR │0.0023│0.0041│0.031 │  │  │
│  │                           │  │  │  ...    │      │      │      │  │  │
│  └───────────────────────────┘  │  └─────────┴──────┴──────┴──────┘  │  │
│                                  └────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PnL Distribution Overlay                                        │   │
│  │  [violin / ridgeline plot]                                       │   │
│  │  One distribution per target trade                               │   │
│  │  Blue: predicted distribution │ Grey: target distribution        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data workflow — By Cluster

```
1. store = session.get_prediction_store(split)   # cached
   catalogue = global_trade_catalogue             # cached

2. Slice store for selected cluster:
     cid_mask = store.cluster_ids == selected_cluster_id
     cluster_preds   = store.predictions[:, cid_mask]   # [n_scenarios, n_cluster_targets]
     cluster_targets = store.targets[:, cid_mask]
     cluster_trades  = store.trade_ids[cid_mask]

3. Heatmap: cluster_preds - cluster_targets
4. Scatter: cluster_preds.sum(axis=1) vs cluster_targets.sum(axis=1)
5. Trade metrics: per-column MAE, RMSE, MaxAE
6. Violin: per-column distributions
7. target_attributes from catalogue for labelling and tooltips
```

---

## 8. Tab 3 — Cluster Deep Dive

**Purpose:** Forensic single-cluster analysis for the structurer or quant
who owns that product. Full model transparency.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLUSTER DEEP DIVE                                                      │
│                                                                         │
│  Cluster: [cluster_0 (GBP / EUROPEAN / FLOW_RATES) ▼]                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Cluster Header Card                                             │   │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐        │   │
│  │  │ Cluster  │ GBP      │ EUROPEAN │FLOW_RATES│ v2.3.1   │        │   │
│  │  │ ID: c_0  │ CCY      │ Product  │ Desk     │ Version  │        │   │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘        │   │
│  │  n_elementary: 142 │ n_target: 47 │ seq_length: 30               │   │
│  │  transform: signed_log_robust │ loss: huber_quantile             │   │
│  │  trained: 2026-03-20 │ best_val_loss: 0.00231                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Split Comparison                                                │   │
│  │                                                                  │   │
│  │  ┌─────────┬───────────┬───────────┬───────────┐                 │   │
│  │  │ Metric  │   Train   │    Val    │   Test    │                 │   │
│  │  ├─────────┼───────────┼───────────┼───────────┤                 │   │
│  │  │ MAE     │  0.0012   │  0.0018   │  0.0023   │                 │   │
│  │  │ RMSE    │  0.0021   │  0.0032   │  0.0041   │                 │   │
│  │  │ Max AE  │  0.0089   │  0.0156   │  0.0312   │                 │   │
│  │  │ P95     │  0.0034   │  0.0058   │  0.0089   │                 │   │
│  │  └─────────┴───────────┴───────────┴───────────┘                 │   │
│  │                                                                  │   │
│  │  Visual: grouped bar chart (train/val/test side by side)         │   │
│  │  Green→amber→red gradient signals overfitting                    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Training Convergence     │  │  Per-Trade Scatter Matrix          │  │
│  │                           │  │                                    │  │
│  │  X: epoch                 │  │  Select up to 6 trades:            │  │
│  │  Y: loss                  │  │  [trade_1 ▼] [trade_2 ▼] ...      │  │
│  │  Blue: train loss         │  │                                    │  │
│  │  Orange: val loss         │  │  Grid of pred vs target scatters   │  │
│  │  Vertical: early stop pt  │  │  for selected individual trades    │  │
│  └───────────────────────────┘  └────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Elementary PnL Explorer  │  │  Trade Graph Visualisation         │  │
│  │                           │  │                                    │  │
│  │  Time-series of input     │  │  Network diagram of GNN adjacency  │  │
│  │  elementary PnL (scaled)  │  │  from graph_results                │  │
│  │                           │  │                                    │  │
│  │  Select elementary trades │  │  Node size: importance/degree      │  │
│  │  to overlay               │  │  Colour: elementary (blue) vs      │  │
│  │                           │  │          target (orange)            │  │
│  │  Shows what the model     │  │  Edge thickness: adjacency weight  │  │
│  │  "sees" as input          │  │                                    │  │
│  └───────────────────────────┘  └────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Data Configuration Summary                                      │   │
│  │                                                                  │   │
│  │  ┌──────────────────┬──────────────────┬──────────────────┐      │   │
│  │  │ Data Pipeline    │ Model            │ Training         │      │   │
│  │  ├──────────────────┼──────────────────┼──────────────────┤      │   │
│  │  │ seq_length: 30   │ hidden_dim: 128  │ epochs: 200      │      │   │
│  │  │ batch_size: 32   │ n_gnn_layers: 3  │ lr: 1e-3         │      │   │
│  │  │ transform: s_l_r │ rnn_type: LSTM   │ loss: huber_q    │      │   │
│  │  │ n_scenarios: 5000│ dropout: 0.1     │ patience: 20     │      │   │
│  │  │ train/val/test:  │ attn_dim: 64     │ strategy: cpu    │      │   │
│  │  │ 3500/750/750     │                  │ compile: false    │      │   │
│  │  └──────────────────┴──────────────────┴──────────────────┘      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data workflow — Cluster Deep Dive

```
Phase 2 (cached):
  - ClusterDisplayState.eval_metrics  → split comparison table
  - ClusterDisplayState.plot_paths    → training convergence PNG
  - ClusterDisplayState.trade_universe → split sizes, scenario metadata
  - data_config.json                  → config summary cards

Lazy load on entry:
  - predictions_{split}.npz per split → split comparison metrics
  - target_attributes.json            → trade selector dropdowns
  - elementary_pnl.parquet            → elementary PnL explorer
  - graph_results.joblib              → trade graph visualisation
  - encoder_results.joblib            → feature importance (if available)
```

---

## 9. Tab 4 — Market Data

**Purpose:** Give traders and Risk full transparency into the **underlying
market data** (risk factor shocks, rate curves, vol surfaces, spot levels)
that drives model inputs. Traders need to verify that the scenarios the model
was trained on are realistic and representative of current market conditions,
and that new inference scenarios are sensible before trusting the PnL output.

**Data source:** `cluster_assets.joblib` per member (the `asset_portfolio`
dictionary saved during training). Each asset object contains
`risk_factor_shocks` — a dict of `{risk_factor_name: shock_data}` where
shock data is typically a dict-of-dicts (scenario → RF level) or a DataFrame.

### Sub-tab structure

```
  ┌────────────┬─────────────────┬──────────────────┬──────────────────┐
  │ RF Summary │ Shock Explorer  │ Scenario Heatmap │ Distribution     │
  └────────────┴─────────────────┴──────────────────┴──────────────────┘
```

### 9.1 Sub-tab: RF Summary

**Portfolio-wide view of all risk factors across all clusters.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MARKET DATA — RF Summary                                               │
│                                                                         │
│  Cluster: [All ▼]  (or select specific cluster)                         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Risk Factor Inventory                                [export]  │   │
│  │  ┌────────────┬───────────────┬───────────┬───────┬───────────┐ │   │
│  │  │ Asset      │ Risk Factor   │ Type      │ #Scen │ Cluster   │ │   │
│  │  ├────────────┼───────────────┼───────────┼───────┼───────────┤ │   │
│  │  │ GBP_IR     │ GBP_1Y_RATE   │ IR Curve  │ 5000  │ cluster_0 │ │   │
│  │  │ GBP_IR     │ GBP_5Y_RATE   │ IR Curve  │ 5000  │ cluster_0 │ │   │
│  │  │ GBPUSD_FX  │ GBPUSD_SPOT   │ FX Spot   │ 5000  │ cluster_0 │ │   │
│  │  │ GBPUSD_FX  │ GBPUSD_1M_VOL │ FX Vol    │ 5000  │ cluster_0 │ │   │
│  │  │ USD_IR     │ USD_3M_RATE   │ IR Curve  │ 5000  │ cluster_1 │ │   │
│  │  │    ...     │               │           │       │           │ │   │
│  │  └────────────┴───────────────┴───────────┴───────┴───────────┘ │   │
│  │                                                                  │   │
│  │  Summary cards:                                                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │ Assets   │ │ RF Count │ │ Scenarios│ │ Clusters │           │   │
│  │  │   12     │ │    87    │ │  5,000   │ │    8     │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Cross-Cluster RF Coverage Matrix                                │   │
│  │                                                                  │   │
│  │  Rows: risk factors │ Cols: clusters                             │   │
│  │  Cell: ● present / ○ absent                                      │   │
│  │  Colour: n_scenarios (intensity)                                 │   │
│  │                                                                  │   │
│  │  Shows which RFs are shared across clusters and which are        │   │
│  │  cluster-specific. Helps Risk verify completeness.               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Sub-tab: Shock Explorer

**Drill into individual risk factor shock time-series.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MARKET DATA — Shock Explorer                                           │
│                                                                         │
│  Cluster: [cluster_0 ▼]   Asset: [GBP_IR ▼]   RF: [GBP_1Y_RATE ▼]    │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Shock Time-Series                                               │   │
│  │                                                                  │   │
│  │  X: scenario index (or date)                                     │   │
│  │  Y: risk factor level / shock value                              │   │
│  │  Line: shock trajectory across scenarios                         │   │
│  │                                                                  │   │
│  │  Overlay: mean (solid), ±1σ band (shaded), ±2σ band (lighter)   │   │
│  │  Interactive: hover shows exact value, zoom to region             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Shock Distribution       │  │  Summary Statistics                │  │
│  │                           │  │                                    │  │
│  │  Histogram / KDE of       │  │  ┌──────────┬───────────┐         │  │
│  │  shock values across      │  │  │ Statistic│ Value     │         │  │
│  │  all scenarios            │  │  ├──────────┼───────────┤         │  │
│  │                           │  │  │ Mean     │  0.0234   │         │  │
│  │  Normal overlay for       │  │  │ Std      │  0.0156   │         │  │
│  │  comparison (are shocks   │  │  │ Skew     │ -0.32     │         │  │
│  │  realistic?)              │  │  │ Kurtosis │  3.41     │         │  │
│  │                           │  │  │ Min      │ -0.0523   │         │  │
│  │  Annotation: Jarque-Bera  │  │  │ Max      │  0.0712   │         │  │
│  │  test p-value             │  │  │ P5       │ -0.0198   │         │  │
│  └───────────────────────────┘  │  │ P95      │  0.0489   │         │  │
│                                  │  └──────────┴───────────┘         │  │
│                                  └────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Multi-RF Overlay                                                │   │
│  │                                                                  │   │
│  │  Select up to 5 RFs: [GBP_1Y ▼] [GBP_5Y ▼] [GBP_10Y ▼] ...   │   │
│  │                                                                  │   │
│  │  Overlaid time-series (normalised to z-score for comparability)  │   │
│  │  Shows co-movement / decorrelation between risk factors          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Sub-tab: Scenario Heatmap

**Full-portfolio risk factor heatmap for scenario analysis.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MARKET DATA — Scenario Heatmap                                         │
│                                                                         │
│  Cluster: [cluster_0 ▼]                                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  RF × Scenario Heatmap                                           │   │
│  │                                                                  │   │
│  │  Rows: risk factors (sorted by asset, then tenor/type)           │   │
│  │  Cols: scenarios (sampled to ~200 if > 500)                      │   │
│  │  Colour: shock magnitude (diverging: blue = down, red = up)      │   │
│  │                                                                  │   │
│  │  Interactive: click scenario column → highlights in all other    │   │
│  │  views, shows scenario detail panel                              │   │
│  │                                                                  │   │
│  │  Dendogram clustering on rows (optional): groups correlated RFs  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Scenario Detail (on click)                                      │   │
│  │                                                                  │   │
│  │  Scenario: scen_1042                                             │   │
│  │  ┌──────────────┬───────────┬──────────────────────────────┐     │   │
│  │  │ Risk Factor  │ Shock     │ Percentile (in training set) │     │   │
│  │  ├──────────────┼───────────┼──────────────────────────────┤     │   │
│  │  │ GBP_1Y_RATE  │ -0.0312   │ P3 (extreme down)            │     │   │
│  │  │ GBPUSD_SPOT  │ +0.0156   │ P72 (moderate up)            │     │   │
│  │  │    ...       │           │                              │     │   │
│  │  └──────────────┴───────────┴──────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Sub-tab: Distribution

**Compare distributions of risk factor shocks across clusters or between
training data and new inference scenarios.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MARKET DATA — Distribution Comparison                                  │
│                                                                         │
│  Mode: [● Cross-Cluster  ○ Training vs Inference]                       │
│  RF:   [GBP_1Y_RATE ▼]                                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Cross-Cluster Mode:                                             │   │
│  │  Overlaid histograms / violin plots of the same RF across        │   │
│  │  clusters that share it. Verifies consistency of shock            │   │
│  │  generation across the portfolio.                                 │   │
│  │                                                                  │   │
│  │  Training vs Inference Mode:                                     │   │
│  │  Side-by-side: training shock distribution vs new scenario       │   │
│  │  shock distribution. Flags extrapolation risk — if new shocks    │   │
│  │  lie outside training range, highlight in red.                   │   │
│  │                                                                  │   │
│  │  QQ-plot: training quantiles vs inference quantiles.             │   │
│  │  Departure from 45° line = distributional shift.                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Correlation Matrix                                              │   │
│  │                                                                  │   │
│  │  Select cluster: [cluster_0 ▼]                                   │   │
│  │  Heatmap of pairwise Pearson correlations between all RFs        │   │
│  │  within this cluster's asset portfolio.                          │   │
│  │                                                                  │   │
│  │  Highlights: strong positive (> 0.7), strong negative (< -0.7)   │   │
│  │  Useful for Risk to verify scenario plausibility.                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data workflow — Market Data tab

```
Phase 2 (lazy per cluster):
  cluster_assets.joblib   → asset_portfolio dict
                            Each asset: asset.risk_factor_shocks → {rf_name: shock_data}
                            asset.asset_name → asset identifier

Build on first access (per cluster):
  1. Load cluster_assets.joblib from version_dir
  2. Walk asset_portfolio:
       for asset_name, asset in portfolio.items():
           for rf_name, shocks in asset.risk_factor_shocks.items():
               # shocks is dict-of-dicts or DataFrame
               # Convert to np.ndarray [n_scenarios, 1] or [n_scenarios, n_tenors]
  3. Build RF inventory table: asset, rf_name, rf_type, n_scenarios, cluster_id
  4. Build shock arrays: {(cluster_id, asset, rf_name): np.ndarray}
  5. Cache in session-level dict

For "Training vs Inference" mode:
  Requires Phase 3 inference context (cluster_assets from inference context)
  OR the user uploading / pointing to new scenario shocks directory.
  Compare distribution of each RF between the two sources.

Memory: shock arrays are typically [n_scenarios × 1] floats per RF.
  100 RFs × 5000 scenarios × 4 bytes = ~2 MB per cluster. Negligible.
```

---

## 10. Tab 5 — Trade Graph Explorer

**Purpose:** Visualise the GNN adjacency graph that defines trade
relationships within each cluster. This is the core structural
insight of the Hybrid GNN-RNN model — traders and quants need to
understand **which trades influence which** and verify the graph
makes economic sense.

**Data source:** `graph_results.joblib` (sparse adjacency: indices, values,
shape) + `encoder_results.joblib` (combined features) + `trade_universe.json`
(trade IDs) per member version directory.

### Sub-tab structure

```
  ┌──────────────┬────────────────┬─────────────────┬──────────────────┐
  │ Graph View   │ Adjacency      │ Node Analytics  │ Cross-Cluster    │
  │              │ Analysis       │                 │ Comparison       │
  └──────────────┴────────────────┴─────────────────┴──────────────────┘
```

### 10.1 Sub-tab: Graph View

**Interactive network diagram of the trade graph.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRADE GRAPH — Graph View                                               │
│                                                                         │
│  Cluster: [cluster_0 (GBP / EUROPEAN / FLOW_RATES) ▼]                  │
│                                                                         │
│  Layout: [● Force-directed  ○ Circular  ○ Hierarchical]                 │
│  Colour by: [● Trade type (elem/target)  ○ Product  ○ MAE  ○ Degree]   │
│  Size by: [● Degree  ○ MAE  ○ Uniform  ○ PnL volatility]               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │                    [ Interactive Graph ]                          │   │
│  │                                                                  │   │
│  │  Nodes:                                                          │   │
│  │    ● Blue circles: elementary trades                              │   │
│  │    ● Orange circles: target trades                                │   │
│  │    Size: proportional to node degree (or selected metric)        │   │
│  │                                                                  │   │
│  │  Edges:                                                          │   │
│  │    Line thickness: adjacency weight (RBF kernel similarity)      │   │
│  │    Opacity: weight magnitude (faint = weak, solid = strong)      │   │
│  │                                                                  │   │
│  │  Interactions:                                                   │   │
│  │    Hover node → tooltip with trade_id, attributes, degree,       │   │
│  │                 top-3 neighbours, MAE (if target)                 │   │
│  │    Click node → highlight its neighbourhood (1-hop, 2-hop)       │   │
│  │    Zoom / pan / drag nodes to rearrange                          │   │
│  │    Search box: find and centre on a specific trade_id             │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Node Detail Panel (on click)                                    │   │
│  │                                                                  │   │
│  │  Trade: GBP|EUROPEAN|1.30|0.5Y|CALL                              │   │
│  │  Type: target │ Product: EUROPEAN │ Moneyness: 1.02              │   │
│  │  Delta: 0.52 │ Vega: 0.13 │ TTM: 0.5Y │ Degree: 8               │   │
│  │                                                                  │   │
│  │  Top neighbours:                                                 │   │
│  │  ┌──────────────────────────────────┬──────────┬───────┐         │   │
│  │  │ Trade ID                         │ Weight   │ Type  │         │   │
│  │  ├──────────────────────────────────┼──────────┼───────┤         │   │
│  │  │ GBP|EUROPEAN|1.28|0.5Y|PUT      │  0.934   │ elem  │         │   │
│  │  │ GBP|EUROPEAN|1.30|1Y|CALL       │  0.891   │ target│         │   │
│  │  │ GBP|DIGITAL|1.30|0.5Y|CALL      │  0.756   │ elem  │         │   │
│  │  └──────────────────────────────────┴──────────┴───────┘         │   │
│  │                                                                  │   │
│  │  If target: MAE = 0.0015, RMSE = 0.0028 (from eval)             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Sub-tab: Adjacency Analysis

**Quantitative view of graph structure — sparsity, weight distribution,
connectivity.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRADE GRAPH — Adjacency Analysis                                       │
│                                                                         │
│  Cluster: [cluster_0 ▼]                                                 │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Graph Statistics Cards   │  │  Edge Weight Distribution          │  │
│  │                           │  │                                    │  │
│  │  ┌──────────┐┌──────────┐│  │  Histogram of all non-zero edge    │  │
│  │  │ Nodes    ││ Edges    ││  │  weights in the adjacency matrix.  │  │
│  │  │  189     ││  1,247   ││  │                                    │  │
│  │  │ (142 E,  ││ Density: ││  │  Annotation: mean, median, P95     │  │
│  │  │  47 T)   ││ 0.070    ││  │  weight. Long tail = few very      │  │
│  │  └──────────┘└──────────┘│  │  strongly connected trade pairs.   │  │
│  │  ┌──────────┐┌──────────┐│  │                                    │  │
│  │  │ Avg Deg  ││ Max Deg  ││  └────────────────────────────────────┘  │
│  │  │  13.2    ││  34      ││                                          │
│  │  └──────────┘└──────────┘│                                          │
│  └───────────────────────────┘                                          │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Degree Distribution                                             │   │
│  │                                                                  │   │
│  │  Histogram: node degree distribution                             │   │
│  │  Overlay: elementary (blue) vs target (orange) degree dists      │   │
│  │  Identifies hub trades (high degree) vs peripheral trades        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Adjacency Spy Plot (sparse matrix pattern)                      │   │
│  │                                                                  │   │
│  │  Rows/cols: trade_ids (sorted: elementary first, then target)    │   │
│  │  Dots: non-zero entries                                          │   │
│  │  Colour: weight magnitude                                       │   │
│  │                                                                  │   │
│  │  Block structure reveals: are elem↔elem, elem↔target,           │   │
│  │  target↔target connections different?                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Sub-tab: Node Analytics

**Per-node feature exploration — links graph structure to model accuracy.**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRADE GRAPH — Node Analytics                                           │
│                                                                         │
│  Cluster: [cluster_0 ▼]                                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Degree vs MAE Scatter (target trades only)                      │   │
│  │                                                                  │   │
│  │  X: node degree                                                  │   │
│  │  Y: per-trade MAE                                                │   │
│  │  Colour: product_type                                            │   │
│  │  Size: PnL volatility                                            │   │
│  │                                                                  │   │
│  │  Insight: do well-connected trades have lower error?             │   │
│  │  Regression line + R² annotation                                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Feature Embedding (2D)   │  │  Node Table                        │  │
│  │                           │  │                                    │  │
│  │  t-SNE / UMAP of encoded │  │  ┌────────┬──────┬──────┬───────┐  │  │
│  │  trade features from      │  │  │ Trade  │ Deg  │ MAE  │ Type  │  │  │
│  │  encoder_results           │  │  ├────────┼──────┼──────┼───────┤  │  │
│  │                           │  │  │ GBP|.. │  12  │.0015 │target │  │  │
│  │  Colour: cluster or type  │  │  │ GBP|.. │   8  │  -   │ elem  │  │  │
│  │  Hover: trade_id, attribs │  │  │  ...   │      │      │       │  │  │
│  │                           │  │  └────────┴──────┴──────┴───────┘  │  │
│  │  Shows feature space      │  │                                    │  │
│  │  clustering that the GNN  │  │  Sortable by degree, MAE, type    │  │
│  │  operates on              │  │  Click → highlights in graph view  │  │
│  └───────────────────────────┘  └────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Neighbourhood Analysis                                          │   │
│  │                                                                  │   │
│  │  Select trade: [GBP|EUROPEAN|1.30|0.5Y|CALL ▼]                   │   │
│  │                                                                  │   │
│  │  1-hop neighbours table + 2-hop expansion                        │   │
│  │  Aggregate stats: avg weight to neighbours, avg neighbour MAE    │   │
│  │  Mini-graph: ego network (selected node + all neighbours)        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.4 Sub-tab: Cross-Cluster Comparison

**Compare graph structure across clusters — are exotic product clusters
denser? Do higher-degree clusters have lower error?**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRADE GRAPH — Cross-Cluster Comparison                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Cluster Graph Summary Table                            [export] │   │
│  │  ┌───────────┬───────┬───────┬─────────┬──────────┬───────────┐  │   │
│  │  │ Cluster   │ Nodes │ Edges │ Density │ Avg Deg  │ Ens MAE   │  │   │
│  │  ├───────────┼───────┼───────┼─────────┼──────────┼───────────┤  │   │
│  │  │ cluster_0 │  189  │ 1,247 │  0.070  │   13.2   │  0.0023   │  │   │
│  │  │ cluster_1 │   95  │   412 │  0.092  │    8.7   │  0.0045   │  │   │
│  │  │    ...    │       │       │         │          │           │  │   │
│  │  └───────────┴───────┴───────┴─────────┴──────────┴───────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Density vs MAE Scatter   │  │  Degree Distribution Comparison   │  │
│  │                           │  │                                    │  │
│  │  Each point = 1 cluster   │  │  Overlaid degree distributions    │  │
│  │  X: graph density         │  │  for selected clusters            │  │
│  │  Y: ensemble MAE          │  │                                    │  │
│  │  Size: n_trades            │  │  Are exotic clusters sparser?     │  │
│  │  Colour: product type     │  │  Do denser graphs → better fit?   │  │
│  └───────────────────────────┘  └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data workflow — Trade Graph Explorer

```
Phase 2 (lazy per cluster):
  graph_results.joblib      → sparse_indices, sparse_values, sparse_shape
  encoder_results.joblib    → combined_features, combined_encoded
  trade_universe.json       → elementary_ids, target_ids
  target_attributes.json    → per-trade attributes for tooltips

Build on first access (per cluster):
  1. Load graph_results.joblib from version_dir
  2. Reconstruct sparse adjacency:
       from scipy.sparse import coo_matrix
       adj = coo_matrix(
           (sparse_values, (sparse_indices[0], sparse_indices[1])),
           shape=sparse_shape
       )
  3. Compute per-node statistics:
       degree = np.array(adj.sum(axis=1)).flatten()
       avg_weight = np.array(adj.mean(axis=1)).flatten()
  4. Load encoder_results.joblib for feature embeddings
  5. Build node DataFrame:
       node_id | type (elem/target) | degree | avg_weight | attributes...
  6. For interactive graph: use plotly network traces or dash-cytoscape
  7. Cache in session-level dict

For cross-cluster comparison:
  Repeat step 1-3 for all clusters, aggregate into summary table.

Library choice:
  - dash-cytoscape: best for interactive node-link diagrams with
    click/hover/search. Handles 200+ nodes smoothly.
  - plotly go.Scatter for scatter-based views (degree vs MAE, embeddings)
  - plotly go.Heatmap for adjacency spy plot

Memory: sparse adjacency + features for one cluster ≈ 1-5 MB.
  Cross-cluster summary table: negligible.
```

---

## 11. Tab 6 — Inference

**Purpose:** Run new scenarios (or future: new trades) through the
ensemble. This is the only tab that triggers Phase 3 model loading.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INFERENCE                                                              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Configuration                                                   │   │
│  │                                                                  │   │
│  │  Mode: [● New Scenarios  ○ New Trades (coming soon)]             │   │
│  │                                                                  │   │
│  │  Scenario Directory: [/data/shocks/stress_2026Q1        ] [📁]   │   │
│  │                                                                  │   │
│  │  Cluster Selection:                                              │   │
│  │  [■ Select All]  [■ cluster_0] [■ cluster_1] [□ cluster_2] ...  │   │
│  │                                                                  │   │
│  │  ┌─────────────────────────────────┐                             │   │
│  │  │  ▶  Run Inference               │  [Estimated: ~45s for 3]   │   │
│  │  └─────────────────────────────────┘                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Loading Progress                                                │   │
│  │  ████████████░░░░░░░░░  cluster_0 ✓  cluster_1 ⏳  cluster_2 ○  │   │
│  │  Phase 3: Loading models + inference context...                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ── Results (shown after completion) ──────────────────────────────     │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Portfolio Predictions    │  │  Per-Cluster Summary               │  │
│  │                           │  │                                    │  │
│  │  Histogram of predicted   │  │  ┌───────────┬──────┬──────────┐  │  │
│  │  portfolio PnL across     │  │  │ Cluster   │ Mean │ P95 Loss │  │  │
│  │  new scenarios            │  │  ├───────────┼──────┼──────────┤  │  │
│  │                           │  │  │ cluster_0 │ -120 │ -2,400   │  │  │
│  │  Key stats:               │  │  │ cluster_1 │  340 │ -1,800   │  │  │
│  │  VaR(95), VaR(99),        │  │  └───────────┴──────┴──────────┘  │  │
│  │  ES(95), mean, median     │  │                                    │  │
│  └───────────────────────────┘  └────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Scenario-Level Table                                  [export]  │   │
│  │  ┌──────────┬──────────┬───────────┬───────────┬──────────────┐  │   │
│  │  │ Scenario │ PortfPnL │ cluster_0 │ cluster_1 │ cluster_2    │  │   │
│  │  ├──────────┼──────────┼───────────┼───────────┼──────────────┤  │   │
│  │  │ scen_001 │   -2,340 │    -1,200 │      -890 │        -250  │  │   │
│  │  │ scen_002 │      450 │       180 │       200 │          70  │  │   │
│  │  │    ...   │          │           │           │              │  │   │
│  │  └──────────┴──────────┴───────────┴───────────┴──────────────┘  │   │
│  │                                                                  │   │
│  │  [Download CSV]  [Download Full NPZ]                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Stress Scenario Analysis (if applicable)                        │   │
│  │                                                                  │   │
│  │  Compare baseline (eval test) distribution vs stressed           │   │
│  │  (new scenarios) distribution for selected clusters              │   │
│  │                                                                  │   │
│  │  [Overlaid histograms: baseline vs stressed for portfolio PnL]   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data workflow — Inference tab

```
1. User selects clusters + scenario directory
2. Dash callback triggers (via background callback / long_callback):
     session.load_inference_state(cluster_ids=selected_clusters)
3. Poll session.inference_ready_clusters via dcc.Interval
4. Once ready:
     result = EnsembleInferencePipeline(
         ensemble_config=session.config,
         session=session,
     ).run()
   OR
     result = session.run_inference(
         mode="new_scenarios",
         cluster_pnl_histories=...  # if needed
     )
5. result.predictions → [n_scenarios, n_targets]
6. Portfolio: sum across columns
7. Per-cluster: slice by cluster_trade_indices
8. Display + store for export
```

---

## 12. Tab 7 — Model Governance

**Purpose:** Audit trail for model validation, Risk sign-off, regulatory
evidence. Not daily use — but essential for model governance framework.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MODEL GOVERNANCE                                                       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Ensemble Manifest                                               │   │
│  │                                                                  │   │
│  │  Version:    ens_20260324_143055_d4e5f6                          │   │
│  │  Tags:       production, hybrid_gnn_rnn                          │   │
│  │  Created:    2026-03-24 14:30:55                                 │   │
│  │  Members:    12 clusters                                         │   │
│  │  Aggregation: concat (disjoint)                                  │   │
│  │  Total trades: 564                                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Member Registry                                       [export]  │   │
│  │  ┌───────────┬──────────────┬────────────┬────────┬───────────┐  │   │
│  │  │ Cluster   │ Version      │ Model      │ Params │ Val Loss  │  │   │
│  │  ├───────────┼──────────────┼────────────┼────────┼───────────┤  │   │
│  │  │ cluster_0 │ v_20260320.. │ HybridGnn  │ 1.2M   │ 0.00231  │  │   │
│  │  │ cluster_1 │ v_20260321.. │ HybridGnn  │ 0.8M   │ 0.00345  │  │   │
│  │  │    ...    │              │            │        │           │  │   │
│  │  └───────────┴──────────────┴────────────┴────────┴───────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌───────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  Config Inspector         │  │  Version Comparison                │  │
│  │                           │  │                                    │  │
│  │  [Expandable JSON tree]   │  │  Version A: [production   ▼]      │  │
│  │                           │  │  Version B: [staging      ▼]      │  │
│  │  EnsembleConfig           │  │                                    │  │
│  │    ├─ member_configs      │  │  [Grouped bar chart: metric by     │  │
│  │    │   ├─ cluster_0       │  │   metric comparison A vs B]        │  │
│  │    │   │   ├─ data_config │  │                                    │  │
│  │    │   │   ├─ model_config│  │  Delta table:                      │  │
│  │    │   │   └─ training..  │  │  metric | A    | B    | Δ   | %Δ  │  │
│  │    │   └─ cluster_1       │  │  MAE    | .002 | .003 | +.001| +50%│  │
│  │    ├─ cluster_mapping     │  │                                    │  │
│  │    ├─ aggregation         │  │  Highlights regressions in red     │  │
│  │    └─ metadata            │  │                                    │  │
│  └───────────────────────────┘  └────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Trade-Cluster Map                                    [search]  │   │
│  │  ┌────────────────────────────────────┬───────────┬──────────┐  │   │
│  │  │ Trade ID                           │ Cluster   │ Desk     │  │   │
│  │  ├────────────────────────────────────┼───────────┼──────────┤  │   │
│  │  │ GBP|EUROPEAN|1.30|0.5Y|CALL       │ cluster_0 │ FLOW_RTS │  │   │
│  │  │ GBP|BARRIER|1.25-1.35|1Y|KO       │ cluster_3 │ EXOTICS  │  │   │
│  │  │    ...                             │           │          │  │   │
│  │  └────────────────────────────────────┴───────────┴──────────┘  │   │
│  │  Search: [________________]  (filters across all columns)       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data workflow — Governance

```
Phase 1 only:
  - EnsembleConfig.to_dict()   → JSON tree
  - member_versions            → member registry table
  - trade_cluster_map          → trade-cluster map table
  - global_trade_catalogue     → enriched trade-cluster map with attributes

Phase 2 (optional):
  - EnsembleModel.get_member_metadata() → n_parameters per member
    (requires Phase 3 if not cached in member_summary)

For version comparison:
  - Load second version's ensemble_metrics.json
  - Compute deltas
```

---

## 13. Session Gaps & Required Changes

### 13.1 Eval Pipeline — Persist prediction/target arrays

**File:** `src/rade_ml_pt/pipelines/ensemble/eval.py`

In `_save_artifacts`, after saving JSON metrics, also save:

```python
# Per-member prediction/target arrays
for cid in split_preds.get(split, {}):
    member_dir = eval_dir / "members" / cid / "predictions"
    member_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        member_dir / f"{split}.npz",
        predictions=split_preds[split][cid],
        targets=split_targets[split][cid],
    )

# Combined (portfolio-level) arrays
if combined_preds is not None:
    combined_dir = eval_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        combined_dir / f"{split}.npz",
        predictions=combined_preds,
        targets=combined_targets,
    )
```

Also save a manifest:

```python
manifest = {
    "trade_ids": config.all_trade_ids,
    "cluster_ids": config.cluster_ids,
    "cluster_trade_indices": cluster_trade_indices,
    "splits_available": list(all_split_results.keys()),
}
with open(eval_dir / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
```

### 13.2 Session — Fix plot path scanning

**File:** `src/rade_ml_pt/ensemble/session.py`

Change `_load_cluster_display` plot path logic from:

```python
member_plots_dir = artifacts_dir / "ensemble" / version / "members" / cluster_id
```

to scan the actual eval pipeline output path:

```python
for split in ("train", "val", "test"):
    plots_dir = (
        self.artifacts_dir / "ensemble" / self._ensemble_version
        / "evaluation" / "plots" / split
    )
    if plots_dir.exists():
        for p in plots_dir.glob("*.png"):
            state.plot_paths[f"{split}/{p.stem}"] = str(p)
```

### 13.3 Session — Load ensemble-level eval metrics

Add a new method and state for portfolio-level metrics:

```python
@dataclass
class EnsembleDisplayState:
    """Ensemble-wide (portfolio) evaluation artifacts."""
    ensemble_metrics: Dict[str, Dict[str, Any]]  # {split: {mae, rmse, ...}}
    member_rollup: Dict[str, Dict[str, Any]]      # {split: rollup}
    per_member_metrics: Dict[str, Dict[str, Any]]  # {split: {cid: metrics}}
    manifest: Dict[str, Any]                       # trade_ids, cluster mapping
```

Load from `evaluation/ensemble_metrics_{split}.json` etc. in Phase 2.

### 13.4 Session — Load per-split eval metrics per cluster

Change `_load_cluster_display` to read per-split metric files from the
evaluation directory, not just from `member_summary`:

```python
for split in ("train", "val", "test"):
    suffix = "" if split == "test" else f"_{split}"
    pm_path = eval_dir / f"per_member_metrics{suffix}.json"
    if pm_path.exists():
        with open(pm_path) as f:
            all_pm = json.load(f)
        if cluster_id in all_pm:
            state.eval_metrics[split] = all_pm[cluster_id]
```

### 13.5 Session — Load target_attributes.json for trade catalogue

In `_load_cluster_display`, add:

```python
target_attribs_path = version_dir / "target_attributes.json"
if target_attribs_path.exists():
    with open(target_attribs_path) as f:
        state.target_attributes = json.load(f)
```

Add `target_attributes` field to `ClusterDisplayState`.

### 13.6 Session — Fix Phase 3 stale imports

In `_load_cluster_inference`, replace the broken import:

```python
# OLD (broken):
from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import load_inference_context

# NEW:
from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
    load_inference_context_from_dir,
)
context = load_inference_context_from_dir(version_dir)
```

In `run_inference`, replace the broken `build_static_dict` /
`build_model_input_dict` imports with the same pattern used in the
updated `ensemble/infer.py` (delegate to
`HybridGnnRnnInferencePipeline` static methods).

### 13.7 Session — Add cluster attribute lookup property

```python
@property
def cluster_attributes(self) -> Dict[str, Dict[str, Any]]:
    """Return {cluster_id: {ccy: ..., desk: ..., product: ...}}."""
    if self._config is None:
        return {}
    return self._config.get_cluster_keys_for_router() or {}
```

### 13.8 Session — Add global trade catalogue builder

```python
def build_global_trade_catalogue(self) -> "pd.DataFrame":
    """
    Build a DataFrame of all target trades with their attributes,
    cluster membership, and cluster-level attributes. Used by the
    dashboard for cross-cluster filtering by desk/product/ccy.
    """
    import pandas as pd
    rows = []
    cluster_attrs = self.cluster_attributes
    for cid in self._config.cluster_ids:
        display = self.load_cluster_display(cid)
        attribs = getattr(display, "target_attributes", None)
        if not attribs:
            # Fallback: use trade_universe target_ids with cluster attrs only
            for tid in self._config.cluster_mapping.get(cid, []):
                row = {"trade_id": tid, "cluster_id": cid}
                row.update(cluster_attrs.get(cid, {}))
                rows.append(row)
            continue
        n = len(attribs.get("trade_id", []))
        for i in range(n):
            row = {"cluster_id": cid}
            for key, values in attribs.items():
                row[key] = values[i] if i < len(values) else None
            row.update(cluster_attrs.get(cid, {}))
            rows.append(row)
    return pd.DataFrame(rows)
```

---

### 13.9 Session — Add Global Prediction Store builder

```python
def build_global_prediction_store(
    self, split: str, manifest: Dict[str, Any],
) -> "GlobalPredictionStore":
    """
    Assemble unified prediction/target arrays from per-member .npz files.
    Built lazily on first access per split, cached for the session lifetime.
    """
    # See §4 for full implementation.
    # Key points:
    # - Reads each member's predictions/{split}.npz
    # - Slots columns into global trade order from manifest
    # - Returns GlobalPredictionStore dataclass
    ...
```

Add caching:

```python
# In EnsembleSession.__init__:
self._prediction_stores: Dict[str, GlobalPredictionStore] = {}

def get_prediction_store(self, split: str) -> GlobalPredictionStore:
    if split not in self._prediction_stores:
        manifest = self._load_manifest()
        self._prediction_stores[split] = build_global_prediction_store(
            self, split, manifest,
        )
    return self._prediction_stores[split]
```

### 13.10 Session — Add market data loader

```python
def load_cluster_market_data(self, cluster_id: str) -> Dict[str, Any]:
    """
    Load cluster_assets.joblib and extract risk factor shock data.
    Returns {asset_name: {rf_name: np.ndarray}}.
    Cached in _display or a dedicated dict.
    """
    version = self._member_versions[cluster_id]
    version_dir = self.registry_dir / version
    assets_path = version_dir / "cluster_assets.joblib"
    if not assets_path.exists():
        return {}
    import joblib
    portfolio = joblib.load(str(assets_path))
    result = {}
    for asset_name, asset in portfolio.items():
        result[asset_name] = {}
        for rf_name, shocks in asset.risk_factor_shocks.items():
            # Convert to numpy if needed
            if isinstance(shocks, dict):
                result[asset_name][rf_name] = np.array(
                    list(shocks.values()), dtype=np.float64
                )
            else:
                result[asset_name][rf_name] = np.asarray(shocks)
    return result
```

### 13.11 Session — Add graph data loader

```python
def load_cluster_graph_data(self, cluster_id: str) -> Dict[str, Any]:
    """
    Load graph_results.joblib and encoder_results.joblib.
    Returns dict with sparse adjacency components, features, trade IDs.
    Cached in _display or a dedicated dict.
    """
    import joblib
    version = self._member_versions[cluster_id]
    version_dir = self.registry_dir / version

    data = {}
    graph_path = version_dir / "graph_results.joblib"
    if graph_path.exists():
        data["graph_results"] = joblib.load(str(graph_path))

    encoder_path = version_dir / "encoder_results.joblib"
    if encoder_path.exists():
        data["encoder_results"] = joblib.load(str(encoder_path))

    display = self.load_cluster_display(cluster_id)
    data["trade_universe"] = display.trade_universe
    return data
```

---

## 14. Performance Budget

| Phase | Operation | Target | Notes |
|-------|-----------|--------|-------|
| **App start** | Phase 1 (metadata) | < 500ms | JSON reads only |
| **App start** | Phase 2 (display artifacts) | < 3s for 100 clusters | JSON + path scanning, no arrays |
| **Tab switch** | Any eval tab | < 200ms | Data already in memory |
| **First eval view** | Build GlobalPredictionStore (1 split) | < 3s | One-time per split, cached |
| **Subsequent eval** | Slice + aggregate from store | < 50ms | NumPy vectorised mask + sum |
| **Filter** | By desk/product/ccy | < 100ms | In-memory catalogue filter + store slice |
| **Portfolio view** | Sum + plot | < 500ms | NumPy vectorised ops |
| **Market data** | Load cluster_assets.joblib (1 cluster) | < 2s | Lazy, cached |
| **Trade graph** | Load graph_results.joblib (1 cluster) | < 1s | Lazy, cached |
| **Trade graph** | Render cytoscape (200 nodes) | < 500ms | Client-side rendering |
| **Inference** | Phase 3 load (per cluster) | 2–10s each | Parallel, progress bar |
| **Inference** | Forward pass (per cluster) | < 1s | GPU or CPU, model.eval() |

### Memory budget (100 clusters, ~50 targets each)

| Component | Estimate |
|-----------|----------|
| Phase 1 metadata | ~1 MB |
| Phase 2 JSONs (all clusters) | ~5 MB |
| Global trade catalogue | ~2 MB |
| GlobalPredictionStore (1 split, 5000 × 500) | ~20 MB (preds + targets + IDs) |
| GlobalPredictionStore (all 3 splits) | ~60 MB |
| Market data (1 cluster, 100 RFs × 5000 scen) | ~2 MB |
| Graph data (1 cluster) | ~1–5 MB |
| Phase 3 per-cluster model | ~5–50 MB each |

**Strategy:** The GlobalPredictionStore replaces per-cluster lazy `.npz`
loading for evaluation views. It is built once per split on first access
(~20 MB per split is trivially cacheable). Market data and graph data are
loaded lazily per cluster. Use `dcc.Store` for lightweight state
(< 5 MB). Keep arrays in server-side Python objects.

---

## 15. Visual Style Guide

### Theme

- **Dark background**: `#0D1117` (GitHub dark) or `#1B2028` (Bloomberg-esque)
- **Cards / panels**: `#161B22` with `1px solid #30363D` border
- **Text**: `#E6EDF3` (primary), `#8B949E` (secondary)
- **Accent**: `#3B82F6` (blue), `#10B981` (green), `#EF4444` (red)

### Typography

- **Headers**: Inter / SF Pro Display, 600 weight
- **Body / tables**: JetBrains Mono or SF Mono, 400 weight
- **Numbers**: tabular-nums for aligned columns

### Charts

- **Library**: Plotly (native Dash integration)
- **Template**: `plotly_dark` base, customised with palette above
- **Consistent palette**: Use `_PALETTE` from `ensemble/plots.py`
- **Grid**: subtle `#30363D` gridlines, no heavy axes
- **Hover**: white tooltip with key metrics, no chart junk
- **Export**: every chart gets a small camera icon for PNG download

### Tables

- **Library**: `dash_ag_grid` (AG Grid) for sortable, filterable, export-ready tables
- **Row striping**: alternate `#161B22` / `#1B2028`
- **Conditional formatting**:
  - Green: metric < P25 across clusters (good)
  - Amber: P25–P75 (acceptable)
  - Red: > P75 (needs attention)
- **Sticky headers** for scrollable tables

### KPI Cards

```
┌────────────────────┐
│  MAE               │
│  0.0023            │  ← large, bold, mono font
│  ▼ 12% vs staging  │  ← small, green/red delta
└────────────────────┘
```

- Fixed width, horizontal row of 5
- Colour border-bottom: green (good), amber, red (bad)
- Delta vs previous version shown if version comparison is active

### Responsiveness

- Minimum viewport: 1440 × 900 (typical trading desk monitor)
- Optimised for 2560 × 1440 (common dual-screen setup)
- No mobile layout required — this is a desk-mounted tool

---

## Appendix: Artifact Directory Structure

After training + evaluation, the artifacts directory should look like:

```
{artifacts_dir}/
  ensemble/
    {ensemble_version}/
      evaluation/
        manifest.json                    # trade_ids, cluster mapping, splits
        ensemble_metrics.json            # test (primary, no suffix)
        ensemble_metrics_train.json
        ensemble_metrics_val.json
        per_member_metrics.json
        per_member_metrics_train.json
        per_member_metrics_val.json
        member_rollup.json
        member_rollup_train.json
        member_rollup_val.json
        combined/
          test.npz                       # {predictions, targets} portfolio-level
          train.npz
          val.npz
        members/
          cluster_0/
            predictions/
              test.npz                   # {predictions, targets} for this member
              train.npz
              val.npz
          cluster_1/
            predictions/
              test.npz
              ...
        plots/
          test/
            member_comparison_mae.png
            cluster_performance_heatmap.png
          train/
            ...
          val/
            ...
      inference/
        predictions.csv
        inference_result.json
```

Registry version directory (per member):

```
{registry_dir}/
  {member_version}/
    model.pt
    data_config.json
    trade_universe.json
    target_attributes.json
    elementary_attributes.json
    elementary_pnl.parquet
    target_pnl.parquet
    graph_builder.pkl
    graph_results.joblib
    encoder.pkl
    encoder_results.joblib
    elementary_scaler.pkl
    target_scaler.pkl
    cluster_info.joblib
    cluster_assets.joblib
    cluster_elem_trades.joblib
    datasets/
      train.pt
      val.pt
      test.pt
```

---

## 16. Implementation Plan

### 16.1 Folder Structure

The dashboard lives under `src/ui/apps/ensemble_analytics/`, following
the existing `src/ui/` conventions (`run.py` registry, `_shared/` reuse,
`create_app()` factory).

```
src/ui/apps/ensemble_analytics/
├── __init__.py                     # create_app() export
├── app.py                          # Dash app factory, top-level layout, tab routing
├── config.py                       # App constants (default port, app title, tab IDs)
│
├── theme/                          # Dark theme (local to this app)
│   ├── __init__.py
│   ├── colors.py                   # Color palette: BG, CARD, TEXT, ACCENT, STATUS
│   ├── styles.py                   # Style dicts: container, navbar, card, table, kpi
│   └── plotly_template.py          # Custom plotly template (plotly_dark + overrides)
│
├── data/                           # Server-side data layer (bridges Session → Dash)
│   ├── __init__.py
│   ├── session_manager.py          # Singleton EnsembleSession wrapper, Phase 1+2 load
│   ├── trade_catalogue.py          # build_global_trade_catalogue() → cached DataFrame
│   ├── prediction_store.py         # GlobalPredictionStore dataclass + builder
│   ├── market_data_loader.py       # Load cluster_assets.joblib, extract RF arrays
│   └── graph_data_loader.py        # Load graph_results.joblib, adjacency + node stats
│
├── components/                     # Reusable Dash components (ensemble-specific)
│   ├── __init__.py
│   ├── kpi_card.py                 # KPI card with value + delta badge
│   ├── split_toggle.py             # Train/Val/Test radio buttons (local toggle)
│   ├── cluster_selector.py         # Cluster dropdown with attribute labels
│   ├── metric_table.py             # AG Grid table with conditional formatting
│   ├── filter_bar.py               # Multi-select filter (desk/product/ccy)
│   └── loading_progress.py         # Phase 3 loading progress bar
│
├── tabs/                           # One module per top-level tab (layout only)
│   ├── __init__.py
│   ├── overview.py                 # Tab 1: KPI cards, member table, heatmap, scatter
│   ├── evaluation/                 # Tab 2: 5 sub-tabs
│   │   ├── __init__.py             # Sub-tab container layout
│   │   ├── portfolio.py            # PnL time-series, scatter, residual, worst scenarios
│   │   ├── by_desk.py              # Desk-aggregated views
│   │   ├── by_product.py           # Product-aggregated views
│   │   ├── by_ccy.py               # CCY-aggregated views
│   │   └── by_cluster.py           # Per-cluster heatmap, trade-level metrics
│   ├── cluster_deep_dive.py        # Tab 3: forensic single-cluster view
│   ├── market_data/                # Tab 4: 4 sub-tabs
│   │   ├── __init__.py
│   │   ├── rf_summary.py           # RF inventory + coverage matrix
│   │   ├── shock_explorer.py       # Drill into RF time-series + distributions
│   │   ├── scenario_heatmap.py     # RF × scenario heatmap
│   │   └── distribution.py         # Cross-cluster + train vs inference comparison
│   ├── trade_graph/                # Tab 5: 4 sub-tabs
│   │   ├── __init__.py
│   │   ├── graph_view.py           # Interactive cytoscape network
│   │   ├── adjacency_analysis.py   # Sparsity, weight dist, degree dist
│   │   ├── node_analytics.py       # Degree vs MAE, embeddings, neighbourhood
│   │   └── cross_cluster.py        # Compare graph structure across clusters
│   ├── inference.py                # Tab 6: scenario upload, run, results
│   └── governance.py               # Tab 7: manifest, config tree, version comparison
│
├── callbacks/                      # Callback registration (one module per tab)
│   ├── __init__.py                 # register_all_callbacks(app)
│   ├── overview_cb.py
│   ├── evaluation_cb.py            # Handles all 5 eval sub-tabs
│   ├── cluster_deep_dive_cb.py
│   ├── market_data_cb.py
│   ├── trade_graph_cb.py
│   ├── inference_cb.py
│   └── governance_cb.py
│
└── figures/                        # Plotly figure builders (pure functions)
    ├── __init__.py
    ├── scatter.py                  # pred_vs_target_scatter(), residual_scatter()
    ├── timeseries.py               # pnl_timeseries(), overlaid_group_timeseries()
    ├── distributions.py            # residual_histogram(), violin_overlay(), qq_plot()
    ├── heatmaps.py                 # cluster_heatmap(), rf_scenario_heatmap(), spy()
    ├── bar_charts.py               # member_comparison_bar(), grouped_split_bar()
    ├── network.py                  # build_cytoscape_elements(), ego_network()
    └── tables.py                   # percentile_table_data(), worst_scenarios_data()
```

### 16.2 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Theme** | New dark theme local to `ensemble_analytics/theme/` | Existing `_shared/styles.py` uses a light theme for other apps; avoid breaking them |
| **Graph library** | `dash-cytoscape` | Purpose-built for interactive node-link diagrams with click/hover/search; handles 200+ nodes smoothly |
| **Separation of concerns** | `tabs/` = layout, `callbacks/` = logic, `figures/` = chart builders, `data/` = session bridge | Each layer is independently testable; callbacks never build figures directly |
| **Figure builders** | Pure functions in `figures/`: `(np.ndarray, config) → go.Figure` | No Dash imports; fully unit-testable without running a server |
| **State management** | `data/session_manager.py` holds a singleton `EnsembleSession`; `dcc.Store` for lightweight client state (< 5 MB) | Arrays stay server-side in Python; only IDs and filter selections go to the browser |

### 16.3 Phased Implementation

The dashboard is built in **4 phases**, each standalone and testable.

#### Phase 1 — Skeleton + Theme + Data Layer + Overview Tab

**Goal:** App launches, loads an ensemble from registry, renders the
Overview tab with real data.

**Estimated files:** ~25

**Steps:**

1. **Backend prerequisites** (changes to existing files):
   - `src/rade_ml_pt/pipelines/ensemble/eval.py` — add `.npz` persistence
     for predictions/targets per member per split, combined arrays, and
     `manifest.json` inside `_save_artifacts`.
   - `src/rade_ml_pt/ensemble/session.py` — fix plot path scanning, add
     `target_attributes` to `ClusterDisplayState`, add `cluster_attributes`
     property, add `build_global_trade_catalogue()`, add
     `get_prediction_store()`, fix Phase 3 stale imports to use
     `load_inference_context_from_dir`.

2. **Theme** (`theme/`):
   - `colors.py` — dark palette constants (`BG`, `CARD_BG`, `BORDER`,
     `TEXT_PRIMARY`, `TEXT_SECONDARY`, `ACCENT_BLUE`, `STATUS_GREEN`,
     `STATUS_AMBER`, `STATUS_RED`).
   - `styles.py` — style dicts for container, navbar, card, KPI card,
     table row, tab bar.
   - `plotly_template.py` — `make_dark_template()` returning a
     `plotly.graph_objects.layout.Template` customised from `plotly_dark`.

3. **Data layer** (`data/`):
   - `session_manager.py` — `init_session(registry_dir, artifacts_dir,
     version)`, `get_session()`, Phase 1 + 2 load on init.
   - `trade_catalogue.py` — `get_trade_catalogue()` (lazy build, cached).
   - `prediction_store.py` — `GlobalPredictionStore` dataclass,
     `get_prediction_store(split)` (lazy build, cached).

4. **Components** (`components/`):
   - `kpi_card.py` — `kpi_card(label, value, delta=None, status=None)`.
   - `split_toggle.py` — `split_toggle(id_prefix)` returning radio buttons.
   - `cluster_selector.py` — `cluster_dropdown(id)` with attribute labels.
   - `metric_table.py` — `metric_table(id, columns, data)` with AG Grid.

5. **Overview tab** (`tabs/overview.py`):
   - Layout: KPI card row, scatter + bar chart row, member table, heatmap.
   - Uses `split_toggle` for local split selection.

6. **Overview callbacks** (`callbacks/overview_cb.py`):
   - Split toggle → update KPI cards, scatter, bar chart, table, heatmap.
   - Ensemble version change → reload session.

7. **App shell** (`app.py`):
   - `create_app()` factory.
   - Top-level layout: header with ensemble dropdown, tab bar (all 7 tabs
     listed but only Overview wired), content area.
   - Register in `src/ui/run.py` under `APP_REGISTRY`.

8. **Figures** (`figures/`):
   - `scatter.py` — `pred_vs_target_scatter(pred, target)`.
   - `bar_charts.py` — `member_comparison_bar(metrics_dict, metric_name)`.
   - `heatmaps.py` — `cluster_performance_heatmap(metrics_dict)`.

**Test:** `python -m src.ui.run ensemble_analytics --registry-dir /path
--artifacts-dir /path --version production`

---

#### Phase 2 — Evaluation Tab (all 5 sub-tabs)

**Goal:** Full pred vs target analytics at portfolio, desk, product, ccy,
and cluster level.

**Estimated files:** ~12

**Prerequisites:** Phase 1 complete; `GlobalPredictionStore` and
`global_trade_catalogue` working.

**Steps:**

1. **Evaluation sub-tab container** (`tabs/evaluation/__init__.py`):
   - Sub-tab bar: Portfolio | By Desk | By Product | By CCY | By Cluster.
   - Split toggle shared across sub-tabs.

2. **Portfolio sub-tab** (`tabs/evaluation/portfolio.py` +
   `callbacks/evaluation_cb.py`):
   - PnL time-series (pred vs target overlaid).
   - Pred vs target scatter with R2.
   - Residual histogram with stats annotation.
   - Percentile comparison table.
   - Worst scenarios table (clickable → cluster deep dive).

3. **By Desk / By Product / By CCY sub-tabs** — same layout pattern,
   parameterised by the grouping column from the trade catalogue:
   - Filter bar (multi-select).
   - Overlaid group time-series.
   - Residual box plot.
   - Per-group metrics table.
   - Small-multiples scatter grid.

4. **By Cluster sub-tab**:
   - Cluster selector dropdown.
   - Per-trade PnL residual heatmap.
   - Cluster-level scatter.
   - Trade-level metrics table.
   - Violin / ridgeline overlay.

5. **New figures** (`figures/`):
   - `timeseries.py` — `pnl_timeseries()`, `overlaid_group_timeseries()`.
   - `distributions.py` — `residual_histogram()`, `violin_overlay()`.
   - `tables.py` — `percentile_table_data()`, `worst_scenarios_data()`.

**Test:** Navigate to Evaluation tab; select splits and filters; verify
charts update correctly with real eval data.

---

#### Phase 3 — Cluster Deep Dive + Market Data + Trade Graph

**Goal:** Forensic single-cluster analysis, market data transparency,
and GNN graph exploration.

**Estimated files:** ~15

**Prerequisites:** Phase 2 complete.

**Steps:**

1. **Cluster Deep Dive** (`tabs/cluster_deep_dive.py` +
   `callbacks/cluster_deep_dive_cb.py`):
   - Cluster header card (attributes, config summary).
   - Split comparison table + grouped bar chart.
   - Training convergence plot (from plot PNGs).
   - Per-trade scatter matrix (select up to 6 trades).
   - Elementary PnL explorer (time-series of input PnL).
   - Data configuration summary cards.

2. **Market Data** (`tabs/market_data/` + `callbacks/market_data_cb.py`):
   - `data/market_data_loader.py` — load `cluster_assets.joblib`, walk
     asset portfolio, extract RF shock arrays into
     `{(cluster, asset, rf): np.ndarray}`.
   - RF Summary: inventory table + coverage matrix.
   - Shock Explorer: RF time-series + distribution + multi-RF overlay.
   - Scenario Heatmap: RF x scenario diverging heatmap.
   - Distribution: cross-cluster comparison + correlation matrix.

3. **Trade Graph** (`tabs/trade_graph/` + `callbacks/trade_graph_cb.py`):
   - `data/graph_data_loader.py` — load `graph_results.joblib`, reconstruct
     sparse adjacency via `scipy.sparse.coo_matrix`, compute per-node
     degree/weight stats.
   - Graph View: `dash-cytoscape` interactive network, colour/size
     controls, node detail panel, search.
   - Adjacency Analysis: graph stats cards, edge weight histogram, degree
     distribution, adjacency spy plot.
   - Node Analytics: degree vs MAE scatter, t-SNE/UMAP embedding,
     neighbourhood table.
   - Cross-Cluster Comparison: summary table + density vs MAE scatter.

4. **New figures** (`figures/`):
   - `network.py` — `build_cytoscape_elements()`, `ego_network()`.
   - `heatmaps.py` — extend with `rf_scenario_heatmap()`,
     `adjacency_spy_plot()`.

**Test:** Select a cluster; verify market data loads and charts render;
verify trade graph is interactive and node clicks work.

---

#### Phase 4 — Inference + Model Governance + Polish

**Goal:** Complete the app with live inference capability and governance
audit trail.

**Estimated files:** ~10

**Prerequisites:** Phase 3 complete; `EnsembleSession.load_inference_state`
and `EnsembleInferencePipeline` working.

**Steps:**

1. **Inference** (`tabs/inference.py` + `callbacks/inference_cb.py`):
   - Configuration panel: mode selector, scenario directory input, cluster
     selection checkboxes.
   - Run button with `dash.long_callback` (or `background_callback_manager`).
   - Loading progress: poll `session.inference_ready_clusters` via
     `dcc.Interval`.
   - Results: portfolio PnL histogram with VaR/ES, per-cluster summary
     table, scenario-level table with export.
   - Stress comparison: baseline vs stressed distribution overlay.
   - `components/loading_progress.py` — progress bar component.

2. **Model Governance** (`tabs/governance.py` +
   `callbacks/governance_cb.py`):
   - Ensemble manifest card.
   - Member registry table.
   - Config inspector (expandable JSON tree via `dash_renderjson` or
     recursive `html.Details`).
   - Version comparison: two dropdowns + delta table + grouped bar chart.
   - Trade-cluster map: searchable AG Grid table.

3. **Polish**:
   - Loading skeletons / shimmer placeholders while data loads.
   - Error boundaries (graceful messages when artifacts are missing).
   - Keyboard shortcuts (if useful).
   - Final styling pass: consistent spacing, alignment, font sizes.

**Test:** Full end-to-end: load ensemble → browse evaluation → drill into
cluster → inspect market data → explore trade graph → run inference →
review governance. Export predictions CSV.

### 16.4 Dependencies

Required Python packages (add to `requirements-ui.txt`):

```
dash>=2.14
dash-ag-grid>=31.0
dash-cytoscape>=0.3
plotly>=5.18
scipy>=1.11          # for sparse adjacency reconstruction
scikit-learn>=1.3    # for t-SNE/UMAP in node analytics (optional)
```

### 16.5 Registration

After Phase 1, register the app in `src/ui/run.py`:

```python
APP_REGISTRY = {
    "pricing_calculator": ("src.ui.apps.pricing_calculator.app", "create_app"),
    "ensemble_analytics": ("src.ui.apps.ensemble_analytics.app", "create_app"),
}
```

Run with:

```bash
python -m src.ui.run ensemble_analytics \
    --registry-dir /data/model_store \
    --artifacts-dir /data/eval_output \
    --version production \
    --port 8051
```
