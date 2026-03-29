# Ensemble Analytics Dashboard

> **Professional analytics dashboard for Hybrid GNN-RNN ensemble models.**
> Built for Traders, Front Office, and Risk teams to evaluate model
> performance, inspect market data, explore trade graphs, run live
> inference, and audit model governance — all from a single interface.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Running the Dashboard](#4-running-the-dashboard)
5. [Folder Structure](#5-folder-structure)
6. [Data Layer and EnsembleSession](#6-data-layer-and-ensemblesession)
7. [Required Data Structures](#7-required-data-structures)
8. [Theme and Styling](#8-theme-and-styling)
9. [Tabs and Visualisations](#9-tabs-and-visualisations)
   - 9.1 [Overview](#91-tab-1--overview)
   - 9.2 [Evaluation](#92-tab-2--evaluation)
   - 9.3 [Cluster Deep Dive](#93-tab-3--cluster-deep-dive)
   - 9.4 [Market Data](#94-tab-4--market-data)
   - 9.5 [Trade Graph Explorer](#95-tab-5--trade-graph-explorer)
   - 9.6 [Inference](#96-tab-6--inference)
   - 9.7 [Model Governance](#97-tab-7--model-governance)
10. [Reusable Components](#10-reusable-components)
11. [Figure Builders](#11-figure-builders)
12. [Callback Architecture](#12-callback-architecture)
13. [Performance Considerations](#13-performance-considerations)
14. [Adding New Tabs or Figures](#14-adding-new-tabs-or-figures)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Overview

The Ensemble Analytics dashboard is a Plotly Dash application that provides
a complete analytical interface for an ensemble of Hybrid GNN-RNN models.
Each ensemble comprises multiple *member models*, one per cluster of trades.
The dashboard presents:

- **Portfolio-level KPIs** — ensemble-wide MAE, RMSE, P95/P99 absolute error
- **Multi-dimensional evaluation** — predictions vs targets sliced by desk, product, currency, and individual cluster
- **Forensic cluster drill-down** — per-trade scatter, convergence plots, residual analysis, data config inspection
- **Market data exploration** — risk-factor inventories, shock distributions, scenario heatmaps, QQ plots
- **Trade graph visualisation** — interactive Cytoscape network, adjacency analysis, degree-vs-error scatter
- **Live inference** — Phase 3 model loading, new-scenario pricing, stress-test comparison, CSV export
- **Model governance** — ensemble manifest, version comparison, config inspector, trade-cluster audit trail

### Target Audience

| User | Primary Tabs |
|------|-------------|
| **Traders** | Overview, Evaluation (Portfolio, Desk, CCY), Inference |
| **Front Office** | Overview, Evaluation (all), Cluster Deep Dive, Inference |
| **Risk** | Evaluation (all), Market Data, Trade Graph, Governance |
| **Quant Dev** | All tabs, especially Cluster Deep Dive and Governance |

---

## 2. Architecture

The dashboard follows a strict **unidirectional data flow** architecture:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EnsembleSession                              │
│              (Phase 1: metadata, Phase 2: display, Phase 3: models)  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                     Data Layer (singletons)
           ┌─────────────────┼──────────────────┐
           │                 │                   │
    session_manager   trade_catalogue    prediction_store
           │           market_data_loader   graph_data_loader
           │
    ┌──────┴──────┐
    │  Callbacks  │  ← Dash reactive event handlers
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │   Figures   │  ← Pure functions: data → go.Figure
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ Components  │  ← Reusable UI building blocks
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │   Layouts   │  ← Static tab skeletons with id placeholders
    └─────────────┘
```

**Key principles:**

- **Separation of concerns** — data access, figure building, component construction, layout, and callback logic are each in their own layer.
- **No side effects in figures** — all figure builders are pure functions that receive data and return `go.Figure`. They never call the session or modify state.
- **Lazy loading** — expensive data (prediction stores, market data, graph data) is loaded on first access, not at startup. Phase 3 model loading only triggers when the user explicitly requests it.
- **Singleton session** — the `session_manager` module holds one `EnsembleSession` instance. All data access routes through it. Version changes tear down and rebuild the session.

---

## 3. Prerequisites

### Python Packages

```
dash>=2.14
dash-bootstrap-components>=1.5
dash-ag-grid>=31.0
dash-cytoscape>=0.3
plotly>=5.18
numpy
pandas
scipy
```

### Backend Dependencies

The dashboard consumes data produced by:

1. **Ensemble Training Pipeline** (`EnsembleTrainPipeline`) — trains member models, registers versions
2. **Ensemble Evaluation Pipeline** (`EnsembleEvalPipeline`) — evaluates all members, saves metrics/predictions/plots
3. **EnsembleRegistry** — stores ensemble config, member versions, trade-cluster maps

All three must have run successfully before the dashboard can display meaningful data.

---

## 4. Running the Dashboard

### Startup Script

```python
from src.ui.apps.ensemble_analytics import create_app

app = create_app(
    registry_dir="/path/to/model_registry",
    artifacts_dir="/path/to/evaluation_artifacts",
    version="latest",       # or a tag like "production", or a specific version ID
    debug=True,
)

app.run(host="0.0.0.0", port=8051)
```

### What Happens at Startup

1. The `ensemble_dark` Plotly template is registered globally
2. The Dash app is created with `dbc.themes.DARKLY` stylesheet
3. `session_manager.initialise()` runs:
   - Creates `EnsembleSession(registry_dir, artifacts_dir)`
   - Runs **Phase 1** (`load_metadata`) — loads config, member versions, trade-cluster map
   - Runs **Phase 2** (`load_display_artifacts`) — loads eval metrics, plot paths, prediction arrays
4. The top-level layout is built (navbar + tab bar + content container)
5. All callbacks are registered via `register_all_callbacks(app)`

### Version Switching

The navbar contains a version dropdown. Selecting a different version triggers:

```python
session_manager.reload(new_version)     # tears down session, re-runs Phase 1 + 2
trade_catalogue.invalidate()            # clears cached catalogue DataFrame
# Overview tab re-renders automatically
```

---

## 5. Folder Structure

```
ensemble_analytics/
├── __init__.py                          # exports create_app
├── app.py                               # Dash app factory
├── config.py                            # Tab IDs, split names, display constants
│
├── theme/
│   ├── __init__.py
│   ├── colors.py                        # Colour palette tokens (BG, text, accent)
│   ├── styles.py                        # CSS-in-Python style dicts
│   └── plotly_template.py               # Custom Plotly template (ensemble_dark)
│
├── data/                                # Data layer — bridges session to callbacks
│   ├── __init__.py
│   ├── session_manager.py               # Singleton EnsembleSession wrapper
│   ├── trade_catalogue.py               # Cached global trade catalogue (pd.DataFrame)
│   ├── prediction_store.py              # Cached GlobalPredictionStore per split
│   ├── market_data_loader.py            # Delegates to session.load_cluster_market_data
│   └── graph_data_loader.py             # Delegates to session.load_cluster_graph_data
│
├── components/                          # Reusable UI building blocks
│   ├── __init__.py
│   ├── kpi_card.py                      # Metric display card with delta badge
│   ├── split_toggle.py                  # Train/Val/Test radio buttons
│   ├── cluster_selector.py              # Cluster dropdown with attribute labels
│   ├── metric_table.py                  # Dark-themed AG Grid wrapper
│   ├── filter_bar.py                    # Multi-select dropdowns (desk, product, ccy)
│   └── loading_progress.py             # Phase 3 loading progress bar
│
├── figures/                             # Pure figure builders: data → go.Figure
│   ├── __init__.py
│   ├── scatter.py                       # pred_vs_target_scatter, residual_scatter
│   ├── timeseries.py                    # pnl_timeseries, overlaid_group_timeseries
│   ├── distributions.py                 # residual_histogram, violin_overlay, qq_plot
│   ├── heatmaps.py                      # cluster_heatmap, multi_metric_cluster_heatmap,
│   │                                    #   rf_scenario_heatmap, adjacency_spy
│   ├── bar_charts.py                    # member_comparison_bar, grouped_split_bar
│   ├── tables.py                        # percentile_table_data, worst_scenarios_data
│   └── network.py                       # build_cytoscape_elements, ego_network
│
├── tabs/                                # Layout modules (static skeletons)
│   ├── __init__.py
│   ├── overview.py                      # Tab 1
│   ├── evaluation/                      # Tab 2 (5 sub-tabs)
│   │   ├── __init__.py                  # Sub-tab container
│   │   ├── portfolio.py
│   │   ├── by_desk.py
│   │   ├── by_product.py
│   │   ├── by_ccy.py
│   │   └── by_cluster.py
│   ├── cluster_deep_dive.py             # Tab 3
│   ├── market_data/                     # Tab 4 (4 sub-tabs)
│   │   ├── __init__.py
│   │   ├── rf_summary.py
│   │   ├── shock_explorer.py
│   │   ├── scenario_heatmap.py
│   │   └── distribution.py
│   ├── trade_graph/                     # Tab 5 (4 sub-tabs)
│   │   ├── __init__.py
│   │   ├── graph_view.py
│   │   ├── adjacency_analysis.py
│   │   ├── node_analytics.py
│   │   └── cross_cluster.py
│   ├── inference.py                     # Tab 6
│   └── governance.py                    # Tab 7
│
└── callbacks/                           # Dash callback registrations
    ├── __init__.py                      # register_all_callbacks + tab router
    ├── overview_cb.py                   # Tab 1 callbacks
    ├── evaluation_cb.py                 # Tab 2 callbacks (all 5 sub-tabs)
    ├── cluster_deep_dive_cb.py          # Tab 3 callbacks
    ├── market_data_cb.py                # Tab 4 callbacks (all 4 sub-tabs)
    ├── trade_graph_cb.py                # Tab 5 callbacks (all 4 sub-tabs)
    ├── inference_cb.py                  # Tab 6 callbacks
    └── governance_cb.py                 # Tab 7 callbacks
```

**57 files total**, organised into 7 layers: config, theme, data, components,
figures, tabs (layouts), callbacks.

---

## 6. Data Layer and EnsembleSession

Every piece of data displayed in the dashboard originates from
`EnsembleSession`, a three-phase lifecycle manager documented in
[`docs/session.md`](../../../docs/session.md).

### How the Dashboard Accesses Data

The `data/` package provides thin wrapper modules that callbacks import
instead of touching the session directly:

```python
# In any callback file:
from src.ui.apps.ensemble_analytics.data.session_manager import get_session
from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data
from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data
```

### Data Flow Per Module

| Data Module | Wraps | Caching | Invalidated On |
|-------------|-------|---------|----------------|
| `session_manager` | `EnsembleSession` singleton | Module-level global | `reload()` creates new session |
| `trade_catalogue` | `session.build_global_trade_catalogue()` | Module-level `_cache` | `invalidate()` on version change |
| `prediction_store` | `session.get_prediction_store(split)` | Session-internal `_prediction_stores` | New session on reload |
| `market_data_loader` | `session.load_cluster_market_data(cid)` | Session-internal `_market_data_cache` | New session on reload |
| `graph_data_loader` | `session.load_cluster_graph_data(cid)` | Session-internal `_graph_data_cache` | New session on reload |

### Session Phase to Dashboard Mapping

| Session Phase | What It Provides | Dashboard Usage |
|---------------|-----------------|-----------------|
| **Phase 1** (metadata) | `config`, `member_versions`, `trade_cluster_map`, `cluster_attributes` | Navbar stats, all dropdown selectors, Governance manifest |
| **Phase 2** (display) | Per-cluster metrics, plot paths, prediction `.npz`, ensemble metrics, manifest | Overview KPIs, all Evaluation charts, Cluster Deep Dive, Market Data, Trade Graph |
| **Phase 3** (inference) | Loaded `nn.Module` models, inference contexts, baseline PnL | Inference tab only (triggered by "Load Models" button) |

---

## 7. Required Data Structures

The dashboard expects the following artifacts to exist on disk. All are
produced by the ensemble training and evaluation pipelines.

### From EnsembleRegistry (Phase 1)

```
registry_dir/ensemble/{version}/
  ensemble_config.json          # EnsembleConfig: cluster_mapping, aggregation, etc.
  member_versions.json          # {cluster_id: member_version_string}
  trade_cluster_map.json        # {trade_id: cluster_id}
  member_summary.json           # {cluster_id: {n_trades, mae, ...}}
  index.json                    # {tag: version} mapping
```

### From Member Registry (Phase 2 + 3)

```
registry_dir/{member_version}/
  trade_universe.json           # {elementary_ids: [...], target_ids: [...]}
  target_attributes.json        # {trade_id: [...], ccy: [...], desk: [...], ...}
  data_config.json              # HybridGnnRnnDataConfig (seq_length, transform_type, ...)
  cluster_assets.joblib         # {asset_name: Asset object with risk_factor_shocks}
  graph_results.joblib          # {sparse_indices, sparse_values, sparse_shape}
  encoder_results.joblib        # Encoder feature data
  model.pt                      # Phase 3 only: nn.Module state dict
  elementary_pnl.parquet        # Phase 3 only: baseline PnL for inference
```

### From Evaluation Pipeline (Phase 2)

```
artifacts_dir/ensemble/{version}/evaluation/
  manifest.json                 # {trade_ids, cluster_ids, cluster_trade_indices, splits_available}
  ensemble_metrics.json         # {mae, rmse, max_ae, p95_ae, p99_ae, ...} for test
  ensemble_metrics_train.json   # Same for train split
  ensemble_metrics_val.json     # Same for val split
  per_member_metrics.json       # {cluster_id: {mae, rmse, ...}} for test
  per_member_metrics_train.json
  per_member_metrics_val.json
  member_rollup.json            # Aggregated member stats
  plots/{split}/*.png           # Saved evaluation plots
  members/{cluster_id}/predictions/{split}.npz  # {predictions, targets} arrays
```

### GlobalPredictionStore (Built at Runtime)

The `GlobalPredictionStore` is assembled lazily from per-member `.npz` files.
It produces unified arrays where columns are aligned to the global trade order
from `manifest.json`:

```python
store = get_prediction_store("test")
store.predictions   # shape: [n_scenarios, n_total_targets]
store.targets       # shape: [n_scenarios, n_total_targets]
store.trade_ids     # list[str], length = n_total_targets
store.cluster_ids   # list[str], per-target cluster membership
```

This enables arbitrary cross-cluster slicing:

```python
# Get predictions for all EUR trades across all clusters
catalogue = get_trade_catalogue()
mask = catalogue["ccy"] == "EUR"
eur_preds = store.predictions[:, mask.values]
```

### Global Trade Catalogue (Built at Runtime)

The trade catalogue is a flat `pd.DataFrame` built from per-member
`target_attributes.json` joined with cluster-level attributes:

```python
catalogue = get_trade_catalogue()
# Columns: trade_id, cluster_id, ccy, desk, product_type, ...
```

---

## 8. Theme and Styling

The dashboard uses a dark theme inspired by GitHub's dark palette.

### Colour Tokens (`theme/colors.py`)

| Token | Hex | Usage |
|-------|-----|-------|
| `BG_PRIMARY` | `#0d1117` | Page background |
| `BG_CARD` | `#161b22` | Card and panel backgrounds |
| `BORDER` | `#30363d` | Borders and grid lines |
| `TEXT_PRIMARY` | `#e6edf3` | Primary text |
| `TEXT_SECONDARY` | `#8b949e` | Labels, secondary text |
| `ACCENT_BLUE` | `#58a6ff` | Primary accent, prediction lines |
| `ACCENT_GREEN` | `#3fb950` | Target lines, success states |
| `ACCENT_RED` | `#f85149` | Error states, worst-case indicators |
| `ACCENT_AMBER` | `#d29922` | Warning states, validation split |
| `ACCENT_PURPLE` | `#bc8cff` | Tertiary accent |

### Plotly Template (`theme/plotly_template.py`)

A custom template `ensemble_dark` extends `plotly_dark` with the dashboard
palette. It is registered globally at app startup:

```python
pio.templates["ensemble_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "ensemble_dark"
```

All figure builders automatically inherit the dark background, font, grid
colour, and colour cycle — no per-figure styling required.

### Style Dicts (`theme/styles.py`)

Pre-built CSS-in-Python dictionaries for common patterns:

| Dict | Usage |
|------|-------|
| `CONTAINER_STYLE` | Full-page container |
| `NAVBAR_STYLE` | Top navigation bar |
| `CARD_STYLE` | Card panels (border, padding, radius) |
| `CARD_HEADER_STYLE` | Uppercase label above card content |
| `KPI_VALUE_STYLE` | Large metric numbers (28px, bold) |
| `TABLE_STYLE` | AG Grid base styling |
| `SECTION_TITLE_STYLE` | Section headings within tabs |

---

## 9. Tabs and Visualisations

### Global Controls

- **Version selector** (navbar) — dropdown of all registered ensemble versions/tags. Changing version reloads the session.
- **Split toggle** (per-tab) — Train / Val / Test radio buttons. Local to each tab that needs it, not a global control.

---

### 9.1 Tab 1 — Overview

**Purpose:** Production-readiness snapshot. The first thing traders and risk
managers see — is the ensemble performing well?

**Data sources:**
- `session.ensemble_display.ensemble_metrics[split]` — portfolio-level KPIs
- `session.ensemble_display.per_member_metrics[split]` — per-cluster metrics
- `get_prediction_store(split)` — portfolio scatter
- `session.cluster_attributes` — table enrichment

**Visualisations:**

| Element | Description |
|---------|-------------|
| **KPI cards** | MAE, RMSE, Max AE, P95 AE, P99 AE for the selected split |
| **Portfolio scatter** | Aggregated predictions vs targets (summed across all trades), with 45-degree reference line. Uses `go.Scattergl` for performance. |
| **Member comparison bar** | Horizontal bar chart of MAE per cluster, sorted ascending |
| **Multi-metric heatmap** | Rows = metrics (MAE, RMSE, Max AE, P95, P99), columns = clusters. Colour intensity encodes value. |
| **Member table** | Sortable AG Grid with cluster ID, attributes (ccy, desk, product), all metrics, and trade count. Cells are conditionally formatted: green (< P25), amber (P25-P75), red (> P75). |

**Callback:** 1 callback, 5 outputs, triggered by split toggle.

---

### 9.2 Tab 2 — Evaluation

**Purpose:** Deep evaluation analytics sliced by business dimension. This is
the primary analytical tab for Front Office and Risk.

**Structure:** 5 sub-tabs, each with its own layout and callbacks.

#### 9.2.1 Portfolio

Full-book PnL analysis — predictions vs targets summed across all trades.

| Element | Description |
|---------|-------------|
| **PnL time-series** | Overlaid prediction and target lines across scenario index |
| **Pred vs Target scatter** | With 45-degree reference line |
| **Residual histogram** | With annotation box (mean, std, skew, kurtosis, % within 1/2 sigma) |
| **Percentile table** | P1, P5, P25, P50, P75, P95, P99 for predicted, target, diff, abs error |
| **Worst scenarios table** | Top 20 highest absolute error scenarios |

#### 9.2.2 By Desk / By Product / By CCY

All three follow the same pattern — group trades by a catalogue column,
aggregate predictions per group, and display:

| Element | Description |
|---------|-------------|
| **Filter bar** | Multi-select dropdown for the grouping column |
| **Overlaid time-series** | One line per group, colour-coded |
| **Residual violin/box** | Distribution comparison across groups |
| **Scatter grid** | Small multiples — one pred-vs-target scatter per group (up to 4 columns) |
| **Metrics table** | MAE, RMSE, and trade count per group |

The **By CCY** sub-tab includes an additional **cross-currency residual
correlation heatmap** — a matrix showing how residual errors correlate
across currencies, useful for identifying hedging blind spots.

#### 9.2.3 By Cluster

Per-cluster forensics within the Evaluation context.

| Element | Description |
|---------|-------------|
| **Cluster selector** | Dropdown with attribute labels |
| **Per-trade PnL heatmap** | Rows = scenarios, columns = trades, colour = residual (capped at 500 scenarios) |
| **Cluster scatter** | Aggregated pred vs target for the selected cluster |
| **Per-trade violin** | Residual distribution for each trade in the cluster |
| **Trade-level metrics** | MAE, RMSE, Max AE per individual trade |

**Data sources:**
- `get_prediction_store(split)` — sliced by catalogue filter
- `get_trade_catalogue()` — grouping and filtering

**Callbacks:** 12 callbacks across all sub-tabs.

---

### 9.3 Tab 3 — Cluster Deep Dive

**Purpose:** Forensic single-cluster view. Quant devs use this to
investigate why a specific cluster is underperforming or to validate
training convergence.

**Data sources:**
- `session.load_cluster_display(cluster_id)` — per-cluster display state
- `session.ensemble_display.per_member_metrics` — cross-split comparison
- `get_prediction_store(split)` — scatter and residual data
- `get_trade_catalogue()` — column index resolution

**Visualisations:**

| Element | Description |
|---------|-------------|
| **Header card** | Cluster ID, version, attributes (ccy, desk, product), n_elementary, n_target, seq_length, transform_type |
| **Split comparison table** | Side-by-side metrics for train/val/test |
| **Convergence plot** | Training loss curve loaded from saved PNG |
| **Pred vs Target scatter** | For the selected cluster and split |
| **Residual histogram** | With statistical annotations |
| **Per-trade scatter matrix** | Small multiples of first 6 trades using `plotly.subplots` |
| **Elementary PnL explorer** | Placeholder for elementary-level drill-down |
| **Config summary table** | Structured display of data_config.json parameters |

**Callbacks:** 2 callbacks (cluster selector render + main update with 8 outputs).

---

### 9.4 Tab 4 — Market Data

**Purpose:** Explore the risk-factor shock data that drives model inputs.
Risk teams use this to validate scenario coverage and identify distributional
anomalies.

**Data source:** `session.load_cluster_market_data(cluster_id)` — returns
`{asset_name: {rf_name: np.ndarray}}` per cluster.

**Structure:** 4 sub-tabs.

#### 9.4.1 RF Summary

| Element | Description |
|---------|-------------|
| **RF inventory table** | Risk factor name, number of clusters using it, cluster list |
| **Coverage heatmap** | Binary matrix: rows = risk factors, columns = clusters, blue = present |

#### 9.4.2 Shock Explorer

| Element | Description |
|---------|-------------|
| **Asset/RF selectors** | Dynamically populated from cluster market data |
| **Shock time-series** | Line chart of shock values across scenarios |
| **Shock distribution** | Histogram of shock values |
| **Summary statistics** | Mean, Std, Min, Max, N |

#### 9.4.3 Scenario Heatmap

| Element | Description |
|---------|-------------|
| **RF x Scenario heatmap** | Full shock surface for the selected cluster. Rows = risk factors, columns = scenarios, colour = shock magnitude (RdBu diverging, centred at 0). Downsampled to 200 scenarios for rendering performance. |

#### 9.4.4 Distribution

| Element | Description |
|---------|-------------|
| **RF violin overlay** | Violin plots of shock distributions for up to 10 risk factors |
| **QQ plot** | Normal QQ plot of the first risk factor |
| **RF correlation matrix** | Heatmap of pairwise correlations between risk factors (up to 20) |

**Callbacks:** 6 callbacks (selector + sub-tab routing + 4 content updaters).

---

### 9.5 Tab 5 — Trade Graph Explorer

**Purpose:** Visualise and analyse the trade adjacency graph that the GNN
component uses. Quant devs and risk managers can verify graph structure and
understand how trade connectivity relates to model error.

**Data source:** `session.load_cluster_graph_data(cluster_id)` — returns
`{graph_results, encoder_results, trade_universe}`.

**Structure:** 4 sub-tabs.

#### 9.5.1 Graph View

| Element | Description |
|---------|-------------|
| **Layout selector** | Force-directed (COSE), Circular, Grid |
| **Edge threshold slider** | Filter edges by minimum absolute weight |
| **Search box** | Find a specific trade node |
| **Interactive Cytoscape** | Nodes coloured by type (blue = elementary, amber = target), edges from sparse adjacency. Click a node for details. |
| **Node detail panel** | Displays trade ID, type, and any extra attributes from the graph data |

#### 9.5.2 Adjacency Analysis

| Element | Description |
|---------|-------------|
| **Graph statistics** | Node count, edge count, density, mean weight, max weight |
| **Edge weight histogram** | Distribution of edge weights |
| **Degree distribution** | Histogram of node degrees |
| **Adjacency spy plot** | Scatter plot of non-zero entries in the adjacency matrix, coloured by weight |

#### 9.5.3 Node Analytics

| Element | Description |
|---------|-------------|
| **Degree vs MAE scatter** | Each node plotted by its graph degree (x) vs its model prediction error (y). Helps identify whether highly connected nodes are easier or harder to predict. |
| **Node feature table** | Trade ID, degree, and MAE for each node |

#### 9.5.4 Cross-Cluster

| Element | Description |
|---------|-------------|
| **Graph structure table** | Nodes, edges, density, and mean weight for every cluster |
| **Density bar chart** | Visual comparison of graph density across clusters |

**Callbacks:** 7 callbacks.

---

### 9.6 Tab 6 — Inference

**Purpose:** Run live model inference for new scenarios. The only tab that
triggers Phase 3 (model loading). Designed for traders who want to price
new scenario sets and compare against the evaluation baseline.

**Data sources:**
- `session.all_inference_ready` / `session.inference_ready_clusters` — loading status
- `session.load_inference_state()` — Phase 3 trigger
- `session.run_inference()` — execution
- `get_prediction_store("test")` — baseline for stress comparison

**Visualisations:**

| Element | Description |
|---------|-------------|
| **Loading status** | Shows how many clusters are loaded, with progress bar |
| **Load Models button** | Triggers parallel Phase 3 loading |
| **Inference controls** | Mode selector (New Scenarios / New Trades), scenario directory input, Run button |
| **Results summary** | Mode, scenario count, target count |
| **Portfolio PnL histogram** | Distribution of inferred portfolio PnL |
| **VaR / ES** | Value-at-Risk (95%) and Expected Shortfall displayed inline |
| **Scenario-level table** | Per-scenario portfolio PnL and per-cluster breakdown (first 200 rows) |
| **Stress comparison** | Overlaid histograms of baseline (test) vs inference (stressed) portfolio PnL |
| **CSV export** | Download inference results as CSV |

**Callbacks:** 4 callbacks (status check, model loading, inference execution, CSV download).

---

### 9.7 Tab 7 — Model Governance

**Purpose:** Audit trail and configuration inspection. Risk and model
validation teams use this to verify what is deployed and compare versions.

**Data sources:**
- `session.config` — ensemble configuration
- `session.ensemble_display.manifest` — evaluation manifest
- `session.member_versions` — per-cluster versions
- `session.trade_cluster_map` — trade-to-cluster assignment
- `session._ens_registry.list_versions()` — available versions
- Disk: `ensemble_metrics.json` from comparison version

**Visualisations:**

| Element | Description |
|---------|-------------|
| **Ensemble manifest** | Version, cluster count, trade count, aggregation strategy, execution strategy, splits available |
| **Member registry table** | Cluster ID, version string, trade count for each member |
| **Config inspector** | Collapsible JSON tree of the full `EnsembleConfig` |
| **Version comparison** | Select a second version to compare. Shows a delta table (metric, current, compare, delta, % change) and a grouped bar chart side by side. |
| **Trade-cluster map** | Searchable AG Grid table of every trade ID and its assigned cluster |

**Callbacks:** 2 callbacks (populate comparison dropdown, render comparison content).

---

## 10. Reusable Components

All components live in `components/` and are imported by tab layouts
and callbacks.

| Component | Module | Description |
|-----------|--------|-------------|
| `kpi_card(title, value, delta, delta_color, card_id)` | `kpi_card.py` | Compact metric card with optional delta badge. Green for improvements (negative delta), red otherwise. |
| `split_toggle(id_prefix, default)` | `split_toggle.py` | Horizontal radio buttons for Train/Val/Test selection. ID is `"{id_prefix}-split-toggle"`. |
| `cluster_selector(cluster_ids, cluster_attrs, id_prefix, multi, default)` | `cluster_selector.py` | Dropdown showing cluster IDs with attribute labels. ID is `"{id_prefix}-cluster-dropdown"`. |
| `metric_table(column_defs, row_data, table_id, height, sort_model)` | `metric_table.py` | Dark-themed AG Grid with sortable/filterable columns. Supports conditional formatting via `cellStyle.styleConditions`. |
| `filter_bar(catalogue, id_prefix, columns)` | `filter_bar.py` | Multi-select dropdowns for catalogue columns. Each dropdown ID is `"{id_prefix}-filter-{column}"`. |
| `loading_progress(total, loaded, id_prefix)` | `loading_progress.py` | Progress bar with text showing Phase 3 loading status. |

### ID Convention

All component IDs follow the pattern `{tab_prefix}-{component_type}`. This
prevents collisions across tabs and makes callback wiring predictable:

```python
split_toggle(id_prefix="overview")    # → "overview-split-toggle"
split_toggle(id_prefix="eval")        # → "eval-split-toggle"
split_toggle(id_prefix="deep-dive")   # → "deep-dive-split-toggle"
```

---

## 11. Figure Builders

All figure builders are pure functions in `figures/`. They accept data
(usually `np.ndarray`) and return `go.Figure`. They never access the
session or data layer.

| Module | Functions | Used By |
|--------|-----------|---------|
| `scatter.py` | `pred_vs_target_scatter`, `residual_scatter` | Overview, Evaluation, Cluster Deep Dive |
| `timeseries.py` | `pnl_timeseries`, `overlaid_group_timeseries` | Evaluation (Portfolio, By Desk/Product/CCY) |
| `distributions.py` | `residual_histogram`, `violin_overlay`, `qq_plot` | Evaluation, Cluster Deep Dive, Market Data |
| `heatmaps.py` | `cluster_heatmap`, `multi_metric_cluster_heatmap`, `rf_scenario_heatmap`, `adjacency_spy` | Overview, Market Data, Trade Graph |
| `bar_charts.py` | `member_comparison_bar`, `grouped_split_bar` | Overview, Cluster Deep Dive |
| `tables.py` | `percentile_table_data`, `worst_scenarios_data` | Evaluation (Portfolio) |
| `network.py` | `build_cytoscape_elements`, `ego_network` | Trade Graph |

### Performance Notes

- Scatter plots use `go.Scattergl` (WebGL) for datasets > 1,000 points
- `pred_vs_target_scatter` downsamples to 5,000 points via random selection
- `rf_scenario_heatmap` downsamples to 200 scenarios via `np.linspace`
- Per-trade heatmaps cap at 500 scenario rows

---

## 12. Callback Architecture

### Registration

All callbacks are registered through `callbacks/__init__.py`:

```python
def register_all_callbacks(app):
    # Top-level tab router
    @app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
    def render_tab(tab_id): ...

    # Version reload
    @app.callback(Output("tab-content", "children", allow_duplicate=True),
                  Input("ensemble-version-selector", "value"), prevent_initial_call=True)
    def reload_version(version): ...

    # Per-tab callbacks
    reg_overview(app)
    reg_evaluation(app)
    reg_deep_dive(app)
    reg_market_data(app)
    reg_trade_graph(app)
    reg_inference(app)
    reg_governance(app)
```

### Pattern: Guard → Load → Build → Return

Every callback follows the same pattern:

```python
@app.callback(
    Output("some-container", "children"),
    Input("some-trigger", "value"),
)
def update_something(trigger_value):
    # 1. Guard: skip if wrong tab or missing input
    if trigger_value != expected:
        return no_update

    # 2. Load: get data from the data layer
    store = get_prediction_store(split)
    catalogue = get_trade_catalogue()

    # 3. Build: construct figures and components
    fig = pred_vs_target_scatter(preds, targets)
    table = metric_table(col_defs, row_data, "table-id")

    # 4. Return: Dash components wrapped in dcc.Graph / html.Div
    return dcc.Graph(figure=fig), table
```

### Callback Counts Per Tab

| Tab | Callback Module | Callbacks |
|-----|----------------|-----------|
| Global | `__init__.py` | 2 (tab router + version reload) |
| Overview | `overview_cb.py` | 1 |
| Evaluation | `evaluation_cb.py` | 12 |
| Cluster Deep Dive | `cluster_deep_dive_cb.py` | 2 |
| Market Data | `market_data_cb.py` | 6 |
| Trade Graph | `trade_graph_cb.py` | 7 |
| Inference | `inference_cb.py` | 4 |
| Governance | `governance_cb.py` | 2 |
| **Total** | | **36** |

---

## 13. Performance Considerations

| Concern | Mitigation |
|---------|------------|
| **100+ clusters** | Phase 2 loads only small JSON/PNG/NPZ files (~1-5s total). Phase 3 uses `ThreadPoolExecutor` (4 workers default) for parallel model loading. |
| **Large prediction arrays** | `GlobalPredictionStore` uses `np.float32` and column-indexed slicing. All filtering operates on NumPy masks, not DataFrame joins. |
| **Heavy scatter plots** | `go.Scattergl` (WebGL rendering) with automatic downsampling to 5,000 points. |
| **Heatmap scenarios** | Capped at 200-500 scenarios for heatmap rendering. |
| **Phase 3 model loading** | Deferred to user action ("Load Models" button). Only the Inference tab triggers it. |
| **Market/graph data** | Loaded lazily per-cluster on first access, then cached for the session lifetime. |
| **Version switching** | Creates a new session, clears all caches, re-renders from Overview. |

---

## 14. Adding New Tabs or Figures

### Adding a New Tab

1. **Create layout** — add `tabs/my_new_tab.py` with a `layout() -> html.Div` function
2. **Add tab ID** — add `TAB_MY_NEW: str = "tab-my-new"` to `config.py` and append to `TAB_ORDER`
3. **Create callback module** — add `callbacks/my_new_tab_cb.py` with a `register(app)` function
4. **Wire routing** — add `elif tab_id == TAB_MY_NEW:` to the tab router in `callbacks/__init__.py`
5. **Register callbacks** — import and call `reg_my_new(app)` in `register_all_callbacks`

### Adding a New Figure Builder

1. Add the function to the appropriate `figures/*.py` module (or create a new one)
2. Keep the function pure: `(data_args) -> go.Figure`
3. Use `CHART_COLORS` from `theme/colors.py` for colour consistency
4. All figures inherit the `ensemble_dark` template automatically

---

## 15. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Session not initialised` | `create_app()` was not called, or `initialise()` failed | Check `registry_dir` and `artifacts_dir` paths exist |
| Tab shows "No prediction data available" | Missing `.npz` files in `artifacts_dir/ensemble/{version}/evaluation/members/` | Run the `EnsembleEvalPipeline` first |
| Dropdown shows no clusters | `load_metadata()` could not find `ensemble_config.json` | Verify the ensemble version exists in the registry |
| Market Data tab is empty | No `cluster_assets.joblib` in member version directory | Ensure the training pipeline saves asset portfolios |
| Trade Graph tab is empty | No `graph_results.joblib` in member version directory | Ensure the training pipeline saves graph builder output |
| Inference "Models not loaded" | Phase 3 has not been triggered | Click "Load Models" button first |
| Version dropdown has only one entry | Only one ensemble version is registered | Register more versions or check `index.json` |
| Slow startup | Large number of clusters or large JSON files | This is expected for 100+ clusters; Phase 2 typically takes 1-5 seconds |
| Port conflict | Another process on port 8051 | Change `port` in `app.run()` |
| Blank figures | `ensemble_dark` template not registered | Ensure `create_app()` runs the template registration before layout build |

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [`docs/session.md`](../../../docs/session.md) | Complete `EnsembleSession` reference — three-phase lifecycle, dataclasses, all methods, source code |
| [`docs/ensemble_hybrid_gnn_rnn_ui.md`](../../../docs/ensemble_hybrid_gnn_rnn_ui.md) | Full implementation guide with complete source code for all 57 files |
| [`docs/ensemble_dashboard_design.md`](../../../docs/ensemble_dashboard_design.md) | Original design specification — tab layouts, data contracts, performance budget |
| [`docs/ensemble_implementation.md`](../../../docs/ensemble_implementation.md) | Ensemble component and pipeline reference — training, evaluation, inference workflows |
