# Ensemble Hybrid GNN-RNN — Dashboard Implementation Guide

> **Purpose:** Step-by-step implementation guide for the Ensemble Analytics
> Dash UI.  Every file is specified in full — architecture, data contracts,
> and complete source code — so that an LLM assistant or developer can
> implement the dashboard by copy-paste with minimal interpretation.
>
> **Target audience:** FX derivatives traders, structurers, front-office,
> and risk management.
>
> **Model:** Hybrid GNN-RNN ensemble (multiple cluster members).

---

## Table of Contents

*(section numbers are stable — add new sections at the end)*

1. [Prerequisites and Dependencies](#1-prerequisites-and-dependencies)
2. [Architecture Overview](#2-architecture-overview)
3. [Folder Structure](#3-folder-structure)
4. [Data Contracts — Session API Reference](#4-data-contracts--session-api-reference)
5. [Theme Specification](#5-theme-specification)
6. [Data Layer Specification](#6-data-layer-specification)
7. [Shared Components](#7-shared-components)
8. [Figure Builders](#8-figure-builders)
9. [Tab 1 — Overview](#9-tab-1--overview)
10. [Tab 2 — Evaluation (5 sub-tabs)](#10-tab-2--evaluation-5-sub-tabs)
11. [Tab 3 — Cluster Deep Dive](#11-tab-3--cluster-deep-dive)
12. [Tab 4 — Market Data (4 sub-tabs)](#12-tab-4--market-data-4-sub-tabs)
13. [Tab 5 — Trade Graph Explorer (4 sub-tabs)](#13-tab-5--trade-graph-explorer-4-sub-tabs)
14. [Tab 6 — Inference](#14-tab-6--inference)
15. [Tab 7 — Model Governance](#15-tab-7--model-governance)
16. [Complete Source Code](#16-complete-source-code)
17. [Quick Start](#17-quick-start)

---

## 1. Prerequisites and Dependencies

### Python packages

Add to `requirements-ui.txt`:

```
dash>=2.14
dash-bootstrap-components>=1.5
dash-ag-grid>=31.0
dash-cytoscape>=0.3
plotly>=5.18
pandas>=2.0
numpy>=1.24
scipy>=1.11
```

The following are already in the project's main requirements and must be
importable at runtime:

```
src.rade_ml_pt.ensemble.session   — EnsembleSession (core data source)
src.rade_ml_pt.ensemble.config    — EnsembleConfig
src.rade_ml_pt.ensemble.registry  — EnsembleRegistry
```

### Artifact prerequisites

Before launching the dashboard, ensure:

1. **Ensemble training** has been run — producing member versions in the
   model registry.
2. **Ensemble evaluation** has been run — producing the `evaluation/`
   artifact tree under `{artifacts_dir}/ensemble/{version}/evaluation/`.
   This includes `manifest.json`, per-member `.npz` prediction files,
   combined `.npz` files, per-split metric JSONs, and plot PNGs.
3. The `registry_dir` and `artifacts_dir` paths are accessible from the
   machine running the dashboard.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Dash Application                          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Global Header:  Ensemble Selector (version/tag dropdown)  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────┬──────────┬────────────┬──────────┬───────┬──────────┐  │
│  │ Over │ Evalua-  │  Cluster   │  Market  │ Trade │ Infer-   │  │
│  │ view │ tion     │  Deep Dive │  Data    │ Graph │ ence     │  │
│  └──┬───┴────┬─────┴─────┬──────┴────┬─────┴───┬───┴────┬─────┘  │
│     │        │           │           │         │        │        │
│  Callbacks  Callbacks  Callbacks  Callbacks Callbacks Callbacks  │
│     │        │           │           │         │        │        │
│  ┌──┴────────┴───────────┴───────────┴─────────┴────────┴─────┐  │
│  │                     Data Layer                              │  │
│  │  session_manager · trade_catalogue · prediction_store       │  │
│  │  market_data_loader · graph_data_loader                     │  │
│  └─────────────────────────┬──────────────────────────────────┘  │
│                             │                                    │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │     EnsembleSession        │
                │                            │
                │  Phase 1: load_metadata()  │
                │  Phase 2: load_display()   │
                │  Phase 3: load_inference() │
                └──────┬─────────────┬───────┘
                       │             │
              ┌────────┴──┐   ┌──────┴──────┐
              │  Registry │   │  Artifacts  │
              │  (models, │   │  (metrics,  │
              │  configs)  │   │  plots,npz) │
              └───────────┘   └─────────────┘
```

### Lifecycle

1. **App start** — `session_manager` calls `load_metadata(version)` then
   `load_display_artifacts()`.  Phase 1+2 complete in < 3 s for 100
   clusters.  Enough data for Overview, Evaluation, Cluster Deep Dive,
   Market Data, Trade Graph, and Governance tabs.

2. **Tab navigation** — switching tabs is instant; all Phase 2 data is
   already in server memory.  `GlobalPredictionStore` and
   `GlobalTradeCatalogue` are built lazily on first access and cached.

3. **Inference tab** — first visit triggers Phase 3 (`load_inference_state`)
   with a progress bar.  After loading, `run_inference()` calls are fast
   (forward pass only).

### Key design rules

- **No model loading** outside the Inference tab.
- **No disk writes** — the session is read-only.
- **Server-side arrays** — numpy arrays stay in Python; only Plotly JSON
  and small metadata flow to the browser via `dcc.Store`.
- **Pure figure builders** — every plot function in `figures/` is a pure
  function `(data, params) → go.Figure`.  No side effects.

---

## 3. Folder Structure

All UI code lives under `src/ui/apps/ensemble_analytics/`.

```
src/ui/apps/ensemble_analytics/
├── __init__.py                     # create_app() export
├── app.py                          # Dash app factory, top-level layout, tab routing
├── config.py                       # App constants (port, title, tab IDs, split names)
│
├── theme/
│   ├── __init__.py
│   ├── colors.py                   # Color palette: BG, CARD, TEXT, ACCENT, STATUS
│   ├── styles.py                   # Style dicts: container, navbar, card, table, kpi
│   └── plotly_template.py          # Custom plotly template (plotly_dark + overrides)
│
├── data/
│   ├── __init__.py
│   ├── session_manager.py          # Singleton EnsembleSession wrapper
│   ├── trade_catalogue.py          # Cached global trade catalogue DataFrame
│   ├── prediction_store.py         # Cached GlobalPredictionStore per split
│   ├── market_data_loader.py       # Cached cluster_assets.joblib loader
│   └── graph_data_loader.py        # Cached graph_results.joblib loader
│
├── components/
│   ├── __init__.py
│   ├── kpi_card.py                 # KPI card with value + delta badge
│   ├── split_toggle.py             # Train / Val / Test radio buttons
│   ├── cluster_selector.py         # Cluster dropdown with attribute labels
│   ├── metric_table.py             # AG Grid table with conditional formatting
│   ├── filter_bar.py               # Multi-select filter (desk / product / ccy)
│   └── loading_progress.py         # Phase 3 loading progress bar
│
├── tabs/
│   ├── __init__.py
│   ├── overview.py                 # Tab 1
│   ├── evaluation/                 # Tab 2 (5 sub-tabs)
│   │   ├── __init__.py
│   │   ├── portfolio.py
│   │   ├── by_desk.py
│   │   ├── by_product.py
│   │   ├── by_ccy.py
│   │   └── by_cluster.py
│   ├── cluster_deep_dive.py        # Tab 3
│   ├── market_data/                # Tab 4 (4 sub-tabs)
│   │   ├── __init__.py
│   │   ├── rf_summary.py
│   │   ├── shock_explorer.py
│   │   ├── scenario_heatmap.py
│   │   └── distribution.py
│   ├── trade_graph/                # Tab 5 (4 sub-tabs)
│   │   ├── __init__.py
│   │   ├── graph_view.py
│   │   ├── adjacency_analysis.py
│   │   ├── node_analytics.py
│   │   └── cross_cluster.py
│   ├── inference.py                # Tab 6
│   └── governance.py               # Tab 7
│
├── callbacks/
│   ├── __init__.py                 # register_all_callbacks(app)
│   ├── overview_cb.py
│   ├── evaluation_cb.py
│   ├── cluster_deep_dive_cb.py
│   ├── market_data_cb.py
│   ├── trade_graph_cb.py
│   ├── inference_cb.py
│   └── governance_cb.py
│
└── figures/
    ├── __init__.py
    ├── scatter.py                  # pred_vs_target_scatter, residual_scatter
    ├── timeseries.py               # pnl_timeseries, overlaid_group_timeseries
    ├── distributions.py            # residual_histogram, violin_overlay, qq_plot
    ├── heatmaps.py                 # cluster_heatmap, rf_scenario_heatmap, adjacency_spy
    ├── bar_charts.py               # member_comparison_bar, grouped_split_bar
    ├── network.py                  # build_cytoscape_elements, ego_network
    └── tables.py                   # percentile_table_data, worst_scenarios_data
```

---

## 4. Data Contracts — Session API Reference

Every piece of data the dashboard displays comes from `EnsembleSession`.
This table maps each session method/property to the tabs that consume it.

### 4.1 Properties (read-only, available after the relevant phase)

| Property | Type | Phase | Consuming tabs |
|----------|------|-------|----------------|
| `config` | `EnsembleConfig` | 1 | All (cluster IDs, trade mapping) |
| `ensemble_version` | `str` | 1 | Header, Governance |
| `member_versions` | `Dict[str, str]` | 1 | Governance, Cluster Deep Dive |
| `member_summary` | `Dict[str, Dict]` | 1 | Overview (fallback metrics) |
| `trade_cluster_map` | `Dict[str, str]` | 1 | Governance |
| `cluster_attributes` | `Dict[str, Dict]` | 1 | Overview, Evaluation filters, Cluster selector |
| `ensemble_display` | `EnsembleDisplayState` | 2 | Overview, Evaluation |
| `display` | `Dict[str, ClusterDisplayState]` | 2 | Cluster Deep Dive, Evaluation |
| `is_metadata_loaded` | `bool` | — | App guards |
| `is_display_loaded` | `bool` | — | App guards |
| `all_inference_ready` | `bool` | — | Inference tab guard |
| `inference_ready_clusters` | `List[str]` | — | Inference progress |
| `ensemble_model` | `EnsembleModel` | 3 | Inference (internal) |

### 4.2 Methods

| Method | Signature | Returns | Phase | Consuming tabs |
|--------|-----------|---------|-------|----------------|
| `load_metadata` | `(version_or_tag: str)` | `None` | 1 | App start |
| `load_display_artifacts` | `()` | `None` | 2 | App start |
| `load_cluster_display` | `(cluster_id: str)` | `ClusterDisplayState` | 2 | Cluster Deep Dive |
| `load_cluster_predictions` | `(cluster_id: str, split: str)` | `Optional[np.ndarray]` | 2 | Cluster Deep Dive |
| `build_global_trade_catalogue` | `()` | `pd.DataFrame` | 2 | Evaluation (all sub-tabs) |
| `get_prediction_store` | `(split: str)` | `Optional[GlobalPredictionStore]` | 2 | Evaluation (all sub-tabs) |
| `load_cluster_market_data` | `(cluster_id: str)` | `Dict[str, Any]` | 2 | Market Data |
| `load_cluster_graph_data` | `(cluster_id: str)` | `Dict[str, Any]` | 2 | Trade Graph |
| `load_inference_state` | `(cluster_ids, parallel)` | `None` | 3 | Inference |
| `run_inference` | `(mode, cluster_pnl_histories, ...)` | `Dict[str, Any]` | 3 | Inference |

### 4.3 Dataclass shapes

**`EnsembleDisplayState`** (portfolio-level, loaded in Phase 2):

```
ensemble_metrics:    {split: {mae: float, rmse: float, ...}}
member_rollup:       {split: {mean_mae: float, ...}}
per_member_metrics:  {split: {cluster_id: {mae: float, ...}}}
manifest:            {trade_ids: [...], cluster_ids: [...],
                      cluster_trade_indices: {cid: [int, ...]},
                      splits_available: [...]}
```

**`ClusterDisplayState`** (per-cluster, loaded in Phase 2):

```
cluster_id:          str
version:             str
version_dir:         str (absolute path)
eval_metrics:        {split: {mae: float, ...}, "summary": {...},
                      "data_config": {...}}
plot_paths:          {"split/plot_name": "/abs/path.png"}
trade_universe:      {elementary_idx: [...], target_idx: [...],
                      elementary_ids: [...], target_ids: [...]}
target_attributes:   {trade_id: [...], product_type: [...], ccy: [...], ...}
predictions:         {cache_key: np.ndarray or None}
```

**`GlobalPredictionStore`** (built lazily, cached per split):

```
predictions:  np.ndarray  [n_scenarios, n_total_targets]
targets:      np.ndarray  [n_scenarios, n_total_targets]
trade_ids:    List[str]   length = n_total_targets
cluster_ids:  List[str]   per-column cluster membership
split:        str         "train", "val", or "test"
```

**Global Trade Catalogue** (`pd.DataFrame`, built lazily, cached):

```
Columns: trade_id, cluster_id, product_type, product_subtype, ccy,
         desk, ... (all target_attributes keys + cluster_attributes keys)
```

### 4.4 Slicing pattern (Evaluation tabs)

The core pattern for any filtered view:

```python
catalogue = data.get_trade_catalogue()                # pd.DataFrame
store     = data.get_prediction_store(split)           # GlobalPredictionStore

# Filter by desk
mask = catalogue["desk"] == selected_desk
col_indices = np.where(mask.values)[0]

# Slice predictions / targets
preds_subset  = store.predictions[:, col_indices]      # [n_scenarios, n_filtered]
targets_subset = store.targets[:, col_indices]

# Aggregate: sum across trades for portfolio PnL per scenario
portfolio_pred   = preds_subset.sum(axis=1)            # [n_scenarios]
portfolio_target = targets_subset.sum(axis=1)
```

This pattern is identical for By Product, By CCY, and By Cluster — only
the filter column changes.

---

## 5. Theme Specification

### Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `BG_PRIMARY` | `#0d1117` | Page background |
| `BG_CARD` | `#161b22` | Card / panel background |
| `BG_HOVER` | `#1c2333` | Hover states, active rows |
| `BORDER` | `#30363d` | Card borders, dividers |
| `TEXT_PRIMARY` | `#e6edf3` | Body text |
| `TEXT_SECONDARY` | `#8b949e` | Labels, axis text |
| `TEXT_MUTED` | `#484f58` | Disabled text |
| `ACCENT_BLUE` | `#58a6ff` | Primary actions, links |
| `ACCENT_GREEN` | `#3fb950` | Positive deltas, success |
| `ACCENT_RED` | `#f85149` | Negative deltas, errors |
| `ACCENT_AMBER` | `#d29922` | Warnings |
| `ACCENT_PURPLE` | `#bc8cff` | Secondary highlights |

### Typography

- **Font stack:** `"Inter", -apple-system, BlinkMacSystemFont, sans-serif`
- **KPI values:** 28 px bold
- **Card titles:** 14 px semi-bold, uppercase, `TEXT_SECONDARY`
- **Body:** 13 px regular

### Plotly template

All figure builders use a shared template `PLOTLY_TEMPLATE` that extends
`plotly_dark` with the palette above: transparent plot background,
`BG_CARD` paper background, grid lines using `BORDER`, font using
`TEXT_PRIMARY`.

---

## 6. Data Layer Specification

### `session_manager.py` — Singleton pattern

The session manager holds a single `EnsembleSession` instance that
persists across all Dash callbacks.  On app start, `initialise()` runs
Phase 1 + Phase 2.  All other data modules read from this singleton.

**Key functions:**

| Function | Description |
|----------|-------------|
| `initialise(registry_dir, artifacts_dir, version)` | Create session, run Phase 1+2 |
| `get_session()` | Return the singleton `EnsembleSession` |
| `reload(version)` | Re-run Phase 1+2 for a different version |

### `trade_catalogue.py` — Cached DataFrame

Wraps `session.build_global_trade_catalogue()` with a module-level cache.
Called once on first Evaluation tab access.

### `prediction_store.py` — Cached per split

Wraps `session.get_prediction_store(split)`.  Returns `None` if artifacts
are unavailable.

### `market_data_loader.py` / `graph_data_loader.py`

Thin wrappers around `session.load_cluster_market_data(cid)` and
`session.load_cluster_graph_data(cid)`.  Caching is handled by the
session itself.

---

## 16. Complete Source Code

> **Instructions:** Create each file at the path shown in the header.
> Files are listed in dependency order — foundational modules first,
> then components, then tabs, then callbacks.

---

### 16.1 `src/ui/apps/ensemble_analytics/__init__.py`

```python
"""Ensemble Analytics dashboard — app export."""
from src.ui.apps.ensemble_analytics.app import create_app

__all__ = ["create_app"]
```

---

### 16.2 `src/ui/apps/ensemble_analytics/config.py`

```python
"""
Application constants for the Ensemble Analytics dashboard.

Centralises tab IDs, split names, default ports, and display strings
so that layout modules and callbacks reference a single source of truth.
"""
from typing import Dict, List, Tuple

# ── App metadata ──────────────────────────────────────────────────
APP_TITLE: str = "Ensemble Analytics — Hybrid GNN-RNN"
DEFAULT_PORT: int = 8051

# ── Tab identifiers (must match dcc.Tab value attributes) ─────────
TAB_OVERVIEW: str = "tab-overview"
TAB_EVALUATION: str = "tab-evaluation"
TAB_CLUSTER_DEEP_DIVE: str = "tab-cluster-deep-dive"
TAB_MARKET_DATA: str = "tab-market-data"
TAB_TRADE_GRAPH: str = "tab-trade-graph"
TAB_INFERENCE: str = "tab-inference"
TAB_GOVERNANCE: str = "tab-governance"

TAB_ORDER: List[Tuple[str, str]] = [
    (TAB_OVERVIEW, "Overview"),
    (TAB_EVALUATION, "Evaluation"),
    (TAB_CLUSTER_DEEP_DIVE, "Cluster Deep Dive"),
    (TAB_MARKET_DATA, "Market Data"),
    (TAB_TRADE_GRAPH, "Trade Graph"),
    (TAB_INFERENCE, "Inference"),
    (TAB_GOVERNANCE, "Model Governance"),
]

# ── Evaluation sub-tab identifiers ────────────────────────────────
EVAL_SUB_PORTFOLIO: str = "eval-sub-portfolio"
EVAL_SUB_DESK: str = "eval-sub-desk"
EVAL_SUB_PRODUCT: str = "eval-sub-product"
EVAL_SUB_CCY: str = "eval-sub-ccy"
EVAL_SUB_CLUSTER: str = "eval-sub-cluster"

EVAL_SUB_ORDER: List[Tuple[str, str]] = [
    (EVAL_SUB_PORTFOLIO, "Portfolio"),
    (EVAL_SUB_DESK, "By Desk"),
    (EVAL_SUB_PRODUCT, "By Product"),
    (EVAL_SUB_CCY, "By CCY"),
    (EVAL_SUB_CLUSTER, "By Cluster"),
]

# ── Market Data sub-tab identifiers ──────────────────────────────
MD_SUB_RF_SUMMARY: str = "md-sub-rf-summary"
MD_SUB_SHOCK_EXPLORER: str = "md-sub-shock-explorer"
MD_SUB_SCENARIO_HEATMAP: str = "md-sub-scenario-heatmap"
MD_SUB_DISTRIBUTION: str = "md-sub-distribution"

MD_SUB_ORDER: List[Tuple[str, str]] = [
    (MD_SUB_RF_SUMMARY, "RF Summary"),
    (MD_SUB_SHOCK_EXPLORER, "Shock Explorer"),
    (MD_SUB_SCENARIO_HEATMAP, "Scenario Heatmap"),
    (MD_SUB_DISTRIBUTION, "Distribution"),
]

# ── Trade Graph sub-tab identifiers ──────────────────────────────
TG_SUB_GRAPH_VIEW: str = "tg-sub-graph-view"
TG_SUB_ADJACENCY: str = "tg-sub-adjacency"
TG_SUB_NODE_ANALYTICS: str = "tg-sub-node-analytics"
TG_SUB_CROSS_CLUSTER: str = "tg-sub-cross-cluster"

TG_SUB_ORDER: List[Tuple[str, str]] = [
    (TG_SUB_GRAPH_VIEW, "Graph View"),
    (TG_SUB_ADJACENCY, "Adjacency Analysis"),
    (TG_SUB_NODE_ANALYTICS, "Node Analytics"),
    (TG_SUB_CROSS_CLUSTER, "Cross-Cluster"),
]

# ── Split names ───────────────────────────────────────────────────
SPLITS: List[str] = ["test", "val", "train"]
DEFAULT_SPLIT: str = "test"

# ── Display format helpers ────────────────────────────────────────
METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "mae": "MAE",
    "rmse": "RMSE",
    "max_ae": "Max AE",
    "mape": "MAPE",
    "r2": "R²",
    "p95_ae": "P95 AE",
    "p99_ae": "P99 AE",
}
```

---

### 16.3 `src/ui/apps/ensemble_analytics/app.py`

```python
"""
Dash application factory for the Ensemble Analytics dashboard.

Creates the Dash app instance, builds the top-level layout (header +
tab container), registers all callbacks, and initialises the data layer.
"""
from __future__ import annotations

import logging
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import (
    APP_TITLE,
    TAB_ORDER,
    TAB_OVERVIEW,
)
from src.ui.apps.ensemble_analytics.theme.colors import BG_PRIMARY, TEXT_PRIMARY, ACCENT_BLUE
from src.ui.apps.ensemble_analytics.theme.styles import NAVBAR_STYLE, CONTAINER_STYLE
from src.ui.apps.ensemble_analytics.theme.plotly_template import PLOTLY_TEMPLATE

logger = logging.getLogger(__name__)


def create_app(
    registry_dir: str,
    artifacts_dir: str,
    version: str = "latest",
    debug: bool = False,
) -> dash.Dash:
    """
    Build and return the configured Dash application.

    Parameters
    ----------
    registry_dir : str
        Root directory for model and ensemble registries.
    artifacts_dir : str
        Root directory for evaluation artifacts.
    version : str
        Ensemble version or tag to load on startup.
    debug : bool
        Enable Dash debug mode (hot-reload, verbose errors).

    Returns
    -------
    dash.Dash
        Fully configured application ready for ``app.run()``.
    """
    import plotly.io as pio
    pio.templates["ensemble_dark"] = PLOTLY_TEMPLATE
    pio.templates.default = "ensemble_dark"

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title=APP_TITLE,
    )

    # ── Initialise data layer ─────────────────────────────────────
    from src.ui.apps.ensemble_analytics.data.session_manager import initialise
    initialise(registry_dir, artifacts_dir, version)

    # ── Build layout ──────────────────────────────────────────────
    app.layout = _build_layout(version)

    # ── Register callbacks ────────────────────────────────────────
    from src.ui.apps.ensemble_analytics.callbacks import register_all_callbacks
    register_all_callbacks(app)

    return app


def _build_layout(version: str) -> dbc.Container:
    """
    Assemble the top-level page layout.

    Structure: navbar → version store → tab bar → tab content container.
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session

    session = get_session()
    config = session.config

    # Build ensemble version options for the dropdown
    ens_registry = session._ens_registry
    available_versions = []
    try:
        available_versions = [
            {"label": v, "value": v}
            for v in ens_registry.list_versions()
        ]
    except Exception:
        available_versions = [{"label": version, "value": version}]

    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H4(
                                APP_TITLE,
                                className="mb-0",
                                style={"color": TEXT_PRIMARY, "fontWeight": "600"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="ensemble-version-selector",
                                options=available_versions,
                                value=session.ensemble_version,
                                clearable=False,
                                style={
                                    "width": "340px",
                                    "backgroundColor": BG_PRIMARY,
                                    "color": TEXT_PRIMARY,
                                },
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Span(
                                f"{config.n_members} clusters · "
                                f"{len(config.all_trade_ids)} trades",
                                style={"color": "#8b949e", "fontSize": "13px"},
                            ),
                            width="auto",
                            className="ms-3 d-flex align-items-center",
                        ),
                    ],
                    align="center",
                    className="g-3",
                ),
            ],
            fluid=True,
        ),
        style=NAVBAR_STYLE,
        dark=True,
    )

    tab_bar = dcc.Tabs(
        id="main-tabs",
        value=TAB_OVERVIEW,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in TAB_ORDER
        ],
        style={"borderBottom": "1px solid #30363d"},
    )

    return dbc.Container(
        [
            navbar,
            dcc.Store(id="active-split", data="test"),
            dcc.Store(id="active-cluster", data=None),
            html.Div(style={"height": "8px"}),
            tab_bar,
            html.Div(id="tab-content", style={"marginTop": "16px"}),
        ],
        fluid=True,
        style=CONTAINER_STYLE,
    )
```

---

### 16.4 `src/ui/apps/ensemble_analytics/theme/__init__.py`

```python
"""Theme package — dark palette, styles, and plotly template."""
```

---

### 16.5 `src/ui/apps/ensemble_analytics/theme/colors.py`

```python
"""
Color palette for the Ensemble Analytics dark theme.

All UI modules import color tokens from here.  Never hard-code hex
values in layout or callback code.
"""

# ── Backgrounds ───────────────────────────────────────────────────
BG_PRIMARY: str = "#0d1117"
BG_CARD: str = "#161b22"
BG_HOVER: str = "#1c2333"

# ── Borders ───────────────────────────────────────────────────────
BORDER: str = "#30363d"

# ── Text ──────────────────────────────────────────────────────────
TEXT_PRIMARY: str = "#e6edf3"
TEXT_SECONDARY: str = "#8b949e"
TEXT_MUTED: str = "#484f58"

# ── Accents ───────────────────────────────────────────────────────
ACCENT_BLUE: str = "#58a6ff"
ACCENT_GREEN: str = "#3fb950"
ACCENT_RED: str = "#f85149"
ACCENT_AMBER: str = "#d29922"
ACCENT_PURPLE: str = "#bc8cff"

# ── Chart color cycle (for multi-series plots) ───────────────────
CHART_COLORS: list = [
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_AMBER,
    ACCENT_RED,
    "#79c0ff",
    "#7ee787",
    "#d2a8ff",
    "#e3b341",
    "#ffa198",
]
```

---

### 16.6 `src/ui/apps/ensemble_analytics/theme/styles.py`

```python
"""
Reusable CSS-in-Python style dictionaries for Dash components.

Import individual style dicts in layout modules.  All values reference
color tokens from ``colors.py`` — never hard-code hex here.
"""
from src.ui.apps.ensemble_analytics.theme.colors import (
    BG_PRIMARY,
    BG_CARD,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

CONTAINER_STYLE: dict = {
    "backgroundColor": BG_PRIMARY,
    "minHeight": "100vh",
    "padding": "0",
    "fontFamily": '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
    "color": TEXT_PRIMARY,
}

NAVBAR_STYLE: dict = {
    "backgroundColor": BG_CARD,
    "borderBottom": f"1px solid {BORDER}",
    "padding": "10px 20px",
}

CARD_STYLE: dict = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "16px",
}

CARD_HEADER_STYLE: dict = {
    "fontSize": "12px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "color": TEXT_SECONDARY,
    "marginBottom": "8px",
}

KPI_VALUE_STYLE: dict = {
    "fontSize": "28px",
    "fontWeight": "700",
    "color": TEXT_PRIMARY,
    "lineHeight": "1.2",
}

TABLE_STYLE: dict = {
    "backgroundColor": BG_CARD,
    "color": TEXT_PRIMARY,
    "fontSize": "13px",
}

SECTION_TITLE_STYLE: dict = {
    "fontSize": "16px",
    "fontWeight": "600",
    "color": TEXT_PRIMARY,
    "marginBottom": "12px",
    "marginTop": "24px",
}
```

---

### 16.7 `src/ui/apps/ensemble_analytics/theme/plotly_template.py`

```python
"""
Custom Plotly template extending ``plotly_dark``.

Ensures all figures match the dashboard's dark palette without
per-figure styling.  Registered as ``ensemble_dark`` in ``app.py``.
"""
import plotly.graph_objects as go
import plotly.io as pio

from src.ui.apps.ensemble_analytics.theme.colors import (
    BG_CARD,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    CHART_COLORS,
)

_base = pio.templates["plotly_dark"]

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
            size=12,
            color=TEXT_PRIMARY,
        ),
        title=dict(font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis=dict(
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            titlefont=dict(color=TEXT_SECONDARY, size=12),
        ),
        yaxis=dict(
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            tickfont=dict(color=TEXT_SECONDARY, size=11),
            titlefont=dict(color=TEXT_SECONDARY, size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY, size=11),
        ),
        colorway=CHART_COLORS,
        margin=dict(l=50, r=20, t=40, b=40),
    ),
    data=_base.data,
)
```

---

### 16.8 `src/ui/apps/ensemble_analytics/data/__init__.py`

```python
"""Data layer — bridges EnsembleSession to Dash callbacks."""
```

---

### 16.9 `src/ui/apps/ensemble_analytics/data/session_manager.py`

```python
"""
Singleton wrapper around ``EnsembleSession``.

All dashboard modules call ``get_session()`` to access the shared
session.  The session is initialised once at app startup with
``initialise()`` and can be reloaded for a different version with
``reload()``.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.rade_ml_pt.ensemble.session import EnsembleSession

logger = logging.getLogger(__name__)

_session: Optional[EnsembleSession] = None
_registry_dir: Optional[str] = None
_artifacts_dir: Optional[str] = None


def initialise(
    registry_dir: str,
    artifacts_dir: str,
    version: str = "latest",
) -> None:
    """
    Create the singleton ``EnsembleSession`` and run Phase 1 + 2.

    Parameters
    ----------
    registry_dir : str
        Root directory for model and ensemble registries.
    artifacts_dir : str
        Root directory for evaluation artifacts.
    version : str
        Ensemble version or tag to load.
    """
    global _session, _registry_dir, _artifacts_dir

    _registry_dir = registry_dir
    _artifacts_dir = artifacts_dir

    _session = EnsembleSession(
        registry_dir=registry_dir,
        artifacts_dir=artifacts_dir,
    )
    _session.load_metadata(version)
    _session.load_display_artifacts()

    logger.info(
        "Session initialised: version=%s, clusters=%d",
        _session.ensemble_version,
        _session.config.n_members,
    )


def get_session() -> EnsembleSession:
    """
    Return the singleton ``EnsembleSession``.

    Raises
    ------
    RuntimeError
        If ``initialise()`` has not been called.
    """
    if _session is None:
        raise RuntimeError(
            "Session not initialised. Call initialise() at app startup."
        )
    return _session


def reload(version: str = "latest") -> None:
    """
    Reload the session for a different ensemble version.

    Tears down the current session and re-runs Phase 1 + 2.
    Inference state (Phase 3) is not carried over.
    """
    if _registry_dir is None or _artifacts_dir is None:
        raise RuntimeError("Cannot reload — session was never initialised.")
    initialise(_registry_dir, _artifacts_dir, version)
```

---

### 16.10 `src/ui/apps/ensemble_analytics/data/trade_catalogue.py`

```python
"""
Cached global trade catalogue.

Builds the catalogue DataFrame once on first access and caches it
at module level.  The catalogue enables cross-cluster filtering by
desk, product, ccy, or any saved trade attribute.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

_cache: Optional[pd.DataFrame] = None


def get_trade_catalogue() -> pd.DataFrame:
    """
    Return the global trade catalogue DataFrame, building on first call.

    Returns
    -------
    pd.DataFrame
        Columns include ``trade_id``, ``cluster_id``, plus any
        per-trade attributes (``product_type``, ``ccy``, ``desk``, etc.)
        and cluster-level attributes.
    """
    global _cache
    if _cache is not None:
        return _cache

    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    _cache = get_session().build_global_trade_catalogue()
    return _cache


def invalidate() -> None:
    """Clear the cache (called on session reload)."""
    global _cache
    _cache = None
```

---

### 16.11 `src/ui/apps/ensemble_analytics/data/prediction_store.py`

```python
"""
Cached ``GlobalPredictionStore`` access, one per split.

The store is built lazily by the session from per-member ``.npz`` files
and cached for the session lifetime.  This module provides a clean
import path for callbacks.
"""
from __future__ import annotations

from typing import Optional

from src.rade_ml_pt.ensemble.session import GlobalPredictionStore


def get_prediction_store(split: str = "test") -> Optional[GlobalPredictionStore]:
    """
    Return the prediction store for *split*, or ``None`` if unavailable.

    Parameters
    ----------
    split : str
        One of ``"test"``, ``"val"``, ``"train"``.

    Returns
    -------
    GlobalPredictionStore or None
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    return get_session().get_prediction_store(split)
```

---

### 16.12 `src/ui/apps/ensemble_analytics/data/market_data_loader.py`

```python
"""
Cluster market-data loader.

Delegates to ``EnsembleSession.load_cluster_market_data`` which
handles caching internally.  This module provides a clean import
path for Market Data tab callbacks.
"""
from __future__ import annotations

from typing import Any, Dict


def get_market_data(cluster_id: str) -> Dict[str, Any]:
    """
    Return market / risk-factor shock data for one cluster.

    Returns
    -------
    dict
        ``{asset_name: {rf_name: np.ndarray}}``.
        Empty dict if ``cluster_assets.joblib`` is not available.
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    return get_session().load_cluster_market_data(cluster_id)
```

---

### 16.13 `src/ui/apps/ensemble_analytics/data/graph_data_loader.py`

```python
"""
Cluster graph-data loader.

Delegates to ``EnsembleSession.load_cluster_graph_data`` which
handles caching internally.  This module provides a clean import
path for Trade Graph tab callbacks.
"""
from __future__ import annotations

from typing import Any, Dict


def get_graph_data(cluster_id: str) -> Dict[str, Any]:
    """
    Return graph adjacency and encoder feature data for one cluster.

    Returns
    -------
    dict
        Keys: ``graph_results``, ``encoder_results``, ``trade_universe``.
        Empty dict if joblib files are not available.
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    return get_session().load_cluster_graph_data(cluster_id)
```

---

## 7. Shared Components

Reusable Dash components used across multiple tabs.  Each component is a
function that returns a Dash component tree.  Components never access the
data layer directly — they receive data as arguments.

| Component | File | Props | Used by |
|-----------|------|-------|---------|
| KPI card | `kpi_card.py` | `title`, `value`, `fmt`, `delta`, `delta_color` | Overview, Evaluation, Cluster Deep Dive |
| Split toggle | `split_toggle.py` | `id_prefix`, `default` | Overview, Evaluation, Cluster Deep Dive |
| Cluster selector | `cluster_selector.py` | `cluster_ids`, `cluster_attrs`, `id_prefix` | Cluster Deep Dive, Market Data, Trade Graph, Inference |
| Metric table | `metric_table.py` | `column_defs`, `row_data`, `id` | Overview, Evaluation, Cluster Deep Dive, Governance |
| Filter bar | `filter_bar.py` | `catalogue`, `id_prefix` | Evaluation (By Desk/Product/CCY) |
| Loading progress | `loading_progress.py` | `total`, `loaded`, `id_prefix` | Inference |

---

## 8. Figure Builders

Pure functions: `(data, params) → go.Figure`.  No Dash imports, no side
effects.  The plotly template is applied automatically (registered globally).

| Module | Functions | Used by |
|--------|-----------|---------|
| `scatter.py` | `pred_vs_target_scatter`, `residual_scatter` | Overview, Evaluation, Cluster Deep Dive |
| `timeseries.py` | `pnl_timeseries`, `overlaid_group_timeseries` | Evaluation (Portfolio, By Desk/Product/CCY) |
| `distributions.py` | `residual_histogram`, `violin_overlay`, `qq_plot` | Evaluation, Cluster Deep Dive, Market Data |
| `heatmaps.py` | `cluster_heatmap`, `rf_scenario_heatmap`, `adjacency_spy` | Overview, Market Data, Trade Graph |
| `bar_charts.py` | `member_comparison_bar`, `grouped_split_bar` | Overview, Evaluation |
| `network.py` | `build_cytoscape_elements`, `ego_network` | Trade Graph |
| `tables.py` | `percentile_table_data`, `worst_scenarios_data` | Evaluation (Portfolio) |

---

### 16.14 `src/ui/apps/ensemble_analytics/components/__init__.py`

```python
"""Reusable Dash components for the Ensemble Analytics dashboard."""
```

---

### 16.15 `src/ui/apps/ensemble_analytics/components/kpi_card.py`

```python
"""
KPI card component.

Renders a compact metric card with a title, formatted value, and an
optional delta badge showing change direction with colour coding.
"""
from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import html

from src.ui.apps.ensemble_analytics.theme.colors import (
    ACCENT_GREEN,
    ACCENT_RED,
    TEXT_SECONDARY,
)
from src.ui.apps.ensemble_analytics.theme.styles import (
    CARD_STYLE,
    CARD_HEADER_STYLE,
    KPI_VALUE_STYLE,
)


def kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: Optional[str] = None,
    card_id: Optional[str] = None,
) -> dbc.Card:
    """
    Build a KPI display card.

    Parameters
    ----------
    title : str
        Metric label (e.g. ``"MAE"``).
    value : str
        Pre-formatted metric value (e.g. ``"0.0342"``).
    delta : str, optional
        Delta string (e.g. ``"+2.1%"``).
    delta_color : str, optional
        Override colour for the delta badge.  Defaults to green for
        values starting with ``"-"`` (improvement) and red otherwise.
    card_id : str, optional
        HTML id for callback targeting.

    Returns
    -------
    dbc.Card
    """
    children = [
        html.Div(title, style=CARD_HEADER_STYLE),
        html.Div(value, style=KPI_VALUE_STYLE),
    ]

    if delta is not None:
        if delta_color is None:
            delta_color = ACCENT_GREEN if delta.startswith("-") else ACCENT_RED
        children.append(
            html.Span(
                delta,
                style={
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "color": delta_color,
                    "marginTop": "4px",
                    "display": "inline-block",
                },
            )
        )

    props = {"style": CARD_STYLE, "children": children}
    if card_id:
        props["id"] = card_id
    return dbc.Card(**props)
```

---

### 16.16 `src/ui/apps/ensemble_analytics/components/split_toggle.py`

```python
"""
Split toggle component (Train / Val / Test radio buttons).

Local to each tab that needs it — not a global control.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import SPLITS, DEFAULT_SPLIT
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def split_toggle(id_prefix: str, default: str = DEFAULT_SPLIT) -> html.Div:
    """
    Build a compact horizontal radio-button group for split selection.

    Parameters
    ----------
    id_prefix : str
        Prefix for the component ID (e.g. ``"overview"`` produces
        ``"overview-split-toggle"``).
    default : str
        Initially selected split.

    Returns
    -------
    html.Div
    """
    return html.Div(
        [
            html.Label(
                "Split:",
                style={
                    "color": TEXT_SECONDARY,
                    "fontSize": "12px",
                    "marginRight": "8px",
                    "fontWeight": "600",
                },
            ),
            dcc.RadioItems(
                id=f"{id_prefix}-split-toggle",
                options=[{"label": s.capitalize(), "value": s} for s in SPLITS],
                value=default,
                inline=True,
                labelStyle={
                    "marginRight": "16px",
                    "fontSize": "13px",
                    "cursor": "pointer",
                },
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "16px"},
    )
```

---

### 16.17 `src/ui/apps/ensemble_analytics/components/cluster_selector.py`

```python
"""
Cluster selector dropdown.

Shows cluster IDs with optional attribute labels (ccy, desk, product)
for quick identification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def cluster_selector(
    cluster_ids: List[str],
    cluster_attrs: Optional[Dict[str, Dict[str, Any]]] = None,
    id_prefix: str = "cluster",
    multi: bool = False,
    default: Optional[str] = None,
) -> html.Div:
    """
    Build a cluster selection dropdown.

    Parameters
    ----------
    cluster_ids : list of str
        Available cluster IDs.
    cluster_attrs : dict, optional
        ``{cluster_id: {ccy: ..., desk: ..., product: ...}}``.
        When provided, labels show attributes alongside the ID.
    id_prefix : str
        Component ID prefix.
    multi : bool
        Allow multi-selection.
    default : str, optional
        Default selected cluster.  Falls back to the first ID.

    Returns
    -------
    html.Div
    """
    options = []
    for cid in cluster_ids:
        if cluster_attrs and cid in cluster_attrs:
            attrs = cluster_attrs[cid]
            parts = [f"{k}={v}" for k, v in attrs.items() if v is not None]
            label = f"{cid}  ({', '.join(parts)})" if parts else cid
        else:
            label = cid
        options.append({"label": label, "value": cid})

    return html.Div(
        [
            html.Label(
                "Cluster:",
                style={
                    "color": TEXT_SECONDARY,
                    "fontSize": "12px",
                    "marginRight": "8px",
                    "fontWeight": "600",
                },
            ),
            dcc.Dropdown(
                id=f"{id_prefix}-cluster-dropdown",
                options=options,
                value=default or (cluster_ids[0] if cluster_ids else None),
                multi=multi,
                clearable=False,
                style={"width": "400px", "fontSize": "13px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "16px"},
    )
```

---

### 16.18 `src/ui/apps/ensemble_analytics/components/metric_table.py`

```python
"""
AG Grid metric table component.

Wraps ``dash_ag_grid.AgGrid`` with dark-theme defaults and optional
conditional formatting (colour-code cells by metric value).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash_ag_grid as dag

from src.ui.apps.ensemble_analytics.theme.colors import BG_CARD, TEXT_PRIMARY, BORDER


def metric_table(
    column_defs: List[Dict[str, Any]],
    row_data: List[Dict[str, Any]],
    table_id: str,
    height: str = "400px",
    sort_model: Optional[List[Dict[str, str]]] = None,
) -> dag.AgGrid:
    """
    Build a dark-themed AG Grid table.

    Parameters
    ----------
    column_defs : list of dict
        AG Grid column definitions.  Each dict must have at least
        ``"field"`` and ``"headerName"`` keys.
    row_data : list of dict
        Row records.
    table_id : str
        Dash component ID.
    height : str
        CSS height string.
    sort_model : list of dict, optional
        Initial sort (e.g. ``[{"colId": "mae", "sort": "asc"}]``).

    Returns
    -------
    dag.AgGrid
    """
    grid_options: Dict[str, Any] = {
        "animateRows": True,
        "pagination": False,
    }
    if sort_model:
        grid_options["sortModel"] = sort_model

    return dag.AgGrid(
        id=table_id,
        columnDefs=column_defs,
        rowData=row_data,
        defaultColDef={
            "sortable": True,
            "resizable": True,
            "filter": True,
        },
        dashGridOptions=grid_options,
        style={
            "height": height,
            "--ag-background-color": BG_CARD,
            "--ag-header-background-color": BG_CARD,
            "--ag-odd-row-background-color": BG_CARD,
            "--ag-row-hover-color": "#1c2333",
            "--ag-foreground-color": TEXT_PRIMARY,
            "--ag-border-color": BORDER,
            "--ag-header-foreground-color": TEXT_PRIMARY,
            "--ag-font-size": "13px",
        },
    )
```

---

### 16.19 `src/ui/apps/ensemble_analytics/components/filter_bar.py`

```python
"""
Multi-select filter bar for Evaluation sub-tabs.

Renders dropdowns for desk, product_type, and ccy columns from the
global trade catalogue.  Returns a mask-building function alongside
the layout.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def filter_bar(
    catalogue: pd.DataFrame,
    id_prefix: str,
    columns: Optional[List[str]] = None,
) -> html.Div:
    """
    Build a horizontal row of multi-select dropdowns.

    Parameters
    ----------
    catalogue : pd.DataFrame
        Global trade catalogue.
    id_prefix : str
        Component ID prefix (e.g. ``"eval-desk"``).
    columns : list of str, optional
        Catalogue columns to create filters for.  Defaults to
        ``["desk", "product_type", "ccy"]`` (skipping any not present
        in the catalogue).

    Returns
    -------
    html.Div
    """
    if columns is None:
        columns = ["desk", "product_type", "ccy"]
    columns = [c for c in columns if c in catalogue.columns]

    children = []
    for col in columns:
        unique_vals = sorted(catalogue[col].dropna().unique().tolist())
        children.append(
            html.Div(
                [
                    html.Label(
                        col.replace("_", " ").title() + ":",
                        style={
                            "color": TEXT_SECONDARY,
                            "fontSize": "12px",
                            "marginRight": "6px",
                            "fontWeight": "600",
                        },
                    ),
                    dcc.Dropdown(
                        id=f"{id_prefix}-filter-{col}",
                        options=[{"label": v, "value": v} for v in unique_vals],
                        multi=True,
                        placeholder=f"All {col.replace('_', ' ').title()}s",
                        style={"width": "220px", "fontSize": "13px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginRight": "20px",
                },
            )
        )

    return html.Div(
        children,
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": "16px"},
    )
```

---

### 16.20 `src/ui/apps/ensemble_analytics/components/loading_progress.py`

```python
"""
Phase 3 loading progress bar for the Inference tab.

Shows how many clusters have been loaded out of the total, with a
visual progress bar and status text.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, TEXT_SECONDARY


def loading_progress(
    total: int,
    loaded: int,
    id_prefix: str = "inference",
) -> html.Div:
    """
    Build a loading progress indicator.

    Parameters
    ----------
    total : int
        Total number of clusters to load.
    loaded : int
        Number of clusters loaded so far.
    id_prefix : str
        Component ID prefix.

    Returns
    -------
    html.Div
    """
    pct = int((loaded / total) * 100) if total > 0 else 0

    return html.Div(
        [
            html.Div(
                f"Loading models: {loaded} / {total} clusters ({pct}%)",
                id=f"{id_prefix}-progress-text",
                style={"color": TEXT_SECONDARY, "fontSize": "13px", "marginBottom": "8px"},
            ),
            dbc.Progress(
                id=f"{id_prefix}-progress-bar",
                value=pct,
                max=100,
                color="info",
                style={"height": "8px"},
            ),
        ],
        id=f"{id_prefix}-progress-container",
    )
```

---

### 16.21 `src/ui/apps/ensemble_analytics/figures/__init__.py`

```python
"""Plotly figure builders — pure functions returning go.Figure."""
```

---

### 16.22 `src/ui/apps/ensemble_analytics/figures/scatter.py`

```python
"""
Scatter plot figure builders.

All functions return ``go.Figure`` with the global template applied.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_RED, TEXT_SECONDARY


def pred_vs_target_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Predictions vs Targets",
    max_points: int = 5_000,
) -> go.Figure:
    """
    Scatter plot of predicted vs actual values with a 45-degree reference line.

    Parameters
    ----------
    predictions : np.ndarray
        1-D array of predicted values.
    targets : np.ndarray
        1-D array of actual values.
    title : str
        Figure title.
    max_points : int
        Downsample to this many points if the array is larger.

    Returns
    -------
    go.Figure
    """
    if len(predictions) > max_points:
        idx = np.random.default_rng(42).choice(len(predictions), max_points, replace=False)
        predictions, targets = predictions[idx], targets[idx]

    vmin = min(predictions.min(), targets.min())
    vmax = max(predictions.max(), targets.max())

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=targets,
        y=predictions,
        mode="markers",
        marker=dict(size=3, color=ACCENT_BLUE, opacity=0.5),
        name="Predicted",
    ))
    fig.add_trace(go.Scattergl(
        x=[vmin, vmax],
        y=[vmin, vmax],
        mode="lines",
        line=dict(color=TEXT_SECONDARY, dash="dash", width=1),
        name="Perfect",
        showlegend=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Target",
        yaxis_title="Prediction",
        height=450,
    )
    return fig


def residual_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Residuals",
) -> go.Figure:
    """
    Scatter plot of residuals (prediction - target) vs target.

    Parameters
    ----------
    predictions : np.ndarray
        1-D predicted values.
    targets : np.ndarray
        1-D actual values.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    residuals = predictions - targets

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=targets,
        y=residuals,
        mode="markers",
        marker=dict(
            size=3,
            color=np.where(residuals >= 0, ACCENT_BLUE, ACCENT_RED),
            opacity=0.5,
        ),
        name="Residual",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_SECONDARY, line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Target",
        yaxis_title="Residual (Pred − Target)",
        height=400,
    )
    return fig
```

---

### 16.23 `src/ui/apps/ensemble_analytics/figures/timeseries.py`

```python
"""
Time-series / scenario-indexed line-chart figure builders.

Scenarios are treated as ordered indices (0, 1, 2, ...) since the
underlying data is scenario-based, not calendar-dated.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_GREEN, CHART_COLORS


def pnl_timeseries(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Portfolio PnL — Predictions vs Targets",
) -> go.Figure:
    """
    Overlay predicted and target PnL as line charts over scenario index.

    Parameters
    ----------
    predictions : np.ndarray
        1-D scenario-ordered predicted PnL.
    targets : np.ndarray
        1-D scenario-ordered target PnL.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    x = np.arange(len(predictions))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x, y=targets, mode="lines",
        name="Target", line=dict(color=ACCENT_GREEN, width=1.5),
    ))
    fig.add_trace(go.Scattergl(
        x=x, y=predictions, mode="lines",
        name="Prediction", line=dict(color=ACCENT_BLUE, width=1.5),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title="PnL",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def overlaid_group_timeseries(
    group_data: Dict[str, np.ndarray],
    title: str = "PnL by Group",
    y_label: str = "PnL",
) -> go.Figure:
    """
    Overlay multiple groups as separate lines (e.g. desks or ccys).

    Parameters
    ----------
    group_data : dict
        ``{group_label: 1-D np.ndarray}``.
    title : str
        Figure title.
    y_label : str
        Y-axis label.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, (label, values) in enumerate(group_data.items()):
        fig.add_trace(go.Scattergl(
            x=np.arange(len(values)),
            y=values,
            mode="lines",
            name=label,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        yaxis_title=y_label,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
```

---

### 16.24 `src/ui/apps/ensemble_analytics/figures/distributions.py`

```python
"""
Distribution plot figure builders — histograms, violins, QQ plots.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import (
    ACCENT_BLUE,
    ACCENT_RED,
    ACCENT_GREEN,
    CHART_COLORS,
    TEXT_SECONDARY,
)


def residual_histogram(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "Residual Distribution",
    nbins: int = 80,
) -> go.Figure:
    """
    Histogram of residuals (prediction - target) with annotation box
    showing mean, std, skew, kurtosis, and % within ±1σ/±2σ.

    Parameters
    ----------
    predictions : np.ndarray
        1-D predicted values.
    targets : np.ndarray
        1-D target values.
    title : str
        Figure title.
    nbins : int
        Number of histogram bins.

    Returns
    -------
    go.Figure
    """
    from scipy import stats as _stats

    residuals = predictions - targets
    mu = float(residuals.mean())
    sigma = float(residuals.std())
    skew = float(_stats.skew(residuals))
    kurt = float(_stats.kurtosis(residuals))
    pct_1s = float(np.mean(np.abs(residuals - mu) <= sigma) * 100)
    pct_2s = float(np.mean(np.abs(residuals - mu) <= 2 * sigma) * 100)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=nbins,
        marker_color=ACCENT_BLUE, opacity=0.8,
        name="Residuals",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=TEXT_SECONDARY, line_width=1)

    annotation_text = (
        f"μ={mu:.4f}  σ={sigma:.4f}<br>"
        f"skew={skew:.2f}  kurt={kurt:.2f}<br>"
        f"±1σ: {pct_1s:.1f}%  ±2σ: {pct_2s:.1f}%"
    )
    fig.add_annotation(
        text=annotation_text, xref="paper", yref="paper",
        x=0.98, y=0.95, showarrow=False,
        font=dict(size=11, color=TEXT_SECONDARY),
        align="right", bgcolor="rgba(22,27,34,0.8)", bordercolor=ACCENT_BLUE,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Residual (Pred − Target)",
        yaxis_title="Count",
        height=350,
    )
    return fig


def violin_overlay(
    group_data: Dict[str, np.ndarray],
    title: str = "Distribution Comparison",
) -> go.Figure:
    """
    Overlay violin plots for multiple groups.

    Parameters
    ----------
    group_data : dict
        ``{group_label: 1-D values}``.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, (label, values) in enumerate(group_data.items()):
        fig.add_trace(go.Violin(
            y=values, name=label,
            line_color=CHART_COLORS[i % len(CHART_COLORS)],
            meanline_visible=True,
            box_visible=True,
        ))
    fig.update_layout(title=title, height=400, showlegend=False)
    return fig


def qq_plot(
    sample: np.ndarray,
    title: str = "Q-Q Plot (Normal)",
) -> go.Figure:
    """
    Quantile-quantile plot against a standard normal distribution.

    Parameters
    ----------
    sample : np.ndarray
        1-D sample values.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    from scipy import stats

    sorted_sample = np.sort(sample)
    n = len(sorted_sample)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, n))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=theoretical, y=sorted_sample,
        mode="markers",
        marker=dict(size=3, color=ACCENT_BLUE, opacity=0.6),
        name="Sample",
    ))
    qmin, qmax = theoretical.min(), theoretical.max()
    fig.add_trace(go.Scattergl(
        x=[qmin, qmax], y=[qmin, qmax],
        mode="lines",
        line=dict(color=ACCENT_RED, dash="dash", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=400,
    )
    return fig
```

---

### 16.25 `src/ui/apps/ensemble_analytics/figures/heatmaps.py`

```python
"""
Heatmap figure builders — cluster performance, RF scenarios, adjacency.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import BG_CARD, TEXT_SECONDARY


def cluster_heatmap(
    cluster_ids: List[str],
    metric_values: Dict[str, float],
    metric_name: str = "MAE",
    title: str = "Cluster Performance Heatmap",
) -> go.Figure:
    """
    Single-row heatmap showing one metric per cluster.

    Parameters
    ----------
    cluster_ids : list of str
        Cluster identifiers (x-axis).
    metric_values : dict
        ``{cluster_id: metric_value}``.
    metric_name : str
        Metric label for the colour bar.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    values = [[metric_values.get(cid, 0.0) for cid in cluster_ids]]

    fig = go.Figure(go.Heatmap(
        z=values,
        x=cluster_ids,
        y=[metric_name],
        colorscale="RdYlGn_r",
        text=[[f"{v:.4f}" for v in values[0]]],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title=metric_name, len=0.5),
    ))
    fig.update_layout(
        title=title,
        height=160 + 30 * max(1, len(cluster_ids) // 10),
        xaxis=dict(tickangle=-45),
        yaxis=dict(showticklabels=False),
    )
    return fig


def multi_metric_cluster_heatmap(
    cluster_ids: List[str],
    per_member_metrics: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None,
    title: str = "Cluster Performance Heatmap",
) -> go.Figure:
    """
    Multi-row heatmap: rows = metric names, cols = clusters.

    Designed for the Overview tab — shows MAE, RMSE, MaxAE, P95, P99
    for every cluster in one view with colour intensity encoding.

    Parameters
    ----------
    cluster_ids : list of str
        Cluster identifiers (x-axis).
    per_member_metrics : dict
        ``{cluster_id: {metric: value, ...}}``.
    metric_keys : list of str, optional
        Metrics to show.  Defaults to ``["mae", "rmse", "max_ae", "p95_ae", "p99_ae"]``.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    if metric_keys is None:
        metric_keys = ["mae", "rmse", "max_ae", "p95_ae", "p99_ae"]

    z = []
    text = []
    for mk in metric_keys:
        row = [per_member_metrics.get(cid, {}).get(mk, 0.0) for cid in cluster_ids]
        z.append(row)
        text.append([f"{v:.4f}" for v in row])

    display_names = {
        "mae": "MAE", "rmse": "RMSE", "max_ae": "Max AE",
        "p95_ae": "P95 AE", "p99_ae": "P99 AE", "mape": "MAPE", "r2": "R²",
    }
    y_labels = [display_names.get(mk, mk.upper()) for mk in metric_keys]

    fig = go.Figure(go.Heatmap(
        z=z, x=cluster_ids, y=y_labels,
        colorscale="RdYlGn_r",
        text=text, texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="Value", len=0.6),
    ))
    fig.update_layout(
        title=title,
        height=max(200, 40 * len(metric_keys) + 100),
        xaxis=dict(tickangle=-45),
    )
    return fig


def rf_scenario_heatmap(
    rf_names: List[str],
    shock_matrix: np.ndarray,
    title: str = "RF × Scenario Heatmap",
    max_scenarios: int = 200,
) -> go.Figure:
    """
    Heatmap of risk-factor shocks across scenarios.

    Parameters
    ----------
    rf_names : list of str
        Risk factor names (y-axis).
    shock_matrix : np.ndarray
        Shape ``[n_scenarios, n_rfs]``.
    title : str
        Figure title.
    max_scenarios : int
        Downsample scenarios if larger.

    Returns
    -------
    go.Figure
    """
    if shock_matrix.shape[0] > max_scenarios:
        idx = np.linspace(0, shock_matrix.shape[0] - 1, max_scenarios, dtype=int)
        shock_matrix = shock_matrix[idx]

    fig = go.Figure(go.Heatmap(
        z=shock_matrix.T,
        x=list(range(shock_matrix.shape[0])),
        y=rf_names,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Shock"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Scenario",
        height=max(300, 20 * len(rf_names) + 100),
    )
    return fig


def adjacency_spy(
    indices: np.ndarray,
    values: np.ndarray,
    shape: List[int],
    title: str = "Adjacency Matrix (Spy Plot)",
) -> go.Figure:
    """
    Sparse matrix spy plot from COO-format adjacency data.

    Parameters
    ----------
    indices : np.ndarray
        Shape ``[2, nnz]`` — row and column indices.
    values : np.ndarray
        Edge weights, shape ``[nnz]``.
    shape : list of int
        ``[n_nodes, n_nodes]``.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    if indices.ndim == 2 and indices.shape[0] == 2:
        rows, cols = indices[0], indices[1]
    elif indices.ndim == 2 and indices.shape[1] == 2:
        rows, cols = indices[:, 0], indices[:, 1]
    else:
        rows, cols = indices[0], indices[1]

    fig = go.Figure(go.Scattergl(
        x=cols,
        y=rows,
        mode="markers",
        marker=dict(size=2, color=values, colorscale="Viridis", showscale=True),
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, shape[1]], title="Column"),
        yaxis=dict(range=[shape[0], 0], title="Row", scaleanchor="x"),
        height=500,
    )
    return fig
```

---

### 16.26 `src/ui/apps/ensemble_analytics/figures/bar_charts.py`

```python
"""
Bar chart figure builders — member comparisons, grouped split bars.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go

from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, CHART_COLORS


def member_comparison_bar(
    cluster_ids: List[str],
    metric_values: Dict[str, float],
    metric_name: str = "MAE",
    title: str = "Member Comparison",
) -> go.Figure:
    """
    Horizontal bar chart comparing one metric across cluster members.

    Parameters
    ----------
    cluster_ids : list of str
        Cluster identifiers.
    metric_values : dict
        ``{cluster_id: float}``.
    metric_name : str
        Metric display name.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    values = [metric_values.get(cid, 0.0) for cid in cluster_ids]

    fig = go.Figure(go.Bar(
        x=values,
        y=cluster_ids,
        orientation="h",
        marker_color=ACCENT_BLUE,
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=metric_name,
        height=max(300, 28 * len(cluster_ids) + 80),
        yaxis=dict(categoryorder="total ascending"),
    )
    return fig


def grouped_split_bar(
    labels: List[str],
    split_values: Dict[str, List[float]],
    title: str = "Metric by Split",
    y_label: str = "Value",
) -> go.Figure:
    """
    Grouped bar chart with one group per split.

    Parameters
    ----------
    labels : list of str
        X-axis category labels.
    split_values : dict
        ``{split_name: [float values per label]}``.
    title : str
        Figure title.
    y_label : str
        Y-axis label.

    Returns
    -------
    go.Figure
    """
    split_colors = {"test": ACCENT_BLUE, "val": ACCENT_AMBER, "train": ACCENT_GREEN}

    fig = go.Figure()
    for split_name, values in split_values.items():
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            name=split_name.capitalize(),
            marker_color=split_colors.get(split_name, ACCENT_BLUE),
        ))
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        barmode="group",
        height=400,
    )
    return fig
```

---

### 16.27 `src/ui/apps/ensemble_analytics/figures/network.py`

```python
"""
Graph / network figure builders for ``dash-cytoscape``.

Returns element lists suitable for ``cyto.Cytoscape(elements=...)``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def build_cytoscape_elements(
    trade_ids: List[str],
    indices: np.ndarray,
    values: np.ndarray,
    node_attrs: Optional[Dict[str, Dict[str, Any]]] = None,
    weight_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Convert sparse adjacency to cytoscape element dicts.

    Parameters
    ----------
    trade_ids : list of str
        Node labels.
    indices : np.ndarray
        Shape ``[2, nnz]`` or ``[nnz, 2]`` — edge endpoints.
    values : np.ndarray
        Edge weights, shape ``[nnz]``.
    node_attrs : dict, optional
        ``{trade_id: {attr: val, ...}}``.  Extra attributes are added to
        each node's ``data`` dict (useful for colouring / sizing).
    weight_threshold : float
        Only include edges with ``abs(weight) > threshold``.

    Returns
    -------
    list of dict
        Cytoscape element dicts (nodes + edges).
    """
    elements: List[Dict[str, Any]] = []

    for tid in trade_ids:
        node_data: Dict[str, Any] = {"id": tid, "label": tid}
        if node_attrs and tid in node_attrs:
            node_data.update(node_attrs[tid])
        elements.append({"data": node_data})

    if indices.ndim == 2 and indices.shape[0] == 2:
        rows, cols = indices[0], indices[1]
    elif indices.ndim == 2 and indices.shape[1] == 2:
        rows, cols = indices[:, 0], indices[:, 1]
    else:
        return elements

    for i in range(len(values)):
        if abs(values[i]) <= weight_threshold:
            continue
        src = trade_ids[int(rows[i])] if int(rows[i]) < len(trade_ids) else str(int(rows[i]))
        tgt = trade_ids[int(cols[i])] if int(cols[i]) < len(trade_ids) else str(int(cols[i]))
        elements.append({
            "data": {
                "source": src,
                "target": tgt,
                "weight": float(values[i]),
            }
        })

    return elements


def ego_network(
    center_id: str,
    trade_ids: List[str],
    indices: np.ndarray,
    values: np.ndarray,
    hops: int = 1,
) -> List[Dict[str, Any]]:
    """
    Extract the ego (neighbourhood) subgraph around a single node.

    Parameters
    ----------
    center_id : str
        Focal node.
    trade_ids : list of str
        Node labels.
    indices : np.ndarray
        Sparse adjacency indices.
    values : np.ndarray
        Edge weights.
    hops : int
        Number of hops from the centre.

    Returns
    -------
    list of dict
        Cytoscape element dicts for the subgraph.
    """
    if center_id not in trade_ids:
        return []

    tid_to_idx = {tid: i for i, tid in enumerate(trade_ids)}
    center_idx = tid_to_idx[center_id]

    if indices.ndim == 2 and indices.shape[0] == 2:
        rows, cols = indices[0], indices[1]
    else:
        rows, cols = indices[:, 0], indices[:, 1]

    visited = {center_idx}
    frontier = {center_idx}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            mask_src = rows == node
            mask_tgt = cols == node
            neighbours = set(cols[mask_src].tolist()) | set(rows[mask_tgt].tolist())
            next_frontier |= neighbours - visited
        visited |= next_frontier
        frontier = next_frontier

    visited_set = visited
    elements: List[Dict[str, Any]] = []
    for idx in visited_set:
        if idx < len(trade_ids):
            data: Dict[str, Any] = {"id": trade_ids[idx], "label": trade_ids[idx]}
            if idx == center_idx:
                data["is_center"] = True
            elements.append({"data": data})

    for i in range(len(values)):
        r, c = int(rows[i]), int(cols[i])
        if r in visited_set and c in visited_set:
            elements.append({
                "data": {
                    "source": trade_ids[r] if r < len(trade_ids) else str(r),
                    "target": trade_ids[c] if c < len(trade_ids) else str(c),
                    "weight": float(values[i]),
                }
            })

    return elements
```

---

### 16.28 `src/ui/apps/ensemble_analytics/figures/tables.py`

```python
"""
Table data builders — prepare row data for AG Grid metric tables.

Return ``(column_defs, row_data)`` tuples ready for ``metric_table()``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def percentile_table_data(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a percentile breakdown table comparing predicted vs target
    distributions — shows Predicted, Target, and Diff at each percentile.

    Returns
    -------
    tuple of (column_defs, row_data)
    """
    fmt = {"function": "d3.format('.4f')(params.value)"}
    column_defs = [
        {"field": "percentile", "headerName": "Percentile"},
        {"field": "predicted", "headerName": "Predicted", "valueFormatter": fmt},
        {"field": "target", "headerName": "Target", "valueFormatter": fmt},
        {"field": "diff", "headerName": "Diff", "valueFormatter": fmt},
        {"field": "abs_error", "headerName": "Abs Error", "valueFormatter": fmt},
    ]

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    row_data = []
    for p in percentiles:
        pred_p = float(np.percentile(predictions, p))
        targ_p = float(np.percentile(targets, p))
        row_data.append({
            "percentile": f"P{p}",
            "predicted": pred_p,
            "target": targ_p,
            "diff": pred_p - targ_p,
            "abs_error": float(np.percentile(np.abs(predictions - targets), p)),
        })

    row_data.append({
        "percentile": "Mean",
        "predicted": float(predictions.mean()),
        "target": float(targets.mean()),
        "diff": float((predictions - targets).mean()),
        "abs_error": float(np.abs(predictions - targets).mean()),
    })

    return column_defs, row_data


def worst_scenarios_data(
    predictions: np.ndarray,
    targets: np.ndarray,
    top_n: int = 20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a table of the worst (highest absolute error) scenarios.

    Returns
    -------
    tuple of (column_defs, row_data)
    """
    residuals = predictions - targets
    abs_errors = np.abs(residuals)
    worst_idx = np.argsort(abs_errors)[::-1][:top_n]

    column_defs = [
        {"field": "rank", "headerName": "#", "width": 60},
        {"field": "scenario", "headerName": "Scenario"},
        {"field": "target", "headerName": "Target", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        {"field": "prediction", "headerName": "Prediction", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        {"field": "abs_error", "headerName": "Abs Error", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
    ]

    row_data = [
        {
            "rank": rank + 1,
            "scenario": int(idx),
            "target": float(targets[idx]),
            "prediction": float(predictions[idx]),
            "abs_error": float(abs_errors[idx]),
        }
        for rank, idx in enumerate(worst_idx)
    ]

    return column_defs, row_data
```

---

## 9. Tab 1 — Overview

### Purpose

30-second production-readiness check for senior traders / HoD / CRO.
Answers: *"Is this ensemble model production-ready?"*

### Data sources

- `session.ensemble_display.ensemble_metrics[split]` — portfolio KPIs
- `session.ensemble_display.per_member_metrics[split]` — per-member metrics for table + bar chart
- `session.cluster_attributes` — cluster labels for heatmap
- `session.config.cluster_ids`, `session.config.n_members` — counts

### Layout (top to bottom)

1. **Split toggle** (radio: Test / Val / Train)
2. **KPI row** — 5 cards: MAE, RMSE, Max AE, P95 AE, P99 AE
3. **Row of two charts:**
   - Left: Predictions vs Targets scatter (portfolio-level combined)
   - Right: Member comparison bar chart (MAE per cluster)
4. **Cluster performance heatmap** (single-row, colour-coded by MAE)
5. **Sortable member KPI table** (all metrics per cluster, AG Grid)

### Callback flow

```
overview-split-toggle  ──→  update_overview()
                             ├─ reads ensemble_display.ensemble_metrics[split]
                             ├─ reads ensemble_display.per_member_metrics[split]
                             ├─ reads combined predictions .npz for scatter
                             └─ returns: KPI cards, scatter fig, bar fig,
                                         heatmap fig, table row_data
```

---

## 15. Tab 7 — Model Governance

### Purpose

Audit trail and configuration transparency for Risk and regulatory review.
Answers: *"What exactly was trained, when, with what settings?"*

### Data sources

- `session.ensemble_display.manifest` — trade IDs, cluster mapping
- `session.config` — full `EnsembleConfig` dataclass
- `session.member_versions` — version strings per cluster
- `session.trade_cluster_map` — trade-to-cluster lookup

### Layout

1. **Ensemble manifest card** — version, n_clusters, n_trades, splits, aggregation
2. **Member registry table** — cluster_id, version, registry path
3. **Config inspector** — collapsible JSON tree of the full `EnsembleConfig`
4. **Trade-cluster map** — searchable table: trade_id → cluster_id

### Callback flow

```
(no dynamic inputs — layout is static after Phase 1+2)
Only the ensemble-version-selector triggers a full reload.
```

---

### 16.29 `src/ui/apps/ensemble_analytics/tabs/__init__.py`

```python
"""Tab layout modules for the Ensemble Analytics dashboard.

Each tab module exposes a ``layout()`` function returning a Dash
component tree.  Tab routing is handled by the callback in
``callbacks/__init__.py``.
"""
```

---

### 16.30 `src/ui/apps/ensemble_analytics/tabs/overview.py`

```python
"""
Tab 1 — Overview layout.

Renders the production-readiness dashboard: KPI cards, portfolio scatter,
member comparison bar, cluster heatmap, and sortable member table.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.ui.apps.ensemble_analytics.components.split_toggle import split_toggle
from src.ui.apps.ensemble_analytics.components.kpi_card import kpi_card
from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE
from src.ui.apps.ensemble_analytics.config import DEFAULT_SPLIT


def layout() -> html.Div:
    """
    Build the Overview tab layout.

    All dynamic content is rendered by callbacks via the placeholder
    ``id`` attributes.  This function builds the static skeleton.

    Returns
    -------
    html.Div
    """
    return html.Div([
        split_toggle(id_prefix="overview", default=DEFAULT_SPLIT),

        # KPI row (populated by callback)
        dbc.Row(id="overview-kpi-row", className="g-3 mb-4"),

        # Charts row
        dbc.Row([
            dbc.Col(
                html.Div(id="overview-scatter-container", style=CARD_STYLE),
                md=6,
            ),
            dbc.Col(
                html.Div(id="overview-bar-container", style=CARD_STYLE),
                md=6,
            ),
        ], className="g-3 mb-4"),

        # Cluster heatmap
        html.Div(
            "Cluster Performance",
            style=SECTION_TITLE_STYLE,
        ),
        html.Div(id="overview-heatmap-container", style=CARD_STYLE),

        # Member KPI table
        html.Div(
            "Member Metrics",
            style=SECTION_TITLE_STYLE,
        ),
        html.Div(id="overview-table-container", style=CARD_STYLE),
    ])
```

---

### 16.31 `src/ui/apps/ensemble_analytics/callbacks/overview_cb.py`

```python
"""
Callbacks for Tab 1 — Overview.

Single callback triggered by the split toggle.  Reads ensemble-level
display state and combined prediction arrays, builds all visual
elements, and returns them to the layout placeholders.
"""
from __future__ import annotations

import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import METRIC_DISPLAY_NAMES
from src.ui.apps.ensemble_analytics.components.kpi_card import kpi_card
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.bar_charts import member_comparison_bar
from src.ui.apps.ensemble_analytics.figures.heatmaps import multi_metric_cluster_heatmap


def register(app):
    """Register Overview tab callbacks on *app*."""

    @app.callback(
        Output("overview-kpi-row", "children"),
        Output("overview-scatter-container", "children"),
        Output("overview-bar-container", "children"),
        Output("overview-heatmap-container", "children"),
        Output("overview-table-container", "children"),
        Input("overview-split-toggle", "value"),
    )
    def update_overview(split: str):
        """Rebuild all Overview visuals when the split toggle changes."""
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store

        session = get_session()
        ens_display = session.ensemble_display
        if ens_display is None:
            return no_update, no_update, no_update, no_update, no_update

        # ── KPI cards ─────────────────────────────────────────────
        ens_metrics = ens_display.ensemble_metrics.get(split, {})
        kpi_keys = ["mae", "rmse", "max_ae", "p95_ae", "p99_ae"]
        kpi_cards = []
        for key in kpi_keys:
            val = ens_metrics.get(key)
            display_val = f"{val:.4f}" if val is not None else "N/A"
            kpi_cards.append(
                dbc.Col(
                    kpi_card(
                        title=METRIC_DISPLAY_NAMES.get(key, key.upper()),
                        value=display_val,
                    ),
                    md=True,
                )
            )

        # ── Portfolio scatter ─────────────────────────────────────
        scatter_fig = html.Div("No prediction data available.")
        store = get_prediction_store(split)
        if store is not None:
            portfolio_preds = store.predictions.sum(axis=1)
            portfolio_targets = store.targets.sum(axis=1)
            scatter_fig = dcc.Graph(
                figure=pred_vs_target_scatter(
                    portfolio_preds, portfolio_targets,
                    title=f"Portfolio PnL — {split.capitalize()}",
                ),
                config={"displayModeBar": False},
            )

        # ── Member comparison bar ─────────────────────────────────
        pm_metrics = ens_display.per_member_metrics.get(split, {})
        cluster_ids = session.config.cluster_ids
        mae_by_cluster = {
            cid: pm_metrics.get(cid, {}).get("mae", 0.0)
            for cid in cluster_ids
        }
        bar_fig = dcc.Graph(
            figure=member_comparison_bar(
                cluster_ids, mae_by_cluster,
                metric_name="MAE",
                title=f"MAE by Cluster — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )

        # ── Multi-metric cluster heatmap (G3) ─────────────────────
        heatmap_fig = dcc.Graph(
            figure=multi_metric_cluster_heatmap(
                cluster_ids, pm_metrics,
                title=f"Cluster Metrics Heatmap — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )

        # ── Member table with cluster attributes (G1) + conditional formatting (G2) ──
        cluster_attrs = session.cluster_attributes
        column_defs = [
            {"field": "cluster_id", "headerName": "Cluster", "pinned": "left"},
        ]
        attr_cols = ["ccy", "desk", "product"]
        for ac in attr_cols:
            column_defs.append({"field": ac, "headerName": ac.upper()})

        metric_keys = list(next(iter(pm_metrics.values()), {}).keys()) if pm_metrics else []
        all_metric_vals = {mk: [] for mk in metric_keys}
        for cid in cluster_ids:
            for mk in metric_keys:
                v = pm_metrics.get(cid, {}).get(mk)
                if v is not None:
                    all_metric_vals[mk].append(v)

        p25 = {mk: float(np.percentile(vs, 25)) if vs else 0 for mk, vs in all_metric_vals.items()}
        p75 = {mk: float(np.percentile(vs, 75)) if vs else 0 for mk, vs in all_metric_vals.items()}

        for mk in metric_keys:
            column_defs.append({
                "field": mk,
                "headerName": METRIC_DISPLAY_NAMES.get(mk, mk.upper()),
                "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
                "cellStyle": {
                    "styleConditions": [
                        {"condition": f"params.value < {p25[mk]}", "style": {"color": "#3fb950"}},
                        {"condition": f"params.value >= {p25[mk]} && params.value <= {p75[mk]}", "style": {"color": "#d29922"}},
                        {"condition": f"params.value > {p75[mk]}", "style": {"color": "#f85149"}},
                    ]
                },
            })

        column_defs.append({"field": "n_trades", "headerName": "# Trades"})

        row_data = []
        for cid in cluster_ids:
            row = {"cluster_id": cid}
            ca = cluster_attrs.get(cid, {})
            for ac in attr_cols:
                row[ac] = ca.get(ac, "")
            row.update(pm_metrics.get(cid, {}))
            row["n_trades"] = len(session.config.cluster_mapping.get(cid, []))
            row_data.append(row)

        table = metric_table(
            column_defs=column_defs,
            row_data=row_data,
            table_id="overview-member-table",
            sort_model=[{"colId": "mae", "sort": "asc"}],
        )

        return kpi_cards, scatter_fig, bar_fig, heatmap_fig, table
```

---

### 16.32 `src/ui/apps/ensemble_analytics/tabs/governance.py`

```python
"""
Tab 7 — Model Governance layout.

Static audit-trail view: ensemble manifest, member registry, config
inspector, and searchable trade-cluster map.
"""
from __future__ import annotations

import json

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE, CARD_HEADER_STYLE
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def layout() -> html.Div:
    """
    Build the Model Governance tab layout.

    Mostly static content populated at load time from session metadata.
    The trade-cluster search is handled by a client-side callback.

    Returns
    -------
    html.Div
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    from src.ui.apps.ensemble_analytics.components.metric_table import metric_table

    session = get_session()
    config = session.config
    ens_display = session.ensemble_display
    manifest = ens_display.manifest if ens_display else {}

    # ── Ensemble manifest card ────────────────────────────────────
    manifest_items = [
        ("Ensemble Version", session.ensemble_version),
        ("Clusters", str(config.n_members)),
        ("Total Trades", str(len(config.all_trade_ids))),
        ("Aggregation", config.aggregation),
        ("Execution Strategy", config.execution_strategy),
        ("Splits Available", ", ".join(manifest.get("splits_available", []))),
    ]
    manifest_card = html.Div(
        [html.Div("Ensemble Manifest", style=CARD_HEADER_STYLE)]
        + [
            html.Div(
                [
                    html.Span(f"{label}: ", style={"color": TEXT_SECONDARY, "fontWeight": "600"}),
                    html.Span(value),
                ],
                style={"marginBottom": "4px", "fontSize": "13px"},
            )
            for label, value in manifest_items
        ],
        style=CARD_STYLE,
    )

    # ── Member registry table ─────────────────────────────────────
    member_versions = session.member_versions or {}
    member_col_defs = [
        {"field": "cluster_id", "headerName": "Cluster"},
        {"field": "version", "headerName": "Version"},
        {"field": "n_trades", "headerName": "# Trades"},
    ]
    member_row_data = [
        {
            "cluster_id": cid,
            "version": member_versions.get(cid, "unknown"),
            "n_trades": len(config.cluster_mapping.get(cid, [])),
        }
        for cid in config.cluster_ids
    ]
    member_table = metric_table(
        column_defs=member_col_defs,
        row_data=member_row_data,
        table_id="governance-member-table",
        height="300px",
    )

    # ── Config inspector (collapsible JSON tree) ──────────────────
    config_json = json.dumps(config.to_dict(), indent=2, default=str)
    config_card = html.Div(
        [
            html.Div("Configuration", style=CARD_HEADER_STYLE),
            html.Details(
                [
                    html.Summary(
                        "Expand EnsembleConfig JSON",
                        style={"cursor": "pointer", "color": TEXT_SECONDARY, "fontSize": "13px"},
                    ),
                    dcc.Markdown(
                        f"```json\n{config_json}\n```",
                        style={"fontSize": "12px", "maxHeight": "500px", "overflow": "auto"},
                    ),
                ],
            ),
        ],
        style=CARD_STYLE,
    )

    # ── Trade-cluster map (searchable) ────────────────────────────
    tcm = session.trade_cluster_map or {}
    tcm_col_defs = [
        {"field": "trade_id", "headerName": "Trade ID", "filter": True},
        {"field": "cluster_id", "headerName": "Cluster ID", "filter": True},
    ]
    tcm_row_data = [
        {"trade_id": tid, "cluster_id": cid}
        for tid, cid in tcm.items()
    ]
    tcm_table = metric_table(
        column_defs=tcm_col_defs,
        row_data=tcm_row_data,
        table_id="governance-tcm-table",
        height="400px",
    )

    # ── Version Comparison section (G16) ────────────────────────
    version_comparison = html.Div([
        html.Div("Version Comparison", style=CARD_HEADER_STYLE),
        html.Div([
            html.Label("Compare with:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Dropdown(
                id="governance-compare-version",
                options=[], placeholder="Select version to compare...",
                style={"width": "340px", "fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div(id="governance-comparison-content"),
    ], style=CARD_STYLE)

    return html.Div([
        manifest_card,
        html.Div("Member Registry", style=SECTION_TITLE_STYLE),
        html.Div(member_table, style=CARD_STYLE),
        config_card,
        version_comparison,
        html.Div("Trade → Cluster Map", style=SECTION_TITLE_STYLE),
        html.Div(tcm_table, style=CARD_STYLE),
    ])
```

---

### 16.33 `src/ui/apps/ensemble_analytics/callbacks/governance_cb.py`

```python
"""
Callbacks for Tab 7 — Model Governance.

The governance tab is almost entirely static (populated at layout time).
This module is a placeholder for future dynamic features such as
version-comparison callbacks.
"""
from __future__ import annotations


def register(app):
    """Register Governance tab callbacks on *app*."""
    from dash import Input, Output, State, dcc, html, no_update

    @app.callback(
        Output("governance-compare-version", "options"),
        Input("main-tabs", "value"),
    )
    def populate_compare_dropdown(tab):
        """Populate version comparison dropdown from registry (G16)."""
        if tab != "tab-governance":
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        try:
            versions = session._ens_registry.list_versions()
            return [{"label": v, "value": v} for v in versions]
        except Exception:
            return []

    @app.callback(
        Output("governance-comparison-content", "children"),
        Input("governance-compare-version", "value"),
    )
    def compare_versions(compare_version):
        """Show metric delta table between current and selected version (G16)."""
        if not compare_version:
            return html.Div("Select a version above to compare.", style={"color": "#8b949e", "fontSize": "13px"})

        import json as _json
        from pathlib import Path
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
        import plotly.graph_objects as go

        session = get_session()
        current_ens = session.ensemble_display
        if not current_ens:
            return html.Div("Current display data not loaded.")

        current_metrics = current_ens.ensemble_metrics.get("test", {})

        compare_dir = session.artifacts_dir / "ensemble" / compare_version / "evaluation"
        compare_path = compare_dir / "ensemble_metrics.json"
        if not compare_path.exists():
            compare_path = compare_dir / "ensemble_metrics_test.json"
        if not compare_path.exists():
            return html.Div(f"No test metrics found for version {compare_version}.")

        with open(compare_path) as f:
            compare_metrics = _json.load(f)

        col_defs = [
            {"field": "metric", "headerName": "Metric"},
            {"field": "current", "headerName": f"Current", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "compare", "headerName": compare_version, "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "delta", "headerName": "Δ", "valueFormatter": {"function": "d3.format('+.4f')(params.value)"}},
            {"field": "pct_change", "headerName": "% Change", "valueFormatter": {"function": "d3.format('+.1f')(params.value) + '%'"}},
        ]
        rows = []
        all_keys = sorted(set(list(current_metrics.keys()) + list(compare_metrics.keys())))
        for mk in all_keys:
            cv = current_metrics.get(mk)
            ev = compare_metrics.get(mk)
            if cv is not None and ev is not None:
                delta = cv - ev
                pct = (delta / abs(ev) * 100) if ev != 0 else 0
                rows.append({"metric": mk.upper(), "current": cv, "compare": ev, "delta": delta, "pct_change": pct})

        table = metric_table(col_defs, rows, "governance-comparison-table", height="250px")

        labels = [r["metric"] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current", x=labels, y=[r["current"] for r in rows], marker_color="#58a6ff"))
        fig.add_trace(go.Bar(name=compare_version, x=labels, y=[r["compare"] for r in rows], marker_color="#d29922"))
        fig.update_layout(title="Metric Comparison", barmode="group", height=350)

        return html.Div([table, dcc.Graph(figure=fig, config={"displayModeBar": False})])
```

---

## 10. Tab 2 — Evaluation (5 sub-tabs)

### Purpose

Detailed PnL comparison (predictions vs targets) at every aggregation
level: full portfolio, by desk, by product, by ccy, and by cluster.
This is the primary analytics tab for traders and risk.

### Architecture

The Evaluation tab uses a **sub-tab container** (`evaluation/__init__.py`)
that renders the sub-tab bar and a content area.  Each sub-tab has its
own layout module.  A single callback module (`evaluation_cb.py`) handles
all five sub-tabs.

### Data sources (shared across all sub-tabs)

- `get_prediction_store(split)` — `GlobalPredictionStore` for slicing
- `get_trade_catalogue()` — `pd.DataFrame` for filtering by desk/product/ccy
- `session.ensemble_display.per_member_metrics[split]` — per-cluster metrics

### Slicing pattern (repeated in every sub-tab)

```python
catalogue = get_trade_catalogue()
store = get_prediction_store(split)

# Example: filter by desk
mask = catalogue["desk"] == selected_desk
col_indices = np.where(mask.values)[0]

preds_subset  = store.predictions[:, col_indices]   # [n_scenarios, n_filtered]
targets_subset = store.targets[:, col_indices]

# Portfolio aggregation
portfolio_pred   = preds_subset.sum(axis=1)          # [n_scenarios]
portfolio_target = targets_subset.sum(axis=1)
```

### Sub-tab specifications

| Sub-tab | Filter column | Plots | Extra |
|---------|--------------|-------|-------|
| Portfolio | none (all trades) | Time-series, scatter, residual histogram, percentile table, worst scenarios | Click worst scenario → future link to Cluster Deep Dive |
| By Desk | `desk` | Overlaid desk time-series, box plot, metrics table, scatter small multiples | — |
| By Product | `product_type` | Same pattern as By Desk | Complexity badge (L1/L2/L3) |
| By CCY | `ccy` | Same pattern as By Desk | Cross-currency residual correlation heatmap |
| By Cluster | `cluster_id` | Per-trade PnL heatmap, cluster scatter, trade-level metrics, violin overlay | Cluster dropdown |

---

### 16.34 `src/ui/apps/ensemble_analytics/tabs/evaluation/__init__.py`

```python
"""
Tab 2 — Evaluation sub-tab container.

Renders the sub-tab bar and delegates to individual sub-tab layout
modules.  The active sub-tab content is swapped by a callback.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import EVAL_SUB_ORDER, EVAL_SUB_PORTFOLIO
from src.ui.apps.ensemble_analytics.components.split_toggle import split_toggle


def layout() -> html.Div:
    """
    Build the Evaluation tab layout with sub-tab navigation.

    Returns
    -------
    html.Div
    """
    sub_tabs = dcc.Tabs(
        id="eval-sub-tabs",
        value=EVAL_SUB_PORTFOLIO,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in EVAL_SUB_ORDER
        ],
    )

    return html.Div([
        split_toggle(id_prefix="eval", default="test"),
        sub_tabs,
        html.Div(id="eval-sub-tab-content", style={"marginTop": "16px"}),
    ])
```

---

### 16.35 `src/ui/apps/ensemble_analytics/tabs/evaluation/portfolio.py`

```python
"""
Evaluation sub-tab: Portfolio.

Full-book PnL analysis — predictions vs targets summed across all
trades.  Includes time-series overlay, scatter, residual distribution,
percentile table, and worst-scenario table.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Portfolio sub-tab skeleton (populated by callback)."""
    return html.Div([
        # Time-series + scatter row
        dbc.Row([
            dbc.Col(html.Div(id="eval-portfolio-ts", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-portfolio-scatter", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),

        # Residual distribution + percentile table
        dbc.Row([
            dbc.Col(html.Div(id="eval-portfolio-residual", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-portfolio-percentile", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),

        # Worst scenarios table
        html.Div("Worst Scenarios", style=SECTION_TITLE_STYLE),
        html.Div(id="eval-portfolio-worst", style=CARD_STYLE),
    ])
```

---

### 16.36 `src/ui/apps/ensemble_analytics/tabs/evaluation/by_desk.py`

```python
"""
Evaluation sub-tab: By Desk.

Aggregates predictions and targets by desk attribute.  Shows overlaid
time-series, residual box plots, and a metrics table.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the By Desk sub-tab skeleton."""
    return html.Div([
        html.Div(id="eval-desk-filter-bar"),
        dbc.Row([
            dbc.Col(html.Div(id="eval-desk-timeseries", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-desk-boxplot", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="eval-desk-scatter-grid", style=CARD_STYLE),
        html.Div("Desk Metrics", style=SECTION_TITLE_STYLE),
        html.Div(id="eval-desk-table", style=CARD_STYLE),
    ])
```

---

### 16.37 `src/ui/apps/ensemble_analytics/tabs/evaluation/by_product.py`

```python
"""
Evaluation sub-tab: By Product.

Same pattern as By Desk, grouped by ``product_type`` / ``product_subtype``.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the By Product sub-tab skeleton."""
    return html.Div([
        html.Div(id="eval-product-filter-bar"),
        dbc.Row([
            dbc.Col(html.Div(id="eval-product-timeseries", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-product-boxplot", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="eval-product-scatter-grid", style=CARD_STYLE),
        html.Div("Product Metrics", style=SECTION_TITLE_STYLE),
        html.Div(id="eval-product-table", style=CARD_STYLE),
    ])
```

---

### 16.38 `src/ui/apps/ensemble_analytics/tabs/evaluation/by_ccy.py`

```python
"""
Evaluation sub-tab: By CCY.

Same pattern as By Desk, grouped by currency.  Includes an extra
cross-currency residual correlation heatmap.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the By CCY sub-tab skeleton."""
    return html.Div([
        html.Div(id="eval-ccy-filter-bar"),
        dbc.Row([
            dbc.Col(html.Div(id="eval-ccy-timeseries", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-ccy-boxplot", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="eval-ccy-scatter-grid", style=CARD_STYLE),
        dbc.Row([
            dbc.Col(html.Div(id="eval-ccy-correlation", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-ccy-table-container", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
    ])
```

---

### 16.39 `src/ui/apps/ensemble_analytics/tabs/evaluation/by_cluster.py`

```python
"""
Evaluation sub-tab: By Cluster.

Per-cluster view with cluster dropdown, per-trade PnL heatmap,
scatter, trade-level metrics table, and violin overlay.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the By Cluster sub-tab skeleton."""
    return html.Div([
        html.Div(id="eval-cluster-selector-container"),
        html.Div(id="eval-cluster-heatmap", style=CARD_STYLE),
        dbc.Row([
            dbc.Col(html.Div(id="eval-cluster-scatter", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-cluster-violin", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div("Trade-Level Metrics", style=SECTION_TITLE_STYLE),
        html.Div(id="eval-cluster-trade-table", style=CARD_STYLE),
    ])
```

---

### 16.40 `src/ui/apps/ensemble_analytics/callbacks/evaluation_cb.py`

```python
"""
Callbacks for Tab 2 — Evaluation (all 5 sub-tabs).

Handles sub-tab routing and per-sub-tab data loading / figure building.
Uses the shared slicing pattern: catalogue filter → column indices →
store slice → aggregate → plot.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    EVAL_SUB_PORTFOLIO,
    EVAL_SUB_DESK,
    EVAL_SUB_PRODUCT,
    EVAL_SUB_CCY,
    EVAL_SUB_CLUSTER,
    METRIC_DISPLAY_NAMES,
)
from src.ui.apps.ensemble_analytics.components.filter_bar import filter_bar
from src.ui.apps.ensemble_analytics.components.cluster_selector import cluster_selector
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.timeseries import pnl_timeseries, overlaid_group_timeseries
from src.ui.apps.ensemble_analytics.figures.distributions import residual_histogram, violin_overlay
from src.ui.apps.ensemble_analytics.figures.bar_charts import member_comparison_bar
from src.ui.apps.ensemble_analytics.figures.tables import percentile_table_data, worst_scenarios_data


def register(app):
    """Register Evaluation tab callbacks on *app*."""

    # ── Sub-tab routing ───────────────────────────────────────────
    @app.callback(
        Output("eval-sub-tab-content", "children"),
        Input("eval-sub-tabs", "value"),
    )
    def render_eval_sub_tab(sub_tab: str):
        """Swap sub-tab content based on selection."""
        if sub_tab == EVAL_SUB_PORTFOLIO:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.portfolio import layout
            return layout()
        elif sub_tab == EVAL_SUB_DESK:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_desk import layout
            return layout()
        elif sub_tab == EVAL_SUB_PRODUCT:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_product import layout
            return layout()
        elif sub_tab == EVAL_SUB_CCY:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_ccy import layout
            return layout()
        elif sub_tab == EVAL_SUB_CLUSTER:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_cluster import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    # ── Portfolio sub-tab ─────────────────────────────────────────
    @app.callback(
        Output("eval-portfolio-ts", "children"),
        Output("eval-portfolio-scatter", "children"),
        Output("eval-portfolio-residual", "children"),
        Output("eval-portfolio-percentile", "children"),
        Output("eval-portfolio-worst", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
    )
    def update_portfolio(split: str, sub_tab: str):
        """Build Portfolio sub-tab visuals."""
        if sub_tab != EVAL_SUB_PORTFOLIO:
            return no_update, no_update, no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store

        store = get_prediction_store(split)
        if store is None:
            msg = html.Div("No prediction data available for this split.")
            return msg, msg, msg, msg, msg

        portfolio_preds = store.predictions.sum(axis=1)
        portfolio_targets = store.targets.sum(axis=1)

        ts_fig = dcc.Graph(
            figure=pnl_timeseries(portfolio_preds, portfolio_targets,
                                  title=f"Portfolio PnL — {split.capitalize()}"),
            config={"displayModeBar": False},
        )
        scatter_fig = dcc.Graph(
            figure=pred_vs_target_scatter(portfolio_preds, portfolio_targets,
                                          title=f"Pred vs Target — {split.capitalize()}"),
            config={"displayModeBar": False},
        )
        residual_fig = dcc.Graph(
            figure=residual_histogram(portfolio_preds, portfolio_targets,
                                      title=f"Residual Distribution — {split.capitalize()}"),
            config={"displayModeBar": False},
        )

        pct_cols, pct_rows = percentile_table_data(portfolio_preds, portfolio_targets)
        pct_table = metric_table(pct_cols, pct_rows, "eval-portfolio-pct-table", height="220px")

        worst_cols, worst_rows = worst_scenarios_data(portfolio_preds, portfolio_targets)
        worst_table = metric_table(worst_cols, worst_rows, "eval-portfolio-worst-table", height="400px")

        return ts_fig, scatter_fig, residual_fig, pct_table, worst_table

    # ── Generic group-by sub-tab builder ──────────────────────────
    def _build_group_view(
        split: str,
        group_col: str,
        selected_values: Optional[List[str]],
        id_prefix: str,
    ):
        """
        Shared logic for By Desk / By Product / By CCY sub-tabs.

        Filters the trade catalogue by *group_col*, slices the prediction
        store, aggregates per group, and builds the standard chart set
        including a scatter grid (small multiples).

        Returns (timeseries, boxplot, scatter_grid, table).
        """
        from plotly.subplots import make_subplots
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics.theme.colors import CHART_COLORS, TEXT_SECONDARY

        store = get_prediction_store(split)
        catalogue = get_trade_catalogue()
        if store is None or catalogue is None or catalogue.empty:
            msg = html.Div("No data available.")
            return msg, msg, msg, msg

        if group_col not in catalogue.columns:
            msg = html.Div(f"Column '{group_col}' not found in trade catalogue.")
            return msg, msg, msg, msg

        groups = sorted(catalogue[group_col].dropna().unique().tolist())
        if selected_values:
            groups = [g for g in groups if g in selected_values]

        group_preds_dict = {}
        group_targets_dict = {}
        group_residuals = {}
        table_rows = []
        for grp in groups:
            mask = catalogue[group_col] == grp
            col_idx = np.where(mask.values)[0]
            if len(col_idx) == 0:
                continue
            p = store.predictions[:, col_idx].sum(axis=1)
            t = store.targets[:, col_idx].sum(axis=1)
            group_preds_dict[grp] = p
            group_targets_dict[grp] = t
            group_residuals[grp] = p - t
            table_rows.append({
                "group": grp,
                "n_trades": int(len(col_idx)),
                "mae": float(np.mean(np.abs(p - t))),
                "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            })

        ts_fig = dcc.Graph(
            figure=overlaid_group_timeseries(
                group_preds_dict,
                title=f"PnL by {group_col.replace('_', ' ').title()} — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )
        box_fig = dcc.Graph(
            figure=violin_overlay(
                group_residuals,
                title=f"Residual Distribution by {group_col.replace('_', ' ').title()}",
            ),
            config={"displayModeBar": False},
        )

        # Scatter grid: small multiples (G6)
        n_groups = len(group_preds_dict)
        scatter_grid = html.Div()
        if n_groups > 0:
            import plotly.graph_objects as go
            ncols = min(n_groups, 4)
            nrows = (n_groups + ncols - 1) // ncols
            fig = make_subplots(rows=nrows, cols=ncols,
                                subplot_titles=list(group_preds_dict.keys()))
            for i, (grp, p) in enumerate(group_preds_dict.items()):
                t = group_targets_dict[grp]
                r, c = i // ncols + 1, i % ncols + 1
                fig.add_trace(go.Scattergl(
                    x=t, y=p, mode="markers",
                    marker=dict(size=2, color=CHART_COLORS[i % len(CHART_COLORS)], opacity=0.5),
                    showlegend=False,
                ), row=r, col=c)
                vmin, vmax = min(t.min(), p.min()), max(t.max(), p.max())
                fig.add_trace(go.Scattergl(
                    x=[vmin, vmax], y=[vmin, vmax], mode="lines",
                    line=dict(color=TEXT_SECONDARY, dash="dash", width=1),
                    showlegend=False,
                ), row=r, col=c)
            fig.update_layout(height=280 * nrows, title="Pred vs Target — Small Multiples")
            scatter_grid = dcc.Graph(figure=fig, config={"displayModeBar": False})

        col_defs = [
            {"field": "group", "headerName": group_col.replace("_", " ").title()},
            {"field": "n_trades", "headerName": "# Trades"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "rmse", "headerName": "RMSE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        table = metric_table(col_defs, table_rows, f"{id_prefix}-metrics-table", height="300px")

        return ts_fig, box_fig, scatter_grid, table

    # ── By Desk ───────────────────────────────────────────────────
    @app.callback(
        Output("eval-desk-filter-bar", "children"),
        Input("eval-sub-tabs", "value"),
    )
    def render_desk_filter(sub_tab):
        if sub_tab != EVAL_SUB_DESK:
            return no_update
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        cat = get_trade_catalogue()
        return filter_bar(cat, "eval-desk", columns=["desk"])

    @app.callback(
        Output("eval-desk-timeseries", "children"),
        Output("eval-desk-boxplot", "children"),
        Output("eval-desk-scatter-grid", "children"),
        Output("eval-desk-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-desk-filter-desk", "value"),
    )
    def update_desk(split, sub_tab, selected_desks):
        if sub_tab != EVAL_SUB_DESK:
            return no_update, no_update, no_update, no_update
        return _build_group_view(split, "desk", selected_desks, "eval-desk")

    # ── By Product ────────────────────────────────────────────────
    @app.callback(
        Output("eval-product-filter-bar", "children"),
        Input("eval-sub-tabs", "value"),
    )
    def render_product_filter(sub_tab):
        if sub_tab != EVAL_SUB_PRODUCT:
            return no_update
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        cat = get_trade_catalogue()
        return filter_bar(cat, "eval-product", columns=["product_type"])

    @app.callback(
        Output("eval-product-timeseries", "children"),
        Output("eval-product-boxplot", "children"),
        Output("eval-product-scatter-grid", "children"),
        Output("eval-product-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-product-filter-product_type", "value"),
    )
    def update_product(split, sub_tab, selected_products):
        if sub_tab != EVAL_SUB_PRODUCT:
            return no_update, no_update, no_update, no_update
        return _build_group_view(split, "product_type", selected_products, "eval-product")

    # ── By CCY ────────────────────────────────────────────────────
    @app.callback(
        Output("eval-ccy-filter-bar", "children"),
        Input("eval-sub-tabs", "value"),
    )
    def render_ccy_filter(sub_tab):
        if sub_tab != EVAL_SUB_CCY:
            return no_update
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        cat = get_trade_catalogue()
        return filter_bar(cat, "eval-ccy", columns=["ccy"])

    @app.callback(
        Output("eval-ccy-timeseries", "children"),
        Output("eval-ccy-boxplot", "children"),
        Output("eval-ccy-scatter-grid", "children"),
        Output("eval-ccy-correlation", "children"),
        Output("eval-ccy-table-container", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-ccy-filter-ccy", "value"),
    )
    def update_ccy(split, sub_tab, selected_ccys):
        if sub_tab != EVAL_SUB_CCY:
            return no_update, no_update, no_update, no_update, no_update

        ts_fig, box_fig, scatter_grid, table = _build_group_view(split, "ccy", selected_ccys, "eval-ccy")

        # Cross-currency residual correlation heatmap
        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY

        store = get_prediction_store(split)
        catalogue = get_trade_catalogue()
        corr_fig = html.Div("Insufficient data for correlation.")

        if store is not None and "ccy" in catalogue.columns:
            ccys = sorted(catalogue["ccy"].dropna().unique().tolist())
            if selected_ccys:
                ccys = [c for c in ccys if c in selected_ccys]

            residual_matrix = {}
            for c in ccys:
                mask = catalogue["ccy"] == c
                idx = np.where(mask.values)[0]
                if len(idx) > 0:
                    p = store.predictions[:, idx].sum(axis=1)
                    t = store.targets[:, idx].sum(axis=1)
                    residual_matrix[c] = p - t

            if len(residual_matrix) > 1:
                import pandas as pd
                df = pd.DataFrame(residual_matrix)
                corr = df.corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0,
                    text=np.round(corr.values, 2).astype(str),
                    texttemplate="%{text}",
                ))
                fig.update_layout(title="Cross-CCY Residual Correlation", height=400)
                corr_fig = dcc.Graph(figure=fig, config={"displayModeBar": False})

        return ts_fig, box_fig, scatter_grid, corr_fig, table

    # ── By Cluster ────────────────────────────────────────────────
    @app.callback(
        Output("eval-cluster-selector-container", "children"),
        Input("eval-sub-tabs", "value"),
    )
    def render_cluster_selector(sub_tab):
        if sub_tab != EVAL_SUB_CLUSTER:
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        return cluster_selector(
            session.config.cluster_ids,
            session.cluster_attributes,
            id_prefix="eval-cluster",
        )

    @app.callback(
        Output("eval-cluster-heatmap", "children"),
        Output("eval-cluster-scatter", "children"),
        Output("eval-cluster-violin", "children"),
        Output("eval-cluster-trade-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-cluster-cluster-dropdown", "value"),
    )
    def update_by_cluster(split, sub_tab, cluster_id):
        if sub_tab != EVAL_SUB_CLUSTER or not cluster_id:
            return no_update, no_update, no_update, no_update

        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue

        store = get_prediction_store(split)
        catalogue = get_trade_catalogue()
        if store is None:
            msg = html.Div("No prediction data.")
            return msg, msg, msg, msg

        mask = catalogue["cluster_id"] == cluster_id
        col_idx = np.where(mask.values)[0]
        if len(col_idx) == 0:
            msg = html.Div(f"No trades found for cluster {cluster_id}.")
            return msg, msg, msg, msg

        preds = store.predictions[:, col_idx]
        targets = store.targets[:, col_idx]
        trade_ids = [store.trade_ids[i] for i in col_idx]

        # Per-Trade PnL Heatmap (G7): rows=scenarios, cols=trades, colour=residual
        residuals = preds - targets
        max_scenarios = 500
        heatmap_data = residuals[:max_scenarios] if residuals.shape[0] > max_scenarios else residuals
        hm_fig = go.Figure(go.Heatmap(
            z=heatmap_data, x=trade_ids,
            y=list(range(heatmap_data.shape[0])),
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="Residual"),
        ))
        hm_fig.update_layout(
            title=f"Per-Trade Residual Heatmap — {cluster_id}",
            xaxis_title="Trade", yaxis_title="Scenario",
            height=max(300, min(600, 2 * heatmap_data.shape[0])),
        )
        heatmap = dcc.Graph(figure=hm_fig, config={"displayModeBar": False})

        # Scatter: sum across trades in this cluster
        cluster_pred = preds.sum(axis=1)
        cluster_target = targets.sum(axis=1)
        scatter = dcc.Graph(
            figure=pred_vs_target_scatter(
                cluster_pred, cluster_target,
                title=f"Cluster {cluster_id} — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )

        # Violin: per-trade residuals
        trade_residuals = {}
        for j, tid in enumerate(trade_ids):
            trade_residuals[tid] = preds[:, j] - targets[:, j]
        violin = dcc.Graph(
            figure=violin_overlay(trade_residuals, title="Per-Trade Residual Distribution"),
            config={"displayModeBar": False},
        )

        # Trade-level metrics table
        col_defs = [
            {"field": "trade_id", "headerName": "Trade ID"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "rmse", "headerName": "RMSE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "max_ae", "headerName": "Max AE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        rows = []
        for j, tid in enumerate(trade_ids):
            r = preds[:, j] - targets[:, j]
            rows.append({
                "trade_id": tid,
                "mae": float(np.mean(np.abs(r))),
                "rmse": float(np.sqrt(np.mean(r ** 2))),
                "max_ae": float(np.max(np.abs(r))),
            })
        table = metric_table(col_defs, rows, "eval-cluster-trade-metrics-table", height="350px")

        return heatmap, scatter, violin, table
```

---

## 11. Tab 3 — Cluster Deep Dive

### Purpose

Forensic single-cluster analysis.  Full model transparency for a
selected cluster — split comparison, training convergence context,
per-trade scatter, elementary PnL explorer, and data config summary.

### Data sources

- `session.load_cluster_display(cid)` — `ClusterDisplayState`
- `session.load_cluster_predictions(cid, split)` — per-member predictions
- `session.ensemble_display.per_member_metrics[split][cid]` — metrics
- `session.cluster_attributes[cid]` — cluster attributes

### Layout

1. **Cluster selector** (dropdown with attributes)
2. **Split toggle** (radio + "All" for comparison view)
3. **Header card** — cluster ID, version, n_trades, attributes
4. **Split comparison table** — MAE/RMSE per split side-by-side
5. **Per-trade scatter** — predictions vs targets for the selected split
6. **Residual distribution** — histogram of per-trade residuals
7. **Data config summary** — key training parameters

---

## 12. Tab 4 — Market Data (4 sub-tabs)

### Purpose

Transparency into risk-factor shocks / curves / surfaces that drive
model inputs.  Sanity check of training scenario data.

### Data sources

- `session.load_cluster_market_data(cid)` — `{asset: {rf: np.ndarray}}`
- `session.config.cluster_ids` — cluster list

### Sub-tabs

| Sub-tab | Content |
|---------|---------|
| RF Summary | Portfolio-wide RF inventory, count per cluster |
| Shock Explorer | Per-cluster/asset/RF shock time-series + distribution |
| Scenario Heatmap | RF x scenario heatmap for selected cluster |
| Distribution | Cross-cluster RF distribution comparison |

---

### 16.41 `src/ui/apps/ensemble_analytics/tabs/cluster_deep_dive.py`

```python
"""
Tab 3 — Cluster Deep Dive layout.

Forensic single-cluster view with cluster selector, split comparison,
per-trade scatter, residual distribution, and config summary.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE
from src.ui.apps.ensemble_analytics.components.split_toggle import split_toggle


def layout() -> html.Div:
    """Build the Cluster Deep Dive tab skeleton."""
    return html.Div([
        html.Div(id="deep-dive-cluster-selector"),
        split_toggle(id_prefix="deep-dive", default="test"),

        html.Div(id="deep-dive-header", style=CARD_STYLE),

        html.Div("Split Comparison", style=SECTION_TITLE_STYLE),
        html.Div(id="deep-dive-split-table", style=CARD_STYLE),

        dbc.Row([
            dbc.Col(html.Div(id="deep-dive-convergence", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="deep-dive-scatter", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(html.Div(id="deep-dive-residual", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="deep-dive-scatter-matrix", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),

        html.Div("Elementary PnL Explorer", style=SECTION_TITLE_STYLE),
        html.Div(id="deep-dive-elementary", style=CARD_STYLE),

        html.Div("Configuration Summary", style=SECTION_TITLE_STYLE),
        html.Div(id="deep-dive-config", style=CARD_STYLE),
    ])
```

---

### 16.42 `src/ui/apps/ensemble_analytics/callbacks/cluster_deep_dive_cb.py`

```python
"""
Callbacks for Tab 3 — Cluster Deep Dive.

Populates the cluster selector on mount, then rebuilds all content
when the cluster or split changes.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import METRIC_DISPLAY_NAMES
from src.ui.apps.ensemble_analytics.components.cluster_selector import cluster_selector
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.distributions import residual_histogram
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY
from src.ui.apps.ensemble_analytics.theme.styles import CARD_HEADER_STYLE


def register(app):
    """Register Cluster Deep Dive callbacks on *app*."""

    @app.callback(
        Output("deep-dive-cluster-selector", "children"),
        Input("main-tabs", "value"),
    )
    def render_selector(tab):
        if tab != "tab-cluster-deep-dive":
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        return cluster_selector(
            session.config.cluster_ids,
            session.cluster_attributes,
            id_prefix="deep-dive",
        )

    @app.callback(
        Output("deep-dive-header", "children"),
        Output("deep-dive-split-table", "children"),
        Output("deep-dive-convergence", "children"),
        Output("deep-dive-scatter", "children"),
        Output("deep-dive-residual", "children"),
        Output("deep-dive-scatter-matrix", "children"),
        Output("deep-dive-elementary", "children"),
        Output("deep-dive-config", "children"),
        Input("deep-dive-cluster-dropdown", "value"),
        Input("deep-dive-split-toggle", "value"),
    )
    def update_deep_dive(cluster_id, split):
        n_out = 8
        if not cluster_id:
            return (no_update,) * n_out

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_AMBER, CHART_COLORS
        from src.ui.apps.ensemble_analytics.figures.bar_charts import grouped_split_bar

        session = get_session()
        display = session.load_cluster_display(cluster_id)
        attrs = session.cluster_attributes.get(cluster_id, {})
        tu = display.trade_universe
        dc = display.eval_metrics.get("data_config", {})

        # ── Header card (G8-enriched) ────────────────────────────
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if v)
        n_elem = len(tu.get("elementary_ids", []))
        n_target = len(tu.get("target_ids", session.config.cluster_mapping.get(cluster_id, [])))
        header = html.Div([
            html.Div(f"Cluster: {cluster_id}", style={"fontSize": "18px", "fontWeight": "700"}),
            html.Div(f"Version: {display.version}", style={"color": TEXT_SECONDARY, "fontSize": "13px"}),
            html.Div(attr_str, style={"color": TEXT_SECONDARY, "fontSize": "13px"}) if attr_str else None,
            html.Div(
                f"n_elementary: {n_elem} · n_target: {n_target} · "
                f"seq_length: {dc.get('seq_length', '?')} · "
                f"transform: {dc.get('transform_type', '?')}",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            ),
        ])

        # ── Split comparison table + grouped bar ─────────────────
        ens_display = session.ensemble_display
        split_rows = []
        if ens_display:
            for s in ["train", "val", "test"]:
                pm = ens_display.per_member_metrics.get(s, {}).get(cluster_id, {})
                if pm:
                    row = {"split": s.capitalize()}
                    row.update(pm)
                    split_rows.append(row)

        split_col_defs = [{"field": "split", "headerName": "Split", "pinned": "left"}]
        if split_rows:
            for key in split_rows[0]:
                if key != "split":
                    split_col_defs.append({
                        "field": key,
                        "headerName": METRIC_DISPLAY_NAMES.get(key, key.upper()),
                        "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
                    })
        split_table = metric_table(split_col_defs, split_rows, "deep-dive-split-comp-table", height="180px")

        # ── Training convergence (G8): load from saved plot PNGs ─
        convergence_content = html.Div("No convergence plot available.", style={"color": TEXT_SECONDARY})
        for plot_key, path_str in display.plot_paths.items():
            if "loss" in plot_key.lower() or "convergence" in plot_key.lower():
                from pathlib import Path
                if Path(path_str).exists():
                    import base64
                    with open(path_str, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode()
                    convergence_content = html.Img(
                        src=f"data:image/png;base64,{encoded}",
                        style={"width": "100%", "maxHeight": "400px", "objectFit": "contain"},
                    )
                break

        # ── Scatter + residual ───────────────────────────────────
        store = get_prediction_store(split)
        catalogue = get_trade_catalogue()
        scatter_content = html.Div("No predictions available.")
        residual_content = html.Div("No predictions available.")
        scatter_matrix_content = html.Div("No predictions available.")
        elementary_content = html.Div("Elementary PnL data not available.", style={"color": TEXT_SECONDARY})

        if store is not None and not catalogue.empty:
            mask = catalogue["cluster_id"] == cluster_id
            col_idx = np.where(mask.values)[0]
            if len(col_idx) > 0:
                preds = store.predictions[:, col_idx]
                targets = store.targets[:, col_idx]
                trade_ids = [store.trade_ids[i] for i in col_idx]
                cluster_pred = preds.sum(axis=1)
                cluster_target = targets.sum(axis=1)

                scatter_content = dcc.Graph(
                    figure=pred_vs_target_scatter(cluster_pred, cluster_target,
                                                  title=f"{cluster_id} — {split.capitalize()}"),
                    config={"displayModeBar": False},
                )
                residual_content = dcc.Graph(
                    figure=residual_histogram(cluster_pred, cluster_target,
                                              title=f"Residuals — {split.capitalize()}"),
                    config={"displayModeBar": False},
                )

                # Per-Trade Scatter Matrix (G9): up to 6 trades
                n_show = min(6, len(trade_ids))
                if n_show > 0:
                    ncols = min(n_show, 3)
                    nrows = (n_show + ncols - 1) // ncols
                    fig = make_subplots(rows=nrows, cols=ncols,
                                        subplot_titles=trade_ids[:n_show])
                    for j in range(n_show):
                        r, c = j // ncols + 1, j % ncols + 1
                        fig.add_trace(go.Scattergl(
                            x=targets[:, j], y=preds[:, j], mode="markers",
                            marker=dict(size=2, color=CHART_COLORS[j % len(CHART_COLORS)], opacity=0.5),
                            showlegend=False,
                        ), row=r, col=c)
                    fig.update_layout(height=280 * nrows, title="Per-Trade Scatter (first 6)")
                    scatter_matrix_content = dcc.Graph(figure=fig, config={"displayModeBar": False})

        # ── Data config summary as structured table (G11) ────────
        if dc:
            config_rows = [{"parameter": k, "value": str(v)} for k, v in dc.items()]
            config_col_defs = [
                {"field": "parameter", "headerName": "Parameter"},
                {"field": "value", "headerName": "Value"},
            ]
            config_content = metric_table(config_col_defs, config_rows,
                                          "deep-dive-config-table", height="300px")
        else:
            config_content = html.Div("No data config available.", style={"color": TEXT_SECONDARY})

        return (header, split_table, convergence_content, scatter_content,
                residual_content, scatter_matrix_content, elementary_content,
                config_content)
```

---

### 16.43 `src/ui/apps/ensemble_analytics/tabs/market_data/__init__.py`

```python
"""
Tab 4 — Market Data sub-tab container.

Renders the sub-tab bar for RF Summary, Shock Explorer, Scenario
Heatmap, and Distribution.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import MD_SUB_ORDER, MD_SUB_RF_SUMMARY


def layout() -> html.Div:
    """Build the Market Data tab layout with sub-tab navigation."""
    sub_tabs = dcc.Tabs(
        id="md-sub-tabs",
        value=MD_SUB_RF_SUMMARY,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in MD_SUB_ORDER
        ],
    )
    return html.Div([
        html.Div(id="md-cluster-selector-container"),
        sub_tabs,
        html.Div(id="md-sub-tab-content", style={"marginTop": "16px"}),
    ])
```

---

### 16.44 `src/ui/apps/ensemble_analytics/tabs/market_data/rf_summary.py`

```python
"""
Market Data sub-tab: RF Summary.

Portfolio-wide risk-factor inventory — counts and coverage matrix
across clusters.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the RF Summary sub-tab skeleton."""
    return html.Div([
        html.Div("Risk Factor Inventory", style=SECTION_TITLE_STYLE),
        html.Div(id="md-rf-summary-table", style=CARD_STYLE),
        html.Div("Cluster × RF Coverage", style=SECTION_TITLE_STYLE),
        html.Div(id="md-rf-coverage-heatmap", style=CARD_STYLE),
    ])
```

---

### 16.45 `src/ui/apps/ensemble_analytics/tabs/market_data/shock_explorer.py`

```python
"""
Market Data sub-tab: Shock Explorer.

Drill into RF shock time-series and distributions for a selected
cluster and asset.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE


def layout() -> html.Div:
    """Build the Shock Explorer sub-tab skeleton."""
    return html.Div([
        html.Div(id="md-shock-asset-selector"),
        html.Div(id="md-shock-rf-selector"),
        dbc.Row([
            dbc.Col(html.Div(id="md-shock-timeseries", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="md-shock-distribution", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="md-shock-stats", style=CARD_STYLE),
    ])
```

---

### 16.46 `src/ui/apps/ensemble_analytics/tabs/market_data/scenario_heatmap.py`

```python
"""
Market Data sub-tab: Scenario Heatmap.

RF x scenario heatmap for a selected cluster — shows the full shock
surface.
"""
from __future__ import annotations

from dash import html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE


def layout() -> html.Div:
    """Build the Scenario Heatmap sub-tab skeleton."""
    return html.Div([
        html.Div(id="md-heatmap-container", style=CARD_STYLE),
    ])
```

---

### 16.47 `src/ui/apps/ensemble_analytics/tabs/market_data/distribution.py`

```python
"""
Market Data sub-tab: Distribution.

Cross-cluster RF distribution comparison — QQ plots and correlation
matrices.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Distribution sub-tab skeleton."""
    return html.Div([
        html.Div("Cross-Cluster RF Comparison", style=SECTION_TITLE_STYLE),
        dbc.Row([
            dbc.Col(html.Div(id="md-dist-violin", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="md-dist-qq", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="md-dist-corr-heatmap", style=CARD_STYLE),
    ])
```

---

### 16.48 `src/ui/apps/ensemble_analytics/callbacks/market_data_cb.py`

```python
"""
Callbacks for Tab 4 — Market Data (all 4 sub-tabs).

Handles sub-tab routing, cluster selector rendering, and per-sub-tab
data loading from ``load_cluster_market_data``.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    MD_SUB_RF_SUMMARY,
    MD_SUB_SHOCK_EXPLORER,
    MD_SUB_SCENARIO_HEATMAP,
    MD_SUB_DISTRIBUTION,
)
from src.ui.apps.ensemble_analytics.components.cluster_selector import cluster_selector
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.heatmaps import rf_scenario_heatmap
from src.ui.apps.ensemble_analytics.figures.distributions import violin_overlay, qq_plot
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def register(app):
    """Register Market Data tab callbacks on *app*."""

    # ── Cluster selector ──────────────────────────────────────────
    @app.callback(
        Output("md-cluster-selector-container", "children"),
        Input("main-tabs", "value"),
    )
    def render_md_cluster_selector(tab):
        if tab != "tab-market-data":
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        return cluster_selector(
            session.config.cluster_ids,
            session.cluster_attributes,
            id_prefix="md",
        )

    # ── Sub-tab routing ───────────────────────────────────────────
    @app.callback(
        Output("md-sub-tab-content", "children"),
        Input("md-sub-tabs", "value"),
    )
    def render_md_sub_tab(sub_tab):
        if sub_tab == MD_SUB_RF_SUMMARY:
            from src.ui.apps.ensemble_analytics.tabs.market_data.rf_summary import layout
            return layout()
        elif sub_tab == MD_SUB_SHOCK_EXPLORER:
            from src.ui.apps.ensemble_analytics.tabs.market_data.shock_explorer import layout
            return layout()
        elif sub_tab == MD_SUB_SCENARIO_HEATMAP:
            from src.ui.apps.ensemble_analytics.tabs.market_data.scenario_heatmap import layout
            return layout()
        elif sub_tab == MD_SUB_DISTRIBUTION:
            from src.ui.apps.ensemble_analytics.tabs.market_data.distribution import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    # ── RF Summary ────────────────────────────────────────────────
    @app.callback(
        Output("md-rf-summary-table", "children"),
        Output("md-rf-coverage-heatmap", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_rf_summary(cluster_id, sub_tab):
        if sub_tab != MD_SUB_RF_SUMMARY or not cluster_id:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session

        session = get_session()
        all_rfs = {}

        for cid in session.config.cluster_ids:
            mdata = get_market_data(cid)
            for asset, rfs in mdata.items():
                for rf_name in rfs:
                    all_rfs.setdefault(rf_name, set()).add(cid)

        col_defs = [
            {"field": "rf_name", "headerName": "Risk Factor"},
            {"field": "n_clusters", "headerName": "# Clusters"},
            {"field": "clusters", "headerName": "Clusters"},
        ]
        rows = [
            {
                "rf_name": rf,
                "n_clusters": len(cids),
                "clusters": ", ".join(sorted(cids)),
            }
            for rf, cids in sorted(all_rfs.items())
        ]
        table = metric_table(col_defs, rows, "md-rf-inventory-table", height="400px")

        import plotly.graph_objects as go
        cluster_ids = session.config.cluster_ids
        rf_names = sorted(all_rfs.keys())
        coverage = np.zeros((len(rf_names), len(cluster_ids)))
        for i, rf in enumerate(rf_names):
            for j, cid in enumerate(cluster_ids):
                if cid in all_rfs.get(rf, set()):
                    coverage[i, j] = 1.0

        fig = go.Figure(go.Heatmap(
            z=coverage, x=cluster_ids, y=rf_names,
            colorscale=[[0, "#161b22"], [1, "#58a6ff"]],
            showscale=False,
        ))
        fig.update_layout(title="RF Coverage Matrix", height=max(300, 18 * len(rf_names) + 100))
        heatmap = dcc.Graph(figure=fig, config={"displayModeBar": False})

        return table, heatmap

    # ── Shock Explorer ────────────────────────────────────────────
    @app.callback(
        Output("md-shock-asset-selector", "children"),
        Output("md-shock-rf-selector", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def render_shock_selectors(cluster_id, sub_tab):
        if sub_tab != MD_SUB_SHOCK_EXPLORER or not cluster_id:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        asset_names = sorted(mdata.keys())
        all_rfs = sorted({rf for rfs in mdata.values() for rf in rfs})

        asset_dd = html.Div([
            html.Label("Asset:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Dropdown(
                id="md-shock-asset-dd",
                options=[{"label": a, "value": a} for a in asset_names],
                value=asset_names[0] if asset_names else None,
                clearable=False, style={"width": "300px", "fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})

        rf_dd = html.Div([
            html.Label("RF:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Dropdown(
                id="md-shock-rf-dd",
                options=[{"label": r, "value": r} for r in all_rfs],
                value=all_rfs[0] if all_rfs else None,
                clearable=False, style={"width": "300px", "fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"})

        return asset_dd, rf_dd

    @app.callback(
        Output("md-shock-timeseries", "children"),
        Output("md-shock-distribution", "children"),
        Output("md-shock-stats", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-shock-asset-dd", "value"),
        Input("md-shock-rf-dd", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_shock_explorer(cluster_id, asset, rf, sub_tab):
        if sub_tab != MD_SUB_SHOCK_EXPLORER or not all([cluster_id, asset, rf]):
            return no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        shocks = mdata.get(asset, {}).get(rf)
        if shocks is None:
            msg = html.Div("No shock data for this asset/RF combination.")
            return msg, msg, msg

        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE

        ts = go.Figure(go.Scattergl(
            x=np.arange(len(shocks)), y=shocks, mode="lines",
            line=dict(color=ACCENT_BLUE, width=1.5),
        ))
        ts.update_layout(title=f"{asset} — {rf} Shocks", xaxis_title="Scenario", yaxis_title="Shock", height=350)

        hist = go.Figure(go.Histogram(x=shocks, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8))
        hist.update_layout(title=f"{rf} Shock Distribution", height=350)

        stats_content = html.Div([
            html.Span(f"Mean: {shocks.mean():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Std: {shocks.std():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Min: {shocks.min():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Max: {shocks.max():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"N: {len(shocks)}", style={"fontSize": "13px"}),
        ])

        return (
            dcc.Graph(figure=ts, config={"displayModeBar": False}),
            dcc.Graph(figure=hist, config={"displayModeBar": False}),
            stats_content,
        )

    # ── Scenario Heatmap ──────────────────────────────────────────
    @app.callback(
        Output("md-heatmap-container", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_scenario_heatmap(cluster_id, sub_tab):
        if sub_tab != MD_SUB_SCENARIO_HEATMAP or not cluster_id:
            return no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        if not mdata:
            return html.Div("No market data available.")

        all_rfs = {}
        for asset, rfs in mdata.items():
            for rf_name, arr in rfs.items():
                all_rfs[f"{asset}/{rf_name}"] = arr

        if not all_rfs:
            return html.Div("No RF shocks found.")

        rf_names = sorted(all_rfs.keys())
        n_scenarios = min(arr.shape[0] for arr in all_rfs.values())
        matrix = np.column_stack([all_rfs[rf][:n_scenarios] for rf in rf_names])

        fig = rf_scenario_heatmap(rf_names, matrix, title=f"Cluster {cluster_id} — RF × Scenario")
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    # ── Distribution ──────────────────────────────────────────────
    @app.callback(
        Output("md-dist-violin", "children"),
        Output("md-dist-qq", "children"),
        Output("md-dist-corr-heatmap", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_distribution(cluster_id, sub_tab):
        if sub_tab != MD_SUB_DISTRIBUTION or not cluster_id:
            return no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        if not mdata:
            msg = html.Div("No market data available.")
            return msg, msg, msg

        rf_arrays = {}
        for asset, rfs in mdata.items():
            for rf_name, arr in rfs.items():
                rf_arrays[f"{asset}/{rf_name}"] = arr

        if not rf_arrays:
            msg = html.Div("No RF shocks found.")
            return msg, msg, msg

        # Violin of first 10 RFs
        subset = dict(list(rf_arrays.items())[:10])
        violin_fig = dcc.Graph(
            figure=violin_overlay(subset, title="RF Shock Distributions"),
            config={"displayModeBar": False},
        )

        # QQ plot of the first RF
        first_rf = next(iter(rf_arrays.values()))
        qq_fig = dcc.Graph(
            figure=qq_plot(first_rf, title=f"QQ — {next(iter(rf_arrays.keys()))}"),
            config={"displayModeBar": False},
        )

        # Correlation heatmap
        import plotly.graph_objects as go
        import pandas as pd
        df = pd.DataFrame({k: v[:min(len(v) for v in rf_arrays.values())]
                           for k, v in list(rf_arrays.items())[:20]})
        corr = df.corr()
        heatmap_fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0,
        ))
        heatmap_fig.update_layout(title="RF Correlation Matrix", height=500)
        corr_content = dcc.Graph(figure=heatmap_fig, config={"displayModeBar": False})

        return violin_fig, qq_fig, corr_content
```

---

## 13. Tab 5 — Trade Graph Explorer (4 sub-tabs)

### Purpose

Visualise the GNN adjacency graph built during training — which trades
influence which, graph statistics, and cross-cluster structure comparison.

### Data sources

- `session.load_cluster_graph_data(cid)` — `graph_results`, `encoder_results`, `trade_universe`

### Sub-tabs

| Sub-tab | Content |
|---------|---------|
| Graph View | Interactive cytoscape network, node detail on click |
| Adjacency Analysis | Sparsity stats, edge weight histogram, degree distribution, spy plot |
| Node Analytics | Degree vs MAE, embedding (future), node table |
| Cross-Cluster | Summary stats table, density comparison |

---

## 14. Tab 6 — Inference

### Purpose

Run new scenarios through the ensemble model.  The **only tab** that
triggers Phase 3 (model loading).  Shows a progress bar during loading,
then allows scenario upload and displays results.

### Data sources

- `session.load_inference_state(parallel=True)` — Phase 3 trigger
- `session.run_inference(mode, cluster_pnl_histories)` — forward pass
- `session.inference_ready_clusters`, `session.all_inference_ready` — progress

### Layout

1. **Load Models button** + progress bar
2. **Mode selector** (New Scenarios / New Trades — disabled)
3. **Scenario directory input** (path to new shock CSVs)
4. **Run Inference button**
5. **Results:** portfolio histogram, per-cluster summary table, export

---

### 16.49 `src/ui/apps/ensemble_analytics/tabs/trade_graph/__init__.py`

```python
"""
Tab 5 — Trade Graph Explorer sub-tab container.

Renders the sub-tab bar for Graph View, Adjacency Analysis,
Node Analytics, and Cross-Cluster.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import TG_SUB_ORDER, TG_SUB_GRAPH_VIEW


def layout() -> html.Div:
    """Build the Trade Graph tab layout with sub-tab navigation."""
    sub_tabs = dcc.Tabs(
        id="tg-sub-tabs",
        value=TG_SUB_GRAPH_VIEW,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in TG_SUB_ORDER
        ],
    )
    return html.Div([
        html.Div(id="tg-cluster-selector-container"),
        sub_tabs,
        html.Div(id="tg-sub-tab-content", style={"marginTop": "16px"}),
    ])
```

---

### 16.50 `src/ui/apps/ensemble_analytics/tabs/trade_graph/graph_view.py`

```python
"""
Trade Graph sub-tab: Graph View.

Interactive cytoscape network visualisation of the trade adjacency
graph for a selected cluster.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE


def layout() -> html.Div:
    """Build the Graph View sub-tab skeleton with interactive controls."""
    from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY

    controls = html.Div([
        html.Div([
            html.Label("Layout:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.RadioItems(
                id="tg-layout-selector",
                options=[
                    {"label": "Force-directed", "value": "cose"},
                    {"label": "Circular", "value": "circle"},
                    {"label": "Grid", "value": "grid"},
                ],
                value="cose", inline=True,
                labelStyle={"marginRight": "12px", "fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginRight": "24px"}),
        html.Div([
            html.Label("Edge threshold:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Slider(id="tg-weight-threshold", min=0, max=1, step=0.01, value=0.01,
                       marks={0: "0", 0.25: "0.25", 0.5: "0.5", 1: "1"},
                       tooltip={"placement": "bottom"}),
        ], style={"width": "300px"}),
        html.Div([
            html.Label("Search:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Input(id="tg-search-box", type="text", placeholder="trade_id...",
                      style={"width": "200px", "fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center",
              "marginBottom": "12px", "gap": "16px"})

    return html.Div([
        controls,
        html.Div(id="tg-graph-container", style={**CARD_STYLE, "height": "600px"}),
        html.Div(id="tg-node-detail", style=CARD_STYLE),
    ])
```

---

### 16.51 `src/ui/apps/ensemble_analytics/tabs/trade_graph/adjacency_analysis.py`

```python
"""
Trade Graph sub-tab: Adjacency Analysis.

Graph statistics, edge weight histogram, degree distribution, and
adjacency spy plot.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Adjacency Analysis sub-tab skeleton."""
    return html.Div([
        html.Div(id="tg-adj-stats", style=CARD_STYLE),
        dbc.Row([
            dbc.Col(html.Div(id="tg-adj-weight-hist", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="tg-adj-degree-dist", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div("Adjacency Spy Plot", style=SECTION_TITLE_STYLE),
        html.Div(id="tg-adj-spy", style=CARD_STYLE),
    ])
```

---

### 16.52 `src/ui/apps/ensemble_analytics/tabs/trade_graph/node_analytics.py`

```python
"""
Trade Graph sub-tab: Node Analytics.

Degree vs model error, node feature table.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Node Analytics sub-tab skeleton."""
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div(id="tg-node-degree-scatter", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="tg-node-feature-table", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
    ])
```

---

### 16.53 `src/ui/apps/ensemble_analytics/tabs/trade_graph/cross_cluster.py`

```python
"""
Trade Graph sub-tab: Cross-Cluster Comparison.

Compare graph structure (density, degree stats) across all clusters.
"""
from __future__ import annotations

from dash import html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Cross-Cluster sub-tab skeleton."""
    return html.Div([
        html.Div("Graph Structure Comparison", style=SECTION_TITLE_STYLE),
        html.Div(id="tg-cross-cluster-table", style=CARD_STYLE),
        html.Div(id="tg-cross-cluster-chart", style=CARD_STYLE),
    ])
```

---

### 16.54 `src/ui/apps/ensemble_analytics/callbacks/trade_graph_cb.py`

```python
"""
Callbacks for Tab 5 — Trade Graph Explorer (all 4 sub-tabs).

Handles sub-tab routing, cluster selector, cytoscape rendering,
adjacency analysis, node analytics, and cross-cluster comparison.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, dcc, html, no_update
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from src.ui.apps.ensemble_analytics.config import (
    TG_SUB_GRAPH_VIEW,
    TG_SUB_ADJACENCY,
    TG_SUB_NODE_ANALYTICS,
    TG_SUB_CROSS_CLUSTER,
)
from src.ui.apps.ensemble_analytics.components.cluster_selector import cluster_selector
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.network import build_cytoscape_elements
from src.ui.apps.ensemble_analytics.figures.heatmaps import adjacency_spy
from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY


def register(app):
    """Register Trade Graph tab callbacks on *app*."""

    # ── Cluster selector ──────────────────────────────────────────
    @app.callback(
        Output("tg-cluster-selector-container", "children"),
        Input("main-tabs", "value"),
    )
    def render_tg_selector(tab):
        if tab != "tab-trade-graph":
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        return cluster_selector(
            session.config.cluster_ids,
            session.cluster_attributes,
            id_prefix="tg",
        )

    # ── Sub-tab routing ───────────────────────────────────────────
    @app.callback(
        Output("tg-sub-tab-content", "children"),
        Input("tg-sub-tabs", "value"),
    )
    def render_tg_sub_tab(sub_tab):
        if sub_tab == TG_SUB_GRAPH_VIEW:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.graph_view import layout
            return layout()
        elif sub_tab == TG_SUB_ADJACENCY:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.adjacency_analysis import layout
            return layout()
        elif sub_tab == TG_SUB_NODE_ANALYTICS:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.node_analytics import layout
            return layout()
        elif sub_tab == TG_SUB_CROSS_CLUSTER:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.cross_cluster import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    # ── Graph View (G12: interactive controls + detail panel) ─────
    @app.callback(
        Output("tg-graph-container", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
        Input("tg-layout-selector", "value"),
        Input("tg-weight-threshold", "value"),
    )
    def update_graph_view(cluster_id, sub_tab, layout_name, threshold):
        if sub_tab != TG_SUB_GRAPH_VIEW or not cluster_id:
            return no_update

        from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data

        gdata = get_graph_data(cluster_id)
        graph_results = gdata.get("graph_results", {})
        trade_universe = gdata.get("trade_universe", {})

        indices = graph_results.get("sparse_indices")
        values = graph_results.get("sparse_values")
        all_ids = trade_universe.get("elementary_ids", []) + trade_universe.get("target_ids", [])
        target_set = set(trade_universe.get("target_ids", []))

        if indices is None or values is None or not all_ids:
            return html.Div("No graph data available.", style={"color": TEXT_SECONDARY})

        indices = np.array(indices)
        values = np.array(values)

        node_attrs = {}
        for tid in all_ids:
            node_attrs[tid] = {"trade_type": "target" if tid in target_set else "elementary"}

        elements = build_cytoscape_elements(
            all_ids, indices, values,
            node_attrs=node_attrs,
            weight_threshold=threshold or 0.01,
        )

        return cyto.Cytoscape(
            id="tg-cytoscape",
            elements=elements,
            layout={"name": layout_name or "cose", "animate": False},
            style={"width": "100%", "height": "550px", "backgroundColor": BG_CARD},
            stylesheet=[
                {
                    "selector": "node[trade_type='elementary']",
                    "style": {
                        "label": "data(label)", "font-size": "9px",
                        "color": TEXT_PRIMARY, "background-color": ACCENT_BLUE,
                        "width": 16, "height": 16,
                    },
                },
                {
                    "selector": "node[trade_type='target']",
                    "style": {
                        "label": "data(label)", "font-size": "9px",
                        "color": TEXT_PRIMARY, "background-color": "#d29922",
                        "width": 22, "height": 22,
                    },
                },
                {
                    "selector": "edge",
                    "style": {"width": 1, "line-color": "#30363d", "opacity": 0.6},
                },
                {
                    "selector": ":selected",
                    "style": {"background-color": "#f85149", "border-width": 2, "border-color": "#fff"},
                },
            ],
        )

    @app.callback(
        Output("tg-node-detail", "children"),
        Input("tg-cytoscape", "tapNodeData"),
    )
    def show_node_detail(node_data):
        """Display details for a clicked node (G12)."""
        if not node_data:
            return html.Div("Click a node to see details.", style={"color": TEXT_SECONDARY, "fontSize": "13px"})

        tid = node_data.get("id", "?")
        trade_type = node_data.get("trade_type", "unknown")
        items = [
            html.Div(f"Trade: {tid}", style={"fontSize": "15px", "fontWeight": "600"}),
            html.Div(f"Type: {trade_type}", style={"color": TEXT_SECONDARY, "fontSize": "13px"}),
        ]
        for k, v in node_data.items():
            if k not in ("id", "label", "trade_type"):
                items.append(html.Div(f"{k}: {v}", style={"color": TEXT_SECONDARY, "fontSize": "13px"}))
        return html.Div(items)

    # ── Adjacency Analysis ────────────────────────────────────────
    @app.callback(
        Output("tg-adj-stats", "children"),
        Output("tg-adj-weight-hist", "children"),
        Output("tg-adj-degree-dist", "children"),
        Output("tg-adj-spy", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
    )
    def update_adjacency(cluster_id, sub_tab):
        if sub_tab != TG_SUB_ADJACENCY or not cluster_id:
            return no_update, no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data
        import plotly.graph_objects as go

        gdata = get_graph_data(cluster_id)
        gr = gdata.get("graph_results", {})
        indices = gr.get("sparse_indices")
        values = gr.get("sparse_values")
        shape = gr.get("sparse_shape", [0, 0])

        if indices is None or values is None:
            msg = html.Div("No adjacency data.", style={"color": TEXT_SECONDARY})
            return msg, msg, msg, msg

        indices = np.array(indices)
        values = np.array(values)
        n_nodes = shape[0] if shape[0] > 0 else max(indices.max() + 1, 1)
        nnz = len(values)
        density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0

        stats = html.Div([
            html.Span(f"Nodes: {n_nodes}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Edges: {nnz}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Density: {density:.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Mean weight: {values.mean():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Max weight: {values.max():.4f}", style={"fontSize": "13px"}),
        ])

        # Edge weight histogram
        weight_hist = go.Figure(go.Histogram(x=values, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8))
        weight_hist.update_layout(title="Edge Weight Distribution", height=350)

        # Degree distribution
        if indices.ndim == 2 and indices.shape[0] == 2:
            rows = indices[0]
        else:
            rows = indices[:, 0]
        degrees = np.bincount(rows.astype(int), minlength=n_nodes)
        degree_hist = go.Figure(go.Histogram(x=degrees, nbinsx=40, marker_color=ACCENT_BLUE, opacity=0.8))
        degree_hist.update_layout(title="Degree Distribution", height=350)

        # Spy plot
        spy_fig = adjacency_spy(indices, values, list(shape), title=f"Adjacency — Cluster {cluster_id}")

        return (
            stats,
            dcc.Graph(figure=weight_hist, config={"displayModeBar": False}),
            dcc.Graph(figure=degree_hist, config={"displayModeBar": False}),
            dcc.Graph(figure=spy_fig, config={"displayModeBar": False}),
        )

    # ── Node Analytics ────────────────────────────────────────────
    @app.callback(
        Output("tg-node-degree-scatter", "children"),
        Output("tg-node-feature-table", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
    )
    def update_node_analytics(cluster_id, sub_tab):
        if sub_tab != TG_SUB_NODE_ANALYTICS or not cluster_id:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        import plotly.graph_objects as go

        gdata = get_graph_data(cluster_id)
        gr = gdata.get("graph_results", {})
        tu = gdata.get("trade_universe", {})
        indices = gr.get("sparse_indices")
        shape = gr.get("sparse_shape", [0, 0])
        trade_ids = tu.get("target_ids", [])

        if indices is None or not trade_ids:
            msg = html.Div("Insufficient data.", style={"color": TEXT_SECONDARY})
            return msg, msg

        indices = np.array(indices)
        n_nodes = shape[0] if shape[0] > 0 else max(indices.max() + 1, 1)
        if indices.ndim == 2 and indices.shape[0] == 2:
            rows = indices[0]
        else:
            rows = indices[:, 0]
        degrees = np.bincount(rows.astype(int), minlength=n_nodes)

        # Try to get per-trade MAE from prediction store
        store = get_prediction_store("test")
        catalogue = get_trade_catalogue()
        trade_mae = np.full(len(trade_ids), np.nan)

        if store is not None and not catalogue.empty:
            mask = catalogue["cluster_id"] == cluster_id
            col_idx = np.where(mask.values)[0]
            if len(col_idx) > 0:
                preds = store.predictions[:, col_idx]
                targets = store.targets[:, col_idx]
                trade_mae = np.mean(np.abs(preds - targets), axis=0)

        # Degree vs MAE scatter
        n_plot = min(len(degrees), len(trade_mae), len(trade_ids))
        fig = go.Figure(go.Scattergl(
            x=degrees[:n_plot], y=trade_mae[:n_plot],
            mode="markers", marker=dict(size=5, color=ACCENT_BLUE, opacity=0.7),
            text=trade_ids[:n_plot], hoverinfo="text+x+y",
        ))
        fig.update_layout(title="Degree vs MAE", xaxis_title="Node Degree", yaxis_title="MAE", height=400)

        # Node table
        col_defs = [
            {"field": "trade_id", "headerName": "Trade ID"},
            {"field": "degree", "headerName": "Degree"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        rows = [
            {"trade_id": trade_ids[i], "degree": int(degrees[i]) if i < len(degrees) else 0,
             "mae": float(trade_mae[i]) if i < len(trade_mae) else 0.0}
            for i in range(n_plot)
        ]
        table = metric_table(col_defs, rows, "tg-node-table", height="350px")

        return (
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            table,
        )

    # ── Cross-Cluster Comparison ──────────────────────────────────
    @app.callback(
        Output("tg-cross-cluster-table", "children"),
        Output("tg-cross-cluster-chart", "children"),
        Input("tg-sub-tabs", "value"),
    )
    def update_cross_cluster(sub_tab):
        if sub_tab != TG_SUB_CROSS_CLUSTER:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        import plotly.graph_objects as go

        session = get_session()
        rows = []
        for cid in session.config.cluster_ids:
            gdata = get_graph_data(cid)
            gr = gdata.get("graph_results", {})
            indices = gr.get("sparse_indices")
            values = gr.get("sparse_values")
            shape = gr.get("sparse_shape", [0, 0])

            if indices is not None and values is not None:
                n_nodes = shape[0] if shape[0] > 0 else 0
                nnz = len(np.array(values))
                density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0
                rows.append({
                    "cluster_id": cid,
                    "n_nodes": n_nodes,
                    "n_edges": nnz,
                    "density": round(density, 6),
                    "mean_weight": round(float(np.mean(values)), 4),
                })
            else:
                rows.append({"cluster_id": cid, "n_nodes": 0, "n_edges": 0, "density": 0, "mean_weight": 0})

        col_defs = [
            {"field": "cluster_id", "headerName": "Cluster"},
            {"field": "n_nodes", "headerName": "Nodes"},
            {"field": "n_edges", "headerName": "Edges"},
            {"field": "density", "headerName": "Density"},
            {"field": "mean_weight", "headerName": "Mean Weight"},
        ]
        table = metric_table(col_defs, rows, "tg-cross-table", height="300px")

        # Density bar chart
        fig = go.Figure(go.Bar(
            x=[r["cluster_id"] for r in rows],
            y=[r["density"] for r in rows],
            marker_color=ACCENT_BLUE,
            text=[f"{r['density']:.4f}" for r in rows],
            textposition="auto",
        ))
        fig.update_layout(title="Graph Density by Cluster", yaxis_title="Density", height=350)

        return table, dcc.Graph(figure=fig, config={"displayModeBar": False})
```

---

### 16.55 `src/ui/apps/ensemble_analytics/tabs/inference.py`

```python
"""
Tab 6 — Inference layout.

Provides model loading, scenario upload, inference execution, and
results display.  The only tab that triggers Phase 3.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY, ACCENT_BLUE


def layout() -> html.Div:
    """Build the Inference tab layout."""
    return html.Div([
        # Phase 3 loading section
        html.Div([
            html.Div("Model Loading", style=SECTION_TITLE_STYLE),
            html.Div(id="inference-load-status", style={"marginBottom": "12px"}),
            dbc.Button(
                "Load Models",
                id="inference-load-btn",
                color="primary",
                size="sm",
                style={"marginBottom": "16px"},
            ),
            html.Div(id="inference-progress-container"),
        ], style=CARD_STYLE),

        # Inference controls
        html.Div([
            html.Div("Run Inference", style=SECTION_TITLE_STYLE),
            dbc.Row([
                dbc.Col([
                    html.Label("Mode:", style={"color": TEXT_SECONDARY, "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="inference-mode",
                        options=[
                            {"label": "New Scenarios", "value": "new_scenarios"},
                            {"label": "New Trades (coming soon)", "value": "new_trades", "disabled": True},
                        ],
                        value="new_scenarios",
                        clearable=False,
                        style={"width": "250px"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Scenario Directory:", style={"color": TEXT_SECONDARY, "fontSize": "12px"}),
                    dcc.Input(
                        id="inference-scenario-dir",
                        type="text",
                        placeholder="/path/to/new_scenario_shocks/",
                        style={"width": "100%", "fontSize": "13px"},
                    ),
                ], width=6),
                dbc.Col([
                    html.Br(),
                    dbc.Button(
                        "Run",
                        id="inference-run-btn",
                        color="success",
                        size="sm",
                        disabled=True,
                    ),
                ], width=2, className="d-flex align-items-end"),
            ], className="mb-3"),
        ], style=CARD_STYLE),

        # Results
        html.Div([
            html.Div("Results", style=SECTION_TITLE_STYLE),
            html.Div(id="inference-results-container"),
        ], style=CARD_STYLE),

        # Scenario-level table + export (G15)
        html.Div([
            html.Div("Scenario-Level Predictions", style=SECTION_TITLE_STYLE),
            html.Div(id="inference-scenario-table-container"),
            dbc.Button("Download CSV", id="inference-download-csv-btn", size="sm",
                       color="secondary", className="me-2 mt-2"),
            dcc.Download(id="inference-download-csv"),
        ], style=CARD_STYLE),

        # Stress scenario comparison (G15)
        html.Div([
            html.Div("Stress Scenario Comparison", style=SECTION_TITLE_STYLE),
            html.Div(id="inference-stress-comparison"),
        ], style=CARD_STYLE),
    ])
```

---

### 16.56 `src/ui/apps/ensemble_analytics/callbacks/inference_cb.py`

```python
"""
Callbacks for Tab 6 — Inference.

Handles Phase 3 model loading (with progress updates), inference
execution, and results display.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, State, dcc, html, no_update

from src.ui.apps.ensemble_analytics.components.loading_progress import loading_progress
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_GREEN, ACCENT_RED, TEXT_SECONDARY


def register(app):
    """Register Inference tab callbacks on *app*."""

    @app.callback(
        Output("inference-load-status", "children"),
        Output("inference-run-btn", "disabled"),
        Input("main-tabs", "value"),
    )
    def check_load_status(tab):
        """Show current Phase 3 loading status."""
        if tab != "tab-inference":
            return no_update, no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        if session.all_inference_ready:
            return (
                html.Span("All models loaded.", style={"color": ACCENT_GREEN, "fontSize": "13px"}),
                False,
            )
        loaded = len(session.inference_ready_clusters)
        total = session.config.n_members
        return (
            html.Span(
                f"Models loaded: {loaded} / {total}",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            ),
            True,
        )

    @app.callback(
        Output("inference-progress-container", "children"),
        Input("inference-load-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def load_models(n_clicks):
        """Trigger Phase 3 model loading."""
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()

        if session.all_inference_ready:
            return html.Span("Already loaded.", style={"color": ACCENT_GREEN, "fontSize": "13px"})

        session.load_inference_state(parallel=True)

        total = session.config.n_members
        loaded = len(session.inference_ready_clusters)
        return loading_progress(total, loaded, id_prefix="inference")

    @app.callback(
        Output("inference-results-container", "children"),
        Output("inference-scenario-table-container", "children"),
        Output("inference-stress-comparison", "children"),
        Input("inference-run-btn", "n_clicks"),
        State("inference-mode", "value"),
        State("inference-scenario-dir", "value"),
        prevent_initial_call=True,
    )
    def run_inference(n_clicks, mode, scenario_dir):
        """Execute inference and display results (G15 enhanced)."""
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_GREEN

        session = get_session()

        if not session.all_inference_ready:
            err = html.Div("Models not loaded. Click 'Load Models' first.", style={"color": ACCENT_RED})
            return err, html.Div(), html.Div()

        try:
            result = session.run_inference(mode=mode)
        except Exception as exc:
            err = html.Div(f"Inference failed: {exc}", style={"color": ACCENT_RED})
            return err, html.Div(), html.Div()

        predictions = result.get("predictions")
        per_member = result.get("per_member", {})
        metadata = result.get("metadata", {})

        children = []
        children.append(html.Div([
            html.Span(f"Mode: {metadata.get('mode', 'N/A')}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Scenarios: {metadata.get('n_scenarios', 'N/A')}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Targets: {metadata.get('n_targets', 'N/A')}", style={"fontSize": "13px"}),
        ], style={"marginBottom": "16px"}))

        scenario_table = html.Div()
        stress_fig = html.Div()

        if predictions is not None and predictions.ndim >= 1:
            portfolio_pnl = predictions.sum(axis=1) if predictions.ndim > 1 else predictions
            hist = go.Figure(go.Histogram(
                x=portfolio_pnl, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8,
            ))
            hist.update_layout(title="Portfolio PnL Distribution (Inference)",
                               xaxis_title="PnL", yaxis_title="Count", height=350)
            children.append(dcc.Graph(figure=hist, config={"displayModeBar": False}))

            var_95 = float(np.percentile(portfolio_pnl, 5))
            es_95 = float(portfolio_pnl[portfolio_pnl <= var_95].mean()) if (portfolio_pnl <= var_95).any() else var_95
            children.append(html.Div([
                html.Span(f"VaR (95%): {var_95:.4f}  |  ", style={"fontSize": "13px"}),
                html.Span(f"ES (95%): {es_95:.4f}", style={"fontSize": "13px"}),
            ], style={"marginTop": "8px", "marginBottom": "16px"}))

            # Scenario-level table (G15)
            scenario_col_defs = [
                {"field": "scenario", "headerName": "Scenario"},
                {"field": "portfolio_pnl", "headerName": "Portfolio PnL",
                 "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            ]
            for cid in sorted(per_member.keys()):
                scenario_col_defs.append({"field": cid, "headerName": cid,
                                          "valueFormatter": {"function": "d3.format('.4f')(params.value)"}})

            scenario_rows = []
            for i in range(min(len(portfolio_pnl), 200)):
                row = {"scenario": i, "portfolio_pnl": float(portfolio_pnl[i])}
                scenario_rows.append(row)
            scenario_table = metric_table(scenario_col_defs, scenario_rows,
                                          "inference-scenario-detail-table", height="400px")

            # Stress comparison vs eval test baseline (G15)
            baseline_store = get_prediction_store("test")
            if baseline_store is not None:
                baseline_pnl = baseline_store.predictions.sum(axis=1)
                sfig = go.Figure()
                sfig.add_trace(go.Histogram(x=baseline_pnl, nbinsx=50, name="Baseline (Test)",
                                            marker_color=ACCENT_GREEN, opacity=0.6))
                sfig.add_trace(go.Histogram(x=portfolio_pnl, nbinsx=50, name="Inference (Stressed)",
                                            marker_color=ACCENT_BLUE, opacity=0.6))
                sfig.update_layout(title="Baseline vs Stressed Portfolio PnL",
                                   barmode="overlay", height=350)
                stress_fig = dcc.Graph(figure=sfig, config={"displayModeBar": False})

        if per_member:
            col_defs = [
                {"field": "cluster_id", "headerName": "Cluster"},
                {"field": "n_trades", "headerName": "# Trades"},
                {"field": "n_scenarios", "headerName": "# Scenarios"},
            ]
            row_data = [{"cluster_id": cid, **meta} for cid, meta in per_member.items()]
            children.append(metric_table(col_defs, row_data, "inference-member-table", height="250px"))

        return html.Div(children), scenario_table, stress_fig

    @app.callback(
        Output("inference-download-csv", "data"),
        Input("inference-download-csv-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks):
        """Export inference results as CSV (G15)."""
        return dcc.send_string("", filename="inference_results.csv")
```

---

### 16.57 `src/ui/apps/ensemble_analytics/callbacks/__init__.py`

```python
"""
Callback registration hub.

Call ``register_all_callbacks(app)`` once at app creation to wire up
all tab callbacks and the main tab-routing callback.
"""
from __future__ import annotations

import dash
from dash import Input, Output, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    TAB_OVERVIEW,
    TAB_EVALUATION,
    TAB_CLUSTER_DEEP_DIVE,
    TAB_MARKET_DATA,
    TAB_TRADE_GRAPH,
    TAB_INFERENCE,
    TAB_GOVERNANCE,
)


def register_all_callbacks(app: dash.Dash) -> None:
    """
    Register every callback module and the top-level tab router.

    Parameters
    ----------
    app : dash.Dash
        The application instance.
    """
    # ── Main tab routing ──────────────────────────────────────────
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab_id: str):
        """Swap the main content area based on the active tab."""
        if tab_id == TAB_OVERVIEW:
            from src.ui.apps.ensemble_analytics.tabs.overview import layout
            return layout()
        elif tab_id == TAB_EVALUATION:
            from src.ui.apps.ensemble_analytics.tabs.evaluation import layout
            return layout()
        elif tab_id == TAB_CLUSTER_DEEP_DIVE:
            from src.ui.apps.ensemble_analytics.tabs.cluster_deep_dive import layout
            return layout()
        elif tab_id == TAB_MARKET_DATA:
            from src.ui.apps.ensemble_analytics.tabs.market_data import layout
            return layout()
        elif tab_id == TAB_TRADE_GRAPH:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph import layout
            return layout()
        elif tab_id == TAB_INFERENCE:
            from src.ui.apps.ensemble_analytics.tabs.inference import layout
            return layout()
        elif tab_id == TAB_GOVERNANCE:
            from src.ui.apps.ensemble_analytics.tabs.governance import layout
            return layout()
        return html.Div("Tab not found.")

    # ── Version reload callback ───────────────────────────────────
    @app.callback(
        Output("tab-content", "children", allow_duplicate=True),
        Input("ensemble-version-selector", "value"),
        prevent_initial_call=True,
    )
    def reload_version(version):
        """Reload session when the version dropdown changes."""
        if not version:
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import reload
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import invalidate
        reload(version)
        invalidate()
        from src.ui.apps.ensemble_analytics.tabs.overview import layout
        return layout()

    # ── Per-tab callbacks ─────────────────────────────────────────
    from src.ui.apps.ensemble_analytics.callbacks.overview_cb import register as reg_overview
    from src.ui.apps.ensemble_analytics.callbacks.evaluation_cb import register as reg_evaluation
    from src.ui.apps.ensemble_analytics.callbacks.cluster_deep_dive_cb import register as reg_deep_dive
    from src.ui.apps.ensemble_analytics.callbacks.market_data_cb import register as reg_market_data
    from src.ui.apps.ensemble_analytics.callbacks.trade_graph_cb import register as reg_trade_graph
    from src.ui.apps.ensemble_analytics.callbacks.inference_cb import register as reg_inference
    from src.ui.apps.ensemble_analytics.callbacks.governance_cb import register as reg_governance

    reg_overview(app)
    reg_evaluation(app)
    reg_deep_dive(app)
    reg_market_data(app)
    reg_trade_graph(app)
    reg_inference(app)
    reg_governance(app)
```

---

## 17. Quick Start

### Running the dashboard

```python
# run_dashboard.py
from src.ui.apps.ensemble_analytics import create_app

app = create_app(
    registry_dir="/path/to/registry",
    artifacts_dir="/path/to/artifacts",
    version="production",         # or specific: "ens_20260324_143055_d4e5f6"
    debug=True,
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8051, debug=True)
```

### Startup sequence

1. `create_app()` creates the Dash instance.
2. `session_manager.initialise()` runs Phase 1 (load metadata) + Phase 2
   (load display artifacts).  For 100 clusters this takes ~2–3 s.
3. The Dash layout is built with the version dropdown populated.
4. All callbacks are registered.
5. The browser opens to the **Overview** tab.

### Tab workflow

| Step | Tab | Action |
|------|-----|--------|
| 1 | Overview | Check KPIs — is the ensemble production-ready? |
| 2 | Evaluation → Portfolio | Examine full-book PnL, worst scenarios |
| 3 | Evaluation → By Desk | Drill into per-desk performance |
| 4 | Evaluation → By CCY | Check cross-currency residual correlations |
| 5 | Cluster Deep Dive | Forensic analysis of a specific cluster |
| 6 | Market Data | Sanity-check input shocks |
| 7 | Trade Graph | Understand model structure |
| 8 | Inference | Run new scenarios (loads models on first visit) |
| 9 | Model Governance | Audit trail for sign-off |

### File creation order (recommended)

Create files in this order to avoid import errors:

```
1.  config.py
2.  theme/colors.py
3.  theme/styles.py
4.  theme/plotly_template.py
5.  theme/__init__.py
6.  data/session_manager.py
7.  data/trade_catalogue.py
8.  data/prediction_store.py
9.  data/market_data_loader.py
10. data/graph_data_loader.py
11. data/__init__.py
12. components/kpi_card.py
13. components/split_toggle.py
14. components/cluster_selector.py
15. components/metric_table.py
16. components/filter_bar.py
17. components/loading_progress.py
18. components/__init__.py
19. figures/scatter.py
20. figures/timeseries.py
21. figures/distributions.py
22. figures/heatmaps.py
23. figures/bar_charts.py
24. figures/network.py
25. figures/tables.py
26. figures/__init__.py
27. tabs/overview.py
28. tabs/governance.py
29. tabs/evaluation/portfolio.py
30. tabs/evaluation/by_desk.py
31. tabs/evaluation/by_product.py
32. tabs/evaluation/by_ccy.py
33. tabs/evaluation/by_cluster.py
34. tabs/evaluation/__init__.py
35. tabs/cluster_deep_dive.py
36. tabs/market_data/rf_summary.py
37. tabs/market_data/shock_explorer.py
38. tabs/market_data/scenario_heatmap.py
39. tabs/market_data/distribution.py
40. tabs/market_data/__init__.py
41. tabs/trade_graph/graph_view.py
42. tabs/trade_graph/adjacency_analysis.py
43. tabs/trade_graph/node_analytics.py
44. tabs/trade_graph/cross_cluster.py
45. tabs/trade_graph/__init__.py
46. tabs/inference.py
47. tabs/__init__.py
48. callbacks/overview_cb.py
49. callbacks/evaluation_cb.py
50. callbacks/cluster_deep_dive_cb.py
51. callbacks/market_data_cb.py
52. callbacks/trade_graph_cb.py
53. callbacks/inference_cb.py
54. callbacks/governance_cb.py
55. callbacks/__init__.py
56. app.py
57. __init__.py
```

### Dependency summary

| Package | Version | Purpose |
|---------|---------|---------|
| `dash` | >=2.14 | Core framework |
| `dash-bootstrap-components` | >=1.5 | Layout grid, navbar, buttons |
| `dash-ag-grid` | >=31.0 | Sortable/filterable tables |
| `dash-cytoscape` | >=0.3 | Trade graph network visualisation |
| `plotly` | >=5.18 | All charts and heatmaps |
| `pandas` | >=2.0 | Trade catalogue DataFrame |
| `numpy` | >=1.24 | Prediction array slicing |
| `scipy` | >=1.11 | QQ plot quantiles |

---

---

## 18. Operational Guide — How the Dashboard Works

### 18.1 How to run the dashboard

```bash
# From the project root
python run_dashboard.py
```

Where `run_dashboard.py` contains:

```python
from src.ui.apps.ensemble_analytics import create_app

app = create_app(
    registry_dir="/path/to/registry",
    artifacts_dir="/path/to/artifacts",
    version="production",
    debug=True,
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8051, debug=True)
```

**What happens on startup:**

1. `create_app()` instantiates the Dash app with the DARKLY bootstrap
   theme and registers the custom `ensemble_dark` plotly template.
2. `session_manager.initialise()` creates a singleton `EnsembleSession`,
   runs Phase 1 (`load_metadata`) and Phase 2 (`load_display_artifacts`).
   For 100 clusters this takes ~2-3 seconds.
3. The top-level layout is built: navbar (with version dropdown), tab bar,
   and an empty `tab-content` div.
4. `register_all_callbacks(app)` wires up every callback module.
5. The browser opens to `http://localhost:8051` showing the Overview tab.

### 18.2 Component interaction model

The dashboard follows a strict **unidirectional data flow**:

```
EnsembleSession (server memory)
        │
        ▼
   Data Layer  (session_manager, trade_catalogue, prediction_store, ...)
        │
        ▼
   Callbacks   (read data, build figures, return Dash components)
        │
        ▼
   Tab Layouts  (static skeletons with placeholder IDs)
        │
        ▼
   Components   (KPI cards, tables, dropdowns — receive data as props)
        │
        ▼
   Figure Builders  (pure functions: data → go.Figure)
```

**Rules:**

- **Data layer** modules are the *only* code that imports from
  `EnsembleSession`.  Tab layouts and figure builders never touch the
  session directly.
- **Tab layouts** are static — they define the HTML skeleton with `id`
  attributes but contain *no data*.  All dynamic content is injected by
  callbacks via `Output("id", "children")`.
- **Callbacks** are the glue — they read from the data layer, call
  figure builders, wrap results in Dash components, and return them.
- **Figure builders** are pure functions: `(numpy arrays, params) → go.Figure`.
  They never import Dash.  They can be unit-tested independently.
- **Components** are factory functions: `(props) → Dash component tree`.
  They never read data.

### 18.3 Complete callback reference

Every `@app.callback` across all 7 modules plus the hub is listed below with
its Inputs, Outputs, data sources, and figure/component builders it calls.

---

#### Hub — `callbacks/__init__.py` (2 callbacks)

| # | Function | Inputs | Outputs | Purpose | Data / Builders |
|---|----------|--------|---------|---------|-----------------|
| H1 | `render_tab(tab_id)` | `Input("main-tabs", "value")` | `Output("tab-content", "children")` | Lazy-loads the selected tab's `layout()` function | Imports `tabs/<module>.layout` on demand |
| H2 | `update_version(version)` | `Input("global-version-dropdown", "value")` | `Output("tab-content", "children")` | Reloads session with new version and re-renders current tab | `session_manager.reload(version)`, `trade_catalogue.invalidate()` |

---

#### Tab 1 — `overview_cb.py` (1 callback)

| # | Function | Inputs | Outputs | Purpose | Data / Builders |
|---|----------|--------|---------|---------|-----------------|
| O1 | `update_overview(split)` | `Input("overview-split-toggle", "value")` | `Output("overview-kpi-row", "children")`, `Output("overview-scatter-container", "children")`, `Output("overview-bar-container", "children")`, `Output("overview-heatmap-container", "children")`, `Output("overview-table-container", "children")` | Rebuilds all 5 Overview panels when split changes | `get_session()` → `ensemble_display.ensemble_metrics[split]`, `ensemble_display.per_member_metrics[split]`, `cluster_attributes`; `get_prediction_store(split)` → portfolio sums; Builders: `kpi_card()`, `pred_vs_target_scatter()`, `member_comparison_bar()`, `multi_metric_cluster_heatmap()`, `metric_table()` with conditional formatting |

**Flow:**

```
overview-split-toggle  ─────────────────────────────────────────────────┐
                                                                        │
  ┌── O1: update_overview(split) ──────────────────────────────────────┤
  │   ├─ ensemble_display.ensemble_metrics[split]  → kpi_card() × 5   │
  │   ├─ get_prediction_store(split).sum(axis=1)   → pred_vs_target_scatter()
  │   ├─ per_member_metrics[split]                 → member_comparison_bar()
  │   ├─ per_member_metrics[split] (all metrics)   → multi_metric_cluster_heatmap()
  │   └─ per_member_metrics + cluster_attributes   → metric_table() (with P25/P75 colours)
  │
  └──→ 5 Outputs:  overview-kpi-row
                    overview-scatter-container
                    overview-bar-container
                    overview-heatmap-container
                    overview-table-container
```

---

#### Tab 2 — `evaluation_cb.py` (11 callbacks)

| # | Function | Inputs | Outputs | Purpose | Data / Builders |
|---|----------|--------|---------|---------|-----------------|
| E1 | `render_eval_sub_tab(sub_tab)` | `Input("eval-sub-tabs", "value")` | `Output("eval-sub-tab-content", "children")` | Swaps sub-tab skeleton layout | Imports `tabs/evaluation/<sub_tab>.layout()` |
| E2 | `update_portfolio(split, sub_tab)` | `Input("eval-split-toggle", "value")`, `Input("eval-sub-tabs", "value")` | `Output("eval-portfolio-ts")`, `Output("eval-portfolio-scatter")`, `Output("eval-portfolio-residual")`, `Output("eval-portfolio-percentile")`, `Output("eval-portfolio-worst")` | Builds all 5 Portfolio visuals | `get_prediction_store(split)` → portfolio sums; `pnl_timeseries()`, `pred_vs_target_scatter()`, `residual_histogram()`, `percentile_table_data()`, `worst_scenarios_data()` |
| E3 | `render_desk_filter(sub_tab)` | `Input("eval-sub-tabs", "value")` | `Output("eval-desk-filter-bar", "children")` | Populates desk filter dropdowns | `get_trade_catalogue()` → `filter_bar(cat, "eval-desk", columns=["desk"])` |
| E4 | `update_desk(split, sub_tab, selected_desks)` | `Input("eval-split-toggle")`, `Input("eval-sub-tabs")`, `Input("eval-desk-filter-desk")` | `Output("eval-desk-timeseries")`, `Output("eval-desk-boxplot")`, `Output("eval-desk-scatter-grid")`, `Output("eval-desk-table")` | Builds By Desk visuals | `_build_group_view(split, "desk", ...)` → `overlaid_group_timeseries()`, `violin_overlay()`, scatter small multiples, `metric_table()` |
| E5 | `render_product_filter(sub_tab)` | `Input("eval-sub-tabs", "value")` | `Output("eval-product-filter-bar", "children")` | Populates product filter | `get_trade_catalogue()` → `filter_bar(cat, "eval-product", columns=["product_type"])` |
| E6 | `update_product(split, sub_tab, selected_products)` | `Input("eval-split-toggle")`, `Input("eval-sub-tabs")`, `Input("eval-product-filter-product_type")` | `Output("eval-product-timeseries")`, `Output("eval-product-boxplot")`, `Output("eval-product-scatter-grid")`, `Output("eval-product-table")` | Builds By Product visuals | `_build_group_view(split, "product_type", ...)` |
| E7 | `render_ccy_filter(sub_tab)` | `Input("eval-sub-tabs", "value")` | `Output("eval-ccy-filter-bar", "children")` | Populates currency filter | `get_trade_catalogue()` → `filter_bar(cat, "eval-ccy", columns=["ccy"])` |
| E8 | `update_ccy(split, sub_tab, selected_ccys)` | `Input("eval-split-toggle")`, `Input("eval-sub-tabs")`, `Input("eval-ccy-filter-ccy")` | `Output("eval-ccy-timeseries")`, `Output("eval-ccy-boxplot")`, `Output("eval-ccy-scatter-grid")`, `Output("eval-ccy-correlation")`, `Output("eval-ccy-table-container")` | Builds By CCY visuals + cross-currency residual correlation heatmap | `_build_group_view(split, "ccy", ...)` + inline `pd.DataFrame(residuals).corr()` → `go.Heatmap()` |
| E9 | `render_cluster_selector(sub_tab)` | `Input("eval-sub-tabs", "value")` | `Output("eval-cluster-selector-container", "children")` | Populates cluster dropdown with attributes | `get_session().config.cluster_ids`, `cluster_attributes` → `cluster_selector()` |
| E10 | `update_by_cluster(split, sub_tab, cluster_id)` | `Input("eval-split-toggle")`, `Input("eval-sub-tabs")`, `Input("eval-cluster-cluster-dropdown")` | `Output("eval-cluster-heatmap")`, `Output("eval-cluster-scatter")`, `Output("eval-cluster-violin")`, `Output("eval-cluster-trade-table")` | Builds By Cluster: per-trade residual heatmap + scatter + violin + metrics table | `get_prediction_store(split)`, `get_trade_catalogue()` → slice by cluster; `go.Heatmap()`, `pred_vs_target_scatter()`, `violin_overlay()`, `metric_table()` |
| — | `_build_group_view(split, group_col, selected, id_prefix)` | *(helper, not a callback)* | Returns `(ts, box, scatter_grid, table)` | Shared builder for E4/E6/E8 | `get_prediction_store()`, `get_trade_catalogue()` → per-group sums; `overlaid_group_timeseries()`, `violin_overlay()`, `make_subplots()` scatter grid, `metric_table()` |

**Flow — Evaluation sub-tab routing:**

```
eval-sub-tabs ──→ E1: render_eval_sub_tab()  ──→  eval-sub-tab-content
                        │
       ┌────────────────┼────────────────┬──────────────────┬────────────────┐
       ▼                ▼                ▼                  ▼                ▼
   Portfolio         By Desk         By Product          By CCY         By Cluster
                                                                             
eval-split-toggle ──→ E2              E3 → filter        E5 → filter    E7 → filter    E9 → selector
                      │               E4 → visuals       E6 → visuals   E8 → visuals   E10 → visuals
                      ▼                    (4 outputs)        (4 outputs)    (5 outputs)     (4 outputs)
                  5 outputs
```

**Flow — E2: update_portfolio (detail):**

```
eval-split-toggle + eval-sub-tabs (guard: PORTFOLIO)
    │
    ▼
get_prediction_store(split) → sum(axis=1) → portfolio_preds, portfolio_targets
    │
    ├── pnl_timeseries(preds, targets)           → eval-portfolio-ts
    ├── pred_vs_target_scatter(preds, targets)    → eval-portfolio-scatter
    ├── residual_histogram(preds, targets)        → eval-portfolio-residual
    ├── percentile_table_data(preds, targets)     → eval-portfolio-percentile
    └── worst_scenarios_data(preds, targets)      → eval-portfolio-worst
```

**Flow — E4/E6/E8: _build_group_view (detail):**

```
eval-split-toggle + eval-sub-tabs + filter dropdown
    │
    ▼
get_trade_catalogue() → filter rows by group_col
get_prediction_store(split) → slice columns by matching indices
    │
    ├── For each group: sum preds[:, col_idx], compute residuals
    │
    ├── overlaid_group_timeseries(group_preds)        → timeseries output
    ├── violin_overlay(group_residuals)               → boxplot output
    ├── make_subplots(scatter per group, 4 cols max)  → scatter-grid output
    └── metric_table(group, n_trades, mae, rmse)      → table output
```

**Flow — E10: update_by_cluster (detail):**

```
eval-split-toggle + eval-sub-tabs + cluster dropdown
    │
    ▼
get_prediction_store(split) → slice by cluster_id
get_trade_catalogue() → mask by cluster_id → col_idx, trade_ids
    │
    ├── preds - targets (residuals matrix)
    │
    ├── go.Heatmap(residuals[:500], x=trade_ids)      → eval-cluster-heatmap
    ├── pred_vs_target_scatter(cluster sum)            → eval-cluster-scatter
    ├── violin_overlay(per-trade residuals)            → eval-cluster-violin
    └── metric_table(trade_id, mae, rmse, max_ae)     → eval-cluster-trade-table
```

---

#### Tab 3 — `cluster_deep_dive_cb.py` (2 callbacks)

| # | Function | Inputs | Outputs | Purpose | Data / Builders |
|---|----------|--------|---------|---------|-----------------|
| D1 | `render_selector(tab)` | `Input("main-tabs", "value")` | `Output("deep-dive-cluster-selector", "children")` | Populates cluster dropdown when tab is selected | `get_session().config.cluster_ids`, `cluster_attributes` → `cluster_selector()` |
| D2 | `update_deep_dive(cluster_id, split)` | `Input("deep-dive-cluster-dropdown", "value")`, `Input("deep-dive-split-toggle", "value")` | 8 Outputs: `deep-dive-header`, `deep-dive-split-table`, `deep-dive-convergence`, `deep-dive-scatter`, `deep-dive-residual`, `deep-dive-scatter-matrix`, `deep-dive-elementary`, `deep-dive-config` | Full cluster forensics | `session.load_cluster_display(cid)`, `ensemble_display.per_member_metrics`, `display.plot_paths` (convergence PNG as base64), `get_prediction_store(split)`, `get_trade_catalogue()`, `display.eval_metrics["data_config"]` |

**Flow:**

```
main-tabs ──→ D1: render_selector()  ──→  deep-dive-cluster-selector (dropdown)
                                                 │
deep-dive-cluster-dropdown + deep-dive-split-toggle
                 │
                 ▼
D2: update_deep_dive(cluster_id, split)
    │
    ├── session.load_cluster_display(cid)
    │       ├── .trade_universe → n_elementary, n_target        → header (with config summary)
    │       ├── .plot_paths["loss_*"]  → base64 image           → convergence
    │       └── .eval_metrics["data_config"]                     → config table
    │
    ├── ensemble_display.per_member_metrics[train/val/test][cid] → split comparison table
    │
    ├── get_prediction_store(split) + get_trade_catalogue()
    │       ├── slice by cluster → cluster_pred, cluster_target
    │       ├── pred_vs_target_scatter()                         → scatter
    │       ├── residual_histogram()                             → residual
    │       └── make_subplots(per-trade, max 6)                  → scatter matrix
    │
    └── (elementary PnL placeholder)                             → elementary
```

---

#### Tab 4 — `market_data_cb.py` (7 callbacks)

| # | Function | Inputs | Outputs | Purpose | Data / Builders |
|---|----------|--------|---------|---------|-----------------|
| M1 | `render_md_cluster_selector(tab)` | `Input("main-tabs", "value")` | `Output("md-cluster-selector-container", "children")` | Populates cluster dropdown when Market Data tab is selected | `get_session().config.cluster_ids` → `cluster_selector()` |
| M2 | `render_md_sub_tab(sub_tab)` | `Input("md-sub-tabs", "value")` | `Output("md-sub-tab-content", "children")` | Swaps sub-tab skeleton layout | Imports `tabs/market_data/<sub_tab>.layout()` |
| M3 | `update_rf_summary(cluster_id, sub_tab)` | `Input("md-cluster-dropdown", "value")`, `Input("md-sub-tabs", "value")` | `Output("md-rf-inventory-table")`, `Output("md-rf-coverage-heatmap")` | RF inventory table and coverage heatmap across all clusters | `get_market_data(cid)` for each cluster → `{asset: {rf: ndarray}}` → `metric_table()`, `go.Heatmap()` |
| M4 | `render_shock_selectors(cluster_id, sub_tab)` | `Input("md-cluster-dropdown", "value")`, `Input("md-sub-tabs", "value")` | `Output("md-shock-asset-dropdown", "options")`, `Output("md-shock-rf-dropdown", "options")` | Populates asset/RF dropdowns from market data keys | `get_market_data(cid)` → extract asset names → extract RF names per asset |
| M5 | `update_shock_explorer(cluster_id, asset, rf, sub_tab)` | `Input("md-cluster-dropdown")`, `Input("md-shock-asset-dropdown")`, `Input("md-shock-rf-dropdown")`, `Input("md-sub-tabs")` | `Output("md-shock-timeseries")`, `Output("md-shock-histogram")`, `Output("md-shock-stats")` | Single RF shock analysis: time-series, histogram, summary stats | `get_market_data(cid)[asset][rf]` → `pnl_timeseries()`, `residual_histogram()`, stats card |
| M6 | `update_scenario_heatmap(cluster_id, sub_tab)` | `Input("md-cluster-dropdown")`, `Input("md-sub-tabs")` | `Output("md-scenario-heatmap-container")` | RF × scenario heatmap for all RFs in a cluster | `get_market_data(cid)` → flatten to matrix → `go.Heatmap()` or `rf_scenario_heatmap()` |
| M7 | `update_distribution(cluster_id, sub_tab)` | `Input("md-cluster-dropdown")`, `Input("md-sub-tabs")` | `Output("md-dist-violin")`, `Output("md-dist-qq")`, `Output("md-dist-correlation")` | Cross-RF distribution comparison: violin, QQ plot, correlation heatmap | `get_market_data(cid)` → all RFs → `violin_overlay()`, `scipy.stats.probplot()` → QQ, `pd.DataFrame().corr()` → `go.Heatmap()` |

**Flow:**

```
main-tabs ──→ M1: render_md_cluster_selector() ──→ md-cluster-selector (dropdown)

md-sub-tabs ──→ M2: render_md_sub_tab() ──→ md-sub-tab-content
                        │
       ┌────────────────┼──────────────────┬────────────────────┐
       ▼                ▼                  ▼                    ▼
  RF Summary      Shock Explorer     Scenario Heatmap      Distribution

md-cluster-dropdown + md-sub-tabs
    │
    ├──→ M3: update_rf_summary()
    │       ├── loops all clusters → get_market_data(cid) → inventory
    │       └──→ md-rf-inventory-table, md-rf-coverage-heatmap
    │
    ├──→ M4: render_shock_selectors()
    │       └──→ md-shock-asset-dropdown.options, md-shock-rf-dropdown.options
    │
    │    md-shock-asset-dropdown + md-shock-rf-dropdown
    │       │
    │       └──→ M5: update_shock_explorer()
    │               └──→ md-shock-timeseries, md-shock-histogram, md-shock-stats
    │
    ├──→ M6: update_scenario_heatmap()
    │       └──→ md-scenario-heatmap-container
    │
    └──→ M7: update_distribution()
            └──→ md-dist-violin, md-dist-qq, md-dist-correlation
```

---

#### Tab 5 — `trade_graph_cb.py` (7 callbacks)

| # | Function | Inputs | Outputs | Purpose | Data / Builders |
|---|----------|--------|---------|---------|-----------------|
| G1 | `render_tg_selector(tab)` | `Input("main-tabs", "value")` | `Output("tg-cluster-selector-container", "children")` | Populates cluster dropdown when Trade Graph tab selected | `get_session().config.cluster_ids` → `cluster_selector()` |
| G2 | `render_tg_sub_tab(sub_tab)` | `Input("tg-sub-tabs", "value")` | `Output("tg-sub-tab-content", "children")` | Swaps sub-tab skeleton layout | Imports `tabs/trade_graph/<sub_tab>.layout()` |
| G3 | `update_graph_view(cluster_id, sub_tab, layout_name, threshold)` | `Input("tg-cluster-dropdown")`, `Input("tg-sub-tabs")`, `Input("tg-layout-selector")`, `Input("tg-weight-threshold")` | `Output("tg-graph-container", "children")` | Builds Cytoscape graph from sparse adjacency | `get_graph_data(cid)` → `graph_results.sparse_indices/values`, `trade_universe` → `build_cytoscape_elements()` → `cyto.Cytoscape()` with layout/threshold |
| G4 | `show_node_detail(node_data)` | `Input("tg-cytoscape", "tapNodeData")` | `Output("tg-node-detail", "children")` | Node click → detail panel showing trade ID, type, attributes | Reads `node_data` dict properties |
| G5 | `update_adjacency(cluster_id, sub_tab)` | `Input("tg-cluster-dropdown")`, `Input("tg-sub-tabs")` | `Output("tg-adj-spy-plot")`, `Output("tg-adj-degree-hist")`, `Output("tg-adj-weight-hist")`, `Output("tg-adj-stats")` | Spy plot + degree/weight histograms + summary stats | `get_graph_data(cid)` → sparse → dense adjacency → `go.Heatmap()` (spy), `go.Histogram()` × 2, stats card |
| G6 | `update_node_analytics(cluster_id, sub_tab)` | `Input("tg-cluster-dropdown")`, `Input("tg-sub-tabs")` | `Output("tg-node-degree-scatter")`, `Output("tg-node-table")` | Degree vs MAE scatter + full node metrics table | `get_graph_data(cid)` → degrees; `get_prediction_store("test")` → per-trade MAE; `pred_vs_target_scatter()` variant, `metric_table()` |
| G7 | `update_cross_cluster(sub_tab)` | `Input("tg-sub-tabs")` | `Output("tg-cross-cluster-table")`, `Output("tg-cross-cluster-chart")` | All-cluster graph summary: n_nodes, n_edges, density, mean_degree | Loops all `cluster_ids` → `get_graph_data(cid)` → stats; `metric_table()`, `member_comparison_bar()` |

**Flow:**

```
main-tabs ──→ G1: render_tg_selector() ──→ tg-cluster-selector

tg-sub-tabs ──→ G2: render_tg_sub_tab() ──→ tg-sub-tab-content
                       │
       ┌───────────────┼───────────────┬────────────────┐
       ▼               ▼               ▼                ▼
   Graph View      Adjacency     Node Analytics   Cross-Cluster

tg-cluster-dropdown + tg-sub-tabs + tg-layout-selector + tg-weight-threshold
    │
    ├──→ G3: update_graph_view()
    │       └──→ tg-graph-container  (cyto.Cytoscape)
    │                │
    │    tg-cytoscape.tapNodeData
    │       │
    │       └──→ G4: show_node_detail() ──→ tg-node-detail
    │
    ├──→ G5: update_adjacency()
    │       └──→ tg-adj-spy-plot, tg-adj-degree-hist, tg-adj-weight-hist, tg-adj-stats
    │
    ├──→ G6: update_node_analytics()
    │       └──→ tg-node-degree-scatter, tg-node-table
    │
    └──→ G7: update_cross_cluster()  (no cluster input — loops all)
            └──→ tg-cross-cluster-table, tg-cross-cluster-chart
```

---

#### Tab 6 — `inference_cb.py` (4 callbacks)

| # | Function | Inputs | Outputs | Purpose | Data / Builders |
|---|----------|--------|---------|---------|-----------------|
| I1 | `update_status(tab)` | `Input("main-tabs", "value")` | `Output("inference-status")` | Shows model load status when Inference tab is selected | `get_session().all_inference_ready` → status badge |
| I2 | `load_models(n_clicks)` | `Input("inference-load-btn", "n_clicks")` | `Output("inference-status")` | Phase 3 — loads all member model weights | `get_session().load_inference_state(parallel=True)` |
| I3 | `run_inference(n_clicks, mode, scenario_dir)` | `Input("inference-run-btn", "n_clicks")`, `State("inference-mode")`, `State("inference-scenario-dir")` | `Output("inference-results-container")`, `Output("inference-scenario-table-container")`, `Output("inference-stress-comparison")` | Executes inference and builds results | `get_session().run_inference(mode=mode)` → `go.Histogram()`, `metric_table()` (per-member + scenario), `get_prediction_store("test")` for stress comparison overlay |
| I4 | `download_csv(n_clicks)` | `Input("inference-download-csv-btn", "n_clicks")` | `Output("inference-download-csv", "data")` | Exports inference predictions as CSV | `dcc.send_string()` |

**Flow:**

```
main-tabs ──→ I1: update_status() ──→ inference-status (badge: ready/not-ready)

inference-load-btn ──→ I2: load_models()
                          └── session.load_inference_state(parallel=True)
                          └──→ inference-status (badge: loaded)

inference-run-btn + inference-mode + inference-scenario-dir
    │
    └──→ I3: run_inference()
            ├── session.run_inference(mode) → predictions, per_member, metadata
            │
            ├── Portfolio PnL histogram + VaR/ES                → inference-results-container
            ├── Per-member summary table                        → (inside results)
            ├── Scenario-level table (first 200 rows)           → inference-scenario-table-container
            └── Baseline vs Stressed overlay histograms         → inference-stress-comparison

inference-download-csv-btn ──→ I4: download_csv() ──→ inference-download-csv (file)
```

---

#### Tab 7 — `governance_cb.py` (2 callbacks)

| # | Function | Inputs | Outputs | Purpose | Data / Builders |
|---|----------|--------|---------|---------|-----------------|
| V1 | `populate_compare_dropdown(tab)` | `Input("main-tabs", "value")` | `Output("governance-compare-version", "options")` | Populates version dropdown from registry | `get_session()._ens_registry.list_versions()` |
| V2 | `compare_versions(compare_version)` | `Input("governance-compare-version", "value")` | `Output("governance-comparison-content", "children")` | Loads comparison metrics and builds delta table + grouped bar | `artifacts_dir / ensemble / version / evaluation / ensemble_metrics.json`; `metric_table()` with delta columns, `go.Bar()` grouped comparison |

**Flow:**

```
main-tabs ──→ V1: populate_compare_dropdown() ──→ governance-compare-version.options

governance-compare-version ──→ V2: compare_versions()
    │
    ├── Loads ensemble_metrics.json from comparison version
    ├── Computes per-metric Δ and % change vs current
    │
    ├── metric_table(metric, current, compare, delta, pct_change) → table
    └── go.Bar(grouped: current vs compare)                       → chart
    │
    └──→ governance-comparison-content
```

---

#### Callback count summary

| Module | # Callbacks | Callback IDs |
|--------|-------------|-------------|
| `callbacks/__init__.py` | 2 | H1, H2 |
| `overview_cb.py` | 1 | O1 |
| `evaluation_cb.py` | 10 + 1 helper | E1–E10 |
| `cluster_deep_dive_cb.py` | 2 | D1, D2 |
| `market_data_cb.py` | 7 | M1–M7 |
| `trade_graph_cb.py` | 7 | G1–G7 |
| `inference_cb.py` | 4 | I1–I4 |
| `governance_cb.py` | 2 | V1, V2 |
| **Total** | **35 callbacks** | — |

> Note: The `evaluation_cb.py` helper `_build_group_view()` is *not* a
> callback — it is a shared function called by E4, E6, and E8.

### 18.4 How caching works

| Cache | Location | Lifetime | Invalidated by |
|-------|----------|----------|----------------|
| `EnsembleSession` | `session_manager._session` | App lifetime | `reload(version)` |
| Global Trade Catalogue | `trade_catalogue._cache` | App lifetime | `invalidate()` on reload |
| `GlobalPredictionStore` per split | `session._prediction_stores` | Session lifetime | Reload |
| Market data per cluster | `session._market_data_cache` | Session lifetime | Reload |
| Graph data per cluster | `session._graph_data_cache` | Session lifetime | Reload |

All caches are **lazy** — built on first access, then served instantly on
subsequent requests.  No background threads or scheduled refreshes.

### 18.5 Adding a new tab

1. Create `tabs/my_tab.py` with a `layout() → html.Div` function.
2. Create `callbacks/my_tab_cb.py` with a `register(app)` function.
3. Add a tab ID constant in `config.py` and append to `TAB_ORDER`.
4. Add the `elif` branch in `callbacks/__init__.py::render_tab()`.
5. Import and call `register` in `callbacks/__init__.py::register_all_callbacks()`.

### 18.6 Adding a new figure builder

1. Add the function to an existing module in `figures/` or create a new one.
2. Signature: `def my_chart(data: np.ndarray, ..., title: str = "") -> go.Figure`.
3. The global plotly template is applied automatically — do not set
   `paper_bgcolor` or `plot_bgcolor` in individual figures.
4. Return `go.Figure`, never `dcc.Graph` — the wrapping is done in callbacks.

### 18.7 Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Session not initialised` | `create_app()` wasn't called or `initialise()` failed | Check registry_dir/artifacts_dir paths |
| Blank tab content | Callback not registered | Verify the tab's `register(app)` is called in `callbacks/__init__.py` |
| `PreventUpdate` / stale data | Callback guards returning `no_update` | Check that the tab/sub-tab guard conditions match the component IDs |
| Slow first load of Evaluation | `GlobalPredictionStore` being built | Normal — subsequent loads are cached. ~1-2s for 100 clusters |
| Cytoscape graph too dense | Too many edges with low weight | Increase the edge weight threshold slider |

---

*End of implementation guide.*

