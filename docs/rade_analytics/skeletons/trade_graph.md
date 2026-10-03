# Skeleton — Evaluation › Trade Graph

> **Status:** Draft, awaiting sign-off.
> **Purpose:** Single source of truth for the layout, controls and data
> sources of the Evaluation › Trade Graph sub-tab.  Once approved, the
> layout file `src/ui/apps/rade_analytics/layouts/evaluation/trade_graph.py`
> must match this document section-for-section.  Any deviation requires
> updating this document first.
> **Mock:** `docs/platform_designs/rade_trade_graph.png`
> **URL:** `/evaluation/trade-graph`
> **Code:**
> * Layout — `src/ui/apps/rade_analytics/layouts/evaluation/trade_graph.py`
> * Callbacks — `src/ui/apps/rade_analytics/callbacks/trade_graph_cb.py`
> * Stylesheet — colocated with the layout file (`CYTOSCAPE_STYLESHEET`)

---

## 1. Page anatomy

The page lives inside the standard Evaluation chrome (top-bar, sidebar,
filter bar) and consists of three rows:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Row 1 · Header band   ──────────────────────────────────────────────── │
│  Cluster · Layout · Zoom · Fit view · Export PNG                        │
├─────────────────────────────────────────────────────────────────────────┤
│ Row 2 · Graph pane (2/3)        │ Side panel (1/3)                      │
│  ┌───────────────────────────┐  │  ┌─────────────────────────────────┐  │
│  │                           │  │  │ Selected Trade card             │  │
│  │   Cytoscape network       │  │  ├─────────────────────────────────┤  │
│  │   + mini-map overlay      │  │  │ Legend card                     │  │
│  │                           │  │  ├─────────────────────────────────┤  │
│  │                           │  │  │ Cluster stats card              │  │
│  │                           │  │  ├─────────────────────────────────┤  │
│  │                           │  │  │ Ensemble summary card           │  │
│  └───────────────────────────┘  │  └─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────┤
│ Row 3 · Secondary charts (split 50/50)                                  │
│  ┌──────────────────────────┐  ┌────────────────────────────────────┐  │
│  │ Density distribution     │  │ Edges vs nodes scatter             │  │
│  └──────────────────────────┘  └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Row 1 — Header band

Sticky band, `rade-card`, flex-wrap.  Order from left to right:

| # | Control | Type | Notes |
|---|---------|------|-------|
| 1 | **Cluster** | `dmc.Select` (searchable) | Populated from `/clusters`.  Drives the visible network. |
| 2 | **Layout** | `dmc.SegmentedControl` | Three options exactly: `Force` (value `cose`), `Concentric` (value `concentric`), `Circle` (value `circle`).  Default `Force`. |
| 3 | **Zoom** | `dcc.Slider` | Range `0.2 – 3.0`, default `1.0`, step `0.1`.  Drives the Cytoscape `zoom` prop via clientside callback. |
| 4 | **Fit view** | `dmc.Button` (outline, icon `tabler:focus-2`) | Calls `cy.fit()` clientside. |
| 5 | spacer | `flex-1` div | Pushes Export PNG to the right. |
| 6 | **Export PNG** | `dmc.Button` (outline, icon `tabler:download`) | Saves the current Cytoscape canvas as PNG. |

**Removed from header (don't add back):**
* ~~Min weight slider~~ — never referenced in the mock.
* ~~Color by: Residual dropdown~~ — explicitly excluded by the user.
* ~~`grid` / `breadthfirst` layout options~~ — not in the mock.

---

## 3. Row 2 — Graph pane (2/3 width)

### 3.1 Pane header strip
A single row inside the pane card:
* Left: `Trade network` (semibold, slate-200).
* Right: status text (`pane_status`) — populated by callbacks (e.g. `42 nodes · 187 edges · cluster_007`).

### 3.2 Cytoscape canvas
* Component: `cyto.Cytoscape`.
* Height: `560px` fixed.
* Default layout: `cose` (Force).
* Stylesheet (see §5).
* `minZoom=0.2`, `maxZoom=3.0`, `boxSelectionEnabled=False`.
* Listens on `tapNodeData` (node clicks).  Edge / background taps ignored.

**Callback contract:** the entire `cyto.Cytoscape(...)` component is
returned by the render callback (rebuild pattern, not prop updates), so
the layout pass re-runs on every cluster / layout / data change.  See
`callbacks/trade_graph_cb.py` § "render_graph".

### 3.3 Mini-map overlay
* Position: absolute, top-right inside the pane, `12px` from the edges.
* Size: `160 × 100 px`.
* Background: `rgba(15, 23, 42, 0.6)` with a `1px` slate-700 border.
* Renders the same elements as the main Cytoscape but with
  `userZoomingEnabled=False`, `userPanningEnabled=False`,
  `boxSelectionEnabled=False`.
* A semi-transparent rectangle ("viewport indicator") shows the area
  currently visible in the main canvas.  Updated via clientside
  callback on the main `cy` `viewport` event.

---

## 4. Row 2 — Side panel (1/3 width)

Four cards, in this exact order, top-to-bottom.  Stack with `gap-3`.

### 4.1 Selected Trade card

Empty state until a node is tapped.

| Element | Notes |
|---------|-------|
| Header — label `Selected trade` (uppercase, slate-400) + cluster chip (right-aligned, violet badge) | Cluster chip empty until a node is selected. |
| Trade ID row — `<code>` with the id, plus a copy `ActionIcon` | Truncate at `220px`. |
| **Attribute chips row** *(new)* | One `dmc.Badge` per attribute (e.g. `FX`, `GBPUSD`, `European Option`).  Source: per-trade attributes from the API.  Layout: `flex-wrap gap-1`. |
| **Top-K neighbours panel** *(new — see §6)* | Heading + segmented control (3/5/10) + adjacency list. |
| `Open in Cluster Deep Dive` button | Disabled until a trade is selected; navigates to `/evaluation/cluster` and writes `deep_dive_cluster_id` + `deep_dive_selected_trade_id` to session. |

### 4.2 Legend card

**Keep both elements** — the mock's gradient bar **and** the existing
anchored binary-swatch rows.

| Element | Notes |
|---------|-------|
| Heading `Legend` (uppercase, slate-400) | |
| **Edge-weight gradient bar** *(new)* | Horizontal bar, ~`12px` tall, full card width, gradient from `rgba(148,163,184,0.2)` (left) to `#8b5cf6` (right).  Two anchor labels under the bar: `weak link` (left, slate-500) ↔ `strong link` (right, slate-300). |
| Divider | `Divider` from dmc, slate-700. |
| Anchored swatch rows *(existing — keep)* | Two rows: amber circle `Target — Priced directly by the member`; violet circle `Elementary — Building block used in recalculation`. |

### 4.3 Cluster stats card *(unchanged from current implementation)*

| Element | Notes |
|---------|-------|
| Heading `Cluster stats` | |
| 2×2 KPI grid: `Nodes`, `Edges`, `Density`, `Mean weight` | Filled by the render callback from `/graph-stats?cluster_id=…`. |

### 4.4 Ensemble summary card *(unchanged from current implementation)*

| Element | Notes |
|---------|-------|
| Heading `Ensemble summary` | |
| 2×2 KPI grid: `Total clusters`, `Total edges`, `Avg density`, `Avg mean weight` | Filled by the render callback from `/graph-stats` aggregated across all clusters. |

---

## 5. Cytoscape stylesheet contract

Two **explicit** node selectors so a missing `trade_type` cannot
default-render as a target-coloured node.

```python
[
    {
        "selector": "node[trade_type = 'elementary']",
        "style": {
            "background-color": "#8b5cf6",   # violet
            "width": 12, "height": 12,
            "border-color": "#0f172a", "border-width": 1,
            "transition-property":  "background-color, width, height, border-color",
            "transition-duration":  "150ms",
        },
    },
    {
        "selector": "node[trade_type = 'target']",
        "style": {
            "background-color": "#f59e0b",   # amber
            "width": 18, "height": 18,
            "border-color": "#0f172a", "border-width": 1,
        },
    },
    {
        "selector": "node:selected",
        "style": {
            "border-color": "#10b981",       # emerald ring
            "border-width": 3,
            "width": 22, "height": 22,
        },
    },
    {
        "selector": "edge",
        "style": {
            "width":          "mapData(weight, 0, 1, 0.5, 3)",
            "line-color":     "rgba(148, 163, 184, 0.35)",
            "curve-style":    "haystack",
            "haystack-radius": 0.5,
        },
    },
]
```

Colour values must stay in sync with `figures/cluster_deep_dive_charts.py`
(the `_TRADE_TYPE_COLOR` dict) so the violin / scatter / Cytoscape
legend agree on what amber and violet mean.

---

## 6. Top-K neighbours panel (new feature)

Inside the Selected Trade card, between the attribute chips and the
"Open in Cluster Deep Dive" button.

### 6.1 Layout

| Element | Notes |
|---------|-------|
| Heading row | `Nearest neighbours` (left, slate-400, uppercase) + `dmc.SegmentedControl` (right) with options `3 / 5 / 10`, default `5`. |
| List | Up to *k* rows, sorted by edge `weight` descending.  Each row: `<code>{trade_id}</code>` (left) · weight badge (centre) · `trade_type` swatch (right, 8×8 px circle).  Hover background: slate-800/40.  Click row → select that node in the graph and update the Selected Trade card. |
| Empty state | When no node is selected: `Click a node in the graph to inspect the trade.` (italic, slate-500).  When a node is selected but has no neighbours: `No neighbours in the current cluster.` |

### 6.2 Data source

Adjacency for the selected node is derived from the existing trade-graph
payload (no new API endpoint needed).  In the render callback, store
`{ "<trade_id>": [(neighbour_id, weight, neighbour_trade_type), ...] }`
keyed by the active cluster, in the existing `store_nodes_edges`
`dcc.Store`.  The neighbour-list callback reads from this store; no
network round-trip on selection or on `k` change.

### 6.3 Cross-highlight behaviour

Clicking a neighbour row writes the new `selected_trade_id` to session
and triggers the same callback chain as a direct node tap.  The
emerald ring style (`node:selected`) handles the visual highlight.

---

## 7. Row 3 — Secondary charts *(unchanged from current implementation)*

Two charts, 50/50 split, `ChartContainer` wrappers, both `280px` tall.

| Chart | Subtitle | ID |
|-------|----------|----|
| `Density distribution` | "Per cluster, selected cluster highlighted" | `density_chart` |
| `Edges vs nodes` | "Marker size = density · colour = mean weight" | `edges_vs_nodes_chart` |

---

## 8. State / session contract

### 8.1 Session fields touched by this page

| Field | Type | Written by | Read by |
|-------|------|-----------|---------|
| `evaluation.trade_graph_cluster_id` | `Optional[str]` | Header cluster select | Render callback |
| `evaluation.trade_graph_layout` | `"cose" \| "concentric" \| "circle"` | Header layout radio | Render callback |
| `evaluation.trade_graph_zoom` *(new)* | `float` (0.2 – 3.0) | Header zoom slider | Clientside `cy.zoom()` call |
| `evaluation.trade_graph_selected_trade_id` | `Optional[str]` | `tapNodeData` + neighbour-row click | Selected Trade card, neighbour panel |
| `evaluation.trade_graph_neighbour_k` *(new)* | `Literal[3, 5, 10]` | Neighbour panel segmented control | Neighbour-list callback |

### 8.2 Session fields removed by this page *(schema bump)*

| Field | Reason |
|-------|--------|
| `evaluation.trade_graph_weight_threshold` | Min-weight slider removed from header. |

### 8.3 Schema version

`SESSION_SCHEMA_VERSION` bumps from `6` → `7`.  `from_dict` drops
`trade_graph_weight_threshold` if present in a stale payload.
`EVALUATION_TRADE_GRAPH_LAYOUTS` shrinks to `("cose", "concentric", "circle")`.

---

## 9. Explicitly NOT on this page

To avoid future regressions, things that **must not** be added to this
page without first updating this document:

* **Color by: Residual dropdown** — user explicitly excluded.
* **Min weight slider** — was a previous-implementation artefact, not in
  the mock.
* **`grid` / `breadthfirst` layout options** — not in the mock.
* **Provenance footer line** — explicitly excluded.
* **Any chart on the right column of Row 2** — the side panel is cards
  only.

---

## 10. Open questions

### 10.1 `trade_type` classification

**How it worked in the previous dashboard (`ensemble_analytics`)**
The Dash callback loaded `trade_universe.json` directly off disk for
the selected cluster, built `target_set = set(trade_universe["target_ids"])`,
and for every trade id stamped `data.trade_type = "target" if id in target_set else "elementary"`.
Stylesheet selectors `node[trade_type='target']` and
`node[trade_type='elementary']` then coloured the nodes.

**How it works in `rade_analytics`**
Identical logic, moved one layer up — into the FastAPI service
(`src/rade_ml_pt/ensemble/api/services/reader.py::trade_graph`).
The API reads `members/{cluster_id}/trade_universe.json`, classifies
each node, and returns nodes already typed (`node.trade_type`).  The
UI just consumes that field and writes it into the Cytoscape element's
`data` dict.

**The API already handles missing data defensively**
* If `trade_universe.json` is missing → all nodes classified as
  `elementary`, a warning is added to `response.warnings`.
* If the file is present but `target_ids` is empty → same effect.
* If a node index is out of range → synthetic id, classified as
  `elementary`, warning appended.

**Three failure modes to distinguish**
The "elementaries invisible / same colour as targets" bug must be one of:

| # | Cause | Symptom in API | Symptom in UI |
|---|-------|----------------|---------------|
| 1 | `trade_universe.json` missing for active version's clusters | Every cluster's response carries the "missing" warning; every node has `trade_type='elementary'` | All nodes violet, no amber targets, no error |
| 2 | File exists but `target_ids = []` | No warning, but every node has `trade_type='elementary'` | All nodes violet, no warning surfaced |
| 3 | API returns `trade_type` correctly, UI drops it on the way to Cytoscape `data` | API response carries mixed `trade_type`s | Stylesheet selectors don't match → every node renders with the default style |

**Probe** — run before implementing §5

```python
# scripts/probe_trade_universe.py
from pathlib import Path
import json, joblib
from src.rade_ml_pt.ensemble.config import EnsembleConfig

cfg = EnsembleConfig.load()                          # picks active version
members_dir = Path(cfg.eval_artifacts_dir) / "members"

print(f"Scanning {members_dir}")
for cluster_dir in sorted(members_dir.glob("cluster_*")):
    cid = cluster_dir.name
    tu_path = cluster_dir / "trade_universe.json"
    gr_path = cluster_dir / "graph_results.joblib"
    if not tu_path.exists():
        print(f"  {cid:24s}  trade_universe.json MISSING")
        continue
    tu = json.loads(tu_path.read_text())
    n_t = len(tu.get("target_ids") or [])
    n_e = len(tu.get("elementary_ids") or [])
    n_g = 0
    if gr_path.exists():
        n_g = int(joblib.load(gr_path).get("sparse_shape", [0])[0])
    flag = "ok" if n_t > 0 and n_e > 0 else "EMPTY"
    print(f"  {cid:24s}  targets={n_t:4d}  elementaries={n_e:5d}  graph_n={n_g:5d}  [{flag}]")
```

Output decides which fix is needed:

* All clusters print `MISSING` → fix is **upstream** (re-run the eval
  pipeline so `_copy_member_graph_artifacts` stages
  `trade_universe.json`).
* Some clusters print `targets=0` (or only `targets=0`) → fix is
  **upstream in the eval pipeline** (the trade-universe writer isn't
  populating `target_ids`).
* Every cluster prints `ok` → fix is **in the UI's element builder**;
  add a unit test asserting `cyto_element["data"]["trade_type"]` is
  preserved end-to-end from the API response.

The doc cannot guess which of the three; the probe answers it
unambiguously in <30s.

### 10.2 Mini-map plugin

`cytoscape-navigator` is a Cytoscape.js plugin; `dash_cytoscape`
exposes only a subset.  Spike before §3.3 ships:

* If `dash_cytoscape >= 1.0` ships the navigator extension natively,
  use it.
* Otherwise fall back to a small synced second `cyto.Cytoscape` with
  `userZoomingEnabled=False`, `userPanningEnabled=False`,
  `boxSelectionEnabled=False`, listening on the main canvas's
  `viewport` event via clientside callback to draw the viewport
  rectangle.

---

## 11. Sign-off

| Role | Name | Date |
|------|------|------|
| Owner (design) | Joe | _pending_ |
| Implementer (AI) | — | _pending_ |
