# Rade UI Design Spec

Short, living spec for the Rade Dash UI. The **mock images in this folder
(`rade_*.png`) are the design contract** — every new tab, card or callback
must visually and functionally align with one of them. When the mocks and
the code disagree, update one of them deliberately, never silently.

---

## 1. Brand

| Token | Value |
|---|---|
| Name | **Rade** |
| Tagline | Quantitative Model Intelligence |
| Logomark | Violet → cyan geometric "R" prism |
| Wordmark | Inter / Geist, weight 600, tracking -0.02em |
| Tone | Quietly confident, low-ornament, numerically dense. Linear / Vercel / Raycast school. |

---

## 2. Palette

Dark-only for v1. All tokens are Tailwind defaults — no bespoke colours.

| Role | Tailwind | Hex |
|---|---|---|
| Background | `slate-950` | `#020617` |
| Surface / card | `slate-900` | `#0f172a` |
| Sunken / track | `slate-950` inside card | `#020617` |
| Border | `slate-800` | `#1e293b` |
| Border subtle | `slate-800/60` | rgba |
| Text primary | `slate-100` | `#f1f5f9` |
| Text secondary | `slate-400` | `#94a3b8` |
| Text muted | `slate-500` | `#64748b` |
| Brand primary | `violet-500` | `#8b5cf6` |
| Brand accent | `cyan-400` | `#22d3ee` |
| Brand gradient | `from-violet-500 to-cyan-400` | — |
| Success | `emerald-500` | `#10b981` |
| Warning | `amber-500` | `#f59e0b` |
| Danger | `rose-500` | `#f43f5e` |
| Info | `sky-400` | `#38bdf8` |

**Rule:** Brand gradient is reserved for primary CTAs, selected nav
indicators, and the splash logomark. Do not apply it to card backgrounds
or chart fills — use 20–30% opacity of the violet for gradient chart fills.

---

## 3. Typography

- **Font**: Inter (fallback: system). Geist is acceptable for brand assets only.
- **Numeric font**: `JetBrains Mono` / `IBM Plex Mono` in tables and diff views
  (Tailwind `font-mono`).
- **Scale**:

| Usage | Size | Weight | Tracking |
|---|---|---|---|
| Page title | 28–36px | 600 | -0.01em |
| Section title | 18–20px | 600 | 0 |
| Card title | 14–15px | 600 | 0 |
| Body | 13–14px | 400 | 0 |
| Muted / caption | 11–12px | 400 | 0.01em |
| Code / numeric | 13px mono | 400 | 0 |

---

## 4. Layout grid

- **Viewport floor**: 1280px. Responsive down to 1024px (sidebar collapses).
- **Sidebar**: 220px wide, `bg-slate-900`, `border-r border-slate-800`.
- **Content padding**: `px-8 py-6` on the main area.
- **Card padding**: `p-5` (default), `p-4` (compact tiles).
- **Gap**: `gap-4` within a card, `gap-6` between cards.
- **Rounded**: `rounded-2xl` on cards, `rounded-xl` on inputs, `rounded-lg` on
  small chips.
- **Shadow**: none by default, `shadow-sm` max. No glow on real content.
  Glow is reserved for: splash logo, selected graph node, ai-assistant
  accent ring.

---

## 5. Page skeleton

Every tab follows the same top-down rhythm — see `rade_landing_dashboard.png`
and `rade_eval_portfolio.png`:

1. **Breadcrumb** (slate-400, 12–13px) — `Section › Sub-section › Tab`.
2. **Title row** — H1 + subtitle + right-aligned meta (version chip, bell,
   search).
3. **Filter bar** (optional) — on a `bg-slate-900 rounded-2xl` strip.
4. **KPI tile row** — 3–6 equal-width cards.
5. **Main content grid** — 1–2–3 column layouts; charts left-weighted, tables
   right-weighted.
6. **Footer strip** — source artifact caption, export actions.

---

## 6. Component library

| Concept | Dash component | Notes |
|---|---|---|
| App shell | `dash-mantine-components.AppShell` | `padding="md"`, custom sidebar. |
| Navigation | `html.A` + Tailwind | Selected = `bg-slate-800` + `border-l-2 border-violet-500`. |
| Breadcrumb | `html.Div` + chevron icons | `dash-iconify` for icons. |
| Segmented control | `dmc.SegmentedControl` | Default style, brand color on active. |
| Dropdown | `dcc.Dropdown` | Theme overridden via `className`. |
| Multi-select | `dmc.MultiSelect` | For cluster/trade pickers. |
| Range slider | `dmc.RangeSlider` | For scenario windows. |
| Button primary | `dmc.Button` with gradient | Variant `gradient`, from violet-500 to cyan-400. |
| Button outline | `dmc.Button variant="outline"` | |
| Pill / chip | `dmc.Badge` | With per-status color. |
| Metric tile | `html.Div` + Tailwind | `rounded-2xl bg-slate-900 border border-slate-800 p-5`. |
| Chart | `dcc.Graph` (Plotly) | Shared dark template — see §7. |
| Table | `dash-ag-grid` | Shared theme config — see §8. |
| Network graph | `dash-cytoscape` | Layout `cose-bilkent`. |
| File upload | `dcc.Upload` | Used in Inference Console. |
| Timeline | `dmc.Timeline` | Governance lineage. |
| Drag canvas | `dash-draggable` | Report Builder only. |
| Command palette | `dmc.Spotlight` | Cmd+K globally. |
| Modal | `dmc.Modal` | Approvals, confirmations. |
| Skeleton | `dmc.Skeleton` | Loading state for every card. |
| Alert | `dmc.Alert` | Error / warning banners. |
| Toast | `dmc.Notification` | Save/export confirmations. |

---

## 7. Chart defaults (Plotly)

One shared template `RADE_DARK` applied via `go.Figure(layout=...)` or a
helper `rade_figure()`:

- `template="plotly_dark"` as base, then override:
- `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"` (transparent; card shows through).
- `font=dict(family="Inter, sans-serif", size=12, color="#94a3b8")`.
- Gridlines: `gridcolor="rgba(148,163,184,0.1)"`, `zerolinecolor` same.
- Axis ticks: `tickcolor="rgba(148,163,184,0.2)"`, `ticks="outside"`, `tickfont.color="#94a3b8"`.
- Margin: `dict(l=40, r=20, t=30, b=40)`.
- Legend: top-right, horizontal, chip-style (`bgcolor="rgba(15,23,42,0.6)"`, `bordercolor="#1e293b"`).
- Hovermode: `"x unified"` for time series, `"closest"` otherwise.
- **Primary series**: `violet-500`. **Secondary**: `cyan-400`. **Reference /
  actual**: dashed `slate-400`. **Residual band**: `rgba(139,92,246,0.18)` fill.
- Colormaps:
  - Residual heatmap: `[[0,"#7c3aed"],[0.5,"#334155"],[1,"#f43f5e"]]`.
  - Correlation heatmap: `"RdBu_r"` centred at 0.
  - Completeness heatmap: `[[0,"#f43f5e"],[0.5,"#334155"],[1,"#7c3aed"]]`.
  - Sequential: Plotly `Viridis` (keep for simple magnitudes).

All chart code lives under `ui/components/charts/` so the template is
enforced centrally.

---

## 8. Table defaults (`dash-ag-grid`)

Single shared `RADE_GRID_DEFAULTS` dict:

```python
RADE_GRID_DEFAULTS = dict(
    className="ag-theme-alpine-dark rade-grid",
    defaultColDef={
        "sortable": True,
        "filter": True,
        "resizable": True,
        "cellClass": "text-slate-200",
    },
    dashGridOptions={
        "rowHeight": 36,
        "headerHeight": 40,
        "suppressMovableColumns": False,
        "animateRows": True,
    },
)
```

- Numeric columns: `type="numericColumn"`, right-aligned, `font-mono`.
- Status pills: `cellRenderer` returning `dmc.Badge`.
- Inline bars: custom `cellRenderer` using a Tailwind-styled `<div>` with
  width proportional to value; bar colour follows status (`bg-rose-500/30`
  for breaches, `bg-emerald-500/30` for healthy).
- Row stripes: `ag-row-odd` = `bg-slate-950/40`.

---

## 9. State reference

Every card must handle the 6 states shown in `rade_states_reference.png`:

1. **Loading** — `dmc.Skeleton` matching the final layout's shape.
2. **Empty (no data)** — outlined icon + heading + body + primary action.
3. **Empty (filter too narrow)** — funnel icon + "Reset filters" link.
4. **Error (404 artifact)** — specific, names the missing file, lists
   available alternatives.
5. **Error (backend down)** — cites the `/health` status, offers Retry.
6. **Authz denied** — names the restricted resource, offers access request.

Do not ship a card without at least the Loading and one Empty state.

---

## 10. Accessibility

- All interactive elements reachable by keyboard; focus ring =
  `ring-2 ring-violet-500 ring-offset-2 ring-offset-slate-950`.
- Colour contrast ≥ 4.5:1 for body text against `slate-900` (checked: all
  palette entries pass).
- Charts ship an accompanying data table under a "View raw data" disclosure
  for screen-reader access.
- Status colour is never the only signal — pair with icon + label.
- Cmd+K command palette exposes every navigable destination.

---

## 11. Design contract — the 20 mocks

| # | File | Purpose | Data source(s) |
|---|---|---|---|
| 1 | `rade_splash.png` | Boot / loading | `/health`, `/versions` |
| 2 | `rade_login.png` | SSO sign-in | — |
| 3 | `rade_landing_dashboard.png` | Home / overview | `/overview`, `/metrics/ensemble`, `/portfolio`, `/clusters` |
| 4 | `rade_eval_portfolio.png` | Evaluation — portfolio split | `/portfolio`, `/metrics/ensemble` |
| 5 | `rade_eval_by_desk.png` | Evaluation grouped by attribute | `/metrics/per-member`, `/clusters` |
| 6 | `rade_cluster_deep_dive.png` | Per-cluster drill-down | `/clusters`, `/metrics/per-member`, `/cluster-timeseries`, `/trades`, `/graph-stats` |
| 7 | `rade_cross_cluster.png` | Correlation matrix + group stats | `/group-correlations`, `/clusters` |
| 8 | `rade_trade_graph.png` | Network graph | `/graph-stats`, `/trades` |
| 9 | `rade_data_quality.png` | Completeness + feature summary | `/quality/completeness`, `/quality/feature-summary` |
| 10 | `rade_model_monitoring.png` | Live drift + alerts | new `/monitoring/*` (future) |
| 11 | `rade_training_curves.png` | Per-cluster training traces | `training_curves.parquet` (served via `/training/curves`) |
| 12 | `rade_prediction_explorer.png` | Raw trade × scenario NPZ drill-in | `/predictions`, `/trades` |
| 13 | `rade_governance.png` | Registry + lineage + approvals | `/versions`, new `/governance/*` |
| 14 | `rade_version_comparison.png` | Side-by-side diff of two versions | compose any `/metrics/*` + `/portfolio` for two versions |
| 15 | `rade_inference.png` | Run ensemble on new trades | new `/inference/run` (future) |
| 16 | `rade_scenario_lab.png` | Synthetic scenario builder + score | new `/scenarios/*` (future) |
| 17 | `rade_report_builder.png` | Block-canvas report authoring | reads anything above, emits PDF / HTML |
| 18 | `rade_ai_assistant.png` | Anomaly investigation side-panel | LLM backend + tool calls into the rest of the API |
| 19 | `rade_command_palette.png` | Global cmd+K navigation | static + `/overview` |
| 20 | `rade_states_reference.png` | Loading / empty / error atlas | — |

---

## 12. Extension process

When a new tab or card is proposed:

1. Draft or regenerate a mock under `docs/platform_designs/rade_<name>.png`.
2. Append a row to §11 table naming its data source and required endpoints.
3. Check charts/tables against §7 / §8 defaults — add to shared helpers if
   a new pattern emerges.
4. Add state coverage (§9) to the implementation PR checklist.

No mock, no merge.

---

## Appendix A — `ensemble/infer.py` (full refactored source, copy-paste friendly)

> **Status**: Lean Plan implementation complete in this env. This appendix
> contains the complete source of `src/rade_ml_pt/pipelines/ensemble/infer.py`
> after the staged-pipeline + Tier-2 optimisation refactor, split into copy-
> friendly sections. Drop each section into the corresponding region of
> your work-env file, in order.
>
> The Phase 1 work (event hook + `RadeBackend.run_inference` wrapper) is
> still in effect — the new staged pipeline emits exactly the same set
> of lifecycle events, plus a few new ones for the per-stage envelope.
> The previous Phase 1 appendix has been replaced by this one; the
> companion long-form spec
> [`docs/rade_analytics/inference_implementation.md`](../rade_analytics/inference_implementation.md)
> remains the source of truth for the event protocol and Phase 2 UI
> wiring.
>
> Companion porting tracker:
> [`docs/rade_analytics/ensemble_infer_refactor.md`](../rade_analytics/ensemble_infer_refactor.md).

---

### Appendix A · 0 — Reading guide

**File layout** (single file, 1366 lines):

| Section | Lines | What it contains |
|---|---|---|
| §1 | 1–44 | Module docstring + imports |
| §2 | 47–84 | `_dict_to_inference_context` (the one module-level helper) |
| §3 | 87–315 | Three frozen dataclasses: `LoadedScenariosReport`, `ClusterRoutingDecision`, `ValidationReport` |
| §4 | 318–381 | `EnsembleInferencePipeline.__init__` (contract preserved) |
| §5 | 383–810 | Orchestration: `run` / `load` / `load_scenarios` / `validate` / `run_inference` / `_run_with_prebuilt` |
| §6 | 812–907 | Loading helpers: `_load_from_registry`, `_load_from_session` |
| §7 | 909–1046 | Validation helpers: four cheap, private, no-IO methods |
| §8 | 1048–1269 | Input building: dispatcher → routing dispatcher → per-cluster builders (full / cheap) |
| §9 | 1271–1366 | Post-inference + `_build_result` |

**Contracts preserved** (identical to pre-refactor signatures — *do not change these*):

- `EnsembleInferencePipeline.__init__(ensemble_config, ensemble_version="latest", session=None, *, on_event=None)`
- `EnsembleInferencePipeline.run() -> InferenceResult`

The `__init__` adds three new private cache slots (`_new_scenario_shocks`,
`_loaded_scenarios`, `_validation_report`) initialised to `None`; existing
public attributes (`config`, `ensemble_version`, `_session`, `_ensemble`,
`_ens_config`, `_member_versions`, `_inference_contexts`, `_emit`) stay
exactly as before.

**Two flow paths** that `run()` dispatches between:

1. **Pre-built short-circuit** — when
   `config.metadata["inference"]["member_inputs"]` is populated, skip
   scenario loading + validation and call `_run_with_prebuilt`. This is the
   legacy path used by smoke tests, model-agnostic harnesses, and the
   `new_trades` mode (until the staged `new_trades` builder is implemented).
2. **Staged path** (default for production new-scenarios runs) —
   `load() → load_scenarios() → validate() → run_inference()`. The UI
   workflow drives these four stages directly via individual buttons;
   programmatic callers can either call them in sequence or call `run()`
   to orchestrate.

**Tier-2 optimisation**: inside `run_inference()` the per-cluster input
build dispatches on `ClusterRoutingDecision.is_affected`:

- **Affected cluster** (cluster RFs ∩ shock RFs ≠ ∅) → `_build_cluster_inputs_full`
  re-prices elementary PnL under new shocks (the expensive path).
- **Unaffected cluster** → `_build_cluster_inputs_cheap` looks up
  pre-scaled, pre-reduced historical PnL via
  `ctx.elementary_pnl.loc[scenario_labels]` (the cheap path; eligibility
  is validated up-front by `validate()`).

Both paths land on the same downstream pipeline
(`build_new_pnl_sequences` + `build_model_inputs`), so the model sees
identical input shapes for affected and unaffected members.

**State machine** (implicit, no enum):

```
__init__   →   _ensemble = None,  _loaded_scenarios = None,  _validation_report = None
   ↓ load()
_ensemble  ↑
   ↓ load_scenarios()
_new_scenario_shocks ↑,  _loaded_scenarios ↑,  _validation_report = None  (reset)
   ↓ validate()
_validation_report ↑
   ↓ run_inference()
returns InferenceResult
```

Each public stage checks its prerequisites at the top and raises
`RuntimeError` with an actionable message when a prior stage is missing.

---

### Appendix A · 1 — Module docstring + imports

Header of the file (lines 1–44). All imports are alphabetised within
their group; the `TYPE_CHECKING` block holds forward references that
would otherwise cause circular imports (`EnsembleSession` and
`InferenceContext` both import from this module transitively).

```python
"""
Ensemble inference pipeline.

Two usage patterns:

1. **Standalone (non-UI):** Construct with an ``EnsembleConfig``, call ``run()``.
   The pipeline loads the ensemble from the registry, builds per-member inputs
   via the single-cluster inference helpers, predicts, and aggregates.

2. **Via EnsembleSession (UI):** The session pre-loads models + inference
   contexts.  Call ``run(session=session)`` or use ``EnsembleSession.run_inference()``
   directly.  The pipeline skips registry loading and uses the cached state.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.rade_ml_pt.core.types import InferenceResult
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    EmitFn,
    STAGE_INFERENCE,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_RUNNING,
    event,
    noop_emit,
)

if TYPE_CHECKING:
    from src.rade_ml_pt.ensemble.session import EnsembleSession
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

logger = logging.getLogger(__name__)
```

---

### Appendix A · 2 — Helper: `_dict_to_inference_context`

The only module-level helper (lines 47–84). Converts the loose dict
returned by `load_inference_context_from_dir` into the typed
`InferenceContext` dataclass that the rest of the pipeline operates on.
Reused by both `_load_from_registry` (cold start) and `_load_from_session`
(warm start).

```python
# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dict_to_inference_context(
    raw: Dict[str, Any],
    data_config_override: Any = None,
) -> "InferenceContext":
    """
    Convert a raw context dict (as returned by ``load_inference_context_from_dir``)
    into an ``InferenceContext`` dataclass.

    Parameters
    ----------
    raw : dict
        Keys match those produced by ``load_inference_context_from_dir``.
    data_config_override : optional
        If provided, replaces the ``data_config`` from *raw*.
    """
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

    dc = data_config_override if data_config_override is not None else raw.get("data_config")
    return InferenceContext(
        data_config=dc,
        encoder=raw.get("encoder"),
        encoder_results=raw.get("encoder_results"),
        graph_builder=raw.get("graph_builder"),
        graph_results=raw.get("graph_results"),
        elementary_pnl=raw.get("elementary_pnl"),
        elementary_scaler=raw.get("elementary_scaler"),
        elementary_attributes=raw.get("elementary_attribs"),
        target_scaler=raw.get("target_scaler"),
        target_attributes=raw.get("target_attribs"),
        trade_universe=raw.get("trade_universe"),
        cluster_info=raw.get("cluster_info"),
        cluster_assets=raw.get("cluster_assets"),
        cluster_elem_trades=raw.get("cluster_elem_trades"),
    )
```

---

### Appendix A · 3 — Stage report dataclasses

Three module-level frozen dataclasses (lines 87–315). One per pipeline
stage (`load_scenarios` returns `LoadedScenariosReport`; `validate`
returns `ValidationReport`, which contains a list of
`ClusterRoutingDecision`). All are **intentionally small and JSON-
serialisable** so the UI can store them in `dcc.Store` and pass them over
HTTP. Heavy artefacts (shock DataFrames, ensemble model, predictions
array) stay on the pipeline instance.

#### Appendix A · 3.1 — `LoadedScenariosReport`

What `load_scenarios()` parsed from the new-scenario directory. Used by
the UI's *manifest preview* card and by `validate()` downstream.

```python
# ------------------------------------------------------------------
# Stage reports
# ------------------------------------------------------------------
#
# The pipeline returns one frozen dataclass per stage, used by both
# programmatic callers and the UI activity log.  Heavy artefacts
# (shock DataFrames, ensemble model, predictions array) stay inside
# the pipeline; the reports below are intentionally small and JSON-
# serialisable so they can live in a ``dcc.Store`` or HTTP response.
#

@dataclass(frozen=True)
class LoadedScenariosReport:
    """Summary of what :meth:`EnsembleInferencePipeline.load_scenarios`
    parsed from the new-scenario directory.

    Attributes
    ----------
    new_scenario_dir
        Absolute or relative path to the folder that was parsed.
    risk_factor_names
        Risk-factor stems (filename minus ``.csv``).  Order is the
        order ``os.listdir`` returned them — not significant.
    n_risk_factors
        Convenience: ``len(risk_factor_names)``.  Stored for cheap
        UI consumption.
    n_scenarios
        Number of rows in every shock CSV.  Validated equal across
        files at parse time.
    scenario_labels
        The canonical scenario index, shared across all shock files
        in this run.  Used by :meth:`load_scenarios` /
        :meth:`validate` to look up rows in each cluster's
        ``elementary_pnl`` history.

    Notes
    -----
    The actual shock dictionaries — i.e. the heavy
    ``Dict[rf_name, Dict[scenario_label, Dict[knot, value]]]``
    parsed from disk — live on the pipeline as
    ``self._new_scenario_shocks`` and are *not* exposed via this
    report.  UI code consumes only the lightweight summary.
    """
    new_scenario_dir:    str
    risk_factor_names:   List[str]
    n_risk_factors:      int
    n_scenarios:         int
    scenario_labels:     List[str]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view (defensive copies of list fields)."""
        return {
            "new_scenario_dir":  self.new_scenario_dir,
            "risk_factor_names": list(self.risk_factor_names),
            "n_risk_factors":    self.n_risk_factors,
            "n_scenarios":       self.n_scenarios,
            "scenario_labels":   list(self.scenario_labels),
        }
```

#### Appendix A · 3.2 — `ClusterRoutingDecision`

One per cluster. Drives the per-cluster branch inside `run_inference()`
(affected → full path; unaffected → cheap path) and gives the UI's
routing-table card everything it needs without further context lookups.

```python
@dataclass(frozen=True)
class ClusterRoutingDecision:
    """Per-cluster decision made by :meth:`EnsembleInferencePipeline.validate`.

    Drives the per-cluster branch inside
    :meth:`EnsembleInferencePipeline.run_inference`: affected clusters
    take the full price-and-predict path
    (``_build_cluster_inputs``); unaffected clusters take the cheap
    historical-lookup path (``_load_cluster_inputs``).

    Attributes
    ----------
    cluster_id
        The cluster this decision is about.
    is_affected
        True iff the cluster shares at least one risk factor with the
        loaded scenario shocks.
    intersecting_risk_factors
        Sorted list of RF names that are in both the cluster's
        instrument universe and ``new_scenario_shocks``.  Empty when
        ``is_affected`` is False.
    n_elementary_trades
        Cluster's reduced elementary-trade count (post dimensionality
        reduction).  Surfaced here so the UI can render trade counts
        in the routing table without re-loading contexts.
    n_target_trades
        Cluster's target-trade count.  Same UI-convenience rationale.
    missing_scenario_labels
        Populated by ``validate()`` *only* for unaffected clusters
        whose history doesn't cover all requested scenario labels.
        Affected clusters always have ``[]`` here (they take the full
        re-pricing path, so historical-label lookup never happens).
        Non-empty on an unaffected cluster ⇒ cheap path blocked ⇒
        :class:`ValidationReport` gets a corresponding entry in
        ``errors``, flipping ``is_valid`` to False.
    """
    cluster_id:                  str
    is_affected:                 bool
    intersecting_risk_factors:   List[str]
    n_elementary_trades:         int
    n_target_trades:             int
    missing_scenario_labels:     List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view (defensive copies of list fields)."""
        return {
            "cluster_id":                self.cluster_id,
            "is_affected":               self.is_affected,
            "intersecting_risk_factors": list(self.intersecting_risk_factors),
            "n_elementary_trades":       self.n_elementary_trades,
            "n_target_trades":           self.n_target_trades,
            "missing_scenario_labels":   list(self.missing_scenario_labels),
        }
```

#### Appendix A · 3.3 — `ValidationReport`

Output of `validate()`. The pipeline never raises on a validation issue —
the caller (programmatic or UI) decides whether to proceed. `errors`
block the run (`is_valid` is False); `warnings` are surfaced but don't
block. Derived properties (`affected_cluster_ids`, `cheap_path_used`,
etc.) ensure there's a single source of truth — the `cluster_decisions`
list.

```python
@dataclass(frozen=True)
class ValidationReport:
    """Output of :meth:`EnsembleInferencePipeline.validate`.

    Two kinds of issues are surfaced separately:

    * **errors** — *block the run*.  ``is_valid`` is False whenever
      this list is non-empty.  Typical errors: scenario labels not
      found in an unaffected cluster's history; no cluster intersects
      any shocked RF (nothing to run).
    * **warnings** — surfaced to the user but **do not** block the
      run.  e.g. a single cluster intersects shocks, shock magnitudes
      look small.

    The pipeline itself never raises on a validation issue.  Those
    are user-input problems that should be fixable by editing scenario
    files or selecting a different ensemble version; the caller
    (programmatic, backend, or UI) decides what to do with the report.

    Attributes
    ----------
    ensemble_version
        The resolved ensemble version this report was built against.
    n_scenarios
        Number of scenarios in the loaded shock files.
    scenario_labels
        Canonical scenario index used during validation.
    cluster_decisions
        One :class:`ClusterRoutingDecision` per cluster in the
        ensemble, in the order the ensemble's ``cluster_ids`` lists
        them.
    errors, warnings
        Validation issues — see notes above.

    Properties
    ----------
    is_valid
        ``len(errors) == 0``.
    affected_cluster_ids, unaffected_cluster_ids
        Derived from ``cluster_decisions``.  Single source of truth
        (these cannot drift from ``cluster_decisions``).
    affected_count, unaffected_count, cheap_path_used
        Convenience counters / boolean for UI display.
    """
    ensemble_version:    str
    n_scenarios:         int
    scenario_labels:     List[str]
    cluster_decisions:   List[ClusterRoutingDecision] = field(default_factory=list)
    errors:              List[str]                    = field(default_factory=list)
    warnings:            List[str]                    = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True iff no blocking errors were recorded."""
        return len(self.errors) == 0

    @property
    def affected_cluster_ids(self) -> List[str]:
        """Cluster IDs that intersect with at least one shocked RF."""
        return [d.cluster_id for d in self.cluster_decisions if d.is_affected]

    @property
    def unaffected_cluster_ids(self) -> List[str]:
        """Cluster IDs with no shocked RF — eligible for the cheap path."""
        return [d.cluster_id for d in self.cluster_decisions if not d.is_affected]

    @property
    def affected_count(self) -> int:
        return len(self.affected_cluster_ids)

    @property
    def unaffected_count(self) -> int:
        return len(self.unaffected_cluster_ids)

    @property
    def cheap_path_used(self) -> bool:
        """True iff any cluster will take the cheap (historical-lookup) path."""
        return self.unaffected_count > 0

    def decision_for(
        self, cluster_id: str,
    ) -> Optional[ClusterRoutingDecision]:
        """Look up the routing decision for one cluster (linear scan).

        Returns ``None`` if the cluster ID isn't in the report — which
        should never happen for an ensemble member the pipeline knows
        about, but the method is defensive.
        """
        for d in self.cluster_decisions:
            if d.cluster_id == cluster_id:
                return d
        return None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view, including derived properties.

        The derived fields (``is_valid``, ``affected_cluster_ids``
        etc.) are included so UI code can render the report without
        having to re-derive them.
        """
        return {
            "ensemble_version":       self.ensemble_version,
            "n_scenarios":            self.n_scenarios,
            "scenario_labels":        list(self.scenario_labels),
            "cluster_decisions":      [d.to_dict() for d in self.cluster_decisions],
            "errors":                 list(self.errors),
            "warnings":               list(self.warnings),
            "is_valid":               self.is_valid,
            "affected_cluster_ids":   self.affected_cluster_ids,
            "unaffected_cluster_ids": self.unaffected_cluster_ids,
            "affected_count":         self.affected_count,
            "unaffected_count":       self.unaffected_count,
            "cheap_path_used":        self.cheap_path_used,
        }
```

---

### Appendix A · 4 — `EnsembleInferencePipeline.__init__`

Class header and constructor (lines 318–381). **The `__init__` signature
is unchanged from pre-refactor** — three new private cache slots
(`_new_scenario_shocks`, `_loaded_scenarios`, `_validation_report`) are
added at the bottom of the constructor body. These three slots are
populated by `load_scenarios()` / `validate()`; the pre-built short-
circuit in `run()` leaves them as `None` (and the result builder reads
this to decide which metadata keys to add).

```python
# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class EnsembleInferencePipeline:
    """
    Run inference through an ensemble of models.

    Supports two input modes (set in config.metadata["inference"]["input_mode"]):
    - ``new_scenarios``: Same trades, new risk-factor scenario data.
    - ``new_trades``: New trade attributes to route and predict (not yet implemented).

    Parameters
    ----------
    ensemble_config : EnsembleConfig
        Must have ``registry_dir`` set (ignored when *session* is provided).
    ensemble_version : str
        Ensemble version or tag to load.
    session : EnsembleSession or None
        If provided, skip registry loading and use the session's cached
        models + inference contexts.  The session must have Phase 3 loaded.
    """

    def __init__(
        self,
        ensemble_config: EnsembleConfig,
        ensemble_version: str = "latest",
        session: Optional["EnsembleSession"] = None,
        *,
        on_event: Optional[EmitFn] = None,
    ) -> None:
        """Initialise the pipeline.

        Parameters
        ----------
        ensemble_config, ensemble_version, session
            Existing arguments — unchanged behaviour.
        on_event
            Optional lifecycle-event callback.  When ``None``
            (default), events are silenced via :data:`noop_emit` and
            the pipeline behaves exactly as it did before this hook
            was added — so all existing call-sites (CLI, tests,
            batch) require no code change.

            Pass an :data:`EmitFn` (e.g. an
            :class:`~src.rade_ml_pt.pipelines.ensemble.infer_events.EventCollector`
            instance) to capture a streaming narration of the run
            for the *Inference Console* activity log.
        """
        self.config = ensemble_config
        self.ensemble_version = ensemble_version
        self._session = session
        self._emit: EmitFn = on_event if on_event is not None else noop_emit

        self._ensemble = None
        self._ens_config: Optional[EnsembleConfig] = None
        self._member_versions: Optional[Dict[str, str]] = None
        self._inference_contexts: Dict[str, Any] = {}

        # Staged-flow cache: populated by load_scenarios() / validate().
        # The pre-built short-circuit in run() leaves these as None.
        self._new_scenario_shocks: Optional[Dict[str, Any]] = None
        self._loaded_scenarios:    Optional[LoadedScenariosReport] = None
        self._validation_report:   Optional[ValidationReport] = None
```

---

### Appendix A · 5 — Orchestration methods

> **Paste target**: append these six methods inside the
> `EnsembleInferencePipeline` class, after `__init__`, in this order.
> They share the `# Orchestration` divider banner.

The orchestration group (lines 383–810) is the public surface that callers
interact with. `run()` orchestrates the staged path or short-circuits to
the pre-built path; the four staged methods (`load`, `load_scenarios`,
`validate`, `run_inference`) can also be called individually by the UI
between buttons.

Each staged method (a) checks its preconditions at the top and raises
`RuntimeError` with an actionable message if a prior stage is missing,
(b) emits a `running` event when it starts and an `ok`/`fail` event when
it finishes, and (c) caches its result on the instance so subsequent
calls can read it.

#### Appendix A · 5.1 — `run()` — top-level entry (contract preserved)

`run() -> InferenceResult` is the unchanged public entry point. Internally
it now dispatches between the staged path (default) and the pre-built
short-circuit. The `Pipeline started` / `Pipeline complete` /
`Pipeline failed` envelope events are emitted here; the individual stages
emit their own start/OK/fail events inside.

```python
    # ==================================================================
    # Orchestration
    # ==================================================================

    def run(self) -> InferenceResult:
        """
        Execute the full ensemble inference pipeline (top-level entry).

        Two paths share this signature:

        1. **Staged path** — the default for production new-scenarios
           runs.  Internally chains
           ``load() → load_scenarios() → validate() → run_inference()``.
           Validation failures (user-input problems) raise
           :class:`ValueError`; system failures propagate the
           underlying exception.

        2. **Pre-built short-circuit** — if
           ``metadata['inference']['member_inputs']`` is populated, skip
           scenario loading + validation and go straight from
           ``load() → predict() → post_infer()``.  This is the legacy
           model-agnostic path used by smoke tests and any caller that
           supplies its own member inputs.

        Lifecycle events
        ----------------
        When ``on_event`` was passed to ``__init__``, this method
        emits a structured narration — see
        :mod:`src.rade_ml_pt.pipelines.ensemble.infer_events`.  The
        outer ``Pipeline started`` / ``Pipeline complete`` /
        ``Pipeline failed`` envelope is emitted here; the individual
        stages emit their own start/OK/fail events inside.

        Returns
        -------
        InferenceResult
        """
        logger.info("EnsembleInferencePipeline: starting")
        self._emit(event(
            STAGE_INFERENCE, "Pipeline started",
            status=STATUS_RUNNING, target=self.ensemble_version,
        ))
        t0 = time.perf_counter()

        try:
            infer_meta = self.config.metadata.get("inference", {})
            input_mode = infer_meta.get("input_mode", "new_scenarios")
            if input_mode not in {"new_scenarios", "new_trades"}:
                raise ValueError(
                    f"Unknown input_mode '{input_mode}'. "
                    f"Supported modes: 'new_scenarios', 'new_trades'."
                )

            if infer_meta.get("member_inputs"):
                # Pre-built short-circuit: load model, then predict
                # straight from caller-supplied member inputs.  Works
                # for any input_mode (the caller has already taken
                # responsibility for what's in member_inputs).
                self.load()
                result = self._run_with_prebuilt(infer_meta)
            else:
                # Staged path requires per-mode build logic; new_trades
                # is not yet implemented end-to-end.  Callers that want
                # new_trades behaviour today must use the pre-built
                # short-circuit above.
                if input_mode == "new_trades":
                    raise NotImplementedError(
                        "new_trades inference is not yet supported in "
                        "the staged ensemble pipeline. Provide "
                        "metadata['inference']['member_inputs'] to use "
                        "the model-agnostic pre-built path."
                    )
                # Staged path: load → load_scenarios → validate →
                # run_inference.  Each stage caches its result on the
                # instance so the UI flow can resume mid-pipeline.
                self.load()
                self.load_scenarios()
                report = self.validate()
                if not report.is_valid:
                    raise ValueError(
                        f"Inference validation failed with "
                        f"{len(report.errors)} error(s): {report.errors}"
                    )
                result = self.run_inference()

            wall = time.perf_counter() - t0
            logger.info(
                "EnsembleInferencePipeline: done (%.3fs, %d samples)",
                wall, result.n_samples,
            )
            self._emit(event(
                STAGE_INFERENCE, "Pipeline complete",
                status=STATUS_OK,
                target=f"{wall * 1000:.0f} ms · {result.n_samples} samples",
            ))
            return result

        except Exception as exc:
            # Always surface failures into the activity log so the UI
            # can render a red-cross row instead of a silent timeout.
            self._emit(event(
                STAGE_INFERENCE, "Pipeline failed",
                status=STATUS_FAIL,
                target=type(exc).__name__,
                detail=str(exc),
            ))
            raise
```

#### Appendix A · 5.2 — `load()`

Loads the ensemble model + per-cluster inference contexts. Idempotent —
returns immediately if already loaded. Bound to the "Load model" UI button.

```python
    def load(self) -> None:
        """Load the ensemble model + per-cluster inference contexts.

        Idempotent — returns immediately if the ensemble is already
        loaded.  Dispatches to :meth:`_load_from_session` when an
        :class:`EnsembleSession` was provided to ``__init__`` and is
        fully ready, otherwise to :meth:`_load_from_registry`.

        Whether to load per-cluster inference contexts is inferred
        from ``config.metadata["inference"]["member_inputs"]``: pre-
        built inputs ⇒ contexts not needed (saves the per-cluster
        artifact reads).

        UI usage
        --------
        Bound to the "Load model" button on the *Inference Console*.
        Failures emit a ``status="fail"`` event from the underlying
        loader before re-raising.
        """
        if self._ensemble is not None:
            return

        infer_meta = self.config.metadata.get("inference", {})
        need_contexts = not bool(infer_meta.get("member_inputs"))

        if self._session is not None and self._session.all_inference_ready:
            self._load_from_session()
        else:
            self._load_from_registry(need_contexts=need_contexts)
```

#### Appendix A · 5.3 — `load_scenarios()`

Parses every shock CSV in the new-scenario directory and extracts the
canonical scenario index (validated equal across files). Returns a small
JSON-friendly report; the heavy shock dict stays on `self._new_scenario_shocks`.
Bound to the "Load scenarios" UI button.

```python
    def load_scenarios(
        self,
        new_scenario_dir: Optional[Union[str, Path]] = None,
    ) -> LoadedScenariosReport:
        """Parse shock CSVs and extract the canonical scenario index.

        Parameters
        ----------
        new_scenario_dir
            Folder containing one CSV per shocked risk factor (filename
            minus ``.csv`` is the RF key).  If ``None``, falls back to
            ``self.config.metadata["inference"]["new_scenario_dir"]``.

        Returns
        -------
        LoadedScenariosReport
            Small JSON-friendly summary of what was parsed.  The heavy
            shock dict itself stays in ``self._new_scenario_shocks``.

        Raises
        ------
        RuntimeError
            If :meth:`load` hasn't been called yet.
        ValueError
            If no scenario directory is provided, or shock files
            disagree on scenario labels / counts.

        Side effects
        ------------
        Resets ``self._validation_report`` to ``None`` — any prior
        validation is invalidated by re-loading scenarios.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before load_scenarios()."
            )

        if new_scenario_dir is None:
            new_scenario_dir = (
                self.config.metadata
                    .get("inference", {})
                    .get("new_scenario_dir")
            )
        if not new_scenario_dir:
            raise ValueError(
                "new_scenario_dir is required (pass as argument or set "
                "config.metadata['inference']['new_scenario_dir'])."
            )
        new_scenario_dir = str(new_scenario_dir)

        self._emit(event(
            STAGE_INFERENCE, "Loading new-scenario shocks",
            status=STATUS_RUNNING, target=new_scenario_dir,
        ))

        try:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                HybridGnnRnnInferencePipeline,
            )
            shocks = HybridGnnRnnInferencePipeline.load_new_scenarios(
                new_scenario_dir,
            )
            canonical = self._extract_canonical_index(shocks)
        except Exception as exc:
            self._emit(event(
                STAGE_INFERENCE, "Scenario load failed",
                status=STATUS_FAIL, target=new_scenario_dir,
                detail=str(exc),
            ))
            raise

        self._new_scenario_shocks = shocks
        self._validation_report = None  # invalidate downstream state

        rf_names = sorted(shocks.keys())
        report = LoadedScenariosReport(
            new_scenario_dir=new_scenario_dir,
            risk_factor_names=rf_names,
            n_risk_factors=len(rf_names),
            n_scenarios=len(canonical),
            scenario_labels=canonical,
        )
        self._loaded_scenarios = report

        self._emit(event(
            STAGE_INFERENCE, "Shocks loaded",
            status=STATUS_OK, target=new_scenario_dir,
            detail=(
                f"{report.n_risk_factors} RFs · "
                f"{report.n_scenarios} scenarios"
            ),
        ))
        return report
```

#### Appendix A · 5.4 — `validate()`

Classifies each ensemble member as affected/unaffected, and for unaffected
clusters verifies that every requested scenario label exists in the
historical `elementary_pnl` index. Returns a `ValidationReport` — *does
not raise* on user-input problems (those go into `report.errors`). Bound
to the "Validate" UI button.

```python
    def validate(self) -> ValidationReport:
        """Classify each cluster and check cheap-path eligibility.

        Returns a :class:`ValidationReport` with one
        :class:`ClusterRoutingDecision` per ensemble member.  Errors
        and warnings live in ``report.errors`` / ``report.warnings``;
        ``report.is_valid`` is ``True`` iff no errors were found.

        Validation issues (missing scenario labels in a cluster's
        history, no cluster intersects any shocked RF, etc.) are
        treated as user-input problems and **never raise** here — the
        caller (`run()` or the UI) decides what to do.

        Raises
        ------
        RuntimeError
            If :meth:`load` or :meth:`load_scenarios` haven't run.
        """
        if self._ensemble is None:
            raise RuntimeError("load() must be called before validate().")
        if self._loaded_scenarios is None:
            raise RuntimeError(
                "load_scenarios() must be called before validate()."
            )

        self._emit(event(
            STAGE_INFERENCE, "Validating scenarios",
            status=STATUS_RUNNING, target=self.ensemble_version,
        ))

        labels = self._loaded_scenarios.scenario_labels
        shock_rfs = sorted(self._new_scenario_shocks.keys())

        decisions: List[ClusterRoutingDecision] = []
        errors:    List[str] = []
        warnings:  List[str] = []

        for cid in self._ens_config.cluster_ids:
            ctx = self._inference_contexts.get(cid)
            if ctx is None:
                errors.append(
                    f"Cluster '{cid}': no inference context loaded."
                )
                continue

            decision = self._classify_cluster(cid, ctx, shock_rfs)

            if not decision.is_affected:
                ok, missing = self._validate_cluster_for_labels(
                    cid, ctx, labels,
                )
                if not ok:
                    # Rebuild the decision with missing labels recorded
                    # (frozen dataclass — can't mutate in place).
                    decision = ClusterRoutingDecision(
                        cluster_id=decision.cluster_id,
                        is_affected=decision.is_affected,
                        intersecting_risk_factors=decision.intersecting_risk_factors,
                        n_elementary_trades=decision.n_elementary_trades,
                        n_target_trades=decision.n_target_trades,
                        missing_scenario_labels=missing,
                    )
                    errors.append(
                        f"Cluster '{cid}' is unaffected but its history "
                        f"is missing {len(missing)}/{len(labels)} "
                        f"requested scenario labels "
                        f"(first 3: {missing[:3]})."
                    )

            decisions.append(decision)

        if decisions and not any(d.is_affected for d in decisions):
            errors.append(
                "No clusters intersect any shocked risk factor — nothing to run."
            )

        report = ValidationReport(
            ensemble_version=self.ensemble_version,
            n_scenarios=self._loaded_scenarios.n_scenarios,
            scenario_labels=labels,
            cluster_decisions=decisions,
            errors=errors,
            warnings=warnings,
        )
        self._validation_report = report

        self._emit(event(
            STAGE_INFERENCE, "Validation complete",
            status=STATUS_OK if report.is_valid else STATUS_FAIL,
            target=(
                f"{report.affected_count} affected · "
                f"{report.unaffected_count} unaffected"
            ),
            detail=(
                f"{len(report.errors)} error(s)" if report.errors else None
            ),
        ))
        return report
```

#### Appendix A · 5.5 — `run_inference()`

The granular forward-pass stage. Builds per-cluster inputs (routing each
cluster to the full or cheap path), calls `self._ensemble.predict`,
wraps the result. Bound to the "Run" UI button — only enabled once
`load`, `load_scenarios`, and `validate` (with `is_valid=True`) have all
completed.

```python
    def run_inference(self) -> InferenceResult:
        """Build inputs (per-cluster routed), predict, post-process.

        Assumes :meth:`load`, :meth:`load_scenarios`, :meth:`validate`
        have all completed successfully (i.e. ``report.is_valid`` is
        True).  Each cluster's :class:`ClusterRoutingDecision` selects
        the full re-pricing path or the cheap historical-lookup path.

        Raises
        ------
        RuntimeError
            If any prior stage is missing, or :meth:`validate`
            reported errors.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before run_inference()."
            )
        if self._loaded_scenarios is None:
            raise RuntimeError(
                "load_scenarios() must be called before run_inference()."
            )
        if self._validation_report is None:
            raise RuntimeError(
                "validate() must be called before run_inference()."
            )
        if not self._validation_report.is_valid:
            first = (
                self._validation_report.errors[0]
                if self._validation_report.errors else "?"
            )
            raise RuntimeError(
                f"Cannot run inference — validation reported "
                f"{len(self._validation_report.errors)} error(s). "
                f"First: {first}"
            )

        t0 = time.perf_counter()
        self._emit(event(
            STAGE_INFERENCE, "Forward pass started",
            status=STATUS_RUNNING,
            target=(
                f"{self._validation_report.affected_count} affected · "
                f"{self._validation_report.unaffected_count} unaffected"
            ),
        ))

        infer_meta = self.config.metadata.get("inference", {})
        input_mode = infer_meta.get("input_mode", "new_scenarios")
        member_inputs, extra_meta = self._build_member_inputs(input_mode)
        combined = self._ensemble.predict(member_inputs)

        self._emit(event(
            STAGE_INFERENCE, "Forward pass complete",
            status=STATUS_OK,
            target=f"{combined.shape[0]} scenarios",
        ))

        result = self._build_result(combined, infer_meta, extra_meta)
        result.latency_seconds = time.perf_counter() - t0
        self.post_infer(result)
        return result
```

#### Appendix A · 5.6 — `_run_with_prebuilt()`

Private fast path for callers that already have `member_inputs`. Skips
scenario loading + validation entirely.

```python
    def _run_with_prebuilt(
        self, infer_meta: Dict[str, Any],
    ) -> InferenceResult:
        """Predict using caller-supplied ``member_inputs`` (legacy path).

        Bypasses ``load_scenarios`` / ``validate`` /
        ``_build_member_inputs`` entirely — the caller is responsible
        for the contents of ``infer_meta["member_inputs"]``.  Used by
        the synthetic-smoke test path and any model-agnostic harness.
        """
        t0 = time.perf_counter()
        member_inputs = infer_meta["member_inputs"]

        self._emit(event(
            STAGE_INFERENCE, "Using pre-built member inputs",
            status=STATUS_OK,
            target=f"{len(member_inputs)} members",
        ))
        self._emit(event(
            STAGE_INFERENCE, "Forward pass started",
            status=STATUS_RUNNING,
            target=f"{len(member_inputs)} members",
        ))
        combined = self._ensemble.predict(member_inputs)
        self._emit(event(
            STAGE_INFERENCE, "Forward pass complete",
            status=STATUS_OK,
            target=f"{combined.shape[0]} scenarios",
        ))

        result = self._build_result(combined, infer_meta, {})
        result.latency_seconds = time.perf_counter() - t0
        self.post_infer(result)
        return result
```

---

### Appendix A · 6 — Loading helpers

> **Paste target**: append after §5 inside the
> `EnsembleInferencePipeline` class. Banner divider precedes them.

Two private methods (lines 812–907) called by `load()`:
`_load_from_registry` for a cold start (reads from `EnsembleRegistry`)
and `_load_from_session` for a warm start (reuses `EnsembleSession`'s
cache). Both end with the ensemble model + per-cluster
`InferenceContext`s populated on the instance.

```python
    # ==================================================================
    # Loading
    # ==================================================================

    def _load_from_registry(self, need_contexts: bool = True) -> None:
        """Cold-start: load ensemble + per-member inference contexts from registry."""
        self._emit(event(
            STAGE_INFERENCE, "Loading ensemble from registry",
            status=STATUS_RUNNING, target=self.ensemble_version,
        ))
        ens_registry = EnsembleRegistry(self.config.registry_dir)
        config, member_versions, version = ens_registry.load(self.ensemble_version)
        self._ens_config = config
        self._member_versions = member_versions

        from src.rade_ml_pt.registry.store import ModelRegistry

        registry = ModelRegistry(self.config.registry_dir)
        builder = EnsembleBuilder(registry)
        self._ensemble = builder.build(config, member_versions)
        self._emit(event(
            STAGE_INFERENCE, "Ensemble assembled",
            status=STATUS_OK,
            target=f"{config.n_members} members",
        ))

        if need_contexts:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                load_inference_context_from_dir,
            )

            for cid in config.cluster_ids:
                ver = member_versions[cid]
                version_dir = Path(self.config.registry_dir) / ver
                self._emit(event(
                    STAGE_INFERENCE, "Cluster context loading",
                    status=STATUS_RUNNING, target=cid,
                ))
                try:
                    raw = load_inference_context_from_dir(version_dir)
                    self._inference_contexts[cid] = _dict_to_inference_context(raw)
                    self._emit(event(
                        STAGE_INFERENCE, "Cluster context loaded",
                        status=STATUS_OK, target=cid,
                    ))
                except Exception as exc:
                    self._emit(event(
                        STAGE_INFERENCE, "Cluster context failed",
                        status=STATUS_FAIL, target=cid, detail=str(exc),
                    ))
                    raise ValueError(
                        f"Could not load inference context for cluster '{cid}' "
                        f"(version '{ver}', dir={version_dir}). "
                        f"Provide metadata['inference']['member_inputs'] for "
                        f"model-agnostic ensemble inference, or ensure model-"
                        f"specific inference artifacts are present. "
                        f"Root cause: {exc}"
                    ) from exc

        logger.info(
            "Loaded ensemble '%s' (%d members) from registry",
            version, config.n_members,
        )

    def _load_from_session(self) -> None:
        """Warm-start: reuse the session's pre-loaded models + contexts."""
        self._emit(event(
            STAGE_INFERENCE, "Reusing pre-loaded session",
            status=STATUS_RUNNING,
            target=self._session.ensemble_version,
        ))
        self._ens_config = self._session.config
        self._member_versions = self._session.member_versions
        self._ensemble = self._session.ensemble_model

        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

        for cid in self._ens_config.cluster_ids:
            state = self._session._inference[cid]
            ctx = state.inference_context
            if isinstance(ctx, InferenceContext):
                self._inference_contexts[cid] = ctx
            else:
                self._inference_contexts[cid] = _dict_to_inference_context(
                    ctx, data_config_override=state.data_config,
                )

        logger.info(
            "Using pre-loaded session (ensemble '%s', %d members)",
            self._session.ensemble_version, self._ens_config.n_members,
        )
        self._emit(event(
            STAGE_INFERENCE, "Session ready",
            status=STATUS_OK,
            target=f"{self._ens_config.n_members} members",
        ))
```

---

### Appendix A · 7 — Validation helpers

> **Paste target**: append after §6 inside the
> `EnsembleInferencePipeline` class. Banner divider precedes them.

Four private methods (lines 909–1046). All are no-IO, no-forward-pass —
they operate on already-parsed shocks and already-loaded
`InferenceContext` instances. Cheap to call. Together they implement the
classification + eligibility logic that `validate()` orchestrates.

```python
    # ==================================================================
    # Validation helpers
    # ==================================================================
    #
    # All private to the pipeline.  Operate on a single cluster's
    # ``InferenceContext`` (or — for ``_extract_canonical_index`` — the
    # already-parsed shock dict) and return primitive types or the
    # frozen ``ClusterRoutingDecision`` dataclass.  No IO; no model
    # forward pass; cheap to call.
    #

    def _extract_canonical_index(
        self,
        new_scenario_shocks: Dict[str, Dict[Any, Any]],
    ) -> List[str]:
        """Cross-shock-file: extract + validate the canonical scenario index.

        Every shock CSV must share the same scenario labels in the
        same order.  Returns the canonical list.

        Raises
        ------
        ValueError
            If no shocks were provided, or files disagree on labels
            or counts.  Errors include up to 3 mismatch examples so
            the user can fix their input files.
        """
        if not new_scenario_shocks:
            raise ValueError(
                "No shock files were loaded — empty new_scenario_dir?"
            )

        canonical:    Optional[List[str]] = None
        canonical_rf: Optional[str]       = None

        for rf, scenarios in new_scenario_shocks.items():
            labels = [str(k) for k in scenarios.keys()]
            if canonical is None:
                canonical    = labels
                canonical_rf = rf
                continue
            if len(labels) != len(canonical):
                raise ValueError(
                    f"Shock files disagree on scenario count: "
                    f"'{canonical_rf}' has {len(canonical)} scenarios, "
                    f"'{rf}' has {len(labels)}."
                )
            if labels != canonical:
                mismatches = [
                    (i, a, b)
                    for i, (a, b) in enumerate(zip(canonical, labels))
                    if a != b
                ]
                raise ValueError(
                    f"Shock files disagree on scenario labels "
                    f"('{canonical_rf}' vs '{rf}'). First mismatches "
                    f"(index, canonical, observed): {mismatches[:3]}"
                )

        assert canonical is not None  # for type checker
        return canonical

    def _extract_cluster_risk_factors(
        self, ctx: "InferenceContext",
    ) -> List[str]:
        """Per-cluster: union of RFs across all assets in the cluster.

        Reads ``ctx.cluster_assets[*].risk_factor_shocks.keys()``.
        Returns a sorted, deterministic list (used in intersection
        checks against the loaded-scenario RF set).
        """
        if ctx.cluster_assets is None:
            return []
        rfs: set = set()
        for asset in ctx.cluster_assets.values():
            rfs.update(asset.risk_factor_shocks.keys())
        return sorted(rfs)

    def _classify_cluster(
        self,
        cluster_id: str,
        ctx: "InferenceContext",
        shock_rfs: Iterable[str],
    ) -> ClusterRoutingDecision:
        """Per-cluster: produce the routing decision (affected vs unaffected).

        Cluster is *affected* iff at least one of its risk factors is
        present in the shocked-RF set.  Trade counts are pulled from
        ``ctx.trade_universe`` so the routing-decision payload is rich
        enough to drive the UI's routing table without any further
        context lookups.
        """
        cluster_rfs  = self._extract_cluster_risk_factors(ctx)
        intersecting = sorted(set(cluster_rfs) & set(shock_rfs))

        n_elementary = 0
        n_target     = 0
        if ctx.trade_universe is not None:
            n_elementary = len(
                ctx.trade_universe.get("elementary_ids", []) or []
            )
            n_target = len(
                ctx.trade_universe.get("target_ids", []) or []
            )

        return ClusterRoutingDecision(
            cluster_id=cluster_id,
            is_affected=len(intersecting) > 0,
            intersecting_risk_factors=intersecting,
            n_elementary_trades=n_elementary,
            n_target_trades=n_target,
        )

    def _validate_cluster_for_labels(
        self,
        cluster_id: str,
        ctx: "InferenceContext",
        scenario_labels: List[str],
    ) -> Tuple[bool, List[str]]:
        """Per-cluster cheap-path eligibility check.

        Returns ``(is_valid, missing_labels)``.  Only meaningful for
        unaffected clusters — they reuse historical elementary PnL
        looked up by scenario label from
        ``ctx.elementary_pnl.loc[labels]``, which requires every
        requested label to be present in the parquet's index.

        Returns ``False`` when ``ctx.elementary_pnl`` is missing or
        has a numeric (``RangeIndex``) — that index can't satisfy
        label lookups even if it has the right row count.
        """
        if ctx.elementary_pnl is None:
            return False, list(scenario_labels)
        if isinstance(ctx.elementary_pnl.index, pd.RangeIndex):
            return False, list(scenario_labels)
        existing = set(ctx.elementary_pnl.index)
        missing = [lab for lab in scenario_labels if lab not in existing]
        return len(missing) == 0, missing
```

---

### Appendix A · 8 — Input building

> **Paste target**: append after §7 inside the
> `EnsembleInferencePipeline` class. Banner divider precedes them.

Four methods (lines 1048–1269) that build the per-member input dicts
fed to `EnsembleModel.predict`. Layered as dispatcher → routing
dispatcher → per-cluster builders:

```
_build_member_inputs(mode)                # by input_mode
  └─ _build_new_scenarios_inputs()        # routing over cluster decisions
        ├─ _build_cluster_inputs_full(...)    # affected → re-price
        └─ _build_cluster_inputs_cheap(...)   # unaffected → historical lookup
```

Both per-cluster builders compose existing `@staticmethod` helpers on
`HybridGnnRnnInferencePipeline` (see `src/rade_ml_pt/pipelines/hybrid_gnn_rnn/infer.py`)
so there is **no duplicated business logic** — only the call graph
differs.

#### Appendix A · 8.1 — `_build_member_inputs()` (mode dispatcher)

```python
    # ==================================================================
    # Input building
    # ==================================================================

    def _build_member_inputs(
        self,
        input_mode: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Mode dispatcher for input building (staged flow only).

        The pre-built short-circuit is handled at :meth:`run` level —
        this method is only invoked for the staged flow, after
        ``validate()`` has populated ``self._validation_report``.

        Returns
        -------
        tuple of (member_inputs, extra_meta)
            *member_inputs*: ``{cluster_id: 7-key model input dict}``
            ready for ``EnsembleModel.predict``.
            *extra_meta*: auxiliary info (sample_ids per member).
        """
        if input_mode == "new_scenarios":
            return self._build_new_scenarios_inputs()

        if input_mode == "new_trades":
            raise NotImplementedError(
                "new_trades inference is not yet supported in the ensemble "
                "pipeline. The underlying HybridGnnRnnInferencePipeline does "
                "not implement _prepare_new_trade_inputs yet."
            )

        raise ValueError(f"Unknown input_mode: {input_mode}")
```

#### Appendix A · 8.2 — `_build_new_scenarios_inputs()` (routing dispatcher)

Walks `self._validation_report.cluster_decisions` and dispatches each
cluster to the full or cheap path. Emits per-cluster build events.

```python
    def _build_new_scenarios_inputs(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Routing dispatcher for new-scenarios mode (staged flow).

        Walks ``self._validation_report.cluster_decisions`` in order
        and dispatches each cluster to the full or cheap path:

        * Affected → :meth:`_build_cluster_inputs_full` (re-prices
          under new shocks; identical to the pre-refactor logic).
        * Unaffected → :meth:`_build_cluster_inputs_cheap` (looks up
          historical PnL via ``ctx.elementary_pnl.loc[labels]`` —
          no re-pricing, no re-scaling).

        Returns
        -------
        tuple of (member_inputs, extra_meta)
        """
        assert self._validation_report is not None, "validate() must run first"
        assert self._new_scenario_shocks is not None, "load_scenarios() must run first"

        member_inputs: Dict[str, Any] = {}
        extra_meta:    Dict[str, Any] = {"sample_ids": {}}

        labels = self._validation_report.scenario_labels
        shocks = self._new_scenario_shocks

        for decision in self._validation_report.cluster_decisions:
            cid  = decision.cluster_id
            ctx  = self._inference_contexts[cid]
            path = "full" if decision.is_affected else "cheap"

            self._emit(event(
                STAGE_INFERENCE, "Building cluster inputs",
                status=STATUS_RUNNING, target=cid,
                detail=f"path={path}",
            ))
            try:
                if decision.is_affected:
                    result = self._build_cluster_inputs_full(cid, ctx, shocks)
                else:
                    result = self._build_cluster_inputs_cheap(cid, ctx, labels)

                member_inputs[cid] = result["inputs"]
                extra_meta["sample_ids"][cid] = result.get("sample_ids")

                n_windows = result["metadata"]["n_scenarios"]
                logger.info(
                    "Built inputs for cluster '%s' (path=%s, %d windows)",
                    cid, path, n_windows,
                )
                self._emit(event(
                    STAGE_INFERENCE, "Cluster inputs ready",
                    status=STATUS_OK, target=cid,
                    detail=f"path={path} · {n_windows} scenarios",
                ))
            except Exception as exc:
                self._emit(event(
                    STAGE_INFERENCE, "Cluster input build failed",
                    status=STATUS_FAIL, target=cid, detail=str(exc),
                ))
                raise

        return member_inputs, extra_meta
```

#### Appendix A · 8.3 — `_build_cluster_inputs_full()` (affected → re-price)

The full path: re-price elementary PnL under the new shocks, then run it
through the same sequencing + scaling pipeline used at training time.
This is the expensive step — `calculate_elementary_pnl` invokes the
static-replication kernels per asset.

```python
    def _build_cluster_inputs_full(
        self,
        cluster_id: str,
        ctx: "InferenceContext",
        new_scenario_shocks: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Full re-pricing path for an *affected* cluster.

        Composes the existing ``@staticmethod`` helpers on
        :class:`HybridGnnRnnInferencePipeline`:

        1. ``_inject_unchanged_inputs`` — trade features, adjacency,
           indices (all static, registry-loaded).
        2. ``_update_asset_portfolio`` — deep-copy + inject new shocks.
        3. Filter elementary trades to the cluster's reduced population.
        4. ``calculate_elementary_pnl`` — static-replication kernels.
           This is the expensive step.
        5. ``pd.concat`` + column reorder to training schema.
        6. ``_standardise_pnl`` — fit scaler from training.
        7. ``build_new_pnl_sequences`` — same windowing as training.
        8. ``build_model_inputs`` — assemble the 7-key dict.

        Returns the full ``{"inputs", "sample_ids", "metadata"}`` dict
        from ``build_model_inputs``; the caller peels off
        ``result["inputs"]`` and ``result["sample_ids"]``.
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        inputs = HybridGnnRnnInferencePipeline._inject_unchanged_inputs(
            ctx, mode="new_scenarios",
        )

        new_asset_portfolio = HybridGnnRnnInferencePipeline._update_asset_portfolio(
            ctx.cluster_assets, new_scenario_shocks,
        )

        elem_trades = {
            z: [
                x for x in ctx.cluster_elem_trades[z]
                if x["id"] in inputs.elementary_ids
            ]
            for z in ctx.cluster_elem_trades.keys()
        }

        asset_elementary_pnl = HybridGnnRnnInferencePipeline.calculate_elementary_pnl(
            asset_portfolio=new_asset_portfolio,
            elementary_trades=elem_trades,
        )

        new_pnl = pd.concat(asset_elementary_pnl.values(), axis=1)
        new_pnl = new_pnl[ctx.elementary_attributes["trade_id"]]

        new_pnl_scaled = HybridGnnRnnInferencePipeline._standardise_pnl(
            pnl_unscaled=new_pnl, scaler=ctx.elementary_scaler,
        )

        inputs.elementary_pnl = pd.DataFrame(
            new_pnl_scaled,
            columns=ctx.elementary_pnl.columns.tolist(),
            index=new_pnl.index.tolist(),
        )

        elem_seq = HybridGnnRnnInferencePipeline.build_new_pnl_sequences(
            elementary_pnl=inputs.elementary_pnl,
            seq_length=ctx.data_config.seq_length,
            n_targets=len(inputs.target_indices),
        )

        return HybridGnnRnnInferencePipeline.build_model_inputs(
            elem_seq=elem_seq,
            inputs=inputs,
            seq_length=ctx.data_config.seq_length,
        )
```

#### Appendix A · 8.4 — `_build_cluster_inputs_cheap()` (unaffected → historical lookup)

The Tier-2 cheap path: for an unaffected cluster, elementary PnL under
the new scenarios *is* the historical elementary PnL on disk (by
definition — none of its risk factors are shocked). Skip re-pricing,
skip re-scaling; only the sequencing + assembly remain. Eligibility
(every requested scenario label exists in the historical index) was
verified up-front by `_validate_cluster_for_labels` during `validate()`,
so a missing label can never reach here.

```python
    def _build_cluster_inputs_cheap(
        self,
        cluster_id: str,
        ctx: "InferenceContext",
        scenario_labels: List[str],
    ) -> Dict[str, Any]:
        """Cheap historical-lookup path for an *unaffected* cluster.

        ``ctx.elementary_pnl`` was saved during training as a parquet
        of scaled, population-reduced elementary PnL indexed by
        scenario label.  Since this cluster's risk factors don't
        intersect any of the loaded scenario shocks, the elementary
        PnL under the "new" scenarios is — by definition — identical
        to the historical values stored on disk.

        We therefore skip re-pricing and re-scaling entirely:

        1. ``_inject_unchanged_inputs`` — same static inputs.
        2. ``ctx.elementary_pnl.loc[scenario_labels]`` — direct lookup
           (eligibility was verified by ``_validate_cluster_for_labels``
           during ``validate()``, so missing labels can't reach here).
        3. ``build_new_pnl_sequences`` — same windowing as the full
           path; cluster sees the same shape.
        4. ``build_model_inputs`` — same assembly.

        Returns the full ``{"inputs", "sample_ids", "metadata"}`` dict.
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        inputs = HybridGnnRnnInferencePipeline._inject_unchanged_inputs(
            ctx, mode="new_scenarios",
        )

        inputs.elementary_pnl = ctx.elementary_pnl.loc[scenario_labels, :]

        elem_seq = HybridGnnRnnInferencePipeline.build_new_pnl_sequences(
            elementary_pnl=inputs.elementary_pnl,
            seq_length=ctx.data_config.seq_length,
            n_targets=len(inputs.target_indices),
        )

        return HybridGnnRnnInferencePipeline.build_model_inputs(
            elem_seq=elem_seq,
            inputs=inputs,
            seq_length=ctx.data_config.seq_length,
        )
```

---

### Appendix A · 9 — Post-inference + result building

> **Paste target**: append after §8 inside the
> `EnsembleInferencePipeline` class — these are the final methods in
> the file.

The post-inference group (lines 1271–1366):

- `post_infer` logs a summary and (if `artifacts_dir` is set) writes
  `predictions.csv` + `inference_result.json`.
- `_save_predictions` is the CSV writer (kept separate so it can be
  unit-tested in isolation).
- `_build_result` wraps the aggregated predictions into the public
  `InferenceResult` dataclass. **Important**: when the staged path was
  used (`self._validation_report` is not None), four extra metadata
  keys are added: `affected_clusters`, `unaffected_clusters`,
  `cheap_path_used`, `scenario_labels`. The pre-built short-circuit
  leaves `_validation_report` as `None`, so those keys are omitted and
  the legacy result schema is preserved.

```python
    # ==================================================================
    # Post-inference
    # ==================================================================

    def post_infer(self, result: InferenceResult) -> None:
        """Post-inference analytics: log summary, save predictions CSV."""
        if result.predictions is not None:
            preds = result.predictions
            logger.info(
                "Ensemble inference summary: n_samples=%d, mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                result.n_samples, np.mean(preds), np.std(preds),
                np.min(preds), np.max(preds),
            )

        if self.config.artifacts_dir and result.predictions is not None:
            self._save_predictions(result)

    def _save_predictions(self, result: InferenceResult) -> None:
        """Save predictions to CSV in the artifacts directory."""
        out_dir = Path(self.config.artifacts_dir) / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "predictions.csv"
        preds = result.predictions

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            if preds.ndim == 2:
                header = [f"target_{i}" for i in range(preds.shape[1])]
                if result.sample_ids:
                    header = ["sample_id"] + header
                writer.writerow(header)
                for row_idx in range(preds.shape[0]):
                    row = list(preds[row_idx])
                    if result.sample_ids and row_idx < len(result.sample_ids):
                        row = [result.sample_ids[row_idx]] + row
                    writer.writerow(row)
            else:
                writer.writerow(["prediction"])
                for val in preds.flat:
                    writer.writerow([val])

        logger.info("Predictions saved to %s", csv_path)

        result.to_json(out_dir / "inference_result.json")

    # ==================================================================
    # Result building
    # ==================================================================

    def _build_result(
        self,
        combined: np.ndarray,
        infer_meta: Dict[str, Any],
        extra_meta: Dict[str, Any],
    ) -> InferenceResult:
        """Wrap aggregated predictions into an ``InferenceResult``.

        Includes routing metadata (affected/unaffected clusters,
        cheap_path_used flag, scenario labels) when the run went
        through the staged flow.  The pre-built short-circuit leaves
        ``self._validation_report`` as ``None``, in which case these
        keys are omitted from the result metadata.
        """
        meta: Dict[str, Any] = {
            "input_mode": infer_meta.get("input_mode", "new_scenarios"),
            "cluster_ids": self._ensemble.router.cluster_ids if self._ensemble else [],
            "n_members": self._ens_config.n_members if self._ens_config else 0,
        }
        if self._validation_report is not None:
            meta["affected_clusters"]    = self._validation_report.affected_cluster_ids
            meta["unaffected_clusters"]  = self._validation_report.unaffected_cluster_ids
            meta["cheap_path_used"]      = self._validation_report.cheap_path_used
            meta["scenario_labels"]      = self._validation_report.scenario_labels
        if extra_meta.get("sample_ids"):
            meta["per_member_sample_ids"] = extra_meta["sample_ids"]

        all_sample_ids = None
        per_member_ids = extra_meta.get("sample_ids", {})
        if per_member_ids:
            all_sample_ids = []
            for cid in sorted(per_member_ids.keys()):
                ids = per_member_ids[cid]
                if ids:
                    all_sample_ids.extend(ids)
            all_sample_ids = all_sample_ids or None

        return InferenceResult(
            predictions=combined,
            n_samples=combined.shape[0],
            sample_ids=all_sample_ids or infer_meta.get("sample_ids"),
            model_version=self.ensemble_version,
            metadata=meta,
        )
```

---

### Appendix A · 10 — Porting checklist

When transferring this file to your work env:

1. **Replace the entire `src/rade_ml_pt/pipelines/ensemble/infer.py`** with
   sections §1 → §9 concatenated in order. There is no shared state
   between sections beyond what's already inlined; the file is self-
   contained.
2. **Verify** `src/rade_ml_pt/pipelines/ensemble/infer_events.py` exists
   in your work env (it shipped in Phase 1 — unchanged in this refactor).
3. **No changes needed** in `src/rade_ml_pt/pipelines/hybrid_gnn_rnn/infer.py`
   — every helper called from `_build_cluster_inputs_full` /
   `_build_cluster_inputs_cheap` is an existing `@staticmethod` on
   `HybridGnnRnnInferencePipeline`. A future PR may refactor that file to
   add a `build_cluster_inputs(...)` method, but it is intentionally
   deferred (see `ensemble_infer_refactor.md` decisions log).
4. **Run the verification suite**:

```bash
# Lints — file is clean in this env
ruff check src/rade_ml_pt/pipelines/ensemble/infer.py

# Existing event tests — all 18 must pass
pytest tests/rade_ml_pt/pipelines/ensemble/test_infer_events.py -v

# Existing pipeline tests — must pass at the same baseline as before
# (3 of 26 failures in this env are caused by a missing rade_sr import,
#  unrelated to this refactor; they pass in the work env which has
#  rade_sr properly installed)
pytest tests/rade_ml_pt/pipelines/ensemble/test_ensemble_pipelines.py -v
```

5. **Optional smoke test for the staged flow** (requires a real
   ensemble version and a folder of shock CSVs on disk):

```python
from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.pipelines.ensemble.infer import EnsembleInferencePipeline

config = EnsembleConfig(
    registry_dir="path/to/registry",
    artifacts_dir="path/to/artifacts",
    metadata={
        "inference": {
            "input_mode": "new_scenarios",
            "new_scenario_dir": "path/to/new_scenarios/",
        }
    },
)

pipe = EnsembleInferencePipeline(config, ensemble_version="latest")

# Either: one-shot
result = pipe.run()

# Or: staged (UI-style)
pipe.load()
scenarios = pipe.load_scenarios()
print(scenarios.to_dict())
report = pipe.validate()
print(report.to_dict())
assert report.is_valid
result = pipe.run_inference()
```

Both paths return the same `InferenceResult`; the staged path additionally
exposes `LoadedScenariosReport` and `ValidationReport` as intermediate
artefacts the UI can render between buttons.
