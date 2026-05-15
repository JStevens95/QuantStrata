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

## Appendix A — Inference pipeline & API source (Stages 1-10, copy-paste sync)

> **Status**: Pipeline + Stage 6 post_infer + Stage 7-10 API endpoints are
> all in this repo and verified working end-to-end (FastAPI app boots, all
> routes register, lints clean). This appendix exists so the **work env**
> can be brought to the exact same state by copy-paste, no judgement calls.
>
> **Companion**: `docs/inference_pipeline_contract.md` is the canonical
> signature / dataclass / on-disk-layout reference. The code below is the
> reference implementation of that contract; if anything ever drifts, the
> contract wins and this appendix gets updated.
>
> **What's NOT included here** (assumed already aligned in your work env):
>
> * `src/rade_ml_pt/pipelines/hybrid_gnn_rnn/infer.py` — host of
>   `InferenceContext`, `intersecting_risk_factors`,
>   `missing_scenario_labels`, `_build_affected_inputs`,
>   `_build_unaffected_inputs`, `transform_predictions`,
>   `predict_member_chunked`. You implemented these in your work env
>   directly; the contract doc captures their signatures.
> * `src/rade_ml_pt/ensemble/api/services/reader.py`, `paths.py`,
>   `version.py`, `config.py`, and all of `models/meta.py` — pre-existing
>   API plumbing that hasn't changed.
> * `src/rade_ml_pt/pipelines/ensemble/infer_events.py` —
>   `EventCollector`, `event()`, `ActivityEntry`, `STAGE_*`, `STATUS_*`.
>   Unchanged since the v2 appendix.

### Reading guide

Seven files in this appendix, in dependency order (lowest-level first):

| § | File | Purpose | Stages |
|---|---|---|---|
| A.1 | `src/rade_ml_pt/pipelines/ensemble/infer.py` | Inference pipeline orchestrator + Stage 6 post_infer | 1-6 |
| A.2 | `src/rade_ml_pt/ensemble/api/services/inference_state.py` | Run-state manager + background thread dispatch | 7-9 |
| A.3 | `src/rade_ml_pt/ensemble/api/services/result_reader.py` | On-disk artifact reader for historical runs | 10 |
| A.4 | `src/rade_ml_pt/ensemble/api/routers/inference.py` | REST endpoints (control plane + data plane) | 7-10 |
| A.5 | `src/rade_ml_pt/ensemble/api/models/inference.py` | Pydantic request / response schemas | 7-10 |
| A.6 | `src/rade_ml_pt/ensemble/api/dependencies.py` | FastAPI dependency-injection wiring | 7-10 |
| A.7 | `src/rade_ml_pt/ensemble/api/app.py` | App factory + lifespan startup | 7-10 |

**On-disk layout this code produces / reads** (single source of truth: the
constants at the top of A.1):

```
<config.artifacts_dir>/                       (= <base_artifacts_dir>/inference_runs/<run_id>/ for API runs)
└── inference/                                  (= INFERENCE_DIRNAME)
    ├── manifest.json
    ├── cluster_summary/
    │   └── cluster_predictions.parquet
    ├── portfolio_summary/
    │   └── portfolio_predictions.parquet
    └── trade_predictions/
        ├── <cluster_id>_scaled.parquet
        └── <cluster_id>_original.parquet
```

---

### A · 1 — `src/rade_ml_pt/pipelines/ensemble/infer.py` (Stages 1-6)

The ensemble inference pipeline. Public entry points: `__init__`, `run()`,
`load()`, `load_scenarios()`, `validate_scenarios()`, `run_inference()`,
`post_infer()`. Stage 6 lives in `_post_infer_cluster`, `post_infer`,
`_write_run_manifest`, and the seven on-disk layout constants right
after the dataclasses.

```python
"""
Ensemble inference pipeline.

Two ways to drive a run:

1. **One-shot (programmatic / CLI):**
   Construct with an ``EnsembleConfig`` (and optionally an
   :class:`EnsembleSession`) and call ``run()``.  The pipeline
   executes the staged flow internally —
   ``load → load_scenarios → validate_scenarios → run_inference`` —
   and returns an :class:`InferenceResult`.

2. **Staged (UI):**
   Construct with an ``EnsembleConfig`` *plus* an
   :class:`EnsembleSession` so the model + contexts are warm-loaded
   from RAM instead of disk, then invoke each stage method
   individually.  Each button on the *Inference Console* triggers
   one stage and surfaces its report (or errors) to the user.

Per-cluster work (input building, routing classification) lives
in :class:`HybridGnnRnnInferencePipeline` as static methods; this
pipeline is the orchestrator + routing-decision layer.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
        cluster_rf_keys=raw.get("cluster_rf_keys"),
        _cluster_assets_path=raw.get("_cluster_assets_path"),
        cluster_elem_trades=raw.get("cluster_elem_trades"),
    )


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
        :meth:`validate_scenarios` to look up rows in each cluster's
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


@dataclass(frozen=True)
class LoadedNewTradesReport:
    """Stub — populated when ``new_trades`` mode is implemented.

    Counterpart to :class:`LoadedScenariosReport` for the
    ``new_trades`` input mode.  Currently carries only the path
    that was loaded; additional summary fields land alongside the
    real :meth:`EnsembleInferencePipeline.load_new_trades` body.
    """
    new_trades_path: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view."""
        return {"new_trades_path": self.new_trades_path}


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
        Populated by ``validate_scenarios()`` *only* for unaffected
        clusters whose history doesn't cover all requested scenario
        labels.
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


@dataclass(frozen=True)
class ValidationReport:
    """Output of :meth:`EnsembleInferencePipeline.validate_scenarios`.

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


# ------------------------------------------------------------------
# Inference artifact on-disk layout (Stage 6)
#
# Mirrors the eval pipeline's convention of one subdirectory per
# artifact family.  Single source of truth for the names —
# `services.result_reader` re-imports these so writer and reader
# can never drift again.
# ------------------------------------------------------------------

INFERENCE_DIRNAME:        str = "inference"
CLUSTER_SUMMARY_DIRNAME:  str = "cluster_summary"
PORTFOLIO_SUMMARY_DIRNAME:str = "portfolio_summary"
TRADE_PREDICTIONS_DIRNAME:str = "trade_predictions"

CLUSTER_SUMMARY_FILENAME:   str = "cluster_predictions.parquet"
PORTFOLIO_SUMMARY_FILENAME: str = "portfolio_predictions.parquet"
MANIFEST_FILENAME:          str = "manifest.json"


# ------------------------------------------------------------------
# Per-cluster artifact (Stage 6)
# ------------------------------------------------------------------

@dataclass
class ClusterArtifact:
    """Lightweight handle on one cluster's persisted predictions.

    Built inside :meth:`EnsembleInferencePipeline._post_infer_cluster`
    right after the two per-cluster parquets are written, and
    accumulated through :meth:`run_inference` for downstream
    aggregation in :meth:`post_infer`.

    ``summary_df`` is the per-scenario aggregate frame returned by
    :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
    (columns: ``scenario_label``, ``cluster_id``, ``sum_pnl_scaled``,
    ``sum_pnl_original``, ``mean_pnl_original``, ``std_pnl_original``,
    ``min_pnl_original``, ``max_pnl_original``).  Kept in-memory only
    long enough for ``post_infer`` to stack across clusters and write
    the run-level cluster + portfolio summaries; the heavy wide
    frames (``scaled_wide`` / ``original_wide``) have already been
    flushed to disk by the time this dataclass exists.

    ``eq=False`` so the dataclass doesn't try to compare its
    embedded DataFrame (pandas refuses ``==`` on heterogeneous
    frames); this dataclass is identity-compared in practice.

    Attributes
    ----------
    cluster_id
        Cluster this artifact belongs to.
    n_trades
        Number of target trades in the cluster (== width of both
        wide parquets).
    n_scenarios
        Number of scenario rows scored for this cluster.
    scaled_path, original_path
        Absolute filesystem paths to the two wide parquets the API
        result-reader will serve.
    trade_ids
        Canonical column ordering of both wide parquets (taken from
        ``target_scaler.feature_names_in_``).  Surfaced in the
        manifest so the UI can list cluster contents without
        opening the parquet.
    summary_df
        Per-scenario summary aggregates for this cluster.
        ``repr=False`` because pandas DataFrames make logs unreadable.
    """

    cluster_id:    str
    n_trades:      int
    n_scenarios:   int
    scaled_path:   str
    original_path: str
    trade_ids:     List[str]                  = field(default_factory=list)
    summary_df:    Optional[pd.DataFrame]     = field(default=None, repr=False, compare=False)

    def to_manifest_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view written into ``manifest.json``.

        Deliberately excludes ``summary_df`` (heavy + redundant — the
        same data is rolled up into the run-level cluster summary
        parquet).
        """
        return {
            "cluster_id":    self.cluster_id,
            "n_trades":      self.n_trades,
            "n_scenarios":   self.n_scenarios,
            "scaled_path":   self.scaled_path,
            "original_path": self.original_path,
            "trade_ids":     list(self.trade_ids),
        }


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
        ensemble_config : EnsembleConfig
            Pipeline configuration.  Must have ``registry_dir`` set
            when ``session`` is ``None``.
            ``metadata['inference']['input_mode']`` selects
            ``'new_scenarios'`` (default) or ``'new_trades'``.
        ensemble_version : str
            Ensemble version or tag to load.  When a ``session`` is
            provided this must match ``session.ensemble_version`` —
            see :meth:`load` for the safety check.
        session : EnsembleSession, optional
            Pre-loaded session (UI / backend path).  When provided,
            :meth:`load` reuses the session's models + contexts
            instead of reading the registry.  When ``None``,
            :meth:`load` cold-loads from ``ensemble_config.registry_dir``.
        on_event : EmitFn, optional
            Lifecycle-event callback.  ``None`` (default) silences
            events — the pipeline runs exactly as it did before the
            hook existed.  Pass an :class:`EventCollector` to stream
            a narration into the UI activity log.
        """
        # Constructor inputs — never reassigned after this point.
        self.config           = ensemble_config
        self.ensemble_version = ensemble_version
        self._session         = session
        self._emit: EmitFn    = on_event if on_event is not None else noop_emit

        # Model layer — populated by load().
        self._ensemble                                       = None
        self._ens_config:         Optional[EnsembleConfig]   = None
        self._member_versions:    Optional[Dict[str, str]]   = None
        self._inference_contexts: Dict[str, Any]             = {}

        # new_scenarios input — populated by load_scenarios().
        self._new_scenario_shocks: Optional[Dict[str, Any]]        = None
        self._loaded_scenarios:    Optional[LoadedScenariosReport] = None

        # new_trades input — populated by load_new_trades() (stub for now;
        # LoadedNewTradesReport dataclass lands with the real body).
        self._new_trades_payload:  Optional[Dict[str, Any]]            = None
        self._loaded_new_trades:   Optional["LoadedNewTradesReport"]   = None

        # Validation — written by validate_scenarios() / validate_new_trades(),
        # read by run_inference() and _build_result().
        self._validation_report:   Optional[ValidationReport]          = None

    # ==================================================================
    # Orchestration
    # ==================================================================

    def run(self) -> InferenceResult:
        """Execute the full staged inference pipeline.

        Convenience wrapper that calls the four staged methods in
        order.  The middle pair is mode-specific — the input mode is
        read from ``config.metadata['inference']['input_mode']``::

            load()
            ├─ new_scenarios → load_scenarios()  → validate_scenarios()
            └─ new_trades    → load_new_trades() → validate_new_trades()
            run_inference()

        Validation failures (user-input problems surfaced in
        ``report.errors``) raise :class:`ValueError`.  System
        failures propagate the underlying exception.

        Lifecycle events
        ----------------
        Emits the outer ``Pipeline started`` / ``Pipeline complete`` /
        ``Pipeline failed`` envelope here; each inner stage emits its
        own start/OK/fail events.

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
            input_mode = (
                self.config.metadata
                    .get("inference", {})
                    .get("input_mode", "new_scenarios")
            )
            if input_mode not in {"new_scenarios", "new_trades"}:
                raise ValueError(
                    f"Unknown input_mode '{input_mode}'. "
                    f"Supported: 'new_scenarios', 'new_trades'."
                )

            self.load()

            if input_mode == "new_scenarios":
                self.load_scenarios()
                report = self.validate_scenarios()
            else:  # "new_trades" — validated above
                self.load_new_trades()
                report = self.validate_new_trades()

            if not report.is_valid:
                raise ValueError(
                    f"Inference validation failed with "
                    f"{len(report.errors)} error(s): {report.errors}"
                )

            result = self.run_inference()
        except Exception as exc:
            self._emit(event(
                STAGE_INFERENCE, "Pipeline failed",
                status=STATUS_FAIL,
                target=type(exc).__name__, detail=str(exc),
            ))
            raise

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

    def load(self) -> None:
        """Load the ensemble model + per-cluster inference contexts.

        Idempotent — returns immediately if already loaded.  Source
        selection is decided at ``__init__`` time, not here:

        * **Session given** → reuse the pre-loaded models + contexts
          (warm path).  The session must match the requested
          ``ensemble_version`` and have finished loading.
        * **No session** → cold-load from the registry at
          ``config.registry_dir``.

        There is no silent fallback: a mismatched or not-ready
        session raises rather than falling back to the registry.

        Raises
        ------
        RuntimeError
            If a session was provided whose version mismatches, or
            whose inference contexts have not finished loading.

        UI usage
        --------
        Bound to the "Load model" button on the *Inference Console*.
        Failures emit a ``status="fail"`` event from the underlying
        loader before re-raising.
        """
        if self._ensemble is not None:
            return  # idempotent — already loaded

        if self._session is None:
            self._load_from_registry()
            return

        if self._session.ensemble_version != self.ensemble_version:
            raise RuntimeError(
                f"Session is loaded for ensemble version "
                f"'{self._session.ensemble_version}' but the pipeline was "
                f"constructed for '{self.ensemble_version}'. Construct "
                f"the pipeline with the matching session, or pass "
                f"session=None."
            )
        if not self._session.all_inference_ready:
            raise RuntimeError(
                f"Session for '{self._session.ensemble_version}' has "
                f"not finished loading per-cluster inference contexts "
                f"(all_inference_ready=False). Wait for the session to "
                f"finish loading before constructing the pipeline."
            )

        self._load_from_session()

    def load_scenarios(
        self,
        new_scenario_dir: Optional[Union[str, Path]] = None,
    ) -> LoadedScenariosReport:
        """Parse new-scenario shock CSVs into the pipeline state.

        Three steps:

          1. Resolve the scenario directory (argument or config fallback).
          2. Parse every ``<risk_factor>.csv`` under that directory.
          3. Cross-check that all shock files share the same scenario
             label index, and extract the canonical list.

        Returns a small JSON-friendly :class:`LoadedScenariosReport`
        summarising what was parsed; the heavy shock dict itself
        stays in ``self._new_scenario_shocks`` for
        :meth:`validate_scenarios` and the input builder to consume.

        Parameters
        ----------
        new_scenario_dir
            Folder containing one CSV per shocked risk factor
            (filename minus ``.csv`` is the RF key).  Falls back to
            ``self.config.metadata['inference']['new_scenario_dir']``
            if ``None``.

        Raises
        ------
        RuntimeError
            If :meth:`load` hasn't been called yet.
        ValueError
            If no scenario directory is provided, or shock files
            disagree on scenario labels / counts.

        Side effects
        ------------
        Invalidates any prior :meth:`validate_scenarios` result by
        resetting ``self._validation_report`` to ``None``.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before load_scenarios()."
            )

        # --- Step 1: resolve the directory ---
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

        # --- Step 2 + 3: parse + cross-check ---
        self._emit(event(
            STAGE_INFERENCE, "Loading new-scenario shocks",
            status=STATUS_RUNNING, target=new_scenario_dir,
        ))

        try:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                HybridGnnRnnInferencePipeline,
            )
            shocks    = HybridGnnRnnInferencePipeline.load_new_scenarios(new_scenario_dir)
            canonical = self._extract_canonical_index(shocks)
        except Exception as exc:
            self._emit(event(
                STAGE_INFERENCE, "Scenario load failed",
                status=STATUS_FAIL, target=new_scenario_dir, detail=str(exc),
            ))
            raise

        # --- Commit to instance state + build the report ---
        self._new_scenario_shocks = shocks
        self._validation_report   = None  # any prior validation is now stale

        rf_names = sorted(shocks.keys())
        report = LoadedScenariosReport(
            new_scenario_dir  = new_scenario_dir,
            risk_factor_names = rf_names,
            n_risk_factors    = len(rf_names),
            n_scenarios       = len(canonical),
            scenario_labels   = canonical,
        )
        self._loaded_scenarios = report

        self._emit(event(
            STAGE_INFERENCE, "Shocks loaded",
            status=STATUS_OK, target=new_scenario_dir,
            detail=f"{report.n_risk_factors} RFs · {report.n_scenarios} scenarios",
        ))
        return report

    def load_new_trades(
        self,
        new_trades_path: Optional[Union[str, Path]] = None,
    ) -> "LoadedNewTradesReport":
        """Parse a new-trades payload into the pipeline state.

        Counterpart to :meth:`load_scenarios` for the ``new_trades``
        input mode.  Will populate ``self._new_trades_payload`` and
        ``self._loaded_new_trades`` (analogous to the new-scenarios
        slots) so :meth:`validate_new_trades` and :meth:`run_inference`
        can pick them up.

        Raises
        ------
        RuntimeError
            If :meth:`load` hasn't been called yet.
        NotImplementedError
            Always — the per-cluster trade routing helper on
            :class:`HybridGnnRnnInferencePipeline` does not exist
            yet.  Extension point: drop the parsing + routing body
            in here once that helper is added.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before load_new_trades()."
            )
        raise NotImplementedError(
            "new_trades input mode is not yet implemented. The "
            "staged pipeline scaffold exists; the missing piece is "
            "per-cluster trade routing — see "
            "HybridGnnRnnInferencePipeline."
        )

    def validate_scenarios(self) -> ValidationReport:
        """Classify each cluster and check cheap-path eligibility.

        For each ensemble member, decides whether the cluster is
        *affected* by the loaded shocks (cluster RFs ∩ shock RFs ≠ ∅).
        For unaffected clusters, additionally checks that the cluster's
        historical ``elementary_pnl`` index contains every requested
        scenario label (cheap-path eligibility).

        Returns a :class:`ValidationReport` with one
        :class:`ClusterRoutingDecision` per ensemble member.  Errors
        and warnings live in ``report.errors`` / ``report.warnings``;
        ``report.is_valid`` is True iff no errors were found.

        Validation issues are treated as user-input problems and
        **never raise** here — the caller (``run()`` or the UI)
        decides what to do with the report.

        Raises
        ------
        RuntimeError
            If :meth:`load` or :meth:`load_scenarios` haven't run.

        UI usage
        --------
        Bound to the "Validate" button on the *Inference Console*.
        The routing-table card renders ``report.cluster_decisions``;
        the alert banner renders ``report.errors`` / ``report.warnings``.
        The Run button is enabled only when ``report.is_valid`` is True.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before validate_scenarios()."
            )
        if self._loaded_scenarios is None:
            raise RuntimeError(
                "load_scenarios() must be called before validate_scenarios()."
            )

        self._emit(event(
            STAGE_INFERENCE, "Validating scenarios",
            status=STATUS_RUNNING, target=self.ensemble_version,
        ))

        scenario_labels = self._loaded_scenarios.scenario_labels
        shock_rfs       = sorted(self._new_scenario_shocks.keys())

        decisions: List[ClusterRoutingDecision] = []
        errors:    List[str]                    = []
        warnings:  List[str]                    = []

        # --- Per-cluster classification + eligibility check ---
        for cid in self._ens_config.cluster_ids:
            ctx = self._inference_contexts.get(cid)
            if ctx is None:
                errors.append(f"Cluster '{cid}': no inference context loaded.")
                continue

            decision = self._build_routing_decision(
                cluster_id         = cid,
                ctx                = ctx,
                shock_risk_factors = shock_rfs,
                scenario_labels    = scenario_labels,
            )

            # Surface cheap-path eligibility failures into the report
            # so the UI can render them in the alert banner.
            if not decision.is_affected and decision.missing_scenario_labels:
                missing = decision.missing_scenario_labels
                errors.append(
                    f"Cluster '{cid}' is unaffected but its history "
                    f"is missing {len(missing)}/{len(scenario_labels)} "
                    f"requested scenario labels (first 3: {missing[:3]})."
                )

            decisions.append(decision)

        # --- Cross-cluster sanity check ---
        if decisions and not any(d.is_affected for d in decisions):
            errors.append(
                "No clusters intersect any shocked risk factor — nothing to run."
            )

        # --- Build and cache the report ---
        report = ValidationReport(
            ensemble_version  = self.ensemble_version,
            n_scenarios       = self._loaded_scenarios.n_scenarios,
            scenario_labels   = scenario_labels,
            cluster_decisions = decisions,
            errors            = errors,
            warnings          = warnings,
        )
        self._validation_report = report

        self._emit(event(
            STAGE_INFERENCE, "Validation complete",
            status=STATUS_OK if report.is_valid else STATUS_FAIL,
            target=f"{report.affected_count} affected · {report.unaffected_count} unaffected",
            detail=f"{len(report.errors)} error(s)" if report.errors else None,
        ))
        return report

    def validate_new_trades(self) -> ValidationReport:
        """Validate the loaded new-trades payload.

        Counterpart to :meth:`validate_scenarios` for the
        ``new_trades`` input mode.  Will populate
        ``self._validation_report`` with one routing decision per
        cluster — likely a different dataclass shape (a future
        ``TradeRoutingDecision``) sitting alongside
        :class:`ClusterRoutingDecision` in the union typing of
        ``ValidationReport.cluster_decisions``.

        Raises
        ------
        RuntimeError
            If :meth:`load` or :meth:`load_new_trades` haven't run.
        NotImplementedError
            Always — pending the matching trade-routing logic.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before validate_new_trades()."
            )
        if self._loaded_new_trades is None:
            raise RuntimeError(
                "load_new_trades() must be called before validate_new_trades()."
            )
        raise NotImplementedError(
            "new_trades validation is not yet implemented. The "
            "plug-in point is here — populate self._validation_report "
            "with one ClusterRoutingDecision (or TradeRoutingDecision) "
            "per cluster and return it."
        )

    def run_inference(self) -> InferenceResult:
        """Build inputs (per-cluster routed), predict, post-process.

        Final stage of the staged pipeline.  Mode-agnostic: dispatches
        on ``config.metadata['inference']['input_mode']`` via
        :meth:`_build_member_inputs`.

        Five steps:

          1. Prerequisite checks  (load / load_* / validate_* completed).
          2. Build member inputs  (per-mode, per-cluster routing inside).
          3. Forward pass         (``EnsembleModel.predict``).
          4. Build the result     (:meth:`_build_result`).
          5. Post-inference       (logging + optional artifact writes).

        Raises
        ------
        RuntimeError
            If any prior stage is missing, or the cached validation
            report contains errors (``report.is_valid`` is False).

        Returns
        -------
        InferenceResult
        """
        # --- Step 1: prerequisite checks ---
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before run_inference()."
            )
        if self._loaded_scenarios is None and self._loaded_new_trades is None:
            raise RuntimeError(
                "load_scenarios() or load_new_trades() must be called "
                "before run_inference()."
            )
        if self._validation_report is None:
            raise RuntimeError(
                "validate_scenarios() (or validate_new_trades()) "
                "must be called before run_inference()."
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

        # --- Step 2: build member inputs (per-mode dispatch) ---
        t0         = time.perf_counter()
        infer_meta = self.config.metadata.get("inference", {})
        input_mode = infer_meta.get("input_mode", "new_scenarios")
        batch_size = int(infer_meta.get("batch_size", 128))
        member_inputs, extra_meta = self._build_member_inputs(input_mode)

        # --- Step 3: forward pass — predict per cluster, write parquets ---
        #
        # We iterate cluster-by-cluster (rather than the legacy
        # ``self._ensemble.predict(member_inputs)`` one-shot call) so we
        # can:
        #
        #   * use ``predict_member_chunked`` to bound activation peak
        #     memory per cluster,
        #   * call ``_post_infer_cluster`` immediately after each
        #     cluster's predictions land — flushing wide parquets to
        #     disk so the in-memory footprint stays bounded across
        #     long runs, and
        #   * free ``member_inputs[cid]`` as soon as its predictions
        #     are computed (deep-copied risk_factor_shocks are the
        #     biggest contributor and never need to outlive the
        #     forward pass).
        #
        # ``_ensemble._combine`` is called once at the end with the
        # full per-cluster prediction dict — same aggregation contract
        # as ``predict``, just sourced one cluster at a time.
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        artifacts_root: Optional[Path] = None
        if self.config.artifacts_dir:
            artifacts_root = Path(self.config.artifacts_dir) / INFERENCE_DIRNAME
            artifacts_root.mkdir(parents=True, exist_ok=True)

        scenario_labels   = list(self._validation_report.scenario_labels)
        cluster_artifacts: List[ClusterArtifact]    = []
        member_preds:      Dict[str, np.ndarray]    = {}

        self._emit(event(
            STAGE_INFERENCE, "Forward pass started",
            status=STATUS_RUNNING,
            target=(
                f"{self._validation_report.affected_count} affected · "
                f"{self._validation_report.unaffected_count} unaffected"
            ),
        ))

        for cid in self._ensemble.router.cluster_ids:
            if cid not in self._ensemble.members:
                logger.warning(
                    "run_inference: no member model for cluster %s; "
                    "skipping (router/ensemble drift).", cid,
                )
                continue
            cluster_inputs = member_inputs.get(cid)
            if cluster_inputs is None:
                continue

            # Per-cluster memory-bounded forward pass.  Returns
            # predictions in the model's SCALED output space —
            # transform back to original units inside
            # _post_infer_cluster via transform_predictions.
            cluster_preds = HybridGnnRnnInferencePipeline.predict_member_chunked(
                ensemble       = self._ensemble,
                cluster_id     = cid,
                cluster_inputs = cluster_inputs,
                batch_size     = batch_size,
            )
            member_preds[cid] = cluster_preds

            # Stage 6: persist scaled + original parquets and capture
            # a lightweight artifact handle for downstream aggregation.
            # Only written when artifacts_dir is configured — pure
            # in-memory runs still produce an InferenceResult, just
            # without per-cluster parquets.
            if artifacts_root is not None:
                ctx = self._inference_contexts.get(cid)
                if ctx is None:
                    logger.warning(
                        "run_inference: no inference context for cluster %s; "
                        "skipping per-cluster artifact write.", cid,
                    )
                else:
                    cluster_artifacts.append(
                        self._post_infer_cluster(
                            cluster_id      = cid,
                            cluster_preds   = cluster_preds,
                            scenario_labels = scenario_labels,
                            ctx             = ctx,
                            out_dir         = artifacts_root,
                        )
                    )

            # Release the per-cluster input dict as soon as we no
            # longer need it.  Affected clusters carry deep-copied
            # risk_factor_shocks which are by far the largest object
            # in the member input dict.
            member_inputs[cid] = None

        if not member_preds:
            raise RuntimeError(
                "run_inference: no member produced predictions — "
                "ensemble may be empty or all clusters were skipped."
            )

        # Single aggregation call across all per-cluster predictions —
        # identical contract to EnsembleModel.predict, just with the
        # iteration lifted into this pipeline so we can interleave
        # per-cluster post-processing above.
        combined = self._ensemble._combine(member_preds)   # noqa: SLF001 — internal-but-stable

        self._emit(event(
            STAGE_INFERENCE, "Forward pass complete",
            status=STATUS_OK, target=f"{combined.shape[0]} scenarios",
        ))

        # --- Step 4: build result ---
        result = self._build_result(combined, infer_meta, extra_meta)
        result.latency_seconds = time.perf_counter() - t0

        # --- Step 5: post-inference (manifest + run-level summaries) ---
        self.post_infer(result, cluster_artifacts=cluster_artifacts)
        return result

    # ==================================================================
    # Loading
    # ==================================================================

    def _load_from_registry(self) -> None:
        """Cold-load: read ensemble + per-cluster contexts from registry.

        Three layers, one event per layer:

          1. Ensemble config + member versions (``EnsembleRegistry``).
          2. Member models, assembled into one ``EnsembleModel``
             (``EnsembleBuilder`` + ``ModelRegistry``).
          3. Per-cluster inference contexts
             (``load_inference_context_from_dir``).

        Step 3 reads the same on-disk artifacts that
        ``hybrid_gnn_rnn/infer.py`` consumes for single-cluster
        inference; the encoder / graph-builder dispatch happens
        inside ``_dict_to_inference_context``.
        """
        # --- Step 1: ensemble config + member versions ---
        self._emit(event(
            STAGE_INFERENCE, "Loading ensemble from registry",
            status=STATUS_RUNNING, target=self.ensemble_version,
        ))
        ens_registry = EnsembleRegistry(self.config.registry_dir)
        config, member_versions, version = ens_registry.load(self.ensemble_version)
        self._ens_config      = config
        self._member_versions = member_versions

        # --- Step 2: assemble the ensemble model ---
        from src.rade_ml_pt.registry.store import ModelRegistry

        model_registry = ModelRegistry(self.config.registry_dir)
        builder        = EnsembleBuilder(model_registry)
        self._ensemble = builder.build(config, member_versions)
        self._emit(event(
            STAGE_INFERENCE, "Ensemble assembled",
            status=STATUS_OK, target=f"{config.n_members} members",
        ))

        # --- Step 3: per-cluster inference contexts ---
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            load_inference_context_from_dir,
        )

        for cid in config.cluster_ids:
            ver         = member_versions[cid]
            version_dir = Path(self.config.registry_dir) / ver
            self._emit(event(
                STAGE_INFERENCE, "Cluster context loading",
                status=STATUS_RUNNING, target=cid,
            ))
            try:
                raw = load_inference_context_from_dir(version_dir)
                self._inference_contexts[cid] = _dict_to_inference_context(raw)
            except Exception as exc:
                self._emit(event(
                    STAGE_INFERENCE, "Cluster context failed",
                    status=STATUS_FAIL, target=cid, detail=str(exc),
                ))
                raise ValueError(
                    f"Could not load inference context for cluster '{cid}' "
                    f"(version '{ver}', dir={version_dir}). Ensure the "
                    f"model-specific inference artifacts are present. "
                    f"Root cause: {exc}"
                ) from exc
            self._emit(event(
                STAGE_INFERENCE, "Cluster context loaded",
                status=STATUS_OK, target=cid,
            ))

        logger.info(
            "Loaded ensemble '%s' (%d members) from registry",
            version, config.n_members,
        )

    def _load_from_session(self) -> None:
        """Warm-load: reuse the session's pre-loaded models + contexts.

        Called by :meth:`load` only after the session has been
        validated (matching version, fully loaded).  Just copies
        references — no IO, no model construction.
        """
        self._emit(event(
            STAGE_INFERENCE, "Reusing pre-loaded session",
            status=STATUS_RUNNING, target=self._session.ensemble_version,
        ))

        # Copy refs from the session (no IO).
        self._ens_config      = self._session.config
        self._member_versions = self._session.member_versions
        self._ensemble        = self._session.ensemble_model

        # Per-cluster inference contexts can live on the session as
        # either already-typed InferenceContext objects or the raw
        # dicts returned by load_inference_context_from_dir (legacy
        # sessions).  Normalise to InferenceContext here so downstream
        # code only deals with one shape.
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

        for cid in self._ens_config.cluster_ids:
            state = self._session._inference[cid]
            ctx   = state.inference_context
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

    def _build_routing_decision(
        self,
        cluster_id: str,
        ctx: "InferenceContext",
        shock_risk_factors: Iterable[str],
        scenario_labels: List[str],
    ) -> ClusterRoutingDecision:
        """Assemble one cluster's routing decision from primitives.

        Calls into :class:`HybridGnnRnnInferencePipeline` for the
        cluster-level classification work (RF intersection, cheap-path
        eligibility), then wraps the result + trade counts in the
        ensemble-level :class:`ClusterRoutingDecision` dataclass.

        Trade counts are pulled from ``ctx.trade_universe`` so the
        returned decision is rich enough to drive the UI's routing-
        table card without any further context lookups.
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        intersecting = HybridGnnRnnInferencePipeline.intersecting_risk_factors(
            ctx, shock_risk_factors,
        )
        is_affected  = bool(intersecting)

        # Eligibility check only matters for unaffected clusters
        # (affected clusters take the re-pricing path, which doesn't
        # depend on the historical label index).
        missing: List[str] = []
        if not is_affected:
            missing = HybridGnnRnnInferencePipeline.missing_scenario_labels(
                ctx, scenario_labels,
            )

        n_elementary = 0
        n_target     = 0
        if ctx.trade_universe is not None:
            n_elementary = len(ctx.trade_universe.get("elementary_ids") or [])
            n_target     = len(ctx.trade_universe.get("target_ids")     or [])

        return ClusterRoutingDecision(
            cluster_id                = cluster_id,
            is_affected               = is_affected,
            intersecting_risk_factors = intersecting,
            n_elementary_trades       = n_elementary,
            n_target_trades           = n_target,
            missing_scenario_labels   = missing,
        )

    # ==================================================================
    # Input building
    # ==================================================================

    def _build_member_inputs(
        self,
        input_mode: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Mode dispatcher for input building.

        Called from :meth:`run_inference` after the matching
        ``validate_*`` method has populated ``self._validation_report``.

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

    def _build_new_scenarios_inputs(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Per-cluster routing loop for new-scenarios mode.

        Walks ``self._validation_report.cluster_decisions`` in order
        and delegates each cluster's input building to
        :meth:`HybridGnnRnnInferencePipeline.build_new_scenario_inputs`,
        which internally dispatches on ``decision.is_affected``:

        * **Affected** clusters take the full re-pricing path.
        * **Unaffected** clusters take the cheap historical-lookup path.

        Returns
        -------
        tuple of (member_inputs, extra_meta)
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        assert self._validation_report is not None, \
            "validate_scenarios() must run first"
        assert self._new_scenario_shocks is not None, \
            "load_scenarios() must run first"

        member_inputs: Dict[str, Any] = {}
        extra_meta:    Dict[str, Any] = {"sample_ids": {}}

        scenario_labels = self._validation_report.scenario_labels
        shocks          = self._new_scenario_shocks

        for decision in self._validation_report.cluster_decisions:
            cid  = decision.cluster_id
            ctx  = self._inference_contexts[cid]
            path = "affected" if decision.is_affected else "unaffected"

            self._emit(event(
                STAGE_INFERENCE, "Building cluster inputs",
                status=STATUS_RUNNING, target=cid,
                detail=f"path={path}",
            ))
            try:
                result = HybridGnnRnnInferencePipeline.build_new_scenario_inputs(
                    ctx                 = ctx,
                    new_scenario_shocks = shocks,
                    scenario_labels     = scenario_labels,
                    is_affected         = decision.is_affected,
                )

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

    # ==================================================================
    # Post-inference (Stage 6)
    #
    # Three responsibilities:
    #
    #   1. ``_post_infer_cluster`` — per-cluster: inverse-scale +
    #      restore notional sign via ``transform_predictions``, write
    #      ``_scaled.parquet`` / ``_original.parquet``, return a
    #      ``ClusterArtifact`` handle.
    #   2. ``post_infer`` — run-level: aggregate the per-cluster
    #      summaries into ``cluster_predictions.parquet`` and
    #      ``portfolio_predictions.parquet``, then write the
    #      manifest.
    #   3. ``_write_run_manifest`` — emit the canonical
    #      ``manifest.json`` the API result-reader and the dashboard
    #      use as the entry point to every artifact this run
    #      produced.
    # ==================================================================

    def _post_infer_cluster(
        self,
        cluster_id:      str,
        cluster_preds:   np.ndarray,
        scenario_labels: List[str],
        ctx:             "InferenceContext",
        out_dir:         Path,
    ) -> ClusterArtifact:
        """Transform one cluster's scaled predictions and persist them.

        Steps
        -----
        1. Call :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
           to get ``(scaled_wide, original_wide, summary_df)``.
        2. Write both wide DataFrames to
           ``<out_dir>/trade_predictions/<cluster_id>_{scaled,original}.parquet``.
           Wide-format is intentional — the dashboard's per-cluster
           trade drill-down reads scenarios as rows and trades as
           columns, matching the parquet layout directly.
        3. Emit an activity event so the UI can render progress
           per cluster.
        4. Return a :class:`ClusterArtifact` carrying the summary
           frame (for post_infer aggregation) and the parquet
           paths (for the manifest).

        Returns
        -------
        ClusterArtifact
            Lightweight handle suitable for accumulation in a list
            without retaining the heavy wide frames (those have
            already been flushed to disk).
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        scaled_wide, original_wide, summary_df = (
            HybridGnnRnnInferencePipeline.transform_predictions(
                cluster_id      = cluster_id,
                cluster_preds   = cluster_preds,
                scenario_labels = scenario_labels,
                ctx             = ctx,
            )
        )

        # Per-cluster trade-level outputs live under
        # ``trade_predictions/<cid>_<space>.parquet`` — symmetric with
        # the eval pipeline's ``members/<cid>/predictions/<split>.npz``
        # convention (one subdirectory per artifact family).
        trade_predictions_dir = out_dir / TRADE_PREDICTIONS_DIRNAME
        trade_predictions_dir.mkdir(parents=True, exist_ok=True)
        scaled_path   = trade_predictions_dir / f"{cluster_id}_scaled.parquet"
        original_path = trade_predictions_dir / f"{cluster_id}_original.parquet"

        scaled_wide.to_parquet(scaled_path)
        original_wide.to_parquet(original_path)

        n_scenarios = int(len(scaled_wide))
        n_trades    = int(len(scaled_wide.columns))
        trade_ids   = list(scaled_wide.columns)

        self._emit(event(
            STAGE_INFERENCE,
            f"Wrote cluster artifacts",
            status = STATUS_OK,
            target = cluster_id,
            detail = f"{n_trades} trades × {n_scenarios} scenarios",
        ))

        return ClusterArtifact(
            cluster_id    = cluster_id,
            n_trades      = n_trades,
            n_scenarios   = n_scenarios,
            scaled_path   = str(scaled_path),
            original_path = str(original_path),
            trade_ids     = trade_ids,
            summary_df    = summary_df,
        )

    def post_infer(
        self,
        result:            InferenceResult,
        *,
        cluster_artifacts: Optional[List[ClusterArtifact]] = None,
    ) -> None:
        """Aggregate per-cluster artifacts and write run-level summaries.

        Replaces the legacy flat-CSV writer.  When invoked with no
        ``cluster_artifacts`` (e.g. by a programmatic caller running
        the pipeline against an in-memory configuration) the method
        is a no-op aside from the summary log line — preserving
        backwards compatibility for callers that don't care about
        on-disk artifacts.

        When called from :meth:`run_inference` with the accumulated
        list of :class:`ClusterArtifact` objects, three outputs land
        under ``<artifacts_dir>/inference/``:

          * ``cluster_summary/cluster_predictions.parquet`` — long-format
            per-cluster × per-scenario aggregate stats.  Stacked
            directly from ``ClusterArtifact.summary_df`` so the column
            schema matches :meth:`transform_predictions`'s contract.
          * ``portfolio_summary/portfolio_predictions.parquet`` —
            per-scenario portfolio totals (sum across clusters per
            scenario).  ``sum_pnl_*`` is sum-additive across clusters;
            mean / std / min / max are not, so we re-derive them from
            ``sum_pnl_original`` at the portfolio level (one value
            per scenario).
          * ``manifest.json`` — canonical entry-point pointing at
            every artifact this run produced (see
            :meth:`_write_run_manifest`).

        Parameters
        ----------
        result
            The :class:`InferenceResult` :meth:`_build_result` just
            produced — used for the summary log line and the
            ``n_scenarios`` / ``latency_seconds`` fields in the
            manifest.
        cluster_artifacts
            The list :meth:`run_inference` accumulated from
            per-cluster ``_post_infer_cluster`` calls.  None ⇒ this
            invocation came from outside :meth:`run_inference`;
            parquets + manifest are not written.
        """
        if result.predictions is not None:
            preds = result.predictions
            logger.info(
                "Ensemble inference summary: n_samples=%d, mean=%.4f, "
                "std=%.4f, min=%.4f, max=%.4f",
                result.n_samples, np.mean(preds), np.std(preds),
                np.min(preds), np.max(preds),
            )

        if not self.config.artifacts_dir or cluster_artifacts is None:
            return

        out_dir = Path(self.config.artifacts_dir) / INFERENCE_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)

        cluster_summary_path:   Optional[Path] = None
        portfolio_summary_path: Optional[Path] = None

        if cluster_artifacts:
            # Stack per-cluster summary frames into one long DataFrame.
            # transform_predictions guarantees the column schema is
            # uniform across clusters, so a vanilla concat is safe.
            cluster_summary_df  = pd.concat(
                [a.summary_df for a in cluster_artifacts if a.summary_df is not None],
                ignore_index=True,
            )
            cluster_summary_dir = out_dir / CLUSTER_SUMMARY_DIRNAME
            cluster_summary_dir.mkdir(parents=True, exist_ok=True)
            cluster_summary_path = cluster_summary_dir / CLUSTER_SUMMARY_FILENAME
            cluster_summary_df.to_parquet(cluster_summary_path)

            # Portfolio-level: sum sum_pnl_* across clusters per
            # scenario (these are linear-additive).  Mean / std /
            # min / max are NOT additive across clusters so we
            # surface them as the portfolio-level aggregates derived
            # from per-cluster sums — one row per scenario.
            portfolio_summary_df  = (
                cluster_summary_df
                .groupby("scenario_label", as_index=False)
                .agg(
                    sum_pnl_scaled   = ("sum_pnl_scaled",   "sum"),
                    sum_pnl_original = ("sum_pnl_original", "sum"),
                    n_clusters       = ("cluster_id",       "nunique"),
                )
            )
            portfolio_summary_dir = out_dir / PORTFOLIO_SUMMARY_DIRNAME
            portfolio_summary_dir.mkdir(parents=True, exist_ok=True)
            portfolio_summary_path = portfolio_summary_dir / PORTFOLIO_SUMMARY_FILENAME
            portfolio_summary_df.to_parquet(portfolio_summary_path)

            self._emit(event(
                STAGE_INFERENCE, "Run summaries written",
                status = STATUS_OK,
                target = f"{len(cluster_artifacts)} clusters",
            ))

        manifest_path = self._write_run_manifest(
            out_dir                = out_dir,
            result                 = result,
            cluster_artifacts      = cluster_artifacts,
            cluster_summary_path   = cluster_summary_path,
            portfolio_summary_path = portfolio_summary_path,
        )
        logger.info("Inference manifest written to %s", manifest_path)
        self._emit(event(
            STAGE_INFERENCE, "Manifest written",
            status = STATUS_OK,
            target = str(manifest_path.name),
        ))

    def _write_run_manifest(
        self,
        out_dir:                Path,
        result:                 InferenceResult,
        cluster_artifacts:      List[ClusterArtifact],
        cluster_summary_path:   Optional[Path],
        portfolio_summary_path: Optional[Path],
    ) -> Path:
        """Write the canonical ``manifest.json`` describing the run.

        The manifest is the result-reader's entry point — the
        dashboard fetches it first, then deep-links to whichever
        per-cluster / run-level parquets it needs.  Schema is
        intentionally permissive (values are not validated by a
        pydantic model on the server side) so future fields can be
        added without breaking older clients.

        Returns
        -------
        Path
            Path of the manifest that was written (always
            ``<out_dir>/manifest.json``).
        """
        manifest: Dict[str, Any] = {
            "schema_version":          "1",
            "generated_at":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ensemble_version":        self.ensemble_version,
            "input_mode":              (result.metadata or {}).get(
                "input_mode", "new_scenarios",
            ),
            "n_scenarios":             int(result.n_samples) if result.n_samples else 0,
            "scenario_labels": (
                list(self._validation_report.scenario_labels)
                if self._validation_report is not None else []
            ),
            "clusters": [a.to_manifest_dict() for a in (cluster_artifacts or [])],
            "cluster_summary_path":    str(cluster_summary_path) if cluster_summary_path   else None,
            "portfolio_summary_path":  str(portfolio_summary_path) if portfolio_summary_path else None,
            "validation": (
                self._validation_report.to_dict()
                if self._validation_report is not None else None
            ),
            "scenarios": (
                self._loaded_scenarios.to_dict()
                if self._loaded_scenarios is not None else None
            ),
            "latency_seconds":         (
                float(result.latency_seconds)
                if getattr(result, "latency_seconds", None) is not None else None
            ),
        }

        path = out_dir / MANIFEST_FILENAME
        path.write_text(json.dumps(manifest, indent=2))
        return path

    # ==================================================================
    # Result building
    # ==================================================================

    def _build_result(
        self,
        combined: np.ndarray,
        infer_meta: Dict[str, Any],
        extra_meta: Dict[str, Any],
    ) -> InferenceResult:
        """Wrap aggregated predictions into an :class:`InferenceResult`.

        Always called from :meth:`run_inference`, whose prerequisite
        checks guarantee that the ensemble model + validation report
        are populated by the time we land here.

        The result's ``metadata`` carries everything the UI needs to
        render the run summary card: input mode, cluster identifiers,
        routing breakdown (affected/unaffected lists, cheap-path
        flag), scenario labels, and per-member sample IDs.
        """
        assert self._validation_report is not None, \
            "_build_result called before validation (run_inference should guard this)"

        meta: Dict[str, Any] = {
            "input_mode":          infer_meta.get("input_mode", "new_scenarios"),
            "cluster_ids":         self._ensemble.router.cluster_ids if self._ensemble else [],
            "n_members":           self._ens_config.n_members if self._ens_config else 0,
            "affected_clusters":   self._validation_report.affected_cluster_ids,
            "unaffected_clusters": self._validation_report.unaffected_cluster_ids,
            "cheap_path_used":     self._validation_report.cheap_path_used,
            "scenario_labels":     self._validation_report.scenario_labels,
        }
        if extra_meta.get("sample_ids"):
            meta["per_member_sample_ids"] = extra_meta["sample_ids"]

        # Flatten per-member sample IDs into a single ordered list,
        # falling back to whatever the caller stashed in infer_meta.
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
            predictions   = combined,
            n_samples     = combined.shape[0],
            sample_ids    = all_sample_ids or infer_meta.get("sample_ids"),
            model_version = self.ensemble_version,
            metadata      = meta,
        )
```

---

### A · 2 — `src/rade_ml_pt/ensemble/api/services/inference_state.py` (Stages 7-9)

Run-state holder for the API. Owns `InferenceRunState` (one per active
run) and `InferenceStateManager` (process-wide singleton). Stage 9 adds
the `_thread` field on the state and `start_run_in_background()` on the
manager for the non-blocking `/run` endpoint.

```python
"""Inference run-state holder for the PRISM API.

Holds the live :class:`EnsembleInferencePipeline` between HTTP requests
so the staged workflow (``load → load_scenarios → validate →
run_inference``) survives the stateless boundary between calls.

State model — Option A (single-user, single-process)
----------------------------------------------------
The API today holds **one** :class:`InferenceRunState` at a time on
the :class:`InferenceStateManager`.  Calling ``create_run`` while a
prior run exists replaces it (the prior pipeline and its activity log
are dropped).  Sufficient for the single-user dashboard use case.

Migrating to multi-user (Option B) is a small change
----------------------------------------------------
All public methods of :class:`InferenceStateManager` already take a
``run_id`` argument; today there is only one and the manager validates
that callers pass it through correctly.  Moving to per-user / per-run
isolation is then a localised refactor:

  * Replace the single ``_active`` slot with
    ``_runs: Dict[str, InferenceRunState]``.
  * Drop the "single active run" assertion in
    :meth:`InferenceStateManager.create_run`.
  * Surface ``run_id`` in the API by upgrading each router method to
    take it from the URL / a header instead of the singleton.

The router and dependency call sites do **not** need to change.

Threading
---------
Stage 9 (background-task execution for ``/run``) will dispatch
:meth:`EnsembleInferencePipeline.run_inference` on a worker thread.
:class:`EventCollector` is already thread-safe so the activity-log
contract holds; the rest of the state object is read-only after
``create_run`` returns, save for ``status`` / ``last_error`` /
``artifacts_dir`` which are written exactly once each.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.pipelines.ensemble.infer import EnsembleInferencePipeline
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    ActivityEntry,
    EventCollector,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Status enum (str-valued so it round-trips through JSON cleanly)
# ──────────────────────────────────────────────────────────────────────

# Lifecycle of one run.  String-typed for transparent JSON encoding;
# kept as module-level constants rather than an Enum because the wire
# shape is just ``str`` and the router doesn't need branching logic on
# the type.
STATUS_CREATED:     str = "created"
STATUS_LOADING:     str = "loading"
STATUS_LOADED:      str = "loaded"
STATUS_SCENARIOS:   str = "scenarios_loaded"
STATUS_VALIDATED:   str = "validated"
STATUS_RUNNING:     str = "running"
STATUS_COMPLETE:    str = "complete"
STATUS_FAILED:      str = "failed"


# ──────────────────────────────────────────────────────────────────────
# Per-run state
# ──────────────────────────────────────────────────────────────────────

@dataclass
class InferenceRunState:
    """One inference run's mutable state.

    Built once by :meth:`InferenceStateManager.create_run`; thereafter
    only the lifecycle fields (``status``, ``last_error``,
    ``artifacts_dir``, ``manifest_path``) are mutated.

    Attributes
    ----------
    run_id
        Stable identifier for this run.  Today derived from
        ``<ensemble_version>__<UTC_yyyymmdd_HHMMSS>``; not a UUID so
        run IDs are human-greppable in logs and on disk.
    ensemble_version
        Ensemble version this run was constructed against.
    pipeline
        Live pipeline instance.  All staged operations (load,
        load_scenarios, validate_scenarios, run_inference) are
        invoked through this object.
    activity_log
        Thread-safe buffer of events emitted by the pipeline.  Passed
        in as ``on_event`` at pipeline construction; readable via
        ``activity_log.snapshot()`` for HTTP polling.
    status
        Current lifecycle state — one of the ``STATUS_*`` constants.
    created_at
        UTC ISO-8601 timestamp at construction.
    last_error
        Detail message for the most recent failure (set whenever
        ``status`` transitions to ``STATUS_FAILED``).  ``None`` when
        no failure has occurred.
    artifacts_dir
        Resolved per-run artifacts directory.  Set when the run
        actually executes (``run_inference``) so the API can locate
        manifest / parquet outputs afterwards.
    manifest_path
        Convenience pointer to ``<artifacts_dir>/inference/manifest.json``
        populated once the run completes successfully.  ``None`` until
        then.
    """

    run_id:           str
    ensemble_version: str
    pipeline:         EnsembleInferencePipeline
    activity_log:     EventCollector
    status:           str           = STATUS_CREATED
    created_at:       str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    last_error:       Optional[str] = None
    artifacts_dir:    Optional[str] = None
    manifest_path:    Optional[str] = None

    # Background-execution machinery (Stage 9).  Populated when the
    # ``/run`` router dispatches ``pipeline.run_inference()`` onto a
    # worker thread.  Excluded from ``repr`` so log lines stay readable.
    _thread:          Optional[threading.Thread] = field(default=None, repr=False)

    @property
    def is_alive(self) -> bool:
        """True iff a background run thread is currently executing.

        Distinct from ``status == STATUS_RUNNING`` because the worker
        transitions the run to COMPLETE / FAILED *inside* the thread —
        there's a small window where ``status`` is already terminal but
        ``Thread.is_alive()`` is still True until the worker unwinds.
        ``is_alive`` is the authoritative concurrency probe; ``status``
        is the user-visible state.
        """
        return self._thread is not None and self._thread.is_alive()

    def transition(self, new_status: str, *, error: Optional[str] = None) -> None:
        """Move the run to ``new_status`` and clear / set ``last_error``.

        Single mutator for the run lifecycle so the status / error
        invariant (``last_error`` is only set when status ⇒ FAILED) is
        enforced in one place.

        Parameters
        ----------
        new_status
            Target lifecycle state (one of the ``STATUS_*`` constants).
        error
            Detail message — only respected when transitioning to
            :data:`STATUS_FAILED`.  Pass ``None`` to clear.
        """
        self.status = new_status
        self.last_error = error if new_status == STATUS_FAILED else None


# ──────────────────────────────────────────────────────────────────────
# State manager
# ──────────────────────────────────────────────────────────────────────

class InferenceStateManager:
    """Single source of truth for the API's active inference run.

    Today (Option A) holds **one** :class:`InferenceRunState` at a time.
    All accessors take ``run_id`` explicitly so the future Option B
    refactor (per-run dict) is mechanical.
    """

    def __init__(self) -> None:
        # The single active run, or None when the server hasn't been
        # asked to construct one yet.  Switching to a dict for Option
        # B replaces only this slot.
        self._active: Optional[InferenceRunState] = None
        self._lock = threading.Lock()

    # ── Construction / lookup ─────────────────────────────────────

    def create_run(
        self,
        ensemble_config:  EnsembleConfig,
        ensemble_version: str,
    ) -> InferenceRunState:
        """Construct a new pipeline + state object and make it the active run.

        Any previously-active run is silently replaced — the prior
        pipeline and its activity log are dropped.  Sufficient for the
        single-user dashboard.  Option B replaces this with a dict
        insertion keyed on ``run_id``.

        Parameters
        ----------
        ensemble_config
            Already-populated :class:`EnsembleConfig`.  Constructed by
            the router from the API settings (``registry_dir``,
            ``artifacts_dir``).
        ensemble_version
            Ensemble version or tag (passed through to the pipeline's
            ``ensemble_version`` constructor argument).

        Returns
        -------
        InferenceRunState
            The freshly registered run.  Use ``state.run_id`` for any
            subsequent operation against this run.
        """
        ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{ensemble_version}__{ts}"

        # Construct the activity-log collector first; the pipeline
        # captures the callable in __init__ and emits into it for
        # every subsequent stage.
        activity_log = EventCollector()

        pipeline = EnsembleInferencePipeline(
            ensemble_config  = ensemble_config,
            ensemble_version = ensemble_version,
            session          = None,           # cold-load path for v1
            on_event         = activity_log,
        )

        state = InferenceRunState(
            run_id           = run_id,
            ensemble_version = ensemble_version,
            pipeline         = pipeline,
            activity_log     = activity_log,
        )

        with self._lock:
            self._active = state

        logger.info("InferenceStateManager: created run %s", run_id)
        return state

    def get_run(self, run_id: Optional[str] = None) -> InferenceRunState:
        """Look up a run by ID.

        When ``run_id`` is None (the v1 single-user convention) the
        currently-active run is returned.  Once Option B lands this
        argument becomes mandatory; callers that pass ``None`` will
        need to be updated then.

        Raises
        ------
        RuntimeError
            If no run has been created yet, or the requested
            ``run_id`` doesn't match the active run.
        """
        with self._lock:
            state = self._active

        if state is None:
            raise RuntimeError(
                "No active inference run — call POST /inference/load first."
            )
        if run_id is not None and run_id != state.run_id:
            raise RuntimeError(
                f"Inference run '{run_id}' not found; active run is "
                f"'{state.run_id}'."
            )
        return state

    @property
    def has_active_run(self) -> bool:
        """True iff at least one run has been created."""
        with self._lock:
            return self._active is not None

    # ── Activity-log helpers (HTTP polling) ───────────────────────

    def events_since(
        self,
        run_id: Optional[str],
        cursor: int,
    ) -> Tuple[List[ActivityEntry], int]:
        """Return events emitted since the given cursor for one run.

        Used by ``GET /inference/events?cursor=N`` polling — the
        cursor is an opaque integer offset into the run's activity
        log.  The next cursor (= current log length) is returned
        alongside the slice so callers can chain polls.

        Parameters
        ----------
        run_id
            The run to read events from.  ``None`` ⇒ active run.
        cursor
            Number of events the caller has already seen.  Clamped to
            ``[0, len(log)]``.

        Returns
        -------
        events : list of ActivityEntry
            Events strictly after ``cursor`` in emit order.
        next_cursor : int
            Total events emitted so far.  Pass back on the next poll.
        """
        state    = self.get_run(run_id)
        snapshot = state.activity_log.snapshot()
        if cursor < 0:
            cursor = 0
        if cursor > len(snapshot):
            cursor = len(snapshot)
        return snapshot[cursor:], len(snapshot)

    # ── Background-execution helpers (Stage 9) ────────────────────

    def start_run_in_background(
        self,
        target: Callable[[], None],
    ) -> threading.Thread:
        """Spawn the background run thread for the active run.

        Single-thread-per-run by construction (Option A).  The caller
        has already transitioned ``state.status`` to ``STATUS_RUNNING``
        before invoking this; the ``target`` closure is responsible
        for the terminal transition (``STATUS_COMPLETE`` /
        ``STATUS_FAILED``) once the pipeline returns or raises.

        Daemon mode is on so a ``SIGTERM`` doesn't hang the server
        waiting for a long-running inference to finish — partial runs
        are recoverable from the on-disk manifest by future filesystem
        rehydration logic (Stage 9.5).

        Raises
        ------
        RuntimeError
            If a background thread is already running for this run —
            prevents accidental double-dispatch from a refreshed UI.
        """
        state = self.get_run()
        if state.is_alive:
            raise RuntimeError(
                f"Run {state.run_id} already has a background thread running."
            )

        thread = threading.Thread(
            target = target,
            name   = f"infer-{state.run_id}",
            daemon = True,
        )
        state._thread = thread        # noqa: SLF001 — module-internal
        thread.start()
        logger.info(
            "InferenceStateManager: started background thread for run %s",
            state.run_id,
        )
        return thread


# ──────────────────────────────────────────────────────────────────────
# Helpers used by router and dependency
# ──────────────────────────────────────────────────────────────────────

def build_ensemble_config(
    registry_dir:    str,
    artifacts_dir:   str,
    new_scenario_dir: Optional[str] = None,
) -> EnsembleConfig:
    """Construct a minimal :class:`EnsembleConfig` for an API-driven run.

    The configuration the pipeline needs at runtime is small —
    ``registry_dir`` (for cold-load) and ``artifacts_dir`` (where
    ``post_infer`` writes the manifest + parquets).  Any further
    metadata (``input_mode``, ``new_scenario_dir`` default) is plumbed
    through the ``metadata`` dict.

    Parameters
    ----------
    registry_dir
        Where the ensemble version + member models live on disk.
    artifacts_dir
        Per-run artifacts root (e.g.
        ``<base_artifacts>/inference_runs/<run_id>``).  ``post_infer``
        writes ``inference/manifest.json`` and friends under this path.
    new_scenario_dir
        Optional default new-scenario directory; the API's
        ``/scenarios`` call passes its own path explicitly so this is
        only used by the convenience ``run()`` orchestrator.

    Returns
    -------
    EnsembleConfig
        Minimal config sufficient for ``load → load_scenarios →
        validate_scenarios → run_inference``.
    """
    metadata = {"inference": {"input_mode": "new_scenarios"}}
    if new_scenario_dir is not None:
        metadata["inference"]["new_scenario_dir"] = new_scenario_dir

    return EnsembleConfig(
        registry_dir  = registry_dir,
        artifacts_dir = artifacts_dir,
        metadata      = metadata,
    )


def per_run_artifacts_dir(base_artifacts_dir: str, run_id: str) -> str:
    """Convention for where per-run artifacts go on disk.

    ``<base_artifacts_dir>/inference_runs/<run_id>``.  The pipeline's
    ``post_infer`` further nests an ``inference/`` directory underneath
    for the manifest + parquets, so the final manifest path is
    ``<base_artifacts_dir>/inference_runs/<run_id>/inference/manifest.json``.

    Centralised here so the router, the state manager, and any future
    listing endpoint agree on the layout.
    """
    return str(Path(base_artifacts_dir) / "inference_runs" / run_id)


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton
# ──────────────────────────────────────────────────────────────────────

_manager: Optional[InferenceStateManager] = None


def set_inference_state_manager(manager: InferenceStateManager) -> None:
    """Inject the manager at app-lifespan startup."""
    global _manager
    _manager = manager


def get_inference_state_manager() -> InferenceStateManager:
    """FastAPI ``Depends`` — returns the singleton manager."""
    if _manager is None:
        raise RuntimeError(
            "InferenceStateManager not initialised. Server not ready."
        )
    return _manager
```

---

### A · 3 — `src/rade_ml_pt/ensemble/api/services/result_reader.py` (Stage 10)

Reader-side counterpart to Stage 6's writers. Single source of truth
for the on-disk layout — imports the directory / filename constants
**directly** from `pipelines.ensemble.infer` (A.1) so writer and reader
cannot drift.

```python
"""Read-side counterpart to Stage 6's writers.

Serves the manifest + parquets written by
:meth:`EnsembleInferencePipeline.post_infer` for any inference run
on disk.  The dashboard's data plane reads exclusively through this
class; the control plane (``/load``, ``/scenarios``, ``/validate``,
``/run``, ``/status``, ``/events``) goes through
:class:`InferenceStateManager` for the *active* run.

On-disk layout
--------------
Mirrors the eval pipeline's "one subdir per artifact family"
convention.  The directory and filename constants are imported
from :mod:`pipelines.ensemble.infer` (the writer side) — single
source of truth, so writer and reader can never drift.  Per run::

    <base_artifacts_dir>/inference_runs/<run_id>/inference/
    ├── manifest.json                                    ← entry point
    ├── cluster_summary/
    │   └── cluster_predictions.parquet                  ← long-format
    ├── portfolio_summary/
    │   └── portfolio_predictions.parquet                ← long-format
    └── trade_predictions/
        ├── <cluster_id>_scaled.parquet                  ← wide, model space
        └── <cluster_id>_original.parquet                ← wide, original space

Companion to
------------
:class:`ArtifactReader` (eval artifacts).  Two readers rather than
one because:

* The layouts differ — eval is keyed by ensemble version, inference
  is keyed by run_id.
* The caching story differs — eval artifacts are updated in place
  for the active version (mtime cache); inference artifacts are
  append-only by run_id (no cache needed).

State model — Option A (process-wide singleton)
-----------------------------------------------
Mirrors :class:`InferenceStateManager`: one reader per process,
constructed at lifespan startup with the base artifacts dir from
:class:`Settings`.  Migrating to per-tenant readers (Option B for
multi-tenant deployments) is mechanical — wrap the singleton slot
in a dict keyed by tenant ID.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

# Single source of truth for directory / filename constants.  Defined
# alongside the writer in pipelines.ensemble.infer so both sides
# of the contract stay in lockstep.
from src.rade_ml_pt.pipelines.ensemble.infer import (
    CLUSTER_SUMMARY_DIRNAME,
    CLUSTER_SUMMARY_FILENAME,
    INFERENCE_DIRNAME,
    MANIFEST_FILENAME,
    PORTFOLIO_SUMMARY_DIRNAME,
    PORTFOLIO_SUMMARY_FILENAME,
    TRADE_PREDICTIONS_DIRNAME,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# On-disk layout constants (reader-only)
# ──────────────────────────────────────────────────────────────────────

# Subdirectory under ``settings.artifacts_dir`` where all inference
# runs land.  Matches :func:`per_run_artifacts_dir` in
# :mod:`services.inference_state` so the writer (which builds
# per-run artifacts dirs) and the reader (which discovers them)
# agree without cross-coupling.
INFERENCE_RUNS_DIR: str = "inference_runs"

# Re-export the inference subdirectory constant under its
# historical name so any external caller that imports
# ``INFERENCE_SUBDIR`` from this module keeps working.
INFERENCE_SUBDIR:   str = INFERENCE_DIRNAME

# Allowed values for the ``space=`` query parameter on
# ``GET /inference/runs/{id}/clusters/{cid}/trades``.  Defined as a
# module constant so the router can use it in OpenAPI docs and the
# reader can use it for validation.
VALID_SPACES = ("scaled", "original")


# ──────────────────────────────────────────────────────────────────────
# Per-run filesystem layout
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunPaths:
    """Resolved on-disk paths for one inference run.

    Frozen so the dataclass is hashable / cache-keyable.  Pure
    derived data — does NOT verify any path exists.  Callers check
    existence to raise the appropriate ``HTTPException`` themselves.

    Layout (single source of truth: ``pipelines.ensemble.infer``)::

        <inference_dir>/
        ├── manifest.json
        ├── cluster_summary/<CLUSTER_SUMMARY_FILENAME>
        ├── portfolio_summary/<PORTFOLIO_SUMMARY_FILENAME>
        └── trade_predictions/<cid>_{scaled,original}.parquet
    """

    run_id:                 str
    run_root:               Path   # <base>/inference_runs/<run_id>
    inference_dir:          Path   # <base>/inference_runs/<run_id>/inference
    manifest_path:          Path
    cluster_summary_path:   Path   # cluster_summary/cluster_predictions.parquet
    portfolio_summary_path: Path   # portfolio_summary/portfolio_predictions.parquet
    trade_predictions_dir:  Path   # holds <cid>_{scaled,original}.parquet

    def cluster_parquet(self, cluster_id: str, space: str) -> Path:
        """Resolve the per-cluster wide parquet path.

        ``space`` must be one of :data:`VALID_SPACES`; the API
        validates this upstream but we re-check defensively to keep
        the file-path layer self-contained.
        """
        if space not in VALID_SPACES:
            raise ValueError(
                f"space must be one of {VALID_SPACES}; got {space!r}"
            )
        return self.trade_predictions_dir / f"{cluster_id}_{space}.parquet"


# ──────────────────────────────────────────────────────────────────────
# Module-level loaders
# ──────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file into a plain dict."""
    return json.loads(path.read_text())


def _load_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a pandas DataFrame.

    Routed through ``pyarrow.parquet.read_table`` (rather than
    ``pd.read_parquet``) for consistency with :class:`ArtifactReader`
    and to keep pyarrow as the only parquet engine dependency.
    """
    return pq.read_table(path).to_pandas()


# ──────────────────────────────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────────────────────────────

class InferenceResultReader:
    """Typed accessor over the inference-runs directory.

    Instances are essentially stateless — every method resolves the
    on-disk layout from ``run_id`` and reads the file.  No mtime
    cache (runs are append-only) but typical access patterns are
    cold (one read per UI navigation event) so caching wouldn't pay
    for itself.

    Attributes
    ----------
    base_dir
        Resolved ``settings.artifacts_dir`` — root under which the
        ``inference_runs/<run_id>/inference/`` tree lives.
    """

    def __init__(self, base_artifacts_dir: str):
        self.base_dir = Path(base_artifacts_dir)

    # ── Path resolution ─────────────────────────────────────────────

    def paths_for(self, run_id: str) -> RunPaths:
        """Resolve the per-run filesystem layout.

        Pure path arithmetic — does NOT check that any file exists.
        The router does an existence check and raises a 404 / 409
        as appropriate.
        """
        run_root      = self.base_dir / INFERENCE_RUNS_DIR / run_id
        inference_dir = run_root / INFERENCE_SUBDIR
        return RunPaths(
            run_id                 = run_id,
            run_root               = run_root,
            inference_dir          = inference_dir,
            manifest_path          = inference_dir / MANIFEST_FILENAME,
            cluster_summary_path   = (
                inference_dir / CLUSTER_SUMMARY_DIRNAME / CLUSTER_SUMMARY_FILENAME
            ),
            portfolio_summary_path = (
                inference_dir / PORTFOLIO_SUMMARY_DIRNAME / PORTFOLIO_SUMMARY_FILENAME
            ),
            trade_predictions_dir  = inference_dir / TRADE_PREDICTIONS_DIRNAME,
        )

    # ── Discovery ───────────────────────────────────────────────────

    def list_run_ids(self) -> List[str]:
        """Return all run IDs on disk, most recent first.

        Sorts by directory name descending.  The pipeline's run_id
        format is ``<ensemble_version>__<UTC_yyyymmdd_HHMMSS>`` so
        descending alphabetical order is also descending temporal
        order within the same ensemble version.

        Returns
        -------
        list of str
            Empty list when ``inference_runs/`` doesn't exist yet
            (fresh deployment with no runs).
        """
        root = self.base_dir / INFERENCE_RUNS_DIR
        if not root.is_dir():
            return []
        return sorted(
            (p.name for p in root.iterdir() if p.is_dir()),
            reverse=True,
        )

    # ── Manifest ────────────────────────────────────────────────────

    def load_manifest(self, run_id: str) -> Dict[str, Any]:
        """Read ``manifest.json`` for ``run_id``.

        Raises
        ------
        FileNotFoundError
            If the manifest doesn't exist — typically because the
            run is still in flight (the worker thread hasn't reached
            ``post_infer`` yet) or never completed successfully.
        """
        path = self.paths_for(run_id).manifest_path
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest not found for run '{run_id}': {path}"
            )
        return _load_json(path)

    # ── Run summary ─────────────────────────────────────────────────

    def run_summary(self, run_id: str) -> Dict[str, Any]:
        """Lightweight summary suitable for the run-history table.

        Reads only the manifest (small JSON), not the parquets.
        When the manifest doesn't exist we still return a
        well-formed summary with ``status='in_progress'`` so the
        UI can list in-flight runs.

        Returns
        -------
        dict
            Keys match :class:`RunSummary` (pydantic model).
        """
        paths = self.paths_for(run_id)
        try:
            manifest = self.load_manifest(run_id)
            status   = "complete"
        except FileNotFoundError:
            manifest = {}
            status   = "in_progress"

        return {
            "run_id":           run_id,
            "ensemble_version": manifest.get("ensemble_version"),
            "generated_at":     manifest.get("generated_at"),
            "n_scenarios":      int(manifest.get("n_scenarios") or 0),
            "n_clusters":       len(manifest.get("clusters") or []),
            "status":           status,
            "manifest_path":    str(paths.manifest_path),
            "latency_seconds":  manifest.get("latency_seconds"),
        }

    # ── Long-format parquets (run-level summaries) ──────────────────

    def load_portfolio(self, run_id: str) -> pd.DataFrame:
        """Read ``portfolio_predictions.parquet`` for ``run_id``.

        Long-format: one row per scenario, columns are
        ``scenario_label``, ``sum_pnl_scaled``, ``sum_pnl_original``,
        ``n_clusters``.
        """
        path = self.paths_for(run_id).portfolio_summary_path
        if not path.exists():
            raise FileNotFoundError(
                f"Portfolio summary not found for run '{run_id}': {path}"
            )
        return _load_parquet(path)

    def load_clusters_summary(self, run_id: str) -> pd.DataFrame:
        """Read ``cluster_predictions.parquet`` for ``run_id``.

        Long-format: one row per cluster × scenario.  Schema is
        whatever :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
        emits in its ``summary_df`` — currently ``scenario_label``,
        ``cluster_id``, ``sum_pnl_scaled``, ``sum_pnl_original``,
        ``mean_pnl_original``, ``std_pnl_original``,
        ``min_pnl_original``, ``max_pnl_original``.
        """
        path = self.paths_for(run_id).cluster_summary_path
        if not path.exists():
            raise FileNotFoundError(
                f"Cluster summary not found for run '{run_id}': {path}"
            )
        return _load_parquet(path)

    # ── Wide-format parquets (per-cluster trade-level) ──────────────

    def load_cluster_trades(
        self,
        run_id:     str,
        cluster_id: str,
        space:      str = "original",
    ) -> pd.DataFrame:
        """Read one per-cluster wide parquet.

        Parameters
        ----------
        space
            Either ``'scaled'`` (model output space) or
            ``'original'`` (inverse-transformed, notional-restored).
            See :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
            for the difference.  Default is ``'original'`` since
            that's what the user-facing dashboard reads by default.

        Returns
        -------
        pd.DataFrame
            Wide format — index = ``scenario_label``, columns =
            trade IDs, values = predicted PnL in the requested
            space.
        """
        path = self.paths_for(run_id).cluster_parquet(cluster_id, space)
        if not path.exists():
            raise FileNotFoundError(
                f"Cluster trades not found: run='{run_id}', "
                f"cluster='{cluster_id}', space='{space}': {path}"
            )
        return _load_parquet(path)

    # ── Convenience: snapshots embedded in the manifest ─────────────

    def load_validation_snapshot(
        self, run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Pull the ``ValidationReport`` snapshot out of the manifest.

        Cheaper than the active run's ``GET /validate`` because no
        on-disk parquet is involved — the snapshot is captured in
        ``manifest.json`` at write time.

        Returns ``None`` if the manifest doesn't carry one (older
        runs written before the snapshot was added).
        """
        return self.load_manifest(run_id).get("validation")

    def load_scenarios_snapshot(
        self, run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Pull the ``LoadedScenariosReport`` snapshot out of the manifest."""
        return self.load_manifest(run_id).get("scenarios")


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton (mirrors InferenceStateManager pattern)
# ──────────────────────────────────────────────────────────────────────

_reader: Optional[InferenceResultReader] = None


def set_result_reader(reader: InferenceResultReader) -> None:
    """Inject the reader at app-lifespan startup."""
    global _reader
    _reader = reader


def get_result_reader() -> InferenceResultReader:
    """FastAPI ``Depends`` — returns the singleton reader.

    Raises
    ------
    RuntimeError
        If called before the lifespan startup has run — indicates
        a misconfigured server.
    """
    if _reader is None:
        raise RuntimeError(
            "InferenceResultReader not initialised. Server not ready."
        )
    return _reader
```

---

### A · 4 — `src/rade_ml_pt/ensemble/api/routers/inference.py` (Stages 7-10)

The full FastAPI router. Control plane (Stages 7-9) at the top: `/load`,
`/scenarios`, `/validate`, `/run` (threaded), `/status`, `/events`,
`/manifest`. Data plane (Stage 10) at the bottom: seven `/runs/...`
endpoints reading off the result reader.

```python
"""``/prism/v1/inference`` — staged inference workflow over HTTP.

Exposes the four stages of :class:`EnsembleInferencePipeline` as REST
endpoints, plus two polling endpoints for the UI activity log and
status probe, plus a manifest fetcher for completed runs, plus the
historical run reader (Stage 10) for cross-session dashboarding.

Endpoint map
------------
Control plane (active run)::

    POST /prism/v1/inference/load          → load the ensemble (cold-load).
    POST /prism/v1/inference/scenarios     → load + parse a scenario folder.
    POST /prism/v1/inference/validate      → run validation on loaded scenarios.
    POST /prism/v1/inference/run           → dispatch inference onto a worker thread.
    GET  /prism/v1/inference/status        → cheap status probe (gate the next button).
    GET  /prism/v1/inference/events        → cursor-based activity log poll.
    GET  /prism/v1/inference/manifest      → read the active run's manifest.json.

Data plane (historical runs, Stage 10)::

    GET  /prism/v1/inference/runs                                            → list every run on disk.
    GET  /prism/v1/inference/runs/{run_id}/manifest                          → read any run's manifest.
    GET  /prism/v1/inference/runs/{run_id}/portfolio                         → portfolio-level summary parquet.
    GET  /prism/v1/inference/runs/{run_id}/clusters                          → cluster-level summary parquet.
    GET  /prism/v1/inference/runs/{run_id}/clusters/{cid}/trades?space=...   → per-cluster trade-level wide parquet.
    GET  /prism/v1/inference/runs/{run_id}/validation                        → ValidationReport snapshot from manifest.
    GET  /prism/v1/inference/runs/{run_id}/scenarios                         → LoadedScenariosReport snapshot from manifest.

State model
-----------
Today (Option A) a single :class:`InferenceRunState` is held on the
process-wide :class:`InferenceStateManager`.  Each ``POST /load`` call
replaces any prior run.  See
:mod:`services.inference_state` for the migration path to multi-user.

The data-plane endpoints are stateless — they go straight through
:class:`InferenceResultReader` to disk, so they're safe to call from
any number of concurrent UI sessions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.config import Settings, get_settings
from src.rade_ml_pt.ensemble.api.dependencies import (
    get_inference_state_manager,
    get_result_reader,
)
from src.rade_ml_pt.ensemble.api.models.inference import (
    ClusterRoutingDecisionModel,
    ClusterSummaryResponse,
    ClusterSummaryRow,
    ClusterTradesResponse,
    EventModel,
    EventsResponse,
    LoadResponse,
    LoadScenariosRequest,
    LoadScenariosResponse,
    ManifestResponse,
    PortfolioResponse,
    PortfolioRow,
    RunRequest,
    RunResponse,
    RunSummary,
    RunsListResponse,
    ScenariosSnapshotResponse,
    StatusResponse,
    ValidateResponse,
    ValidationSnapshotResponse,
)
from src.rade_ml_pt.ensemble.api.services.inference_state import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_LOADED,
    STATUS_LOADING,
    STATUS_RUNNING,
    STATUS_SCENARIOS,
    STATUS_VALIDATED,
    InferenceStateManager,
    build_ensemble_config,
    per_run_artifacts_dir,
)
from src.rade_ml_pt.ensemble.api.services.result_reader import (
    VALID_SPACES,
    InferenceResultReader,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prism/v1/inference", tags=["inference"])


# ──────────────────────────────────────────────────────────────────────
# POST /load
# ──────────────────────────────────────────────────────────────────────

@router.post("/load", response_model=LoadResponse)
def load_ensemble(
    settings: Settings                  = Depends(get_settings),
    manager:  InferenceStateManager     = Depends(get_inference_state_manager),
) -> LoadResponse:
    """Cold-load the ensemble model + per-cluster inference contexts.

    Wraps :meth:`EnsembleInferencePipeline.load`.  Constructs a fresh
    run state on the manager and returns the assigned ``run_id`` plus
    a count of clusters that the UI can render immediately.

    The active run (if any) is silently replaced — appropriate for
    the single-user dashboard today; Option B will key on per-request
    ``run_id`` and reject the implicit replacement.
    """
    # Build a minimal EnsembleConfig from settings.  artifacts_dir is
    # filled with a placeholder; the real per-run path is set when
    # /run is called (post_infer needs it before writing).
    ensemble_config = build_ensemble_config(
        registry_dir  = settings.registry_dir,
        artifacts_dir = settings.artifacts_dir,
    )

    state = manager.create_run(
        ensemble_config  = ensemble_config,
        ensemble_version = settings.resolved_version,
    )

    state.transition(STATUS_LOADING)
    try:
        state.pipeline.load()
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Ensemble load failed for run %s", state.run_id)
        raise HTTPException(status_code=500, detail=f"Ensemble load failed: {exc}")

    n_clusters = (
        len(state.pipeline._ensemble.members)  # noqa: SLF001 — internal-but-stable
        if state.pipeline._ensemble is not None else 0
    )
    state.transition(STATUS_LOADED)

    return LoadResponse(
        run_id           = state.run_id,
        ensemble_version = state.ensemble_version,
        n_clusters       = n_clusters,
        status           = state.status,
    )


# ──────────────────────────────────────────────────────────────────────
# POST /scenarios
# ──────────────────────────────────────────────────────────────────────

@router.post("/scenarios", response_model=LoadScenariosResponse)
def load_scenarios(
    body:     LoadScenariosRequest,
    manager:  InferenceStateManager = Depends(get_inference_state_manager),
) -> LoadScenariosResponse:
    """Parse a folder of shock CSVs into the pipeline state.

    Wraps :meth:`EnsembleInferencePipeline.load_scenarios`.  The path
    must be readable by the API host process.  File-upload (multipart)
    is deferred to a later stage.

    Side-effect: invalidates any prior validate result (the pipeline
    resets ``_validation_report`` to ``None`` internally).
    """
    state = manager.get_run()
    if state.status not in (STATUS_LOADED, STATUS_SCENARIOS, STATUS_VALIDATED, STATUS_COMPLETE, STATUS_FAILED):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot load scenarios from status '{state.status}'. Call /load first.",
        )

    try:
        report = state.pipeline.load_scenarios(new_scenario_dir=body.new_scenario_dir)
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Scenario load failed for run %s", state.run_id)
        raise HTTPException(status_code=400, detail=f"Scenario load failed: {exc}")

    state.transition(STATUS_SCENARIOS)
    return LoadScenariosResponse(**report.to_dict())


# ──────────────────────────────────────────────────────────────────────
# POST /validate
# ──────────────────────────────────────────────────────────────────────

@router.post("/validate", response_model=ValidateResponse)
def validate_scenarios(
    manager: InferenceStateManager = Depends(get_inference_state_manager),
) -> ValidateResponse:
    """Compute the per-cluster routing decisions + run-level errors.

    Wraps :meth:`EnsembleInferencePipeline.validate_scenarios`.  Does
    NOT raise for user-input issues (those land in
    ``report.errors``); only system failures bubble up as 500s.

    A non-empty ``errors`` list keeps the run in the
    ``scenarios_loaded`` state so the user can fix and re-validate
    without re-loading scenarios.
    """
    state = manager.get_run()
    if state.status not in (STATUS_SCENARIOS, STATUS_VALIDATED):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot validate from status '{state.status}'. "
                "Call /scenarios first."
            ),
        )

    try:
        report = state.pipeline.validate_scenarios()
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Validation failed for run %s", state.run_id)
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")

    # Only advance to "validated" when the report is clean — keeps the
    # state machine truthful (an invalid report shouldn't unlock /run).
    if report.is_valid:
        state.transition(STATUS_VALIDATED)

    decisions = [
        ClusterRoutingDecisionModel(**d.to_dict()) for d in report.cluster_decisions
    ]
    return ValidateResponse(
        ensemble_version       = report.ensemble_version,
        n_scenarios            = report.n_scenarios,
        scenario_labels        = list(report.scenario_labels),
        cluster_decisions      = decisions,
        errors                 = list(report.errors),
        warnings               = list(report.warnings),
        is_valid               = report.is_valid,
        affected_cluster_ids   = report.affected_cluster_ids,
        unaffected_cluster_ids = report.unaffected_cluster_ids,
        affected_count         = report.affected_count,
        unaffected_count       = report.unaffected_count,
        cheap_path_used        = report.cheap_path_used,
    )


# ──────────────────────────────────────────────────────────────────────
# POST /run
# ──────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=RunResponse)
def run_inference(
    body:     RunRequest                = RunRequest(),
    settings: Settings                  = Depends(get_settings),
    manager:  InferenceStateManager     = Depends(get_inference_state_manager),
) -> RunResponse:
    """Dispatch the inference run onto a background thread.

    Wraps :meth:`EnsembleInferencePipeline.run_inference`.  Stage 9:
    the call is **non-blocking** — the pipeline executes on a worker
    thread and the HTTP response returns immediately with
    ``status='running'``.  The UI then polls
    :func:`get_status` (terminal-state probe) and :func:`get_events`
    (live activity-log tail) until ``status`` becomes ``'complete'``
    or ``'failed'``.

    Gated on:
      * ``status == 'validated'`` — validation must have produced an
        error-free report.
      * ``not state.is_alive`` — no prior run thread still executing
        (defends against double-dispatch from a refreshed browser).

    Side effects
    ------------
    Pins ``config.artifacts_dir`` on the pipeline to a per-run path
    so ``post_infer`` writes its manifest + parquets into a stable
    predictable place the API can serve.  The resolved path is
    returned immediately for the UI to deep-link to even while the
    run is still in flight.

    Terminal metrics (``n_scenarios``, ``n_clusters``,
    ``n_predictions``, ``manifest_path``) are unknown at dispatch —
    they are populated on the state object by the worker thread once
    the run completes, and surfaced through ``GET /status`` and
    ``GET /manifest``.
    """
    state = manager.get_run()
    if state.status != STATUS_VALIDATED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot run from status '{state.status}'. "
                "Validation must complete cleanly (status='validated')."
            ),
        )
    if state.is_alive:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run {state.run_id} is already executing in the "
                "background.  Wait for completion before re-dispatching."
            ),
        )

    # Resolve and pin the per-run artifacts directory BEFORE dispatch
    # so the worker closure captures a stable path — the request
    # scope ends before the worker thread does.
    artifacts_dir = body.artifacts_dir or per_run_artifacts_dir(
        settings.artifacts_dir, state.run_id,
    )
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    state.pipeline.config.artifacts_dir = artifacts_dir
    state.artifacts_dir                  = artifacts_dir
    state.transition(STATUS_RUNNING)

    # Worker closure — captured by the background thread.  Any
    # exception lands on ``state.last_error`` via the FAILED
    # transition; the HTTP response has already been returned so the
    # client must poll /status to discover the outcome.
    def _execute() -> None:
        try:
            state.pipeline.run_inference()
        except Exception as exc:
            state.transition(STATUS_FAILED, error=str(exc))
            logger.exception(
                "Inference run failed for run %s", state.run_id,
            )
            return

        # post_infer writes <artifacts_dir>/inference/manifest.json —
        # pin the path on the state so /manifest can read it back
        # without re-deriving the layout.
        manifest_path = Path(artifacts_dir) / "inference" / "manifest.json"
        state.manifest_path = (
            str(manifest_path) if manifest_path.exists() else None
        )
        state.transition(STATUS_COMPLETE)

    manager.start_run_in_background(_execute)

    return RunResponse(
        run_id        = state.run_id,
        status        = state.status,        # "running"
        artifacts_dir = artifacts_dir,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /status
# ──────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
def get_status(
    manager: InferenceStateManager = Depends(get_inference_state_manager),
) -> StatusResponse:
    """Cheap status probe — drives the next-button gate in the UI.

    Designed to be polled at 1-2 Hz between stages without
    appreciable load on the server.  Returns a no-op-shaped response
    when no run has been created yet.
    """
    if not manager.has_active_run:
        return StatusResponse(has_active_run=False)

    state = manager.get_run()
    return StatusResponse(
        has_active_run   = True,
        run_id           = state.run_id,
        ensemble_version = state.ensemble_version,
        status           = state.status,
        last_error       = state.last_error,
        n_events         = len(state.activity_log),
        created_at       = state.created_at,
        artifacts_dir    = state.artifacts_dir,
        manifest_path    = state.manifest_path,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /events
# ──────────────────────────────────────────────────────────────────────

@router.get("/events", response_model=EventsResponse)
def get_events(
    cursor:  int                       = Query(0, ge=0, description="Number of events the caller has already seen."),
    manager: InferenceStateManager     = Depends(get_inference_state_manager),
) -> EventsResponse:
    """Cursor-paginated tail of the activity log.

    Used by the UI to render the live activity feed.  Returns only
    events strictly after ``cursor``; the response's ``next_cursor``
    is the total event count so far — pass it back on the next poll
    to chain efficiently.

    Falls back to an empty response when no run has been created so
    the UI can mount the polling loop before /load is called.
    """
    if not manager.has_active_run:
        return EventsResponse(events=[], next_cursor=0)

    events, next_cursor = manager.events_since(run_id=None, cursor=cursor)
    return EventsResponse(
        events      = [EventModel(**e) for e in events],
        next_cursor = next_cursor,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /manifest
# ──────────────────────────────────────────────────────────────────────

@router.get("/manifest", response_model=ManifestResponse)
def get_manifest(
    manager: InferenceStateManager = Depends(get_inference_state_manager),
) -> ManifestResponse:
    """Return the per-run ``manifest.json`` written by ``post_infer``.

    Only available after the run completes successfully.  The
    manifest is the dashboard's entry point — it lists the artifact
    paths for every per-cluster parquet plus the run-level
    portfolio / cluster summaries.
    """
    state = manager.get_run()
    if state.status != STATUS_COMPLETE or state.manifest_path is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Manifest not available — run status is "
                f"'{state.status}'.  Wait for status='complete'."
            ),
        )

    path = Path(state.manifest_path)
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Manifest path recorded but file is missing: {path}",
        )

    with open(path, "r") as f:
        manifest = json.load(f)

    return ManifestResponse(run_id=state.run_id, manifest=manifest)


# ══════════════════════════════════════════════════════════════════════
# Stage 10 — historical run reader
#
# All endpoints below are stateless: they go straight through
# :class:`InferenceResultReader` to disk and don't touch
# :class:`InferenceStateManager` at all.  Safe to call from any
# number of concurrent UI sessions.
#
# Error model
# -----------
# * ``FileNotFoundError`` from the reader → 404.  Covers the
#   common case of polling for results before the worker thread
#   finished writing them (manifest exists but parquets don't yet,
#   or run_id is invalid).
# * Any other reader exception → 500 (signals data corruption).
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# GET /runs — discovery
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs", response_model=RunsListResponse)
def list_runs(
    reader: InferenceResultReader = Depends(get_result_reader),
) -> RunsListResponse:
    """Return every inference run on disk, most recent first.

    Reads only ``manifest.json`` per run (small JSON), not parquets,
    so the response is fast even with hundreds of runs.  Returns an
    empty list when the ``inference_runs/`` directory doesn't exist
    yet (fresh deployment).
    """
    run_ids = reader.list_run_ids()
    runs    = [RunSummary(**reader.run_summary(rid)) for rid in run_ids]
    return RunsListResponse(runs=runs, count=len(runs))


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/manifest
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/manifest", response_model=ManifestResponse)
def get_run_manifest(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ManifestResponse:
    """Read ``manifest.json`` for any historical run.

    Counterpart to ``GET /manifest`` (active run); use this one
    when navigating the run-history table.
    """
    try:
        manifest = reader.load_manifest(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ManifestResponse(run_id=run_id, manifest=manifest)


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/portfolio
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/portfolio", response_model=PortfolioResponse)
def get_run_portfolio(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> PortfolioResponse:
    """Return the per-scenario portfolio summary for ``run_id``.

    Sourced from ``portfolio_predictions.parquet``.  Long-format:
    one row per scenario with ``sum_pnl_*`` (additive across
    clusters) and ``n_clusters`` (sanity check for missing
    contributions).
    """
    try:
        df = reader.load_portfolio(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    rows = [
        PortfolioRow(
            scenario_label   = str(r["scenario_label"]),
            sum_pnl_scaled   = float(r["sum_pnl_scaled"]),
            sum_pnl_original = float(r["sum_pnl_original"]),
            n_clusters       = int(r["n_clusters"]),
        )
        for r in df.to_dict(orient="records")
    ]
    return PortfolioResponse(run_id=run_id, n_scenarios=len(rows), rows=rows)


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/clusters
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/clusters", response_model=ClusterSummaryResponse)
def get_run_clusters(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ClusterSummaryResponse:
    """Return the per-cluster × per-scenario summary for ``run_id``.

    Sourced from ``cluster_predictions.parquet``.  Long-format —
    one row per cluster × scenario, schema matches
    :meth:`HybridGnnRnnInferencePipeline.transform_predictions`.
    """
    try:
        df = reader.load_clusters_summary(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    rows = [
        ClusterSummaryRow(
            scenario_label    = str(r["scenario_label"]),
            cluster_id        = str(r["cluster_id"]),
            sum_pnl_scaled    = float(r["sum_pnl_scaled"]),
            sum_pnl_original  = float(r["sum_pnl_original"]),
            mean_pnl_original = float(r["mean_pnl_original"]),
            std_pnl_original  = float(r["std_pnl_original"]),
            min_pnl_original  = float(r["min_pnl_original"]),
            max_pnl_original  = float(r["max_pnl_original"]),
        )
        for r in df.to_dict(orient="records")
    ]
    return ClusterSummaryResponse(
        run_id      = run_id,
        n_clusters  = int(df["cluster_id"].nunique()) if len(df) else 0,
        n_scenarios = int(df["scenario_label"].nunique()) if len(df) else 0,
        rows        = rows,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/clusters/{cluster_id}/trades
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/clusters/{cluster_id}/trades",
    response_model=ClusterTradesResponse,
)
def get_run_cluster_trades(
    run_id:     str,
    cluster_id: str,
    space:      str = Query(
        "original",
        description=(
            "Output space — 'original' (inverse-scaled + notional-restored) "
            "or 'scaled' (model output space)."
        ),
        pattern=f"^({'|'.join(VALID_SPACES)})$",
    ),
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ClusterTradesResponse:
    """Return the per-cluster trade-level wide parquet.

    Wide-format on the wire (matrix + axis labels) rather than
    long-format because typical sizes are ~30k cells and a row-per-
    cell response would be 5-10× heavier.  The dashboard's AG Grid
    consumes the matrix directly; histograms / aggregates pivot
    cheaply from this shape.
    """
    try:
        df = reader.load_cluster_trades(run_id, cluster_id, space)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        # space validated by the regex above, but defence-in-depth.
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ClusterTradesResponse(
        run_id          = run_id,
        cluster_id      = cluster_id,
        space           = space,
        n_scenarios     = int(len(df)),
        n_trades        = int(len(df.columns)),
        scenario_labels = [str(i) for i in df.index],
        trade_ids       = [str(c) for c in df.columns],
        values          = df.astype(float).values.tolist(),
    )


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/validation
# GET /runs/{run_id}/scenarios
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/validation", response_model=ValidationSnapshotResponse,
)
def get_run_validation(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ValidationSnapshotResponse:
    """Return the ``ValidationReport`` snapshot embedded in the run's manifest.

    Cheaper than the active run's ``GET /validate`` because no
    on-disk parquet is involved — ``post_infer`` captures the
    report into the manifest at write time.
    """
    try:
        snapshot = reader.load_validation_snapshot(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ValidationSnapshotResponse(run_id=run_id, snapshot=snapshot)


@router.get(
    "/runs/{run_id}/scenarios", response_model=ScenariosSnapshotResponse,
)
def get_run_scenarios(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ScenariosSnapshotResponse:
    """Return the ``LoadedScenariosReport`` snapshot embedded in the manifest."""
    try:
        snapshot = reader.load_scenarios_snapshot(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ScenariosSnapshotResponse(run_id=run_id, snapshot=snapshot)
```

---

### A · 5 — `src/rade_ml_pt/ensemble/api/models/inference.py` (Stages 7-10)

All pydantic request / response schemas for the `/inference` router.
Stages 7-9 models at the top, Stage 10 data-plane models below the
divider.

```python
"""Pydantic schemas for the ``/inference`` router.

Type-safe wire contracts for the three-button workflow:

    POST /inference/load          → LoadResponse
    POST /inference/scenarios     ← LoadScenariosRequest    → LoadScenariosResponse
    POST /inference/validate                                → ValidateResponse
    POST /inference/run           ← RunRequest              → RunResponse
    GET  /inference/status                                  → StatusResponse
    GET  /inference/events?cursor=N                         → EventsResponse
    GET  /inference/manifest                                → ManifestResponse

Most response models mirror the dataclass ``to_dict()`` shapes the
pipeline already exposes (``LoadedScenariosReport.to_dict``,
``ValidationReport.to_dict``).  Wrapping them in Pydantic gives the
typed client and OpenAPI docs full schema coverage without duplicating
the underlying state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Activity-log event (mirror of ``infer_events.ActivityEntry``)
# ──────────────────────────────────────────────────────────────────────

class EventModel(BaseModel):
    """One activity-log event emitted by the pipeline.

    Wire shape identical to :data:`infer_events.ActivityEntry` (a plain
    dict).  Re-declared as a Pydantic model purely for OpenAPI schema
    coverage — the pipeline itself stays Pydantic-free.
    """

    id:     str            = Field(..., description="Hex UUID, set at event-construction time.")
    stage:  str            = Field(..., description="One of 'ingest', 'validate', 'inference'.")
    phase:  str            = Field(..., description="Human-readable phase label (e.g. 'Loading new-scenario shocks').")
    status: str            = Field(..., description="One of 'ok', 'running', 'fail', 'pending'.")
    ts:     str            = Field(..., description="UTC ISO-8601 timestamp.")
    target: Optional[str]  = Field(None, description="Optional secondary label (filename, cluster id, etc.).")
    detail: Optional[str]  = Field(None, description="Optional detail / error text.")


# ──────────────────────────────────────────────────────────────────────
# POST /inference/load
# ──────────────────────────────────────────────────────────────────────

class LoadResponse(BaseModel):
    """Result of the cold-load stage.

    Returned after :meth:`EnsembleInferencePipeline.load` completes.
    ``n_clusters`` lets the UI render the routing table skeleton before
    scenarios are loaded.
    """

    run_id:           str  = Field(..., description="Identifier for the newly-created run.")
    ensemble_version: str  = Field(..., description="Resolved ensemble version.")
    n_clusters:       int  = Field(..., description="Number of clusters in the loaded ensemble.")
    status:           str  = Field(..., description="Run lifecycle state (one of 'loaded' / 'failed').")


# ──────────────────────────────────────────────────────────────────────
# POST /inference/scenarios
# ──────────────────────────────────────────────────────────────────────

class LoadScenariosRequest(BaseModel):
    """Request body for ``POST /inference/scenarios``.

    For v1 the API accepts a server-side path; file-upload (multipart)
    is deferred to a later stage to keep the contract simple.  The
    path must be readable by the API process.
    """

    new_scenario_dir: str = Field(
        ...,
        description=(
            "Path on the API host to a folder containing one CSV "
            "per shocked risk factor.  Filename minus '.csv' is the "
            "risk-factor key."
        ),
    )


class LoadScenariosResponse(BaseModel):
    """Mirror of :meth:`LoadedScenariosReport.to_dict`."""

    new_scenario_dir:  str
    risk_factor_names: List[str]
    n_risk_factors:    int
    n_scenarios:       int
    scenario_labels:   List[str]


# ──────────────────────────────────────────────────────────────────────
# POST /inference/validate
# ──────────────────────────────────────────────────────────────────────

class ClusterRoutingDecisionModel(BaseModel):
    """Per-cluster decision in :class:`ValidationReport`."""

    cluster_id:                str
    is_affected:               bool
    intersecting_risk_factors: List[str]
    n_elementary_trades:       int
    n_target_trades:           int
    missing_scenario_labels:   List[str]


class ValidateResponse(BaseModel):
    """Mirror of :meth:`ValidationReport.to_dict`, including derived fields.

    Derived properties (``is_valid``, counts, etc.) are returned so
    the UI can render the validate-stage report without re-deriving
    them client-side.
    """

    ensemble_version:       str
    n_scenarios:            int
    scenario_labels:        List[str]
    cluster_decisions:      List[ClusterRoutingDecisionModel]
    errors:                 List[str]
    warnings:               List[str]

    is_valid:               bool
    affected_cluster_ids:   List[str]
    unaffected_cluster_ids: List[str]
    affected_count:         int
    unaffected_count:       int
    cheap_path_used:        bool


# ──────────────────────────────────────────────────────────────────────
# POST /inference/run
# ──────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Optional knobs for the run stage.

    Both fields are optional — sensible defaults are taken from the
    API settings + the run state when omitted.
    """

    artifacts_dir: Optional[str] = Field(
        None,
        description=(
            "Override for the per-run artifacts directory.  Defaults "
            "to <settings.artifacts_dir>/inference_runs/<run_id>."
        ),
    )
    batch_size:    Optional[int] = Field(
        None,
        description=(
            "Per-cluster chunk size for the chunked forward pass "
            "(see HybridGnnRnnInferencePipeline.predict_member_chunked). "
            "Defaults to the pipeline's built-in default."
        ),
    )


class RunResponse(BaseModel):
    """Acknowledgement of a background ``POST /inference/run``.

    Returned immediately after the pipeline is dispatched onto a
    worker thread (Stage 9).  ``status`` is therefore ``'running'``
    on success; the run's terminal outcome (``'complete'`` /
    ``'failed'``) and its metrics are surfaced through
    ``GET /status`` and ``GET /manifest`` once the worker finishes.

    The metric fields below remain on the schema so the client type
    stays stable even when Stage 9.5 adds a synchronous "wait until
    done" variant or eager completion in fast-path tests.
    """

    run_id:        str           = Field(..., description="Run identifier.")
    status:        str           = Field(..., description="Run lifecycle state at dispatch time — normally 'running'.")
    artifacts_dir: str           = Field(..., description="Resolved per-run artifacts root (populated immediately).")
    n_scenarios:   Optional[int] = Field(None, description="Number of scenarios scored (None until the worker completes).")
    n_clusters:    Optional[int] = Field(None, description="Number of clusters scored (None until the worker completes).")
    n_predictions: Optional[int] = Field(None, description="Total prediction cells (None until the worker completes).")
    manifest_path: Optional[str] = Field(None, description="Absolute path to manifest.json (None until run completes).")
    error:         Optional[str] = Field(None, description="Detail message; set only when status='failed'.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/status
# ──────────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    """Lightweight current-state probe.

    Used by the UI between stages to gate the next button.  Designed
    to be cheap to poll (no I/O, no heavy serialisation).
    """

    has_active_run:   bool          = Field(..., description="True once /load has been called.")
    run_id:           Optional[str] = None
    ensemble_version: Optional[str] = None
    status:           Optional[str] = Field(None, description="Run lifecycle state (see services.inference_state.STATUS_*).")
    last_error:       Optional[str] = None
    n_events:         int           = Field(0, description="Total activity-log events emitted so far for this run.")
    created_at:       Optional[str] = None
    artifacts_dir:    Optional[str] = None
    manifest_path:    Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# GET /inference/events
# ──────────────────────────────────────────────────────────────────────

class EventsResponse(BaseModel):
    """Slice of the activity log for cursor-based polling.

    Pass ``next_cursor`` from the previous response as ``?cursor=`` on
    the next request to get only new events.
    """

    events:      List[EventModel] = Field(..., description="Events emitted since the cursor.")
    next_cursor: int              = Field(..., description="Total events emitted so far — pass on the next poll.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/manifest
# ──────────────────────────────────────────────────────────────────────

class ManifestResponse(BaseModel):
    """The run-level ``manifest.json`` written by ``post_infer``.

    Untyped on the values so the manifest schema can evolve without a
    contract churn — the dashboard reads keys it knows about and is
    permissive on the rest.  See
    :meth:`EnsembleInferencePipeline._write_run_manifest` for the
    canonical fields.
    """

    run_id:    str           = Field(..., description="Run identifier.")
    manifest:  Dict[str, Any] = Field(..., description="Raw manifest.json contents.")


# ══════════════════════════════════════════════════════════════════════
# Stage 10 — data-plane response models
#
# All models below describe responses served by the result-reading
# endpoints (`GET /inference/runs/...`).  They mirror the on-disk
# layout written by Stage 6 (`EnsembleInferencePipeline.post_infer`
# + `_post_infer_cluster`):
#
#   * Run-level summaries are LONG-format parquets → list-of-rows
#     responses (`PortfolioResponse`, `ClusterSummaryResponse`).
#   * Per-cluster trade-level outputs are WIDE-format parquets →
#     compact matrix response (`ClusterTradesResponse`) that the
#     dashboard's AG Grid can render directly.
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs
# ──────────────────────────────────────────────────────────────────────

class RunSummary(BaseModel):
    """Lightweight summary of one inference run on disk.

    Built by :meth:`InferenceResultReader.run_summary` from
    ``manifest.json``.  When the manifest doesn't exist yet (run is
    still in flight) ``status`` is ``'in_progress'`` and most
    fields are ``None`` / zero.
    """

    run_id:           str             = Field(..., description="Run identifier — ``<ensemble_version>__<UTC_yyyymmdd_HHMMSS>``.")
    ensemble_version: Optional[str]   = Field(None, description="Ensemble version this run scored against.")
    generated_at:     Optional[str]   = Field(None, description="UTC ISO timestamp when post_infer wrote the manifest.")
    n_scenarios:      int             = Field(0,    description="Scenarios scored in the run.")
    n_clusters:       int             = Field(0,    description="Clusters that produced artifacts.")
    status:           str             = Field("complete", description="``complete`` once manifest exists; ``in_progress`` otherwise.")
    manifest_path:    str             = Field(..., description="Absolute path to ``manifest.json`` on the server.")
    latency_seconds:  Optional[float] = Field(None, description="End-to-end run latency from the pipeline.")


class RunsListResponse(BaseModel):
    """Body of ``GET /inference/runs`` — every run discoverable on disk."""

    runs:  List[RunSummary] = Field(..., description="One entry per run, most recent first.")
    count: int              = Field(..., description="Total runs returned (== len(runs)).")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs/{run_id}/portfolio
# ──────────────────────────────────────────────────────────────────────

class PortfolioRow(BaseModel):
    """One row of ``portfolio_predictions.parquet`` — one scenario.

    ``sum_pnl_*`` is the SUM of per-cluster ``sum_pnl_*`` for the
    same scenario, hence portfolio-level PnL in the requested space.
    ``n_clusters`` reports the number of clusters that contributed
    (so the dashboard can flag scenarios where some cluster was
    missing).
    """

    scenario_label:   str   = Field(..., description="Scenario identifier (row index of the parquet).")
    sum_pnl_scaled:   float = Field(..., description="Portfolio PnL in the model's scaled output space.")
    sum_pnl_original: float = Field(..., description="Portfolio PnL in the original (notional-restored) space.")
    n_clusters:       int   = Field(..., description="Distinct clusters that contributed to this scenario.")


class PortfolioResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/portfolio``."""

    run_id:      str                = Field(..., description="Run identifier.")
    n_scenarios: int                = Field(..., description="Total scenarios (== len(rows)).")
    rows:        List[PortfolioRow] = Field(..., description="One row per scenario.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs/{run_id}/clusters
# ──────────────────────────────────────────────────────────────────────

class ClusterSummaryRow(BaseModel):
    """One row of ``cluster_predictions.parquet`` — one cluster × one scenario.

    Mirrors the ``summary_df`` returned by
    :meth:`HybridGnnRnnInferencePipeline.transform_predictions`.
    All ``*_pnl_original`` stats are per-trade aggregates *within
    the cluster*; ``*_pnl_scaled`` exists only as a sanity check
    against the model's output space.
    """

    scenario_label:    str   = Field(..., description="Scenario identifier.")
    cluster_id:        str   = Field(..., description="Cluster identifier.")
    sum_pnl_scaled:    float = Field(..., description="Cluster PnL in scaled space.")
    sum_pnl_original:  float = Field(..., description="Cluster PnL in original space.")
    mean_pnl_original: float = Field(..., description="Mean per-trade PnL across the cluster, original space.")
    std_pnl_original:  float = Field(..., description="Std-dev per-trade PnL across the cluster, original space.")
    min_pnl_original:  float = Field(..., description="Min per-trade PnL within the cluster, original space.")
    max_pnl_original:  float = Field(..., description="Max per-trade PnL within the cluster, original space.")


class ClusterSummaryResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/clusters``."""

    run_id:      str                     = Field(..., description="Run identifier.")
    n_clusters:  int                     = Field(..., description="Distinct cluster IDs in the response.")
    n_scenarios: int                     = Field(..., description="Distinct scenarios in the response.")
    rows:        List[ClusterSummaryRow] = Field(..., description="One row per cluster × scenario.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs/{run_id}/clusters/{cluster_id}/trades
# ──────────────────────────────────────────────────────────────────────

class ClusterTradesResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/clusters/{cid}/trades``.

    Returns the wide-format parquet as a (small) two-axis labelled
    matrix instead of a list of rows because:

    * Typical sizes are ``[n_scenarios=300, n_trades=100]`` — 30k
      floats; a long-format response would be 30k JSON objects, 5-10×
      heavier on the wire.
    * The dashboard's AG Grid renders an in-memory matrix directly.
    * The pivoting cost (long → wide for any other view, e.g.
      transposing to a histogram) is identical from this shape.
    """

    run_id:          str               = Field(..., description="Run identifier.")
    cluster_id:      str               = Field(..., description="Cluster identifier.")
    space:           str               = Field(..., description="Output space served — ``'scaled'`` or ``'original'``.")
    n_scenarios:     int               = Field(..., description="Scenarios scored (rows of ``values``).")
    n_trades:        int               = Field(..., description="Trades in the cluster (cols of ``values``).")
    scenario_labels: List[str]         = Field(..., description="Row labels of ``values`` — scenario IDs.")
    trade_ids:       List[str]         = Field(..., description="Column labels of ``values`` — canonical trade IDs.")
    values:          List[List[float]] = Field(..., description="2-D matrix ``[n_scenarios][n_trades]``.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs/{run_id}/validation
# GET /inference/runs/{run_id}/scenarios
# ──────────────────────────────────────────────────────────────────────

class ValidationSnapshotResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/validation``.

    Returns the ``ValidationReport`` that ``post_infer`` snapshotted
    into the manifest at write time.  ``snapshot`` is ``None`` for
    runs whose manifest didn't carry one (e.g. very old manifests).
    """

    run_id:   str                       = Field(..., description="Run identifier.")
    snapshot: Optional[Dict[str, Any]]  = Field(None, description="The ``ValidationReport.to_dict()`` payload at run time.")


class ScenariosSnapshotResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/scenarios``.

    Returns the ``LoadedScenariosReport`` that ``post_infer``
    snapshotted into the manifest.
    """

    run_id:   str                       = Field(..., description="Run identifier.")
    snapshot: Optional[Dict[str, Any]]  = Field(None, description="The ``LoadedScenariosReport.to_dict()`` payload at run time.")
```

---

### A · 6 — `src/rade_ml_pt/ensemble/api/dependencies.py`

FastAPI dependency-injection module. Re-exports the three singleton
hooks (`ArtifactReader`, `InferenceStateManager`, `InferenceResultReader`)
so routers can import everything from one canonical surface.

```python
"""FastAPI dependency injection.

Process-wide singletons constructed during the app's lifespan startup
and exposed here as ``Depends`` providers.  Three collaborators live here:

* :class:`ArtifactReader` — serves on-disk **evaluation** artifacts
  (parquet + JSON) to the existing PRISM routers.
* :class:`InferenceStateManager` — holds the live
  :class:`EnsembleInferencePipeline` for the staged inference workflow
  exposed by the ``/inference`` router (Stage 8).
* :class:`InferenceResultReader` — serves on-disk **inference run**
  artifacts (manifest + parquets) to the historical-run endpoints
  added in Stage 10.

The injection points are re-exported from their respective service
modules; this file is the canonical FastAPI surface so routers can
``from .dependencies import …`` consistently.
"""
from __future__ import annotations

from src.rade_ml_pt.ensemble.api.services.inference_state import (
    InferenceStateManager,
    get_inference_state_manager,
    set_inference_state_manager,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader
from src.rade_ml_pt.ensemble.api.services.result_reader import (
    InferenceResultReader,
    get_result_reader,
    set_result_reader,
)

_reader: ArtifactReader | None = None


def set_reader(reader: ArtifactReader) -> None:
    """Called once during the lifespan startup event."""
    global _reader
    _reader = reader


def get_reader() -> ArtifactReader:
    """FastAPI ``Depends`` — returns the singleton reader."""
    if _reader is None:
        raise RuntimeError("ArtifactReader not initialised. Server not ready.")
    return _reader


# Re-export the inference-state manager + result-reader hooks so
# routers can import them from this single dependencies module
# (consistent with get_reader).
__all__ = [
    "ArtifactReader",
    "InferenceResultReader",
    "InferenceStateManager",
    "get_inference_state_manager",
    "get_reader",
    "get_result_reader",
    "set_inference_state_manager",
    "set_reader",
    "set_result_reader",
]
```

---

### A · 7 — `src/rade_ml_pt/ensemble/api/app.py`

App factory + lifespan startup. Initialises the three singletons
(eval-side `ArtifactReader`, `InferenceStateManager`,
`InferenceResultReader`) and registers every router including
`inference_router`.

```python
"""PRISM API — FastAPI application factory.

Serves ensemble evaluation artifacts (parquet + JSON) for low-latency
dashboard consumption.

CLI launch
----------
.. code-block:: bash

    export PRISM_ARTIFACTS_DIR=/path/to/artifacts
    export PRISM_REGISTRY_DIR=/path/to/registry
    export PRISM_ENSEMBLE_VERSION=latest          # optional, default "latest"

    uvicorn src.rade_ml_pt.ensemble.api.app:get_app --factory --port 8000

Programmatic launch
-------------------
.. code-block:: python

    from src.rade_ml_pt.ensemble.api.app import create_app
    from src.rade_ml_pt.ensemble.api.config import Settings, set_settings

    set_settings(Settings(
        artifacts_dir="/path/to/artifacts",
        registry_dir="/path/to/registry",
        ensemble_version="latest",
    ))
    app = create_app()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

OpenAPI docs are available at ``http://localhost:8000/docs``.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.rade_ml_pt.ensemble.api.config import (
    Settings,
    get_settings,
    set_settings,
)
from src.rade_ml_pt.ensemble.api.dependencies import (
    set_inference_state_manager,
    set_reader,
    set_result_reader,
)
from src.rade_ml_pt.ensemble.api.models.meta import HealthResponse, VersionsResponse
from src.rade_ml_pt.ensemble.api.routers.cluster_timeseries import (
    router as cluster_timeseries_router,
)
from src.rade_ml_pt.ensemble.api.routers.clusters import router as clusters_router
from src.rade_ml_pt.ensemble.api.routers.elementary_pnl import (
    router as elementary_pnl_router,
)
from src.rade_ml_pt.ensemble.api.routers.governance import (
    router as governance_router,
)
from src.rade_ml_pt.ensemble.api.routers.inference import (
    router as inference_router,
)
from src.rade_ml_pt.ensemble.api.routers.graph_stats import (
    router as graph_stats_router,
)
from src.rade_ml_pt.ensemble.api.routers.group_correlations import (
    router as group_correlations_router,
)
from src.rade_ml_pt.ensemble.api.routers.metrics import router as metrics_router
from src.rade_ml_pt.ensemble.api.routers.overview import router as overview_router
from src.rade_ml_pt.ensemble.api.routers.portfolio import router as portfolio_router
from src.rade_ml_pt.ensemble.api.routers.predictions import (
    router as predictions_router,
)
from src.rade_ml_pt.ensemble.api.routers.quality import router as quality_router
from src.rade_ml_pt.ensemble.api.routers.trade_graph import (
    router as trade_graph_router,
)
from src.rade_ml_pt.ensemble.api.routers.trades import router as trades_router
from src.rade_ml_pt.ensemble.api.routers.training_curves import (
    router as training_curves_router,
)
from src.rade_ml_pt.ensemble.api.services.inference_state import (
    InferenceStateManager,
)
from src.rade_ml_pt.ensemble.api.services.paths import ArtifactPaths
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader
from src.rade_ml_pt.ensemble.api.services.result_reader import (
    InferenceResultReader,
)
from src.rade_ml_pt.ensemble.api.services.version import list_versions

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: resolve version, build paths, register the reader."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    version = settings.resolved_version
    logger.info(
        "PRISM API starting — version=%s (requested=%s), artifacts=%s, registry=%s",
        version,
        settings.ensemble_version,
        settings.artifacts_dir,
        settings.registry_dir,
    )

    paths = ArtifactPaths(
        artifacts_dir=settings.artifacts_path,
        version=version,
    )
    reader = ArtifactReader(paths)
    set_reader(reader)

    # Inference state manager — holds the live EnsembleInferencePipeline
    # for the /inference router.  Empty until POST /inference/load.
    set_inference_state_manager(InferenceStateManager())

    # Inference result reader — serves the on-disk artifacts written
    # by EnsembleInferencePipeline.post_infer (Stage 6) to the
    # historical-run endpoints under /inference/runs/... (Stage 10).
    # Pure read-side; no warm-loading required.
    set_result_reader(InferenceResultReader(settings.artifacts_dir))

    splits = reader.available_splits()
    logger.info(
        "PRISM API ready — version=%s, splits=%s — http://%s:%d/docs",
        version, splits, settings.host, settings.port,
    )

    yield

    logger.info("PRISM API shutting down.")


def create_app(
    artifacts_dir: Optional[str] = None,
    registry_dir: Optional[str] = None,
    ensemble_version: str = "latest",
) -> FastAPI:
    """Build the FastAPI application.

    If *artifacts_dir* and *registry_dir* are passed they override any
    env-var-driven settings — convenient for script-based launches.
    """
    if artifacts_dir is not None and registry_dir is not None:
        set_settings(Settings(
            artifacts_dir=artifacts_dir,
            registry_dir=registry_dir,
            ensemble_version=ensemble_version,
        ))

    settings = get_settings()

    app = FastAPI(
        title="PRISM — Quantitative Model Intelligence",
        description=(
            "REST API serving ensemble evaluation artifacts for the "
            "PRISM dashboard.  Consumed by Dash and (optionally) a "
            "TypeScript/React/Tailwind front-end."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Discovery / liveness endpoints ────────────────────────────
    # Inlined here because they don't fit any artifact-family router
    # and aren't worth their own module.

    @app.get("/health", tags=["meta"], response_model=HealthResponse)
    def health() -> HealthResponse:
        s = get_settings()
        return HealthResponse(
            status="ok",
            version=s.resolved_version,
            artifacts_dir=s.artifacts_dir,
        )

    @app.get("/versions", tags=["meta"], response_model=VersionsResponse)
    def versions() -> VersionsResponse:
        s = get_settings()
        return VersionsResponse(
            active=s.resolved_version,
            available=list_versions(s.registry_path),
        )

    # ── Artifact routers ──────────────────────────────────────────
    app.include_router(overview_router)
    app.include_router(portfolio_router)
    app.include_router(metrics_router)
    app.include_router(clusters_router)
    app.include_router(cluster_timeseries_router)
    app.include_router(trades_router)
    app.include_router(group_correlations_router)
    app.include_router(graph_stats_router)
    app.include_router(trade_graph_router)
    app.include_router(training_curves_router)
    app.include_router(elementary_pnl_router)
    app.include_router(quality_router)
    app.include_router(predictions_router)
    app.include_router(governance_router)
    app.include_router(inference_router)

    return app


def get_app() -> FastAPI:
    """Factory entry-point for ``uvicorn ... --factory`` CLI launch."""
    return create_app()
```

---

### Verification commands (run in your work env after copy-paste)

```bash
# Syntax check on all 7 files
python -m py_compile \
    src/rade_ml_pt/pipelines/ensemble/infer.py \
    src/rade_ml_pt/ensemble/api/services/inference_state.py \
    src/rade_ml_pt/ensemble/api/services/result_reader.py \
    src/rade_ml_pt/ensemble/api/routers/inference.py \
    src/rade_ml_pt/ensemble/api/models/inference.py \
    src/rade_ml_pt/ensemble/api/dependencies.py \
    src/rade_ml_pt/ensemble/api/app.py

# Import check (transitively imports torch + pyarrow + fastapi)
python -c "from src.rade_ml_pt.ensemble.api.app import create_app; print('OK')"

# Boot the server against your real registry + artifacts
PRISM_ARTIFACTS_DIR=/path/to/artifacts \
PRISM_REGISTRY_DIR=/path/to/registry \
PRISM_ENSEMBLE_VERSION=latest \
uvicorn src.rade_ml_pt.ensemble.api.app:get_app --factory --port 8000

# Confirm all 14 inference endpoints are registered
curl -s http://localhost:8000/openapi.json | python -c "
import sys, json
spec = json.load(sys.stdin)
paths = sorted(p for p in spec['paths'] if '/inference' in p)
for p in paths: print(p)
"
```

Expected output of the last command — 14 inference endpoints:

```
/prism/v1/inference/events
/prism/v1/inference/load
/prism/v1/inference/manifest
/prism/v1/inference/run
/prism/v1/inference/runs
/prism/v1/inference/runs/{run_id}/clusters
/prism/v1/inference/runs/{run_id}/clusters/{cluster_id}/trades
/prism/v1/inference/runs/{run_id}/manifest
/prism/v1/inference/runs/{run_id}/portfolio
/prism/v1/inference/runs/{run_id}/scenarios
/prism/v1/inference/runs/{run_id}/validation
/prism/v1/inference/scenarios
/prism/v1/inference/status
/prism/v1/inference/validate
```
