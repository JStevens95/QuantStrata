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

## Appendix A — Phase M.2: Ensemble monitoring pipeline (copy-paste sync)

> **Status**: Seven files below land in the repo. Three production
> modules + an updated `__init__.py` re-export under
> `src/rade_ml_pt/monitoring/` and `src/rade_ml_pt/pipelines/ensemble/`,
> plus three test files under `tests/`. **96/96 tests pass** (53 M.1
> + 43 M.2), lint-clean.
>
> **Scope**: Compose the existing `EnsembleInferencePipeline` to reuse
> its `load → load_scenarios → validate_scenarios` stages, then
> compute per-cluster drift against the training-time baselines from
> `monitoring.baselines.save_feature_baseline`. **Zero edits to
> `infer.py`** — we avoid the work-env drift trap entirely.
>
> **What M.2 ships**:
> * `EnsembleMonitoringPipeline.run(new_scenario_dir)` →
>   `MonitoringResult` with `run_id`, `portfolio_summary`,
>   `drift_tables`, and the manifest path.
> * Per-cluster `drift_table.parquet` written under
>   `monitoring_runs/<run_id>/monitoring/clusters/<cid>/`.
> * Run-level `drift_summary.json` + `manifest.json` written under
>   `monitoring_runs/<run_id>/monitoring/`.
> * Symmetric path conventions in `monitoring.run_paths` (mirrors
>   inference's `inference_runs/<run_id>/inference/...`).
> * NaN/Inf-safe JSON writers — every artefact parses with strict
>   `json.loads` (no `NaN` / `Infinity` literals).
>
> **What M.2 explicitly DOES NOT do** (deferred):
> * **M.3** — `promote_to_predictions(run_id)`: load an existing
>   monitoring manifest, chain into
>   `EnsembleInferencePipeline.run_inference()`, write predictions
>   alongside the drift artifacts in the same run dir.
> * **M.4** — API routers + result reader.
> * **M.5/M.6** — UI layout + callbacks for the Monitoring tab.
>
> **Key design decisions baked in**:
> 1. **Composition over extraction** — `EnsembleMonitoringPipeline`
>    instantiates an `EnsembleInferencePipeline` internally and calls
>    its public `load()` / `load_scenarios()` / `validate_scenarios()`
>    methods. Reads three private attrs of the inference pipeline
>    (`_inference_contexts`, `_validation_report`,
>    `_new_scenario_shocks`) directly. No edits to `infer.py`.
> 2. **Deferred hybrid import** — the
>    `HybridGnnRnnInferencePipeline` import is inside the
>    `if decision.is_affected:` branch so a pure unaffected-only run
>    doesn't need the static-replication module chain (works in this
>    env without `src.rade_sr.market_data_manager`).
> 3. **`version_dir` via `ctx._cluster_assets_path.parent`** — same
>    resolution inference uses; works for both session-warm and
>    registry-cold loads with zero code duplication.
> 4. **Unaffected drift = historical PnL at requested scenario
>    labels** — `ctx.elementary_pnl.loc[scenario_labels]` (same data
>    the cheap-path inference loads). Tells us "did today's scenarios
>    pick a slice of history that looks different from training?".
> 5. **`no_data` fallback for missing baselines** — a cluster without
>    a baseline parquet still gets a `severity=no_data` row so it
>    counts in `n_clusters` and renders as a grey heatmap cell.
> 6. **JSON-strict sanitisation** — `NaN` / `Infinity` /
>    `np.ndarray` / `np.int64` / `Path` all coerced through
>    `_sanitise_for_json` so every artefact is strict-JSON-parseable
>    by non-Python tooling.

### Artifact layout (mirrors `inference_runs/`)

```
{artifacts_dir}/monitoring_runs/<run_id>/monitoring/
  ├── manifest.json            # ensemble_version, run_id, ts, drift_summary, ...
  ├── drift_summary.json       # output of build_portfolio_drift_summary()
  └── clusters/<cid>/
      └── drift_table.parquet  # output of build_drift_table()
                               # (M.3 will add predictions_*.parquet alongside)
```

Where `run_id = "<ensemble_version>__monitor__<UTC_ISO_TS>"` (mirrors
inference's `"<ensemble_version>__<TS>"` with `monitor` segment for
clarity in flat listings).

### Pipeline waterfall

```text
EnsembleMonitoringPipeline.run(new_scenario_dir)
│
├─ 1. inference_pipeline.load()                  (model + per-cluster contexts)
├─ 2. inference_pipeline.load_scenarios(...)     (parse shock CSVs)
├─ 3. inference_pipeline.validate_scenarios()    (routing + cheap-path check)
├─ 4. For each cluster:
│       today_features = _today_features(ctx, decision, shocks, labels)
│       baseline_df    = load_baseline(version_dir(ctx) / "monitoring" / "...")
│       drift_table    = build_drift_table(baseline_df, today_features, cid)
│       write_drift_table_parquet(...)
├─ 5. portfolio_summary = build_portfolio_drift_summary([drift_tables])
│     write_drift_summary_json(...)
│     write_monitoring_manifest_json(...)
│
└─ return MonitoringResult(run_id, summary, drift_tables, manifest_path)
```

### File 1: `src/rade_ml_pt/monitoring/run_paths.py` (full source)

```python
"""Path conventions + run-id helpers for monitoring runs.

Mirrors ``ensemble.api.services.inference_state.per_run_artifacts_dir``
for inference, kept in monitoring/ so the writer / pipeline / API layers
agree on the layout without anyone importing the inference state
manager.  Single source of truth for "where does monitoring write?".

Layout (mirrors ``inference_runs/<run_id>/inference/...``)::

    {artifacts_dir}/monitoring_runs/<run_id>/monitoring/
      ├── manifest.json
      ├── drift_summary.json
      └── clusters/<cid>/drift_table.parquet
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

# Top-level directory under ``artifacts_dir`` for monitoring runs.
# Symmetric with ``inference_runs`` so callers can list both with a
# single ``Path(artifacts_dir).iterdir()`` if they want a unified runs
# index later (M.4 / M.5).
MONITORING_RUNS_DIRNAME: str = "monitoring_runs"

# Nested sub-directory holding the manifest + per-cluster artifacts.
# Mirrors inference's ``inference/`` sub-dir.  Keep it stable across
# M.2 / M.3 / M.4 — the API result reader and the UI both join from
# this constant.
MONITORING_SUBDIRNAME:   str = "monitoring"

# Default file names — also constants so writers + readers stay in
# sync without string typos.
MANIFEST_FILENAME:        str = "manifest.json"
DRIFT_SUMMARY_FILENAME:   str = "drift_summary.json"
DRIFT_TABLE_FILENAME:     str = "drift_table.parquet"

# Run-id segment that distinguishes monitoring runs from inference runs
# when both share an ``ensemble_version`` prefix.  Lets future tooling
# (or grep) tell them apart at a glance.
RUN_ID_SEGMENT:           str = "monitor"


def monitoring_run_id(
    ensemble_version: str,
    *,
    timestamp: Union[datetime, None] = None,
) -> str:
    """Build a deterministic monitoring run-id.

    Format::

        <ensemble_version>__monitor__<UTC_ISO_TS>

    Where ``UTC_ISO_TS`` is ``YYYY-MM-DDTHH-MM-SSZ`` (colon → dash so
    the id is safe as a directory name on every filesystem we care
    about).  Mirrors inference's ``<ensemble_version>__<TS>`` pattern,
    inserting ``monitor`` so monitoring + inference runs are easy to
    distinguish in a flat listing.

    Parameters
    ----------
    ensemble_version
        Ensemble version or tag this run is monitoring.
    timestamp
        Optional override (UTC).  Defaults to ``datetime.now(timezone.utc)``.
        Tests inject a fixed value for determinism.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        # Defensive: callers passing naive datetimes get UTC assumed,
        # not local time — the artefact path must be reproducible.
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    ts_str = timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ensemble_version}__{RUN_ID_SEGMENT}__{ts_str}"


# Run-id parser regex.  Lazy import-time compile so callers that never
# parse run-ids don't pay the cost.  Matches the format produced by
# :func:`monitoring_run_id` strictly — anything looser would invite
# silent collisions with future run-id variants.
_RUN_ID_REGEX = re.compile(
    r"^(?P<ensemble_version>.+)__"
    rf"{re.escape(RUN_ID_SEGMENT)}__"
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)$"
)


def parse_monitoring_run_id(run_id: str) -> dict:
    """Reverse :func:`monitoring_run_id` — return ``{ensemble_version, ts}``.

    Returns an empty dict on parse failure rather than raising; callers
    that need strict validation should check ``bool(...)`` on the
    result.  Used by the M.4 API to populate ``MonitoringRunMeta``
    from a directory listing without re-reading every manifest.
    """
    m = _RUN_ID_REGEX.match(run_id)
    if not m:
        return {}
    return {"ensemble_version": m.group("ensemble_version"), "ts": m.group("ts")}


def per_run_artifacts_dir(base_artifacts_dir: Union[str, Path], run_id: str) -> str:
    """Where this run's artifacts live on disk.

    ``{base_artifacts_dir}/monitoring_runs/<run_id>``.  The pipeline's
    writers further nest a ``monitoring/`` directory underneath for the
    manifest + parquets, so the final manifest path is
    ``{base}/monitoring_runs/<run_id>/monitoring/manifest.json``.

    Centralised here so the pipeline, the writer, and any future
    listing endpoint agree on the layout — same convention used by
    inference (see ``ensemble.api.services.inference_state``).
    """
    return str(Path(base_artifacts_dir) / MONITORING_RUNS_DIRNAME / run_id)


@dataclass(frozen=True)
class MonitoringRunPaths:
    """Resolved on-disk paths for a single monitoring run.

    Constructing this dataclass does NOT touch the filesystem; calling
    :meth:`ensure_dirs` does.  Keep the two concerns separate so unit
    tests can exercise path resolution without ``tmp_path``.

    Attributes
    ----------
    artifacts_dir
        Base artifacts directory (the same value passed to the
        :class:`EnsembleMonitoringPipeline`).  All other paths are
        rooted here.
    run_id
        Monitoring run identifier (see :func:`monitoring_run_id`).
    ensemble_version
        Ensemble version this run is monitoring.  Kept on the dataclass
        so callers don't have to re-derive it from ``run_id`` by
        regex.
    """

    artifacts_dir:    Path
    run_id:           str
    ensemble_version: str

    # ─── Run-level paths ─────────────────────────────────────────────

    @property
    def run_dir(self) -> Path:
        """``{artifacts_dir}/monitoring_runs/<run_id>``."""
        return self.artifacts_dir / MONITORING_RUNS_DIRNAME / self.run_id

    @property
    def monitoring_dir(self) -> Path:
        """``{run_dir}/monitoring`` — holds manifest + summary + clusters/."""
        return self.run_dir / MONITORING_SUBDIRNAME

    @property
    def manifest_path(self) -> Path:
        """``{monitoring_dir}/manifest.json``."""
        return self.monitoring_dir / MANIFEST_FILENAME

    @property
    def drift_summary_path(self) -> Path:
        """``{monitoring_dir}/drift_summary.json`` — portfolio KPIs."""
        return self.monitoring_dir / DRIFT_SUMMARY_FILENAME

    @property
    def clusters_dir(self) -> Path:
        """``{monitoring_dir}/clusters`` — parent of per-cluster sub-dirs."""
        return self.monitoring_dir / "clusters"

    # ─── Per-cluster paths ───────────────────────────────────────────

    def cluster_dir(self, cluster_id: str) -> Path:
        """``{clusters_dir}/<cluster_id>``."""
        return self.clusters_dir / cluster_id

    def cluster_drift_table_path(self, cluster_id: str) -> Path:
        """``{cluster_dir}/drift_table.parquet`` — output of build_drift_table."""
        return self.cluster_dir(cluster_id) / DRIFT_TABLE_FILENAME

    # ─── Filesystem side-effects ─────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create the run + monitoring + clusters directories.

        Per-cluster sub-directories are created lazily by the writer
        each time it writes a cluster table — saves us from having to
        know the cluster list at path-construction time.
        """
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.clusters_dir.mkdir(parents=True, exist_ok=True)


__all__ = [
    # constants
    "MONITORING_RUNS_DIRNAME",
    "MONITORING_SUBDIRNAME",
    "MANIFEST_FILENAME",
    "DRIFT_SUMMARY_FILENAME",
    "DRIFT_TABLE_FILENAME",
    "RUN_ID_SEGMENT",
    # functions
    "monitoring_run_id",
    "parse_monitoring_run_id",
    "per_run_artifacts_dir",
    # dataclass
    "MonitoringRunPaths",
]
```

### File 2: `src/rade_ml_pt/monitoring/writers.py` (full source)

```python
"""Writers + readers for monitoring run artifacts.

Symmetric with inference's ``_post_infer_cluster`` / ``post_infer`` —
two layers:

* **Per-cluster**: :func:`write_drift_table_parquet` (the per-cluster
  output of :func:`monitoring.drift.build_drift_table`).
* **Run-level**:   :func:`write_drift_summary_json` (portfolio KPIs)
  and :func:`write_monitoring_manifest_json` (entry-point pointer that
  the API result reader / UI use to discover every artifact this run
  produced).

JSON writers sanitise ``NaN`` → ``null`` because ``json.dump`` with
``allow_nan=True`` produces ``NaN`` literals that are NOT valid JSON
and break every standards-compliant reader (incl. JavaScript /
``json.loads`` in non-Python tooling).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Bumped whenever the manifest / drift_table schemas change in a
# back-incompatible way.  Mirrors ``monitoring.baselines.SCHEMA_VERSION``
# and ``monitoring.drift.SCHEMA_VERSION``; all three are independent
# but conventionally move together.
SCHEMA_VERSION: int = 1


# ═════════════════════════════════════════════════════════════════════
# Per-cluster drift table (parquet)
# ═════════════════════════════════════════════════════════════════════

def write_drift_table_parquet(
    drift_table: pd.DataFrame,
    out_path:    Union[str, Path],
    *,
    cluster_id:       str,
    ensemble_version: str,
    run_id:           str,
) -> None:
    """Persist one cluster's drift table.

    Schema lives in :data:`monitoring.drift._DRIFT_TABLE_COLUMNS`; this
    writer enforces those columns and embeds ``cluster_id`` /
    ``ensemble_version`` / ``run_id`` / ``_schema_version`` as pyarrow
    schema metadata so a stray parquet file can be reconciled with its
    parent run without consulting the manifest.

    Parameters
    ----------
    drift_table
        Output of :func:`monitoring.drift.build_drift_table`.  Must
        contain the canonical column set (validated lightly here —
        a missing column raises rather than silently dropping data).
    out_path
        Destination file.  Parent dirs are created on demand.
    cluster_id, ensemble_version, run_id
        Provenance triplet stamped on the schema metadata.

    Raises
    ------
    ValueError
        If ``drift_table`` is missing required drift-table columns.
    """
    required = {
        "cluster_id", "feature_name", "psi", "js_divergence",
        "mean_shift", "std_ratio", "severity",
    }
    missing = required - set(drift_table.columns)
    if missing:
        raise ValueError(
            f"drift_table for cluster '{cluster_id}' is missing columns "
            f"{sorted(missing)}.  Did you build it via "
            f"monitoring.drift.build_drift_table?"
        )

    # See ``monitoring.baselines.save_feature_baseline`` for the
    # rationale on ``pa.table`` over ``Table.from_pandas`` (the latter
    # trips a pyarrow stub bug in PyCharm).
    table = pa.table(drift_table)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version":   str(SCHEMA_VERSION).encode(),
        b"cluster_id":        cluster_id.encode(),
        b"ensemble_version":  ensemble_version.encode(),
        b"run_id":            run_id.encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    logger.info(
        "Wrote drift table for cluster '%s' (%d features, severity counts %s)",
        cluster_id,
        len(drift_table),
        dict(drift_table["severity"].value_counts()),
    )


def read_drift_table_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read a drift table parquet written by :func:`write_drift_table_parquet`.

    Pure pass-through to ``pd.read_parquet`` plus a friendlier
    ``FileNotFoundError`` message.  The result has the exact same
    column / dtype schema produced by
    :func:`monitoring.drift.build_drift_table`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"drift table parquet not found: {path}")
    return pd.read_parquet(path)


# ═════════════════════════════════════════════════════════════════════
# Run-level JSON artifacts (drift_summary, manifest)
# ═════════════════════════════════════════════════════════════════════

def write_drift_summary_json(
    summary:  Mapping[str, Any],
    out_path: Union[str, Path],
) -> None:
    """Write the portfolio-level drift summary as JSON.

    ``summary`` is typically the output of
    :func:`monitoring.drift.build_portfolio_drift_summary`.  NaN
    values are coerced to ``None`` (rendered as JSON ``null``) so the
    file is strictly valid JSON — see module docstring for why.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sanitised = _sanitise_for_json(summary)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sanitised, f, indent=2, sort_keys=True)
    logger.info("Wrote drift summary → %s", out_path)


def read_drift_summary_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a drift_summary.json written by :func:`write_drift_summary_json`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"drift summary json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_monitoring_manifest_json(
    manifest: Mapping[str, Any],
    out_path: Union[str, Path],
) -> None:
    """Write the run manifest as JSON.

    The manifest is the canonical entry point that the API result
    reader and the UI use to discover every artifact this run
    produced.  See :class:`pipelines.ensemble.monitor.MonitoringResult`
    for the producer side and ``RADE_UI_DESIGN.md`` for the schema
    expected by the M.4 API.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sanitised = _sanitise_for_json(manifest)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sanitised, f, indent=2, sort_keys=True)
    logger.info("Wrote monitoring manifest → %s", out_path)


def read_monitoring_manifest_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a manifest.json written by :func:`write_monitoring_manifest_json`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"monitoring manifest json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════
# Internal — JSON sanitisation
# ═════════════════════════════════════════════════════════════════════

def _sanitise_for_json(obj: Any) -> Any:
    """Recursive NaN/Inf → None + ndarray → list normalisation.

    ``json.dump(..., allow_nan=True)`` emits ``NaN`` / ``Infinity``
    literals that are NOT in the JSON spec and silently break every
    non-Python consumer.  We normalise here so call-sites don't have
    to remember.

    Also coerces:
    * ``np.ndarray`` → ``list`` (so cluster-id arrays / hist arrays
      embedded in metadata don't crash the encoder)
    * ``np.integer`` / ``np.floating`` → native Python ``int`` /
      ``float`` (round-trip parity across pandas / numpy versions)
    * ``Path`` → ``str``  (manifests embed paths frequently)
    """
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitise_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitise_for_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


__all__ = [
    "SCHEMA_VERSION",
    # parquet
    "write_drift_table_parquet",
    "read_drift_table_parquet",
    # json
    "write_drift_summary_json",
    "read_drift_summary_json",
    "write_monitoring_manifest_json",
    "read_monitoring_manifest_json",
]
```

### File 3: `src/rade_ml_pt/pipelines/ensemble/monitor.py` (full source)

```python
"""Ensemble monitoring pipeline.

Composes :class:`EnsembleInferencePipeline` to reuse the already-stable
load → load_scenarios → validate_scenarios stages, then computes per-
cluster drift against the training-time baselines emitted by
:mod:`monitoring.baselines`.

Pipeline waterfall::

    EnsembleMonitoringPipeline.run(new_scenario_dir)
    │
    ├─ 1. inference.load()                       (model + per-cluster contexts)
    ├─ 2. inference.load_scenarios(...)          (parse shock CSVs)
    ├─ 3. inference.validate_scenarios()         (routing + cheap-path check)
    ├─ 4. for cid in members:
    │       today_features = _today_features(ctx, decision, shocks, labels)
    │       baseline_df    = load_baseline(version_dir(ctx) / "monitoring" / "...")
    │       drift_table    = build_drift_table(baseline_df, today_features, ...)
    │       write_drift_table_parquet(...)
    ├─ 5. portfolio_summary = build_portfolio_drift_summary([drift_tables])
    │     write_drift_summary_json(...)
    │     write_monitoring_manifest_json(...)
    │
    └─ return MonitoringResult(run_id, summary, drift_tables, manifest_path)

NOT implemented in M.2 (deferred to M.3):
* "Promote-to-predictions" step that re-uses the inference pipeline's
  ``run_inference()`` to write ``predictions_*.parquet`` into the SAME
  monitoring run directory.  The hook point is :meth:`run` itself —
  M.3 will add a ``promote_to_predictions(run_id)`` method that
  re-loads the run's manifest and chains into inference.

Threading + parallelism: M.2 runs clusters sequentially.  Drift compute
is sub-second per cluster; the wall-clock bottleneck is the lazy load
of ``cluster_assets`` for affected clusters (shared with inference).
M.4 / M.5 can add ``ThreadPoolExecutor`` if portfolios grow large.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.monitoring.drift import (
    SEVERITY_NO_DATA,
    build_drift_table,
    build_portfolio_drift_summary,
)
from src.rade_ml_pt.monitoring.loaders import load_baseline
from src.rade_ml_pt.monitoring.run_paths import (
    MonitoringRunPaths,
    monitoring_run_id,
)
from src.rade_ml_pt.monitoring.writers import (
    write_drift_summary_json,
    write_drift_table_parquet,
    write_monitoring_manifest_json,
)
from src.rade_ml_pt.pipelines.ensemble.infer import (
    ClusterRoutingDecision,
    EnsembleInferencePipeline,
)
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    EmitFn,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_RUNNING,
    event,
    noop_emit,
)

logger = logging.getLogger(__name__)


# Monitoring schema version stamped onto the manifest.  Bump only on
# back-incompatible manifest shape changes — adding new optional keys
# does NOT require a bump.
MANIFEST_SCHEMA_VERSION: int = 1

# Filename of the baseline parquet under each member's ``version_dir``.
# Must match what ``monitoring.baselines.save_feature_baseline`` writes
# from the training pipeline.
_BASELINE_RELPATH = Path("monitoring") / "baseline_feature_stats.parquet"

# Pipeline stage tag used in activity log emissions.  Kept as a plain
# string (cast at the call-site) because the inference event vocab is
# a Literal[...]; widening it would require touching ``infer_events.py``
# which we explicitly avoid in M.2.  The UI's monitoring activity log
# (M.6) will filter on this tag.
_STAGE_MONITORING: str = "monitoring"


# ═════════════════════════════════════════════════════════════════════
# Result dataclass
# ═════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MonitoringResult:
    """Frozen return value of :meth:`EnsembleMonitoringPipeline.run`.

    Programmatic callers consume this directly; the API layer (M.4)
    will translate it into a Pydantic response model.  Heavy
    per-cluster ``drift_tables`` are also persisted to disk under
    :attr:`MonitoringRunPaths.cluster_drift_table_path` — the in-memory
    copy is convenient for tests and short-lived UI sessions but the
    disk copy is the source of truth for cross-process consumers.
    """

    run_id:            str
    ensemble_version:  str
    artifacts_dir:     Path
    manifest_path:     Path
    portfolio_summary: Dict[str, Any]
    drift_tables:      Dict[str, pd.DataFrame] = field(default_factory=dict)
    n_scenarios:       int                     = 0
    n_clusters:        int                     = 0
    n_affected:        int                     = 0
    n_unaffected:      int                     = 0
    wall_seconds:      float                   = 0.0


# ═════════════════════════════════════════════════════════════════════
# Pipeline
# ═════════════════════════════════════════════════════════════════════

class EnsembleMonitoringPipeline:
    """Run drift monitoring on an ensemble against a new scenario set.

    Composes :class:`EnsembleInferencePipeline` for the load /
    load_scenarios / validate stages — see module docstring for the
    full waterfall.

    Parameters
    ----------
    ensemble_config
        Must have ``registry_dir`` and ``artifacts_dir`` set.  The
        same config a non-UI inference run would use; monitoring uses
        a ``monitoring_runs/<run_id>/`` sibling layout to inference
        artifacts.
    ensemble_version
        Ensemble version or tag to monitor.
    session
        Optional pre-loaded :class:`EnsembleSession` (UI / backend
        warm path).  Must match ``ensemble_version`` — the inference
        pipeline enforces this in :meth:`load`.
    on_event
        Optional lifecycle-event callback.  Forwarded to the
        composed inference pipeline AND used directly by this class
        for monitoring-specific events.  Defaults to a no-op.
    """

    def __init__(
        self,
        ensemble_config:  EnsembleConfig,
        ensemble_version: str = "latest",
        session:          Optional[Any] = None,
        *,
        on_event:         Optional[EmitFn] = None,
    ) -> None:
        self.config           = ensemble_config
        self.ensemble_version = ensemble_version
        self._session         = session
        self._emit: EmitFn    = on_event if on_event is not None else noop_emit

        # Composed inference pipeline — owns the model + contexts +
        # validation report.  Created on every instance so monitoring
        # runs are independent (no cross-run state leak).
        self._inference_pipeline = EnsembleInferencePipeline(
            ensemble_config  = ensemble_config,
            ensemble_version = ensemble_version,
            session          = session,
            on_event         = on_event,
        )

        # Populated by run().
        self._run_paths:    Optional[MonitoringRunPaths] = None
        self._drift_tables: Dict[str, pd.DataFrame]      = {}

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def run(
        self,
        new_scenario_dir: Optional[Union[str, Path]] = None,
    ) -> MonitoringResult:
        """Execute the full monitoring pipeline.

        Convenience wrapper that calls the four staged methods in
        order — symmetric with
        :meth:`EnsembleInferencePipeline.run` so callers can swap one
        for the other when they want a drift-only view of today's
        scenarios.

        Parameters
        ----------
        new_scenario_dir
            Directory containing the new-scenario shock CSVs.  Falls
            back to ``config.metadata['inference']['new_scenario_dir']``
            if ``None`` (matching the inference pipeline's lookup).

        Raises
        ------
        ValueError
            If the inference validation reports user-input errors
            (e.g. unaffected cluster with missing scenario labels).
        """
        logger.info("EnsembleMonitoringPipeline: starting")
        self._emit(_mon_event("Pipeline started", status=STATUS_RUNNING,
                              target=self.ensemble_version))
        t0 = time.perf_counter()

        try:
            self._inference_pipeline.load()
            self._inference_pipeline.load_scenarios(new_scenario_dir)
            report = self._inference_pipeline.validate_scenarios()

            if not report.is_valid:
                raise ValueError(
                    f"Monitoring validation failed with "
                    f"{len(report.errors)} error(s): {report.errors}"
                )

            self._run_paths    = self._init_run_paths()
            self._drift_tables = self._compute_drift_for_all_clusters()
            portfolio_summary  = build_portfolio_drift_summary(
                list(self._drift_tables.values())
            )
            self._write_run_artifacts(portfolio_summary)
        except Exception as exc:
            self._emit(_mon_event(
                "Pipeline failed", status=STATUS_FAIL,
                target=type(exc).__name__, detail=str(exc),
            ))
            raise

        wall = time.perf_counter() - t0
        n_affected   = report.affected_count
        n_unaffected = report.unaffected_count

        assert self._run_paths is not None  # set in the try-block

        result = MonitoringResult(
            run_id            = self._run_paths.run_id,
            ensemble_version  = self.ensemble_version,
            artifacts_dir     = self._run_paths.run_dir,
            manifest_path     = self._run_paths.manifest_path,
            portfolio_summary = portfolio_summary,
            drift_tables      = self._drift_tables,
            n_scenarios       = report.n_scenarios,
            n_clusters        = n_affected + n_unaffected,
            n_affected        = n_affected,
            n_unaffected      = n_unaffected,
            wall_seconds      = wall,
        )

        logger.info(
            "EnsembleMonitoringPipeline: done (%.3fs, %d clusters, severity=%s)",
            wall, result.n_clusters, portfolio_summary.get("severity"),
        )
        self._emit(_mon_event(
            "Pipeline complete", status=STATUS_OK,
            target=f"{wall * 1000:.0f} ms · {result.n_clusters} clusters",
            detail=f"severity={portfolio_summary.get('severity')}",
        ))
        return result

    # ─────────────────────────────────────────────────────────────────
    # Internal — drift compute
    # ─────────────────────────────────────────────────────────────────

    def _compute_drift_for_all_clusters(self) -> Dict[str, pd.DataFrame]:
        """Walk the validation report, compute a drift table per cluster.

        Two paths:
        * **Affected** cluster → today's features = output of
          :meth:`HybridGnnRnnInferencePipeline.build_new_scenario_inputs`
          (same call inference makes; the heavy ``cluster_assets``
          load happens once and is shared by a later promote step).
        * **Unaffected** cluster → today's features =
          ``ctx.elementary_pnl.loc[scenario_labels]`` (the historical
          PnL slice for the requested scenario labels).  This still
          tells us "did today's scenarios pick a slice of history
          that looks different from training?".

        Missing baselines or shape mismatches degrade gracefully — the
        affected feature column emits a ``severity = "no_data"`` row
        rather than crashing the run.
        """
        assert self._inference_pipeline._validation_report is not None, \
            "validate_scenarios() must run first"
        assert self._inference_pipeline._new_scenario_shocks is not None, \
            "load_scenarios() must run first"

        report          = self._inference_pipeline._validation_report
        shocks          = self._inference_pipeline._new_scenario_shocks
        scenario_labels = report.scenario_labels

        drift_tables: Dict[str, pd.DataFrame] = {}

        for decision in report.cluster_decisions:
            cid = decision.cluster_id
            ctx = self._inference_pipeline._inference_contexts[cid]

            self._emit(_mon_event(
                "Computing drift", status=STATUS_RUNNING, target=cid,
                detail="affected" if decision.is_affected else "unaffected",
            ))

            try:
                today_features = self._today_features(
                    ctx              = ctx,
                    decision         = decision,
                    shocks           = shocks,
                    scenario_labels  = scenario_labels,
                )
                baseline_df = self._load_cluster_baseline(ctx, cluster_id=cid)

                drift_table = build_drift_table(
                    baseline_df      = baseline_df,
                    current_features = today_features,
                    cluster_id       = cid,
                )
            except FileNotFoundError as exc:
                # No baseline for this cluster — emit a no_data row per
                # feature in the current frame so the portfolio
                # summary still reflects the cluster's coverage.
                logger.warning(
                    "No baseline found for cluster '%s' (%s) — drift row will "
                    "be emitted as no_data", cid, exc,
                )
                drift_table = _no_data_table_for_cluster(cid)

            drift_tables[cid] = drift_table

            sev_counts = dict(drift_table["severity"].value_counts())
            self._emit(_mon_event(
                "Cluster drift ready", status=STATUS_OK, target=cid,
                detail=f"{len(drift_table)} features · {sev_counts}",
            ))

        return drift_tables

    @staticmethod
    def _today_features(
        ctx:             Any,
        decision:        ClusterRoutingDecision,
        shocks:          Dict[str, Any],
        scenario_labels: List[str],
    ) -> pd.DataFrame:
        """Build today's feature matrix for one cluster.

        Returns a ``DataFrame`` shaped ``(n_scenarios × n_features)``
        in the same coordinate system the training baseline was built
        in (scaled space).  Compatible with
        :func:`monitoring.drift.build_drift_table`.

        Affected path delegates to the same builder inference uses, so
        the two pipelines see byte-identical "today" features for any
        cluster they both touch.  Unaffected path slices the
        cluster's historical ``elementary_pnl`` at the requested
        scenario labels — same data the cheap-path inference loads.

        The heavy ``HybridGnnRnnInferencePipeline`` import is deferred
        to the affected branch so a pure unaffected-only run (or a
        unit test) doesn't pull in the static-replication dependency
        chain.
        """
        if decision.is_affected:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                HybridGnnRnnInferencePipeline,
            )
            result = HybridGnnRnnInferencePipeline.build_new_scenario_inputs(
                ctx                 = ctx,
                new_scenario_shocks = shocks,
                scenario_labels     = scenario_labels,
                is_affected         = True,
            )
            inputs = result["inputs"]
            elementary_pnl = inputs.elementary_pnl
            if elementary_pnl is None:
                raise RuntimeError(
                    f"build_new_scenario_inputs returned no elementary_pnl "
                    f"for cluster '{decision.cluster_id}'."
                )
            return elementary_pnl

        # Unaffected path — historical PnL at the requested labels.
        # Labels are guaranteed present by the validate_scenarios()
        # cheap-path eligibility check (any missing label would have
        # been surfaced as an error and we'd never reach this branch).
        hist = ctx.elementary_pnl
        if hist is None:
            raise RuntimeError(
                f"Cluster '{decision.cluster_id}' is unaffected but its "
                f"InferenceContext has no elementary_pnl loaded."
            )
        return hist.loc[scenario_labels]

    @staticmethod
    def _load_cluster_baseline(ctx: Any, *, cluster_id: str) -> pd.DataFrame:
        """Locate + load the training-time baseline parquet.

        Resolution order (first hit wins):
        1. ``ctx._cluster_assets_path.parent`` — set by
           ``load_inference_context_from_dir``; works for both
           registry cold-loads and session warm-loads.
        2. Future: a ``ctx.version_dir`` field if we add one.

        Raises
        ------
        FileNotFoundError
            If no baseline parquet can be resolved / read.  The caller
            converts this into a no_data drift row.
        """
        if ctx._cluster_assets_path is None:
            raise FileNotFoundError(
                f"Cluster '{cluster_id}': InferenceContext has no "
                f"_cluster_assets_path; cannot derive baseline location."
            )
        version_dir = ctx._cluster_assets_path.parent
        baseline_path = version_dir / _BASELINE_RELPATH
        return load_baseline(baseline_path)

    # ─────────────────────────────────────────────────────────────────
    # Internal — artifact writing
    # ─────────────────────────────────────────────────────────────────

    def _init_run_paths(self) -> MonitoringRunPaths:
        """Mint a fresh run-id + create the on-disk layout."""
        run_id = monitoring_run_id(self.ensemble_version)
        artifacts_dir = Path(self.config.artifacts_dir)
        paths = MonitoringRunPaths(
            artifacts_dir    = artifacts_dir,
            run_id           = run_id,
            ensemble_version = self.ensemble_version,
        )
        paths.ensure_dirs()
        logger.info(
            "Monitoring run dir: %s (run_id=%s)", paths.run_dir, paths.run_id,
        )
        return paths

    def _write_run_artifacts(self, portfolio_summary: Dict[str, Any]) -> None:
        """Persist per-cluster parquets + run-level JSON files."""
        assert self._run_paths is not None

        # Per-cluster drift tables.
        for cid, drift_table in self._drift_tables.items():
            write_drift_table_parquet(
                drift_table      = drift_table,
                out_path         = self._run_paths.cluster_drift_table_path(cid),
                cluster_id       = cid,
                ensemble_version = self.ensemble_version,
                run_id           = self._run_paths.run_id,
            )

        # Run-level summary + manifest.
        write_drift_summary_json(
            summary  = portfolio_summary,
            out_path = self._run_paths.drift_summary_path,
        )
        write_monitoring_manifest_json(
            manifest = self._build_manifest(portfolio_summary),
            out_path = self._run_paths.manifest_path,
        )

    def _build_manifest(self, portfolio_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Construct the canonical run manifest.

        Mirrors the inference manifest schema where it makes sense
        (``schema_version``, ``run_id``, ``ensemble_version``,
        ``created_at``) and adds monitoring-specific keys
        (``drift_summary``, ``cluster_drift_tables``, ``predictions``
        — the last is null in M.2; M.3 populates it on promote).
        """
        assert self._run_paths is not None
        report   = self._inference_pipeline._validation_report
        loaded   = self._inference_pipeline._loaded_scenarios

        # Per-cluster parquet paths stored RELATIVE to the run dir so
        # the manifest moves with the directory tree.  The API reader
        # joins them back against the manifest's own location.
        cluster_drift_tables: Dict[str, str] = {}
        for cid in self._drift_tables:
            abs_path = self._run_paths.cluster_drift_table_path(cid)
            rel_path = abs_path.relative_to(self._run_paths.run_dir)
            cluster_drift_tables[cid] = str(rel_path)

        return {
            "schema_version":       MANIFEST_SCHEMA_VERSION,
            "run_id":                self._run_paths.run_id,
            "ensemble_version":      self.ensemble_version,
            "created_at":            datetime.now(timezone.utc).isoformat(
                                         timespec="seconds"),
            "input_mode":            "new_scenarios",
            "new_scenario_dir":      loaded.new_scenario_dir if loaded else None,
            "n_scenarios":           report.n_scenarios if report else 0,
            "n_clusters":            report.affected_count + report.unaffected_count
                                     if report else 0,
            "n_clusters_affected":   report.affected_count   if report else 0,
            "n_clusters_unaffected": report.unaffected_count if report else 0,
            "drift_summary":         portfolio_summary,
            "cluster_drift_tables":  cluster_drift_tables,
            # M.3 promote step will populate this; nullable in M.2.
            "predictions": None,
        }


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _mon_event(
    phase: str,
    *,
    status: str = STATUS_OK,
    target: Optional[str] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Thin wrapper around :func:`event` that stamps the monitoring stage.

    Suppresses the type-checker warning for passing a non-Literal
    string into ``Stage`` — see ``_STAGE_MONITORING`` docstring.
    """
    return event(
        _STAGE_MONITORING,  # type: ignore[arg-type]
        phase,
        status=status,  # type: ignore[arg-type]
        target=target,
        detail=detail,
    )


def _no_data_table_for_cluster(cluster_id: str) -> pd.DataFrame:
    """Single-row no_data drift table when a cluster has no baseline.

    Lets the portfolio summary's ``n_clusters`` count this cluster
    even though we couldn't score any features for it.  The single
    row carries ``feature_name="__no_baseline__"`` so the UI heatmap
    has something concrete to render in the cluster's column.
    """
    return build_drift_table(
        baseline_df = pd.DataFrame([{
            "feature_name": "__no_baseline__",
            "mean":         float("nan"),
            "std":          float("nan"),
            "hist_edges":   [],
            "hist_counts":  [],
        }]),
        current_features = pd.DataFrame(),
        cluster_id       = cluster_id,
    )


__all__ = [
    "EnsembleMonitoringPipeline",
    "MonitoringResult",
    "MANIFEST_SCHEMA_VERSION",
]
```

### File 4: `src/rade_ml_pt/monitoring/__init__.py` (replace existing — adds M.2 re-exports)

```python
"""Drift monitoring artifacts for the PRISM Model Monitoring surface.

This package houses:

* :mod:`monitoring.baselines` — training-time histogram + summary
  statistics writer (``save_feature_baseline``).
* :mod:`monitoring.loaders`   — reader that decodes baseline parquets
  back into NumPy-ready DataFrames (``load_baseline``).
* :mod:`monitoring.drift`     — pure-NumPy drift primitives + per-cluster
  / portfolio aggregators (PSI, JSD, severity classifier).
* :mod:`monitoring.writers`   — readers/writers for monitoring run
  artifacts (per-cluster drift table parquets + run-level JSON).
* :mod:`monitoring.run_paths` — on-disk layout conventions
  (``monitoring_runs/<run_id>/monitoring/...``) — symmetric with
  inference's ``inference_runs/<run_id>/inference/...``.

See ``docs/platform_designs/prism_retool_migration.md`` Phase 4 for the
full design and ``RADE_UI_DESIGN.md`` for the Monitoring tab consumer.
"""
from .drift import (  # noqa: F401  (re-exported public API)
    PSI_CRITICAL_THRESHOLD,
    PSI_WARN_THRESHOLD,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_NO_DATA,
    SEVERITY_WARN,
    build_drift_table,
    build_portfolio_drift_summary,
    classify_severity,
    js_divergence,
    population_stability_index,
)
from .loaders import load_baseline  # noqa: F401
from .run_paths import (  # noqa: F401
    MonitoringRunPaths,
    monitoring_run_id,
    parse_monitoring_run_id,
    per_run_artifacts_dir,
)
from .writers import (  # noqa: F401
    read_drift_summary_json,
    read_drift_table_parquet,
    read_monitoring_manifest_json,
    write_drift_summary_json,
    write_drift_table_parquet,
    write_monitoring_manifest_json,
)

__all__ = [
    # ─── drift.py ────────────────────────────────────────────────────
    "PSI_WARN_THRESHOLD", "PSI_CRITICAL_THRESHOLD",
    "SEVERITY_INFO", "SEVERITY_WARN", "SEVERITY_CRITICAL", "SEVERITY_NO_DATA",
    "population_stability_index",
    "js_divergence",
    "classify_severity",
    "build_drift_table",
    "build_portfolio_drift_summary",
    # ─── loaders.py ──────────────────────────────────────────────────
    "load_baseline",
    # ─── run_paths.py ────────────────────────────────────────────────
    "MonitoringRunPaths",
    "monitoring_run_id",
    "parse_monitoring_run_id",
    "per_run_artifacts_dir",
    # ─── writers.py ──────────────────────────────────────────────────
    "write_drift_table_parquet",
    "read_drift_table_parquet",
    "write_drift_summary_json",
    "read_drift_summary_json",
    "write_monitoring_manifest_json",
    "read_monitoring_manifest_json",
]
```
### File 5: `tests/rade_ml_pt/monitoring/test_run_paths.py` (full source)

```python
"""Unit tests for ``rade_ml_pt.monitoring.run_paths``.

Pure path-resolution + run-id format tests — no filesystem touching
except for the explicit ``ensure_dirs`` assertion.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.rade_ml_pt.monitoring.run_paths import (
    DRIFT_SUMMARY_FILENAME,
    DRIFT_TABLE_FILENAME,
    MANIFEST_FILENAME,
    MONITORING_RUNS_DIRNAME,
    MONITORING_SUBDIRNAME,
    MonitoringRunPaths,
    monitoring_run_id,
    parse_monitoring_run_id,
    per_run_artifacts_dir,
)


# ═════════════════════════════════════════════════════════════════════
# monitoring_run_id
# ═════════════════════════════════════════════════════════════════════

class TestMonitoringRunId:
    def test_format_with_fixed_timestamp(self):
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        rid = monitoring_run_id("ens_v1", timestamp=ts)
        assert rid == "ens_v1__monitor__2026-05-26T14-30-00Z"

    def test_naive_timestamp_assumed_utc(self):
        ts_naive = datetime(2026, 5, 26, 14, 30, 0)
        rid = monitoring_run_id("ens_v1", timestamp=ts_naive)
        assert rid.endswith("2026-05-26T14-30-00Z")

    def test_now_default_is_zulu_format(self):
        rid = monitoring_run_id("ens_v1")
        # Z suffix + colon-free → safe directory name on every OS
        assert rid.startswith("ens_v1__monitor__")
        assert rid.endswith("Z")
        assert ":" not in rid

    def test_ensemble_version_with_underscores_preserved(self):
        ts = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        rid = monitoring_run_id("ens_v1_2_dev", timestamp=ts)
        assert rid.startswith("ens_v1_2_dev__monitor__")


# ═════════════════════════════════════════════════════════════════════
# parse_monitoring_run_id
# ═════════════════════════════════════════════════════════════════════

class TestParseMonitoringRunId:
    def test_roundtrip(self):
        ts  = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        rid = monitoring_run_id("ens_v1", timestamp=ts)
        parsed = parse_monitoring_run_id(rid)
        assert parsed == {
            "ensemble_version": "ens_v1",
            "ts":               "2026-05-26T14-30-00Z",
        }

    def test_underscored_version_roundtrip(self):
        ts  = datetime(2026, 5, 26, 14, 30, 0, tzinfo=timezone.utc)
        rid = monitoring_run_id("ens_v1_2_dev", timestamp=ts)
        parsed = parse_monitoring_run_id(rid)
        assert parsed["ensemble_version"] == "ens_v1_2_dev"

    @pytest.mark.parametrize("bad", [
        "",
        "ens_v1__monitor__not-a-ts",
        "ens_v1__infer__2026-05-26T14-30-00Z",  # wrong segment
        "no_segments_at_all",
    ])
    def test_invalid_returns_empty_dict(self, bad):
        assert parse_monitoring_run_id(bad) == {}


# ═════════════════════════════════════════════════════════════════════
# per_run_artifacts_dir
# ═════════════════════════════════════════════════════════════════════

class TestPerRunArtifactsDir:
    def test_basic(self):
        out = per_run_artifacts_dir("/tmp/artifacts", "rid")
        assert out == str(Path("/tmp/artifacts") / MONITORING_RUNS_DIRNAME / "rid")

    def test_accepts_path_obj(self):
        out = per_run_artifacts_dir(Path("/tmp/artifacts"), "rid")
        assert MONITORING_RUNS_DIRNAME in out


# ═════════════════════════════════════════════════════════════════════
# MonitoringRunPaths
# ═════════════════════════════════════════════════════════════════════

class TestMonitoringRunPaths:
    def _paths(self, tmp_path: Path) -> MonitoringRunPaths:
        return MonitoringRunPaths(
            artifacts_dir    = tmp_path,
            run_id           = "ens_v1__monitor__2026-05-26T14-30-00Z",
            ensemble_version = "ens_v1",
        )

    def test_run_dir_layout(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.run_dir == tmp_path / MONITORING_RUNS_DIRNAME / p.run_id

    def test_monitoring_dir_nested(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.monitoring_dir == p.run_dir / MONITORING_SUBDIRNAME

    def test_manifest_path(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.manifest_path == p.monitoring_dir / MANIFEST_FILENAME

    def test_drift_summary_path(self, tmp_path):
        p = self._paths(tmp_path)
        assert p.drift_summary_path == p.monitoring_dir / DRIFT_SUMMARY_FILENAME

    def test_cluster_drift_table_path(self, tmp_path):
        p = self._paths(tmp_path)
        cdp = p.cluster_drift_table_path("c0")
        assert cdp == p.clusters_dir / "c0" / DRIFT_TABLE_FILENAME

    def test_ensure_dirs_creates_layout(self, tmp_path):
        p = self._paths(tmp_path)
        assert not p.monitoring_dir.exists()
        p.ensure_dirs()
        assert p.monitoring_dir.is_dir()
        assert p.clusters_dir.is_dir()

    def test_ensure_dirs_idempotent(self, tmp_path):
        p = self._paths(tmp_path)
        p.ensure_dirs()
        p.ensure_dirs()  # second call must not raise
        assert p.monitoring_dir.is_dir()
```

### File 6: `tests/rade_ml_pt/monitoring/test_writers.py` (full source)

```python
"""Unit tests for ``rade_ml_pt.monitoring.writers``.

Round-trip writers ↔ readers + JSON sanitisation edge cases.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.rade_ml_pt.monitoring.drift import (
    SEVERITY_INFO,
    SEVERITY_NO_DATA,
)
from src.rade_ml_pt.monitoring.writers import (
    read_drift_summary_json,
    read_drift_table_parquet,
    read_monitoring_manifest_json,
    write_drift_summary_json,
    write_drift_table_parquet,
    write_monitoring_manifest_json,
)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _drift_table_fixture(cluster_id: str = "c0") -> pd.DataFrame:
    """Canonical drift table — matches build_drift_table output shape."""
    return pd.DataFrame([
        {
            "cluster_id":    cluster_id,
            "feature_name":  "rf_a",
            "psi":           0.05,
            "js_divergence": 0.01,
            "mean_shift":    0.10,
            "std_ratio":     1.02,
            "severity":      SEVERITY_INFO,
        },
        {
            "cluster_id":    cluster_id,
            "feature_name":  "rf_b",
            "psi":           float("nan"),
            "js_divergence": float("nan"),
            "mean_shift":    float("nan"),
            "std_ratio":     float("nan"),
            "severity":      SEVERITY_NO_DATA,
        },
    ]).astype({
        "cluster_id":    "string",
        "feature_name":  "string",
        "psi":           np.float32,
        "js_divergence": np.float32,
        "mean_shift":    np.float32,
        "std_ratio":     np.float32,
        "severity":      "string",
    })


# ═════════════════════════════════════════════════════════════════════
# write_drift_table_parquet ↔ read_drift_table_parquet
# ═════════════════════════════════════════════════════════════════════

class TestDriftTableParquet:
    def test_roundtrip_preserves_columns(self, tmp_path):
        df  = _drift_table_fixture("c0")
        out = tmp_path / "drift_table.parquet"
        write_drift_table_parquet(
            df, out,
            cluster_id="c0", ensemble_version="ens_v1", run_id="rid",
        )
        back = read_drift_table_parquet(out)
        assert list(back.columns) == list(df.columns)
        assert len(back) == len(df)
        assert (back["feature_name"] == df["feature_name"]).all()
        assert back.loc[0, "psi"] == pytest.approx(0.05, abs=1e-5)

    def test_creates_parent_dirs(self, tmp_path):
        df  = _drift_table_fixture()
        out = tmp_path / "deeply" / "nested" / "drift_table.parquet"
        write_drift_table_parquet(
            df, out,
            cluster_id="c0", ensemble_version="ens_v1", run_id="rid",
        )
        assert out.exists()

    def test_schema_metadata_embedded(self, tmp_path):
        df  = _drift_table_fixture("c0")
        out = tmp_path / "drift_table.parquet"
        write_drift_table_parquet(
            df, out,
            cluster_id="c0", ensemble_version="ens_v1", run_id="rid",
        )
        table = pq.read_table(out)
        meta  = table.schema.metadata or {}
        assert meta.get(b"cluster_id")       == b"c0"
        assert meta.get(b"ensemble_version") == b"ens_v1"
        assert meta.get(b"run_id")           == b"rid"
        assert b"_schema_version" in meta

    def test_missing_columns_raises(self, tmp_path):
        bad = pd.DataFrame([{"cluster_id": "c0", "feature_name": "x"}])  # missing psi etc.
        with pytest.raises(ValueError, match="missing columns"):
            write_drift_table_parquet(
                bad, tmp_path / "x.parquet",
                cluster_id="c0", ensemble_version="ens_v1", run_id="rid",
            )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_drift_table_parquet(tmp_path / "nope.parquet")


# ═════════════════════════════════════════════════════════════════════
# write_drift_summary_json
# ═════════════════════════════════════════════════════════════════════

class TestDriftSummaryJson:
    def test_roundtrip(self, tmp_path):
        summary = {
            "n_clusters": 3,
            "mean_psi": 0.12,
            "severity": "warn",
            "worst_cluster": "c1",
        }
        out = tmp_path / "drift_summary.json"
        write_drift_summary_json(summary, out)
        back = read_drift_summary_json(out)
        assert back == summary

    def test_nan_coerced_to_null(self, tmp_path):
        summary = {
            "mean_psi": float("nan"),
            "max_psi":  float("inf"),
            "n_clusters": 0,
        }
        out = tmp_path / "drift_summary.json"
        write_drift_summary_json(summary, out)
        raw = out.read_text()
        assert "NaN" not in raw          # NOT in the JSON spec
        assert "Infinity" not in raw
        assert "null" in raw
        back = read_drift_summary_json(out)
        assert back["mean_psi"]   is None
        assert back["max_psi"]    is None
        assert back["n_clusters"] == 0

    def test_numpy_types_normalised(self, tmp_path):
        summary = {
            "n_clusters":   np.int64(3),
            "mean_psi":     np.float32(0.12),
            "feature_names": np.array(["a", "b"]),
        }
        out = tmp_path / "drift_summary.json"
        write_drift_summary_json(summary, out)
        back = read_drift_summary_json(out)
        assert back["n_clusters"] == 3
        assert back["mean_psi"] == pytest.approx(0.12, abs=1e-6)
        assert back["feature_names"] == ["a", "b"]

    def test_path_values_serialised(self, tmp_path):
        summary = {"some_path": tmp_path / "foo.parquet"}
        out = tmp_path / "drift_summary.json"
        write_drift_summary_json(summary, out)
        back = read_drift_summary_json(out)
        assert isinstance(back["some_path"], str)
        assert back["some_path"].endswith("foo.parquet")

    def test_creates_parent_dirs(self, tmp_path):
        summary = {"k": 1}
        out = tmp_path / "deeply" / "nested" / "ds.json"
        write_drift_summary_json(summary, out)
        assert out.exists()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_drift_summary_json(tmp_path / "nope.json")


# ═════════════════════════════════════════════════════════════════════
# write_monitoring_manifest_json
# ═════════════════════════════════════════════════════════════════════

class TestMonitoringManifestJson:
    def test_roundtrip(self, tmp_path):
        manifest = {
            "schema_version":  1,
            "run_id":          "ens_v1__monitor__2026-05-26T14-30-00Z",
            "ensemble_version": "ens_v1",
            "n_clusters":      3,
            "drift_summary":   {"mean_psi": 0.05},
            "cluster_drift_tables": {
                "c0": "clusters/c0/drift_table.parquet",
            },
            "predictions": None,
        }
        out = tmp_path / "manifest.json"
        write_monitoring_manifest_json(manifest, out)
        back = read_monitoring_manifest_json(out)
        assert back == manifest

    def test_nested_nan_coerced(self, tmp_path):
        manifest = {
            "drift_summary": {
                "mean_psi": float("nan"),
                "nested":   {"deep_nan": float("nan")},
            },
        }
        out = tmp_path / "manifest.json"
        write_monitoring_manifest_json(manifest, out)
        back = read_monitoring_manifest_json(out)
        assert back["drift_summary"]["mean_psi"]            is None
        assert back["drift_summary"]["nested"]["deep_nan"]  is None

    def test_json_is_strictly_valid(self, tmp_path):
        """Confirm the file parses with stdlib json.loads (strict=True)."""
        manifest = {"k": float("nan"), "deep": [float("inf"), 1.0]}
        out = tmp_path / "manifest.json"
        write_monitoring_manifest_json(manifest, out)
        # Default json.loads is strict — would reject NaN/Infinity
        parsed = json.loads(out.read_text())
        assert parsed["k"]      is None
        assert parsed["deep"]   == [None, 1.0]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_monitoring_manifest_json(tmp_path / "nope.json")
```

### File 7: `tests/rade_ml_pt/pipelines/ensemble/test_monitor.py` (full source)

```python
"""Unit + integration tests for ``EnsembleMonitoringPipeline``.

Strategy:
* Test each helper (``_today_features``, ``_load_cluster_baseline``,
  ``_init_run_paths``, ``_build_manifest``, ``_write_run_artifacts``)
  in isolation with synthetic inputs.
* One end-to-end ``run()`` test stitches the helpers together via a
  monkey-patched ``EnsembleInferencePipeline`` so we never touch the
  real ensemble loader or any model.

Mocking convention: we monkey-patch the *names* the module imported
(``monitor.EnsembleInferencePipeline``, etc.) so the real classes are
free to evolve without breaking these tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.rade_ml_pt.monitoring.baselines import save_feature_baseline
from src.rade_ml_pt.monitoring.drift import (
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_NO_DATA,
)
from src.rade_ml_pt.monitoring.run_paths import MonitoringRunPaths


# ═════════════════════════════════════════════════════════════════════
# Lightweight stand-ins for the inference dataclasses
# ═════════════════════════════════════════════════════════════════════

@dataclass
class _FakeCtx:
    """Substitute for InferenceContext — only the fields monitor.py touches."""
    elementary_pnl:       Optional[pd.DataFrame] = None
    _cluster_assets_path: Optional[Path]         = None


@dataclass
class _FakeRoutingDecision:
    """Substitute for ClusterRoutingDecision."""
    cluster_id:               str
    is_affected:              bool
    missing_scenario_labels:  List[str] = field(default_factory=list)


@dataclass
class _FakeValidationReport:
    """Substitute for ValidationReport — exposes the fields run() reads."""
    ensemble_version:   str
    n_scenarios:        int
    scenario_labels:    List[str]
    cluster_decisions:  List[_FakeRoutingDecision]
    errors:             List[str] = field(default_factory=list)
    warnings:           List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def affected_count(self) -> int:
        return sum(d.is_affected for d in self.cluster_decisions)

    @property
    def unaffected_count(self) -> int:
        return sum(not d.is_affected for d in self.cluster_decisions)


@dataclass
class _FakeLoadedScenarios:
    new_scenario_dir:  str
    n_scenarios:       int
    scenario_labels:   List[str]


class _FakeInferencePipeline:
    """In-memory replacement for ``EnsembleInferencePipeline``.

    Exposes the same private attrs ``monitor.py`` reads
    (``_inference_contexts``, ``_validation_report``, etc.) plus the
    three public methods (``load``, ``load_scenarios``,
    ``validate_scenarios``) the monitoring pipeline calls.
    """

    def __init__(
        self,
        ensemble_config,
        ensemble_version: str,
        session=None,
        *,
        on_event=None,
    ):
        self.config           = ensemble_config
        self.ensemble_version = ensemble_version
        self._session         = session
        self._inference_contexts: Dict[str, _FakeCtx]               = {}
        self._validation_report:  Optional[_FakeValidationReport]   = None
        self._loaded_scenarios:   Optional[_FakeLoadedScenarios]    = None
        self._new_scenario_shocks: Optional[Dict[str, Any]]         = None

        # Test hooks — populated by the test BEFORE run() is called.
        self._test_contexts:        Dict[str, _FakeCtx]              = {}
        self._test_decisions:       List[_FakeRoutingDecision]       = []
        self._test_scenario_labels: List[str]                        = []

    def load(self) -> None:
        self._inference_contexts = dict(self._test_contexts)

    def load_scenarios(self, new_scenario_dir=None) -> None:
        self._loaded_scenarios = _FakeLoadedScenarios(
            new_scenario_dir = str(new_scenario_dir),
            n_scenarios      = len(self._test_scenario_labels),
            scenario_labels  = list(self._test_scenario_labels),
        )
        self._new_scenario_shocks = {"rf_x": {lab: 0.01 for lab in self._test_scenario_labels}}

    def validate_scenarios(self) -> _FakeValidationReport:
        self._validation_report = _FakeValidationReport(
            ensemble_version  = self.ensemble_version,
            n_scenarios       = len(self._test_scenario_labels),
            scenario_labels   = list(self._test_scenario_labels),
            cluster_decisions = list(self._test_decisions),
        )
        return self._validation_report


# ═════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture
def patch_inference_pipeline(monkeypatch):
    """Swap the production EnsembleInferencePipeline for the in-memory fake."""
    from src.rade_ml_pt.pipelines.ensemble import monitor
    monkeypatch.setattr(monitor, "EnsembleInferencePipeline", _FakeInferencePipeline)
    return monitor


@pytest.fixture
def ensemble_config(tmp_path):
    """Minimal EnsembleConfig substitute — monitor.py only reads artifacts_dir."""
    return SimpleNamespace(
        registry_dir  = str(tmp_path / "registry"),
        artifacts_dir = str(tmp_path / "artifacts"),
        metadata      = {"inference": {}},
    )


@pytest.fixture
def baseline_parquet_path(tmp_path):
    """Write a real baseline parquet on disk for the unaffected-path test."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame({
        "trade_0": rng.normal(0, 1, 5000),
        "trade_1": rng.normal(2, 1, 5000),
    })
    version_dir = tmp_path / "registry" / "cluster-c0-v1"
    out = version_dir / "monitoring" / "baseline_feature_stats.parquet"
    save_feature_baseline(out, features, cluster_id="c0")
    return out


# ═════════════════════════════════════════════════════════════════════
# Helpers — pipeline construction
# ═════════════════════════════════════════════════════════════════════

def _scenario_labels(n: int) -> List[str]:
    return [f"s{i}" for i in range(n)]


def _historical_pnl(scenario_labels: List[str], rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic historical elementary PnL — index = scenario labels."""
    return pd.DataFrame(
        {
            "trade_0": rng.normal(0, 1, len(scenario_labels)),
            "trade_1": rng.normal(2, 1, len(scenario_labels)),
        },
        index=scenario_labels,
    )


def _make_pipeline(monitor_module, ensemble_config):
    """Construct EnsembleMonitoringPipeline with the patched fake injected."""
    return monitor_module.EnsembleMonitoringPipeline(
        ensemble_config  = ensemble_config,
        ensemble_version = "ens_v1",
    )


# ═════════════════════════════════════════════════════════════════════
# Helper-level tests
# ═════════════════════════════════════════════════════════════════════

class TestTodayFeatures:
    """``_today_features`` — unaffected path is pure; affected path delegates."""

    def test_unaffected_path_returns_historical_slice(self):
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        labels = _scenario_labels(5)
        rng    = np.random.default_rng(0)
        ctx    = _FakeCtx(elementary_pnl=_historical_pnl(labels, rng))
        dec    = _FakeRoutingDecision(cluster_id="c0", is_affected=False)

        out = EnsembleMonitoringPipeline._today_features(
            ctx=ctx, decision=dec, shocks={}, scenario_labels=labels,
        )
        assert list(out.index) == labels
        assert set(out.columns) == {"trade_0", "trade_1"}

    def test_unaffected_path_raises_without_elementary_pnl(self):
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        ctx = _FakeCtx(elementary_pnl=None)
        dec = _FakeRoutingDecision(cluster_id="c0", is_affected=False)
        with pytest.raises(RuntimeError, match="no elementary_pnl"):
            EnsembleMonitoringPipeline._today_features(
                ctx=ctx, decision=dec, shocks={}, scenario_labels=["s0"],
            )

    def test_affected_path_delegates_to_hybrid_builder(self, monkeypatch):
        """The affected path calls HybridGnnRnnInferencePipeline.build_new_scenario_inputs.

        We inject a fake module into ``sys.modules`` before the deferred
        import inside ``_today_features`` runs, so this test works
        even in environments without the hybrid_gnn_rnn dependency
        chain (``rade_sr.market_data_manager`` etc.).
        """
        import sys
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        labels = _scenario_labels(3)
        synthetic_pnl = pd.DataFrame(
            {"trade_0": [0.1, 0.2, 0.3]},
            index=labels,
        )

        def _fake_builder(*, ctx, new_scenario_shocks, scenario_labels, is_affected):
            assert is_affected is True
            return {"inputs": SimpleNamespace(elementary_pnl=synthetic_pnl)}

        fake_module = SimpleNamespace(
            HybridGnnRnnInferencePipeline=SimpleNamespace(
                build_new_scenario_inputs=_fake_builder,
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer",
            fake_module,
        )

        out = EnsembleMonitoringPipeline._today_features(
            ctx              = _FakeCtx(),
            decision         = _FakeRoutingDecision(cluster_id="c0", is_affected=True),
            shocks           = {"rf_x": {"s0": 0.01}},
            scenario_labels  = labels,
        )
        pd.testing.assert_frame_equal(out, synthetic_pnl)


class TestLoadClusterBaseline:
    """``_load_cluster_baseline`` — resolution via ``_cluster_assets_path.parent``."""

    def test_loads_real_baseline_via_ctx_path(self, baseline_parquet_path):
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        version_dir = baseline_parquet_path.parent.parent
        # _cluster_assets_path.parent must equal version_dir
        ctx = _FakeCtx(_cluster_assets_path=version_dir / "cluster_assets.joblib")

        df = EnsembleMonitoringPipeline._load_cluster_baseline(ctx, cluster_id="c0")
        assert "hist_edges"  in df.columns
        assert "hist_counts" in df.columns
        assert set(df["feature_name"]) == {"trade_0", "trade_1"}

    def test_missing_assets_path_raises_file_not_found(self):
        from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

        ctx = _FakeCtx(_cluster_assets_path=None)
        with pytest.raises(FileNotFoundError, match="no _cluster_assets_path"):
            EnsembleMonitoringPipeline._load_cluster_baseline(ctx, cluster_id="c0")


class TestInitRunPaths:
    def test_creates_layout_and_returns_paths(self, patch_inference_pipeline, ensemble_config):
        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        paths = pipe._init_run_paths()
        assert isinstance(paths, MonitoringRunPaths)
        assert paths.monitoring_dir.is_dir()
        assert paths.clusters_dir.is_dir()
        assert paths.run_id.startswith("ens_v1__monitor__")


# ═════════════════════════════════════════════════════════════════════
# End-to-end run() — unaffected cluster against real baseline
# ═════════════════════════════════════════════════════════════════════

class TestRunEndToEnd:
    def test_drift_only_with_unaffected_cluster_writes_all_artifacts(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        # Build a real baseline parquet for cluster c0.
        rng = np.random.default_rng(0)
        training_features = pd.DataFrame({
            "trade_0": rng.normal(0, 1, 5000),
            "trade_1": rng.normal(0, 1, 5000),
        })
        version_dir = Path(ensemble_config.registry_dir) / "cluster-c0-v1"
        save_feature_baseline(
            version_dir / "monitoring" / "baseline_feature_stats.parquet",
            features   = training_features,
            cluster_id = "c0",
        )

        # 500 today obs against 5000 baseline obs keeps sample-size
        # noise in PSI low enough for stable assertions; under-sampling
        # alone can push PSI above the info threshold even when the
        # underlying distribution is identical.
        labels    = _scenario_labels(500)
        today_pnl = pd.DataFrame(
            {
                "trade_0": rng.normal(0, 1, len(labels)),
                "trade_1": rng.normal(0, 1, len(labels)),
            },
            index=labels,
        )

        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        # Inject the test fixtures BEFORE calling run() — the fake
        # inference pipeline copies them when load() / load_scenarios()
        # are called.
        fake = pipe._inference_pipeline
        fake._test_contexts = {
            "c0": _FakeCtx(
                elementary_pnl       = today_pnl,
                _cluster_assets_path = version_dir / "cluster_assets.joblib",
            ),
        }
        fake._test_decisions       = [_FakeRoutingDecision("c0", is_affected=False)]
        fake._test_scenario_labels = labels

        result = pipe.run(new_scenario_dir=tmp_path / "scenarios")

        # ── MonitoringResult shape ─────────────────────────────────
        assert result.run_id.startswith("ens_v1__monitor__")
        assert result.n_clusters     == 1
        assert result.n_affected     == 0
        assert result.n_unaffected   == 1
        assert result.n_scenarios    == 500
        assert "c0" in result.drift_tables

        # ── Portfolio summary ──────────────────────────────────────
        # The point of this test is pipeline wiring, not drift
        # numerics — only assert that summary keys are populated and
        # severity comes from the canonical set.
        summary = result.portfolio_summary
        assert summary["n_clusters"]    == 1
        assert summary["severity"]      in {"info", "warn", "critical", "no_data"}
        assert "mean_psi" in summary

        # ── On-disk artifacts exist ────────────────────────────────
        manifest_path = result.manifest_path
        assert manifest_path.is_file()
        drift_summary_path = manifest_path.parent / "drift_summary.json"
        assert drift_summary_path.is_file()
        cluster_parquet = manifest_path.parent / "clusters" / "c0" / "drift_table.parquet"
        assert cluster_parquet.is_file()

        # ── Manifest content ───────────────────────────────────────
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"]       == 1
        assert manifest["ensemble_version"]     == "ens_v1"
        assert manifest["n_clusters"]            == 1
        assert manifest["n_clusters_unaffected"] == 1
        assert "c0" in manifest["cluster_drift_tables"]
        assert manifest["cluster_drift_tables"]["c0"].endswith("drift_table.parquet")
        # M.2 hasn't promoted yet
        assert manifest["predictions"] is None

    def test_invalid_validation_report_raises(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        fake = pipe._inference_pipeline
        labels = _scenario_labels(3)
        fake._test_contexts        = {"c0": _FakeCtx()}
        fake._test_decisions       = [_FakeRoutingDecision("c0", is_affected=False)]
        fake._test_scenario_labels = labels

        # Override validate_scenarios to inject an error
        def _bad_validate():
            report = _FakeValidationReport(
                ensemble_version  = "ens_v1",
                n_scenarios       = 3,
                scenario_labels   = labels,
                cluster_decisions = fake._test_decisions,
                errors            = ["synthetic validation error"],
            )
            fake._validation_report = report
            return report
        fake.validate_scenarios = _bad_validate  # type: ignore[assignment]

        with pytest.raises(ValueError, match="validation failed"):
            pipe.run(new_scenario_dir=tmp_path / "scenarios")

    def test_missing_baseline_emits_no_data_table(
        self, patch_inference_pipeline, ensemble_config, tmp_path,
    ):
        """Cluster with no baseline parquet → no_data drift row, run still succeeds."""
        labels    = _scenario_labels(5)
        rng       = np.random.default_rng(0)
        today_pnl = _historical_pnl(labels, rng)

        # Note: NO baseline parquet written, but ctx points to a fake
        # version_dir that doesn't have one.
        version_dir = Path(ensemble_config.registry_dir) / "cluster-c0-v1"
        version_dir.mkdir(parents=True, exist_ok=True)

        pipe = _make_pipeline(patch_inference_pipeline, ensemble_config)
        fake = pipe._inference_pipeline
        fake._test_contexts = {
            "c0": _FakeCtx(
                elementary_pnl       = today_pnl,
                _cluster_assets_path = version_dir / "cluster_assets.joblib",
            ),
        }
        fake._test_decisions       = [_FakeRoutingDecision("c0", is_affected=False)]
        fake._test_scenario_labels = labels

        result = pipe.run(new_scenario_dir=tmp_path / "scenarios")
        c0_drift = result.drift_tables["c0"]
        assert (c0_drift["severity"] == SEVERITY_NO_DATA).all()
        # Manifest still written
        assert result.manifest_path.is_file()
```

### Verification (copy-paste into work env)

After copying the seven files above into work env, run:

```bash
# Lint (style + unused imports)
ruff check src/rade_ml_pt/monitoring/run_paths.py \
           src/rade_ml_pt/monitoring/writers.py \
           src/rade_ml_pt/monitoring/__init__.py \
           src/rade_ml_pt/pipelines/ensemble/monitor.py \
           tests/rade_ml_pt/monitoring/test_run_paths.py \
           tests/rade_ml_pt/monitoring/test_writers.py \
           tests/rade_ml_pt/pipelines/ensemble/test_monitor.py

# Run the new tests (M.2 surface only) — should be 43 passing
pytest tests/rade_ml_pt/monitoring/test_run_paths.py \
       tests/rade_ml_pt/monitoring/test_writers.py \
       tests/rade_ml_pt/pipelines/ensemble/test_monitor.py -v

# Full monitoring surface (M.1 + M.2) — should be 96 passing
pytest tests/rade_ml_pt/monitoring/ \
       tests/rade_ml_pt/pipelines/ensemble/test_monitor.py -v
```

Expected output: **96 passed** (53 M.1 + 43 M.2). If the work env
also has `src.rade_sr.market_data_manager` available, the affected-
path test will run against the real `HybridGnnRnnInferencePipeline`
on import; the deferred import in `monitor._today_features` means
the test still passes either way.

### Quick smoke (non-test, manual)

If you want to confirm the pipeline runs against a real ensemble in
work env (NOT a test — pretty-print on the side):

```python
from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

config = EnsembleConfig(...)   # same config you use for inference
pipe   = EnsembleMonitoringPipeline(
    ensemble_config  = config,
    ensemble_version = "ens_v1",
)
result = pipe.run(new_scenario_dir="/path/to/new_scenarios/")

print(result.run_id)
print(result.portfolio_summary)        # → {n_clusters: ..., mean_psi: ..., severity: "info|warn|critical|no_data", ...}
print(list(result.drift_tables.keys()))  # → ["c0", "c1", ...]
print(result.manifest_path)              # → {artifacts_dir}/monitoring_runs/<run_id>/monitoring/manifest.json
```

The run dir contains everything the M.4 API will eventually serve.

### What M.3 will consume

The M.3 promote-to-predictions step will:

1. Load `manifest.json` from a known `run_id`.
2. Re-build the inference pipeline at the same `ensemble_version` and
   `new_scenario_dir` (both stored in the manifest).
3. Call `inference_pipeline.run_inference()` and re-route its writers
   to `manifest_path.parent` so `predictions_*.parquet` land next to
   the drift tables.
4. Update the manifest's `predictions` field to point to the new
   files and bump `schema_version` if the shape changes.

No edits to `infer.py` are needed for M.3 either — `run_inference()`
already accepts a target artifacts directory, and the monitoring
manifest is the only file we need to rewrite.

---
