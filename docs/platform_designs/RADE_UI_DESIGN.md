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
## Appendix A — Inference Console (Phase 1 · pipeline events + backend wrapper)

> **Status:** Phase 1 of 6 complete. The page **layout** lives in
> `src/ui/apps/rade_analytics/layouts/inference.py` (V2 — already
> shipped); this appendix covers the new **runtime substrate**
> Phase 1 adds underneath it: a lifecycle-event protocol on the
> ensemble pipeline plus a thin UI-facing backend wrapper.

The **incredibly detailed** long-form spec — system architecture,
all data structures, six-phase roadmap, testing strategy, glossary
— lives in **[`docs/rade_analytics/inference_implementation.md`](../rade_analytics/inference_implementation.md)**.
Read that first; this appendix is the implementation cookbook.

---

### Appendix A · 0 — TL;DR for reviewers

| Question | Answer |
|---|---|
| What does this PR change? | Adds a lifecycle-event hook to `EnsembleInferencePipeline` and a `RadeBackend.run_inference(...)` wrapper that captures it. No UI behaviour changes in this PR. |
| What does it not change? | Existing pipeline behaviour — `on_event=None` (the default) is a behavioural no-op, proved by `TestPipelineEmits.test_default_on_event_is_noop_behavioural`. |
| Why? | The Inference Console UI (Phase 2) needs a streamable narration of every run for its activity-log card. Building that on `logging` would couple the UI to every other subsystem's chatter; a first-class hook keeps the contract narrow. |
| What's the surface area? | **2 new files**, **2 patched files**, **2 new test files**. ~750 lines total, ~30% of which is docstrings. |
| How do I verify? | `pytest tests/rade_ml_pt/pipelines/ensemble/test_infer_events.py tests/ui/apps/rade_analytics/test_run_inference_backend.py` → 22 tests, ~1.5 s. |

---

### Appendix A · 1 — File touch summary

| Action | Path | Bytes (approx) |
|---|---|---|
| **NEW** | `src/rade_ml_pt/pipelines/ensemble/infer_events.py` | full file in §3.1 |
| **PATCH** | `src/rade_ml_pt/pipelines/ensemble/infer.py` | additive only — diff in §3.2 |
| **PATCH** | `src/ui/apps/rade_analytics/data/backend.py` | additive only — diff in §3.3 |
| **NEW** | `tests/rade_ml_pt/pipelines/ensemble/test_infer_events.py` | snippet in §4.1 |
| **NEW** | `tests/ui/apps/rade_analytics/test_run_inference_backend.py` | snippet in §4.2 |
| **NEW** | `tests/ui/apps/rade_analytics/conftest.py` | re-exports `registry_with_members` so the backend test can reuse the synthetic registry fixture |
| **NEW** | `docs/rade_analytics/inference_implementation.md` | the long-form spec |

The Phase 1 layout (`layouts/inference.py`, `figures/inference_charts.py`,
`router.py`, `assets/rade.css`) is **unchanged**; the layout-only
"V2" appendix that previously lived here was a placeholder for the
runtime contract that Phase 1 now fills in.  The **layout gotcha**
section (Appendix A · 6) is preserved so future phases don't re-
introduce the squashed-grid bug.

---

### Appendix A · 2 — Data structures (one-page summary)

The full catalogue (with rationale, ownership, and lifecycle
notes) is in
[`inference_implementation.md` §4](../rade_analytics/inference_implementation.md).
Quick reference:

| Name | Where it lives | Wire shape | Owner |
|---|---|---|---|
| `ActivityEntry` | `src/rade_ml_pt/pipelines/ensemble/infer_events.py` | `dict` with `id`, `stage`, `phase`, `status`, `ts`, optional `target` / `detail` | Pipeline emits, UI Store consumes (Phase 2). |
| `EmitFn` | same | `Callable[[ActivityEntry], None]` | The pipeline accepts; `EventCollector` is the canonical implementation. |
| `EventCollector` | same | thread-safe in-memory buffer that **is itself an `EmitFn`** | Backend wrapper instantiates one per run. |
| `InferenceRunResult` | `src/ui/apps/rade_analytics/data/backend.py` | frozen dataclass — `ensemble_version`, `n_scenarios`, `n_targets`, `latency_seconds`, `predictions`, `sample_ids`, `activity_log` | `RadeBackend.run_inference` returns. UI callbacks (Phase 2) consume. |
| `BackendResult[InferenceRunResult]` | same | tri-state envelope (`ok` / `error` / `status_code`) | Returned by `run_inference`; UI never sees raw exceptions. |

Stage / status vocabularies — these are part of the public
contract, do not invent new tokens without a UI patch:

- `Stage`  ∈ `"ingest" | "validate" | "inference"`
- `Status` ∈ `"ok" | "fail" | "running" | "pending"`

The Phase 1 pipeline uses only `inference` events.  Ingest and
validate are reserved for Phase 2 callbacks (file upload + manifest
sanity).

---

### Appendix A · 3 — Source files (full code or diff)

#### 3.1 — `src/rade_ml_pt/pipelines/ensemble/infer_events.py`  (NEW · full file)

This module is *new* and self-contained — copy in verbatim.

```python
"""Lifecycle events emitted by :class:`EnsembleInferencePipeline`.
... (full docstring + module body) ...
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional


Stage  = Literal["ingest", "validate", "inference"]
Status = Literal["ok", "fail", "running", "pending"]

STAGE_INGEST    : Stage = "ingest"
STAGE_VALIDATE  : Stage = "validate"
STAGE_INFERENCE : Stage = "inference"

STATUS_OK      : Status = "ok"
STATUS_FAIL    : Status = "fail"
STATUS_RUNNING : Status = "running"
STATUS_PENDING : Status = "pending"


ActivityEntry = Dict[str, Any]


@dataclass(frozen=True)
class TypedActivityEntry:
    stage:   Stage
    phase:   str
    status:  Status
    target:  Optional[str] = None
    detail:  Optional[str] = None
    id:      str           = field(default_factory=lambda: uuid.uuid4().hex)
    ts:      str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    def to_dict(self) -> ActivityEntry:
        d: ActivityEntry = {
            "id":     self.id,
            "stage":  self.stage,
            "phase":  self.phase,
            "status": self.status,
            "ts":     self.ts,
        }
        if self.target is not None:
            d["target"] = self.target
        if self.detail is not None:
            d["detail"] = self.detail
        return d


EmitFn = Callable[[ActivityEntry], None]


def noop_emit(_entry: ActivityEntry) -> None:
    return None


def event(
    stage:   Stage,
    phase:   str,
    *,
    status:  Status              = STATUS_OK,
    target:  Optional[str]       = None,
    detail:  Optional[str]       = None,
) -> ActivityEntry:
    return TypedActivityEntry(
        stage=stage, phase=phase, status=status,
        target=target, detail=detail,
    ).to_dict()


class EventCollector:
    def __init__(self) -> None:
        self._events: List[ActivityEntry] = []
        self._lock = threading.Lock()

    def __call__(self, entry: ActivityEntry) -> None:
        with self._lock:
            self._events.append(entry)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def snapshot(self) -> List[ActivityEntry]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


__all__ = [
    "ActivityEntry", "EmitFn", "EventCollector",
    "Stage", "Status",
    "STAGE_INFERENCE", "STAGE_INGEST", "STAGE_VALIDATE",
    "STATUS_FAIL", "STATUS_OK", "STATUS_PENDING", "STATUS_RUNNING",
    "TypedActivityEntry", "event", "noop_emit",
]
```

> ⚠️ **Subtle pitfall** — `EventCollector` defines `__len__`, so an
> empty collector is *falsy*.  Always default the emit-fn with
> `on_event if on_event is not None else noop_emit`, never
> `on_event or noop_emit` (the latter silently drops every event
> on a fresh collector).

#### 3.2 — `src/rade_ml_pt/pipelines/ensemble/infer.py`  (PATCH · additive)

Three changes; all **additive**, none alter existing behaviour
when `on_event=None`.

**Change 1 — top of file, after the existing imports:**

```python
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    EmitFn,
    STAGE_INFERENCE,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_RUNNING,
    event,
    noop_emit,
)
```

**Change 2 — `EnsembleInferencePipeline.__init__` signature + body:**

```python
def __init__(
    self,
    ensemble_config: EnsembleConfig,
    ensemble_version: str = "latest",
    session: Optional["EnsembleSession"] = None,
    *,
    on_event: Optional[EmitFn] = None,         # NEW
) -> None:
    self.config = ensemble_config
    self.ensemble_version = ensemble_version
    self._session = session
    self._emit: EmitFn = on_event if on_event is not None else noop_emit  # NEW
    # ... rest unchanged ...
```

**Change 3 — emit at the seam points** (full table in
[`inference_implementation.md` §3.2](../rade_analytics/inference_implementation.md#32-srcrade_ml_ptpipelinesensembleinferpy-patch)).

The eight call-sites and their exact phase strings:

| Where | Emit |
|---|---|
| `run()` first line | `event(STAGE_INFERENCE, "Pipeline started", status=STATUS_RUNNING, target=self.ensemble_version)` |
| `_load_from_registry` start | `"Loading ensemble from registry"`, `running` |
| after `builder.build()` | `"Ensemble assembled"`, `ok`, target `"N members"` |
| per-cluster context start / done / fail | `"Cluster context loading / loaded / failed"` |
| `_load_from_session` start / done | `"Reusing pre-loaded session"` / `"Session ready"` |
| `_build_member_inputs` short-circuit | `"Using pre-built member inputs"`, `ok` |
| `_build_new_scenarios_inputs` start / done | `"Loading new-scenario shocks"` / `"Shocks loaded"` |
| per-cluster build start / done / fail | `"Building cluster inputs"` / `"Cluster inputs ready"` / `"Cluster input build failed"` |
| around `self._ensemble.predict(...)` | `"Forward pass started"` (`running`) → `"Forward pass complete"` (`ok`) |
| `run()` last line on success | `"Pipeline complete"`, `ok` |
| `run()` outer `except` | `"Pipeline failed"`, `fail`, target `type(exc).__name__`, detail `str(exc)` — re-raises afterwards |

The full patched file is in the repo at
`src/rade_ml_pt/pipelines/ensemble/infer.py`; the diff is small
(~70 lines added, 0 removed) — read the file in your IDE rather
than reproducing it inline here.

#### 3.3 — `src/ui/apps/rade_analytics/data/backend.py`  (PATCH · additive)

Two additions; both at the bottom of the existing file.

**Change 1 — add the `InferenceRunResult` dataclass next to
`BackendResult`:**

```python
@dataclass(frozen=True)
class InferenceRunResult:
    """Typed payload returned by :meth:`RadeBackend.run_inference`.
    ... see file for full docstring ...
    """
    ensemble_version: str
    n_scenarios:      int
    n_targets:        int
    latency_seconds:  float
    predictions:      "np.ndarray"
    sample_ids:       Optional[List[str]]
    activity_log:     List[Dict[str, Any]]
```

**Change 2 — append `RadeBackend.run_inference(...)` at the
end of the class:**

```python
def run_inference(
    self,
    *,
    registry_dir: Any,
    ensemble_version: str,
    new_scenario_dir: Any,
    member_inputs: Optional[Dict[str, Any]] = None,
) -> "BackendResult[InferenceRunResult]":
    """Execute one ensemble inference run and capture its lifecycle.
    ... see file for full docstring ...
    """
    from pathlib import Path

    from src.rade_ml_pt.ensemble.config import EnsembleConfig  # noqa: F401
    from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
    from src.rade_ml_pt.pipelines.ensemble.infer import (
        EnsembleInferencePipeline,
    )
    from src.rade_ml_pt.pipelines.ensemble.infer_events import (
        EventCollector,
    )

    registry_path = Path(registry_dir)
    try:
        ens_registry = EnsembleRegistry(registry_path)
        config, _member_versions, resolved_version = ens_registry.load(
            ensemble_version,
        )
    except Exception as exc:
        logger.exception("ensemble registry load failed")
        return BackendResult.failure(
            error=f"Could not load ensemble '{ensemble_version}': {exc}",
        )

    config.registry_dir = str(registry_path)

    infer_meta: Dict[str, Any] = {"input_mode": "new_scenarios"}
    if member_inputs is not None:
        infer_meta["member_inputs"] = member_inputs
    else:
        infer_meta["new_scenario_dir"] = str(new_scenario_dir)
    config.metadata["inference"] = infer_meta

    collector = EventCollector()
    try:
        pipeline = EnsembleInferencePipeline(
            config,
            ensemble_version=resolved_version,
            on_event=collector,
        )
        inference_result = pipeline.run()
    except Exception as exc:
        logger.exception("inference run failed")
        return BackendResult.failure(
            error=f"Inference failed: {type(exc).__name__}: {exc}",
        )

    run_result = InferenceRunResult(
        ensemble_version=resolved_version,
        n_scenarios=int(inference_result.n_samples),
        n_targets=int(
            inference_result.predictions.shape[1]
            if inference_result.predictions.ndim > 1 else 1
        ),
        latency_seconds=float(inference_result.latency_seconds),
        predictions=inference_result.predictions,
        sample_ids=inference_result.sample_ids,
        activity_log=collector.snapshot(),
    )
    return BackendResult.success(run_result)
```

**Three invariants codified in this method:**

1. *Lazy imports* — pulls in pytorch / pipeline only on first
   call so other backend methods stay cheap.
2. *Uncached* — caching would silently bypass the activity log on
   a hit, breaking the UI narration.
3. *Tri-state envelope* — load failures + run failures both come
   back as `BackendResult.failure(...)`; the UI never sees a raw
   traceback.

---

### Appendix A · 4 — Test files

Both tests are dependency-free (use the `MagicMock` HTTP client
and the synthetic two-cluster fixture from
`tests/rade_ml_pt/pipelines/ensemble/conftest.py`).  Total runtime
under 2 seconds.

#### 4.1 — `tests/rade_ml_pt/pipelines/ensemble/test_infer_events.py`  (NEW)

Outline (full file in the repo):

- `TestEventFactory` — required keys, ID uniqueness, ISO-8601 ts.
- `TestTypedActivityEntryRoundTrip` — dataclass ↔ wire-dict.
- `TestNoopEmit` — total no-op.
- `TestEventCollector` — append, snapshot-is-a-copy, clear,
  thread-safety under 8 × 250 concurrent appends.
- `TestPipelineEmits` — six integration tests:
  1. Default `on_event=None` is behavioural no-op (existing
     callers untouched).
  2. Required phase strings (`"Pipeline started"` … `"Pipeline
     complete"`) all present after a successful run.
  3. First event is `Pipeline started` / `running`.
  4. Last event on success is `Pipeline complete` / `ok`.
  5. Last event on failure is `Pipeline failed` / `fail`, *before*
     the exception re-raises.
  6. Total event count is in `[5, 30]` for the 2-cluster fixture
     (sanity bound).

Run with:

```bash
pytest tests/rade_ml_pt/pipelines/ensemble/test_infer_events.py -v
```

#### 4.2 — `tests/ui/apps/rade_analytics/test_run_inference_backend.py`  (NEW)

Four assertions over `RadeBackend.run_inference` end-to-end:

1. Returns `BackendResult.success` with the typed
   `InferenceRunResult`.
2. `activity_log[0]` is `Pipeline started`, `activity_log[-1]` is
   `Pipeline complete`.
3. Unknown ensemble version → `BackendResult.failure(...)`,
   no exception escapes the wrapper.
4. Tag inputs (e.g. `"backend_test"`) resolve to a concrete
   `ens_*` version string — so subsequent artifact reads pin to
   one snapshot.

Run with:

```bash
pytest tests/ui/apps/rade_analytics/test_run_inference_backend.py -v
```

---

### Appendix A · 5 — End-to-end Phase 1 verification

After landing the patches:

```bash
# 1) Lints clean (no new warnings introduced)
python -m pyflakes src/rade_ml_pt/pipelines/ensemble/infer_events.py \
                   src/rade_ml_pt/pipelines/ensemble/infer.py \
                   src/ui/apps/rade_analytics/data/backend.py

# 2) New tests pass
pytest tests/rade_ml_pt/pipelines/ensemble/test_infer_events.py \
       tests/ui/apps/rade_analytics/test_run_inference_backend.py -q

# 3) Existing pipeline tests still pass (regression check —
#    proves on_event=None is a behavioural no-op)
pytest tests/rade_ml_pt/pipelines/ensemble/test_ensemble_pipelines.py::TestEnsembleInferencePipeline::test_infer_new_scenarios \
       tests/rade_ml_pt/pipelines/ensemble/test_ensemble_pipelines.py::TestEnsembleInferencePipeline::test_infer_saves_csv \
       tests/rade_ml_pt/pipelines/ensemble/test_ensemble_pipelines.py::TestEnsembleInferencePipeline::test_infer_unknown_mode_raises -q

# 4) Smoke import — the lazy-import path works under a fresh interpreter
python -c "
from src.ui.apps.rade_analytics.data.backend import (
    RadeBackend, NoOpCache, InferenceRunResult, BackendResult,
)
import inspect
sig = inspect.signature(RadeBackend.run_inference)
assert list(sig.parameters) == [
    'self', 'registry_dir', 'ensemble_version', 'new_scenario_dir', 'member_inputs',
]
assert [f.name for f in InferenceRunResult.__dataclass_fields__.values()] == [
    'ensemble_version', 'n_scenarios', 'n_targets',
    'latency_seconds', 'predictions', 'sample_ids', 'activity_log',
]
print('OK')
"
```

Expected output: 22 tests pass, 3 regression tests pass, smoke
imports `OK`.

---

### Appendix A · 6 — Layout gotcha: `rade.css` is hand-compiled, not JIT

*(Preserved from the V2 layout appendix — still relevant for any
new utility classes future phases want to add.)*

`src/ui/apps/rade_analytics/assets/rade.css` is a **hand-compiled
Tailwind bundle**, not a runtime JIT pass.  Only utilities that
already shipped (see `tailwind.input.css` + the previous build
scan) are present at runtime.  Classes that look fine in the source
but **silently no-op in the browser** include:

- `grid-cols-12`, `col-span-5`, `col-span-7`, `col-span-12`
- `lg:col-span-5`, `lg:col-span-7`
- `gap-y-5`
- arbitrary values: `min-h-[…]`, `max-h-[…]`, `max-w-[…]`, `min-w-[…]`
- opacity variants: `bg-slate-900/35`, `bg-slate-900/40`,
  `bg-slate-950/50`, `border-slate-700/70`, `border-slate-700/80`
- `border-dashed`, `italic`, `list-disc`, `space-y-1`,
  logical-property `ps-*` / `pe-*`

**Approved substitutions** (already in `rade.css`):

| Need | Use instead |
| --- | --- |
| Two-column workspace | `grid grid-cols-1 lg:grid-cols-5 gap-4 items-stretch` parent + `lg:col-span-2` / `lg:col-span-3` (Cluster Deep-Dive idiom) |
| Soft transparent surface | `bg-slate-900/60` (or `bg-slate-950/40`) |
| Soft border | `border-slate-800/60` or `border-slate-700` |
| Arbitrary heights | inline `style={"minHeight": "…", "maxHeight": "…"}` |
| Dashed border | inline `style={"borderStyle": "dashed"}` |
| Bullet lists | plain stacked `html.Div`s with leading `"• "` |
| Italic text | drop or replace with `text-slate-500` weight cue |

Whenever you add a new utility class, sanity-check it appears in
`rade.css` (or rebuild the bundle per `assets/README.md`); the page
will render but content will collapse to natural width if the
parent grid loses its column-span utilities.

---

### Appendix A · 7 — What's next (Phase 2 preview)

Phase 2 will:

1. Wire a Dash `long_callback` from the Run button to
   `RadeBackend.run_inference`, draining `EventCollector` to the
   page's `activity_log_store` via `set_progress`.
2. Add an upload callback that turns the file bar + path
   `TextInput` into an `IngestMeta` payload feeding the manifest
   preview card.
3. Compute and bind the KPI strip + distribution chart from
   `InferenceRunResult.predictions`.

The full callback wiring table is in
[`inference_implementation.md` §5](../rade_analytics/inference_implementation.md#5--how-the-ui-consumes-phase-1).
