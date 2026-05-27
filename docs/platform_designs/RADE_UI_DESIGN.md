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

## Appendix A — Phase M.4: Monitoring API layer (copy-paste sync)

> **Status**: 117/117 tests pass (53 M.1 + 43 M.2 + 8 M.3 + 13 M.4),
> lint-clean.  M.4 ships the FastAPI surface for the Monitoring tab:
> 13 endpoints across one new router, mirroring the inference router's
> structure 1:1.  All new code below pastes verbatim into the work env.

### Scope

| Concern              | M.4 status | Notes                                                                                     |
|----------------------|-----------|-------------------------------------------------------------------------------------------|
| Control plane        | SHIPPED   | `/load`, `/scenarios`, `/validate`, `/run`, `/promote`, `/status`, `/events`, `/manifest` |
| Drift data plane     | SHIPPED   | `/runs`, `/runs/{id}/manifest`, `/runs/{id}/drift_summary`, `/runs/{id}/clusters`, `/runs/{id}/clusters/{cid}/drift` |
| Predictions data plane | DEFERRED to M.4.5 | Will compose `InferenceResultReader` over `<run_dir>/monitoring/` when shipped     |
| Multi-user concurrency | DEFERRED  | Same Option-A → Option-B path as inference                                              |
| `load_existing_run()`  | DEFERRED  | Promote rides on the same in-memory state manager that drift created the run on        |

### Endpoint map

```
Control plane (active run)
  POST /prism/v1/monitoring/load        EnsembleMonitoringPipeline.load()
  POST /prism/v1/monitoring/scenarios   .load_scenarios()
  POST /prism/v1/monitoring/validate    .validate_scenarios()
  POST /prism/v1/monitoring/run         .compute_drift()                   non-blocking
  POST /prism/v1/monitoring/promote     .promote_to_predictions()          non-blocking
  GET  /prism/v1/monitoring/status      cheap status probe
  GET  /prism/v1/monitoring/events      cursor-paginated activity log tail
  GET  /prism/v1/monitoring/manifest    active run's manifest.json

Data plane (historical runs)
  GET  /prism/v1/monitoring/runs                                 list every run on disk
  GET  /prism/v1/monitoring/runs/{run_id}/manifest               historical manifest
  GET  /prism/v1/monitoring/runs/{run_id}/drift_summary          portfolio drift KPIs
  GET  /prism/v1/monitoring/runs/{run_id}/clusters               per-cluster severity index
  GET  /prism/v1/monitoring/runs/{run_id}/clusters/{cid}/drift   per-cluster drift table
```

### State machine

```
created → loading → loaded → scenarios_loaded → validated
                                                   ↓
                                                running
                                                   ↓
                                                complete  ──▶ promoting
                                                                  ↓
                                                              promoted
(any state) → failed
```

`/run` gated on `status == validated`.  `/promote` gated on
`status == complete`.  Both dispatch onto a worker thread; the HTTP
response returns immediately and the UI polls `/status` until a
terminal state.

### File layout

```
src/rade_ml_pt/ensemble/api/
├── services/
│   ├── monitoring_state.py      ← NEW (state manager + per-run state)
│   └── monitoring_reader.py     ← NEW (historical artifact reader)
├── models/
│   └── monitoring.py            ← NEW (Pydantic response models)
├── routers/
│   └── monitoring.py            ← NEW (13-endpoint FastAPI router)
├── dependencies.py              ← MODIFIED (re-export new manager + reader)
└── app.py                       ← MODIFIED (register router + singletons)

src/rade_ml_pt/pipelines/ensemble/monitor.py
                                 ← MODIFIED (extracted compute_drift() +
                                   proxy methods so the API router can
                                   drive each stage atomically)

tests/rade_ml_pt/ensemble/api/test_monitoring_api.py
                                 ← NEW (13 happy-path tests, mocks the
                                   monitoring pipeline)
```

### Pipeline refactor (`monitor.py`) — net behaviour zero

The router needs to drive each stage as a separate REST call, so we
extracted a `compute_drift()` method out of `run()`.  `run()` itself
now delegates through the four staged methods — same on-disk output,
same events, same `MonitoringResult`, but the staged path and the
orchestrated path share one implementation.

**Diff summary** (all changes are additive or pure-refactor):

1. `__init__`: add `self._validation_report: Optional[ValidationReport] = None`.
2. Add four thin public proxy methods that forward to the composed
   inference pipeline: `load()`, `load_scenarios()`,
   `validate_scenarios()` (caches the report), and a new
   `compute_drift()`.
3. `run()` body is now a 4-line orchestrator that calls
   `self.load()` → `self.load_scenarios()` → `self.validate_scenarios()`
   → `self.compute_drift()` (with the existing `Pipeline started`
   event emitted before delegation).
4. The post-`validate_scenarios()` body of the old `run()` becomes
   the body of `compute_drift()` verbatim — no semantic changes.

### `dependencies.py` — add three re-exports

```python
from src.rade_ml_pt.ensemble.api.services.monitoring_reader import (
    MonitoringResultReader,
    get_monitoring_result_reader,
    set_monitoring_result_reader,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_state import (
    MonitoringStateManager,
    get_monitoring_state_manager,
    set_monitoring_state_manager,
)
```

Add the six new symbols to `__all__`.

### `app.py` — register manager + reader + mount router

```python
from src.rade_ml_pt.ensemble.api.dependencies import (
    set_inference_state_manager,
    set_monitoring_result_reader,
    set_monitoring_state_manager,
    ...
)
from src.rade_ml_pt.ensemble.api.routers.monitoring import (
    router as monitoring_router,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_reader import (
    MonitoringResultReader,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_state import (
    MonitoringStateManager,
)

# Inside lifespan():
set_monitoring_state_manager(MonitoringStateManager())
set_monitoring_result_reader(MonitoringResultReader(settings.artifacts_dir))

# Inside create_app(), after app.include_router(inference_router):
app.include_router(monitoring_router)
```

### Verification

```bash
# Lint
ruff check src/rade_ml_pt/ensemble/api/

# Unit + integration tests (M.1 → M.4)
pytest tests/rade_ml_pt/monitoring/ \
       tests/rade_ml_pt/pipelines/ensemble/test_monitor.py \
       tests/rade_ml_pt/ensemble/api/test_monitoring_api.py -v

# Endpoint listing smoke test
python -c "
from src.rade_ml_pt.ensemble.api.routers.monitoring import router
for r in router.routes:
    methods = sorted(r.methods - {'HEAD'})
    print(f'{methods[0]:6} {r.path}')
"
```

Expected: 117 passed, 13 endpoints listed.

---

## Full source of new files

The four files below paste verbatim into the work env at their
respective paths under `src/rade_ml_pt/ensemble/api/`.


### `src/rade_ml_pt/ensemble/api/services/monitoring_state.py` (NEW, 410 lines)

```python
"""Monitoring run-state holder for the PRISM API.

Holds the live :class:`EnsembleMonitoringPipeline` between HTTP requests
so the staged workflow (``load → load_scenarios → validate_scenarios →
run → promote_to_predictions``) survives the stateless boundary between
calls.

Mirrors :mod:`services.inference_state` deliberately — same singleton
pattern, same status enum shape, same background-thread dispatch —
so the M.4 router can keep its mental model identical to the
inference router.

State model — Option A (single-user, single-process)
----------------------------------------------------
One active :class:`MonitoringRunState` at a time on the
:class:`MonitoringStateManager`.  Calling ``create_run`` while a prior
run exists replaces it.  Sufficient for the single-user dashboard;
Option B (multi-user, dict-keyed) is the same mechanical refactor
documented on :class:`InferenceStateManager`.

State machine
-------------
::

    created → loading → loaded → scenarios_loaded → validated
                                                       ↓
                                                    running
                                                       ↓
                                                    complete  ──▶ promoting
                                                                       ↓
                                                                   promoted
    (any) → failed

``/run`` is gated on ``status == validated``.  ``/promote`` is gated
on ``status == complete``.  Both endpoints dispatch onto a background
thread (Stage 9 of the inference router precedent) so the HTTP
request returns immediately and the UI polls ``/status`` until a
terminal state is reached.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.monitoring.run_paths import (
    MONITORING_RUNS_DIRNAME,
    MONITORING_SUBDIRNAME,
)
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    ActivityEntry,
    EventCollector,
)
from src.rade_ml_pt.pipelines.ensemble.monitor import EnsembleMonitoringPipeline

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Status enum (str-valued for transparent JSON encoding)
# ──────────────────────────────────────────────────────────────────────

STATUS_CREATED:    str = "created"
STATUS_LOADING:    str = "loading"
STATUS_LOADED:     str = "loaded"
STATUS_SCENARIOS:  str = "scenarios_loaded"
STATUS_VALIDATED:  str = "validated"
STATUS_RUNNING:    str = "running"
STATUS_COMPLETE:   str = "complete"
STATUS_PROMOTING:  str = "promoting"
STATUS_PROMOTED:   str = "promoted"
STATUS_FAILED:     str = "failed"

# Terminal states — used by the router to gate which transitions are
# valid from any given snapshot.  Promoted is terminal w.r.t. this run
# but a fresh run can always be created on top of it.
TERMINAL_STATES = frozenset({STATUS_COMPLETE, STATUS_PROMOTED, STATUS_FAILED})


# ──────────────────────────────────────────────────────────────────────
# Per-run state
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MonitoringRunState:
    """One monitoring run's mutable state.

    Constructed by :meth:`MonitoringStateManager.create_run`; thereafter
    only lifecycle fields (``status``, ``last_error``, ``artifacts_dir``,
    ``manifest_path``, ``predictions_dir``) are mutated.

    Attributes
    ----------
    run_id
        Stable identifier for this monitoring run.  The router's
        ``/load`` endpoint mints a provisional id immediately;
        ``pipeline.run()`` overwrites it with the canonical monitoring
        run id once the underlying :class:`EnsembleMonitoringPipeline`
        decides the on-disk path.  Why provisional → final: load()
        doesn't know the run id yet (that's resolved at run() time)
        but the router needs SOMETHING to return on the load response.
    ensemble_version
        Ensemble version this run was constructed against.
    pipeline
        Live :class:`EnsembleMonitoringPipeline` instance — the
        underlying inference pipeline is composed inside it.
    activity_log
        Thread-safe event buffer.  Captures both monitoring stage
        events and any inference stage events emitted by the
        composed pipeline (monitoring forwards on_event to inference).
    status
        Current lifecycle state — one of the ``STATUS_*`` constants.
    created_at
        UTC ISO-8601 timestamp at construction.
    last_error
        Detail message for the most recent failure.  ``None`` when
        no failure has occurred.
    artifacts_dir
        Resolved per-run monitoring artifacts directory.  Set when
        the drift run completes; equals
        ``<base>/monitoring_runs/<run_id>``.
    manifest_path
        Convenience pointer to the run's ``manifest.json``.  Populated
        when the drift run completes.
    predictions_dir
        Convenience pointer to the predictions sub-directory
        (``<manifest_path.parent>/inference/``).  Populated when
        promote-to-predictions completes.
    """

    run_id:           str
    ensemble_version: str
    pipeline:         EnsembleMonitoringPipeline
    activity_log:     EventCollector
    status:           str           = STATUS_CREATED
    created_at:       str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    last_error:       Optional[str] = None
    artifacts_dir:    Optional[str] = None
    manifest_path:    Optional[str] = None
    predictions_dir:  Optional[str] = None

    # Background-execution machinery — populated when ``/run`` or
    # ``/promote`` dispatches onto a worker thread.  Excluded from
    # ``repr`` so log lines stay readable.
    _thread:          Optional[threading.Thread] = field(default=None, repr=False)

    @property
    def is_alive(self) -> bool:
        """True iff a background run / promote thread is currently executing."""
        return self._thread is not None and self._thread.is_alive()

    def transition(self, new_status: str, *, error: Optional[str] = None) -> None:
        """Move the run to ``new_status`` and clear / set ``last_error``.

        Single mutator for the run lifecycle so the
        ``status ⇒ last_error`` invariant (set only when FAILED) is
        enforced in one place.
        """
        self.status = new_status
        self.last_error = error if new_status == STATUS_FAILED else None


# ──────────────────────────────────────────────────────────────────────
# State manager
# ──────────────────────────────────────────────────────────────────────

class MonitoringStateManager:
    """Single source of truth for the API's active monitoring run.

    Today (Option A) holds **one** :class:`MonitoringRunState` at a
    time.  Same Option-A → Option-B migration story as
    :class:`InferenceStateManager`; ``run_id`` is already plumbed
    through every accessor.
    """

    def __init__(self) -> None:
        self._active: Optional[MonitoringRunState] = None
        self._lock = threading.Lock()

    # ── Construction / lookup ─────────────────────────────────────

    def create_run(
        self,
        ensemble_config:  EnsembleConfig,
        ensemble_version: str,
    ) -> MonitoringRunState:
        """Construct a new monitoring pipeline + state object.

        Any previously-active run is silently replaced.  The
        ``run_id`` returned here is provisional — see the docstring
        on :attr:`MonitoringRunState.run_id` for the
        provisional-vs-final reasoning.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        provisional_run_id = f"{ensemble_version}__monitor__{ts}"

        # EventCollector captures monitoring stage events AND any
        # inference stage events emitted by the composed pipeline
        # (EnsembleMonitoringPipeline forwards on_event into its
        # inner EnsembleInferencePipeline).
        activity_log = EventCollector()

        pipeline = EnsembleMonitoringPipeline(
            ensemble_config  = ensemble_config,
            ensemble_version = ensemble_version,
            session          = None,
            on_event         = activity_log,
        )

        state = MonitoringRunState(
            run_id           = provisional_run_id,
            ensemble_version = ensemble_version,
            pipeline         = pipeline,
            activity_log     = activity_log,
        )

        with self._lock:
            self._active = state

        logger.info(
            "MonitoringStateManager: created run %s (provisional)",
            provisional_run_id,
        )
        return state

    def get_run(self, run_id: Optional[str] = None) -> MonitoringRunState:
        """Look up the active run by ID.

        When ``run_id`` is None the currently-active run is returned
        (v1 single-user convention).
        """
        with self._lock:
            state = self._active

        if state is None:
            raise RuntimeError(
                "No active monitoring run — call POST /monitoring/load first."
            )
        if run_id is not None and run_id != state.run_id:
            raise RuntimeError(
                f"Monitoring run '{run_id}' not found; active run is "
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
        """Return events emitted since ``cursor`` for one run.

        Same cursor protocol as
        :meth:`InferenceStateManager.events_since`.
        """
        state    = self.get_run(run_id)
        snapshot = state.activity_log.snapshot()
        if cursor < 0:
            cursor = 0
        if cursor > len(snapshot):
            cursor = len(snapshot)
        return snapshot[cursor:], len(snapshot)

    # ── Background-execution helpers ──────────────────────────────

    def start_run_in_background(
        self,
        target: Callable[[], None],
    ) -> threading.Thread:
        """Spawn the background run / promote thread for the active run.

        Single-thread-per-run by construction.  The ``target`` closure
        is responsible for the terminal status transition (COMPLETE /
        PROMOTED / FAILED).  Daemon mode is on so SIGTERM doesn't
        hang on a long monitoring run.

        Raises
        ------
        RuntimeError
            If a background thread is already running — prevents
            double-dispatch from a refreshed browser tab.
        """
        state = self.get_run()
        if state.is_alive:
            raise RuntimeError(
                f"Run {state.run_id} already has a background thread running."
            )

        thread = threading.Thread(
            target = target,
            name   = f"monitor-{state.run_id}",
            daemon = True,
        )
        state._thread = thread        # noqa: SLF001 — module-internal
        thread.start()
        logger.info(
            "MonitoringStateManager: started background thread for run %s",
            state.run_id,
        )
        return thread


# ──────────────────────────────────────────────────────────────────────
# Helpers used by router and dependency
# ──────────────────────────────────────────────────────────────────────

def build_monitoring_ensemble_config(
    registry_dir:    str,
    artifacts_dir:   str,
    new_scenario_dir: Optional[str] = None,
) -> EnsembleConfig:
    """Construct a minimal :class:`EnsembleConfig` for an API-driven monitoring run.

    Symmetric with :func:`build_ensemble_config` in
    :mod:`services.inference_state`.  Carries ``input_mode =
    "new_scenarios"`` because monitoring only supports the
    new-scenarios input mode today (new-trades is deferred).
    """
    metadata = {"inference": {"input_mode": "new_scenarios"}}
    if new_scenario_dir is not None:
        metadata["inference"]["new_scenario_dir"] = new_scenario_dir

    return EnsembleConfig(
        registry_dir  = registry_dir,
        artifacts_dir = artifacts_dir,
        metadata      = metadata,
    )


def per_run_monitoring_artifacts_dir(base_artifacts_dir: str, run_id: str) -> str:
    """Convention for where per-run monitoring artifacts go on disk.

    ``<base_artifacts_dir>/monitoring_runs/<run_id>``.  Mirrors
    inference's :func:`per_run_artifacts_dir` exactly — same
    "one subdir per run, named for the family" layout.

    Centralised here so the router, the state manager, and the
    monitoring reader agree on the layout without cross-coupling.
    """
    return str(Path(base_artifacts_dir) / MONITORING_RUNS_DIRNAME / run_id)


def manifest_path_for_run(base_artifacts_dir: str, run_id: str) -> Path:
    """Resolve the canonical manifest.json path for a monitoring run.

    ``<base>/monitoring_runs/<run_id>/monitoring/manifest.json``.
    Same constants the writer uses
    (:mod:`monitoring.run_paths`).
    """
    return (
        Path(base_artifacts_dir)
        / MONITORING_RUNS_DIRNAME
        / run_id
        / MONITORING_SUBDIRNAME
        / "manifest.json"
    )


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton
# ──────────────────────────────────────────────────────────────────────

_manager: Optional[MonitoringStateManager] = None


def set_monitoring_state_manager(manager: MonitoringStateManager) -> None:
    """Inject the manager at app-lifespan startup."""
    global _manager
    _manager = manager


def get_monitoring_state_manager() -> MonitoringStateManager:
    """FastAPI ``Depends`` — returns the singleton manager."""
    if _manager is None:
        raise RuntimeError(
            "MonitoringStateManager not initialised. Server not ready."
        )
    return _manager


__all__ = [
    # ── Status constants ─────────────────────────────────────────
    "STATUS_CREATED", "STATUS_LOADING", "STATUS_LOADED",
    "STATUS_SCENARIOS", "STATUS_VALIDATED", "STATUS_RUNNING",
    "STATUS_COMPLETE", "STATUS_PROMOTING", "STATUS_PROMOTED",
    "STATUS_FAILED", "TERMINAL_STATES",
    # ── State machinery ─────────────────────────────────────────
    "MonitoringRunState",
    "MonitoringStateManager",
    "set_monitoring_state_manager",
    "get_monitoring_state_manager",
    # ── Helpers ─────────────────────────────────────────────────
    "build_monitoring_ensemble_config",
    "per_run_monitoring_artifacts_dir",
    "manifest_path_for_run",
]

```


### `src/rade_ml_pt/ensemble/api/services/monitoring_reader.py` (NEW, 300 lines)

```python
"""Read-side counterpart to the monitoring pipeline's writers.

Serves the manifest + drift artifacts written by
:class:`EnsembleMonitoringPipeline` for any monitoring run on disk.
The dashboard's data plane (``GET /monitoring/runs/...``) reads
exclusively through this class; the control plane (``/load``,
``/scenarios``, ``/validate``, ``/run``, ``/promote``, ``/status``,
``/events``) goes through :class:`MonitoringStateManager` for the
*active* run.

On-disk layout
--------------
Mirrors the monitoring pipeline's
:class:`MonitoringRunPaths`.  Per run::

    <base_artifacts_dir>/monitoring_runs/<run_id>/monitoring/
    ├── manifest.json                             ← entry point
    ├── drift_summary.json
    ├── clusters/<cid>/drift_table.parquet
    └── inference/                                ← optional (after promote)
        ├── manifest.json
        ├── cluster_summary/...
        ├── portfolio_summary/...
        └── trade_predictions/...

The ``inference/`` subdir is M.3's promote-to-predictions output and
is NOT served by this reader in M.4 — predictions data plane is
deferred to M.4.5 (will compose :class:`InferenceResultReader` over
``<run_dir>/monitoring/`` when shipped).

State model — Option A (process-wide singleton)
-----------------------------------------------
Mirrors :class:`InferenceResultReader`: one reader per process,
constructed at lifespan startup with the base artifacts dir from
:class:`Settings`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.rade_ml_pt.monitoring.run_paths import (
    DRIFT_SUMMARY_FILENAME,
    DRIFT_TABLE_FILENAME,
    MANIFEST_FILENAME,
    MONITORING_RUNS_DIRNAME,
    MONITORING_SUBDIRNAME,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Per-run filesystem layout
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonitoringRunReaderPaths:
    """Resolved on-disk paths for one monitoring run (reader view).

    Symmetric with inference's :class:`RunPaths` and with the writer-
    side :class:`MonitoringRunPaths`.  Pure derived data — does NOT
    verify any path exists.  The router checks existence and raises
    the appropriate HTTPException itself.
    """

    run_id:              str
    run_root:            Path   # <base>/monitoring_runs/<run_id>
    monitoring_dir:      Path   # <base>/monitoring_runs/<run_id>/monitoring
    manifest_path:       Path
    drift_summary_path:  Path
    clusters_dir:        Path   # holds <cid>/drift_table.parquet

    def cluster_drift_table_path(self, cluster_id: str) -> Path:
        """``{clusters_dir}/<cluster_id>/drift_table.parquet``."""
        return self.clusters_dir / cluster_id / DRIFT_TABLE_FILENAME


# ──────────────────────────────────────────────────────────────────────
# Module-level loaders
# ──────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


# ──────────────────────────────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────────────────────────────

class MonitoringResultReader:
    """Typed accessor over the monitoring-runs directory.

    Stateless — every method resolves the on-disk layout from
    ``run_id`` and reads the file.  No cache (runs are append-only).

    Attributes
    ----------
    base_dir
        Resolved ``settings.artifacts_dir`` — root under which the
        ``monitoring_runs/<run_id>/monitoring/`` tree lives.
    """

    def __init__(self, base_artifacts_dir: str):
        self.base_dir = Path(base_artifacts_dir)

    # ── Path resolution ─────────────────────────────────────────────

    def paths_for(self, run_id: str) -> MonitoringRunReaderPaths:
        """Resolve the per-run filesystem layout.  Pure path arithmetic."""
        run_root       = self.base_dir / MONITORING_RUNS_DIRNAME / run_id
        monitoring_dir = run_root / MONITORING_SUBDIRNAME
        return MonitoringRunReaderPaths(
            run_id              = run_id,
            run_root            = run_root,
            monitoring_dir      = monitoring_dir,
            manifest_path       = monitoring_dir / MANIFEST_FILENAME,
            drift_summary_path  = monitoring_dir / DRIFT_SUMMARY_FILENAME,
            clusters_dir        = monitoring_dir / "clusters",
        )

    # ── Discovery ───────────────────────────────────────────────────

    def list_run_ids(self) -> List[str]:
        """Return all monitoring run IDs on disk, most recent first.

        The run-id format embeds an ISO-8601 timestamp so descending
        alphabetical order is also descending temporal order for any
        given ensemble version (matches inference's run-id ordering
        contract).

        Returns
        -------
        list of str
            Empty list when ``monitoring_runs/`` doesn't exist yet.
        """
        root = self.base_dir / MONITORING_RUNS_DIRNAME
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
            If the manifest doesn't exist (run still in flight or
            never completed).
        """
        path = self.paths_for(run_id).manifest_path
        if not path.exists():
            raise FileNotFoundError(
                f"Monitoring manifest not found for run '{run_id}': {path}"
            )
        return _load_json(path)

    # ── Drift summary ───────────────────────────────────────────────

    def load_drift_summary(self, run_id: str) -> Dict[str, Any]:
        """Read ``drift_summary.json`` (portfolio aggregate)."""
        path = self.paths_for(run_id).drift_summary_path
        if not path.exists():
            raise FileNotFoundError(
                f"Drift summary not found for run '{run_id}': {path}"
            )
        return _load_json(path)

    # ── Per-cluster drift table ─────────────────────────────────────

    def load_cluster_drift_table(
        self,
        run_id:     str,
        cluster_id: str,
    ) -> pd.DataFrame:
        """Read a single cluster's drift table parquet."""
        path = self.paths_for(run_id).cluster_drift_table_path(cluster_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Drift table not found for run '{run_id}' cluster "
                f"'{cluster_id}': {path}"
            )
        return _load_parquet(path)

    # ── Cluster severity index (cheap aggregate for the UI heatmap) ─

    def load_cluster_severity_index(self, run_id: str) -> List[Dict[str, Any]]:
        """Per-cluster severity rollup derived from the manifest's drift_summary.

        The portfolio drift_summary already aggregates severity at the
        cluster level, so we read it once and project to a flat list
        of ``{cluster_id, severity, max_psi, mean_psi, n_features}``
        rows for the UI heatmap.  Avoids loading every cluster
        parquet just to render the cluster picker.

        Returns
        -------
        list of dict
            One entry per cluster.  Empty list when the summary has
            no per-cluster breakdown (``no_data`` severity at the
            portfolio level).
        """
        summary = self.load_drift_summary(run_id)
        clusters = summary.get("clusters") or []
        # The summary writer normalises clusters to a list of dicts;
        # round-trip defensively here in case the schema evolves.
        return [
            {
                "cluster_id":  str(c.get("cluster_id", "")),
                "severity":    str(c.get("severity", "no_data")),
                "max_psi":     c.get("max_psi"),
                "mean_psi":    c.get("mean_psi"),
                "n_features":  c.get("n_features"),
            }
            for c in clusters
        ]

    # ── Run summary ─────────────────────────────────────────────────

    def run_summary(self, run_id: str) -> Dict[str, Any]:
        """Lightweight summary suitable for the run-history table.

        Reads only the manifest (small JSON), no parquets.  Falls
        back to a well-formed ``status='in_progress'`` summary when
        the manifest doesn't exist yet so the UI can list in-flight
        runs.
        """
        try:
            manifest = self.load_manifest(run_id)
            status   = (
                "promoted"
                if manifest.get("predictions") is not None
                else "complete"
            )
        except FileNotFoundError:
            manifest = {}
            status   = "in_progress"

        drift_summary: Dict[str, Any] = manifest.get("drift_summary") or {}
        predictions: Optional[Dict[str, Any]] = manifest.get("predictions")

        return {
            "run_id":                run_id,
            "ensemble_version":      manifest.get("ensemble_version"),
            "status":                status,
            "created_at":            manifest.get("created_at"),
            "n_scenarios":           manifest.get("n_scenarios"),
            "n_clusters":            manifest.get("n_clusters"),
            "n_clusters_affected":   manifest.get("n_clusters_affected"),
            "n_clusters_unaffected": manifest.get("n_clusters_unaffected"),
            "severity":              drift_summary.get("severity"),
            "mean_psi":              drift_summary.get("mean_psi"),
            "max_psi":               drift_summary.get("max_psi"),
            "has_predictions":       predictions is not None,
        }


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton
# ──────────────────────────────────────────────────────────────────────

_monitoring_reader: Optional[MonitoringResultReader] = None


def set_monitoring_result_reader(reader: MonitoringResultReader) -> None:
    """Inject the reader at app-lifespan startup."""
    global _monitoring_reader
    _monitoring_reader = reader


def get_monitoring_result_reader() -> MonitoringResultReader:
    """FastAPI ``Depends`` — returns the singleton reader."""
    if _monitoring_reader is None:
        raise RuntimeError(
            "MonitoringResultReader not initialised. Server not ready."
        )
    return _monitoring_reader


__all__ = [
    "MonitoringRunReaderPaths",
    "MonitoringResultReader",
    "set_monitoring_result_reader",
    "get_monitoring_result_reader",
]

```


### `src/rade_ml_pt/ensemble/api/models/monitoring.py` (NEW, 354 lines)

```python
"""Pydantic schemas for the ``/monitoring`` router.

Type-safe wire contracts for the five-button monitoring workflow:

    POST /monitoring/load          → LoadResponse
    POST /monitoring/scenarios     ← LoadScenariosRequest → LoadScenariosResponse
    POST /monitoring/validate                             → ValidateResponse
    POST /monitoring/run           ← RunRequest           → RunResponse
    POST /monitoring/promote                              → PromoteResponse
    GET  /monitoring/status                               → StatusResponse
    GET  /monitoring/events?cursor=N                      → EventsResponse
    GET  /monitoring/manifest                             → ManifestResponse
    GET  /monitoring/runs                                 → RunsListResponse
    GET  /monitoring/runs/{run_id}/manifest               → ManifestResponse
    GET  /monitoring/runs/{run_id}/drift_summary          → DriftSummaryResponse
    GET  /monitoring/runs/{run_id}/clusters               → ClusterSeverityResponse
    GET  /monitoring/runs/{run_id}/clusters/{cid}/drift   → ClusterDriftResponse

Most response models mirror the dataclass shapes the monitoring pipeline
already exposes (``MonitoringResult``, ``PromoteResult``) or the JSON
shapes its writers produce (``manifest.json``, ``drift_summary.json``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Activity-log event (re-used from inference vocab)
# ──────────────────────────────────────────────────────────────────────

class MonitoringEventModel(BaseModel):
    """One activity-log event emitted by the monitoring pipeline.

    Wire shape identical to :data:`infer_events.ActivityEntry` — the
    monitoring pipeline emits through the same vocab.  Re-declared
    here so the ``/monitoring/events`` OpenAPI schema is self-
    contained (doesn't bleed inference model imports into the
    monitoring tab consumer).
    """

    id:     str            = Field(..., description="Hex UUID, set at event-construction time.")
    stage:  str            = Field(..., description="One of 'ingest', 'validate', 'inference', 'monitoring'.")
    phase:  str            = Field(..., description="Human-readable phase label.")
    status: str            = Field(..., description="One of 'ok', 'running', 'fail', 'pending'.")
    ts:     str            = Field(..., description="UTC ISO-8601 timestamp.")
    target: Optional[str]  = Field(None, description="Optional secondary label.")
    detail: Optional[str]  = Field(None, description="Optional detail / error text.")


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/load
# ──────────────────────────────────────────────────────────────────────

class LoadResponse(BaseModel):
    """Result of the cold-load stage."""

    run_id:           str  = Field(..., description="Provisional run id (replaced when run() picks the canonical id).")
    ensemble_version: str  = Field(..., description="Resolved ensemble version.")
    n_clusters:       int  = Field(..., description="Number of clusters in the loaded ensemble.")
    status:           str  = Field(..., description="Run lifecycle state (one of 'loaded' / 'failed').")


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/scenarios
# ──────────────────────────────────────────────────────────────────────

class LoadScenariosRequest(BaseModel):
    """Request body for ``POST /monitoring/scenarios``.

    Server-side path; file-upload (multipart) deferred — same contract
    decision as inference.
    """

    new_scenario_dir: str = Field(
        ...,
        description=(
            "Path on the API host to a folder containing the new-scenario "
            "shock CSVs.  Must be readable by the API process."
        ),
    )


class LoadScenariosResponse(BaseModel):
    """Result of parsing the scenario folder.

    Shape mirrors ``LoadedScenariosReport.to_dict()`` from the
    inference pipeline, since the underlying ``load_scenarios()`` is
    the same call the monitoring pipeline composes.
    """

    new_scenario_dir:  str       = Field(..., description="Resolved absolute path.")
    risk_factor_names: List[str] = Field(default_factory=list, description="RF stems parsed from the folder.")
    n_risk_factors:    int       = Field(0,  description="Risk-factor count surfaced by the shocks.")
    n_scenarios:       int       = Field(..., description="Total number of scenarios parsed.")
    scenario_labels:   List[str] = Field(..., description="List of scenario labels.")


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/validate
# ──────────────────────────────────────────────────────────────────────

class ClusterRoutingDecisionModel(BaseModel):
    """One cluster's affected / unaffected decision.

    Re-declared here (rather than re-imported from inference models)
    so the monitoring schema is self-contained and OpenAPI consumers
    don't need to chase cross-router refs.  Field set mirrors
    :meth:`ClusterRoutingDecision.to_dict` exactly so the router can
    ``**decision.to_dict()`` directly into this model.
    """

    cluster_id:                 str       = Field(..., description="Cluster identifier.")
    is_affected:                bool      = Field(..., description="Whether this cluster needs a forward pass.")
    intersecting_risk_factors:  List[str] = Field(default_factory=list)
    n_elementary_trades:        int       = Field(0)
    n_target_trades:            int       = Field(0)
    missing_scenario_labels:    List[str] = Field(default_factory=list)


class ValidateResponse(BaseModel):
    """Result of running the validation report on the loaded scenarios."""

    ensemble_version:       str                                = Field(...)
    n_scenarios:            int                                = Field(...)
    scenario_labels:        List[str]                          = Field(...)
    cluster_decisions:      List[ClusterRoutingDecisionModel]  = Field(...)
    errors:                 List[str]                          = Field(default_factory=list)
    warnings:               List[str]                          = Field(default_factory=list)
    is_valid:               bool                               = Field(...)
    affected_cluster_ids:   List[str]                          = Field(default_factory=list)
    unaffected_cluster_ids: List[str]                          = Field(default_factory=list)
    affected_count:         int                                = Field(0)
    unaffected_count:       int                                = Field(0)


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/run
# ──────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Optional body for ``POST /monitoring/run``.

    All fields default to ``None`` — pass an explicit
    ``new_scenario_dir`` only if you skipped the staged
    ``/scenarios`` call (rare; UI always goes through the stages).
    """

    new_scenario_dir: Optional[str] = Field(
        None,
        description="Fallback scenario dir; usually omitted because /scenarios was already called.",
    )


class RunResponse(BaseModel):
    """Immediate response for ``POST /monitoring/run`` (non-blocking dispatch).

    Terminal metrics (``n_clusters``, ``severity``, ``manifest_path``,
    ...) are unknown at dispatch — they land on the state once the
    worker thread completes.  Clients poll ``GET /monitoring/status``
    + ``GET /monitoring/manifest`` to retrieve them.
    """

    run_id:        str = Field(..., description="Provisional run id at dispatch time.")
    status:        str = Field(..., description="Lifecycle state immediately after dispatch (typically 'running').")
    artifacts_dir: str = Field(..., description="Resolved per-run artifacts directory.")


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/promote
# ──────────────────────────────────────────────────────────────────────

class PromoteResponse(BaseModel):
    """Immediate response for ``POST /monitoring/promote`` (non-blocking).

    Same dispatch-time contract as ``RunResponse``: only the
    lifecycle status is known here; ``predictions_dir`` /
    ``n_clusters_predicted`` populate on the state once the worker
    finishes.
    """

    run_id: str = Field(..., description="Monitoring run id this promote is attached to.")
    status: str = Field(..., description="Lifecycle state after dispatch (typically 'promoting').")


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/status
# ──────────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    """Cheap status probe — drives the UI's next-button gate."""

    has_active_run:    bool          = Field(...)
    run_id:            Optional[str] = Field(None)
    ensemble_version:  Optional[str] = Field(None)
    status:            Optional[str] = Field(None)
    last_error:        Optional[str] = Field(None)
    n_events:          int           = Field(0)
    created_at:        Optional[str] = Field(None)
    artifacts_dir:     Optional[str] = Field(None)
    manifest_path:     Optional[str] = Field(None)
    predictions_dir:   Optional[str] = Field(None)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/events
# ──────────────────────────────────────────────────────────────────────

class EventsResponse(BaseModel):
    """Cursor-paginated activity log slice."""

    events:      List[MonitoringEventModel] = Field(...)
    next_cursor: int                        = Field(..., description="Pass back on the next poll to chain.")


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/manifest
# GET /monitoring/runs/{run_id}/manifest
# ──────────────────────────────────────────────────────────────────────

class ManifestResponse(BaseModel):
    """The full manifest JSON for a monitoring run.

    The inner ``manifest`` payload is left untyped (``Dict[str, Any]``)
    because the manifest schema is owned by the writer
    (:mod:`monitoring.writers.write_monitoring_manifest_json`) and we
    don't want to duplicate that schema in two places.  Consumers
    interested in field-level validation should reference
    :class:`MonitoringResult` / :class:`PromoteResult`.
    """

    run_id:   str            = Field(...)
    manifest: Dict[str, Any] = Field(...)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/runs
# ──────────────────────────────────────────────────────────────────────

class RunSummary(BaseModel):
    """Lightweight summary for the run-history table.

    Reads exclusively from the manifest (small JSON), no parquets —
    safe to compute for hundreds of runs in a single
    ``GET /runs`` call.
    """

    run_id:                str            = Field(...)
    ensemble_version:      Optional[str]  = Field(None)
    status:                str            = Field(..., description="'in_progress' | 'complete' | 'promoted'")
    created_at:            Optional[str]  = Field(None)
    n_scenarios:           Optional[int]  = Field(None)
    n_clusters:            Optional[int]  = Field(None)
    n_clusters_affected:   Optional[int]  = Field(None)
    n_clusters_unaffected: Optional[int]  = Field(None)
    severity:              Optional[str]  = Field(None, description="Portfolio severity rollup.")
    mean_psi:              Optional[float] = Field(None)
    max_psi:               Optional[float] = Field(None)
    has_predictions:       bool           = Field(False, description="True iff promote-to-predictions has run.")


class RunsListResponse(BaseModel):
    """Result of ``GET /monitoring/runs`` — list every run on disk."""

    runs:  List[RunSummary] = Field(...)
    count: int              = Field(...)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/runs/{run_id}/drift_summary
# ──────────────────────────────────────────────────────────────────────

class DriftSummaryResponse(BaseModel):
    """Portfolio-level drift KPIs.

    Inner ``summary`` mirrors
    :func:`monitoring.drift.build_portfolio_drift_summary`'s output —
    same untyped-dict reasoning as :class:`ManifestResponse`.
    """

    run_id:  str            = Field(...)
    summary: Dict[str, Any] = Field(...)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/runs/{run_id}/clusters
# ──────────────────────────────────────────────────────────────────────

class ClusterSeverityRow(BaseModel):
    """One cluster's severity rollup (UI heatmap row)."""

    cluster_id: str             = Field(...)
    severity:   str             = Field(...)
    max_psi:    Optional[float] = Field(None)
    mean_psi:   Optional[float] = Field(None)
    n_features: Optional[int]   = Field(None)


class ClusterSeverityResponse(BaseModel):
    """Aggregate severity per cluster — sourced from drift_summary.json."""

    run_id: str                       = Field(...)
    rows:   List[ClusterSeverityRow]  = Field(...)
    count:  int                       = Field(...)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/runs/{run_id}/clusters/{cluster_id}/drift
# ──────────────────────────────────────────────────────────────────────

class ClusterDriftRow(BaseModel):
    """One per-feature drift-table row."""

    cluster_id:    str             = Field(...)
    feature_name:  str             = Field(...)
    psi:           Optional[float] = Field(None)
    js_divergence: Optional[float] = Field(None)
    mean_shift:    Optional[float] = Field(None)
    std_ratio:     Optional[float] = Field(None)
    severity:      str             = Field(...)


class ClusterDriftResponse(BaseModel):
    """Long-format drift table for a single cluster."""

    run_id:     str                  = Field(...)
    cluster_id: str                  = Field(...)
    rows:       List[ClusterDriftRow] = Field(...)
    n_features: int                  = Field(...)


__all__ = [
    "MonitoringEventModel",
    "LoadResponse",
    "LoadScenariosRequest",
    "LoadScenariosResponse",
    "ClusterRoutingDecisionModel",
    "ValidateResponse",
    "RunRequest",
    "RunResponse",
    "PromoteResponse",
    "StatusResponse",
    "EventsResponse",
    "ManifestResponse",
    "RunSummary",
    "RunsListResponse",
    "DriftSummaryResponse",
    "ClusterSeverityRow",
    "ClusterSeverityResponse",
    "ClusterDriftRow",
    "ClusterDriftResponse",
]

```


### `src/rade_ml_pt/ensemble/api/routers/monitoring.py` (NEW, 695 lines)

```python
"""``/prism/v1/monitoring`` — staged drift-monitoring workflow over HTTP.

Mirrors the inference router 1:1 — see
:mod:`ensemble.api.routers.inference` for the parallel design rationale
(state machine, non-blocking ``/run``, historical-runs data plane).
This router adds one extra control endpoint on top of the four
inference stages: ``POST /promote``, which triggers M.3's
:meth:`EnsembleMonitoringPipeline.promote_to_predictions` against a
completed drift run.

Endpoint map
------------
Control plane (active run)::

    POST /prism/v1/monitoring/load        → cold-load ensemble.
    POST /prism/v1/monitoring/scenarios   → load + parse a scenario folder.
    POST /prism/v1/monitoring/validate    → run validation on loaded scenarios.
    POST /prism/v1/monitoring/run         → compute drift on a worker thread.
    POST /prism/v1/monitoring/promote     → forward-pass inside the same run.
    GET  /prism/v1/monitoring/status      → cheap status probe.
    GET  /prism/v1/monitoring/events      → cursor-paginated activity log tail.
    GET  /prism/v1/monitoring/manifest    → active run's manifest.json.

Data plane (historical runs)::

    GET  /prism/v1/monitoring/runs                                 → list every run on disk.
    GET  /prism/v1/monitoring/runs/{run_id}/manifest               → read any run's manifest.
    GET  /prism/v1/monitoring/runs/{run_id}/drift_summary          → portfolio drift KPIs.
    GET  /prism/v1/monitoring/runs/{run_id}/clusters               → per-cluster severity index.
    GET  /prism/v1/monitoring/runs/{run_id}/clusters/{cid}/drift   → per-cluster drift table.

State model
-----------
Single :class:`MonitoringRunState` on the process-wide
:class:`MonitoringStateManager` (Option A — single user).  Same
migration story to Option B as the inference router.

Data plane (historical runs) goes straight through
:class:`MonitoringResultReader` to disk — stateless, safe for any
number of concurrent UI sessions.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.config import Settings, get_settings
from src.rade_ml_pt.ensemble.api.models.monitoring import (
    ClusterDriftResponse,
    ClusterDriftRow,
    ClusterRoutingDecisionModel,
    ClusterSeverityResponse,
    ClusterSeverityRow,
    DriftSummaryResponse,
    EventsResponse,
    LoadResponse,
    LoadScenariosRequest,
    LoadScenariosResponse,
    ManifestResponse,
    MonitoringEventModel,
    PromoteResponse,
    RunRequest,
    RunResponse,
    RunSummary,
    RunsListResponse,
    StatusResponse,
    ValidateResponse,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_reader import (
    MonitoringResultReader,
    get_monitoring_result_reader,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_state import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_LOADED,
    STATUS_LOADING,
    STATUS_PROMOTED,
    STATUS_PROMOTING,
    STATUS_RUNNING,
    STATUS_SCENARIOS,
    STATUS_VALIDATED,
    MonitoringStateManager,
    build_monitoring_ensemble_config,
    get_monitoring_state_manager,
    per_run_monitoring_artifacts_dir,
)
from src.rade_ml_pt.monitoring.run_paths import (
    MANIFEST_FILENAME,
    MONITORING_SUBDIRNAME,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prism/v1/monitoring", tags=["monitoring"])


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _opt_float(x) -> float | None:
    """Coerce a parquet cell to ``Optional[float]`` for the wire schema.

    Parquet's NA marker varies (``pd.NA``, ``np.nan``, plain Python
    ``None``); the Pydantic models declare these fields as
    ``Optional[float]`` so the router normalises all NA flavours to
    a real ``None`` here.  This keeps the wire shape JSON-strict
    (no NaN-encoded floats slipping through) regardless of how the
    writer side normalised the parquet.
    """
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _opt_int(x) -> int | None:
    """Coerce a parquet cell to ``Optional[int]``."""
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════
# Control plane (active run)
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# POST /load
# ──────────────────────────────────────────────────────────────────────

@router.post("/load", response_model=LoadResponse)
def load_ensemble(
    settings: Settings                  = Depends(get_settings),
    manager:  MonitoringStateManager    = Depends(get_monitoring_state_manager),
) -> LoadResponse:
    """Cold-load the ensemble model + per-cluster contexts.

    Wraps :meth:`EnsembleMonitoringPipeline.load` (which forwards to
    :meth:`EnsembleInferencePipeline.load` under the hood).  Any prior
    active monitoring run is silently replaced — appropriate for the
    single-user dashboard today.
    """
    ensemble_config = build_monitoring_ensemble_config(
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
        logger.exception("Monitoring ensemble load failed for run %s", state.run_id)
        raise HTTPException(status_code=500, detail=f"Ensemble load failed: {exc}")

    # Reach into the composed inference pipeline to count clusters —
    # mirrors the inference router's ``len(state.pipeline._ensemble.members)``
    # lookup.  We can't call a public accessor because none exists on
    # the inference pipeline either (would couple us to infer.py).
    inference_pipeline = state.pipeline._inference_pipeline  # noqa: SLF001
    n_clusters = (
        len(inference_pipeline._ensemble.members)            # noqa: SLF001
        if inference_pipeline._ensemble is not None else 0   # noqa: SLF001
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
    body:    LoadScenariosRequest,
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> LoadScenariosResponse:
    """Parse a folder of shock CSVs into the monitoring pipeline state.

    Wraps :meth:`EnsembleMonitoringPipeline.load_scenarios`.  Same
    server-side-path contract inference uses; file-upload deferred.
    """
    state = manager.get_run()
    if state.status not in (
        STATUS_LOADED, STATUS_SCENARIOS, STATUS_VALIDATED,
        STATUS_COMPLETE, STATUS_PROMOTED, STATUS_FAILED,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot load scenarios from status '{state.status}'. Call /load first.",
        )

    try:
        report = state.pipeline.load_scenarios(new_scenario_dir=body.new_scenario_dir)
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Monitoring scenario load failed for run %s", state.run_id)
        raise HTTPException(status_code=400, detail=f"Scenario load failed: {exc}")

    state.transition(STATUS_SCENARIOS)
    return LoadScenariosResponse(**report.to_dict())


# ──────────────────────────────────────────────────────────────────────
# POST /validate
# ──────────────────────────────────────────────────────────────────────

@router.post("/validate", response_model=ValidateResponse)
def validate_scenarios(
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> ValidateResponse:
    """Compute per-cluster routing decisions + run-level validation errors.

    Wraps :meth:`EnsembleMonitoringPipeline.validate_scenarios`.
    Does NOT raise on user-input errors (those land in ``report.errors``);
    only system failures bubble up as 500s.  A non-empty ``errors``
    list keeps the run in ``scenarios_loaded`` so the user can fix
    and re-validate.
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
        logger.exception("Monitoring validation failed for run %s", state.run_id)
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")

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
    )


# ──────────────────────────────────────────────────────────────────────
# POST /run
# ──────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=RunResponse)
def run_monitoring(
    body:     RunRequest                = RunRequest(),  # noqa: B008 — FastAPI default
    settings: Settings                  = Depends(get_settings),
    manager:  MonitoringStateManager    = Depends(get_monitoring_state_manager),
) -> RunResponse:
    """Dispatch the drift-compute stage onto a background thread.

    Wraps :meth:`EnsembleMonitoringPipeline.compute_drift`.  Non-
    blocking — returns immediately with ``status='running'``; the UI
    polls ``GET /status`` until terminal.  Mirror of
    ``POST /inference/run`` in shape.

    Gated on:
      * ``status == 'validated'`` — validation must have produced an
        error-free report.
      * ``not state.is_alive`` — no prior background thread still
        executing.

    Side effects
    ------------
    Pins ``config.artifacts_dir`` on the composed inference pipeline
    so the monitoring pipeline writes its run under
    ``<artifacts_dir>/monitoring_runs/<run_id>/``.  The router doesn't
    know the final run_id at dispatch (it's minted by
    :func:`monitoring_run_id` inside ``compute_drift``); the on-disk
    convention puts every run under the same parent, so the path
    returned here is the parent ``monitoring_runs/<provisional_id>``
    directory.  The state's ``manifest_path`` is populated by the
    worker after compute_drift completes.
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

    # Pin the per-run artifacts root.  The monitoring pipeline mints
    # its own run_id inside compute_drift; we point ``artifacts_dir``
    # at the BASE (where ``monitoring_runs/<run_id>/`` will land) and
    # surface the resolved final dir on the worker-callback below
    # once the pipeline returns its MonitoringResult.
    base_artifacts_dir = settings.artifacts_dir
    state.pipeline.config.artifacts_dir = base_artifacts_dir
    state.artifacts_dir                  = base_artifacts_dir
    state.transition(STATUS_RUNNING)

    def _execute() -> None:
        try:
            result = state.pipeline.compute_drift()
        except Exception as exc:
            state.transition(STATUS_FAILED, error=str(exc))
            logger.exception(
                "Monitoring run failed for run %s", state.run_id,
            )
            return

        # MonitoringResult exposes the canonical run_id (the pipeline
        # mints this internally via :func:`monitoring_run_id`).
        # Persist it on the state so subsequent /promote and /status
        # calls refer to the same id the manifest carries.
        state.run_id        = result.run_id
        state.artifacts_dir = str(result.artifacts_dir)
        state.manifest_path = (
            str(result.manifest_path) if result.manifest_path.exists() else None
        )
        state.transition(STATUS_COMPLETE)

    manager.start_run_in_background(_execute)

    return RunResponse(
        run_id        = state.run_id,
        status        = state.status,                              # "running"
        artifacts_dir = per_run_monitoring_artifacts_dir(
            base_artifacts_dir, state.run_id,
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# POST /promote
# ──────────────────────────────────────────────────────────────────────

@router.post("/promote", response_model=PromoteResponse)
def promote_to_predictions(
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> PromoteResponse:
    """Dispatch promote-to-predictions onto a background thread.

    Wraps :meth:`EnsembleMonitoringPipeline.promote_to_predictions`.
    Non-blocking — same dispatch model as ``/run``.  The UI polls
    ``GET /status`` until terminal (``status=='promoted'``) and then
    reads the updated manifest via ``GET /manifest``.

    Gated on:
      * ``status == 'complete'`` — the drift run must have finished.
      * ``not state.is_alive`` — no prior background thread still
        executing.

    The pipeline itself enforces single-promote-per-instance
    (raises if a prior promote succeeded); we surface that as a 409
    when the second dispatch reaches the worker.
    """
    state = manager.get_run()
    if state.status != STATUS_COMPLETE:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot promote from status '{state.status}'. "
                "Drift run must complete first (status='complete')."
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

    state.transition(STATUS_PROMOTING)

    def _execute() -> None:
        try:
            promote_result = state.pipeline.promote_to_predictions()
        except Exception as exc:
            state.transition(STATUS_FAILED, error=str(exc))
            logger.exception(
                "Monitoring promote failed for run %s", state.run_id,
            )
            return

        state.predictions_dir = str(promote_result.predictions_dir)
        # promote_to_predictions rewrites the same monitoring manifest,
        # so manifest_path stays the same — but defensively refresh it
        # in case the pipeline ever changes that contract.
        state.manifest_path = (
            str(promote_result.manifest_path)
            if promote_result.manifest_path.exists() else state.manifest_path
        )
        state.transition(STATUS_PROMOTED)

    manager.start_run_in_background(_execute)

    return PromoteResponse(
        run_id = state.run_id,
        status = state.status,                                 # "promoting"
    )


# ──────────────────────────────────────────────────────────────────────
# GET /status
# ──────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
def get_status(
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> StatusResponse:
    """Cheap status probe — drives the next-button gate in the UI."""
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
        predictions_dir  = state.predictions_dir,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /events
# ──────────────────────────────────────────────────────────────────────

@router.get("/events", response_model=EventsResponse)
def get_events(
    cursor:  int                       = Query(0, ge=0, description="Number of events the caller has already seen."),
    manager: MonitoringStateManager    = Depends(get_monitoring_state_manager),
) -> EventsResponse:
    """Cursor-paginated tail of the activity log.

    Same cursor protocol as ``GET /inference/events``.  Falls back to
    an empty response when no run has been created so the UI can mount
    the polling loop before /load.
    """
    if not manager.has_active_run:
        return EventsResponse(events=[], next_cursor=0)

    events, next_cursor = manager.events_since(run_id=None, cursor=cursor)
    return EventsResponse(
        events      = [MonitoringEventModel(**e) for e in events],
        next_cursor = next_cursor,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /manifest
# ──────────────────────────────────────────────────────────────────────

@router.get("/manifest", response_model=ManifestResponse)
def get_manifest(
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> ManifestResponse:
    """Return the active run's monitoring ``manifest.json``.

    Only available after the drift run completes (``status='complete'``
    or ``'promoted'``).  The manifest is the dashboard's entry point —
    lists per-cluster paths + the portfolio drift_summary +
    (if promoted) the predictions block.
    """
    state = manager.get_run()
    if state.status not in (STATUS_COMPLETE, STATUS_PROMOTED) or state.manifest_path is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Manifest not available — run status is '{state.status}'.  "
                "Wait for status='complete' (drift) or 'promoted' (with predictions)."
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
# Data plane (historical runs)
# ══════════════════════════════════════════════════════════════════════
#
# All endpoints below are stateless: they go straight through
# :class:`MonitoringResultReader` to disk and don't touch
# :class:`MonitoringStateManager` at all.
#
# Error model
# -----------
# * ``FileNotFoundError`` from the reader → 404.  Covers the
#   common case of polling for results before the worker thread
#   finished writing them.
# * Any other reader exception → 500 (data corruption).
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# GET /runs — discovery
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs", response_model=RunsListResponse)
def list_runs(
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> RunsListResponse:
    """Return every monitoring run on disk, most recent first."""
    run_ids = reader.list_run_ids()
    runs    = [RunSummary(**reader.run_summary(rid)) for rid in run_ids]
    return RunsListResponse(runs=runs, count=len(runs))


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/manifest
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/manifest", response_model=ManifestResponse)
def get_run_manifest(
    run_id: str,
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> ManifestResponse:
    """Read ``manifest.json`` for any historical monitoring run."""
    try:
        manifest = reader.load_manifest(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ManifestResponse(run_id=run_id, manifest=manifest)


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/drift_summary
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/drift_summary", response_model=DriftSummaryResponse,
)
def get_run_drift_summary(
    run_id: str,
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> DriftSummaryResponse:
    """Return the portfolio-level drift KPIs for ``run_id``.

    Sourced from ``drift_summary.json``.  Cheap (small JSON), so
    safe to poll from the UI's main dashboard view.
    """
    try:
        summary = reader.load_drift_summary(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return DriftSummaryResponse(run_id=run_id, summary=summary)


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/clusters
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/clusters", response_model=ClusterSeverityResponse,
)
def get_run_clusters(
    run_id: str,
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> ClusterSeverityResponse:
    """Return the per-cluster severity index for ``run_id``.

    Derived from the manifest's portfolio drift_summary — cheap
    enough to call on every UI navigation event without paying for
    a per-cluster parquet read.
    """
    try:
        rows_raw = reader.load_cluster_severity_index(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    rows = [
        ClusterSeverityRow(
            cluster_id = str(r.get("cluster_id", "")),
            severity   = str(r.get("severity", "no_data")),
            max_psi    = _opt_float(r.get("max_psi")),
            mean_psi   = _opt_float(r.get("mean_psi")),
            n_features = _opt_int(r.get("n_features")),
        )
        for r in rows_raw
    ]
    return ClusterSeverityResponse(run_id=run_id, rows=rows, count=len(rows))


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/clusters/{cluster_id}/drift
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/clusters/{cluster_id}/drift",
    response_model=ClusterDriftResponse,
)
def get_run_cluster_drift(
    run_id:     str,
    cluster_id: str,
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> ClusterDriftResponse:
    """Return the long-format drift table for one cluster.

    Sourced from
    ``<run>/monitoring/clusters/<cid>/drift_table.parquet``.  Output
    of :func:`monitoring.drift.build_drift_table`.
    """
    try:
        df = reader.load_cluster_drift_table(run_id, cluster_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    rows = [
        ClusterDriftRow(
            cluster_id    = str(r.get("cluster_id", cluster_id)),
            feature_name  = str(r.get("feature_name", "")),
            psi           = _opt_float(r.get("psi")),
            js_divergence = _opt_float(r.get("js_divergence")),
            mean_shift    = _opt_float(r.get("mean_shift")),
            std_ratio     = _opt_float(r.get("std_ratio")),
            severity      = str(r.get("severity", "no_data")),
        )
        for r in df.to_dict(orient="records")
    ]
    return ClusterDriftResponse(
        run_id     = run_id,
        cluster_id = cluster_id,
        rows       = rows,
        n_features = len(rows),
    )


# Re-export the constants the active-run /manifest path uses so the
# router and the on-disk writer can never drift on the manifest
# filename / sub-directory name.  Lint-friendly: explicit reference
# so unused-import linting doesn't strip them.
_ = (MANIFEST_FILENAME, MONITORING_SUBDIRNAME)

```
