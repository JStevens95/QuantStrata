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

## Appendix A — API integration: `/inference` router (Stages 7 + 8)

> **Status**: Stage 7 (run-state singleton + dependency) and Stage 8 (REST routers
> + pydantic models) complete in this env. This appendix contains the full source
> of every new file and the exact, line-scoped edits to every modified file so the
> changes can be copied across to a fresh work env without missing any glue.
>
> The previous Appendix A (full `ensemble/infer.py` source) has been superseded
> by this content; the inference pipeline is now landed in the repo and tracked
> separately in
> [`docs/rade_analytics/ensemble_infer_refactor.md`](../rade_analytics/ensemble_infer_refactor.md).
>
> Stages 1–6 (the inference pipeline itself) are upstream prerequisites: they
> produce the `manifest.json` and summary parquets that the endpoints in this
> appendix expose over HTTP.

---

### Appendix A · 0 — Reading guide

**Files covered in this appendix**:

| # | File | Action | Section |
|---|---|---|---|
| 1 | `src/rade_ml_pt/ensemble/api/services/inference_state.py` | **NEW** | A·1 |
| 2 | `src/rade_ml_pt/ensemble/api/models/inference.py`         | **NEW** | A·2 |
| 3 | `src/rade_ml_pt/ensemble/api/routers/inference.py`        | **NEW** | A·3 |
| 4 | `src/rade_ml_pt/ensemble/api/dependencies.py`             | MODIFIED | A·4 |
| 5 | `src/rade_ml_pt/ensemble/api/app.py`                      | MODIFIED | A·5 |

**Implementation order**: 1 → 2 → 3 → 4 → 5. The router (3) depends on the state
manager (1) and pydantic models (2); the dependency re-exports (4) wire the
router to its provider; the app changes (5) register the router and initialise
the manager at startup.

**Endpoint map** (the entire `/prism/v1/inference` surface):

```
POST /prism/v1/inference/load          → ensemble cold-load.
POST /prism/v1/inference/scenarios     ← { "new_scenario_dir": "..." }
POST /prism/v1/inference/validate      → per-cluster routing decisions + errors.
POST /prism/v1/inference/run           ← optional { artifacts_dir?, batch_size? } — synchronous in Stage 8.
GET  /prism/v1/inference/status        → cheap probe (button-gate).
GET  /prism/v1/inference/events?cursor=N → cursor-paginated activity-log tail.
GET  /prism/v1/inference/manifest      → run-level manifest.json after completion.
```

**State model**: Option A (singleton). All state-manager accessors take an
explicit `run_id` argument so migrating to Option B (per-run dict, multi-user)
is a localised refactor — see the `InferenceStateManager` docstring in A·1.

**Smoke-test sequence** (after wiring up):

```bash
# 1. Load ensemble
curl -X POST http://localhost:8000/prism/v1/inference/load
# 2. Status
curl http://localhost:8000/prism/v1/inference/status
# 3. Scenarios (server-side path)
curl -X POST http://localhost:8000/prism/v1/inference/scenarios \
     -H "Content-Type: application/json" \
     -d '{"new_scenario_dir":"/abs/path/to/scenarios"}'
# 4. Validate
curl -X POST http://localhost:8000/prism/v1/inference/validate
# 5. Run (synchronous — blocks until done)
curl -X POST http://localhost:8000/prism/v1/inference/run \
     -H "Content-Type: application/json" \
     -d '{}'
# 6. Manifest
curl http://localhost:8000/prism/v1/inference/manifest
# Activity log (poll)
curl "http://localhost:8000/prism/v1/inference/events?cursor=0"
```

Or open `http://localhost:8000/docs` and exercise the endpoints interactively.

---

### Appendix A · 1 — NEW: `src/rade_ml_pt/ensemble/api/services/inference_state.py`

**Purpose**: hold the live `EnsembleInferencePipeline` between HTTP requests so
the staged workflow survives the stateless boundary between calls. Defines:

* Lifecycle constants (`STATUS_CREATED` → `STATUS_COMPLETE` / `STATUS_FAILED`).
* `InferenceRunState` dataclass — per-run mutable state with a single `transition` mutator.
* `InferenceStateManager` — Option-A singleton holder; `run_id`-aware for Option-B migration.
* `build_ensemble_config()` — minimal `EnsembleConfig` for API-driven runs.
* `per_run_artifacts_dir()` — disk-layout convention `<base>/inference_runs/<run_id>`.
* Module-level `get_inference_state_manager()` / `set_inference_state_manager()` for FastAPI `Depends`.

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
from typing import List, Optional, Tuple

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

### Appendix A · 2 — NEW: `src/rade_ml_pt/ensemble/api/models/inference.py`

**Purpose**: typed pydantic schemas for every request/response on the
`/prism/v1/inference` surface. Most response models intentionally mirror the
dataclass `to_dict()` shapes the pipeline already exposes (e.g.
`LoadedScenariosReport.to_dict()`), so the typed client and OpenAPI docs cover
the contract without duplicating state.

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
    """Outcome of a synchronous ``POST /inference/run``.

    Stage 8 implementation is synchronous — the HTTP call blocks
    until ``run_inference`` returns.  Stage 9 will switch this to a
    background task and the same response model will be returned
    immediately with ``status='running'``.
    """

    run_id:        str           = Field(..., description="Run identifier.")
    status:        str           = Field(..., description="'complete' on success, 'failed' on error, 'running' under Stage 9 async.")
    artifacts_dir: str           = Field(..., description="Resolved per-run artifacts root.")
    n_scenarios:   Optional[int] = Field(None, description="Number of scenarios scored.")
    n_clusters:    Optional[int] = Field(None, description="Number of clusters scored.")
    n_predictions: Optional[int] = Field(None, description="Total prediction cells in the combined output.")
    manifest_path: Optional[str] = Field(None, description="Absolute path to the per-run manifest.json (set on success).")
    error:         Optional[str] = Field(None, description="Detail message when status='failed'.")


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
```

---

### Appendix A · 3 — NEW: `src/rade_ml_pt/ensemble/api/routers/inference.py`

**Purpose**: the seven `/prism/v1/inference/*` endpoints. Each POST endpoint is
a thin wrapper around the corresponding pipeline stage, gated on the run's
`status` so the state machine can't be violated (returns `409 Conflict` if you
try to validate before loading scenarios, run before validating, etc.).

```python
"""``/prism/v1/inference`` — staged inference workflow over HTTP.

Exposes the four stages of :class:`EnsembleInferencePipeline` as REST
endpoints, plus two polling endpoints for the UI activity log and
status probe, plus a manifest fetcher for completed runs.

Endpoint map
------------
::

    POST /prism/v1/inference/load          → load the ensemble (cold-load).
    POST /prism/v1/inference/scenarios     → load + parse a scenario folder.
    POST /prism/v1/inference/validate      → run validation on loaded scenarios.
    POST /prism/v1/inference/run           → run inference (synchronous in Stage 8;
                                              async in Stage 9).
    GET  /prism/v1/inference/status        → cheap status probe (gate the next button).
    GET  /prism/v1/inference/events        → cursor-based activity log poll.
    GET  /prism/v1/inference/manifest      → read the per-run manifest.json after a run completes.

State model
-----------
Today (Option A) a single :class:`InferenceRunState` is held on the
process-wide :class:`InferenceStateManager`.  Each ``POST /load`` call
replaces any prior run.  See
:mod:`services.inference_state` for the migration path to multi-user.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.config import Settings, get_settings
from src.rade_ml_pt.ensemble.api.dependencies import get_inference_state_manager
from src.rade_ml_pt.ensemble.api.models.inference import (
    ClusterRoutingDecisionModel,
    EventModel,
    EventsResponse,
    LoadResponse,
    LoadScenariosRequest,
    LoadScenariosResponse,
    ManifestResponse,
    RunRequest,
    RunResponse,
    StatusResponse,
    ValidateResponse,
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
    """Run inference end-to-end on the active run.

    Wraps :meth:`EnsembleInferencePipeline.run_inference`.  Today this
    is **synchronous** — the HTTP call blocks until the run completes.
    Stage 9 will swap to a background-task dispatch; the response
    schema is already shaped for that future (``status='running'``
    returned immediately).

    Gated on ``status == 'validated'`` — i.e. ``validate_scenarios``
    must have produced an error-free report.

    Side effects
    ------------
    Sets ``config.artifacts_dir`` on the pipeline to a per-run path
    so ``post_infer`` writes its manifest + parquets into a stable
    location the API can serve.  The resolved path is returned for
    the UI to deep-link to.
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

    # Resolve and pin the per-run artifacts directory before invoking
    # the pipeline so post_infer writes to a stable, predictable place.
    artifacts_dir = body.artifacts_dir or per_run_artifacts_dir(
        settings.artifacts_dir, state.run_id,
    )
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    state.pipeline.config.artifacts_dir = artifacts_dir
    state.artifacts_dir                  = artifacts_dir
    state.transition(STATUS_RUNNING)

    try:
        result = state.pipeline.run_inference()
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Inference run failed for run %s", state.run_id)
        return RunResponse(
            run_id        = state.run_id,
            status        = state.status,
            artifacts_dir = artifacts_dir,
            error         = str(exc),
        )

    # post_infer writes <artifacts_dir>/inference/manifest.json — pin
    # the path on the state so /manifest can read it back without
    # re-deriving the layout.
    manifest_path = Path(artifacts_dir) / "inference" / "manifest.json"
    state.manifest_path = str(manifest_path) if manifest_path.exists() else None
    state.transition(STATUS_COMPLETE)

    preds = result.predictions
    return RunResponse(
        run_id        = state.run_id,
        status        = state.status,
        artifacts_dir = artifacts_dir,
        n_scenarios   = int(preds.shape[0]) if preds is not None and preds.ndim >= 1 else None,
        n_clusters    = (len(state.pipeline._ensemble.members)             # noqa: SLF001
                         if state.pipeline._ensemble is not None else None),
        n_predictions = int(preds.size) if preds is not None else None,
        manifest_path = state.manifest_path,
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
```

---

### Appendix A · 4 — MODIFIED: `src/rade_ml_pt/ensemble/api/dependencies.py`

**What changed**: re-export the new inference-state hooks so routers can import
them from one canonical module (consistent with how `get_reader` is exposed
today). The existing `ArtifactReader` provider is unchanged.

**Two edits to apply** to the existing file:

#### Edit 4.1 — Rewrite the module docstring

```python
# BEFORE
"""FastAPI dependency injection.

A single :class:`ArtifactReader` is constructed during the app's lifespan
startup and registered here.  All routers depend on
:func:`get_reader` to access it.
"""

# AFTER
"""FastAPI dependency injection.

Process-wide singletons constructed during the app's lifespan startup
and exposed here as ``Depends`` providers.  Two collaborators live here:

* :class:`ArtifactReader` — serves on-disk **evaluation** artifacts
  (parquet + JSON) to the existing PRISM routers.
* :class:`InferenceStateManager` — holds the live
  :class:`EnsembleInferencePipeline` for the staged inference workflow
  exposed by the ``/inference`` router (Stage 8).

Both injection points are re-exported via the inference state module
itself; this file is the canonical FastAPI surface.
"""
```

#### Edit 4.2 — Add the inference-state import + `__all__` re-export

Add this `import` block above the existing `from … reader import ArtifactReader`
line, and append the `__all__` block at the bottom of the file:

```python
# Add directly after the module docstring + `from __future__ import annotations`:
from src.rade_ml_pt.ensemble.api.services.inference_state import (
    InferenceStateManager,
    get_inference_state_manager,
    set_inference_state_manager,
)

# Append at the end of the file (after get_reader):
# Re-export the inference-state manager hooks so routers can import them
# from this single dependencies module (consistent with get_reader).
__all__ = [
    "InferenceStateManager",
    "get_inference_state_manager",
    "get_reader",
    "set_inference_state_manager",
    "set_reader",
]
```

#### Final file (full source after both edits)

```python
"""FastAPI dependency injection.

Process-wide singletons constructed during the app's lifespan startup
and exposed here as ``Depends`` providers.  Two collaborators live here:

* :class:`ArtifactReader` — serves on-disk **evaluation** artifacts
  (parquet + JSON) to the existing PRISM routers.
* :class:`InferenceStateManager` — holds the live
  :class:`EnsembleInferencePipeline` for the staged inference workflow
  exposed by the ``/inference`` router (Stage 8).

Both injection points are re-exported via the inference state module
itself; this file is the canonical FastAPI surface.
"""
from __future__ import annotations

from src.rade_ml_pt.ensemble.api.services.inference_state import (
    InferenceStateManager,
    get_inference_state_manager,
    set_inference_state_manager,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

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


# Re-export the inference-state manager hooks so routers can import them
# from this single dependencies module (consistent with get_reader).
__all__ = [
    "InferenceStateManager",
    "get_inference_state_manager",
    "get_reader",
    "set_inference_state_manager",
    "set_reader",
]
```

---

### Appendix A · 5 — MODIFIED: `src/rade_ml_pt/ensemble/api/app.py`

**What changed**: register the new `inference_router` and initialise the
`InferenceStateManager` at lifespan startup so the router can resolve its
`Depends`. The reader/router conventions are preserved.

**Four discrete edits** to apply to the existing file:

#### Edit 5.1 — Import `set_inference_state_manager` from dependencies

```python
# BEFORE
from src.rade_ml_pt.ensemble.api.dependencies import set_reader

# AFTER
from src.rade_ml_pt.ensemble.api.dependencies import (
    set_inference_state_manager,
    set_reader,
)
```

#### Edit 5.2 — Add the inference router import

Insert this block directly under the `governance` router import (the existing
block keeps imports alphabetical, so this slots in between `governance` and
`graph_stats`):

```python
from src.rade_ml_pt.ensemble.api.routers.inference import (
    router as inference_router,
)
```

#### Edit 5.3 — Add the `InferenceStateManager` import

Insert this block above the existing `ArtifactPaths` / `ArtifactReader` /
`list_versions` services imports:

```python
from src.rade_ml_pt.ensemble.api.services.inference_state import (
    InferenceStateManager,
)
```

#### Edit 5.4 — Initialise the manager in `lifespan`

Add the highlighted block immediately after `set_reader(reader)` in
`lifespan` (existing code shown for context):

```python
    reader = ArtifactReader(paths)
    set_reader(reader)

    # Inference state manager — holds the live EnsembleInferencePipeline
    # for the /inference router.  Empty until POST /inference/load.
    set_inference_state_manager(InferenceStateManager())

    splits = reader.available_splits()
```

#### Edit 5.5 — Register the inference router in `create_app`

Append the new include after the existing `governance_router` include (other
routers in this block are listed in registration order, so it goes last):

```python
    app.include_router(predictions_router)
    app.include_router(governance_router)
    app.include_router(inference_router)        # NEW

    return app
```

#### Verification (one command after applying all edits)

```bash
.venv/bin/python -c "
from src.rade_ml_pt.ensemble.api.routers.inference import router
for route in router.routes:
    print(f'{list(route.methods)[0]:6s} {route.path}')
"
# Expected output:
# POST   /prism/v1/inference/load
# POST   /prism/v1/inference/scenarios
# POST   /prism/v1/inference/validate
# POST   /prism/v1/inference/run
# GET    /prism/v1/inference/status
# GET    /prism/v1/inference/events
# GET    /prism/v1/inference/manifest
```

---

### Appendix A · 6 — What's next (Stage 9 preview, not in this appendix)

Three follow-ups remain to fully wire the dashboard:

1. **Threaded `/run`** — wrap `state.pipeline.run_inference()` in a
   `threading.Thread`, return immediately with `status='running'`.
   `EventCollector` is already thread-safe so no extra glue. Response schema
   already supports this transition.
2. **Multipart file upload for `/scenarios`** — parallel `POST /scenarios/upload`
   that accepts a zip of CSVs, unpacks to a temp dir, then calls the existing
   `load_scenarios`. Path-based v1 stays as the default.
3. **Result-reading endpoints** — `GET /manifest/{run_id}/portfolio` and
   `/clusters/{cid}` that read directly off the parquet files the pipeline
   wrote; mirror the existing `ArtifactReader` pattern in this repo.
