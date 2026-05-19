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

## Appendix A — Inference Console UI wiring (Stages 12 + 14, copy-paste sync)

> **Status**: All four files below land in this repo in the
> `src/ui/apps/rade_analytics/` tree. Verified compiling and lint-clean. The
> Stage 11 `RadeApiClient` extensions + Stage 12 `RadeBackend` wrappers
> together with the eight callbacks in A.3 turn the Inference Console page
> into a live, threaded inference driver against the API built in Stages
> 7-10.
>
> The previous **Stages 1-10 appendix** (the pipeline + API source) has
> been retired now that the work env is aligned on that base. The canonical
> reference for Stages 1-10 lives in
> `docs/inference_pipeline_contract.md`. This new appendix covers only
> the UI wiring built on top.

### Reading guide

Four files, in dependency order. Three are diffs (existing files, narrow
edits) and one is a NEW file with its full source.

| § | File | Change | Stage |
|---|---|---|---|
| A.1 | `src/ui/apps/rade_analytics/layouts/inference.py` | +2 IDs, +2 Stores in `_page_stores()` | 14 |
| A.2 | `src/ui/apps/rade_analytics/data/backend.py` | +13 imports, +7 cache bindings, +7 fetchers, +14 public methods | 12 |
| A.3 | `src/ui/apps/rade_analytics/callbacks/inference_cb.py` | **NEW file** — 8 callbacks | 12 |
| A.4 | `src/ui/apps/rade_analytics/callbacks/__init__.py` | +1 import, +1 `register(...)` call | 12 |

**Suggested copy order**: A.1 → A.2 → A.3 → A.4. Each step compiles
independently; you can boot the app after any of them and verify the prior
ones haven't regressed.

**Architecture recap (Stage 12 control flow)**:

```
                                URL hits /inference
                                       │
                                       ▼
              on_mount  ───────────►  POST /load    (or adopt existing run)
                                       │
                                       ▼
              on_upload  ──────────►  POST /scenarios
                                       │
                                       ▼
              on_validate  ────────►  POST /validate    (gates Run button)
                                       │
                                       ▼
              on_run  ─────────────►  POST /run    (returns 'running' immediately)
                                       │ arms polling
                                       ▼
              on_poll  ─ 1Hz tick ─►  GET /events?cursor=N  +  GET /status
                                       │ until terminal
                                       ▼
              hydrate_results  ────►  GET /runs/{id}/portfolio + /clusters + /manifest
                                       │
                                       ▼
                       KPIs + figures + AG Grid populated
```

The polling pair (`polling_store` + `poll_interval`, added in §A.1) is the
only piece of bespoke client state; every other store flows from server
state on demand.

---

### A.1 `src/ui/apps/rade_analytics/layouts/inference.py` — Stage 14 fold-in

Two surgical edits to an existing file. **Do not replace the file** — add
the two snippets below into the locations indicated.

**Edit 1** — Add two keys to `INFERENCE_IDS` (inside the "Row 0 — page-level
data Stores" block):

```python
    # Row 0 — page-level data Stores driving the state machine.
    "activity_log_store":         "inference-activity-log-store",
    "ingest_meta_store":          "inference-ingest-meta-store",
    "run_meta_store":             "inference-run-meta-store",
    "selected_scenario_store":    "inference-selected-scenario-store",

    # Row 0 — polling primitives (Stage 12 / 14).  ``polling_store``
    # carries the live run cursor + run_id + armed flag so the poll
    # callback can advance ``/events`` page-by-page without flicker;
    # ``poll_interval`` is disabled at mount and flipped on by the
    # ``on_run`` callback the moment a run is dispatched.
    "polling_store":              "inference-polling-store",
    "poll_interval":              "inference-poll-interval",
```

**Edit 2** — Replace the body of `_page_stores()` with the version below
(adds a `dcc.Store` and a `dcc.Interval` plus an explanatory docstring;
preserves all five existing stores in their existing order):

```python
def _page_stores() -> List[Any]:
    """All ``dcc.Store`` mounts driving the page state machine.

    Kept as a list so ``build_inference`` can splat them at the top of
    the page tree without tracking individual ids.

    The ``polling_store`` + ``poll_interval`` pair powers the live
    activity-log drain documented in :func:`callbacks.inference_cb.on_poll`:
    ``on_run`` flips ``polling_store.data['armed']`` to ``True`` and
    enables ``poll_interval``; each interval tick reads the current
    cursor, calls ``/events?cursor=N`` + ``/status``, advances the
    cursor, and disarms the polling pair when the API reports a
    terminal status.  Decoupling the cursor from the interval lets a
    completed run pause polling without losing its place if the user
    re-arms (e.g. for re-runs).
    """
    return [
        dcc.Store(
            id=INFERENCE_IDS["mount_signal"],
            data=True,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["activity_log_store"],
            data=[],
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["ingest_meta_store"],
            data=None,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["run_meta_store"],
            data=None,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["selected_scenario_store"],
            data=None,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["polling_store"],
            # Initial shape mirrors the on-the-wire contract the
            # poll callback expects.  ``armed=False`` means the
            # interval will short-circuit even if its disabled flag
            # ever drifts out of sync (defence in depth).
            data={"armed": False, "run_id": None, "cursor": 0},
            storage_type="memory",
        ),
        dcc.Interval(
            id=INFERENCE_IDS["poll_interval"],
            # 1 Hz is a comfortable trade-off between perceived
            # liveness (one new line per second when the worker is
            # busy) and server load (the events endpoint is cheap;
            # /status is a single dict read).  The interval stays
            # disabled until ``on_run`` arms it — pages that never
            # dispatch a run pay zero polling cost.
            interval=1000,
            n_intervals=0,
            disabled=True,
        ),
    ]
```

**Polling store contract**:

| Key | Type | Meaning |
|---|---|---|
| `armed` | `bool` | `True` between `on_run` dispatch and terminal-state detection |
| `run_id` | `str \| None` | Active run_id (kept in sync with `StatusResponse.run_id`) |
| `cursor` | `int` | Number of events the UI has already drained |

---

### A.2 `src/ui/apps/rade_analytics/data/backend.py` — Step 2 wrappers

Three surgical edits. **Do not replace the file** — add the snippets below
into the locations indicated. New lines total ~240; net diff is purely
additive.

**Edit 1** — Add inference Pydantic model imports alongside the existing
model imports near the top of the file:

```python
from src.rade_ml_pt.ensemble.api.client import RadeApiClient, RadeApiError
from src.rade_ml_pt.ensemble.api.models.clusters import ClustersResponse
from src.rade_ml_pt.ensemble.api.models.governance import (
    GovernanceRegistryResponse,
)
from src.rade_ml_pt.ensemble.api.models.inference import (
    EventsResponse,
    LoadResponse,
    LoadScenariosResponse,
    ManifestResponse,
    RunResponse,
    RunsListResponse,
    ScenariosSnapshotResponse,
    StatusResponse,
    ValidateResponse,
    ValidationSnapshotResponse,
)
from src.rade_ml_pt.ensemble.api.models.meta import HealthResponse, VersionsResponse
from src.rade_ml_pt.ensemble.api.models.overview import OverviewResponse
from src.rade_ml_pt.ensemble.api.models.trade_graph import TradeGraphResponse
```

**Edit 2** — Append seven cache bindings to `_bind_cached_methods()` just
before the "Predictions are intentionally NOT cached" comment:

```python
        # Inference (Stage 12) — only the *data-plane* GETs are cached.
        # Per-run artifacts are immutable on disk, so a long TTL is
        # safe and dramatically reduces flicker when the user pivots
        # between tabs.  /runs has a shorter TTL because new runs
        # land continuously while the dashboard is open.  Control-
        # plane endpoints are uncached by design (live state).
        self._inference_runs_list_cached       = cache.memoize(timeout=30)(
            self._fetch_inference_runs_list
        )
        self._inference_run_manifest_cached    = cache.memoize(timeout=ttl)(
            self._fetch_inference_run_manifest
        )
        self._inference_portfolio_df_cached    = cache.memoize(timeout=ttl)(
            self._fetch_inference_portfolio_df
        )
        self._inference_clusters_df_cached     = cache.memoize(timeout=ttl)(
            self._fetch_inference_clusters_df
        )
        self._inference_cluster_trades_cached  = cache.memoize(timeout=ttl)(
            self._fetch_inference_cluster_trades_df
        )
        self._inference_validation_cached      = cache.memoize(timeout=ttl)(
            self._fetch_inference_validation
        )
        self._inference_scenarios_cached       = cache.memoize(timeout=ttl)(
            self._fetch_inference_scenarios
        )
```

**Edit 3** — Append the new section (fetchers + public methods) at the very
end of the `RadeBackend` class. If you've already removed the legacy
`run_inference()` method, just paste this after the last remaining public
method:

```python
    # ══════════════════════════════════════════════════════════════
    # Inference (Stage 12) — API-driven control plane + data plane
    #
    # Thin wrappers over the corresponding :class:`RadeApiClient`
    # methods, wrapped in :class:`BackendResult` for the standard
    # tri-state envelope.  Two design lines kept very deliberately:
    #
    #   * **Control plane is uncached.**  The pipeline state machine
    #     advances on every call — caching it would silently break
    #     the activity-log narrative (stale events) and the gate
    #     between stages (stale ``is_valid`` flag).
    #   * **Data plane is cached.**  Once a run finishes, its
    #     artifacts are immutable, so per-run reads can sit on a
    #     long TTL.  ``/runs`` (the discovery list) uses a shorter
    #     TTL because new runs continue to land while the dashboard
    #     is open.
    # ══════════════════════════════════════════════════════════════

    # ── Raw fetchers — control plane (uncached) ──────────────────

    def _fetch_inference_load(self) -> LoadResponse:
        return self._client.load_ensemble()

    def _fetch_inference_load_scenarios(
        self, new_scenario_dir: str,
    ) -> LoadScenariosResponse:
        return self._client.load_scenarios(new_scenario_dir)

    def _fetch_inference_validate(self) -> ValidateResponse:
        return self._client.validate_scenarios()

    def _fetch_inference_start(
        self, artifacts_dir: Optional[str],
    ) -> RunResponse:
        return self._client.run_inference(artifacts_dir=artifacts_dir)

    def _fetch_inference_status(self) -> StatusResponse:
        return self._client.inference_status()

    def _fetch_inference_events(self, cursor: int) -> EventsResponse:
        return self._client.inference_events(cursor=cursor)

    def _fetch_inference_manifest(self) -> ManifestResponse:
        return self._client.inference_manifest()

    # ── Raw fetchers — data plane (cached above) ─────────────────

    def _fetch_inference_runs_list(self) -> RunsListResponse:
        return self._client.list_inference_runs()

    def _fetch_inference_run_manifest(self, run_id: str) -> ManifestResponse:
        return self._client.inference_run_manifest(run_id)

    def _fetch_inference_portfolio_df(self, run_id: str) -> pd.DataFrame:
        """Portfolio long-format frame — one row per scenario.

        Columns: ``scenario_label``, ``sum_pnl_scaled``,
        ``sum_pnl_original``, ``n_clusters``.  ``run_id`` is stashed
        on ``df.attrs`` so callers can defensively cross-check.
        """
        resp = self._client.inference_portfolio(run_id)
        df = pd.DataFrame([r.model_dump() for r in resp.rows])
        df.attrs["run_id"]      = resp.run_id
        df.attrs["n_scenarios"] = resp.n_scenarios
        return df

    def _fetch_inference_clusters_df(self, run_id: str) -> pd.DataFrame:
        """Cluster-summary long-format frame — one row per (cluster, scenario).

        Columns match :class:`ClusterSummaryRow`: scenario_label,
        cluster_id, sum/mean/std/min/max_pnl_{scaled,original}.
        """
        resp = self._client.inference_clusters_summary(run_id)
        df = pd.DataFrame([r.model_dump() for r in resp.rows])
        df.attrs["run_id"]      = resp.run_id
        df.attrs["n_clusters"]  = resp.n_clusters
        df.attrs["n_scenarios"] = resp.n_scenarios
        return df

    def _fetch_inference_cluster_trades_df(
        self,
        run_id:     str,
        cluster_id: str,
        space:      str,
    ) -> pd.DataFrame:
        """Per-cluster trade-level wide frame.

        Index = scenario_labels, columns = trade_ids, values = PnL
        in the requested ``space``.  Matches the parquet layout
        :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
        emits.
        """
        resp = self._client.inference_cluster_trades(
            run_id, cluster_id, space=space,
        )
        df = pd.DataFrame(
            data    = resp.values,
            index   = pd.Index(resp.scenario_labels, name="scenario_label"),
            columns = pd.Index(resp.trade_ids,       name="trade_id"),
        )
        df.attrs["run_id"]     = resp.run_id
        df.attrs["cluster_id"] = resp.cluster_id
        df.attrs["space"]      = resp.space
        return df

    def _fetch_inference_validation(
        self, run_id: str,
    ) -> ValidationSnapshotResponse:
        return self._client.inference_validation(run_id)

    def _fetch_inference_scenarios(
        self, run_id: str,
    ) -> ScenariosSnapshotResponse:
        return self._client.inference_scenarios(run_id)

    # ── Public — control plane (always live; never cached) ──────

    def inference_load(self) -> BackendResult[LoadResponse]:
        """Cold-load the ensemble + per-cluster contexts on the server."""
        return self._wrap(self._fetch_inference_load)

    def inference_load_scenarios(
        self, new_scenario_dir: str,
    ) -> BackendResult[LoadScenariosResponse]:
        """Parse a folder of shock CSVs into the active run state."""
        return self._wrap(self._fetch_inference_load_scenarios, new_scenario_dir)

    def inference_validate(self) -> BackendResult[ValidateResponse]:
        """Compute the per-cluster routing decisions + run-level checks."""
        return self._wrap(self._fetch_inference_validate)

    def inference_start(
        self,
        *,
        artifacts_dir: Optional[str] = None,
    ) -> BackendResult[RunResponse]:
        """Dispatch the inference run onto the API's worker thread.

        Returns immediately with ``status='running'`` — caller is
        expected to arm the polling loop and drain
        :meth:`inference_events` until terminal state.
        """
        return self._wrap(self._fetch_inference_start, artifacts_dir)

    def inference_status(self) -> BackendResult[StatusResponse]:
        """Cheap probe used to gate the next-button in the UI."""
        return self._wrap(self._fetch_inference_status)

    def inference_events(
        self, *, cursor: int = 0,
    ) -> BackendResult[EventsResponse]:
        """Cursor-paginated tail of the activity log for the live run."""
        return self._wrap(self._fetch_inference_events, cursor)

    def inference_manifest(self) -> BackendResult[ManifestResponse]:
        """Active run's ``manifest.json`` — available once status=complete."""
        return self._wrap(self._fetch_inference_manifest)

    # ── Public — data plane (cached) ─────────────────────────────

    def list_inference_runs(self) -> BackendResult[RunsListResponse]:
        """Discover every inference run on disk, most recent first."""
        return self._wrap(self._inference_runs_list_cached)

    def inference_run_manifest(
        self, run_id: str,
    ) -> BackendResult[ManifestResponse]:
        """Read ``manifest.json`` for a historical run."""
        return self._wrap(self._inference_run_manifest_cached, run_id)

    def inference_portfolio_df(
        self, run_id: str,
    ) -> BackendResult[pd.DataFrame]:
        """Portfolio-level summary frame for ``run_id``."""
        return self._wrap(self._inference_portfolio_df_cached, run_id)

    def inference_clusters_df(
        self, run_id: str,
    ) -> BackendResult[pd.DataFrame]:
        """Cluster-level summary frame for ``run_id``."""
        return self._wrap(self._inference_clusters_df_cached, run_id)

    def inference_cluster_trades_df(
        self,
        run_id:     str,
        cluster_id: str,
        *,
        space:      str = "original",
    ) -> BackendResult[pd.DataFrame]:
        """Per-cluster trade-level wide frame for ``run_id``."""
        return self._wrap(
            self._inference_cluster_trades_cached,
            run_id, cluster_id, space,
        )

    def inference_validation(
        self, run_id: str,
    ) -> BackendResult[ValidationSnapshotResponse]:
        """``ValidationReport`` snapshot embedded in ``run_id``'s manifest."""
        return self._wrap(self._inference_validation_cached, run_id)

    def inference_scenarios(
        self, run_id: str,
    ) -> BackendResult[ScenariosSnapshotResponse]:
        """``LoadedScenariosReport`` snapshot embedded in ``run_id``'s manifest."""
        return self._wrap(self._inference_scenarios_cached, run_id)
```

**Caching policy** (single source of truth, mirrored in the cache bindings):

| Method group | Cached? | TTL |
|---|---|---|
| Control plane (7 methods) | No | — |
| `list_inference_runs` | Yes | 30s |
| Per-run data plane (6 methods) | Yes | 300s (default) |

---

### A.3 `src/ui/apps/rade_analytics/callbacks/inference_cb.py` — NEW file

**Copy the full source below into a new file at this path.** The file is
the entire Stage 12 wiring: 8 callbacks across `_register_capture` and
`_register_render`, plus two private layout helpers.

```python
"""Inference Console callbacks — Stage 12 (API-driven, threaded ``/run``).

Wires the inference page's eight gestures to the
:mod:`~src.rade_ml_pt.ensemble.api.routers.inference` endpoints via
:class:`~src.ui.apps.rade_analytics.data.backend.RadeBackend`.

State machine
-------------

The page is a thin client over the API's run-state machine.  Each
gesture either advances the server-side state or polls it:

    on_mount     ─ url=/inference ──────────►  POST /load        (cold-load)
                                               │
                                               ▼
    on_upload    ─ Upload btn ──────────────►  POST /scenarios
                                               │
                                               ▼
    on_validate  ─ Validate btn ────────────►  POST /validate    (gates run)
                                               │
                                               ▼
    on_run       ─ Run btn ─────────────────►  POST /run         (async)
                                               │
                                               ▼  arms polling
    on_poll      ─ Interval tick ────┬──────►  GET  /events?cursor=N
                                     └──────►  GET  /status
                                               │
                                               ▼  terminal state
                                               hydrate run_meta_store
                                               │
                                               ▼
    hydrate_results  ─ run_meta_store ───────►  GET /runs/{id}/portfolio,
                                                GET /runs/{id}/clusters
                                                renders KPIs + figures + grid

Two ancillary callbacks decouple display from data flow:

    render_activity  ─ activity_log_store ──►  rebuilds the feed DOM
    on_row_select    ─ AG Grid click ───────►  writes selected_scenario_store

The polling pair (``polling_store`` + ``poll_interval``) is the only
piece of bespoke client state; everything else flows from server
state on demand.

Activity-log delivery
---------------------

Server events are emitted by :class:`EventCollector` on the pipeline
and exposed via ``GET /events?cursor=N``.  ``EventModel`` is shape-
compatible with the layout's ``ActivityEntry`` dict, so events flow
through with ``model_dump()`` and need no further mapping.  Local
events (e.g. *"Ensemble loaded"* on mount) use :func:`_local_event`
to keep the timestamp + UUID conventions consistent.

Page Contract anchors
---------------------

* §2.1 — capture/render split: capture writes Stores + side effects,
  render reads Stores + emits children/figures.  Both sections are
  registered through :func:`register`, mirroring ``overview_cb`` /
  ``portfolio_cb``.
* §3 Rule L4 — page identity is the URL pathname, not a per-page
  mount tripwire; ``on_mount`` keys on ``Input(SHELL_IDS["url"], …)``.
* §6 — long-running side effects (the ``/run`` dispatch) write the
  ``polling_store`` *before* the user navigates away; if they
  revisit, ``on_mount`` re-syncs via ``GET /status``.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from ..figures.inference_charts import (
    empty_pnl_distribution,
    empty_pnl_overlay,
    empty_pnl_timeseries,
    empty_risk_attribution,
    empty_stress_tails,
)
from ..layouts.inference import INFERENCE_IDS, render_activity_entries
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

_PLACEHOLDER:        str = "—"
_INFERENCE_ROUTE:    str = "/inference"

# API run statuses that mean the worker thread has finished — the
# poll callback uses this to disarm the polling pair.
_TERMINAL_STATUSES: set[str] = {"complete", "failed"}

# Statuses the API reports while a run is in flight (worker thread
# alive).  Anything outside this set + the terminal set means the
# state machine is idle and the run button should be re-enabled.
_RUNNING_STATUS:     str = "running"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _local_event(
    stage:  str,
    phase:  str,
    *,
    status: str           = "ok",
    target: Optional[str] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct an activity-log entry locally.

    Used by callbacks that want to narrate a client-side gesture
    (e.g. *"Ensemble load requested"*) without waiting for the
    server's confirmation event.  Shape matches
    :class:`~src.rade_ml_pt.ensemble.api.models.inference.EventModel`
    so the local entry can sit alongside server events in the
    activity-log store without a mapping layer.
    """
    return {
        "id":     uuid.uuid4().hex,
        "stage":  stage,
        "phase":  phase,
        "status": status,
        "ts":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,
        "detail": detail,
    }


def _format_currency(value: Any, *, digits: int = 2) -> str:
    """Pretty-print a PnL-style number; ``—`` on missing / NaN."""
    if value is None:
        return _PLACEHOLDER
    try:
        val = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val:,.{digits}f}"


def _format_int(value: Any) -> str:
    if value is None:
        return _PLACEHOLDER
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _PLACEHOLDER


def _status_icon(status: str) -> DashIconify:
    """Render an ingest-status icon matching the activity log's
    semantic-colour palette so the two feedback channels stay
    visually coherent."""
    mapping = {
        "ok":      ("tabler:circle-check",   "#34d399"),
        "fail":    ("tabler:circle-x",       "#fb7185"),
        "running": ("tabler:loader-2",       "#a78bfa"),
        "pending": ("tabler:circle-dashed",  "#94a3b8"),
    }
    icon, colour = mapping.get(status, mapping["pending"])
    return DashIconify(icon=icon, width=18, color=colour)


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Inference Console callback to ``app``.

    Mirrors the Page Contract §2 capture/render split — same shape
    every other page module follows (``overview_cb``, ``portfolio_cb``,
    ``cluster_deep_dive_cb``).
    """
    _register_capture(app, backend)
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# Section dispatchers
# ─────────────────────────────────────────────────────────────────────

def _register_capture(app: "Dash", backend: "RadeBackend") -> None:
    """Capture-side callbacks (gestures → Stores + API side effects)."""
    _register_on_mount(app, backend)
    _register_on_upload(app, backend)
    _register_on_validate(app, backend)
    _register_on_run(app, backend)
    _register_on_poll(app, backend)
    _register_on_row_select(app)


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Render-side callbacks (Stores → DOM, no Store writes)."""
    _register_render_activity(app)
    _register_hydrate_results(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. on_mount — URL hits /inference → POST /load
# ═════════════════════════════════════════════════════════════════════

def _register_on_mount(app: "Dash", backend: "RadeBackend") -> None:
    """Cold-load the ensemble whenever the page first mounts.

    Idempotent on the server side: an already-loaded API silently
    replaces its state.  We still gate with a ``/status`` probe to
    avoid the full registry walk on every route flip.
    """

    @app.callback(
        Output(INFERENCE_IDS["subtitle"],          "children"),
        Output(INFERENCE_IDS["activity_log_store"], "data"),
        Output(INFERENCE_IDS["ingest_meta_store"],  "data"),
        Output(INFERENCE_IDS["run_meta_store"],     "data"),
        Output(INFERENCE_IDS["polling_store"],      "data"),
        Output(INFERENCE_IDS["poll_interval"],      "disabled"),
        Input(SHELL_IDS["url"],                     "pathname"),
        prevent_initial_call=False,
    )
    def _on_mount(
        pathname: Optional[str],
    ) -> Tuple[Any, List[Dict[str, Any]], Optional[Any], Optional[Any], Dict[str, Any], bool]:
        if pathname != _INFERENCE_ROUTE:
            raise PreventUpdate

        # Fresh page mount → reset every page Store to its initial
        # shape.  This is intentional: stale data from a prior visit
        # would otherwise leak into the new render before /status
        # has a chance to refresh it.
        polling_initial = {"armed": False, "run_id": None, "cursor": 0}

        # /status: cheap, no-side-effect probe.  Tells us whether
        # the API already has an active run we should adopt or
        # whether we need to do a cold /load.
        status_res = backend.inference_status()
        if status_res.ok and status_res.data is not None and status_res.data.has_active_run:
            status = status_res.data
            subtitle = (
                f"Run {status.run_id} · ensemble {status.ensemble_version} "
                f"· state '{status.status}'"
            )
            log: List[Dict[str, Any]] = [
                _local_event(
                    "inference",
                    "Resumed existing run",
                    target=status.run_id,
                    detail=f"State: {status.status}",
                )
            ]
            # Adopt the existing run_id so subsequent ``/events`` polls
            # advance the same cursor.  Arm only if the run is still
            # in flight; otherwise leave disarmed so the user must
            # explicitly re-arm via the Run button.
            armed = status.status == _RUNNING_STATUS
            polling_initial = {
                "armed":   armed,
                "run_id":  status.run_id,
                "cursor":  status.n_events or 0,
            }
            interval_disabled = not armed
            return subtitle, log, None, None, polling_initial, interval_disabled

        # No active run → cold-load.  Failures here are surfaced via
        # the activity log so the user sees *why* nothing else works.
        load_res = backend.inference_load()
        if not load_res.ok:
            logger.warning("/load failed: %s", load_res.error)
            return (
                "Failed to load ensemble — check API logs.",
                [_local_event(
                    "inference",
                    "Ensemble load failed",
                    status="fail",
                    detail=load_res.error,
                )],
                None,
                None,
                polling_initial,
                True,
            )

        load = load_res.data
        subtitle = (
            f"Loaded ensemble {load.ensemble_version} "
            f"({load.n_clusters} clusters) — upload scenarios to begin."
        )
        log = [_local_event(
            "inference",
            "Ensemble loaded",
            target=load.ensemble_version,
            detail=f"{load.n_clusters} clusters",
        )]
        return subtitle, log, None, None, polling_initial, True


# ═════════════════════════════════════════════════════════════════════
# 2. on_upload — Upload scenarios btn → POST /scenarios
# ═════════════════════════════════════════════════════════════════════

def _register_on_upload(app: "Dash", backend: "RadeBackend") -> None:
    """Submit the typed folder path to the API's scenario loader."""

    @app.callback(
        Output(INFERENCE_IDS["ingest_meta_store"],   "data", allow_duplicate=True),
        Output(INFERENCE_IDS["ingest_status"],       "children"),
        Output(INFERENCE_IDS["activity_log_store"],  "data", allow_duplicate=True),
        Output(INFERENCE_IDS["upload_scenarios_btn"], "loading"),
        Input(INFERENCE_IDS["upload_scenarios_btn"], "n_clicks"),
        State(INFERENCE_IDS["scenario_folder_path"], "value"),
        State(INFERENCE_IDS["activity_log_store"],   "data"),
        prevent_initial_call=True,
    )
    def _on_upload(
        n_clicks:    Optional[int],
        folder_path: Optional[str],
        log:         Optional[List[Dict[str, Any]]],
    ) -> Tuple[Any, Any, List[Dict[str, Any]], bool]:
        if not n_clicks:
            raise PreventUpdate

        log = list(log or [])

        # Defensive empty-input check.  The button is always
        # clickable but a no-text submission is a no-op.
        if not folder_path or not folder_path.strip():
            log.append(_local_event(
                "ingest",
                "Empty scenario folder path",
                status="fail",
                detail="Type a server-readable folder path before uploading.",
            ))
            return no_update, _status_icon("fail"), log, False

        log.append(_local_event(
            "ingest",
            "Uploading scenarios",
            status="running",
            target=folder_path,
        ))

        # The actual API call.  Synchronous on the server side
        # (parsing a folder of CSVs is fast), so the loading state
        # on the button only ever shows briefly.
        res = backend.inference_load_scenarios(folder_path)
        if not res.ok:
            logger.warning("/scenarios failed: %s", res.error)
            log.append(_local_event(
                "ingest",
                "Scenario load failed",
                status="fail",
                target=folder_path,
                detail=res.error,
            ))
            return no_update, _status_icon("fail"), log, False

        scenarios = res.data
        ingest_meta: Dict[str, Any] = {
            "source":            folder_path,
            "n_risk_factors":    scenarios.n_risk_factors,
            "n_scenarios":       scenarios.n_scenarios,
            "risk_factor_names": list(scenarios.risk_factor_names),
            "scenario_labels":   list(scenarios.scenario_labels),
            "completed_ts":      datetime.now(timezone.utc).isoformat(
                timespec="seconds",
            ),
        }
        log.append(_local_event(
            "ingest",
            "Scenarios ingested",
            target=folder_path,
            detail=(
                f"{scenarios.n_scenarios} scenarios across "
                f"{scenarios.n_risk_factors} risk factors"
            ),
        ))
        return ingest_meta, _status_icon("ok"), log, False


# ═════════════════════════════════════════════════════════════════════
# 3. on_validate — Validate-only btn → POST /validate
# ═════════════════════════════════════════════════════════════════════

def _register_on_validate(app: "Dash", backend: "RadeBackend") -> None:
    """Validate the loaded scenarios and gate the Run button on success."""

    @app.callback(
        Output(INFERENCE_IDS["manifest_preview_container"], "children"),
        Output(INFERENCE_IDS["run_btn"],                    "disabled"),
        Output(INFERENCE_IDS["activity_log_store"],         "data", allow_duplicate=True),
        Output(INFERENCE_IDS["validate_only_btn"],          "loading"),
        Input(INFERENCE_IDS["validate_only_btn"],           "n_clicks"),
        State(INFERENCE_IDS["activity_log_store"],          "data"),
        prevent_initial_call=True,
    )
    def _on_validate(
        n_clicks: Optional[int],
        log:      Optional[List[Dict[str, Any]]],
    ) -> Tuple[List[Any], bool, List[Dict[str, Any]], bool]:
        if not n_clicks:
            raise PreventUpdate

        log = list(log or [])
        log.append(_local_event("validate", "Validation requested", status="running"))

        res = backend.inference_validate()
        if not res.ok:
            logger.warning("/validate failed: %s", res.error)
            log.append(_local_event(
                "validate",
                "Validation failed",
                status="fail",
                detail=res.error,
            ))
            return (
                [_manifest_card_error(res.error or "Unknown error")],
                True,
                log,
                False,
            )

        report = res.data
        is_valid = bool(report.is_valid)
        log.append(_local_event(
            "validate",
            "Validation complete" if is_valid else "Validation failed",
            status="ok" if is_valid else "fail",
            detail=(
                f"{report.affected_count} affected / "
                f"{report.unaffected_count} unaffected clusters"
            ),
        ))
        return (
            _manifest_card_for_validation(report),
            not is_valid,
            log,
            False,
        )


def _manifest_card_for_validation(report: Any) -> List[Any]:
    """Render a compact validation summary into the manifest card."""
    rows: List[Any] = [
        html.Div(
            f"Ensemble · {report.ensemble_version}",
            className="text-xs font-semibold text-slate-200",
        ),
        html.Div(
            f"{report.n_scenarios} scenarios across "
            f"{len(report.cluster_decisions)} clusters",
            className="text-xs text-slate-400",
        ),
        html.Div(
            f"Affected: {report.affected_count} · "
            f"Unaffected: {report.unaffected_count} · "
            f"Cheap path: {'yes' if report.cheap_path_used else 'no'}",
            className="text-xs text-slate-400",
        ),
    ]
    if report.errors:
        rows.append(html.Div(
            "Errors:",
            className="text-xs font-semibold text-rose-300 mt-2",
        ))
        for err in report.errors:
            rows.append(html.Div(f"• {err}", className="text-xs text-rose-300"))
    if report.warnings:
        rows.append(html.Div(
            "Warnings:",
            className="text-xs font-semibold text-amber-300 mt-2",
        ))
        for warn in report.warnings:
            rows.append(html.Div(f"• {warn}", className="text-xs text-amber-300"))
    return rows


def _manifest_card_error(message: str) -> Any:
    """Render an error state in the manifest card."""
    return html.Div(
        className="flex flex-col gap-1 text-rose-300",
        children=[
            html.Div("Validation failed", className="text-xs font-semibold"),
            html.Div(message, className="text-xs"),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# 4. on_run — Run btn → POST /run + arm polling
# ═════════════════════════════════════════════════════════════════════

def _register_on_run(app: "Dash", backend: "RadeBackend") -> None:
    """Dispatch the inference run and arm the polling pair.

    The API's ``/run`` returns immediately with ``status='running'``;
    the worker thread does the actual forward pass.  We mark the
    polling store armed and disable=False the interval so the next
    tick (1Hz later) starts draining ``/events``.
    """

    @app.callback(
        Output(INFERENCE_IDS["polling_store"],        "data", allow_duplicate=True),
        Output(INFERENCE_IDS["poll_interval"],        "disabled", allow_duplicate=True),
        Output(INFERENCE_IDS["activity_log_store"],   "data", allow_duplicate=True),
        Output(INFERENCE_IDS["run_btn"],              "loading"),
        Input(INFERENCE_IDS["run_btn"],               "n_clicks"),
        State(INFERENCE_IDS["polling_store"],         "data"),
        State(INFERENCE_IDS["activity_log_store"],    "data"),
        prevent_initial_call=True,
    )
    def _on_run(
        n_clicks:   Optional[int],
        polling:    Optional[Dict[str, Any]],
        log:        Optional[List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], bool, List[Dict[str, Any]], bool]:
        if not n_clicks:
            raise PreventUpdate

        log = list(log or [])
        log.append(_local_event("inference", "Run dispatched", status="running"))

        res = backend.inference_start()
        if not res.ok:
            logger.warning("/run failed: %s", res.error)
            log.append(_local_event(
                "inference",
                "Run dispatch failed",
                status="fail",
                detail=res.error,
            ))
            return polling or {"armed": False, "run_id": None, "cursor": 0}, True, log, False

        run = res.data
        # Cursor=0 — the worker thread may have already emitted a
        # few events before our HTTP response landed; the first
        # /events poll picks those up.
        new_polling = {
            "armed":  True,
            "run_id": run.run_id,
            "cursor": 0,
        }
        # Interval ``disabled=False`` flips polling on; the next
        # ``on_poll`` tick (≤1s later) will start the drain.
        return new_polling, False, log, True


# ═════════════════════════════════════════════════════════════════════
# 5. on_poll — Interval tick → GET /events + GET /status
# ═════════════════════════════════════════════════════════════════════

def _register_on_poll(app: "Dash", backend: "RadeBackend") -> None:
    """Drain events + watch for terminal state once per interval tick.

    Heart of the live-update story.  Runs at 1 Hz when armed.  Two
    invariants worth highlighting:

    * ``polling_store.cursor`` is the single source of truth for
      pagination.  We never trust the interval's ``n_intervals``.
    * Terminal-state hydration (writing ``run_meta_store``) happens
      *here*, not in :func:`_on_run`, because only the poll knows
      when the worker thread actually finishes.
    """

    @app.callback(
        Output(INFERENCE_IDS["activity_log_store"],  "data", allow_duplicate=True),
        Output(INFERENCE_IDS["polling_store"],       "data", allow_duplicate=True),
        Output(INFERENCE_IDS["poll_interval"],       "disabled", allow_duplicate=True),
        Output(INFERENCE_IDS["run_meta_store"],      "data", allow_duplicate=True),
        Output(INFERENCE_IDS["run_btn"],             "loading", allow_duplicate=True),
        Input(INFERENCE_IDS["poll_interval"],        "n_intervals"),
        State(INFERENCE_IDS["polling_store"],        "data"),
        State(INFERENCE_IDS["activity_log_store"],   "data"),
        prevent_initial_call=True,
    )
    def _on_poll(
        n_intervals: Optional[int],
        polling:     Optional[Dict[str, Any]],
        log:         Optional[List[Dict[str, Any]]],
    ) -> Tuple[Any, Any, Any, Any, Any]:
        polling = polling or {"armed": False, "run_id": None, "cursor": 0}
        if not polling.get("armed"):
            # Defence-in-depth: the interval should be disabled when
            # disarmed, but a stale tick can still land in-flight.
            # PreventUpdate is the cheapest way to short-circuit
            # without firing six no-ops on every output.
            raise PreventUpdate

        cursor = int(polling.get("cursor", 0))
        log = list(log or [])

        # 1. Drain new events.  EventModel is shape-compatible with
        #    the layout's activity-entry contract, so model_dump()
        #    is a direct passthrough.
        events_res = backend.inference_events(cursor=cursor)
        if events_res.ok and events_res.data is not None:
            new_events = events_res.data.events
            for ev in new_events:
                log.append(ev.model_dump())
            cursor = int(events_res.data.next_cursor)

        # 2. Probe terminal state.  We do this every tick (even if
        #    no new events landed) so a fast run that completes
        #    between ticks still gets hydrated promptly.
        status_res = backend.inference_status()
        if not status_res.ok or status_res.data is None:
            # Transport hiccup — keep polling, the next tick may
            # succeed.  Don't disarm here; that would silently
            # strand the page on a temporary network blip.
            logger.debug("/status hiccup: %s", status_res.error)
            return (
                log,
                {**polling, "cursor": cursor},
                no_update,
                no_update,
                no_update,
            )

        status = status_res.data
        if status.status in _TERMINAL_STATUSES:
            # Terminal state — disarm, snapshot run_meta_store, and
            # stop the interval.  The hydrate_results callback fires
            # off run_meta_store and pulls the data-plane artifacts.
            run_meta = {
                "run_id":           status.run_id,
                "ensemble_version": status.ensemble_version,
                "status":           status.status,
                "manifest_path":    status.manifest_path,
                "completed_ts":     datetime.now(timezone.utc).isoformat(
                    timespec="seconds",
                ),
            }
            disarmed = {**polling, "armed": False, "cursor": cursor}
            return log, disarmed, True, run_meta, False

        # Still running — bump the cursor, keep polling.
        return (
            log,
            {**polling, "cursor": cursor},
            no_update,
            no_update,
            no_update,
        )


# ═════════════════════════════════════════════════════════════════════
# 6. render_activity — activity_log_store → activity-feed DOM
# ═════════════════════════════════════════════════════════════════════

def _register_render_activity(app: "Dash") -> None:
    """Rebuild the activity feed whenever its Store mutates.

    The activity log is the user-visible heart of the page during a
    run — every other callback advances it indirectly via
    ``allow_duplicate=True`` writes; this single render callback
    converts the store payload to DOM children.
    """

    @app.callback(
        Output(INFERENCE_IDS["activity_log_container"], "children"),
        Input(INFERENCE_IDS["activity_log_store"],      "data"),
        prevent_initial_call=False,
    )
    def _render(entries: Optional[List[Dict[str, Any]]]) -> List[Any]:
        return render_activity_entries(entries)


# ═════════════════════════════════════════════════════════════════════
# 7. hydrate_results — run_meta_store → KPIs + figures + AG Grid
# ═════════════════════════════════════════════════════════════════════

def _register_hydrate_results(app: "Dash", backend: "RadeBackend") -> None:
    """Pull data-plane artifacts when a run completes and paint the page.

    Fires on:
      * ``run_meta_store`` write (terminal-state hydration from
        :func:`_on_poll`), and
      * each of the three segmented controls so users can switch
        chart modes without re-running.

    Stage 12 still uses :mod:`figures.inference_charts`'s empty-state
    builders for the three figures; Stage 13 swaps them in place.
    Everything else (KPIs, AG Grid, stress mini-KPIs) is live data.
    """

    @app.callback(
        # KPI strip
        Output(INFERENCE_IDS["kpi_scenarios_value"], "children"),
        Output(INFERENCE_IDS["kpi_clusters_value"],  "children"),
        Output(INFERENCE_IDS["kpi_latency_value"],   "children"),
        Output(INFERENCE_IDS["kpi_portfolio_value"], "children"),
        # Three figures
        Output(INFERENCE_IDS["chart_main"],              "figure"),
        Output(INFERENCE_IDS["risk_attribution_chart"],  "figure"),
        Output(INFERENCE_IDS["stress_tails_chart"],      "figure"),
        # Stress mini-KPIs
        Output(INFERENCE_IDS["stress_kpi_var"],   "children"),
        Output(INFERENCE_IDS["stress_kpi_cvar"],  "children"),
        Output(INFERENCE_IDS["stress_kpi_worst"], "children"),
        # AG Grid rows
        Output(INFERENCE_IDS["scenario_results_grid"], "rowData"),
        # Triggers
        Input(INFERENCE_IDS["run_meta_store"],            "data"),
        Input(INFERENCE_IDS["chart_view_mode"],           "value"),
        Input(INFERENCE_IDS["risk_attribution_breakdown"], "value"),
        Input(INFERENCE_IDS["stress_tails_mode"],         "value"),
        prevent_initial_call=False,
    )
    def _hydrate(
        run_meta:   Optional[Dict[str, Any]],
        chart_mode: Optional[str],
        risk_axis:  Optional[str],
        stress_mode: Optional[str],
    ) -> Tuple[Any, ...]:
        del chart_mode, risk_axis, stress_mode  # Stage 13 will dispatch on these

        empty_returns: Tuple[Any, ...] = (
            _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER,
            empty_pnl_distribution(),
            empty_risk_attribution(),
            empty_stress_tails(),
            _stress_kpi_block("VaR (95%)",  _PLACEHOLDER),
            _stress_kpi_block("CVaR (95%)", _PLACEHOLDER),
            _stress_kpi_block("Worst loss", _PLACEHOLDER),
            [],
        )

        # No completed run yet — keep the empty-state painting in
        # place but don't waste an API call.
        if not run_meta or run_meta.get("status") != "complete":
            return empty_returns

        run_id = run_meta.get("run_id")
        if not run_id:
            return empty_returns

        # Portfolio + cluster summary parquets.  Both are tiny
        # (one row per scenario / cluster×scenario) and cached at
        # the backend layer, so re-firing on segmented-control flips
        # is cheap.
        portfolio_res = backend.inference_portfolio_df(run_id)
        clusters_res  = backend.inference_clusters_df(run_id)

        if not portfolio_res.ok or portfolio_res.data is None:
            logger.warning("portfolio fetch failed: %s", portfolio_res.error)
            return empty_returns

        portfolio_df = portfolio_res.data
        clusters_df  = (
            clusters_res.data
            if clusters_res.ok and clusters_res.data is not None
            else pd.DataFrame()
        )

        # --- KPI strip ---------------------------------------------------
        n_scenarios = int(len(portfolio_df))
        n_clusters  = int(
            clusters_df["cluster_id"].nunique()
            if not clusters_df.empty and "cluster_id" in clusters_df.columns
            else 0
        )
        portfolio_total = float(
            portfolio_df["sum_pnl_original"].sum()
            if "sum_pnl_original" in portfolio_df.columns
            else 0.0
        )
        # Latency comes from the manifest — fetch it lazily here
        # rather than passing through ``run_meta`` so a refresh of
        # the manifest after the run (rare but possible) is picked
        # up automatically.
        manifest_res = backend.inference_run_manifest(run_id)
        latency_s    = (
            manifest_res.data.manifest.get("latency_seconds")
            if manifest_res.ok and manifest_res.data is not None else None
        )
        latency_txt  = (
            f"{float(latency_s):.2f}s"
            if latency_s is not None else _PLACEHOLDER
        )

        # --- Stress mini-KPIs --------------------------------------------
        # VaR / CVaR on the portfolio's sum_pnl_original — quick
        # quantile arithmetic, no extra API roundtrip.
        var_txt   = _PLACEHOLDER
        cvar_txt  = _PLACEHOLDER
        worst_txt = _PLACEHOLDER
        if "sum_pnl_original" in portfolio_df.columns and not portfolio_df.empty:
            pnl = portfolio_df["sum_pnl_original"].astype(float)
            var_value  = float(pnl.quantile(0.05))
            cvar_value = float(pnl[pnl <= var_value].mean()) if (pnl <= var_value).any() else var_value
            worst_value = float(pnl.min())
            var_txt   = _format_currency(var_value)
            cvar_txt  = _format_currency(cvar_value)
            worst_txt = _format_currency(worst_value)

        # --- AG Grid rows -----------------------------------------------
        # Map portfolio rows onto the column defs the layout
        # declares: ``scenario_id`` / ``cluster`` / ``predicted`` /
        # ``p95_band`` / ``confidence``.  Two columns have no
        # natural source from portfolio summary (``p95_band``,
        # ``confidence``); we leave them None so the grid renders
        # them as the ``—`` placeholder defined in the column
        # ``valueFormatter``.
        row_data: List[Dict[str, Any]] = [
            {
                "scenario_id": str(row.get("scenario_label", "")),
                "cluster":     f'{int(row.get("n_clusters", 0))} clusters',
                "predicted":   float(row.get("sum_pnl_original", 0.0)),
                "p95_band":    None,
                "confidence":  None,
            }
            for _, row in portfolio_df.iterrows()
        ]

        return (
            _format_int(n_scenarios),
            _format_int(n_clusters),
            latency_txt,
            _format_currency(portfolio_total),
            empty_pnl_distribution(),
            empty_risk_attribution(),
            empty_stress_tails(),
            _stress_kpi_block("VaR (95%)",  var_txt),
            _stress_kpi_block("CVaR (95%)", cvar_txt),
            _stress_kpi_block("Worst loss", worst_txt),
            row_data,
        )


def _stress_kpi_block(label: str, value: str) -> List[Any]:
    """Re-render one stress-tile body (label + value).

    Kept in this module rather than ``layouts/inference.py`` because
    the original layout helper builds the *outer* tile div; this
    helper builds only the children that the callback owns.
    """
    return [
        html.Div(label, className="rade-stress-mini-label"),
        html.Div(value, className="rade-stress-mini-value"),
    ]


# ═════════════════════════════════════════════════════════════════════
# 8. on_row_select — AG Grid row click → selected_scenario_store
# ═════════════════════════════════════════════════════════════════════

def _register_on_row_select(app: "Dash") -> None:
    """Capture the active scenario when the user clicks a grid row.

    Stage 12 only writes the store; chart-level filtering keys off
    this in Stage 13 once the figure builders are real.
    """

    @app.callback(
        Output(INFERENCE_IDS["selected_scenario_store"], "data"),
        Input(INFERENCE_IDS["scenario_results_grid"],    "selectedRows"),
        prevent_initial_call=True,
    )
    def _on_row_select(
        selected: Optional[List[Dict[str, Any]]],
    ) -> Optional[str]:
        if not selected:
            return None
        # Single-row selection (``rowSelection='single'`` per layout)
        # — first entry is the only one we care about.
        row = selected[0]
        return row.get("scenario_id")


__all__ = ["register"]
```

---

### A.4 `src/ui/apps/rade_analytics/callbacks/__init__.py` — Step 4 registration

Two tiny edits — add `inference_cb` to the import block and call
`inference_cb.register(app, backend)` in `register_all`.

**Edit 1** — Add to the existing `from . import (...)` block (alphabetical order):

```python
from . import (
    cluster_deep_dive_cb,
    evaluation_cb,
    inference_cb,
    overview_cb,
    portfolio_cb,
    splash_cb,
    trade_graph_cb,
)
```

**Edit 2** — Add the registration call inside `register_all` (after the
existing `cluster_deep_dive_cb` line, before the `governance_cb` comment):

```python
    register_router(app, backend)
    splash_cb.register(app, backend)
    overview_cb.register(app, backend)
    evaluation_cb.register(app, backend)
    portfolio_cb.register(app, backend)
    trade_graph_cb.register(app, backend)
    cluster_deep_dive_cb.register(app, backend)
    inference_cb.register(app, backend)
    # governance_cb intentionally not registered — see import block.
```

---

### Verification

After applying A.1–A.4 (or after each step independently — every one is
self-contained), confirm the integration:

```bash
# 1. Lints + compile
ruff check src/ui/apps/rade_analytics/
python -m py_compile src/ui/apps/rade_analytics/callbacks/inference_cb.py
python -m py_compile src/ui/apps/rade_analytics/data/backend.py
python -m py_compile src/ui/apps/rade_analytics/layouts/inference.py
python -m py_compile src/ui/apps/rade_analytics/callbacks/__init__.py

# 2. Boot the API + dashboard
uvicorn src.rade_ml_pt.ensemble.api.app:create_app --factory --port 8000 &
python -m src.ui.apps.rade_analytics.app

# 3. Hit the page in a browser → /inference
#    Expected behaviour:
#    * Page mounts → "Loaded ensemble <ver> (N clusters)" appears in subtitle
#    * Activity log shows "Ensemble loaded" green-tick row
#    * Type a scenario folder path → click "Upload scenarios" →
#      activity log narrates ingest; ingest-status slot turns green-tick
#    * Click "Validate only" → manifest card populates with
#      affected/unaffected counts; Run button enables
#    * Click "Run" → button shows loading spinner; activity log
#      narrates each cluster's progress at 1 Hz; on completion,
#      KPI tiles + stress mini-KPIs + AG Grid populate
#    * Three figures remain empty-state until Stage 13 lands

# 4. Programmatic smoke (no UI)
python -c "
from src.rade_ml_pt.ensemble.api.client import RadeApiClient
import time
with RadeApiClient('http://localhost:8000') as c:
    c.load_ensemble()
    c.load_scenarios('/path/to/shocks/')
    rep = c.validate_scenarios()
    assert rep.is_valid
    run = c.run_inference()
    while True:
        s = c.inference_status()
        if s.status in ('complete','failed'):
            break
        time.sleep(1)
    print('Run:', run.run_id, '→ status:', s.status)
    print('Portfolio rows:', len(c.inference_portfolio(run.run_id).rows))
"
```

If all four files compile, the FastAPI app boots without errors, and the
inference page renders the empty-state on first visit, Stage 12 is done.
Stage 13 (real Plotly figure builders) is the next piece — purely a swap
of the three `empty_*` functions in `figures/inference_charts.py` with no
callback-side changes required.

---

## Appendix B — Tenor timeseries waterfall backfill (ad-hoc utility)

> **Status**: Self-contained standalone script. Not part of the Rade /
> ensemble pipeline — pasted here purely as the copy-paste destination for
> the ad-hoc tenor-backfill request. Run from the command line against a
> Bloomberg-style tenor sheet; outputs a colour-coded backfilled workbook.
>
> **Dependencies**: `pip install pandas openpyxl scipy scikit-learn`
>
> **Usage**:
>
> ```bash
> python backfill_waterfall.py <input_file.xlsx>
> # → <input_file>_backfilled.xlsx
> ```

### Waterfall priority order

| Priority | Method | When it applies |
|---|---|---|
| 1 | Cross-tenor OLS regression | Missing date has observed values in correlated tenors (R² ≥ 0.90 from ≥ 30 overlap points) |
| 2 | Natural cubic spline | Interior gap with ≥ 4 surrounding known values in the same tenor |
| 3 | Linear interpolation | Interior gap where the spline can't fit (< 4 known values) |
| 4 | Forward / backward fill | Edge cases only — before first or after last observation |

### Source

```python
"""
Tenor Timeseries Waterfall Backfill
====================================
Backfills missing rates in a Bloomberg tenor timeseries using a four-stage
waterfall that prioritises the most market-informed method available for each
missing cell:

    Priority 1 — Cross-tenor regression
        If correlated tenors have observed values on the missing date, predict
        the missing rate via OLS regression trained on the overlapping history.
        This exploits the strong co-movement between tenors (typical R² > 0.99
        for adjacent swap rates).

    Priority 2 — Cubic spline interpolation (temporal)
        For interior gaps (bounded by known values in the same tenor), fit a
        natural cubic spline through surrounding observations. Preserves local
        curvature and avoids the sharp kinks of linear interpolation.

    Priority 3 — Linear interpolation (temporal)
        Fallback for interior gaps where too few surrounding points exist for a
        stable spline (e.g. fewer than 4 known values in the tenor).

    Priority 4 — Forward / backward fill
        Last resort for edge cases only — missing dates before the first
        observation or after the last observation in a tenor.

Usage
-----
    python backfill_waterfall.py <input_file.xlsx>

Output
------
    <input_file>_backfilled.xlsx with four sheets:
        1. Backfilled Data     — the completed timeseries
        2. Audit Trail         — method code per filled cell (colour-coded)
        3. Per Tenor Breakdown — fill counts by method for each tenor
        4. Summary             — overall statistics and configuration

Dependencies
------------
    pip install pandas openpyxl scipy scikit-learn
"""

import sys
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — adjust these thresholds to suit your dataset
# ═══════════════════════════════════════════════════════════════════════════════

# Cross-tenor regression: minimum overlapping observations to train a model
MIN_REGRESSION_OBSERVATIONS = 30

# Cross-tenor regression: minimum R² to accept a model's predictions
MIN_REGRESSION_R_SQUARED = 0.90

# Cubic spline: minimum known data points required in a tenor to fit a spline
MIN_SPLINE_POINTS = 4


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_timeseries(filepath: str) -> pd.DataFrame:
    """
    Read the input Excel file into a DatetimeIndex DataFrame.

    Expects:
        - Column A: dates (any Excel-recognised date format)
        - Row 1: tenor headers (e.g. Bloomberg tickers)
        - Body: rate values, with blanks for missing observations

    Returns:
        DataFrame with DatetimeIndex sorted ascending, tenor columns, and
        NaN for missing values.
    """
    df = pd.read_excel(filepath, index_col=0, parse_dates=True)
    df.index.name = "Date"
    df = df.sort_index()

    # Coerce any non-numeric entries (e.g. "N/A" strings) to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY 1: CROSS-TENOR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_regression_model(
    df: pd.DataFrame,
    target_tenor: str,
    candidate_tenors: list[str],
) -> tuple:
    """
    Find the best OLS regression model for a target tenor using other tenors
    as predictors.

    Tries all single-predictor and all two-predictor combinations from the
    candidate list. Selects the model with the highest R² that exceeds the
    minimum threshold.

    Args:
        df:               full DataFrame (may contain NaNs)
        target_tenor:     column name of the tenor to predict
        candidate_tenors: list of other tenor column names to use as features

    Returns:
        (model, feature_names, r_squared) if a valid model is found
        (None, None, None) otherwise
    """
    # Dates where the target tenor has observed values (training pool)
    target_observed = df[target_tenor].notna()

    best_model = None
    best_features = None
    best_r2 = -1.0

    # --- Single-predictor models ---
    for predictor in candidate_tenors:
        training_mask = target_observed & df[predictor].notna()
        n_obs = training_mask.sum()

        if n_obs < MIN_REGRESSION_OBSERVATIONS:
            continue

        X = df.loc[training_mask, [predictor]].values
        y = df.loc[training_mask, target_tenor].values

        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

        if r2 > best_r2:
            best_model, best_features, best_r2 = model, [predictor], r2

    # --- Two-predictor models (for potentially better fit) ---
    for i, pred_1 in enumerate(candidate_tenors):
        for pred_2 in candidate_tenors[i + 1:]:
            training_mask = target_observed & df[pred_1].notna() & df[pred_2].notna()
            n_obs = training_mask.sum()

            if n_obs < MIN_REGRESSION_OBSERVATIONS:
                continue

            X = df.loc[training_mask, [pred_1, pred_2]].values
            y = df.loc[training_mask, target_tenor].values

            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)

            if r2 > best_r2:
                best_model, best_features, best_r2 = model, [pred_1, pred_2], r2

    # Only return if R² meets the threshold
    if best_r2 >= MIN_REGRESSION_R_SQUARED and best_model is not None:
        return best_model, best_features, best_r2

    return None, None, None


def fill_via_regression(df: pd.DataFrame, audit: pd.DataFrame) -> int:
    """
    Priority 1: fill missing values using cross-tenor OLS regression.

    For each tenor with missing data, find the best regression model from
    other tenors. Apply predictions only on dates where the predictor tenors
    have observed values.

    Args:
        df:    timeseries DataFrame (modified in place)
        audit: audit trail DataFrame (modified in place)

    Returns:
        Number of cells filled.
    """
    tenors = df.columns.tolist()
    total_filled = 0

    for target_tenor in tenors:
        missing_mask = df[target_tenor].isna()
        if not missing_mask.any():
            continue

        # All other tenors are candidates
        candidates = [t for t in tenors if t != target_tenor]
        model, features, r2 = find_best_regression_model(df, target_tenor, candidates)

        if model is None:
            continue

        # Predict only where the target is missing AND all features are observed
        predict_mask = missing_mask.copy()
        for feature in features:
            predict_mask = predict_mask & df[feature].notna()

        if not predict_mask.any():
            continue

        # Apply predictions
        X_predict = df.loc[predict_mask, features].values
        df.loc[predict_mask, target_tenor] = model.predict(X_predict)

        # Record in audit trail with the R² for transparency
        audit.loc[predict_mask, target_tenor] = f"REG (R²={r2:.3f})"
        total_filled += predict_mask.sum()

    return total_filled


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY 2: CUBIC SPLINE INTERPOLATION (TEMPORAL)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_via_spline(df: pd.DataFrame, audit: pd.DataFrame) -> int:
    """
    Priority 2: fill remaining interior gaps using natural cubic spline
    interpolation along the time axis.

    A 'natural' spline sets the second derivative to zero at the endpoints,
    avoiding the wild oscillations that can occur with other boundary
    conditions. Only interior gaps (between the first and last known value)
    are filled — edge gaps are left for later stages.

    Args:
        df:    timeseries DataFrame (modified in place)
        audit: audit trail DataFrame (modified in place)

    Returns:
        Number of cells filled.
    """
    total_filled = 0

    for tenor in df.columns:
        series = df[tenor]
        missing_mask = series.isna()

        if not missing_mask.any():
            continue

        known_positions = np.where(series.notna())[0]

        # Need at least MIN_SPLINE_POINTS for a stable cubic spline
        if len(known_positions) < MIN_SPLINE_POINTS:
            continue

        # Define the interior region (between first and last known value)
        first_known = known_positions[0]
        last_known = known_positions[-1]

        # Identify interior missing positions only
        all_positions = np.arange(len(series))
        interior_missing = (
            missing_mask
            & (all_positions >= first_known)
            & (all_positions <= last_known)
        )
        missing_positions = np.where(interior_missing)[0]

        if len(missing_positions) == 0:
            continue

        # Fit natural cubic spline through known data points
        x_known = known_positions.astype(float)
        y_known = series.iloc[known_positions].values

        try:
            spline = CubicSpline(x_known, y_known, bc_type="natural")
            interpolated_values = spline(missing_positions.astype(float))

            # Use .iloc on the DataFrame directly to avoid copy-on-write issues
            missing_dates = df.index[missing_positions]
            df.loc[missing_dates, tenor] = interpolated_values
            audit.loc[missing_dates, tenor] = "SPLINE"
            total_filled += len(missing_positions)
        except Exception as e:
            # Spline fitting can fail on degenerate data; skip to next stage
            print(f"  Warning: spline failed for {tenor}: {e}")
            continue

    return total_filled


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY 3: LINEAR INTERPOLATION (TEMPORAL)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_via_linear(df: pd.DataFrame, audit: pd.DataFrame) -> int:
    """
    Priority 3: fill remaining interior gaps using linear interpolation
    along the time axis.

    This catches any interior gaps that the spline stage skipped (e.g. tenors
    with fewer than MIN_SPLINE_POINTS known values).

    Args:
        df:    timeseries DataFrame (modified in place)
        audit: audit trail DataFrame (modified in place)

    Returns:
        Number of cells filled.
    """
    total_filled = 0

    for tenor in df.columns:
        series = df[tenor]
        still_missing = series.isna()

        if not still_missing.any():
            continue

        # pandas interpolate with limit_area='inside' fills interior gaps only
        before = series.copy()
        df[tenor] = series.interpolate(method="linear", limit_area="inside")

        newly_filled = before.isna() & df[tenor].notna()
        audit.loc[newly_filled, tenor] = "LINEAR"
        total_filled += newly_filled.sum()

    return total_filled


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY 4: FORWARD / BACKWARD FILL (EDGE CASES ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_via_flat(df: pd.DataFrame, audit: pd.DataFrame) -> int:
    """
    Priority 4 (last resort): fill any remaining edge gaps by carrying the
    nearest known value forward or backward.

    This only affects cells at the very start or end of a tenor's timeseries
    where no second bounding value exists for interpolation.

    Args:
        df:    timeseries DataFrame (modified in place)
        audit: audit trail DataFrame (modified in place)

    Returns:
        Number of cells filled.
    """
    still_missing = df.isna()
    remaining_count = still_missing.sum().sum()

    if remaining_count == 0:
        return 0

    df[:] = df.ffill().bfill()

    # Mark newly filled cells in audit
    for tenor in df.columns:
        newly_filled = still_missing[tenor] & df[tenor].notna()
        audit.loc[newly_filled, tenor] = "FLAT"

    return remaining_count - df.isna().sum().sum()


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def write_output(
    df: pd.DataFrame,
    audit: pd.DataFrame,
    original_df: pd.DataFrame,
    output_path: str,
    fill_counts: dict,
    empty_tenors: list = None,
    sparse_tenors: list = None,
):
    """
    Write the backfilled data, audit trail, per-tenor breakdown, and summary
    to a formatted Excel workbook.

    Audit trail cells are colour-coded by method:
        Green  = Regression    |  Blue   = Spline
        Gold   = Linear        |  Salmon = Flat fill
        Red    = No data (entirely empty tenor)

    Args:
        df:            backfilled DataFrame
        audit:         audit trail DataFrame
        original_df:   original (pre-fill) DataFrame for comparison stats
        output_path:   file path for output Excel
        fill_counts:   dict mapping method name -> number of cells filled
        empty_tenors:  list of tenor names with zero observations
        sparse_tenors: list of (tenor_name, observation_count) tuples
    """
    empty_tenors = empty_tenors or []
    sparse_tenors = sparse_tenors or []
    tenors = df.columns.tolist()
    n_rows = len(df)

    # --- Per-tenor breakdown ---
    tenor_rows = []
    empty_set = set(empty_tenors)
    sparse_dict = dict(sparse_tenors)
    for tenor in tenors:
        orig_missing = original_df[tenor].isna().sum()

        # Flag the tenor's status for easy scanning
        if tenor in empty_set:
            status = "NO DATA"
        elif tenor in sparse_dict:
            status = f"SPARSE ({sparse_dict[tenor]} obs)"
        else:
            status = "OK"

        tenor_rows.append({
            "Tenor": tenor,
            "Status": status,
            "Total Dates": n_rows,
            "Originally Missing": orig_missing,
            "Missing %": round(orig_missing / n_rows * 100, 1),
            "Regression": audit[tenor].str.startswith("REG").sum(),
            "Spline": (audit[tenor] == "SPLINE").sum(),
            "Linear": (audit[tenor] == "LINEAR").sum(),
            "Flat": (audit[tenor] == "FLAT").sum(),
        })
    breakdown_df = pd.DataFrame(tenor_rows)

    # --- Summary ---
    total_cells = n_rows * len(tenors)
    total_missing = original_df.isna().sum().sum()
    summary_rows = [
        ("Total cells", total_cells),
        ("Originally missing", total_missing),
        ("Missing %", f"{total_missing / total_cells * 100:.1f}%"),
        ("", ""),
        ("Filled by regression", fill_counts.get("regression", 0)),
        ("Filled by cubic spline", fill_counts.get("spline", 0)),
        ("Filled by linear interpolation", fill_counts.get("linear", 0)),
        ("Filled by flat fill", fill_counts.get("flat", 0)),
        ("Still missing", int(df.isna().sum().sum())),
        ("", ""),
        ("Empty tenors (no data)", len(empty_tenors)),
        ("Sparse tenors (<4 obs)", len(sparse_tenors)),
        ("", ""),
        ("CONFIG: Min regression observations", MIN_REGRESSION_OBSERVATIONS),
        ("CONFIG: Min regression R²", MIN_REGRESSION_R_SQUARED),
        ("CONFIG: Min spline points", MIN_SPLINE_POINTS),
    ]

    # List the empty and sparse tenors explicitly in the summary
    if empty_tenors:
        summary_rows.append(("", ""))
        summary_rows.append(("EMPTY TENORS (unfillable)", ""))
        for t in empty_tenors:
            summary_rows.append(("", t))

    if sparse_tenors:
        summary_rows.append(("", ""))
        summary_rows.append(("SPARSE TENORS", ""))
        for t, n in sparse_tenors:
            summary_rows.append(("", f"{t} ({n} observations)"))

    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    # --- Write all sheets ---
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Backfilled Data")
        audit.to_excel(writer, sheet_name="Audit Trail")
        breakdown_df.to_excel(writer, sheet_name="Per Tenor Breakdown", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # --- Apply formatting ---
    wb = load_workbook(output_path)
    _format_audit_sheet(wb["Audit Trail"])
    _format_all_headers(wb)
    wb.save(output_path)


def _format_audit_sheet(ws):
    """Colour-code audit trail cells and add a legend."""
    colour_map = {
        "REG":     PatternFill("solid", fgColor="90EE90"),  # green
        "SPLINE":  PatternFill("solid", fgColor="87CEEB"),  # blue
        "LINEAR":  PatternFill("solid", fgColor="FFD700"),  # gold
        "FLAT":    PatternFill("solid", fgColor="FFA07A"),  # salmon
        "NO DATA": PatternFill("solid", fgColor="FF6666"),  # red — no data at all
    }

    for row in ws.iter_rows(min_row=2, min_col=2):
        for cell in row:
            cell_text = str(cell.value) if cell.value else ""
            for method_prefix, fill in colour_map.items():
                if cell_text.startswith(method_prefix):
                    cell.fill = fill
                    break

    # Legend below the data
    legend_row = ws.max_row + 3
    ws.cell(row=legend_row, column=1, value="Legend:").font = Font(bold=True, name="Arial")

    legend_entries = [
        ("REG",     "Cross-tenor regression (R² shown)",  "90EE90"),
        ("SPLINE",  "Cubic spline interpolation",         "87CEEB"),
        ("LINEAR",  "Linear interpolation",               "FFD700"),
        ("FLAT",    "Forward/backward fill (edges only)", "FFA07A"),
        ("NO DATA", "Tenor has zero observations",        "FF6666"),
        ("(blank)", "Original observed data",             "FFFFFF"),
    ]
    for i, (code, description, hex_colour) in enumerate(legend_entries):
        r = legend_row + 1 + i
        ws.cell(row=r, column=1, value=code).fill = PatternFill("solid", fgColor=hex_colour)
        ws.cell(row=r, column=2, value=description)


def _format_all_headers(wb):
    """Apply consistent header formatting across all sheets."""
    header_font = Font(bold=True, name="Arial", size=11)
    header_fill = PatternFill("solid", fgColor="D9E1F2")

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Entry point: load data, run the four-stage waterfall, and write results.
    """
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = input_path.replace(".xlsx", "_backfilled.xlsx")

    # Load
    print(f"Loading {input_path}...")
    df = load_timeseries(input_path)
    original_df = df.copy()  # preserve original for comparison

    n_rows, n_cols = df.shape
    total_missing = df.isna().sum().sum()
    print(f"  {n_rows} dates × {n_cols} tenors")
    print(f"  {total_missing} missing cells ({total_missing / (n_rows * n_cols) * 100:.1f}%)\n")

    # ── Pre-flight check: detect empty and sparse tenors ──
    empty_tenors = []   # zero observations — cannot be filled by any method
    sparse_tenors = []  # 1–3 observations — too few for spline, limited interpolation

    for tenor in df.columns:
        n_observed = df[tenor].notna().sum()
        if n_observed == 0:
            empty_tenors.append(tenor)
        elif n_observed < MIN_SPLINE_POINTS:
            sparse_tenors.append((tenor, n_observed))

    if empty_tenors:
        print(f"  ⚠ WARNING: {len(empty_tenors)} tenor(s) have NO data at all:")
        for t in empty_tenors:
            print(f"      - {t}  (will remain entirely empty in output)")
        print()

    if sparse_tenors:
        print(f"  ⚠ WARNING: {len(sparse_tenors)} tenor(s) have very few observations:")
        for t, n in sparse_tenors:
            print(f"      - {t}  ({n} values — too few for spline, will use linear/flat only)")
        print()

    # Audit trail: empty string = original data, otherwise = method code
    audit = pd.DataFrame("", index=df.index, columns=df.columns)

    # Mark empty tenors in audit so they are clearly flagged in the output
    for tenor in empty_tenors:
        audit[tenor] = "NO DATA"

    # Run waterfall
    print("Priority 1 — Cross-tenor regression...")
    n_reg = fill_via_regression(df, audit)
    print(f"  → {n_reg} cells filled\n")

    print("Priority 2 — Cubic spline interpolation...")
    n_spline = fill_via_spline(df, audit)
    print(f"  → {n_spline} cells filled\n")

    print("Priority 3 — Linear interpolation...")
    n_linear = fill_via_linear(df, audit)
    print(f"  → {n_linear} cells filled\n")

    print("Priority 4 — Forward/backward fill (edges only)...")
    n_flat = fill_via_flat(df, audit)
    print(f"  → {n_flat} cells filled\n")

    # Results
    still_missing = df.isna().sum().sum()
    print(f"Complete: {total_missing} missing → {int(still_missing)} remaining")
    if still_missing > 0:
        unfillable = sum(n_rows for t in empty_tenors)
        fillable_remaining = int(still_missing) - unfillable
        print(f"  ({unfillable} unfillable from empty tenors, {fillable_remaining} other)")
    print()

    # Write
    print(f"Writing {output_path}...")
    fill_counts = {
        "regression": n_reg,
        "spline": n_spline,
        "linear": n_linear,
        "flat": n_flat,
    }
    write_output(df, audit, original_df, output_path, fill_counts, empty_tenors, sparse_tenors)
    print(f"Done → {output_path}")


if __name__ == "__main__":
    main()
```

---

## Appendix C — Phase 0.4: Risk Management page skeleton (copy-paste sync)

> **Status**: Phase 0.4 ships a new top-level **Risk Management** page —
> a peer of `/overview`, `/evaluation`, `/inference`, etc. — with a
> real-data subset of Mock B (the agreed canonical layout in
> `rade_risk_management.png`). Phase 4 then *upgrades* the same panels to
> true RF-level granularity once Phase 2's trade-attribute API lands.
>
> Three new files, three modified files. No backend / API changes —
> the page is a pure consumer of the existing
> `GET /inference/runs/{run_id}/portfolio` and `…/clusters` endpoints
> already exposed by Stage 10.

### What ships in Phase 0.4

| Panel | Data source | Real today? | Phase 4 upgrade |
|---|---|:---:|---|
| KPIs: VaR / CVaR / Skewness / Excess kurtosis | `portfolio_predictions.parquet` | ✅ | unchanged |
| Cluster waterfall (worst scenario) | `cluster_predictions.parquet` × `sum_pnl_original` | ✅ | swap to RF-level + scenario picker |
| Cluster tornado | aggregate by cluster | ✅ | swap to RF tornado |
| Cluster donut (single-ring) | aggregate by cluster | ✅ | three-ring sunburst (RF group → trade type → cluster) |
| Tail table — Scenario / Date / PnL / Top cluster / VaR contrib | `portfolio_predictions.parquet` + per-scenario argmin on `clusters_df` | ✅ | adds `Top RF` and `Top trade type` columns |

### Files added (full source below)

* `src/ui/apps/rade_analytics/figures/risk_management_charts.py` — figure builders (§C.1)
* `src/ui/apps/rade_analytics/layouts/risk_management.py` — page layout (§C.2)
* `src/ui/apps/rade_analytics/callbacks/risk_management_cb.py` — callbacks (§C.3)

### Files modified (wiring diffs in §C.4)

* `src/ui/apps/rade_analytics/router.py` — new `/risk-management` route
* `src/ui/apps/rade_analytics/components/sidebar.py` — new nav entry
* `src/ui/apps/rade_analytics/callbacks/__init__.py` — register `risk_management_cb`

### Behaviour

On mount the page calls `backend.list_inference_runs()`, picks the
**most recent run**, writes its meta into a page-local `dcc.Store`,
and a single hydration callback fetches the portfolio + cluster
parquets to paint every panel. The waterfall auto-pins to the
**worst portfolio loss** scenario in the run. No run-picker or
scenario-picker controls in Phase 0 — those land in Phase 4.

When there are no inference runs the subtitle reads
*"No inference runs available yet — run the Inference Console to
populate this page."* and every panel stays in its empty state.

---

### C.1 — `src/ui/apps/rade_analytics/figures/risk_management_charts.py` (NEW)

```python
"""Plotly figure builders for the Risk Management page (Phase 0.4).

Three "real" builders cover the page's hero panels:

* :func:`build_cluster_waterfall` — scenario-pinned P&L decomposition
  by cluster.  Each bar = one cluster's contribution to that
  scenario's portfolio total.  Phase 4 swaps in a risk-factor-level
  waterfall once trade attributes land.
* :func:`build_cluster_tornado` — clusters ranked by absolute
  contribution across all scenarios.  Same visual idiom as the
  Inference tab's risk-attribution-by-cluster but sorted by ``|x|``
  instead of signed ``x`` so the biggest movers (positive or
  negative) sit at the top of the chart.
* :func:`build_cluster_donut` — single-ring breakdown of |signed
  contribution| per cluster, with sign-coloured slices.  Phase 4
  upgrades this into a three-ring sunburst (RF group → trade type →
  cluster).

Every figure is run through :func:`figures._theme.rade_layout` so the
violet → cyan / amber / rose palette stays consistent with the
Inference and Overview pages.

The ``empty_*`` builders return placeholders that the layout uses at
mount time (before the auto-load callback has fetched real data).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import empty_figure, rade_layout, rgba


# ─────────────────────────────────────────────────────────────────────
# Palette aliases (kept local so chart code reads naturally; canonical
# hex values live in ``_theme.CATEGORY_PALETTE``).
# ─────────────────────────────────────────────────────────────────────

_VIOLET:    str = "#8b5cf6"
_EMERALD:   str = "#10b981"
_ROSE:      str = "#f43f5e"
_AMBER:     str = "#f59e0b"
_ZERO_LINE: str = "#475569"
_SLATE:     str = "#64748b"

# How many bars the cluster-waterfall keeps visible by default.  The
# user typically cares about the top-N magnitude contributors; the
# remainder are folded into an "Other" terminal bar so the chart
# stays readable on a ~25-cluster portfolio.
_WATERFALL_TOP_N: int = 7


# ═════════════════════════════════════════════════════════════════════
# Empty-state builders — kept as public API so the layout can paint
# placeholders before data is fetched.
# ═════════════════════════════════════════════════════════════════════

def empty_cluster_waterfall() -> go.Figure:
    """Placeholder for the cluster-level P&L waterfall."""
    return empty_figure("Awaiting inference run — P&L decomposition")


def empty_cluster_tornado() -> go.Figure:
    """Placeholder for the cluster-level tornado chart."""
    return empty_figure("Awaiting inference run — cluster sensitivity")


def empty_cluster_donut() -> go.Figure:
    """Placeholder for the cluster-level contribution donut."""
    return empty_figure("Awaiting inference run — contribution donut")


# ═════════════════════════════════════════════════════════════════════
# Shared adapters
# ═════════════════════════════════════════════════════════════════════

def _portfolio_pnl(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """Pull the canonical portfolio-PnL series from a frame.

    Returns ``None`` if the frame is empty / missing the expected
    column, so callers can fall back to the matching empty figure.
    """
    if df is None or df.empty or "sum_pnl_original" not in df.columns:
        return None
    return df["sum_pnl_original"].astype(float)


def _aggregate_cluster_contributions(
    clusters_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Aggregate per-cluster signed contribution across all scenarios.

    Returns a frame with two columns (``cluster_id``,
    ``sum_pnl_original``) or ``None`` when the input frame is empty
    or lacking the required columns.
    """
    if clusters_df is None or clusters_df.empty:
        return None
    if not {"cluster_id", "sum_pnl_original"}.issubset(clusters_df.columns):
        return None
    return (
        clusters_df.groupby("cluster_id", as_index=False)["sum_pnl_original"]
        .sum()
    )


# ═════════════════════════════════════════════════════════════════════
# 1. Cluster waterfall — scenario-pinned P&L decomposition
# ═════════════════════════════════════════════════════════════════════

def build_cluster_waterfall(
    clusters_df:    Optional[pd.DataFrame],
    scenario_label: Optional[str] = None,
    *,
    top_n:          int = _WATERFALL_TOP_N,
) -> go.Figure:
    """P&L waterfall for a single scenario, broken down by cluster.

    The chart shows how each cluster's contribution to the chosen
    scenario stacks up step-by-step to the portfolio total.  Top
    ``top_n`` clusters (by absolute contribution) get individual
    bars; everything else is folded into a single "Other" bar so
    the chart stays legible on 25+ cluster portfolios.

    When ``scenario_label`` is None or not present in the data, the
    builder defaults to the **worst** scenario — i.e. the scenario
    with the largest negative portfolio PnL — which is almost always
    what the user wants to land on first.

    Parameters
    ----------
    clusters_df
        Long-format cluster summary parquet (one row per cluster ×
        scenario), with at least ``cluster_id``, ``scenario_label``,
        ``sum_pnl_original`` columns.
    scenario_label
        Scenario to decompose.  None → worst portfolio PnL.
    top_n
        How many top-magnitude clusters keep their own bar.
    """
    if clusters_df is None or clusters_df.empty:
        return empty_cluster_waterfall()
    needed = {"cluster_id", "scenario_label", "sum_pnl_original"}
    if not needed.issubset(clusters_df.columns):
        return empty_cluster_waterfall()

    # Resolve scenario_label — default to the worst portfolio loss.
    portfolio_by_scenario = (
        clusters_df.groupby("scenario_label", as_index=False)[
            "sum_pnl_original"
        ].sum()
    )
    if scenario_label is None or scenario_label not in set(portfolio_by_scenario["scenario_label"]):
        worst_idx = portfolio_by_scenario["sum_pnl_original"].idxmin()
        scenario_label = str(portfolio_by_scenario.loc[worst_idx, "scenario_label"])

    sub = clusters_df.loc[
        clusters_df["scenario_label"].astype(str) == str(scenario_label),
        ["cluster_id", "sum_pnl_original"],
    ].copy()
    if sub.empty:
        return empty_cluster_waterfall()
    sub["sum_pnl_original"] = sub["sum_pnl_original"].astype(float)

    # Rank by |contribution| and split top_n vs rest.
    sub = sub.assign(abs_pnl=sub["sum_pnl_original"].abs()).sort_values(
        "abs_pnl", ascending=False,
    )
    top    = sub.head(top_n)
    rest   = sub.iloc[top_n:]
    other  = float(rest["sum_pnl_original"].sum()) if not rest.empty else 0.0
    total  = float(sub["sum_pnl_original"].sum())

    # Keep top bars in their |contribution| order so the biggest
    # movers land at the LEFT (most-read position).
    cluster_labels  = list(top["cluster_id"].astype(str))
    cluster_values  = list(top["sum_pnl_original"].astype(float))
    if not rest.empty:
        cluster_labels.append(f"Other ({len(rest)} clusters)")
        cluster_values.append(other)

    # Waterfall measure: every cluster bar is relative; final bar is total.
    measures = ["relative"] * len(cluster_values) + ["total"]
    x_labels = [*cluster_labels, "Portfolio PnL"]
    y_values = [*cluster_values, total]
    text     = [f"{v:,.0f}" for v in y_values]

    fig = go.Figure()
    fig.add_trace(go.Waterfall(
        name           = scenario_label,
        orientation    = "v",
        measure        = measures,
        x              = x_labels,
        y              = y_values,
        text           = text,
        textposition   = "outside",
        textfont       = {"size": 10, "color": "#e2e8f0"},
        connector      = {"line": {"color": rgba(_ZERO_LINE, 0.6)}},
        increasing     = {"marker": {"color": _EMERALD}},
        decreasing     = {"marker": {"color": _ROSE}},
        totals         = {"marker": {"color": _VIOLET}},
        hovertemplate  = "%{x}<br>Contribution: %{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(
        y=0,
        line={"color": _ZERO_LINE, "width": 1, "dash": "dot"},
    )
    fig.update_layout(**rade_layout(
        hovermode = "x",
        margin    = {"l": 36, "r": 16, "t": 20, "b": 56},
        xaxis     = {"title": "", "tickangle": -25, "automargin": True},
        yaxis     = {"title": "PnL contribution (original units)"},
    ))
    return fig


# ═════════════════════════════════════════════════════════════════════
# 2. Cluster tornado — magnitude-ranked sensitivity bars
# ═════════════════════════════════════════════════════════════════════

def build_cluster_tornado(
    clusters_df: Optional[pd.DataFrame],
    *,
    top_n:       int = 12,
) -> go.Figure:
    """Horizontal bar chart of clusters ranked by |contribution|.

    Same data source as the cluster donut but sorted by absolute
    value (descending), so the biggest movers — regardless of sign —
    land at the top of the chart.  Phase 4 swaps this for a true
    risk-factor tornado once trade-RF mappings exist.

    Parameters
    ----------
    clusters_df
        Long-format cluster summary parquet.
    top_n
        Cap on visible bars.  Beyond this, the rest are aggregated
        into an "Other" bar so the chart stays scannable.
    """
    agg = _aggregate_cluster_contributions(clusters_df)
    if agg is None:
        return empty_cluster_tornado()

    agg = agg.assign(abs_pnl=agg["sum_pnl_original"].abs()).sort_values(
        "abs_pnl", ascending=False,
    )
    top  = agg.head(top_n)
    rest = agg.iloc[top_n:]
    rows = list(top[["cluster_id", "sum_pnl_original"]].itertuples(index=False))
    if not rest.empty:
        rows.append(("Other (%d)" % len(rest), float(rest["sum_pnl_original"].sum())))

    # Plotly horizontal bars draw bottom-up; reverse so biggest is on top.
    rows = list(reversed(rows))
    y_labels = [str(r[0]) for r in rows]
    x_values = [float(r[1]) for r in rows]
    colors   = [_EMERALD if v >= 0 else _ROSE for v in x_values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x             = x_values,
        y             = y_labels,
        orientation   = "h",
        marker        = {"color": colors, "opacity": 0.9},
        text          = [f"{v:,.0f}" for v in x_values],
        textposition  = "outside",
        textfont      = {"size": 10, "color": "#cbd5e1"},
        hovertemplate = "%{y}<br>Contribution: %{x:,.2f}<extra></extra>",
    ))
    fig.add_vline(
        x=0,
        line={"color": _ZERO_LINE, "width": 1, "dash": "dot"},
    )
    fig.update_layout(**rade_layout(
        hovermode = "y",
        margin    = {"l": 140, "r": 56, "t": 8, "b": 36},
        xaxis     = {"title": "Contribution (original units)"},
        yaxis     = {"title": "", "automargin": True},
    ))
    return fig


# ═════════════════════════════════════════════════════════════════════
# 3. Cluster donut — single-ring contribution breakdown
# ═════════════════════════════════════════════════════════════════════

def build_cluster_donut(
    clusters_df: Optional[pd.DataFrame],
    *,
    top_n:       int = 8,
) -> go.Figure:
    """Donut of |contribution| share per cluster, slice-coloured by sign.

    Phase 0 deliberately stays single-ring; Phase 4 upgrades to the
    full RF-group / trade-type / cluster sunburst once Phase 2's
    trade attribute API lands.

    Parameters
    ----------
    clusters_df
        Long-format cluster summary parquet.
    top_n
        Slice count cap.  The rest are folded into a single "Other"
        slice so the chart doesn't degenerate into a hairline pie on
        large portfolios.
    """
    agg = _aggregate_cluster_contributions(clusters_df)
    if agg is None:
        return empty_cluster_donut()

    agg = agg.assign(abs_pnl=agg["sum_pnl_original"].abs()).sort_values(
        "abs_pnl", ascending=False,
    )
    top  = agg.head(top_n)
    rest = agg.iloc[top_n:]

    labels:  list[str]   = list(top["cluster_id"].astype(str))
    signed:  list[float] = list(top["sum_pnl_original"].astype(float))
    values:  list[float] = list(top["abs_pnl"].astype(float))
    if not rest.empty:
        labels.append(f"Other ({len(rest)})")
        signed.append(float(rest["sum_pnl_original"].sum()))
        values.append(float(rest["abs_pnl"].sum()))

    if not values or sum(values) <= 0:
        return empty_cluster_donut()

    colors = [_EMERALD if s >= 0 else _ROSE for s in signed]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels        = labels,
        values        = values,
        customdata    = [[s] for s in signed],
        hole          = 0.55,
        sort          = False,
        marker        = {"colors": colors, "line": {"color": "#0f172a", "width": 1}},
        textinfo      = "label+percent",
        textfont      = {"size": 11, "color": "#e2e8f0"},
        hovertemplate = (
            "%{label}<br>"
            "|Contribution|: %{value:,.2f}<br>"
            "Signed: %{customdata[0]:,.2f}"
            "<extra></extra>"
        ),
    ))
    total_signed = float(sum(signed))
    fig.update_layout(
        **rade_layout(
            show_legend = False,
            margin      = {"l": 8, "r": 8, "t": 8, "b": 8},
        ),
        annotations = [
            {
                "text":      (
                    f"<b>{total_signed:,.0f}</b><br>"
                    f"<span style='font-size:11px;color:#94a3b8'>"
                    f"net contribution</span>"
                ),
                "showarrow": False,
                "x":         0.5, "y": 0.5,
                "font":      {"size": 18, "color": "#e2e8f0"},
            },
        ],
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
# 4. KPI helpers (stats only — the tile DOM is built in the callback)
# ═════════════════════════════════════════════════════════════════════

def compute_tail_stats(
    portfolio_df: Optional[pd.DataFrame],
    *,
    var_quantile: float = 0.05,
) -> dict[str, Optional[float]]:
    """Compute VaR / CVaR / skew / excess-kurtosis on portfolio PnL.

    Single-shot helper so the callback doesn't have to duplicate the
    quantile / mean / scipy-equivalents arithmetic.  Returns a dict
    keyed by metric name with ``None`` for any metric we can't
    derive from the input (e.g. empty frame).
    """
    pnl = _portfolio_pnl(portfolio_df)
    if pnl is None or pnl.empty:
        return {"var": None, "cvar": None, "skew": None, "kurt": None}

    var_value  = float(pnl.quantile(var_quantile))
    mask       = pnl <= var_value
    cvar_value = float(pnl[mask].mean()) if mask.any() else var_value
    # Pandas' ``skew`` / ``kurtosis`` use the bias-corrected
    # Fisher-Pearson estimators — same convention as scipy.stats so
    # downstream comparisons match.  Excess kurtosis ⇒ Gaussian = 0.
    skew_value = float(pnl.skew()) if len(pnl) > 2 else None
    kurt_value = float(pnl.kurtosis()) if len(pnl) > 3 else None

    return {
        "var":  var_value,
        "cvar": cvar_value,
        "skew": skew_value,
        "kurt": kurt_value,
    }


def compute_tail_table(
    portfolio_df: Optional[pd.DataFrame],
    clusters_df:  Optional[pd.DataFrame],
    *,
    n:            int = 10,
    var_quantile: float = 0.05,
) -> pd.DataFrame:
    """Build the tail-conditional table — the worst-N scenarios.

    Columns (Phase 0 — what we can compute without Phase 2 attrs):

    * ``scenario_label`` — scenario identifier (string).
    * ``portfolio_pnl`` — total PnL for the scenario.
    * ``top_cluster`` — cluster with the largest negative contribution.
    * ``top_cluster_pnl`` — that cluster's contribution to this scenario.
    * ``var_contribution_pct`` — |scenario_pnl| / |CVaR| × 100, for
      colour-bar shading in the UI.

    Phase 4 swaps in ``top_rf`` and ``top_trade_type`` columns once
    the trade-attribute API ships.
    """
    pnl = _portfolio_pnl(portfolio_df)
    if pnl is None or pnl.empty or "scenario_label" not in portfolio_df.columns:
        return pd.DataFrame(columns=[
            "scenario_label", "portfolio_pnl",
            "top_cluster", "top_cluster_pnl",
            "var_contribution_pct",
        ])

    portfolio_sub = portfolio_df[["scenario_label", "sum_pnl_original"]].copy()
    portfolio_sub.columns = ["scenario_label", "portfolio_pnl"]
    portfolio_sub = portfolio_sub.nsmallest(n, "portfolio_pnl")

    # Per-scenario worst cluster (most negative contribution).
    top_cluster_lookup: dict[str, tuple[str, float]] = {}
    if clusters_df is not None and not clusters_df.empty and {
        "cluster_id", "scenario_label", "sum_pnl_original",
    }.issubset(clusters_df.columns):
        cl = clusters_df[["cluster_id", "scenario_label", "sum_pnl_original"]]
        cl = cl[cl["scenario_label"].isin(set(portfolio_sub["scenario_label"]))]
        # idxmin per scenario → cluster_id with smallest sum_pnl_original.
        for label, grp in cl.groupby("scenario_label", sort=False):
            row = grp.loc[grp["sum_pnl_original"].idxmin()]
            top_cluster_lookup[str(label)] = (
                str(row["cluster_id"]),
                float(row["sum_pnl_original"]),
            )

    portfolio_sub["top_cluster"] = portfolio_sub["scenario_label"].map(
        lambda lbl: top_cluster_lookup.get(str(lbl), ("—", np.nan))[0]
    )
    portfolio_sub["top_cluster_pnl"] = portfolio_sub["scenario_label"].map(
        lambda lbl: top_cluster_lookup.get(str(lbl), ("—", np.nan))[1]
    )

    # VaR-contribution % — gives the UI's mini-bar a 0–100 scale.
    var_value = float(pnl.quantile(var_quantile))
    cvar_mask = pnl <= var_value
    cvar_abs  = abs(float(pnl[cvar_mask].mean())) if cvar_mask.any() else abs(var_value)
    cvar_abs  = cvar_abs if cvar_abs > 0 else 1.0
    portfolio_sub["var_contribution_pct"] = (
        portfolio_sub["portfolio_pnl"].abs() / cvar_abs * 100.0
    ).clip(lower=0.0, upper=200.0)

    return portfolio_sub.reset_index(drop=True)


__all__ = [
    # Dispatchers
    "build_cluster_waterfall",
    "build_cluster_tornado",
    "build_cluster_donut",
    # Empty-state builders
    "empty_cluster_waterfall",
    "empty_cluster_tornado",
    "empty_cluster_donut",
    # KPI/stats helpers
    "compute_tail_stats",
    "compute_tail_table",
]
```

---

### C.2 — `src/ui/apps/rade_analytics/layouts/risk_management.py` (NEW)

```python
"""Risk Management page layout (Phase 0.4 skeleton).

Top-level page for tail-risk decomposition and shock attribution.
Phase 0 ships a real-data subset (KPIs, cluster waterfall, tornado,
donut, tail table); Phase 4 then *upgrades* the same panels to true
risk-factor / trade-type granularity once Phase 2's trade-attribute
API lands.

Page anatomy
------------

* **Page header** — title + subtitle (page-mount-time copy).
* **Run-info banner** — subtitle line populated with the
  most-recent run ID / ensemble version once the auto-load
  callback resolves the run.
* **KPI strip** — four tiles (VaR / CVaR / Skewness /
  Excess-kurtosis) computed from ``portfolio_predictions.parquet``.
* **Cluster waterfall** — scenario-pinned P&L decomposition.
  Defaults to the worst portfolio scenario; Phase 4 adds a scenario
  picker.
* **Tornado + donut grid** — clusters ranked by |contribution| on
  the left, single-ring contribution donut on the right.
* **Tail-conditional table** — top-N worst scenarios with
  cluster-level enrichment columns.  ``top_rf`` / ``top_trade_type``
  columns are placeholdered (``—``) until Phase 4 lands those data.
* **Footer** — source manifest path (mirrors the Inference tab
  footer line for visual continuity).

Data flow
---------

Single mount-time callback (``callbacks/risk_management_cb._on_mount``)
fetches the most-recent inference run via
``backend.list_inference_runs()`` and writes its ``run_id`` into
:data:`RISK_MANAGEMENT_IDS["run_meta_store"]`.  Every panel then
hydrates from a follow-up callback that fetches portfolio + cluster
parquets and renders KPIs / figures / table.

Design-spec anchors
-------------------
* §1 — typography & palette (Inter UI text, JetBrains-Mono numerics,
  violet → cyan brand).
* §5 — page anatomy (KPI strip on top, hero chart, supporting grid,
  tabular content, footer caption).
* §6 — ``rade-card`` / ``rade-stress-mini-kpi`` shared utilities.
* §11 — top-level nav taxonomy (Risk Management is a peer of
  Overview, Evaluation, Inference, etc.).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ..components.chart_container import ChartContainer
from ..figures.risk_management_charts import (
    empty_cluster_donut,
    empty_cluster_tornado,
    empty_cluster_waterfall,
)

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────
# DOM identifiers — every dynamic id used by the page lives here so
# callbacks never hard-code strings.
# ─────────────────────────────────────────────────────────────────────

RISK_MANAGEMENT_IDS: Dict[str, str] = {
    "root":                 "risk-management-root",
    "subtitle":             "risk-management-subtitle",

    # Page-wide Store: ``{run_id, ensemble_version, generated_at, ...}``.
    # Hydrated on page mount; every panel keys off it.
    "run_meta_store":       "risk-management-run-meta-store",

    # KPI tiles (4-card row at top of page).
    "kpi_var_value":        "risk-management-kpi-var-value",
    "kpi_cvar_value":       "risk-management-kpi-cvar-value",
    "kpi_skew_value":       "risk-management-kpi-skew-value",
    "kpi_kurt_value":       "risk-management-kpi-kurt-value",

    # Hero chart — cluster-level P&L waterfall for the worst scenario.
    "waterfall_chart":      "risk-management-waterfall-chart",
    "waterfall_caption":    "risk-management-waterfall-caption",

    # Supporting grid — magnitude-ranked tornado + signed donut.
    "tornado_chart":        "risk-management-tornado-chart",
    "donut_chart":          "risk-management-donut-chart",

    # Tail-conditional table — populated by the callback.
    "tail_table_container": "risk-management-tail-table-container",

    # Footer — source manifest line.
    "footer_caption":       "risk-management-footer-caption",
}


_PLACEHOLDER: str = "—"


_PAGE_SUBTITLE_DEFAULT: str = (
    "Tail-risk decomposition & shock attribution — auto-loaded "
    "from the most recent inference run."
)


# ─────────────────────────────────────────────────────────────────────
# KPI strip — small label-over-value tiles (NOT the spark KpiCard).
# ─────────────────────────────────────────────────────────────────────


def _kpi_tile(label: str, value_id: str) -> html.Div:
    """One KPI tile (label + big mono value).

    Uses ``rade-stress-mini-kpi`` styling for visual continuity with
    the Inference page's stress tiles; the parent grid lays four of
    them in a row.
    """
    return html.Div(
        className="rade-stress-mini-kpi",
        children=[
            html.Div(label.upper(), className="rade-stress-mini-label"),
            html.Div(
                _PLACEHOLDER,
                id=value_id,
                className="rade-stress-mini-value font-mono",
            ),
        ],
    )


def _kpi_strip() -> html.Div:
    return html.Div(
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3",
        children=[
            _kpi_tile("VaR (95%)",     RISK_MANAGEMENT_IDS["kpi_var_value"]),
            _kpi_tile("CVaR (95%)",    RISK_MANAGEMENT_IDS["kpi_cvar_value"]),
            _kpi_tile("Skewness",      RISK_MANAGEMENT_IDS["kpi_skew_value"]),
            _kpi_tile("Excess kurtosis", RISK_MANAGEMENT_IDS["kpi_kurt_value"]),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Cards — waterfall (hero), tornado + donut (supporting), tail table
# ─────────────────────────────────────────────────────────────────────


def _waterfall_card() -> html.Div:
    return html.Div(
        className="flex flex-col gap-2",
        children=[
            ChartContainer(
                title="P&L decomposition — by cluster",
                subtitle=(
                    "How each cluster contributes to the portfolio total "
                    "for the worst scenario in the run.  Phase 4 swaps "
                    "this for a risk-factor-level waterfall."
                ),
                graph_id=RISK_MANAGEMENT_IDS["waterfall_chart"],
                figure=empty_cluster_waterfall(),
                height=360,
                actions=[
                    html.Div(
                        id=RISK_MANAGEMENT_IDS["waterfall_caption"],
                        className="text-[11px] text-slate-500 italic",
                        children="Awaiting run — scenario pin appears once data loads.",
                    ),
                ],
            ),
        ],
    )


def _tornado_card() -> html.Div:
    return ChartContainer(
        title="Cluster sensitivity",
        subtitle=(
            "Clusters ranked by absolute contribution across all "
            "scenarios.  Sign-coloured: emerald = positive, rose = "
            "negative."
        ),
        graph_id=RISK_MANAGEMENT_IDS["tornado_chart"],
        figure=empty_cluster_tornado(),
        height=360,
    )


def _donut_card() -> html.Div:
    return ChartContainer(
        title="Contribution share",
        subtitle=(
            "|Contribution| share per cluster; signed colour. "
            "Phase 4 nests this into a three-ring "
            "RF-group → trade-type → cluster sunburst."
        ),
        graph_id=RISK_MANAGEMENT_IDS["donut_chart"],
        figure=empty_cluster_donut(),
        height=360,
    )


def _tornado_donut_row() -> html.Div:
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-2 gap-3",
        children=[
            html.Div(className="min-w-0", children=[_tornado_card()]),
            html.Div(className="min-w-0", children=[_donut_card()]),
        ],
    )


def _tail_table_placeholder() -> List[Any]:
    return [
        html.Div("Tail-conditional table",
                 className="text-sm font-semibold text-slate-200"),
        html.Div(
            "Scenarios at or below the 5th-percentile portfolio loss. "
            "``Top RF`` and ``Top trade type`` columns light up in "
            "Phase 4 once trade attributes are exposed.",
            className="text-xs text-slate-500",
        ),
        html.Div(
            "Awaiting inference run — the table will appear here.",
            className="text-xs text-slate-500 italic px-2 py-3",
        ),
    ]


def _tail_table_card() -> html.Div:
    return html.Div(
        id=RISK_MANAGEMENT_IDS["tail_table_container"],
        className="rade-card flex flex-col gap-2 min-w-0",
        children=_tail_table_placeholder(),
    )


# ─────────────────────────────────────────────────────────────────────
# Page composition
# ─────────────────────────────────────────────────────────────────────


def _page_header() -> html.Div:
    """Title row mirrored on every Rade top-level page."""
    return html.Div(
        className="flex flex-col gap-1 mb-2",
        children=[
            html.Div(
                "Risk Management",
                className="text-2xl font-semibold text-slate-100",
            ),
            html.Div(
                _PAGE_SUBTITLE_DEFAULT,
                id=RISK_MANAGEMENT_IDS["subtitle"],
                className="text-sm text-slate-400",
            ),
        ],
    )


def _page_footer() -> html.Div:
    """Subtle source-of-truth line at page bottom.

    Hydrated by the auto-load callback to show the inference run's
    manifest path; placeholder text until then.
    """
    return html.Div(
        className="flex justify-end mt-6",
        children=[
            html.Span(
                "Source: awaiting inference run …",
                id=RISK_MANAGEMENT_IDS["footer_caption"],
                className="text-xs text-slate-500 font-mono",
            ),
        ],
    )


def _page_stores() -> List[Any]:
    """``dcc.Store`` instances driving the page's state."""
    return [
        dcc.Store(
            id=RISK_MANAGEMENT_IDS["run_meta_store"],
            data=None,
            storage_type="memory",
        ),
    ]


def build_risk_management(
    *,
    session: Optional["Session"] = None,
) -> html.Div:
    """Compose the Risk Management page.

    Follows Page Contract §2.1 (uniform ``build_*(session=...)``
    signature); ``session`` is accepted for parity with every other
    page builder and is currently unused — the page hydrates entirely
    from the API rather than the session store.
    """
    del session  # unused — page loads its own most-recent run

    return html.Div(
        id=RISK_MANAGEMENT_IDS["root"],
        className="rade-page flex flex-col gap-4 p-4",
        children=[
            *_page_stores(),
            _page_header(),
            _kpi_strip(),
            _waterfall_card(),
            _tornado_donut_row(),
            _tail_table_card(),
            _page_footer(),
        ],
    )


__all__ = [
    "RISK_MANAGEMENT_IDS",
    "build_risk_management",
]
```

---

### C.3 — `src/ui/apps/rade_analytics/callbacks/risk_management_cb.py` (NEW)

```python
"""Risk Management page callbacks (Phase 0.4 skeleton).

Two callbacks drive the page:

1. **on_mount** — fires when the URL hits ``/risk-management``.
   Resolves the most-recent inference run via
   ``backend.list_inference_runs()`` and writes its meta into
   :data:`RISK_MANAGEMENT_IDS["run_meta_store"]`.  Pre-empts the
   page render's empty-state with a real run_id reference.

2. **hydrate** — fires on ``run_meta_store`` writes.  Fetches the
   manifest + portfolio + cluster parquets, computes VaR / CVaR /
   skew / kurtosis from the portfolio frame, and emits real figures
   for the waterfall / tornado / donut, plus the tail-conditional
   table.

Auto-pinning
------------

Phase 0 deliberately ships *without* run-picker or scenario-picker
controls.  The page always auto-resolves to the most-recent run and
the worst-portfolio-loss scenario inside that run.  Phase 4 swaps
both for real interactive controls when the Risk Management tab
becomes a first-class drill-down surface.

Tail-table shape
----------------

The tail table renders five columns today (``Scenario``, ``Date``,
``Portfolio PnL``, ``Top cluster``, ``VaR contribution``); Phase 4
appends ``Top RF`` and ``Top trade type`` once trade attributes are
exposed by Phase 2.  The placeholder em-dash cells today hint at
where those values will appear so the user gets a stable visual
shape across the upgrade.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from dash import Input, Output, html
from dash.exceptions import PreventUpdate

from ..figures.risk_management_charts import (
    build_cluster_donut,
    build_cluster_tornado,
    build_cluster_waterfall,
    compute_tail_stats,
    compute_tail_table,
    empty_cluster_donut,
    empty_cluster_tornado,
    empty_cluster_waterfall,
)
from ..layouts.risk_management import RISK_MANAGEMENT_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

_PLACEHOLDER:        str = "—"
_RISK_MGMT_ROUTE:    str = "/risk-management"


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach the Risk Management callbacks to ``app``.

    Mirrors the Inference page's capture/render split: one callback
    writes the run-meta Store, a second hydrates every panel from it.
    """
    _register_on_mount(app, backend)
    _register_hydrate(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. on_mount — URL hits /risk-management → resolve most recent run
# ═════════════════════════════════════════════════════════════════════

def _register_on_mount(app: "Dash", backend: "RadeBackend") -> None:
    """Resolve the most-recent inference run on page mount.

    Idempotent on the API side (a no-op when no runs exist).  We
    surface the empty case to the user by leaving the placeholder
    subtitle in place; the hydrate callback won't fire either, so
    the page reads as "nothing to show yet" rather than crashing
    on missing data.
    """

    @app.callback(
        Output(RISK_MANAGEMENT_IDS["run_meta_store"], "data"),
        Output(RISK_MANAGEMENT_IDS["subtitle"],       "children"),
        Input(SHELL_IDS["url"],                       "pathname"),
        prevent_initial_call=False,
    )
    def _on_mount(
        pathname: Optional[str],
    ) -> Tuple[Optional[Dict[str, Any]], Any]:
        if pathname != _RISK_MGMT_ROUTE:
            raise PreventUpdate

        runs_res = backend.list_inference_runs()
        if not runs_res.ok or runs_res.data is None or not runs_res.data.runs:
            logger.info(
                "Risk Management mount: no inference runs available "
                "(error=%s).",
                runs_res.error,
            )
            return None, _empty_subtitle()

        latest = runs_res.data.runs[0]
        meta: Dict[str, Any] = {
            "run_id":           latest.run_id,
            "ensemble_version": latest.ensemble_version,
            "generated_at":     latest.generated_at,
            "n_scenarios":      latest.n_scenarios,
            "n_clusters":       latest.n_clusters,
            "manifest_path":    latest.manifest_path,
        }
        return meta, _populated_subtitle(meta)


def _empty_subtitle() -> str:
    return (
        "No inference runs available yet — run the Inference Console "
        "to populate this page."
    )


def _populated_subtitle(meta: Dict[str, Any]) -> Any:
    """Subtitle line shown once a run is resolved."""
    parts = [
        html.Span("Run ", className="text-slate-500"),
        html.Span(str(meta.get("run_id") or _PLACEHOLDER),
                  className="font-mono text-violet-300"),
        html.Span(" · ensemble ", className="text-slate-500"),
        html.Span(str(meta.get("ensemble_version") or _PLACEHOLDER),
                  className="font-mono text-slate-300"),
    ]
    n_scenarios = meta.get("n_scenarios")
    if n_scenarios:
        parts.extend([
            html.Span(" · ", className="text-slate-500"),
            html.Span(f"{int(n_scenarios):,} scenarios",
                      className="text-slate-300"),
        ])
    return html.Span(parts)


# ═════════════════════════════════════════════════════════════════════
# 2. hydrate — run_meta_store → KPIs + figures + tail table + footer
# ═════════════════════════════════════════════════════════════════════

def _register_hydrate(app: "Dash", backend: "RadeBackend") -> None:
    """Fetch portfolio + cluster parquets and paint every panel."""

    @app.callback(
        # KPI strip (4 values)
        Output(RISK_MANAGEMENT_IDS["kpi_var_value"],   "children"),
        Output(RISK_MANAGEMENT_IDS["kpi_cvar_value"],  "children"),
        Output(RISK_MANAGEMENT_IDS["kpi_skew_value"],  "children"),
        Output(RISK_MANAGEMENT_IDS["kpi_kurt_value"],  "children"),
        # Three figures
        Output(RISK_MANAGEMENT_IDS["waterfall_chart"], "figure"),
        Output(RISK_MANAGEMENT_IDS["tornado_chart"],   "figure"),
        Output(RISK_MANAGEMENT_IDS["donut_chart"],     "figure"),
        # Hero chart caption
        Output(RISK_MANAGEMENT_IDS["waterfall_caption"], "children"),
        # Tail table
        Output(RISK_MANAGEMENT_IDS["tail_table_container"], "children"),
        # Footer source line
        Output(RISK_MANAGEMENT_IDS["footer_caption"], "children"),
        Input(RISK_MANAGEMENT_IDS["run_meta_store"], "data"),
        prevent_initial_call=False,
    )
    def _hydrate(
        run_meta: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        empty_returns: Tuple[Any, ...] = (
            _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER,
            empty_cluster_waterfall(),
            empty_cluster_tornado(),
            empty_cluster_donut(),
            "Awaiting run — scenario pin appears once data loads.",
            _tail_table_placeholder(),
            "Source: awaiting inference run …",
        )

        if not run_meta:
            return empty_returns
        run_id = run_meta.get("run_id")
        if not run_id:
            return empty_returns

        # Data-plane fetches.  Both calls are cached at the backend
        # layer so a route flip back to the page doesn't trigger
        # re-reads.
        portfolio_res = backend.inference_portfolio_df(run_id)
        clusters_res  = backend.inference_clusters_df(run_id)

        if not portfolio_res.ok or portfolio_res.data is None:
            logger.warning(
                "Risk Management hydrate: portfolio fetch failed for "
                "%s: %s", run_id, portfolio_res.error,
            )
            return empty_returns

        portfolio_df = portfolio_res.data
        clusters_df  = (
            clusters_res.data
            if clusters_res.ok and clusters_res.data is not None
            else pd.DataFrame()
        )

        # --- KPI strip ----------------------------------------------
        stats = compute_tail_stats(portfolio_df)
        var_txt  = _format_currency(stats["var"])
        cvar_txt = _format_currency(stats["cvar"])
        skew_txt = _format_signed(stats["skew"], digits=2)
        kurt_txt = _format_signed(stats["kurt"], digits=2)

        # --- Waterfall ----------------------------------------------
        worst_scenario = _resolve_worst_scenario(portfolio_df)
        waterfall_fig  = build_cluster_waterfall(
            clusters_df,
            scenario_label=worst_scenario,
        )
        waterfall_caption = _waterfall_caption_text(
            portfolio_df,
            worst_scenario,
        )

        # --- Tornado + donut -----------------------------------------
        tornado_fig = build_cluster_tornado(clusters_df)
        donut_fig   = build_cluster_donut(clusters_df)

        # --- Tail table ----------------------------------------------
        tail_df = compute_tail_table(portfolio_df, clusters_df, n=10)
        tail_card = _tail_table_card_body(tail_df, stats=stats)

        # --- Footer --------------------------------------------------
        footer = _format_footer(run_meta)

        return (
            var_txt, cvar_txt, skew_txt, kurt_txt,
            waterfall_fig, tornado_fig, donut_fig,
            waterfall_caption,
            tail_card,
            footer,
        )


# ─────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────

def _format_currency(value: Optional[float], *, digits: int = 0) -> str:
    """Pretty-print a PnL-style number; ``—`` for missing/NaN."""
    if value is None:
        return _PLACEHOLDER
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(v):
        return _PLACEHOLDER
    return f"{v:,.{digits}f}"


def _format_signed(value: Optional[float], *, digits: int = 2) -> str:
    """Pretty-print a unitless statistic (e.g. skew, kurt); ``—`` if NaN."""
    if value is None:
        return _PLACEHOLDER
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(v):
        return _PLACEHOLDER
    return f"{v:+.{digits}f}"


def _resolve_worst_scenario(portfolio_df: pd.DataFrame) -> Optional[str]:
    """Find the worst-portfolio-loss scenario label.

    Returns ``None`` for an empty / malformed frame so the figure
    builder can fall back to its empty placeholder.
    """
    if (
        portfolio_df is None
        or portfolio_df.empty
        or "sum_pnl_original" not in portfolio_df.columns
        or "scenario_label" not in portfolio_df.columns
    ):
        return None
    worst_idx = portfolio_df["sum_pnl_original"].idxmin()
    return str(portfolio_df.loc[worst_idx, "scenario_label"])


def _waterfall_caption_text(
    portfolio_df: pd.DataFrame,
    worst_scenario: Optional[str],
) -> Any:
    """One-line note under the waterfall stating the pinned scenario."""
    if worst_scenario is None:
        return "Awaiting run — scenario pin appears once data loads."
    if (
        portfolio_df is None
        or portfolio_df.empty
        or "scenario_label" not in portfolio_df.columns
    ):
        return f"Pinned: {worst_scenario}"
    row = portfolio_df.loc[
        portfolio_df["scenario_label"].astype(str) == str(worst_scenario)
    ]
    if row.empty or "sum_pnl_original" not in row.columns:
        return f"Pinned: {worst_scenario}"
    total = float(row["sum_pnl_original"].iloc[0])
    return html.Span([
        html.Span("Pinned: ", className="text-slate-500"),
        html.Span(str(worst_scenario), className="font-mono text-violet-300"),
        html.Span(" · portfolio PnL ", className="text-slate-500"),
        html.Span(
            f"{total:,.0f}",
            className="font-mono " + (
                "text-rose-300" if total < 0 else "text-emerald-300"
            ),
        ),
        html.Span(
            " · Phase 4 enables scenario pick + RF-level decomposition.",
            className="text-slate-600",
        ),
    ])


def _format_footer(run_meta: Dict[str, Any]) -> str:
    manifest_path = run_meta.get("manifest_path") or _PLACEHOLDER
    return f"Source: {manifest_path}"


# ─────────────────────────────────────────────────────────────────────
# Tail table — HTML table builder
# ─────────────────────────────────────────────────────────────────────

def _tail_table_placeholder() -> List[Any]:
    """Pre-run state — single-line italic note."""
    return [
        html.Div("Tail-conditional table",
                 className="text-sm font-semibold text-slate-200"),
        html.Div(
            "Scenarios at or below the 5th-percentile portfolio loss. "
            "``Top RF`` and ``Top trade type`` columns light up in "
            "Phase 4 once trade attributes are exposed.",
            className="text-xs text-slate-500",
        ),
        html.Div(
            "Awaiting inference run — the table will appear here.",
            className="text-xs text-slate-500 italic px-2 py-3",
        ),
    ]


def _tail_table_card_body(
    tail_df: pd.DataFrame,
    *,
    stats:   Dict[str, Optional[float]],
) -> List[Any]:
    """Populated tail table.

    Five real columns (``Scenario``, ``Date``, ``PnL``, ``Top
    cluster``, ``VaR contribution``) plus two placeholder columns
    (``Top RF``, ``Top trade type``) rendered as em-dash so the
    layout is stable when Phase 4 fills them.
    """
    header_block: List[Any] = [
        html.Div("Tail-conditional table",
                 className="text-sm font-semibold text-slate-200"),
        html.Div(
            f"Scenarios at or below the 5th-percentile portfolio loss · "
            f"showing top {len(tail_df)} worst.",
            className="text-xs text-slate-500",
        ),
    ]

    if tail_df is None or tail_df.empty:
        header_block.append(html.Div(
            "No tail scenarios — portfolio PnL series did not yield any "
            "below-quantile observations.",
            className="text-xs text-slate-500 italic px-2 py-3",
        ))
        return header_block

    header = html.Thead(html.Tr(className="rade-diag-header", children=[
        html.Th("Scenario",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Date",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Portfolio PnL",
            className="text-right px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Top cluster",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Cluster PnL",
            className="text-right px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Top RF",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-500"),
        html.Th("Top trade type",
            className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-500"),
        html.Th("VaR contrib %",
            className="text-right px-2 py-1.5 text-xs font-semibold text-slate-300"),
    ]))

    body_rows: List[Any] = []
    for _, row in tail_df.iterrows():
        scenario = str(row.get("scenario_label", ""))
        date_str = _date_from_scenario_label(scenario)
        pnl_value = float(row.get("portfolio_pnl") or 0.0)
        top_cluster = str(row.get("top_cluster") or _PLACEHOLDER)
        cluster_pnl = float(row.get("top_cluster_pnl") or 0.0) if pd.notna(row.get("top_cluster_pnl")) else None
        var_pct = float(row.get("var_contribution_pct") or 0.0)

        body_rows.append(html.Tr(
            className="rade-diag-row border-t border-slate-800",
            children=[
                html.Td(scenario,
                    className="rade-grid-mono px-2 py-1.5 text-xs text-violet-300",
                ),
                html.Td(date_str,
                    className="px-2 py-1.5 text-xs text-slate-400",
                ),
                html.Td(f"{pnl_value:,.0f}",
                    className=(
                        "rade-grid-mono px-2 py-1.5 text-xs text-right " +
                        ("text-rose-300" if pnl_value < 0 else "text-emerald-300")
                    ),
                ),
                html.Td(top_cluster,
                    className="rade-grid-mono px-2 py-1.5 text-xs text-slate-300",
                ),
                html.Td(
                    f"{cluster_pnl:,.0f}" if cluster_pnl is not None else _PLACEHOLDER,
                    className=(
                        "rade-grid-mono px-2 py-1.5 text-xs text-right " +
                        ("text-rose-300" if (cluster_pnl or 0) < 0 else "text-slate-400")
                    ),
                ),
                html.Td(_PLACEHOLDER,
                    className="px-2 py-1.5 text-xs text-slate-600 italic",
                ),
                html.Td(_PLACEHOLDER,
                    className="px-2 py-1.5 text-xs text-slate-600 italic",
                ),
                html.Td(_var_contribution_cell(var_pct),
                    className="px-2 py-1.5",
                ),
            ],
        ))

    table = html.Table(
        className="w-full text-xs",
        children=[header, html.Tbody(body_rows)],
    )

    return header_block + [
        html.Div(
            className="overflow-y-auto min-w-0",
            style={"maxHeight": "360px"},
            children=table,
        ),
    ]


def _var_contribution_cell(pct: float) -> Any:
    """Mini horizontal bar with the VaR-contribution percentage."""
    width = max(0.0, min(pct, 100.0))
    return html.Div(
        className="flex items-center gap-2 justify-end",
        children=[
            html.Span(
                f"{pct:.1f}%",
                className="rade-grid-mono text-xs text-slate-300",
            ),
            html.Div(
                className=(
                    "h-1.5 rounded-full bg-slate-800 overflow-hidden"
                ),
                style={"width": "60px"},
                children=html.Div(
                    style={
                        "width":      f"{width}%",
                        "background": "#f43f5e",
                        "height":     "100%",
                    },
                ),
            ),
        ],
    )


def _date_from_scenario_label(label: str) -> str:
    """Best-effort date extraction from a scenario label.

    Many of our scenario labels follow a ``yyyy-mm-dd``-shaped
    convention (eval splits + new-scenario folder names from the
    data pipeline).  If the input doesn't parse, we silently fall
    back to ``—`` so the column reads cleanly across mixed schemes.
    """
    if not label:
        return _PLACEHOLDER
    try:
        ts = pd.to_datetime(label, errors="coerce")
    except Exception:  # pragma: no cover — defensive
        return _PLACEHOLDER
    if pd.isna(ts):
        return _PLACEHOLDER
    return ts.strftime("%Y-%m-%d")


__all__ = ["register"]
```

---

### C.4 — Wiring diffs (three small edits)

#### C.4.1 — `src/ui/apps/rade_analytics/router.py`

**Add to imports** (alongside the other `from .layouts.* import build_*` lines):

```python
from .layouts.risk_management import build_risk_management
```

**Add to the `ROUTES` dict**, after the existing `/inference` entry:

```python
"/risk-management": PageSpec(
    path="/risk-management",
    title="Risk Management",
    build=build_risk_management,
),
```

#### C.4.2 — `src/ui/apps/rade_analytics/components/sidebar.py`

**Insert into the `NAV_ITEMS` list**, after the `/inference` entry:

```python
NavItem("/risk-management", "Risk Management", "tabler:shield-half"),
```

#### C.4.3 — `src/ui/apps/rade_analytics/callbacks/__init__.py`

**Add to the `from . import (...)` block**:

```python
from . import (
    cluster_deep_dive_cb,
    evaluation_cb,
    inference_cb,
    overview_cb,
    portfolio_cb,
    risk_management_cb,          # ← new
    splash_cb,
    trade_graph_cb,
)
```

**Add to `register_all()`**, right after the `inference_cb` line:

```python
inference_cb.register(app, backend)
risk_management_cb.register(app, backend)   # ← new
```

---

### Sanity checks after applying

1. Sidebar shows a new "Risk Management" entry between "Inference" and "Scenario Lab".
2. Clicking it navigates to `/risk-management` and the page header reads "Risk Management".
3. **With at least one completed inference run on disk**: KPI strip, waterfall, tornado, donut and tail table all populate.
4. **With no runs on disk**: subtitle reads *"No inference runs available yet — run the Inference Console to populate this page."* and panels stay in their empty state. No crash.
5. The tail table shows `Top RF` and `Top trade type` columns as `—` (em-dash) — these light up in Phase 4.

### What's next

Phase 0 is now complete (0.1 KPI rounding · 0.2 cluster zero-pad · 0.3 Diagnostics sub-tab · 0.4 Risk Management skeleton). The plan continues with:

| Phase | Description |
|---|---|
| **Phase 1a** | Sensitivity sub-tab + RF attribution Approach B (cluster intersection share) — no new data needed |
| **Phase 3** | Scaled / Original toggle on Overview + Evaluation (Eval pipeline emits `_original` parquets, API `?space=…`, UI app-shell toggle) |
| **Phase 2** | `GET /runs/{run_id}/clusters/{cid}/trade_attributes` API |
| **Phase 1b** | Risk attribution by `product` / `currency` / `desk` / `trade_type` (uses Phase 2) |
| **Phase 4** | Risk Management upgrade — RF-level waterfall, three-ring donut, tail-table Top RF / Top trade type columns |
