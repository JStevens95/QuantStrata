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

## Appendix C — Phase 3.1: Eval pipeline `_original` parquets (copy-paste sync)

> **Status**: Phase 3.1 extends `src/rade_ml_pt/pipelines/ensemble/eval.py` to
> emit original-space PnL artefacts that mirror the inference pipeline's
> output. Trade-level data lives in separate `_scaled.parquet` /
> `_original.parquet` files per cluster (matches inference exactly);
> aggregate parquets gain additional `_original` columns side-by-side with
> the existing scaled columns. The manifest now declares which spaces are
> available, so the Phase 3.2 API can discover the layout without
> grovelling through parquet footers.
>
> All edits are contained in **one file** —
> `src/rade_ml_pt/pipelines/ensemble/eval.py` — so the work-env sync is a
> single-file diff.

### What ships in Phase 3.1

| Artefact | Treatment | Schema |
|---|---|:---:|
| `portfolio_summary/portfolio_timeseries_{split}.parquet` | + 5 `*_original` cols (NaN-fill on legacy runs) | v2 → v3 |
| `portfolio_summary/cluster_timeseries_{split}.parquet`   | + 5 `*_original` cols (per-cluster NaN-fill) | v2 → v3 |
| `trade_predictions/{cid}/{split}_{scaled,original}.parquet` | **new family** — wide scenarios × trade_ids | v1 |
| `manifest.json` | + `spaces_available` + `artifacts` blocks | v1 → v2 |
| `ensemble_metrics.parquet`, `per_member_metrics.parquet`, `trade_metrics.parquet`, `group_correlations.parquet`, `graph_stats.parquet`, `quality_*.parquet` | **unchanged** (dimensionless metrics) | — |

### Scope cuts (deliberate)

- Eval's metric parquets (RMSE / correlations / quality) stay scaled-only — they're dimensionless, so an "original" view makes no semantic sense.
- Targets are **not** persisted at trade level in the new family — mirrors inference. Trade-level targets remain in `members/{cid}/predictions/{split}.npz` (scaled space).
- Inference aggregates (`cluster_predictions.parquet`, `portfolio_predictions.parquet`) **not** widened in this phase — strict scope.

### Single source of truth: `transform_predictions`

The scaled→original transformation reuses the existing static helper
`HybridGnnRnnInferencePipeline.transform_predictions` (which inference
already calls inside `_post_infer_cluster`). Eval now calls it twice per
cluster — once for predictions, once for targets — so the eval-side
`predictions_original` / `targets_original` columns are computed
identically to inference's per-cluster artefacts.

---

### Edit map

Nine discrete blocks inside `src/rade_ml_pt/pipelines/ensemble/eval.py`:

| # | Section | Location | Change |
|:-:|---|---|---|
| 1 | Imports | Top of file | `+ joblib`, `+ Tuple` |
| 2 | Schema constants | After imports | Bump v2→v3, add new `_TRADE_PREDICTIONS_*` and `_EVAL_MANIFEST_SCHEMA_VERSION` |
| 3 | `_load_cluster_contexts` | Insert before `_save_portfolio_timeseries_parquet` | NEW helper |
| 4 | `_transform_cluster_to_original` | Insert directly after #3 | NEW helper |
| 5 | `_save_trade_predictions_parquet` | Insert directly after #4 | NEW helper |
| 6 | `_save_portfolio_timeseries_parquet` | Widen signature + docstring + frame build | + 5 columns |
| 7 | `_save_cluster_timeseries_parquet` | Widen signature + docstring + frame build | + 5 columns |
| 8 | `_save_all_artifacts` — load contexts | After `cluster_attrs = _build_cluster_attributes(config)` | Load `cluster_contexts` once per run |
| 9 | `_save_all_artifacts` — per-split loop | Replace the existing `cluster_pred_sums` … `_save_cluster_timeseries_parquet(...)` block | Compute originals per cluster, save trade-level parquets, pass originals to widened writers |
| 10 | `_save_all_artifacts` — manifest | Replace the existing `_write_json(eval_dir / "manifest.json", {...})` call | Add `spaces_available` + `artifacts` blocks |

---

### Phase 3.1.1 — Imports

Replace the top-of-file imports block with:

```python
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
```

Diff: `+ Tuple` in `typing` import, `+ import joblib`.

---

### Phase 3.1.2 — Schema-version constants

Replace the existing schema-version block (just below imports) with:

```python
_PORTFOLIO_TIMESERIES_SCHEMA_VERSION = 3  # v3: +predictions_original / targets_original / *_original cols (Phase 3.1)
_CLUSTER_TIMESERIES_SCHEMA_VERSION   = 3  # v3: +predictions_original / targets_original / *_original cols (Phase 3.1)
_TRADE_METRICS_SCHEMA_VERSION = 1
_ENSEMBLE_METRICS_SCHEMA_VERSION = 1
_PER_MEMBER_METRICS_SCHEMA_VERSION = 1
_GROUP_CORRELATIONS_SCHEMA_VERSION = 1
_CLUSTER_ATTRIBUTES_SCHEMA_VERSION = 1
_GRAPH_STATS_SCHEMA_VERSION = 1
_COMPLETENESS_SCHEMA_VERSION = 1
_FEATURE_SUMMARY_SCHEMA_VERSION = 1
_TRADE_PREDICTIONS_SCHEMA_VERSION = 1     # Phase 3.1: per-cluster trade-level prediction parquets

# Per-cluster trade-level family (Phase 3.1).  Mirrors the inference
# pipeline's ``trade_predictions/<cid>_<space>.parquet`` layout but adds
# a per-cluster subdirectory because eval emits one prediction matrix
# per (cluster, split) pair, not one per run.
_TRADE_PREDICTIONS_DIRNAME = "trade_predictions"

# Manifest schema version — bumped to v2 to reflect the new
# ``spaces_available`` / ``artifacts`` blocks that Phase 3.1 added.  v1
# consumers (legacy) read the file as before — the extra keys are
# additive and don't disturb the canonical entry points.
_EVAL_MANIFEST_SCHEMA_VERSION = 2
```

---

### Phase 3.1.3 — `_load_cluster_contexts` helper

Insert **directly before** `_save_portfolio_timeseries_parquet` (which is currently the first function under the "Module-level helpers (stateless, no self)" block):

```python
# ----------------------------------------------------------------------
# Phase 3.1 — original-space helpers
# ----------------------------------------------------------------------

def _load_cluster_contexts(
    config: EnsembleConfig,
    member_versions: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Build per-cluster ``InferenceContext`` instances for scaled→original transforms.

    Only the two fields the inference-side ``transform_predictions`` static
    helper reads are populated:

    * ``target_scaler`` — sklearn-compatible scaler (with
      ``feature_names_in_`` and ``inverse_transform``) loaded from each
      member's ``target_scaler.pkl`` under the registry.
    * ``target_attributes`` — dict carrying ``trade_id`` and
      ``NotionalSign`` lists, loaded from ``target_attributes.json``.

    All other context fields stay ``None`` because eval doesn't need
    graph builders / cluster assets / elementary PnL for the inverse
    transform.

    Clusters whose member-version artefacts are missing or unreadable
    are silently dropped from the returned mapping with a warning;
    downstream writers fall back to scaled-only output for those
    clusters rather than failing the whole eval run.

    Resolution of ``member_versions``: prefers the explicit argument, then
    falls back to ``config.metadata["job"]["member_versions"]`` — mirrors
    the fallback chain in :func:`_load_scenario_labels_by_split` so legacy
    runs with implicit member versioning still work.
    """
    # Local import keeps the torch-transitive cost out of module load —
    # eval's lightweight metrics-only branch never hits this code path.
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

    if not member_versions:
        member_versions = (
            config.metadata.get("job", {}).get("member_versions", {}) or {}
        )
    if not isinstance(member_versions, dict):
        return {}

    contexts: Dict[str, Any] = {}
    for cid in config.cluster_ids:
        member_version = member_versions.get(cid)
        if not member_version:
            continue
        version_dir = Path(config.registry_dir) / str(member_version)
        scaler_path = version_dir / "target_scaler.pkl"
        attrs_path  = version_dir / "target_attributes.json"

        if not scaler_path.exists() or not attrs_path.exists():
            logger.warning(
                "Original-space artefacts skipped for cluster '%s' — missing "
                "target_scaler.pkl or target_attributes.json under %s",
                cid, version_dir,
            )
            continue

        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            logger.warning(
                "Could not load target_scaler.pkl for cluster '%s': %s — "
                "original-space skipped.", cid, exc,
            )
            continue
        try:
            with open(attrs_path) as f:
                attrs = json.load(f)
        except Exception as exc:
            logger.warning(
                "Could not load target_attributes.json for cluster '%s': %s — "
                "original-space skipped.", cid, exc,
            )
            continue

        contexts[cid] = InferenceContext(
            target_scaler     = scaler,
            target_attributes = attrs,
        )

    return contexts
```

---

### Phase 3.1.4 — `_transform_cluster_to_original` helper

Insert directly after `_load_cluster_contexts`:

```python
def _transform_cluster_to_original(
    cluster_id:      str,
    preds_2d:        np.ndarray,
    targets_2d:      np.ndarray,
    scenario_labels: List[str],
    ctx:             Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inverse-scale + notional-sign restore a single cluster's preds + targets.

    Reuses :meth:`HybridGnnRnnInferencePipeline.transform_predictions` —
    the single source of truth for the scaled→original transformation
    chain across the codebase.  We call it twice (once for predictions,
    once for targets) because the helper treats its array argument as
    "predictions" and we need the same math applied to targets so the
    eval-side ``predictions_original`` / ``targets_original`` columns
    are computed identically to inference's per-cluster artefacts.

    Returns
    -------
    preds_scaled_wide : pd.DataFrame
        Shape ``[n_scenarios, n_trades]`` — model output in z-space.
        Suitable for direct parquet write as ``<cid>/{split}_scaled.parquet``.
    preds_original_wide : pd.DataFrame
        Same shape — inverse-scaled + notional-sign-restored predictions.
        Suitable for direct parquet write as ``<cid>/{split}_original.parquet``.
    targets_original_wide : pd.DataFrame
        Same shape — inverse-scaled + notional-sign-restored targets.
        **Not** persisted at trade level (mirrors inference, which has no
        targets at all); used only to compute aggregate
        ``targets_original`` columns for the cluster / portfolio
        timeseries parquets.
    """
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
        HybridGnnRnnInferencePipeline,
    )

    preds_scaled_wide, preds_original_wide, _ = (
        HybridGnnRnnInferencePipeline.transform_predictions(
            cluster_id      = cluster_id,
            cluster_preds   = preds_2d,
            scenario_labels = scenario_labels,
            ctx             = ctx,
        )
    )
    _, targets_original_wide, _ = (
        HybridGnnRnnInferencePipeline.transform_predictions(
            cluster_id      = cluster_id,
            cluster_preds   = targets_2d,
            scenario_labels = scenario_labels,
            ctx             = ctx,
        )
    )
    return preds_scaled_wide, preds_original_wide, targets_original_wide
```

---

### Phase 3.1.5 — `_save_trade_predictions_parquet` helper

Insert directly after `_transform_cluster_to_original`:

```python
def _save_trade_predictions_parquet(
    eval_dir:       Path,
    cluster_id:     str,
    split:          str,
    *,
    scaled_wide:    pd.DataFrame,
    original_wide:  pd.DataFrame,
) -> Tuple[Path, Path]:
    """Write a cluster's trade-level prediction matrix in both spaces.

    Layout (Phase 3.1) ::

        trade_predictions/{cluster_id}/
            {split}_scaled.parquet     # [scenarios × trade_ids] z-space
            {split}_original.parquet   # [scenarios × trade_ids] notional units

    Mirrors the inference pipeline's
    ``trade_predictions/<cid>_<space>.parquet`` family with an extra
    ``{cluster_id}/`` directory level because eval emits one matrix per
    (cluster, split) pair, not one per run.  Wide format
    (scenarios as rows, trade IDs as columns) matches inference exactly,
    so the dashboard's per-cluster trade drill-down can use the same
    parquet-reader code path across the two pipelines.

    Returns the (scaled_path, original_path) tuple so the caller can
    record both in the run manifest.
    """
    out_dir = eval_dir / _TRADE_PREDICTIONS_DIRNAME / cluster_id
    out_dir.mkdir(parents=True, exist_ok=True)

    scaled_path   = out_dir / f"{split}_scaled.parquet"
    original_path = out_dir / f"{split}_original.parquet"

    scaled_wide.to_parquet(scaled_path)
    original_wide.to_parquet(original_path)

    return scaled_path, original_path
```

---

### Phase 3.1.6 — Widen `_save_portfolio_timeseries_parquet`

**Signature + docstring**: replace just the signature line and the existing v2 docstring with the v3 versions below. The body changes only in the frame-construction block (see further down).

OLD signature:
```python
def _save_portfolio_timeseries_parquet(
    eval_dir: Path,
    split: str,
    preds_1d: np.ndarray,
    targets_1d: np.ndarray,
    scenario_labels: List[str],
) -> None:
```

NEW signature:
```python
def _save_portfolio_timeseries_parquet(
    eval_dir: Path,
    split: str,
    preds_1d: np.ndarray,
    targets_1d: np.ndarray,
    scenario_labels: List[str],
    *,
    preds_1d_original:   Optional[np.ndarray] = None,
    targets_1d_original: Optional[np.ndarray] = None,
) -> None:
    """Write ``portfolio_timeseries_{split}.parquet`` (PRISM contract §11.15.2, B1).

    Long-format columnar time series for portfolio-level predictions /
    actuals, with precomputed error fields.  Replaces the legacy
    ``portfolio_summary/{split}.npz`` for API consumption.

    Schema (v3):

    - ``scenario_idx``           int32   — 0-based within this split
    - ``scenario_label``         string  — date-like label from elementary PnL index
    - ``split``                  string  — "train" / "val" / "test"
    - ``predictions``            float32 — portfolio predicted PnL (scaled space)
    - ``targets``                float32 — portfolio actual PnL (scaled space)
    - ``error``                  float32 — predictions − targets (scaled)
    - ``abs_error``              float32 — |error|
    - ``squared_error``          float32 — error²
    - ``predictions_original``   float32 — portfolio predicted PnL in original (notional) units (Phase 3.1, NaN if no scaler)
    - ``targets_original``       float32 — portfolio actual PnL in original units (Phase 3.1, NaN if no scaler)
    - ``error_original``         float32 — predictions_original − targets_original (Phase 3.1)
    - ``abs_error_original``     float32 — |error_original| (Phase 3.1)
    - ``squared_error_original`` float32 — error_original² (Phase 3.1)

    File-level metadata: ``_schema_version`` = ``3``, ``split``.

    Schema-change history
    ---------------------
    v2: ``predicted`` → ``predictions``, ``actual`` → ``targets`` so the
    column convention matches the per-member NPZ shards and the SQLite
    ``portfolio_summary`` table downstream.

    v3 (current, Phase 3.1): add five ``*_original`` columns carrying the
    inverse-scaled + notional-sign-restored portfolio PnL.  The legacy
    scaled columns are preserved unchanged for backwards compatibility;
    readers that pre-date Phase 3.1 see the new columns as extras.  When
    a caller can't compute original-space (no scaler for any cluster),
    the new columns are filled with NaN — schema stays canonical.

    Raises
    ------
    RuntimeError
        If ``scenario_labels`` is non-empty but its length does not match
        ``preds_1d``.  An empty ``scenario_labels`` list is treated as a
        legacy-run fallback: positional labels (``str(i)``) are used.
    """
```

**Frame-construction block**: replace the existing block that starts with `preds_f32 = preds_1d.astype(...)` through the closing `}).astype({...})` with:

```python
    preds_f32   = preds_1d.astype(np.float32, copy=False)
    targets_f32 = targets_1d.astype(np.float32, copy=False)
    error       = preds_f32 - targets_f32

    # Original-space columns (Phase 3.1).  NaN-fill when the caller can't
    # compute original-space — this keeps the schema canonical (v3) so
    # readers always see the same column list and can detect "no
    # original" via column-wide NaN rather than column-presence checks.
    if preds_1d_original is not None and targets_1d_original is not None:
        preds_orig_f32   = preds_1d_original.astype(np.float32, copy=False)
        targets_orig_f32 = targets_1d_original.astype(np.float32, copy=False)
    else:
        preds_orig_f32   = np.full(n, np.nan, dtype=np.float32)
        targets_orig_f32 = np.full(n, np.nan, dtype=np.float32)
    error_orig = preds_orig_f32 - targets_orig_f32

    df = pd.DataFrame({
        "scenario_idx":            np.arange(n, dtype=np.int32),
        "scenario_label":          labels,
        "split":                   [split] * n,
        "predictions":             preds_f32,
        "targets":                 targets_f32,
        "error":                   error,
        "abs_error":               np.abs(error),
        "squared_error":           error * error,
        "predictions_original":    preds_orig_f32,
        "targets_original":        targets_orig_f32,
        "error_original":          error_orig,
        "abs_error_original":      np.abs(error_orig),
        "squared_error_original":  error_orig * error_orig,
    }).astype({
        "scenario_idx":            np.int32,
        "scenario_label":          "string",
        "split":                   "string",
        "predictions":             np.float32,
        "targets":                 np.float32,
        "error":                   np.float32,
        "abs_error":               np.float32,
        "squared_error":           np.float32,
        "predictions_original":    np.float32,
        "targets_original":        np.float32,
        "error_original":          np.float32,
        "abs_error_original":      np.float32,
        "squared_error_original":  np.float32,
    })
```

Everything below (the `table = pa.table(df)` block and the file-write block) stays unchanged — the bumped `_PORTFOLIO_TIMESERIES_SCHEMA_VERSION` from §3.1.2 propagates automatically.

---

### Phase 3.1.7 — Widen `_save_cluster_timeseries_parquet`

**Signature + docstring**: replace just the signature and v2 docstring with the v3 versions:

```python
def _save_cluster_timeseries_parquet(
    eval_dir: Path,
    split: str,
    cluster_pred_sums: Dict[str, np.ndarray],
    cluster_target_sums: Dict[str, np.ndarray],
    scenario_labels: List[str],
    *,
    cluster_pred_sums_original:   Optional[Dict[str, np.ndarray]] = None,
    cluster_target_sums_original: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write ``cluster_timeseries_{split}.parquet`` (PRISM contract §11.15.2, B2).

    Long-format per-cluster × per-scenario portfolio sums.  One row per
    ``(cluster_id, scenario_idx)``; replaces the wide-format
    ``cluster_summary/{split}.npz`` for API consumption.

    Schema (v3):

    - ``cluster_id``             string  — e.g. "cluster_0"
    - ``scenario_idx``           int32   — 0-based within this split
    - ``scenario_label``         string  — date-like label from elementary PnL index
    - ``split``                  string  — "train" / "val" / "test"
    - ``predictions``            float32 — cluster portfolio predicted PnL (scaled)
    - ``targets``                float32 — cluster portfolio actual PnL (scaled)
    - ``error``                  float32 — predictions − targets (scaled)
    - ``abs_error``              float32 — |error|
    - ``squared_error``          float32 — error²
    - ``predictions_original``   float32 — cluster predicted PnL in original (notional) units (Phase 3.1)
    - ``targets_original``       float32 — cluster actual PnL in original units (Phase 3.1)
    - ``error_original``         float32 — predictions_original − targets_original (Phase 3.1)
    - ``abs_error_original``     float32 — |error_original| (Phase 3.1)
    - ``squared_error_original`` float32 — error_original² (Phase 3.1)

    Rows are sorted by ``(cluster_id, scenario_idx)`` for read locality
    under the dominant filter-by-cluster access pattern.  File metadata:
    ``_schema_version`` = ``3``, ``split``.

    Schema-change history
    ---------------------
    v2: ``predicted`` → ``predictions``, ``actual`` → ``targets`` so the
    column convention matches the per-member NPZ shards.

    v3 (current, Phase 3.1): five ``*_original`` columns carrying the
    inverse-scaled + notional-sign-restored cluster PnL.  Per-cluster
    NaN-fill is supported: a cluster with no scaler in
    ``cluster_pred_sums_original`` / ``cluster_target_sums_original``
    still emits its rows but with NaN in the original columns.  This
    keeps the schema canonical and makes the "no original data"
    detectable client-side via column-wide NaN.

    Raises
    ------
    RuntimeError
        If ``scenario_labels`` is non-empty and any cluster's series
        length disagrees with it, or if cluster pred/target lengths
        disagree with each other.  Empty ``scenario_labels`` triggers a
        warning and positional fallback (``str(i)``).
    """
```

**Frame-construction block**: replace the existing block that starts with `preds = np.concatenate([...])` through the closing `}).astype({...})` with:

```python
    preds = np.concatenate([
        cluster_pred_sums[cid].astype(np.float32, copy=False)
        for cid in cluster_ids
    ])
    actuals = np.concatenate([
        cluster_target_sums[cid].astype(np.float32, copy=False)
        for cid in cluster_ids
    ])
    error = preds - actuals

    # Original-space columns (Phase 3.1).  NaN-fill per-cluster when no
    # scaler was available — preserves the canonical v3 schema while
    # letting partial-coverage runs still emit a valid parquet.
    def _stack_or_nan(
        sums_by_cid: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        if sums_by_cid is None:
            return np.full(n_rows, np.nan, dtype=np.float32)
        out = np.empty(n_rows, dtype=np.float32)
        for i, cid in enumerate(cluster_ids):
            block = sums_by_cid.get(cid)
            start = i * n_scenarios
            end   = start + n_scenarios
            if block is None:
                out[start:end] = np.nan
            else:
                out[start:end] = np.asarray(block, dtype=np.float32)
        return out

    preds_orig   = _stack_or_nan(cluster_pred_sums_original)
    targets_orig = _stack_or_nan(cluster_target_sums_original)
    error_orig   = preds_orig - targets_orig

    df = pd.DataFrame({
        "cluster_id":              cluster_col,
        "scenario_idx":            scenario_idx_col,
        "scenario_label":          scenario_label_col,
        "split":                   np.full(n_rows, split, dtype=object),
        "predictions":             preds,
        "targets":                 actuals,
        "error":                   error,
        "abs_error":               np.abs(error),
        "squared_error":           error * error,
        "predictions_original":    preds_orig,
        "targets_original":        targets_orig,
        "error_original":          error_orig,
        "abs_error_original":      np.abs(error_orig),
        "squared_error_original":  error_orig * error_orig,
    }).astype({
        "cluster_id":              "string",
        "scenario_idx":            np.int32,
        "scenario_label":          "string",
        "split":                   "string",
        "predictions":             np.float32,
        "targets":                 np.float32,
        "error":                   np.float32,
        "abs_error":               np.float32,
        "squared_error":           np.float32,
        "predictions_original":    np.float32,
        "targets_original":        np.float32,
        "error_original":          np.float32,
        "abs_error_original":      np.float32,
        "squared_error_original":  np.float32,
    })
```

Everything below (the `table = pa.table(df)` block and the file-write block) stays unchanged.

---

### Phase 3.1.8 — `_save_all_artifacts`: load cluster contexts once per run

Inside the `_save_all_artifacts` method, **immediately after** the existing block:

```python
        # Shared inputs — computed once, reused across multiple writers.
        scenario_labels_by_split = _load_scenario_labels_by_split(
            config, member_versions,
        )
        cluster_attrs = _build_cluster_attributes(config)
```

…insert the following:

```python
        # Phase 3.1 — per-cluster (target_scaler, target_attributes) for
        # scaled→original transforms.  Loaded *once* (a single deserialise
        # per member, not per split) and reused across all three splits.
        # Empty mapping for legacy runs without scaler artefacts; per-split
        # logic falls back to NaN-filled original columns in that case.
        cluster_contexts = _load_cluster_contexts(config, member_versions)
        if not cluster_contexts:
            logger.warning(
                "No per-cluster scaler artefacts found — eval will emit "
                "scaled-space only (Phase 3.1 original columns will be NaN)."
            )
```

---

### Phase 3.1.9 — `_save_all_artifacts`: per-split loop changes

Inside the `for split in all_split_results:` loop, **replace** the block that currently runs from:

```python
            # Cluster / portfolio sums — computed once, reused by B1/B2/B6.
            cluster_pred_sums: Dict[str, np.ndarray] = {
                ...
            }
            ...
            _save_cluster_timeseries_parquet(
                eval_dir, split,
                cluster_pred_sums=cluster_pred_sums,
                cluster_target_sums=cluster_target_sums,
                scenario_labels=scenario_labels,
            )
```

…down to and including the closing `_save_cluster_timeseries_parquet(...)` call — with this expanded block:

```python
            # Cluster / portfolio sums — computed once, reused by B1/B2/B6.
            cluster_pred_sums: Dict[str, np.ndarray] = {
                cid: preds_by_cluster[cid].sum(axis=1)
                for cid in sorted(preds_by_cluster)
            }
            cluster_target_sums: Dict[str, np.ndarray] = {
                cid: targets_by_cluster[cid].sum(axis=1)
                for cid in sorted(targets_by_cluster)
            }
            if cluster_pred_sums:
                preds_1d = np.add.reduce(list(cluster_pred_sums.values()))
                targets_1d = np.add.reduce(list(cluster_target_sums.values()))
            else:
                preds_1d = np.zeros(0, dtype=np.float32)
                targets_1d = np.zeros(0, dtype=np.float32)

            scenario_labels = scenario_labels_by_split.get(split, [])

            # ── Phase 3.1 — original-space transforms ─────────────────
            #
            # For every cluster carrying a scaler context, run the canonical
            # inverse-scale + notional-sign-restore chain once for predictions
            # and once for targets.  Two outputs:
            #
            # (a) Per-cluster wide DataFrames (scenarios × trade_ids) for
            #     trade-level parquets — mirrors inference's
            #     ``trade_predictions/<cid>_<space>.parquet`` layout.
            # (b) Per-cluster 1-D sums used by the cluster / portfolio
            #     timeseries writers to fill the new ``*_original`` columns.
            #
            # Clusters with no scaler context are silently skipped here
            # (logged once at startup); the writers NaN-fill their rows so
            # the v3 schema stays canonical regardless of coverage.
            cluster_pred_sums_original:   Dict[str, np.ndarray] = {}
            cluster_target_sums_original: Dict[str, np.ndarray] = {}
            t_orig = time.perf_counter()
            for cid in sorted(preds_by_cluster):
                ctx = cluster_contexts.get(cid)
                if ctx is None:
                    continue
                try:
                    (
                        preds_scaled_wide,
                        preds_original_wide,
                        targets_original_wide,
                    ) = _transform_cluster_to_original(
                        cluster_id      = cid,
                        preds_2d        = preds_by_cluster[cid],
                        targets_2d      = targets_by_cluster[cid],
                        scenario_labels = scenario_labels or None,
                        ctx             = ctx,
                    )
                except Exception as exc:
                    logger.warning(
                        "  [%s] cluster '%s' original-space transform "
                        "failed: %s — falling back to scaled-only for "
                        "this cluster.", split, cid, exc,
                    )
                    continue

                _save_trade_predictions_parquet(
                    eval_dir, cid, split,
                    scaled_wide   = preds_scaled_wide,
                    original_wide = preds_original_wide,
                )

                cluster_pred_sums_original[cid] = (
                    preds_original_wide.to_numpy().sum(axis=1)
                )
                cluster_target_sums_original[cid] = (
                    targets_original_wide.to_numpy().sum(axis=1)
                )
            if cluster_pred_sums_original:
                logger.info(
                    "  [%s] original-space transforms: %.1fs (%d clusters)",
                    split, time.perf_counter() - t_orig,
                    len(cluster_pred_sums_original),
                )

            # Portfolio-level original-space sum across clusters.  ``+`` over
            # the values fans out per-scenario across the cluster axis, so
            # the result lines up 1-for-1 with the scaled ``preds_1d``
            # ordering.  ``None`` propagates when no cluster had original
            # data, which the writer translates to a NaN-filled column.
            if cluster_pred_sums_original:
                preds_1d_original = np.add.reduce(
                    list(cluster_pred_sums_original.values())
                )
                targets_1d_original = np.add.reduce(
                    list(cluster_target_sums_original.values())
                )
            else:
                preds_1d_original   = None
                targets_1d_original = None

            _save_portfolio_timeseries_parquet(
                eval_dir, split,
                preds_1d=preds_1d,
                targets_1d=targets_1d,
                scenario_labels=scenario_labels,
                preds_1d_original=preds_1d_original,
                targets_1d_original=targets_1d_original,
            )
            _save_cluster_timeseries_parquet(
                eval_dir, split,
                cluster_pred_sums=cluster_pred_sums,
                cluster_target_sums=cluster_target_sums,
                scenario_labels=scenario_labels,
                cluster_pred_sums_original=(
                    cluster_pred_sums_original or None
                ),
                cluster_target_sums_original=(
                    cluster_target_sums_original or None
                ),
            )
```

The remainder of the per-split loop (`_save_trade_metrics_parquet`, `_save_group_correlations_parquet`, `_save_quality_parquets_for_split`, the trailing log line) stays unchanged.

---

### Phase 3.1.10 — Manifest extension

Inside `_save_all_artifacts`, replace the existing `_write_json(eval_dir / "manifest.json", {...})` call with:

```python
        # Manifest written *upfront* so a partial / failing run still
        # leaves a structurally-valid manifest behind.  The ``artifacts``
        # block enumerates which families this writer produces; the
        # Phase 3.2 API uses this to discover the available spaces /
        # paths without having to peek at every parquet's footer.
        _write_json(eval_dir / "manifest.json", {
            "_schema_version": _EVAL_MANIFEST_SCHEMA_VERSION,
            "version": version,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trade_ids": config.all_trade_ids,
            "cluster_ids": config.cluster_ids,
            "cluster_trade_indices": {
                k: v.tolist() if isinstance(v, np.ndarray) else list(v)
                for k, v in cluster_trade_indices.items()
            },
            "splits_available": sorted(all_split_results.keys()),
            # Phase 3.1 — declared at manifest level so a single read of
            # the manifest tells the API which spaces are available
            # without grovelling through every parquet's column list.
            "spaces_available": ["scaled", "original"],
            "artifacts": {
                "portfolio_timeseries": {
                    "schema_version": _PORTFOLIO_TIMESERIES_SCHEMA_VERSION,
                    "path_template":  "portfolio_summary/portfolio_timeseries_{split}.parquet",
                    "spaces": ["scaled", "original"],
                    "columns": {
                        "scaled":   ["predictions", "targets", "error", "abs_error", "squared_error"],
                        "original": [
                            "predictions_original", "targets_original",
                            "error_original", "abs_error_original",
                            "squared_error_original",
                        ],
                    },
                },
                "cluster_timeseries": {
                    "schema_version": _CLUSTER_TIMESERIES_SCHEMA_VERSION,
                    "path_template":  "portfolio_summary/cluster_timeseries_{split}.parquet",
                    "spaces": ["scaled", "original"],
                    "columns": {
                        "scaled":   ["predictions", "targets", "error", "abs_error", "squared_error"],
                        "original": [
                            "predictions_original", "targets_original",
                            "error_original", "abs_error_original",
                            "squared_error_original",
                        ],
                    },
                },
                "trade_predictions": {
                    "schema_version": _TRADE_PREDICTIONS_SCHEMA_VERSION,
                    "path_template":  (
                        _TRADE_PREDICTIONS_DIRNAME
                        + "/{cluster_id}/{split}_{space}.parquet"
                    ),
                    "spaces": ["scaled", "original"],
                    "shape": "wide_scenarios_x_trades",
                    "note": "Predictions only (no targets) — mirrors inference output.",
                },
            },
        })
```

The `_write_json(eval_dir / "trade_cluster_map.json", ...)` call directly below stays unchanged.

---

### Sanity checks after applying

Run the eval pipeline against a registered ensemble and verify:

1. **Manifest is v2 with the new blocks**:
   ```bash
   jq '._schema_version, .spaces_available, (.artifacts | keys)' artifacts/ensemble/<version>/evaluation/manifest.json
   # → 2
   #   ["scaled", "original"]
   #   ["cluster_timeseries", "portfolio_timeseries", "trade_predictions"]
   ```

2. **`portfolio_timeseries_{split}.parquet` is v3 with the new columns**:
   ```python
   import pandas as pd
   df = pd.read_parquet("artifacts/ensemble/<version>/evaluation/portfolio_summary/portfolio_timeseries_test.parquet")
   assert {"predictions_original","targets_original","error_original",
           "abs_error_original","squared_error_original"}.issubset(df.columns)
   assert not df["predictions_original"].isna().all()   # populated, not NaN-fill
   ```

3. **`cluster_timeseries_{split}.parquet` is v3 with the new columns** — same check on the cluster file.

4. **New trade-level family exists per cluster × split × space**:
   ```bash
   ls artifacts/ensemble/<version>/evaluation/trade_predictions/
   # → cluster_0/  cluster_1/  cluster_2/  ...
   ls artifacts/ensemble/<version>/evaluation/trade_predictions/cluster_0/
   # → test_original.parquet  test_scaled.parquet  train_original.parquet  ...
   ```

5. **Wide layout matches inference**:
   ```python
   wide = pd.read_parquet("artifacts/ensemble/<version>/evaluation/trade_predictions/cluster_0/test_scaled.parquet")
   # Row index → scenario_label (a single named column after to_parquet)
   # Other columns → trade_ids in the scaler's canonical order
   ```

6. **Legacy-run fallback (no scalers)**: if `target_scaler.pkl` is missing for every member, eval still runs to completion; the new `*_original` columns are all NaN; no `trade_predictions/` directory is created; the manifest still declares `spaces_available: ["scaled","original"]` (advertised even when the data is NaN — readers detect this via column-wide NaN).

### What's next

Phase 3.2 — API `?space=scaled|original` on Overview / Eval endpoints:

- Adds a `space: Literal["scaled","original"] = "scaled"` query param to the relevant endpoints.
- Server-side projects the requested space's columns onto a canonical schema (e.g. `predictions` always means "the value in the chosen space") so the UI stays oblivious to suffixes.
- New endpoint family for the trade-level parquets:
  `GET /eval/runs/{version}/clusters/{cid}/trade_predictions/{split}?space=…`.
- Pydantic response models updated to the canonical shape.
