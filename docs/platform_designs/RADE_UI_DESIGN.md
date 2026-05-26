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

## Appendix A — Phase M.3: `promote_to_predictions()` (copy-paste sync)

> **Status**: 104/104 tests pass (53 M.1 + 43 M.2 + 8 M.3), lint-clean.
> Only **one method** is new in `monitor.py` for M.3 — the full source
> is below for verbatim paste into work env.  Surrounding additions
> (the `PromoteResult` dataclass, the two private helpers, the
> `__init__` cache fields, the `_build_manifest` kwarg, the
> `__all__` bump) are already documented in chat — copy those over
> first if you haven't already.
>
> **Where this method goes**: paste it as a new public method on
> `EnsembleMonitoringPipeline`, immediately after `run()` and before
> the `# ─── Internal — drift compute ────` section divider.

### `EnsembleMonitoringPipeline.promote_to_predictions` (full source)

```python
    def promote_to_predictions(self) -> PromoteResult:
        """Forward-pass the same scenarios this run() just drifted on.

        Optional follow-up to :meth:`run` — re-uses the already-loaded
        ensemble + per-cluster contexts + validated scenarios to
        produce per-trade predicted PnL artifacts inside the SAME
        monitoring run directory.

        On-disk layout after a successful promote::

            monitoring_runs/<run_id>/monitoring/
              ├── manifest.json              ← rewritten with predictions block
              ├── drift_summary.json
              ├── clusters/<cid>/drift_table.parquet
              └── inference/                 ← NEW
                  ├── manifest.json
                  ├── cluster_summary/cluster_predictions.parquet
                  ├── portfolio_summary/portfolio_predictions.parquet
                  └── trade_predictions/<cid>_<space>.parquet

        Mechanism: temporarily set
        ``inference_pipeline.config.artifacts_dir`` to ``monitoring_dir``
        so the existing ``run_inference()`` writes its standard
        ``<root>/inference/...`` tree INSIDE the monitoring run rather
        than to the global inference dir.  The swap is wrapped in
        ``try/finally`` so partial failures restore the config
        cleanly.

        Returns
        -------
        PromoteResult
            Self-describing handle on the new predictions
            (``predictions_dir``, ``inference_result``, timing).

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called on this instance, OR
            if a promote has already succeeded on this instance
            (promotes are NOT re-runnable — construct a new pipeline
            for a fresh run_id).
        """
        if self._run_paths is None or self._cached_portfolio_summary is None:
            raise RuntimeError(
                "promote_to_predictions() requires a successful prior run() "
                "on this instance.  Call pipeline.run(new_scenario_dir) first."
            )
        if self._promotion is not None:
            raise RuntimeError(
                "promote_to_predictions() has already succeeded on this "
                "instance.  Construct a new EnsembleMonitoringPipeline if "
                "you need to monitor + promote a fresh scenario set."
            )

        logger.info(
            "EnsembleMonitoringPipeline.promote_to_predictions: starting "
            "(run_id=%s)", self._run_paths.run_id,
        )
        self._emit(_mon_event(
            "Promote started", status=STATUS_RUNNING,
            target=self.ensemble_version,
        ))
        t0 = time.perf_counter()

        config            = self._inference_pipeline.config
        original_root     = config.artifacts_dir
        # Stash-swap target: ``run_inference()`` joins ``INFERENCE_DIRNAME``
        # onto this, so predictions land at ``monitoring_dir/inference/``.
        # We mkdir explicitly first because run_inference's own mkdir
        # is conditional on ``config.artifacts_dir`` being truthy and
        # we want to fail loudly if the swap target is unwritable.
        swap_target = self._run_paths.monitoring_dir
        swap_target.mkdir(parents=True, exist_ok=True)

        try:
            config.artifacts_dir = str(swap_target)
            inference_result = self._inference_pipeline.run_inference()
        except Exception as exc:
            self._emit(_mon_event(
                "Promote failed", status=STATUS_FAIL,
                target=type(exc).__name__, detail=str(exc),
            ))
            raise
        finally:
            # Restore even on failure so the inference pipeline (and
            # any callers sharing this config object) sees the
            # original artifacts_dir afterwards.
            config.artifacts_dir = original_root

        wall         = time.perf_counter() - t0
        promoted_at  = datetime.now(timezone.utc).isoformat(timespec="seconds")
        predictions_dir = swap_target / _PROMOTE_PREDICTIONS_SUBDIR

        # Derive ``n_clusters_predicted`` from the inference result
        # metadata when present; fall back to counting cluster
        # parquets on disk so the UI's "N clusters scored" KPI is
        # never silently wrong even if ``infer.py`` evolves.
        n_clusters_predicted = self._derive_n_clusters_predicted(
            inference_result, predictions_dir,
        )

        predictions_block = self._build_predictions_block(
            predictions_dir      = predictions_dir,
            promoted_at          = promoted_at,
            wall_seconds         = wall,
            n_clusters_predicted = n_clusters_predicted,
        )

        # Re-write the monitoring manifest with the new predictions
        # pointer.  All other manifest fields stay byte-identical
        # because we rebuild from the cached portfolio_summary.
        manifest = self._build_manifest(
            self._cached_portfolio_summary,
            predictions_block=predictions_block,
        )
        write_monitoring_manifest_json(
            manifest = manifest,
            out_path = self._run_paths.manifest_path,
        )

        promote_result = PromoteResult(
            run_id               = self._run_paths.run_id,
            predictions_dir      = predictions_dir,
            manifest_path        = self._run_paths.manifest_path,
            inference_result     = inference_result,
            promoted_at          = promoted_at,
            wall_seconds         = wall,
            n_clusters_predicted = n_clusters_predicted,
        )
        self._promotion = promote_result

        logger.info(
            "EnsembleMonitoringPipeline.promote_to_predictions: done "
            "(%.3fs, %d clusters predicted, predictions_dir=%s)",
            wall, n_clusters_predicted, predictions_dir,
        )
        self._emit(_mon_event(
            "Promote complete", status=STATUS_OK,
            target=f"{wall * 1000:.0f} ms · {n_clusters_predicted} clusters",
            detail=f"predictions_dir={predictions_dir.name}",
        ))
        return promote_result
```

### Symbols the method depends on (must already exist in `monitor.py`)

| Symbol                            | Provided by                                                            |
|-----------------------------------|------------------------------------------------------------------------|
| `PromoteResult`                   | Frozen dataclass added to this module in M.3                           |
| `self._run_paths`                 | Populated by `run()` (M.2)                                             |
| `self._cached_portfolio_summary`  | New `__init__` field cached at end of `run()` (M.3)                    |
| `self._promotion`                 | New `__init__` field (M.3); one-shot guard                             |
| `self._inference_pipeline`        | Composed `EnsembleInferencePipeline` (M.2)                             |
| `self._build_manifest(..., predictions_block=...)` | M.2 method now with optional kwarg (M.3)              |
| `self._build_predictions_block(...)` | New private helper (M.3)                                           |
| `self._derive_n_clusters_predicted(...)` | New static helper (M.3)                                          |
| `_PROMOTE_PREDICTIONS_SUBDIR`     | New module constant `= "inference"` (M.3)                              |
| `_mon_event(...)`                 | M.2 helper                                                             |
| `write_monitoring_manifest_json`  | M.2 writer                                                             |
| `STATUS_OK`, `STATUS_FAIL`, `STATUS_RUNNING` | Imported from `infer_events` (M.2)                          |

### Quick smoke (manual)

```python
pipe   = EnsembleMonitoringPipeline(ensemble_config=config, ensemble_version="ens_v1")
drift  = pipe.run(new_scenario_dir="/path/to/new_scenarios/")
# user reviews drift, decides to promote
promote = pipe.promote_to_predictions()

print(promote.predictions_dir)
#   → <artifacts_dir>/monitoring_runs/<run_id>/monitoring/inference
print(promote.n_clusters_predicted)
# manifest.json now has predictions block populated
```

---
