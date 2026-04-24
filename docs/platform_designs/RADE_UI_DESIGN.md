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

## Appendix A — Phase E.4 · Evaluation → Cluster Deep-Dive

Self-contained change-set for the Cluster Deep-Dive sub-tab at
`/evaluation/cluster`. Applying the file operations below — in order
— lands the full page with live data + preview script + no extra
wiring.

This appendix supersedes every prior revision. The **Row 2 revision**
moved KPIs out of the header and alongside a new per-cluster
training-curves chart with a metric-picker chip row.

### A.0 · Layout at a glance

Five-row page scoped to a single cluster (no Evaluation filter bar,
split inherits from the topbar):

| Row | Purpose |
|----|---------|
| 1 | Header band · cluster picker · attribute chips · "Trade-Graph" link |
| 2 | 2×2 KPI grid (MAE / RMSE / R² / Coverage) + training-curves chart with overlay chip filter |
| 3 | Residual over time · predicted vs target PnL (shaded error band) |
| 4 | Per-trade residual violin (target / elementary) · per-trade scatter (click to highlight) |
| 5 | Trades AgGrid — per-trade MAE / RMSE / p95 / mean_residual; row click ↔ Row 4 scatter |

KPI definitions:

* **MAE / RMSE** — from `per_member_metrics.parquet` (split-scoped).
* **R²** — 1 − SSres/SStot between cluster `predicted` and `actual`
  (computed client-side from the cluster timeseries; matches
  `rade_ml_pt.evaluation.metrics.r_squared`).
* **Coverage** — fraction of scenarios where `|error| ≤ MAE`. Uses
  the cluster's own MAE as the tolerance so the metric is
  scale-invariant across clusters.

Training-curves behaviour:

* `train_loss` is **always** plotted — the chart is meaningless
  without it, so it's not a chip.
* Additional metrics (auto-discovered from
  `training_curves.parquet`) appear as chips. When a `val_*` sibling
  exists (e.g. `val_mae` for `mae`) it is drawn dashed in the same
  hue as its `train_*` sibling.
* Chip selection is persisted in `session.evaluation.deep_dive_curve_metrics`
  and reset on cluster change (different trainer runs may emit
  different metric sets).

### A.1 · File checklist (apply order)

| # | Action | Path | Scope |
|---|--------|------|-------|
| F1 | PATCH   | `src/rade_ml_pt/pipelines/ensemble/eval.py` | `files_to_copy` tuple — stage `training_curves.parquet` |
| F2 | PATCH   | `src/rade_ml_pt/ensemble/api/services/paths.py` | add `member_training_curves` + doc-block |
| F3 | PATCH   | `src/rade_ml_pt/ensemble/api/services/reader.py` | add `training_curves()` method |
| F4 | CREATE  | `src/rade_ml_pt/ensemble/api/models/training_curves.py` | Pydantic response model |
| F5 | CREATE  | `src/rade_ml_pt/ensemble/api/routers/training_curves.py` | FastAPI router |
| F6 | PATCH   | `src/rade_ml_pt/ensemble/api/app.py` | import + register router |
| F7 | PATCH   | `src/rade_ml_pt/ensemble/api/client.py` | import + `training_curves()` client method |
| U1 | REPLACE | `src/ui/apps/rade_analytics/data/session.py` | schema v6 + `deep_dive_curve_metrics` |
| U2 | PATCH   | `src/ui/apps/rade_analytics/data/backend.py` | cache binding + `_fetch_training_curves` + `training_curves_df()` |
| U3 | CREATE  | `src/ui/apps/rade_analytics/figures/training_curves.py` | new Plotly builder |
| U4 | REPLACE | `src/ui/apps/rade_analytics/figures/__init__.py` | re-export `training_curves_chart` |
| U5 | REPLACE | `src/ui/apps/rade_analytics/layouts/evaluation/cluster_deep_dive.py` | 5-row layout |
| U6 | REPLACE | `src/ui/apps/rade_analytics/callbacks/cluster_deep_dive_cb.py` | 11 callbacks |
| U7 | REPLACE | `examples/rade_analytics/_mock_backend.py` | synth training curves generator |
| U8 | REPLACE | `examples/rade_analytics/09_cluster_deep_dive_preview_live.py` | preview script |

> **Smoke test** — after pasting, run from the project root:
>
> ```bash
> PYTHONPATH=. python examples/rade_analytics/09_cluster_deep_dive_preview_live.py
> ```
>
> Open http://localhost:8055/evaluation/cluster and
> http://localhost:8055/evaluation/cluster?cid=cluster_03. Both should
> render without errors. Row 2 should show a 2×2 KPI grid on the left
> and a training-loss chart (with a chip row for `val_loss`, `mae`,
> `val_mae`) on the right.

---

### A.F1 · PATCH `src/rade_ml_pt/pipelines/ensemble/eval.py`

Within `_copy_member_graph_artifacts`, the `files_to_copy` tuple now
includes `training_curves.parquet` and the doc-block and log line are
updated. Replace the function body with:

```python
def _copy_member_graph_artifacts(
    eval_dir: Path,
    config: EnsembleConfig,
    member_versions: Optional[Dict[str, str]] = None,
) -> None:
    """Stage per-member graph & training artefacts into the eval directory.

    For each cluster, copies the following files from
    ``{registry_dir}/{member_version}/`` into
    ``{eval_dir}/members/{cluster_id}/``:

    * ``graph_results.joblib`` — sparse adjacency used by the Trade-Graph
      UI tab (``sparse_indices`` / ``sparse_values`` / ``sparse_shape``).
    * ``trade_universe.json`` — target / elementary trade split used to
      colour nodes and populate the Selected-Trade card.
    * ``training_curves.parquet`` — per-epoch ``train_loss`` (and any
      additional per-epoch series the trainer emitted, e.g. ``val_loss``,
      ``mae``, ``val_mae``, …).  Consumed by the Cluster Deep-Dive
      training-curves chart.

    All three files are already referenced by the member artefacts in
    the registry; copying them into the eval bundle makes the evaluation
    directory self-contained (shippable to a read-only environment
    without the training registry) and keeps the API a pure reader over
    ``artifacts_dir``.  This matches the pattern set by
    ``members/{cluster_id}/predictions/{split}.npz``.

    Missing source files are logged at debug and skipped — the UI
    already tolerates the absence of trade-graph / training-curve data
    gracefully, so legacy runs without these files continue to work.
    """
    if not member_versions:
        member_versions = config.metadata.get("job", {}).get("member_versions", {})

    if not isinstance(member_versions, dict) or not member_versions:
        logger.info("  skipping graph-artefact staging (no member versions)")
        return

    files_to_copy = (
        "graph_results.joblib",
        "trade_universe.json",
        "training_curves.parquet",
    )
    n_clusters_staged = 0
    n_files_staged = 0

    for cid in config.cluster_ids:
        member_version = member_versions.get(cid)
        if not member_version:
            continue

        src_dir = Path(config.registry_dir) / member_version
        dst_dir = eval_dir / "members" / cid
        dst_dir.mkdir(parents=True, exist_ok=True)

        cluster_files_staged = 0
        for filename in files_to_copy:
            src = src_dir / filename
            if not src.exists():
                logger.debug(
                    "  member graph artefact missing: %s (cluster=%s)",
                    src, cid,
                )
                continue
            try:
                shutil.copy2(src, dst_dir / filename)
                cluster_files_staged += 1
            except Exception as exc:
                logger.warning(
                    "  could not stage %s for cluster '%s': %s",
                    filename, cid, exc,
                )

        if cluster_files_staged > 0:
            n_clusters_staged += 1
            n_files_staged += cluster_files_staged

    logger.info(
        "  staged member graph + training artefacts: %d files across "
        "%d clusters", n_files_staged, n_clusters_staged,
    )
```

---

### A.F2 · REPLACE `src/rade_ml_pt/ensemble/api/services/paths.py`

```python
"""Path resolution for an ensemble evaluation run.

Canonical layout written by :mod:`rade_ml_pt.pipelines.ensemble.eval`:

.. code-block:: text

    {artifacts_dir}/ensemble/{version}/evaluation/
        manifest.json
        trade_cluster_map.json
        cluster_attributes.parquet
        ensemble_metrics.parquet
        per_member_metrics.parquet
        graph_stats.parquet
        portfolio_summary/
            portfolio_timeseries_{split}.parquet
            cluster_timeseries_{split}.parquet
        trade_metrics/
            trade_metrics_{split}.parquet
        group_correlations/
            group_correlations_{split}.parquet
        quality/
            completeness_{split}.parquet
            feature_summary_{split}.parquet
        members/
            {cluster_id}/
                predictions/{split}.npz
                graph_results.joblib
                trade_universe.json
                training_curves.parquet

This module is the single source of truth for those filenames inside the
API.  If the evaluation pipeline ever renames or relocates an artifact,
update this class and the readers follow automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Immutable path resolver for one ``(artifacts_dir, version)`` pair."""

    artifacts_dir: Path
    version: str

    # ── Root ──────────────────────────────────────────────────────
    @property
    def eval_dir(self) -> Path:
        return self.artifacts_dir / "ensemble" / self.version / "evaluation"

    # ── Run metadata ──────────────────────────────────────────────
    @property
    def manifest(self) -> Path:
        return self.eval_dir / "manifest.json"

    @property
    def trade_cluster_map(self) -> Path:
        return self.eval_dir / "trade_cluster_map.json"

    # ── Ensemble-scoped parquets ──────────────────────────────────
    @property
    def cluster_attributes(self) -> Path:
        return self.eval_dir / "cluster_attributes.parquet"

    @property
    def ensemble_metrics(self) -> Path:
        return self.eval_dir / "ensemble_metrics.parquet"

    @property
    def per_member_metrics(self) -> Path:
        return self.eval_dir / "per_member_metrics.parquet"

    @property
    def graph_stats(self) -> Path:
        return self.eval_dir / "graph_stats.parquet"

    # ── Per-split parquets ────────────────────────────────────────
    def portfolio_timeseries(self, split: str) -> Path:
        return (
            self.eval_dir
            / "portfolio_summary"
            / f"portfolio_timeseries_{split}.parquet"
        )

    def cluster_timeseries(self, split: str) -> Path:
        return (
            self.eval_dir
            / "portfolio_summary"
            / f"cluster_timeseries_{split}.parquet"
        )

    def trade_metrics(self, split: str) -> Path:
        return self.eval_dir / "trade_metrics" / f"trade_metrics_{split}.parquet"

    def group_correlations(self, split: str) -> Path:
        return (
            self.eval_dir
            / "group_correlations"
            / f"group_correlations_{split}.parquet"
        )

    def completeness(self, split: str) -> Path:
        return self.eval_dir / "quality" / f"completeness_{split}.parquet"

    def feature_summary(self, split: str) -> Path:
        return self.eval_dir / "quality" / f"feature_summary_{split}.parquet"

    # ── Raw per-member shards (NPZ) ───────────────────────────────
    def member_predictions(self, cluster_id: str, split: str) -> Path:
        return (
            self.eval_dir
            / "members"
            / cluster_id
            / "predictions"
            / f"{split}.npz"
        )

    # ── Per-member graph artefacts (staged by the eval pipeline) ──
    def member_graph_results(self, cluster_id: str) -> Path:
        """Sparse adjacency joblib staged from the training registry.

        Used by the Trade-Graph UI tab to render cluster networks.
        The joblib is expected to contain keys ``sparse_indices``,
        ``sparse_values`` and ``sparse_shape`` (see
        ``_save_graph_stats_parquet`` in the eval pipeline for the
        producer side).
        """
        return self.eval_dir / "members" / cluster_id / "graph_results.joblib"

    def member_trade_universe(self, cluster_id: str) -> Path:
        """Trade-universe JSON staged from the training registry.

        Provides ``target_ids`` / ``elementary_ids`` for node colouring
        and trade-attribute lookup in the Selected-Trade card.
        """
        return self.eval_dir / "members" / cluster_id / "trade_universe.json"

    def member_training_curves(self, cluster_id: str) -> Path:
        """Per-epoch training curves parquet staged from the registry.

        Schema (B-level contract, trainer-side §11.15.1):

        ``epoch``      int32    (required — monotonically increasing)
        ``train_loss`` float32  (required — always present)
        *other cols*   float32  (any additional per-epoch series the
                                 trainer emitted — e.g. ``val_loss``,
                                 ``mae``, ``val_mae``).

        Consumed by the Cluster Deep-Dive training-curves chart.
        Column set is not fixed so the UI introspects available
        metrics at render time.
        """
        return self.eval_dir / "members" / cluster_id / "training_curves.parquet"
```

---

### A.F3 · PATCH `src/rade_ml_pt/ensemble/api/services/reader.py`

Add the following method to the `ArtifactReader` class immediately
after the `trade_graph(...)` method (no other changes):

```python
    # ── Per-member training curves (parquet) ──────────────────────
    def training_curves(self, cluster_id: str) -> pd.DataFrame:
        """Per-epoch training curves for one cluster member.

        Staged by the eval pipeline from
        ``{registry_dir}/{member_version}/training_curves.parquet``.

        Always contains ``epoch`` + ``train_loss``; any additional
        per-epoch series the trainer emitted (``val_loss``, ``mae``,
        ``val_mae``, …) are preserved untouched.  The UI introspects
        the available metric columns at render time — the column set
        is deliberately open so new trainer metrics flow through
        without coordinated schema bumps.

        Raises
        ------
        FileNotFoundError
            If ``training_curves.parquet`` is missing for the cluster.
            Callers should return a 404 — the absence of curves
            typically means a legacy run or a pipeline that skipped
            the trainer-side writer.
        """
        path = self.paths.member_training_curves(cluster_id)
        if not path.exists():
            raise FileNotFoundError(
                f"training_curves.parquet missing for cluster "
                f"'{cluster_id}' (looked for {path}). Re-run the eval "
                f"pipeline to stage member training artefacts."
            )
        return read_with_mtime_cache(path, _load_parquet)
```

The method reuses the module-local `_load_parquet` and
`read_with_mtime_cache` helpers already present in `reader.py` — no
new imports required.

---

### A.F4 · CREATE `src/rade_ml_pt/ensemble/api/models/training_curves.py`

```python
"""Pydantic response schemas for the training-curves endpoint.

Mirrors ``members/{cluster_id}/training_curves.parquet``, which the eval
pipeline stages from the training registry (trainer-side contract
§11.15.1).  Shape:

* ``epoch``       int    (required — monotonically increasing)
* ``train_loss``  float  (required — always emitted by the trainer)
* *other cols*    float  (any additional per-epoch series — e.g.
                          ``val_loss``, ``mae``, ``val_mae``)

The response is columnar (parallel arrays) so the UI can plot any metric
without having to re-zip rows client-side, and so the payload stays
compact even for long training runs.  ``metrics`` lists the optional
series names (excluding ``epoch`` + ``train_loss``) so consumers can
drive a metric picker without inspecting every column of ``series``.
"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class TrainingCurvesResponse(BaseModel):
    """Per-epoch training curves for one cluster member.

    ``series`` is a ``{metric_name: [values]}`` dict where every value
    array has length ``n_epochs``.  ``train_loss`` is always present;
    other keys are optional and depend on what the trainer emitted.
    ``epoch`` lives alongside the series dict for convenience — it is
    redundant with ``range(n_epochs)`` but the explicit representation
    keeps the x-axis correct even if the trainer ever switches to a
    non-zero start epoch.
    """

    cluster_id: str
    n_epochs: int
    epoch: List[int]
    series: Dict[str, List[float]] = Field(
        ...,
        description=(
            "Per-metric per-epoch values.  Always contains "
            "'train_loss'; other keys are optional per-epoch series "
            "the trainer emitted."
        ),
    )
    metrics: List[str] = Field(
        ...,
        description=(
            "Names of the optional metric series (i.e. keys of "
            "`series` excluding 'train_loss').  Surfaces the set of "
            "columns available for a UI metric picker."
        ),
    )


__all__ = ["TrainingCurvesResponse"]
```

---

### A.F5 · CREATE `src/rade_ml_pt/ensemble/api/routers/training_curves.py`

```python
"""``/prism/v1/clusters/{cluster_id}/training-curves`` — per-cluster
training curves.

Reads ``members/{cluster_id}/training_curves.parquet``, staged by the
eval pipeline from the training registry (trainer-side contract
§11.15.1).  Used by the Evaluation → Cluster Deep-Dive UI tab to render
a multi-series training curve plot with a metric-picker chip row.

The endpoint is cluster-scoped because each ensemble member carries its
own training history — the UI lets the user pivot to whichever cluster
they're inspecting without refetching other clusters' curves.
"""
from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.training_curves import (
    TrainingCurvesResponse,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["training-curves"])


@router.get(
    "/clusters/{cluster_id}/training-curves",
    response_model=TrainingCurvesResponse,
)
def get_training_curves(
    cluster_id: str,
    reader: ArtifactReader = Depends(get_reader),
) -> TrainingCurvesResponse:
    """Return per-epoch training curves for one cluster member.

    Responses are columnar — one array per metric — so the UI can plot
    any subset of metrics without having to re-zip rows.  ``train_loss``
    is always present; other series are whatever the trainer emitted.
    """
    try:
        df = reader.training_curves(cluster_id=cluster_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if df is None or df.empty or "epoch" not in df.columns or "train_loss" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=(
                f"training_curves.parquet for cluster '{cluster_id}' "
                "is present but malformed (missing 'epoch' / 'train_loss'). "
                "This is a pipeline contract break — file a bug."
            ),
        )

    df_sorted = df.sort_values("epoch").reset_index(drop=True)
    n_epochs = int(len(df_sorted))

    series: dict[str, list[float]] = {
        col: _to_clean_floats(df_sorted[col])
        for col in df_sorted.columns
        if col != "epoch"
    }
    metrics = sorted(k for k in series if k != "train_loss")

    return TrainingCurvesResponse(
        cluster_id=cluster_id,
        n_epochs=n_epochs,
        epoch=[int(v) for v in df_sorted["epoch"].tolist()],
        series=series,
        metrics=metrics,
    )


def _to_clean_floats(series) -> list[float]:
    """Coerce to ``list[float]`` and replace ``NaN`` / ``inf`` with ``0.0``.

    Pydantic / JSON can't round-trip non-finite floats, so anything
    pathological is clamped before serialisation.  Training runs that
    diverge (``nan`` loss) still produce a plottable payload; the UI
    flags the divergence separately via the summary chip strip.
    """
    out: list[float] = []
    for v in series.tolist():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if not math.isfinite(fv):
            fv = 0.0
        out.append(fv)
    return out
```

---

### A.F6 · PATCH `src/rade_ml_pt/ensemble/api/app.py`

Add the router import alongside the other router imports at the top of
the file:

```python
from src.rade_ml_pt.ensemble.api.routers.training_curves import (
    router as training_curves_router,
)
```

Register it inside `create_app()` alongside the existing routers (right
after `trade_graph_router`):

```python
    app.include_router(trade_graph_router)
    app.include_router(training_curves_router)
    app.include_router(quality_router)
    app.include_router(predictions_router)
```

No other changes.

---

### A.F7 · PATCH `src/rade_ml_pt/ensemble/api/client.py`

Add the response-model import alongside the other model imports at the
top of `client.py`:

```python
from src.rade_ml_pt.ensemble.api.models.training_curves import (
    TrainingCurvesResponse,
)
```

Add the method inside the `RadeApiClient` class (anywhere among the
resource methods — e.g. immediately after the `trade_graph` method):

```python
    # ── Training curves (per-cluster, per-epoch) ──────────────────

    def training_curves(self, *, cluster_id: str) -> TrainingCurvesResponse:
        """Per-epoch training curves for one cluster member.

        Always returns ``train_loss``; additional series depend on what
        the trainer emitted (``val_loss``, ``mae``, …).  See
        :mod:`..models.training_curves` for the schema.
        """
        return TrainingCurvesResponse(
            **self._get_json(
                f"/prism/v1/clusters/{cluster_id}/training-curves"
            )
        )
```

---

### A.U1 · REPLACE `src/ui/apps/rade_analytics/data/session.py`

```python
"""Typed per-user session state for the Rade Analytics UI.

Persisted to a ``dcc.Store(storage_type="session")`` as plain JSON.  The
contract here is:

* callbacks write a :class:`Session` instance, serialised via
  :meth:`Session.to_store`, into the store component;
* callbacks read the store via :meth:`Session.from_store`, which is
  defensive against partial / missing / stale payloads so the UI never
  crashes on a schema bump.

Adding a new field?  Bump :data:`SESSION_SCHEMA_VERSION` and leave the
field optional on :class:`Session`.  :meth:`from_store` will drop old
payloads whose version doesn't match.

Schema versions
---------------
* ``1`` — Phase A baseline (active_version, split, cluster_id, theme).
* ``2`` — Phase E.0 adds :class:`EvaluationState` under ``evaluation``.
* ``3`` — Phase E.1 adds ``portfolio_scatter_focus`` to
  :class:`EvaluationState` for the click-to-focus scatter behaviour.
* ``4`` — Phase E.3 adds ``trade_graph_cluster_id``,
  ``trade_graph_layout``, ``trade_graph_weight_threshold`` and
  ``trade_graph_selected_trade_id`` to :class:`EvaluationState` for the
  Cytoscape trade-graph tab.
* ``5`` — Phase E.4 adds ``deep_dive_cluster_id`` and
  ``deep_dive_selected_trade_id`` to :class:`EvaluationState` for the
  Cluster Deep-Dive sub-tab.  Kept separate from the trade-graph fields
  so the two sub-tabs can focus on different clusters without stepping
  on each other.
* ``6`` — Phase E.4 follow-up: Row 2 of the Cluster Deep-Dive now hosts
  a KPI card grid + a training-curves chart.  Adds
  ``deep_dive_curve_metrics`` (the optional metric chips the user has
  toggled on alongside ``train_loss``).  Empty list means "just the
  train curve".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

SESSION_SCHEMA_VERSION = 6


# ─────────────────────────────────────────────────────────────────────
# Evaluation sub-state (Phase E.0)
# ─────────────────────────────────────────────────────────────────────

# Valid values for :attr:`EvaluationState.active_subtab`.  Kept as a
# module constant so the router can validate URL slugs against the same
# list the session trusts.
EVALUATION_SUBTABS = ("portfolio", "cross-cluster", "trade-graph", "cluster")
DEFAULT_EVALUATION_SUBTAB: str = "portfolio"

# Valid values for the Portfolio sub-tab's "Group by" toggle.  ``None``
# means "no group-by" — render the aggregate view without a leaderboard.
EVALUATION_PORTFOLIO_GROUP_BY = (
    "desk", "product", "currency", "asset_class", "cluster",
)

# Cytoscape layout presets exposed in the Trade-Graph tab header.  Keep
# in sync with the radio options in ``layouts/evaluation/trade_graph.py``
# — the strings are what the Cytoscape component consumes verbatim.
EVALUATION_TRADE_GRAPH_LAYOUTS = ("cose", "concentric", "circle", "grid", "breadthfirst")
DEFAULT_TRADE_GRAPH_LAYOUT: str = "cose"

# Default weight threshold for edge rendering — drops edges whose
# ``weight < threshold``.  0.0 shows everything.
DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD: float = 0.0


@dataclass
class EvaluationFilters:
    """The global WHERE clause applied across every Evaluation sub-tab.

    All fields are **inclusive** filters — an empty list / ``None`` means
    "no filter on this dimension", which is the default.

    Date bounds are ISO-8601 strings (``YYYY-MM-DD``) rather than
    ``datetime`` objects so the value survives the ``dcc.Store`` JSON
    round-trip without custom (de)serialisers.
    """

    asset_class: List[str] = field(default_factory=list)
    currency:    List[str] = field(default_factory=list)
    desk:        List[str] = field(default_factory=list)
    product:     List[str] = field(default_factory=list)
    date_from:   Optional[str] = None
    date_to:     Optional[str] = None

    # ── Convenience ───────────────────────────────────────────────

    def active_chip_count(self) -> int:
        """Number of distinct dimensions that have at least one filter.

        Drives the "``n`` active" label next to the toggle button and
        the visibility of the "Clear all" action.
        """
        return sum(
            bool(x) for x in (
                self.asset_class,
                self.currency,
                self.desk,
                self.product,
                self.date_from or self.date_to,
            )
        )

    def is_empty(self) -> bool:
        return self.active_chip_count() == 0

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EvaluationFilters":
        if not isinstance(data, dict):
            return cls()
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class EvaluationState:
    """Everything the Evaluation page needs between sub-tab switches.

    Lives under :attr:`Session.evaluation`.  Each sub-tab reads what it
    needs from here; no Evaluation callback ever touches top-level
    ``Session`` fields directly.

    Portfolio-specific fields
    -------------------------
    * :attr:`portfolio_group_by` — which dimension the residual violin /
      scatter / leaderboard pivot on.  ``None`` means aggregate view.
    * :attr:`portfolio_scatter_focus` — a click-to-drill focus value on
      the grouped scatter (e.g. ``"rates"`` after the user clicks a
      Rates point when grouped by asset class).  Only meaningful while
      :attr:`portfolio_group_by` is set; callbacks clear it defensively
      whenever the group-by dimension changes so stale focus values
      never survive.

    Trade-graph-specific fields
    ---------------------------
    * :attr:`trade_graph_cluster_id` — local override of the top-bar
      cluster picker; when set, the Trade-Graph tab renders this cluster
      instead of :attr:`Session.cluster_id`.  ``None`` → follow the
      top-bar.
    * :attr:`trade_graph_layout` — Cytoscape layout name
      (``cose``, ``concentric``, ``circle``, ``grid``, ``breadthfirst``).
    * :attr:`trade_graph_weight_threshold` — edges with ``weight`` below
      this float are hidden; 0.0 shows everything.
    * :attr:`trade_graph_selected_trade_id` — currently-selected node,
      populates the Selected-Trade card.  Cleared on cluster switch.

    Deep-dive-specific fields
    -------------------------
    * :attr:`deep_dive_cluster_id` — local override of the top-bar
      cluster picker; when set, the Deep-Dive tab focuses on this
      cluster instead of :attr:`Session.cluster_id`.  The Trade-Graph
      "Open in Cluster Deep Dive" button writes this field then
      navigates to ``/evaluation/cluster``.
    * :attr:`deep_dive_selected_trade_id` — currently-highlighted trade
      on the Deep-Dive page.  Driven by scatter click-to-focus and the
      trades AgGrid row selection.  Cleared on cluster switch.
    * :attr:`deep_dive_curve_metrics` — optional metric chips the user
      has toggled on for the Row-2 training-curves chart.  The
      ``train_loss`` series is always shown implicitly; chips only
      control the extra overlays (``val_loss``, ``mae``, …).  Empty
      list ⇒ "just train_loss".  Persisted in session so the chip
      selection survives sub-tab navigation.
    """

    filters:                        EvaluationFilters = field(default_factory=EvaluationFilters)
    filter_bar_open:                bool = False
    active_subtab:                  str = DEFAULT_EVALUATION_SUBTAB
    portfolio_group_by:             Optional[str] = None
    portfolio_scatter_focus:        Optional[str] = None
    trade_graph_cluster_id:         Optional[str] = None
    trade_graph_layout:             str = DEFAULT_TRADE_GRAPH_LAYOUT
    trade_graph_weight_threshold:   float = DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD
    trade_graph_selected_trade_id:  Optional[str] = None
    deep_dive_cluster_id:           Optional[str] = None
    deep_dive_selected_trade_id:    Optional[str] = None
    deep_dive_curve_metrics:        List[str] = field(default_factory=list)

    # ── Serialisation ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters":                       self.filters.to_dict(),
            "filter_bar_open":               self.filter_bar_open,
            "active_subtab":                 self.active_subtab,
            "portfolio_group_by":            self.portfolio_group_by,
            "portfolio_scatter_focus":       self.portfolio_scatter_focus,
            "trade_graph_cluster_id":        self.trade_graph_cluster_id,
            "trade_graph_layout":            self.trade_graph_layout,
            "trade_graph_weight_threshold":  self.trade_graph_weight_threshold,
            "trade_graph_selected_trade_id": self.trade_graph_selected_trade_id,
            "deep_dive_cluster_id":          self.deep_dive_cluster_id,
            "deep_dive_selected_trade_id":   self.deep_dive_selected_trade_id,
            "deep_dive_curve_metrics":       list(self.deep_dive_curve_metrics),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EvaluationState":
        if not isinstance(data, dict):
            return cls()

        subtab = data.get("active_subtab", DEFAULT_EVALUATION_SUBTAB)
        if subtab not in EVALUATION_SUBTABS:
            subtab = DEFAULT_EVALUATION_SUBTAB

        group_by = data.get("portfolio_group_by")
        if group_by is not None and group_by not in EVALUATION_PORTFOLIO_GROUP_BY:
            group_by = None

        scatter_focus = data.get("portfolio_scatter_focus")
        if group_by is None:
            scatter_focus = None
        if scatter_focus is not None and not isinstance(scatter_focus, str):
            scatter_focus = None

        tg_cluster = data.get("trade_graph_cluster_id")
        if tg_cluster is not None and not isinstance(tg_cluster, str):
            tg_cluster = None

        tg_layout = data.get("trade_graph_layout", DEFAULT_TRADE_GRAPH_LAYOUT)
        if tg_layout not in EVALUATION_TRADE_GRAPH_LAYOUTS:
            tg_layout = DEFAULT_TRADE_GRAPH_LAYOUT

        raw_threshold = data.get(
            "trade_graph_weight_threshold", DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD,
        )
        try:
            tg_threshold = float(raw_threshold)
            if tg_threshold < 0.0:
                tg_threshold = 0.0
        except (TypeError, ValueError):
            tg_threshold = DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD

        tg_selected = data.get("trade_graph_selected_trade_id")
        if tg_selected is not None and not isinstance(tg_selected, str):
            tg_selected = None

        dd_cluster = data.get("deep_dive_cluster_id")
        if dd_cluster is not None and not isinstance(dd_cluster, str):
            dd_cluster = None

        dd_selected = data.get("deep_dive_selected_trade_id")
        if dd_selected is not None and not isinstance(dd_selected, str):
            dd_selected = None

        raw_metrics = data.get("deep_dive_curve_metrics")
        if isinstance(raw_metrics, list):
            # Dedup + stringify; anything non-string (e.g. None) is
            # dropped.  Ordering is preserved so the UI doesn't
            # reshuffle the user's chip selection on round-trip.
            seen: set[str] = set()
            dd_metrics: List[str] = []
            for m in raw_metrics:
                if isinstance(m, str) and m and m not in seen:
                    seen.add(m)
                    dd_metrics.append(m)
        else:
            dd_metrics = []

        return cls(
            filters=EvaluationFilters.from_dict(data.get("filters")),
            filter_bar_open=bool(data.get("filter_bar_open", False)),
            active_subtab=subtab,
            portfolio_group_by=group_by,
            portfolio_scatter_focus=scatter_focus,
            trade_graph_cluster_id=tg_cluster,
            trade_graph_layout=tg_layout,
            trade_graph_weight_threshold=tg_threshold,
            trade_graph_selected_trade_id=tg_selected,
            deep_dive_cluster_id=dd_cluster,
            deep_dive_selected_trade_id=dd_selected,
            deep_dive_curve_metrics=dd_metrics,
        )


# ─────────────────────────────────────────────────────────────────────
# Top-level session
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Session:
    """User session state.  Every field must be JSON-serialisable."""

    # Which ensemble version this tab is looking at.  ``None`` on a
    # fresh tab before the splash picker resolves.
    active_version: Optional[str] = None

    # Global filters shared across tabs.
    split: Literal["train", "val", "test"] = "test"
    cluster_id: Optional[str] = None

    # Presentation toggles.
    theme: Literal["dark", "light"] = "dark"

    # Per-page sub-state.
    evaluation: EvaluationState = field(default_factory=EvaluationState)

    # Schema book-keeping — never surface to UI.
    schema_version: int = field(default=SESSION_SCHEMA_VERSION, repr=False)

    # ── Serialisation ─────────────────────────────────────────────

    def to_store(self) -> Dict[str, Any]:
        """Return a JSON-safe dict suitable for ``dcc.Store.data``."""
        return {
            "active_version": self.active_version,
            "split":          self.split,
            "cluster_id":     self.cluster_id,
            "theme":          self.theme,
            "evaluation":     self.evaluation.to_dict(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_store(cls, data: Optional[Dict[str, Any]]) -> "Session":
        """Hydrate from the store; fall back to defaults on any issue."""
        if not data or not isinstance(data, dict):
            return cls()
        if data.get("schema_version") != SESSION_SCHEMA_VERSION:
            # Stale schema — drop everything rather than risk mixing
            # fields.  A fresh Session() has sensible defaults.
            return cls()
        split = data.get("split", "test")
        if split not in ("train", "val", "test"):
            split = "test"
        theme = data.get("theme", "dark")
        if theme not in ("dark", "light"):
            theme = "dark"
        return cls(
            active_version=data.get("active_version"),
            split=split,                                      # type: ignore[arg-type]
            cluster_id=data.get("cluster_id"),
            theme=theme,                                      # type: ignore[arg-type]
            evaluation=EvaluationState.from_dict(data.get("evaluation")),
        )

    # ── Convenience mutators (returns new instance; Session is logically immutable per render) ──

    def with_version(self, version: str) -> "Session":
        payload = self.to_store()
        payload["active_version"] = version
        return Session.from_store(payload)

    def with_split(self, split: str) -> "Session":
        payload = self.to_store()
        payload["split"] = split
        return Session.from_store(payload)

    def with_cluster(self, cluster_id: Optional[str]) -> "Session":
        payload = self.to_store()
        payload["cluster_id"] = cluster_id
        return Session.from_store(payload)

    def with_evaluation(self, evaluation: EvaluationState) -> "Session":
        """Swap in a new :class:`EvaluationState`."""
        payload = self.to_store()
        payload["evaluation"] = evaluation.to_dict()
        return Session.from_store(payload)


__all__ = [
    "DEFAULT_EVALUATION_SUBTAB",
    "DEFAULT_TRADE_GRAPH_LAYOUT",
    "DEFAULT_TRADE_GRAPH_WEIGHT_THRESHOLD",
    "EVALUATION_PORTFOLIO_GROUP_BY",
    "EVALUATION_SUBTABS",
    "EVALUATION_TRADE_GRAPH_LAYOUTS",
    "EvaluationFilters",
    "EvaluationState",
    "SESSION_SCHEMA_VERSION",
    "Session",
]
```

---

### A.U2 · PATCH `src/ui/apps/rade_analytics/data/backend.py`

Two additions — all other code in `backend.py` is unchanged.

**1.** In `_bind_cached_methods` (between the trade-graph and
completeness cache lines), register a cache for training curves:

```python
        # Training curves are per-cluster + per-epoch (tens of floats);
        # payload is small so the default TTL is fine and re-entering
        # the deep-dive tab is always instant.
        self._training_curves_cached = cache.memoize(timeout=ttl)(
            self._fetch_training_curves
        )
```

**2.** Add a raw fetcher and a public UI method:

```python
    def _fetch_training_curves(self, cluster_id: str) -> pd.DataFrame:
        """Per-epoch training curves for one cluster, as a DataFrame.

        One row per epoch.  Columns: ``epoch``, ``train_loss``, plus
        whatever additional per-epoch series the trainer emitted.  The
        available metric names (i.e. ``series`` keys excluding
        ``train_loss``) are stashed on ``df.attrs["metrics"]`` so the
        metric-picker chip row can render without a second server hit.
        """
        resp = self._client.training_curves(cluster_id=cluster_id)
        data: Dict[str, Any] = {"epoch": resp.epoch}
        for name, values in resp.series.items():
            data[name] = values
        df = pd.DataFrame(data)
        df.attrs["metrics"] = list(resp.metrics)
        df.attrs["cluster_id"] = resp.cluster_id
        return df

    # ── Training curves (per-cluster, per-epoch) ──────────────────

    def training_curves_df(
        self, *, cluster_id: str,
    ) -> BackendResult[pd.DataFrame]:
        """Per-epoch training curves for one cluster as a DataFrame.

        Always carries ``epoch`` + ``train_loss``; any additional
        per-epoch series the trainer emitted (``val_loss``, ``mae``,
        ``val_mae``, …) are preserved and surfaced via
        ``df.attrs["metrics"]`` so a metric picker can be driven
        without a second round-trip.
        """
        return self._wrap(self._training_curves_cached, cluster_id)
```

Place `_fetch_training_curves` alongside the other `_fetch_*` methods
(e.g. next to `_fetch_trade_graph`); place `training_curves_df`
alongside the other public `*_df` methods (e.g. after `trade_graph`).

---

### A.U3 · CREATE `src/ui/apps/rade_analytics/figures/training_curves.py`

```python
"""Training-curves chart for the Cluster Deep-Dive sub-tab (Phase E.4).

Consumes the DataFrame returned by
:meth:`~src.ui.apps.rade_analytics.data.backend.RadeBackend.training_curves_df`
— one row per epoch, columns = ``epoch`` + any per-epoch series the
trainer emitted (``train_loss``, ``val_loss``, ``mae``, …).

Render behaviour
----------------
* ``train_loss`` is **always** plotted — it is the canonical "what did
  the model learn" trace and the chart is meaningless without it.
* Additional metrics selected via the chip row are each drawn as a
  secondary line.  ``val_*`` counterparts are auto-paired in the same
  hue (dashed) as their ``train_*`` sibling, so "train + val of mae"
  reads as one colour family instead of two unrelated lines.
* Palette pickeffs through :func:`color_for_index` so chip order in
  the UI matches trace order in the legend.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout, rgba


# Dashed trace for val_* series; plain line for everything else.  Kept
# module-local so it's trivial to tweak later (e.g. dotted for test_*).
_VAL_DASH = "dash"


def _is_val_pair(primary: str, candidate: str) -> bool:
    """True when ``candidate`` is the ``val_*`` sibling of ``primary``.

    Matches both ``val_loss`` (paired with ``train_loss`` / ``loss``)
    and ``val_<metric>`` (paired with ``<metric>``).
    """
    if primary in ("train_loss", "loss") and candidate == "val_loss":
        return True
    if not primary.startswith("val_") and candidate == f"val_{primary}":
        return True
    return False


def training_curves_chart(
    df: Optional[pd.DataFrame],
    *,
    selected_metrics: Optional[Sequence[str]] = None,
    available_metrics: Optional[Iterable[str]] = None,
) -> go.Figure:
    """Render training curves, always including ``train_loss``.

    Parameters
    ----------
    df
        DataFrame from ``training_curves_df`` (one row per epoch).
        ``None`` or empty returns an empty-state figure.
    selected_metrics
        Optional metric names to overlay alongside ``train_loss``.
        Values that are not columns of ``df`` are silently dropped.
        ``val_*`` counterparts of selected metrics are auto-added as
        dashed overlays in the same hue.
    available_metrics
        Optional pre-computed list of available metric names (from
        ``df.attrs["metrics"]``).  Only used to validate
        ``selected_metrics`` — passing it lets callers share one
        look-up across KPIs + chart + chip row without reading
        ``df.columns`` twice.
    """
    if df is None or df.empty or "epoch" not in df.columns or "train_loss" not in df.columns:
        return empty_figure("Training curves unavailable for this cluster.")

    epochs = df["epoch"].tolist()
    validated = _validated_selection(df, selected_metrics, available_metrics)

    fig = go.Figure()

    # Primary trace — always train_loss.  Uses palette[0] so the colour
    # stays stable no matter what secondary metric is also selected.
    _add_primary(fig, epochs, df["train_loss"].tolist(), color_for_index(0))

    # val_loss is paired with train_loss (same hue, dashed) when it is
    # in the user's chip selection — always shown alongside its sibling
    # rather than using a separate palette slot.
    val_loss_selected = "val_loss" in validated and "val_loss" in df.columns
    if val_loss_selected:
        _add_pair(
            fig, epochs, df["val_loss"].tolist(), color_for_index(0), "val_loss",
        )

    # Secondary metrics — pair train/val per colour slot.
    slot = 1
    for metric in validated:
        if metric == "val_loss":
            # Already rendered alongside train_loss above.
            continue
        hue = color_for_index(slot)
        _add_primary(fig, epochs, df[metric].tolist(), hue, name=metric)
        val_name = f"val_{metric}" if not metric.startswith("val_") else None
        if val_name and val_name in df.columns:
            _add_pair(fig, epochs, df[val_name].tolist(), hue, val_name)
        slot += 1

    fig.update_layout(
        **rade_layout(
            show_legend=True,
            hovermode="x unified",
            xaxis={"title": "Epoch"},
            yaxis={"title": "Value"},
        )
    )
    return fig


# ── internal helpers ───────────────────────────────────────────────

def _validated_selection(
    df: pd.DataFrame,
    selected: Optional[Sequence[str]],
    available: Optional[Iterable[str]],
) -> List[str]:
    """Drop unknown / duplicate / ``train_loss`` entries from selection.

    ``train_loss`` is always drawn so we strip it from the user
    selection to avoid a duplicate primary trace; every other name
    must exist as a column in ``df``.
    """
    if not selected:
        return []
    cols = set(df.columns)
    avail = set(available) if available is not None else cols
    out: List[str] = []
    seen: set[str] = set()
    for m in selected:
        if not isinstance(m, str) or not m:
            continue
        if m == "train_loss":
            continue
        if m not in cols or m not in avail:
            continue
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def _add_primary(
    fig: go.Figure,
    x: Sequence[int],
    y: Sequence[float],
    color: str,
    *,
    name: str = "train_loss",
) -> None:
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": 2.2},
        hovertemplate=f"{name}: %{{y:.6f}}<extra>epoch %{{x}}</extra>",
    ))


def _add_pair(
    fig: go.Figure,
    x: Sequence[int],
    y: Sequence[float],
    color: str,
    name: str,
) -> None:
    """Add a ``val_*`` trace in the same hue but dashed + translucent."""
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        name=name,
        line={"color": rgba(color, 0.75), "width": 1.6, "dash": _VAL_DASH},
        hovertemplate=f"{name}: %{{y:.6f}}<extra>epoch %{{x}}</extra>",
    ))


__all__ = ["training_curves_chart"]
```

---

### A.U4 · REPLACE `src/ui/apps/rade_analytics/figures/__init__.py`

```python
"""Figure builders for the Rade Analytics Dash UI.

This is the single place every callback goes to for a Plotly
``go.Figure``.  Keeping chart construction out of the callback modules
means:

* Callbacks stay focused on fetch + state plumbing.
* Figures can be unit-tested headlessly (``fig.to_dict()`` snapshots).
* Visual tweaks land in one diff — no hunting through every callback
  module for a font-size bump.

Modules shipped
---------------
* :mod:`._theme`                   — shared layout defaults, palette, helpers.
* :mod:`.cluster_deep_dive_charts` — per-cluster PnL band + per-trade
  violin / scatter (Phase E.4).
* :mod:`.distributions`            — residual violin (aggregate + grouped).
* :mod:`.graph_charts`             — graph density histogram + edges vs nodes.
* :mod:`.scatter`                  — predicted-vs-actual scatter (+ focus).
* :mod:`.timeseries`               — portfolio PnL + rolling error band.
* :mod:`.training_curves`          — per-cluster training loss + metric
  overlays (Phase E.4 Row 2).
"""
from __future__ import annotations

from ._theme import (
    CATEGORY_PALETTE,
    color_for_index,
    empty_figure,
    rade_layout,
    rgba,
)
from .cluster_deep_dive_charts import (
    per_trade_residual_violin,
    per_trade_scatter,
    predicted_vs_actual_band,
)
from .distributions import residual_violin
from .graph_charts import density_distribution, edges_vs_nodes_scatter
from .scatter import pred_actual_scatter
from .timeseries import error_over_time, portfolio_pnl
from .training_curves import training_curves_chart

__all__ = [
    "CATEGORY_PALETTE",
    "color_for_index",
    "density_distribution",
    "edges_vs_nodes_scatter",
    "empty_figure",
    "error_over_time",
    "per_trade_residual_violin",
    "per_trade_scatter",
    "portfolio_pnl",
    "pred_actual_scatter",
    "predicted_vs_actual_band",
    "rade_layout",
    "residual_violin",
    "rgba",
    "training_curves_chart",
]
```

---

### A.U5 · REPLACE `src/ui/apps/rade_analytics/layouts/evaluation/cluster_deep_dive.py`

```python
"""Evaluation → Cluster Deep-Dive sub-tab layout (Phase E.4).

Five-row layout laser-focused on a single cluster:

    Row 1 · Header band       (cluster picker · attribute chips ·
                               open Trade-Graph link)
    Row 2 · Training          (KPI grid 2×2: MAE/RMSE/R²/Coverage  |
                               training curves with a metric chip
                               filter for optional overlays)
    Row 3 · Time-series       (residual over time  |  predicted vs
                               target PnL with error band)
    Row 4 · Per-trade charts  (residual violin target/elementary  |
                               per-trade scatter target/elementary,
                               with click-to-highlight into Row 5)
    Row 5 · Trades grid       (AgGrid — per-trade MAE / RMSE / p95 /
                               mean_residual; row click ↔ Row 4
                               scatter)

The page deliberately has no filter bar — Cluster Deep-Dive scopes
everything to the single selected cluster, so the top-level Evaluation
filter chrome is redundant here.  Split inherits from the topbar as
everywhere else.

All dynamic ids live in :data:`CLUSTER_DEEP_DIVE_IDS` so callbacks
never hardcode strings.  Callbacks live in
:mod:`..callbacks.cluster_deep_dive_cb`.
"""
from __future__ import annotations

from typing import Any, Dict, List

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ...components.ag_grid_table import AgGridTable
from ...components.chart_container import ChartContainer
from ...components.kpi_card import KpiCard


CLUSTER_DEEP_DIVE_IDS: Dict[str, str] = {
    "root":                "eval-cluster-root",

    # Header band
    "cluster_select":      "eval-cluster-cluster-select",
    "attribute_chips":     "eval-cluster-attribute-chips",
    "open_trade_graph_btn": "eval-cluster-open-trade-graph-btn",

    # Row 2 — KPI grid
    "kpi_mae_card":        "eval-cluster-kpi-mae-card",
    "kpi_mae_value":       "eval-cluster-kpi-mae-value",
    "kpi_rmse_card":       "eval-cluster-kpi-rmse-card",
    "kpi_rmse_value":      "eval-cluster-kpi-rmse-value",
    "kpi_r2_card":         "eval-cluster-kpi-r2-card",
    "kpi_r2_value":        "eval-cluster-kpi-r2-value",
    "kpi_coverage_card":   "eval-cluster-kpi-coverage-card",
    "kpi_coverage_value":  "eval-cluster-kpi-coverage-value",

    # Row 2 — training curves
    "curves_chart":        "eval-cluster-curves-chart",
    "curves_chip_group":   "eval-cluster-curves-chip-group",
    "curves_chip_empty":   "eval-cluster-curves-chip-empty",

    # Row 3 — timeseries
    "residual_ts_chart":   "eval-cluster-residual-ts-chart",
    "pnl_band_chart":      "eval-cluster-pnl-band-chart",

    # Row 4 — per-trade
    "per_trade_violin":    "eval-cluster-per-trade-violin",
    "per_trade_scatter":   "eval-cluster-per-trade-scatter",
    "selected_trade_chip": "eval-cluster-selected-trade-chip",
    "selected_trade_label": "eval-cluster-selected-trade-label",
    "selected_trade_clear_btn": "eval-cluster-selected-trade-clear-btn",

    # Row 5 — trades grid
    "trades_grid":         "eval-cluster-trades-grid",
    "trades_grid_card":    "eval-cluster-trades-grid-card",
    "trades_grid_empty":   "eval-cluster-trades-grid-empty",
    "trades_grid_wrap":    "eval-cluster-trades-grid-wrap",

    # Ephemeral stores
    "store_trade_types":   "eval-cluster-trade-types-store",
    "store_curve_metrics": "eval-cluster-curve-metrics-store",
}


# ─────────────────────────────────────────────────────────────────────
# Row 1 — Header band
# ─────────────────────────────────────────────────────────────────────


def _header_band() -> html.Div:
    """Sticky header with cluster picker + attribute chips.

    KPIs used to live here but Phase E.4 (Row 2 revision) moved them
    alongside the training-curves chart so the header stays a thin
    navigation / context band.
    """
    cluster_picker = html.Div(
        className="flex flex-col gap-1 min-w-[220px]",
        children=[
            html.Span(
                "Cluster",
                className="text-[11px] uppercase tracking-wider text-slate-400",
            ),
            dmc.Select(
                id=CLUSTER_DEEP_DIVE_IDS["cluster_select"],
                data=[],
                placeholder="Select a cluster…",
                searchable=True,
                clearable=False,
                size="sm",
            ),
        ],
    )

    attribute_chips = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["attribute_chips"],
        className="flex items-center gap-1 flex-wrap min-h-[32px]",
    )

    open_trade_graph_btn = dmc.Button(
        "Trade-Graph",
        id=CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"],
        variant="light",
        color="violet",
        size="sm",
        leftSection=DashIconify(icon="tabler:share-2", width=16),
    )

    top_row = html.Div(
        className="flex items-end gap-4 flex-wrap",
        children=[
            cluster_picker,
            html.Div(
                className="flex flex-col gap-1 flex-1 min-w-[240px]",
                children=[
                    html.Span(
                        "Attributes",
                        className="text-[11px] uppercase tracking-wider text-slate-400",
                    ),
                    attribute_chips,
                ],
            ),
            open_trade_graph_btn,
        ],
    )

    return html.Div(
        className="rade-card flex flex-col gap-3 sticky top-0 z-10",
        children=[top_row],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — KPI grid + training curves
# ─────────────────────────────────────────────────────────────────────


def _kpi_grid() -> html.Div:
    """2×2 KPI grid — MAE, RMSE, R², Coverage.

    R² and Coverage are derived client-side from the cluster time-series
    (``rade_analytics.callbacks.cluster_deep_dive_cb``); MAE and RMSE
    come straight from ``per_member_metrics``.
    """
    return html.Div(
        className="grid grid-cols-2 gap-3 self-start",
        children=[
            KpiCard(
                label="MAE",
                value="—",
                card_id=CLUSTER_DEEP_DIVE_IDS["kpi_mae_card"],
                value_id=CLUSTER_DEEP_DIVE_IDS["kpi_mae_value"],
                icon="tabler:arrow-narrow-down",
            ),
            KpiCard(
                label="RMSE",
                value="—",
                card_id=CLUSTER_DEEP_DIVE_IDS["kpi_rmse_card"],
                value_id=CLUSTER_DEEP_DIVE_IDS["kpi_rmse_value"],
                icon="tabler:square-root",
            ),
            KpiCard(
                label="R²",
                value="—",
                card_id=CLUSTER_DEEP_DIVE_IDS["kpi_r2_card"],
                value_id=CLUSTER_DEEP_DIVE_IDS["kpi_r2_value"],
                icon="tabler:chart-dots",
            ),
            KpiCard(
                label="Coverage",
                value="—",
                card_id=CLUSTER_DEEP_DIVE_IDS["kpi_coverage_card"],
                value_id=CLUSTER_DEEP_DIVE_IDS["kpi_coverage_value"],
                icon="tabler:target",
            ),
        ],
    )


def _curves_chip_group() -> html.Div:
    """Multi-select metric chips for the training-curves overlay filter.

    ``train_loss`` is always shown on the chart and so is deliberately
    absent from the chip group — chips only pick extra series to
    overlay (``val_loss``, ``mae``, ``val_mae``, …).  The callback
    populates the chip group based on ``df.attrs["metrics"]``; the
    empty state message renders while the list is still being fetched
    or when the trainer emitted only ``train_loss``.
    """
    return html.Div(
        className="flex flex-col gap-2",
        children=[
            html.Div(
                className="flex items-center justify-between",
                children=[
                    html.Span(
                        "Overlay metrics",
                        className="text-[11px] uppercase tracking-wider text-slate-400",
                    ),
                    html.Span(
                        "train_loss always shown",
                        className="text-[11px] text-slate-500",
                    ),
                ],
            ),
            html.Div(
                className="flex items-center gap-1 flex-wrap min-h-[28px]",
                children=[
                    dmc.ChipGroup(
                        id=CLUSTER_DEEP_DIVE_IDS["curves_chip_group"],
                        multiple=True,
                        value=[],
                        children=[],
                    ),
                    html.Span(
                        "No additional metrics emitted for this cluster.",
                        id=CLUSTER_DEEP_DIVE_IDS["curves_chip_empty"],
                        className="text-xs text-slate-500",
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )


def _row_training() -> html.Div:
    """Row 2 — KPI grid on the left, training curves chart on the right."""
    return html.Div(
        # 2/5 : 3/5 split on wide screens — gives the KPI grid enough
        # room to breathe without crowding the chart.  Collapses to a
        # single stack on narrow viewports.
        className="grid grid-cols-1 lg:grid-cols-5 gap-3 items-stretch",
        children=[
            html.Div(
                className="lg:col-span-2 flex flex-col gap-3",
                children=[_kpi_grid()],
            ),
            html.Div(
                className="lg:col-span-3 flex flex-col gap-2",
                children=[
                    ChartContainer(
                        title="Training curves",
                        subtitle="Per-epoch train loss (+ selected overlays)",
                        graph_id=CLUSTER_DEEP_DIVE_IDS["curves_chart"],
                        height=300,
                    ),
                    _curves_chip_group(),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — timeseries
# ─────────────────────────────────────────────────────────────────────


def _row_timeseries() -> html.Div:
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-2 gap-3",
        children=[
            ChartContainer(
                title="Residual over time",
                subtitle="Rolling absolute error with ±1σ band",
                graph_id=CLUSTER_DEEP_DIVE_IDS["residual_ts_chart"],
                height=300,
            ),
            ChartContainer(
                title="Predicted vs Target PnL",
                subtitle="Shaded band = prediction error",
                graph_id=CLUSTER_DEEP_DIVE_IDS["pnl_band_chart"],
                height=300,
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — per-trade charts
# ─────────────────────────────────────────────────────────────────────


def _selected_trade_chip() -> html.Div:
    """Focus-state chip shown in the per-trade scatter card header."""
    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["selected_trade_chip"],
        className="rade-focus-chip flex items-center gap-1",
        style={"display": "none"},
        children=[
            DashIconify(
                icon="tabler:target",
                width=12,
                className="text-emerald-400",
            ),
            html.Span(
                "Trade: —",
                id=CLUSTER_DEEP_DIVE_IDS["selected_trade_label"],
                className="text-xs text-slate-300",
            ),
            html.Button(
                "× Clear",
                id=CLUSTER_DEEP_DIVE_IDS["selected_trade_clear_btn"],
                className="rade-focus-chip-close",
                **{"aria-label": "Clear trade selection"},
            ),
        ],
    )


def _row_per_trade_charts() -> html.Div:
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-2 gap-3",
        children=[
            ChartContainer(
                title="Per-trade residual distribution",
                subtitle="Split by target / elementary",
                graph_id=CLUSTER_DEEP_DIVE_IDS["per_trade_violin"],
                height=360,
            ),
            ChartContainer(
                title="Per-trade bias vs magnitude",
                subtitle=(
                    "x: mean_residual  ·  y: MAE  ·  click a point "
                    "to highlight in the grid"
                ),
                graph_id=CLUSTER_DEEP_DIVE_IDS["per_trade_scatter"],
                height=360,
                actions=[_selected_trade_chip()],
                config={"doubleClick": "reset"},
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 5 — trades grid
# ─────────────────────────────────────────────────────────────────────


def _row_trades_grid() -> html.Div:
    header = html.Div(
        className="flex items-center justify-between",
        children=[
            html.Div(
                className="flex flex-col",
                children=[
                    html.Div(
                        "Trades in this cluster",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Per-trade metrics across the active split.  "
                        "Click a row to highlight the trade in the scatter above.",
                        className="text-xs text-slate-500",
                    ),
                ],
            ),
        ],
    )

    empty_state = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["trades_grid_empty"],
        className="rade-list-empty flex flex-col items-center justify-center gap-2 py-8",
        children=[
            DashIconify(icon="tabler:table-off", width=22, className="text-slate-600"),
            html.Div(
                "Pick a cluster to see its trades.",
                className="text-xs text-slate-500 text-center max-w-sm",
            ),
        ],
    )

    grid = AgGridTable(
        grid_id=CLUSTER_DEEP_DIVE_IDS["trades_grid"],
        column_defs=_initial_column_defs(),
        row_data=[],
        height=360,
        className="rade-cluster-trades-grid",
        grid_options={"rowSelection": "single"},
        getRowId="params.data.trade_id",
    )

    grid_wrapper = html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["trades_grid_wrap"],
        className="rade-cluster-trades-grid-wrap",
        style={"display": "none"},
        children=grid,
    )

    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["trades_grid_card"],
        className="rade-card flex flex-col gap-3",
        children=[header, empty_state, grid_wrapper],
    )


def _initial_column_defs() -> List[Dict[str, Any]]:
    """Bootstrap columnDefs — the callback rewrites these once data lands."""
    return [
        {"field": "trade_id",      "headerName": "Trade",        "flex": 2, "minWidth": 160},
        {"field": "trade_type",    "headerName": "Type",         "flex": 1, "minWidth": 100},
        {"field": "mae",           "headerName": "MAE",          "flex": 1, "type": "numericColumn"},
        {"field": "rmse",          "headerName": "RMSE",         "flex": 1, "type": "numericColumn"},
        {"field": "p95_ae",        "headerName": "P95 |err|",    "flex": 1, "type": "numericColumn"},
        {"field": "mean_residual", "headerName": "Mean resid.",  "flex": 1, "type": "numericColumn"},
        {"field": "n_scenarios",   "headerName": "Scenarios",    "flex": 1, "type": "numericColumn"},
    ]


# ─────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────


def build_cluster_deep_dive() -> html.Div:
    """Assemble the Cluster Deep-Dive sub-tab (pure layout, no callbacks)."""
    return html.Div(
        id=CLUSTER_DEEP_DIVE_IDS["root"],
        className="rade-evaluation-subtab flex flex-col gap-4",
        children=[
            _header_band(),
            _row_training(),
            _row_timeseries(),
            _row_per_trade_charts(),
            _row_trades_grid(),
            dcc.Store(
                id=CLUSTER_DEEP_DIVE_IDS["store_trade_types"],
                data={},
                storage_type="memory",
            ),
            dcc.Store(
                id=CLUSTER_DEEP_DIVE_IDS["store_curve_metrics"],
                data=[],
                storage_type="memory",
            ),
        ],
    )


__all__ = ["CLUSTER_DEEP_DIVE_IDS", "build_cluster_deep_dive"]
```

---

### A.U6 · REPLACE `src/ui/apps/rade_analytics/callbacks/cluster_deep_dive_cb.py`

```python
"""Evaluation → Cluster Deep-Dive sub-tab callbacks (Phase E.4).

Ten callbacks drive the tab — each small, each scoped to a single Output
contract so a future refactor can swap any one without touching the
others:

1. :func:`_register_hydrate_on_entry`            — URL ``?cid=`` +
   ``clusters`` → cluster ``Select`` options & value.
2. :func:`_register_sync_selection`              — cluster picker,
   scatter click, grid row click and the clear-chip button → session
   (``deep_dive_cluster_id`` / ``deep_dive_selected_trade_id``).
3. :func:`_register_sync_curve_metrics`          — overlay-metric chip
   group ``value`` → session (``deep_dive_curve_metrics``).
4. :func:`_register_render_header`               — session → attribute
   chips + "Trade-Graph" link enabled state.
5. :func:`_register_render_kpis`                 — session → Row-2 KPI
   grid values (MAE / RMSE / R² / Coverage).  R² and Coverage are
   derived client-side from the cluster timeseries; Coverage uses a
   ``|error| ≤ MAE`` tolerance.
6. :func:`_register_render_training_curves`      — session → training-
   curves figure + chip group children + chip empty-state visibility.
7. :func:`_register_render_timeseries`           — session → residual-
   over-time + predicted-vs-actual-with-band figures (Row 3).
8. :func:`_register_render_trade_data`           — session → per-trade
   violin + AgGrid rowData/columnDefs + ``trade_types`` memory store
   (Row 4 + Row 5 data feed).
9. :func:`_register_render_scatter`              — trade_types store +
   session → per-trade scatter figure + focus-chip visibility.
10. :func:`_register_navigate_to_trade_graph`    — "Trade-Graph"
    button on this page → ``/evaluation/trade-graph`` with the
    current cluster pinned as an override.
11. :func:`_register_navigate_from_trade_graph`  — "Open in Cluster
    Deep Dive" on the Trade-Graph tab → ``/evaluation/cluster`` with
    the trade-graph's cluster + selected trade copied into the Deep-
    Dive session slots.

Data contract
-------------
Row 4 / Row 5 coloring + violin grouping rely on a
``{trade_id: "target" | "elementary"}`` map which the backend exposes
via :meth:`RadeBackend.trade_graph`.  When a cluster has no graph
staged we degrade gracefully to an aggregate single-violin view and an
uncoloured scatter — the rest of the page keeps working.

Every fetch goes through :class:`RadeBackend`; the cache layer there
coalesces duplicate requests within a render tick.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, html
from dash.exceptions import PreventUpdate

from ..data.session import Session
from ..figures import (
    empty_figure,
    error_over_time,
    per_trade_residual_violin,
    per_trade_scatter,
    predicted_vs_actual_band,
    training_curves_chart,
)
from ..layouts.evaluation.cluster_deep_dive import CLUSTER_DEEP_DIVE_IDS
from ..layouts.evaluation.trade_graph import TRADE_GRAPH_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


_DEEP_DIVE_PATH = "/evaluation/cluster"
_TRADE_GRAPH_PATH = "/evaluation/trade-graph"
_PLACEHOLDER = "—"

# ATTRIBUTE column names expected on ``clusters_df``.  Kept in sync with
# ``ClustersResponse.attribute_names`` the API emits.
_ATTRIBUTE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Asset class", "asset_class"),
    ("Desk",        "desk"),
    ("Product",     "product_code"),
    ("Currency",    "currency_code"),
)


# ─────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────


def _fmt_float(x: Optional[float], *, precision: int = 4) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = float(x)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val:.{precision}f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = float(x)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val * 100:.1f}%"


def _parse_cid_from_search(search: Optional[str]) -> Optional[str]:
    """Extract the ``?cid=`` query param from the URL search string.

    Returns ``None`` on anything we can't parse so callers just fall
    back to the session default.
    """
    if not search:
        return None
    try:
        params = parse_qs(search.lstrip("?"))
    except (ValueError, TypeError):
        return None
    values = params.get("cid")
    if not values:
        return None
    candidate = values[0]
    return candidate if isinstance(candidate, str) and candidate else None


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Cluster-Deep-Dive sub-tab callback to ``app``."""
    _register_hydrate_on_entry(app, backend)
    _register_sync_selection(app)
    _register_sync_curve_metrics(app)
    _register_render_header(app, backend)
    _register_render_kpis(app, backend)
    _register_render_training_curves(app, backend)
    _register_render_timeseries(app, backend)
    _register_render_trade_data(app, backend)
    _register_render_scatter(app)
    _register_navigate_to_trade_graph(app)
    _register_navigate_from_trade_graph(app)


# ═════════════════════════════════════════════════════════════════════
# 1. Hydrate cluster Select on URL entry (optionally from ?cid=)
# ═════════════════════════════════════════════════════════════════════


def _register_hydrate_on_entry(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["cluster_select"], "data"),
        Output(CLUSTER_DEEP_DIVE_IDS["cluster_select"], "value"),
        Input(SHELL_IDS["url"],                         "pathname"),
        Input(SHELL_IDS["url"],                         "search"),
        State(SHELL_IDS["session_store"],               "data"),
    )
    def _hydrate(
        pathname:     Optional[str],
        search:       Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        res = backend.clusters_df()
        if not res.ok or res.data is None or res.data.empty:
            return [], None

        session = Session.from_store(session_data)
        df = res.data
        options = [
            {"value": cid, "label": cid} for cid in sorted(df["cluster_id"].unique())
        ]
        valid_ids = {o["value"] for o in options}

        # URL override wins over session, which wins over top-bar, which
        # wins over "first cluster".  The picker must never be empty.
        candidate = (
            _parse_cid_from_search(search)
            or session.evaluation.deep_dive_cluster_id
            or session.cluster_id
            or (options[0]["value"] if options else None)
        )
        if candidate not in valid_ids:
            candidate = options[0]["value"] if options else None

        return options, candidate


# ═════════════════════════════════════════════════════════════════════
# 2. Sync picker / scatter-click / grid-click / clear-btn → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_selection(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],                "data", allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["cluster_select"],    "value"),
        Input(CLUSTER_DEEP_DIVE_IDS["per_trade_scatter"], "clickData"),
        Input(CLUSTER_DEEP_DIVE_IDS["trades_grid"],       "cellClicked"),
        Input(CLUSTER_DEEP_DIVE_IDS["selected_trade_clear_btn"], "n_clicks"),
        State(SHELL_IDS["session_store"],                 "data"),
        prevent_initial_call=True,
    )
    def _sync(
        cluster:      Optional[str],
        click_data:   Optional[Dict[str, Any]],
        cell_clicked: Optional[Dict[str, Any]],
        clear_clicks: Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        changed = False

        if trigger == CLUSTER_DEEP_DIVE_IDS["cluster_select"]:
            new_cluster = cluster if cluster else None
            if ev.deep_dive_cluster_id != new_cluster:
                ev.deep_dive_cluster_id = new_cluster
                # Cluster change invalidates any per-trade selection —
                # the old trade id is almost certainly not in the new
                # cluster's trade list.  Overlay-metric chips also
                # reset: the new cluster may expose a different metric
                # set (different trainer run).
                ev.deep_dive_selected_trade_id = None
                ev.deep_dive_curve_metrics = []
                changed = True

        elif trigger == CLUSTER_DEEP_DIVE_IDS["per_trade_scatter"]:
            trade_id = _trade_id_from_click(click_data)
            if trade_id and ev.deep_dive_selected_trade_id != trade_id:
                ev.deep_dive_selected_trade_id = trade_id
                changed = True

        elif trigger == CLUSTER_DEEP_DIVE_IDS["trades_grid"]:
            trade_id = _trade_id_from_cell(cell_clicked)
            if trade_id and ev.deep_dive_selected_trade_id != trade_id:
                ev.deep_dive_selected_trade_id = trade_id
                changed = True

        elif trigger == CLUSTER_DEEP_DIVE_IDS["selected_trade_clear_btn"]:
            if clear_clicks and ev.deep_dive_selected_trade_id is not None:
                ev.deep_dive_selected_trade_id = None
                changed = True

        if not changed:
            raise PreventUpdate
        return session.to_store()


def _trade_id_from_click(click_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Pull ``trade_id`` out of Plotly ``clickData.points[0].customdata``.

    Robust to ``None`` / shape drift — any failure just returns ``None``
    so the sync callback short-circuits via the "no change" branch.
    """
    if not click_data:
        return None
    points = click_data.get("points") or []
    if not points:
        return None
    custom = points[0].get("customdata")
    if isinstance(custom, (list, tuple)) and custom:
        tid = custom[0]
        return tid if isinstance(tid, str) and tid else None
    return None


def _trade_id_from_cell(cell_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Pull ``trade_id`` out of AgGrid's ``cellClicked`` payload.

    AgGrid posts ``{rowId, data, colId, ...}`` — ``data`` is the raw
    row dict so the trade id lives under ``data.trade_id``.
    """
    if not cell_data:
        return None
    row = cell_data.get("data") or {}
    tid = row.get("trade_id")
    return tid if isinstance(tid, str) and tid else None


# ═════════════════════════════════════════════════════════════════════
# 3. Overlay-metric chip group → session.deep_dive_curve_metrics
# ═════════════════════════════════════════════════════════════════════


def _register_sync_curve_metrics(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],                    "data", allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["curves_chip_group"],     "value"),
        State(SHELL_IDS["session_store"],                     "data"),
        prevent_initial_call=True,
    )
    def _sync(
        selected:     Optional[List[str]],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        session = Session.from_store(session_data)
        normalised = _normalise_chip_selection(selected)
        if session.evaluation.deep_dive_curve_metrics == normalised:
            raise PreventUpdate
        session.evaluation.deep_dive_curve_metrics = normalised
        return session.to_store()


def _normalise_chip_selection(raw: Optional[List[str]]) -> List[str]:
    """Stable, deduped list of strings; drops anything non-string."""
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for v in raw:
        if not isinstance(v, str) or not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# ═════════════════════════════════════════════════════════════════════
# 4. Header band — attribute chips + nav button enabled state
# ═════════════════════════════════════════════════════════════════════


def _register_render_header(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["attribute_chips"],     "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"], "disabled"),
        Input(SHELL_IDS["url"],                              "pathname"),
        Input(SHELL_IDS["session_store"],                    "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Any], bool]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id

        if not cluster_id:
            return _attribute_chips(None, None), True

        clusters_res = backend.clusters_df(cluster_id=cluster_id)
        attrs_row: Optional[Dict[str, Any]] = None
        if (
            clusters_res.ok
            and clusters_res.data is not None
            and not clusters_res.data.empty
        ):
            attrs_row = clusters_res.data.iloc[0].to_dict()

        return _attribute_chips(attrs_row, cluster_id), False


def _attribute_chips(
    row: Optional[Dict[str, Any]], cluster_id: Optional[str],
) -> List[Any]:
    """Render attribute chips for the header band.

    ``row`` is a single row from ``clusters_df`` (as a dict).  Missing
    attributes are silently skipped — the empty state is handled by the
    caller so the chips area never shows a raw "—".
    """
    if row is None:
        return [
            html.Span(
                "Pick a cluster to see its attributes.",
                className="text-xs italic text-slate-500",
            )
        ]

    chips: List[Any] = [
        html.Span(
            cluster_id or "",
            className=(
                "inline-flex items-center gap-1 px-2 py-0.5 rounded-full "
                "text-[11px] font-mono text-slate-200 bg-slate-800 border border-slate-700"
            ),
        ),
    ]
    for label, column in _ATTRIBUTE_COLUMNS:
        value = row.get(column)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        chips.append(
            html.Span(
                children=[
                    html.Span(label + ":", className="text-slate-500 mr-1"),
                    html.Span(str(value), className="text-slate-200"),
                ],
                className=(
                    "inline-flex items-center gap-1 px-2 py-0.5 rounded-full "
                    "text-[11px] bg-slate-900 border border-slate-800"
                ),
            )
        )
    return chips


# ═════════════════════════════════════════════════════════════════════
# 5. Row 2 left — KPI grid (MAE / RMSE / R² / Coverage)
# ═════════════════════════════════════════════════════════════════════


def _register_render_kpis(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_mae_value"],      "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_rmse_value"],     "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_r2_value"],       "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_coverage_value"], "children"),
        Input(SHELL_IDS["url"],                             "pathname"),
        Input(SHELL_IDS["session_store"],                   "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, str, str]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cluster_id:
            return _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER

        # MAE + RMSE come straight from per_member_metrics (B5).
        mae: Optional[float] = None
        rmse: Optional[float] = None
        metrics_res = backend.per_member_metrics_df(
            split=session.split, cluster_id=cluster_id,
        )
        if (
            metrics_res.ok
            and metrics_res.data is not None
            and not metrics_res.data.empty
        ):
            row = metrics_res.data.iloc[0]
            mae = _nullable_float(row.get("mae"))
            rmse = _nullable_float(row.get("rmse"))

        # R² and Coverage come from the cluster timeseries — both
        # derivations are cheap enough to run client-side and avoid an
        # API schema bump.  Coverage uses ``|error| ≤ MAE`` as its
        # tolerance (see :func:`_compute_coverage`).
        r2: Optional[float] = None
        coverage: Optional[float] = None
        ts_res = backend.cluster_timeseries_df(
            session.split, cluster_id=cluster_id,
        )
        if (
            ts_res.ok
            and ts_res.data is not None
            and not ts_res.data.empty
        ):
            r2 = _compute_r_squared(ts_res.data)
            coverage = _compute_coverage(ts_res.data, mae)

        return (
            _fmt_float(mae),
            _fmt_float(rmse),
            _fmt_float(r2, precision=3),
            _fmt_pct(coverage),
        )


def _nullable_float(v: Any) -> Optional[float]:
    """Convert pandas cell values to ``float`` or ``None`` on NaN / bad types."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(fv):
        return None
    return fv


def _compute_r_squared(df: pd.DataFrame) -> Optional[float]:
    """Coefficient of determination between ``actual`` and ``predicted``.

    Mirrors the reference definition in
    ``src.rade_ml_pt.evaluation.metrics.r_squared``: returns 1.0 for a
    perfect fit, 0.0 when the model is equivalent to the mean, and can
    be negative for worse-than-mean models.  ``None`` when the series
    is empty or totally flat (so the KPI pill falls back to ``—``).
    """
    if "predicted" not in df.columns or "actual" not in df.columns:
        return None
    try:
        y_pred = df["predicted"].to_numpy(dtype=float, copy=False)
        y_true = df["actual"].to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError):
        return None
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if not mask.any():
        return None
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    if y_true.size == 0:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _compute_coverage(
    df: pd.DataFrame, mae: Optional[float],
) -> Optional[float]:
    """Fraction of scenarios where ``|error| ≤ MAE``.

    We use the per-cluster MAE as the tolerance, which makes Coverage
    scale-invariant across clusters (it's the share of scenarios that
    land *better than average* error-wise).  When MAE is unavailable
    we fall back to the mean of ``abs_error`` from the timeseries
    itself — same meaning, computed locally.

    Returns ``None`` if the timeseries doesn't carry an ``abs_error``
    column or carries only NaNs (e.g. a split with no scenarios).
    """
    if "abs_error" in df.columns:
        abs_err = df["abs_error"].to_numpy(dtype=float, copy=False)
    elif "error" in df.columns:
        abs_err = np.abs(df["error"].to_numpy(dtype=float, copy=False))
    elif "predicted" in df.columns and "actual" in df.columns:
        abs_err = np.abs(
            df["predicted"].to_numpy(dtype=float, copy=False)
            - df["actual"].to_numpy(dtype=float, copy=False)
        )
    else:
        return None

    mask = np.isfinite(abs_err)
    if not mask.any():
        return None
    abs_err = abs_err[mask]

    tolerance: float
    if mae is not None and mae > 0.0:
        tolerance = float(mae)
    else:
        tolerance = float(np.mean(abs_err))
    if tolerance <= 0.0:
        return None

    n_cov = int(np.sum(abs_err <= tolerance))
    return n_cov / int(abs_err.size)


# ═════════════════════════════════════════════════════════════════════
# 6. Row 2 right — training curves (figure + chip group population)
# ═════════════════════════════════════════════════════════════════════


def _register_render_training_curves(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["curves_chart"],       "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["curves_chip_group"],  "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["curves_chip_group"],  "value"),
        Output(CLUSTER_DEEP_DIVE_IDS["curves_chip_empty"],  "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["store_curve_metrics"], "data"),
        Input(SHELL_IDS["url"],                             "pathname"),
        Input(SHELL_IDS["session_store"],                   "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, List[Any], List[str], Dict[str, str], List[str]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id

        empty_chip_hidden = {"display": "none"}
        empty_chip_visible = {"display": "inline"}

        if not cluster_id:
            return (
                empty_figure("Pick a cluster to see its training curves."),
                [],
                [],
                empty_chip_visible,
                [],
            )

        res = backend.training_curves_df(cluster_id=cluster_id)
        if not res.ok or res.data is None or res.data.empty:
            err = res.error if not res.ok else "no data"
            logger.info(
                "training_curves fetch failed / empty for cluster %s: %s",
                cluster_id, err,
            )
            return (
                empty_figure("No training curves staged for this cluster."),
                [],
                [],
                empty_chip_visible,
                [],
            )

        df = res.data
        available = list(df.attrs.get("metrics") or [])

        # Validate the persisted chip selection against what the
        # trainer actually emitted for this cluster.  Anything unknown
        # is dropped; order is preserved so the user's choice rides
        # through unchanged when possible.
        persisted = list(session.evaluation.deep_dive_curve_metrics)
        validated = [m for m in persisted if m in available]

        chip_children = [
            dmc.Chip(m, value=m, size="xs", variant="outline")
            for m in available
        ]

        fig = training_curves_chart(
            df, selected_metrics=validated, available_metrics=available,
        )

        empty_style = empty_chip_hidden if available else empty_chip_visible
        return fig, chip_children, validated, empty_style, available


# ═════════════════════════════════════════════════════════════════════
# 7. Row 3 — residual over time + predicted vs actual with error band
# ═════════════════════════════════════════════════════════════════════


def _register_render_timeseries(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["residual_ts_chart"], "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["pnl_band_chart"],    "figure"),
        Input(SHELL_IDS["url"],                            "pathname"),
        Input(SHELL_IDS["session_store"],                  "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cluster_id:
            return (
                empty_figure("Pick a cluster to see its residual over time."),
                empty_figure("Pick a cluster to see its predicted vs target PnL."),
            )

        res = backend.cluster_timeseries_df(
            session.split, cluster_id=cluster_id,
        )
        if not res.ok or res.data is None or res.data.empty:
            return (
                empty_figure("No timeseries data for this cluster."),
                empty_figure("No timeseries data for this cluster."),
            )

        df = res.data
        return error_over_time(df), predicted_vs_actual_band(df)


# ═════════════════════════════════════════════════════════════════════
# 8. Row 4 left + Row 5 — violin + AgGrid rowData + trade_types store
# ═════════════════════════════════════════════════════════════════════


def _register_render_trade_data(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["per_trade_violin"],   "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["store_trade_types"],  "data"),
        Output(CLUSTER_DEEP_DIVE_IDS["trades_grid"],        "rowData"),
        Output(CLUSTER_DEEP_DIVE_IDS["trades_grid"],        "columnDefs"),
        Output(CLUSTER_DEEP_DIVE_IDS["trades_grid_empty"],  "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["trades_grid_wrap"],   "style"),
        Input(SHELL_IDS["url"],                             "pathname"),
        Input(SHELL_IDS["session_store"],                   "data"),
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        empty_state_on = {"display": "flex"}
        grid_on = {"display": "none"}
        grid_visible = {"display": "block"}

        if not cluster_id:
            return (
                empty_figure("Pick a cluster to see per-trade residuals."),
                {},
                [],
                _initial_column_defs(),
                empty_state_on,
                grid_on,
            )

        trades_res = backend.trades_df(session.split, cluster_id=cluster_id)
        graph_res = backend.trade_graph(cluster_id=cluster_id)

        trade_type_map: Dict[str, str] = {}
        if graph_res.ok and graph_res.data is not None:
            for node in graph_res.data.nodes:
                trade_type_map[str(node.trade_id)] = str(node.trade_type)

        if (
            not trades_res.ok
            or trades_res.data is None
            or trades_res.data.empty
        ):
            logger.info(
                "trades fetch failed / empty for cluster %s: %s",
                cluster_id, trades_res.error if not trades_res.ok else "empty",
            )
            return (
                empty_figure("No per-trade metrics for this cluster."),
                trade_type_map,
                [],
                _initial_column_defs(),
                empty_state_on,
                grid_on,
            )

        df = trades_res.data.copy()
        df["trade_type"] = (
            df["trade_id"].map(trade_type_map).fillna("unknown")
            if "trade_id" in df.columns
            else "unknown"
        )

        violin = per_trade_residual_violin(
            df, trade_type_map=trade_type_map or None,
        )

        grid_columns = _grid_columns_for(df)
        grid_rows = df[[c["field"] for c in grid_columns if c["field"] in df.columns]].to_dict(
            orient="records",
        )
        return (
            violin,
            trade_type_map,
            grid_rows,
            grid_columns,
            {"display": "none"},     # empty state off
            grid_visible,
        )


def _initial_column_defs() -> List[Dict[str, Any]]:
    """columnDefs for the empty state — mirrors the layout's bootstrap."""
    return [
        {"field": "trade_id",      "headerName": "Trade",        "flex": 2, "minWidth": 160},
        {"field": "trade_type",    "headerName": "Type",         "flex": 1, "minWidth": 100},
        {"field": "mae",           "headerName": "MAE",          "flex": 1, "type": "numericColumn"},
        {"field": "rmse",          "headerName": "RMSE",         "flex": 1, "type": "numericColumn"},
        {"field": "p95_ae",        "headerName": "P95 |err|",    "flex": 1, "type": "numericColumn"},
        {"field": "mean_residual", "headerName": "Mean resid.",  "flex": 1, "type": "numericColumn"},
        {"field": "n_scenarios",   "headerName": "Scenarios",    "flex": 1, "type": "numericColumn"},
    ]


def _grid_columns_for(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Dynamic columnDefs from the available ``trades_df`` columns.

    We keep the trade id + trade type first for context, then surface
    every numeric metric column in a stable order.  Unknown columns
    fall through untouched so schema evolutions pick up automatically.
    """
    headers = {
        "trade_id":       "Trade",
        "trade_type":     "Type",
        "mae":            "MAE",
        "mse":            "MSE",
        "rmse":           "RMSE",
        "max_ae":         "Max |err|",
        "p95_ae":         "P95 |err|",
        "p99_ae":         "P99 |err|",
        "mean_residual":  "Mean resid.",
        "std_residual":   "Std resid.",
        "n_scenarios":    "Scenarios",
    }
    priority = [
        "trade_id", "trade_type",
        "mae", "rmse", "p95_ae", "p99_ae",
        "mean_residual", "std_residual", "n_scenarios",
    ]

    numeric_cols = {
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col].dtype)
    }

    ordered_cols: List[str] = []
    seen: set = set()
    for col in priority:
        if col in df.columns and col not in seen:
            ordered_cols.append(col)
            seen.add(col)
    for col in df.columns:
        if col in seen:
            continue
        if col in ("cluster_id", "split"):
            continue
        ordered_cols.append(col)
        seen.add(col)

    defs: List[Dict[str, Any]] = []
    for col in ordered_cols:
        col_def: Dict[str, Any] = {
            "field":      str(col),
            "headerName": headers.get(col, str(col)),
        }
        if col == "trade_id":
            col_def.update({"flex": 2, "minWidth": 160})
        elif col == "trade_type":
            col_def.update({"flex": 1, "minWidth": 100})
        else:
            col_def["flex"] = 1
            if col in numeric_cols:
                col_def["type"] = "numericColumn"
                col_def["valueFormatter"] = {
                    "function": "d3.format('.4~g')(params.value)",
                }
        defs.append(col_def)
    return defs


# ═════════════════════════════════════════════════════════════════════
# 9. Row 4 right — per-trade scatter (re-renders on selection change)
# ═════════════════════════════════════════════════════════════════════


def _register_render_scatter(app: "Dash") -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["per_trade_scatter"],  "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["selected_trade_chip"], "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["selected_trade_label"], "children"),
        Input(CLUSTER_DEEP_DIVE_IDS["trades_grid"],         "rowData"),
        Input(CLUSTER_DEEP_DIVE_IDS["store_trade_types"],   "data"),
        Input(SHELL_IDS["session_store"],                   "data"),
    )
    def _render(
        row_data:        Optional[List[Dict[str, Any]]],
        trade_types:     Optional[Dict[str, str]],
        session_data:    Optional[Dict[str, Any]],
    ) -> Tuple[Any, Dict[str, Any], str]:
        session = Session.from_store(session_data)
        selected_trade_id = session.evaluation.deep_dive_selected_trade_id

        hidden_chip_style = {"display": "none"}
        if not row_data:
            return (
                empty_figure("No per-trade scatter data."),
                hidden_chip_style,
                "Trade: —",
            )

        df = pd.DataFrame(row_data)
        fig = per_trade_scatter(
            df,
            trade_type_map=trade_types or None,
            selected_trade_id=selected_trade_id,
        )

        if selected_trade_id:
            return (
                fig,
                {"display": "flex"},
                f"Trade: {selected_trade_id}",
            )
        return fig, hidden_chip_style, "Trade: —"


# ═════════════════════════════════════════════════════════════════════
# 10. "Trade-Graph" button on this page → navigate + pin cluster
# ═════════════════════════════════════════════════════════════════════


def _register_navigate_to_trade_graph(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["url"],                          "pathname",
               allow_duplicate=True),
        Output(SHELL_IDS["url"],                          "search",
               allow_duplicate=True),
        Output(SHELL_IDS["session_store"],                "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"], "n_clicks"),
        State(SHELL_IDS["session_store"],                 "data"),
        prevent_initial_call=True,
    )
    def _navigate(
        n_clicks:     Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cid = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cid:
            raise PreventUpdate

        # Pin the cluster on the Trade-Graph tab so the user lands on
        # the same cluster they were inspecting here.
        session.evaluation.trade_graph_cluster_id = cid
        session.evaluation.trade_graph_selected_trade_id = None

        return _TRADE_GRAPH_PATH, "", session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 11. "Open in Cluster Deep Dive" (Trade-Graph tab) → land here
# ═════════════════════════════════════════════════════════════════════


def _register_navigate_from_trade_graph(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["url"],                          "pathname",
               allow_duplicate=True),
        Output(SHELL_IDS["url"],                          "search",
               allow_duplicate=True),
        Output(SHELL_IDS["session_store"],                "data",
               allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["selected_deep_dive_btn"],  "n_clicks"),
        State(SHELL_IDS["session_store"],                 "data"),
        prevent_initial_call=True,
    )
    def _navigate(
        n_clicks:     Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cid = (
            session.evaluation.trade_graph_cluster_id
            or session.cluster_id
        )
        if not cid:
            raise PreventUpdate

        # Copy the Trade-Graph sub-tab's selection into the Deep-Dive
        # slots so the landing view is primed with the user's context.
        session.evaluation.deep_dive_cluster_id = cid
        session.evaluation.deep_dive_selected_trade_id = (
            session.evaluation.trade_graph_selected_trade_id
        )

        # Keep the query string in sync so deep-links work and a
        # browser-back reproduces the same state.
        return _DEEP_DIVE_PATH, f"?cid={cid}", session.to_store()


__all__ = ["register"]
```

---

### A.U7 · REPLACE `examples/rade_analytics/_mock_backend.py`

```python
"""Shared synthetic :class:`RadeBackend` for every preview script.

Each preview script (``05_overview_preview_live.py``,
``06_portfolio_preview_live.py``, …) builds the full Rade app and
injects this mock so callbacks can be exercised end-to-end without
running the FastAPI backend.

Design
------
* Every public method of :class:`RadeBackend` that any page consumes
  is overridden here.
* Data is deterministic — seeded once at construction — so screenshots
  are reproducible.
* :meth:`_cluster_timeseries` is the **single source** for synthetic
  per-cluster residuals; :meth:`portfolio_df` derives its aggregate
  rollup from it so the cluster-level and portfolio-level views stay
  internally consistent (important once the user starts filtering and
  comparing numbers between the two).
* Pydantic responses are constructed from the real model classes so
  any schema drift breaks at import-time instead of silently shipping
  a bad mock.

Import from a preview script::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _mock_backend import MockRadeBackend
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.rade_ml_pt.ensemble.api.models.clusters import (
    ClusterInfo,
    ClustersResponse,
)
from src.rade_ml_pt.ensemble.api.models.meta import (
    HealthResponse,
    VersionsResponse,
)
from src.rade_ml_pt.ensemble.api.models.trade_graph import (
    TradeGraphEdge,
    TradeGraphNode,
    TradeGraphResponse,
    TradeGraphStats,
)
from src.ui.apps.rade_analytics.data.backend import BackendResult, RadeBackend


# ─────────────────────────────────────────────────────────────────────
# Constants — vocabulary for synthetic cluster attributes
# ─────────────────────────────────────────────────────────────────────

_ASSET_CLASSES = ("rates", "fx", "credit", "equity")
_CURRENCIES    = ("USD", "EUR", "GBP", "JPY")
_DESKS         = ("Alpha", "Beta", "Gamma")
_PRODUCTS      = ("swap", "option", "forward", "bond")
_N_SCENARIOS   = 48


class MockRadeBackend(RadeBackend):
    """Deterministic in-memory backend for Dash UI preview scripts.

    Parameters
    ----------
    n_clusters
        How many synthetic clusters to generate.  Default 12 — enough
        to populate 3 desks × ~4 products without overwhelming the
        heatmap.
    seed
        Seeds both :mod:`random` and :mod:`numpy.random` so sessions
        are reproducible across the two RNGs we use.
    """

    def __init__(self, *, n_clusters: int = 12, seed: int = 42) -> None:
        # NOT calling ``super().__init__`` on purpose — the parent
        # constructor requires a client and cache, neither of which
        # makes sense here.  Every public method is overridden below.
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._seed = seed
        self._n_clusters = n_clusters
        self._cluster_ids: List[str] = [
            f"cluster_{i + 1:02d}" for i in range(n_clusters)
        ]
        self._versions: List[str] = [
            "v2026.04.17-a1b2c",
            "v2026.04.10-f3e4d7",
            "v2026.04.03-9876a2",
        ]
        # Stable attribute map — computed once, reused forever.
        self._cluster_attrs: Dict[str, Dict[str, Any]] = {
            cid: {
                "asset_class":   self._rng.choice(_ASSET_CLASSES),
                "currency_code": self._rng.choice(_CURRENCIES),
                "desk":          self._rng.choice(_DESKS),
                "product_code":  self._rng.choice(_PRODUCTS),
            }
            for cid in self._cluster_ids
        }
        # Stable per-cluster weights used when reconstructing the
        # portfolio rollup — weights sum to ~1.0 so portfolio-level
        # numbers stay on a reasonable scale.
        raw_weights = self._np_rng.uniform(0.2, 1.8, size=n_clusters)
        self._cluster_weights: Dict[str, float] = dict(
            zip(self._cluster_ids, raw_weights / raw_weights.sum() * n_clusters)
        )

    # ─────────────────────────────────────────────────────────────
    # Meta endpoints
    # ─────────────────────────────────────────────────────────────

    def health(self) -> BackendResult[HealthResponse]:
        return BackendResult.success(
            HealthResponse(
                status="ok",
                version=self._versions[0],
                artifacts_dir="/tmp/mock-artifacts",
            )
        )

    def versions(self) -> BackendResult[VersionsResponse]:
        return BackendResult.success(
            VersionsResponse(
                active=self._versions[0],
                available=list(self._versions),
            )
        )

    # ─────────────────────────────────────────────────────────────
    # Metrics endpoints
    # ─────────────────────────────────────────────────────────────

    def ensemble_metrics_df(self) -> BackendResult[pd.DataFrame]:
        rows: List[Dict[str, Any]] = []
        for split, scale in (("train", 0.9), ("val", 1.05), ("test", 1.15)):
            mae = 0.0012 * scale
            mse = (mae ** 2) * 1.4
            rmse = float(np.sqrt(mse))
            rows.append(
                {
                    "split":  split,
                    "mae":    mae,
                    "mse":    mse,
                    "rmse":   rmse,
                    "max_ae": mae * 18.0,
                    "p95_ae": mae * 4.5,
                    "p99_ae": mae * 9.0,
                }
            )
        return BackendResult.success(pd.DataFrame(rows))

    def per_member_metrics_df(
        self,
        *,
        split:      Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        # Derive per-cluster metrics from the synthetic timeseries so
        # rows 1-3 (aggregate) and rows 4-5 (grouped) agree on numbers.
        splits = [split] if split else ["train", "val", "test"]
        frames: List[pd.DataFrame] = []
        for s in splits:
            ts = self._cluster_timeseries(s)
            if cluster_id:
                ts = ts[ts["cluster_id"] == cluster_id]
            grouped = (
                ts.assign(
                    _abs_err=ts["abs_error"],
                    _sq_err=ts["squared_error"],
                )
                .groupby("cluster_id", as_index=False)
                .agg(
                    mae=("_abs_err", "mean"),
                    rmse_sq=("_sq_err", "mean"),
                    max_ae=("_abs_err", "max"),
                    p95_ae=("_abs_err", lambda x: float(np.quantile(x, 0.95))),
                    p99_ae=("_abs_err", lambda x: float(np.quantile(x, 0.99))),
                    n_scenarios=("scenario_idx", "nunique"),
                )
            )
            grouped["rmse"] = np.sqrt(grouped["rmse_sq"].astype(float))
            grouped["mse"] = grouped["rmse"] ** 2
            grouped["split"] = s
            grouped["n_targets"] = [
                self._rng.randint(120, 480) for _ in range(len(grouped))
            ]
            frames.append(
                grouped[
                    [
                        "cluster_id", "split", "mae", "mse", "rmse",
                        "max_ae", "p95_ae", "p99_ae",
                        "n_targets", "n_scenarios",
                    ]
                ]
            )
        if not frames:
            return BackendResult.success(pd.DataFrame())
        return BackendResult.success(pd.concat(frames, ignore_index=True))

    # ─────────────────────────────────────────────────────────────
    # Cluster endpoints
    # ─────────────────────────────────────────────────────────────

    def clusters(
        self, *, cluster_id: Optional[str] = None,
    ) -> BackendResult[ClustersResponse]:
        entries: List[ClusterInfo] = []
        for cid in self._cluster_ids:
            if cluster_id and cid != cluster_id:
                continue
            # n_trades follows a deterministic pattern so re-runs match.
            n_trades = 80 + int(self._cluster_weights[cid] * 180)
            entries.append(
                ClusterInfo(
                    cluster_id=cid,
                    n_trades=n_trades,
                    attributes=dict(self._cluster_attrs[cid]),
                )
            )
        return BackendResult.success(
            ClustersResponse(
                clusters=entries,
                attribute_names=["asset_class", "currency_code", "desk", "product_code"],
            )
        )

    def clusters_df(
        self, *, cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        res = self.clusters(cluster_id=cluster_id)
        if not res.ok or res.data is None:
            return BackendResult.failure(
                error=res.error or "",
                status_code=res.status_code,
            )
        rows = [
            {"cluster_id": c.cluster_id, "n_trades": c.n_trades, **c.attributes}
            for c in res.data.clusters
        ]
        return BackendResult.success(pd.DataFrame(rows))

    # ─────────────────────────────────────────────────────────────
    # Portfolio + cluster timeseries
    # ─────────────────────────────────────────────────────────────

    def portfolio_df(self, split: str) -> BackendResult[pd.DataFrame]:
        ts = self._cluster_timeseries(split)
        if ts.empty:
            return BackendResult.success(ts)
        agg = (
            ts.groupby(["scenario_idx", "scenario_label"], as_index=False)
            .agg(
                predicted=("predicted", "sum"),
                actual=("actual", "sum"),
            )
        )
        agg["error"] = agg["predicted"] - agg["actual"]
        agg["abs_error"] = agg["error"].abs()
        agg["squared_error"] = agg["error"] ** 2
        return BackendResult.success(agg)

    def cluster_timeseries_df(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        df = self._cluster_timeseries(split)
        if cluster_id:
            df = df[df["cluster_id"] == cluster_id]
        return BackendResult.success(df.reset_index(drop=True))

    # ─────────────────────────────────────────────────────────────
    # Graph stats + trade-graph (Phase E.3)
    # ─────────────────────────────────────────────────────────────

    def graph_stats_df(
        self, *, cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        """Per-cluster graph topology summary.

        Topology numbers are derived from the same synthetic graphs
        :meth:`_trade_graph_payload` emits, so the Trade-Graph tab's
        stats cards and the secondary charts always agree with the
        rendered network.
        """
        rows: List[Dict[str, Any]] = []
        for cid in self._cluster_ids:
            payload = self._trade_graph_payload(cid)
            n = payload.stats.n_nodes
            rows.append(
                {
                    "cluster_id":   cid,
                    "n_nodes":      int(n),
                    "n_edges":      int(payload.stats.n_edges),
                    "density":      float(payload.stats.density),
                    "mean_weight":  float(payload.stats.mean_weight),
                }
            )
        df = pd.DataFrame(rows)
        if cluster_id:
            df = df[df["cluster_id"] == cluster_id].reset_index(drop=True)
        return BackendResult.success(df)

    def trade_graph(
        self, *, cluster_id: str,
    ) -> BackendResult[TradeGraphResponse]:
        if cluster_id not in self._cluster_ids:
            return BackendResult.failure(
                error=f"unknown cluster '{cluster_id}'",
                status_code=404,
            )
        return BackendResult.success(self._trade_graph_payload(cluster_id))

    # ─────────────────────────────────────────────────────────────
    # Trades — per-trade metrics (Phase E.4)
    # ─────────────────────────────────────────────────────────────

    def trades_df(
        self,
        split: str,
        *,
        cluster_id: Optional[str] = None,
    ) -> BackendResult[pd.DataFrame]:
        """Synthetic per-trade metrics.

        Trade ids match those emitted by :meth:`_trade_graph_payload`
        (``{cluster_id}_T###`` for targets, ``{cluster_id}_E###`` for
        elementary trades) so the Cluster Deep-Dive page can merge
        trade-type colour on top of these rows without surprises.
        """
        cluster_ids = (
            [cluster_id]
            if cluster_id
            else list(self._cluster_ids)
        )
        frames: List[pd.DataFrame] = []
        for cid in cluster_ids:
            if cid not in self._cluster_ids:
                continue
            frames.append(_trades_df_cached(
                mock_id=id(self),
                cluster_id=cid,
                split=split,
                seed=self._seed,
            ))
        if not frames:
            return BackendResult.success(pd.DataFrame())
        return BackendResult.success(pd.concat(frames, ignore_index=True))

    # ─────────────────────────────────────────────────────────────
    # Training curves — per-cluster per-epoch (Phase E.4 Row 2)
    # ─────────────────────────────────────────────────────────────

    def training_curves_df(
        self, *, cluster_id: str,
    ) -> BackendResult[pd.DataFrame]:
        """Synthetic per-epoch training curves for one cluster.

        Produces a small deterministic curve set — ``train_loss``,
        ``val_loss``, ``mae``, ``val_mae`` — seeded off the cluster
        id so every refetch for the same cluster returns the same
        numbers.  Available metric names are stashed on
        ``df.attrs["metrics"]`` exactly like the real backend so the
        chip-row callback needs no mock-specific branch.
        """
        if cluster_id not in self._cluster_ids:
            return BackendResult.failure(
                error=f"training curves missing for '{cluster_id}'",
                status_code=404,
            )
        df = _training_curves_cached(
            mock_id=id(self),
            cluster_id=cluster_id,
            seed=self._seed,
        )
        return BackendResult.success(df)

    # ─────────────────────────────────────────────────────────────
    # Internal — seeded synthetic data generators
    # ─────────────────────────────────────────────────────────────

    def _trade_graph_payload(self, cluster_id: str) -> TradeGraphResponse:
        """Build a deterministic trade-graph payload for one cluster.

        Uses the cluster id as the RNG seed so every call for the same
        cluster returns identical nodes / edges.  Topology numbers
        (n_nodes, n_edges, density, mean_weight) are derived from the
        result — the graph-stats endpoint builds its mock off the same
        helper, so the two views cannot drift.
        """
        return _trade_graph_cached(
            mock_id=id(self),
            cluster_id=cluster_id,
            seed=self._seed,
        )

    def _cluster_timeseries(self, split: str) -> pd.DataFrame:
        """Deterministic per-cluster per-scenario synthetic frame.

        Memoised per ``(mock_id, split)`` so repeated calls inside one
        callback render don't regenerate (and re-randomise) the
        underlying numbers.  The ``mock_id`` step lets multiple
        MockRadeBackend instances coexist in the same Python process
        (e.g. tests) without bleeding state between them.
        """
        return _cluster_timeseries_cached(
            mock_id=id(self),
            split=split,
            seed=self._seed,
            cluster_ids=tuple(self._cluster_ids),
            cluster_weights=tuple(
                (cid, self._cluster_weights[cid]) for cid in self._cluster_ids
            ),
        )


# ─────────────────────────────────────────────────────────────────────
# Module-level LRU cache for the synthetic timeseries.  Keyed on the
# MockRadeBackend instance id + split so data stays stable within a
# Dash session but regenerates fresh when the server restarts.
# ─────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def _cluster_timeseries_cached(
    *,
    mock_id:          int,
    split:            str,
    seed:             int,
    cluster_ids:      tuple[str, ...],
    cluster_weights:  tuple[tuple[str, float], ...],
) -> pd.DataFrame:
    del mock_id  # cache key only
    rng = np.random.default_rng(seed + hash(split) % 10_000)
    labels = pd.date_range("2025-11-01", periods=_N_SCENARIOS, freq="D").strftime(
        "%Y-%m-%d"
    )
    noise_scale = {"train": 0.008, "val": 0.012, "test": 0.018}.get(split, 0.012)

    weights = dict(cluster_weights)
    frames: List[pd.DataFrame] = []
    for cid in cluster_ids:
        w = float(weights[cid])
        # Each cluster follows its own random walk scaled by weight;
        # the sum across clusters reconstructs a sensible portfolio
        # trend.
        actual_inc = rng.normal(0.003 * w, 0.004 * max(w, 0.3), _N_SCENARIOS)
        actual = np.cumsum(actual_inc) + 0.08 * w
        pred = actual + rng.normal(0, noise_scale * max(w, 0.3), _N_SCENARIOS)
        error = pred - actual
        frames.append(
            pd.DataFrame(
                {
                    "cluster_id":     cid,
                    "scenario_idx":   list(range(_N_SCENARIOS)),
                    "scenario_label": labels,
                    "predicted":      pred,
                    "actual":         actual,
                    "error":          error,
                    "abs_error":      np.abs(error),
                    "squared_error":  error ** 2,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────
# Trade-graph synthesis — seeded per cluster so nodes / edges never
# change for the same (mock_id, cluster_id) pair within one preview run.
# ─────────────────────────────────────────────────────────────────────

# Node budget per cluster.  Random within [lo, hi]; small enough to
# keep Cytoscape renders snappy, large enough to look like a real
# trade graph.
_TRADE_NODE_MIN = 40
_TRADE_NODE_MAX = 90
# Roughly 1/5 of nodes are "target" trades; rest are elementary
# building blocks.
_TARGET_FRACTION = 0.2


@lru_cache(maxsize=256)
def _trade_graph_cached(
    *,
    mock_id:    int,
    cluster_id: str,
    seed:       int,
) -> TradeGraphResponse:
    del mock_id   # cache key only
    # Seed per-cluster so different clusters look different but the
    # same cluster is stable across re-fetches.
    rng = np.random.default_rng(seed + (abs(hash(cluster_id)) % 10_000))

    n = int(rng.integers(_TRADE_NODE_MIN, _TRADE_NODE_MAX + 1))
    n_target = max(1, int(round(n * _TARGET_FRACTION)))
    n_elem = n - n_target

    target_ids = [f"{cluster_id}_T{i:03d}" for i in range(n_target)]
    elem_ids = [f"{cluster_id}_E{i:03d}" for i in range(n_elem)]

    nodes: List[TradeGraphNode] = []
    for tid in target_ids:
        nodes.append(
            TradeGraphNode(trade_id=tid, cluster_id=cluster_id, trade_type="target")
        )
    for tid in elem_ids:
        nodes.append(
            TradeGraphNode(
                trade_id=tid, cluster_id=cluster_id, trade_type="elementary",
            )
        )

    # Edge budget ≈ 2 × n (moderately sparse, enough to look
    # connected without overwhelming the canvas).
    n_edges_target = max(n, 2 * n)
    edges: List[TradeGraphEdge] = []
    seen: set[tuple[int, int]] = set()
    tries = 0
    while len(edges) < n_edges_target and tries < n_edges_target * 4:
        tries += 1
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        weight = float(rng.uniform(0.05, 1.0))
        edges.append(
            TradeGraphEdge(
                source=nodes[i].trade_id,
                target=nodes[j].trade_id,
                weight=weight,
            )
        )

    # Density uses the sparse-values count — consistent with the real
    # pipeline's ``_save_graph_stats_parquet`` formula.
    density = float(len(edges) / (n * n)) if n > 0 else 0.0
    mean_w = float(np.mean([e.weight for e in edges])) if edges else 0.0

    return TradeGraphResponse(
        cluster_id=cluster_id,
        n_target_trades=n_target,
        n_elementary_trades=n_elem,
        stats=TradeGraphStats(
            n_nodes=n,
            n_edges=len(edges),
            density=density,
            mean_weight=mean_w,
        ),
        nodes=nodes,
        edges=edges,
        warnings=None,
    )


# ─────────────────────────────────────────────────────────────────────
# Per-trade metrics synthesis — one row per trade, seeded off the trade
# id so the numbers are stable across re-fetches within one preview run.
# Uses the trade-graph payload as the authoritative trade list so the
# two views stay aligned.
# ─────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=128)
def _trades_df_cached(
    *,
    mock_id:    int,
    cluster_id: str,
    split:      str,
    seed:       int,
) -> pd.DataFrame:
    del mock_id   # cache key only
    graph = _trade_graph_cached(mock_id=0, cluster_id=cluster_id, seed=seed)

    # Noise scale mirrors ``_cluster_timeseries_cached`` so the two
    # pages agree on "test is harder than val, val is harder than
    # train".
    noise_scale = {"train": 0.008, "val": 0.012, "test": 0.018}.get(split, 0.012)
    rng = np.random.default_rng(seed + (abs(hash((cluster_id, split))) % 10_000))

    rows: List[Dict[str, Any]] = []
    for node in graph.nodes:
        # Target trades carry more residual spread than elementary ones
        # — matches the real pipeline's behaviour where targets absorb
        # the model's final prediction error.
        type_scale = 1.8 if node.trade_type == "target" else 1.0
        residuals = rng.normal(
            loc=0.0,
            scale=noise_scale * type_scale,
            size=_N_SCENARIOS,
        )
        # Add a slight systematic bias so mean_residual is non-zero
        # and the violin / scatter look lifelike.
        bias = float(rng.normal(0.0, noise_scale * 0.4 * type_scale))
        residuals += bias
        abs_err = np.abs(residuals)
        sq_err = residuals ** 2
        rows.append(
            {
                "cluster_id":     cluster_id,
                "trade_id":       node.trade_id,
                "split":          split,
                "mae":            float(abs_err.mean()),
                "mse":            float(sq_err.mean()),
                "rmse":           float(np.sqrt(sq_err.mean())),
                "max_ae":         float(abs_err.max()),
                "p95_ae":         float(np.quantile(abs_err, 0.95)),
                "p99_ae":         float(np.quantile(abs_err, 0.99)),
                "mean_residual":  float(residuals.mean()),
                "std_residual":   float(residuals.std()),
                "n_scenarios":    int(_N_SCENARIOS),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Training-curves synthesis — deterministic per-cluster multi-metric
# series.  Mirrors the real pipeline's ``training_curves.parquet`` shape
# closely enough that the UI can't tell the two apart.
# ─────────────────────────────────────────────────────────────────────

_N_EPOCHS = 40


@lru_cache(maxsize=128)
def _training_curves_cached(
    *,
    mock_id:    int,
    cluster_id: str,
    seed:       int,
) -> pd.DataFrame:
    del mock_id   # cache key only
    rng = np.random.default_rng(seed + (abs(hash(("curves", cluster_id))) % 10_000))
    epochs = np.arange(_N_EPOCHS, dtype=np.int32)

    # Per-cluster starting MAE & decay speed so clusters look different.
    start = float(rng.uniform(0.08, 0.20))
    decay = float(rng.uniform(0.08, 0.18))
    noise = rng.normal(0.0, 0.003, size=_N_EPOCHS)
    train_loss = start * np.exp(-decay * epochs) + np.abs(noise)
    # Val lags train slightly and carries a bit more noise — classic
    # learning-curve silhouette.
    val_loss = (
        start * 1.08 * np.exp(-decay * 0.95 * epochs)
        + np.abs(rng.normal(0.0, 0.006, size=_N_EPOCHS))
    )
    # Secondary metrics (mae) track loss but on a different scale so
    # the chart actually needs the metric-picker to show them meaning-
    # fully.
    mae = train_loss * 0.85 + np.abs(rng.normal(0.0, 0.002, size=_N_EPOCHS))
    val_mae = val_loss * 0.88 + np.abs(rng.normal(0.0, 0.004, size=_N_EPOCHS))

    df = pd.DataFrame(
        {
            "epoch":      epochs,
            "train_loss": train_loss.astype(np.float32),
            "val_loss":   val_loss.astype(np.float32),
            "mae":        mae.astype(np.float32),
            "val_mae":    val_mae.astype(np.float32),
        }
    )
    df.attrs["metrics"] = ["val_loss", "mae", "val_mae"]
    df.attrs["cluster_id"] = cluster_id
    return df


__all__ = ["MockRadeBackend"]
```

---

### A.U8 · REPLACE `examples/rade_analytics/09_cluster_deep_dive_preview_live.py`

```python
"""End-to-end smoke test for the Evaluation → Cluster Deep-Dive sub-tab.

Phase E.4 ships the full Cluster Deep-Dive page (Row 2 revision):

* Sticky header with cluster picker + attribute chips (no KPI strip —
  KPIs now live in Row 2).
* Row 2 — 2×2 KPI grid (MAE, RMSE, R², Coverage) on the left, training
  curves chart on the right with a chip-row metric picker.  Coverage
  uses a ``|error| ≤ MAE`` tolerance; R² is computed client-side from
  the cluster timeseries.
* Row 3 — residual-over-time (rolling mean + ±1σ band) and predicted-
  vs-target-PnL with a rose shaded error band.
* Row 4 — per-trade residual violin (target / elementary) and per-trade
  scatter (mean_residual × MAE, coloured by trade type).  Clicking a
  scatter point highlights the corresponding trade id; the focus chip
  in the header clears it.
* Row 5 — trades AgGrid.  Row click also updates the selected trade,
  which the scatter re-renders to emerald-ring the chosen point.
* Header "Trade-Graph" button navigates to the Trade-Graph sub-tab
  with the current cluster pinned; the reciprocal "Open in Cluster
  Deep Dive" button on the Trade-Graph tab navigates back here,
  priming the cluster + selected trade.

Run from the project root::

    python examples/rade_analytics/09_cluster_deep_dive_preview_live.py

Then open http://localhost:8055/evaluation/cluster.

You can also deep-link into a specific cluster via ``?cid=``::

    http://localhost:8055/evaluation/cluster?cid=cluster_04
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── React 18 pin — must happen *before* ``dash.Dash`` is imported.
import dash._dash_renderer  # noqa: E402
dash._dash_renderer._set_react_version("18.2.0")

# Make ``_mock_backend`` importable via sibling-file lookup.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dash import Dash  # noqa: E402

from _mock_backend import MockRadeBackend  # noqa: E402  (sibling import)
from src.ui.apps.rade_analytics.callbacks import register_all  # noqa: E402
from src.ui.apps.rade_analytics.config import RadeUiSettings, set_settings  # noqa: E402
from src.ui.apps.rade_analytics.layouts import (  # noqa: E402
    INDEX_STRING,
    META_TAGS,
    build_shell,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rade.preview.cluster_deep_dive_live")


def build_preview_app() -> Dash:
    """Compose a fully-wired Rade app against :class:`MockRadeBackend`."""
    settings = RadeUiSettings(
        api_url="http://mock",
        cache_type="NullCache",
        debug=True,
    )
    set_settings(settings)

    assets_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "ui" / "apps" / "rade_analytics" / "assets"
    )

    app = Dash(
        __name__,
        title="Rade — Cluster Deep-Dive live preview (mock)",
        update_title=None,
        index_string=INDEX_STRING,
        meta_tags=META_TAGS,
        assets_folder=str(assets_dir),
        assets_ignore=r"tailwind\.(config\.js|input\.css)|README\.md",
        suppress_callback_exceptions=True,
    )

    backend = MockRadeBackend()
    app.server.config["rade_backend"] = backend
    app.server.config["rade_settings"] = settings

    app.layout = build_shell()
    register_all(app, backend)

    log.info(
        "Preview ready — open http://localhost:8055/evaluation/cluster"
    )
    return app


if __name__ == "__main__":
    build_preview_app().run(
        debug=True,
        host="0.0.0.0",
        port=8055,
    )
```
