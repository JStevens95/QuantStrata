# PRISM Retool Migration — Master Plan

> **Status**: Living document. Updated as phases are scoped, implemented, and closed out.
> **Owner**: Joes Stevens
> **Last updated**: 2026-04-17
> **Related**: `docs/platform_designs/ensemble_analytics_db_v3_blueprint.md`, `docs/ensemble_dashboard_design.md`

---

## 0. Purpose & reading order

This document is the single source of truth for the migration of the ensemble analytics surface from the existing Dash UI (`ensemble_analytics_db`) to a Retool-backed PRISM UI, including the training and evaluation pipeline changes that unlock it.

Read in this order:

1. Section 1 — decisions log (what has been agreed).
2. Section 2 — audit findings (what exists today, what is missing).
3. Section 3 — file format policy (NPZ vs parquet).
4. Sections 4–6 — phased implementation plan, starting with training and eval outputs, then API, then Retool scaffolding.
5. Section 7 — full output catalogue (one row per artifact, with endpoint and Retool shape).
6. Section 8 — open questions.

Implementation cells marked *TBD* will be filled in collaboratively as each phase is picked up.

---

## 1. Decisions log

| Date | Decision | Notes |
|---|---|---|
| 2026-04-17 | Retool namespace is `/prism/v1/*` | Sibling to existing `/api/v1/*`; no existing route is renamed. |
| 2026-04-17 | Dual file-format policy: keep NPZ for wide preloaded matrices, use parquet for long-format Retool data | See §3. |
| 2026-04-17 | New artifacts feed **both** Dash and Retool through a single `ArtifactCache` | One cache, two UIs. |
| 2026-04-17 | Start with training + eval output changes, then API fixes, then Retool scaffolding | Ensures data is in place before surfacing it. |
| 2026-04-17 | This document is the master plan; phase details are filled in iteratively | See §4. |
| 2026-04-17 | **Dash is the PRISM reference implementation**: a new Dash app `ui/apps/prism_dash/` mirrors the PRISM mocks 1:1 and consumes `/prism/v1/*`. Existing `ensemble_analytics_db` Dash app is left in place and will be sunset in Phase 10. | See §3a and Phase 8. |

---

## 2. Audit findings

### 2.1 Training pipeline (`EnsembleTrainPipeline` + `HybridGnnRnnTrainPipeline`)

**What it writes today**

- Per member under `{registry_dir}/{member_version}/`:
  `model.pt`, `metadata.json`, `graph_builder.pkl`, `graph_results.joblib`, `encoder.pkl`, `encoder_results.joblib`, `target_scaler.pkl`, `elementary_scaler.pkl`, `data_config.json`, `trade_universe.json`, `target_pnl.parquet`, `elementary_pnl.parquet`, `target_attributes.json`, `elementary_attributes.json`, `datasets/{train,val,test}.pt`, and job-derived joblibs (`cluster_info.joblib`, `cluster_assets.joblib`, `cluster_elem_trades.joblib`).
- Per member under `{artifacts_dir}/training/{member_version}/`:
  `training_analytics.png`, optional PnL distribution PNGs, optional `trade_graph_analytics.png`.
- Ensemble under `{registry_dir}/ensemble/{ens_version}/`:
  `ensemble_config.json`, `member_versions.json`, `trade_cluster_map.json`, `member_summary.json`.
  Mirror of `member_summary.json` under `{artifacts_dir}/ensemble/{ens_version}/`.

**Confirmed gaps**

| ID | Issue | Consequence |
|---|---|---|
| T1 | Trainer writes `training_analytics.png`; `ArtifactCache` reads `training_plots.png` | Silent empty convergence on fresh runs (the exact work-env bug). |
| T2 | `_plot_graph` referenced in `HybridGnnRnnTrainPipeline.build_data` but not defined in `rade_ml_pt` | `plot_trade_graph=True` raises `AttributeError`. |
| T3 | `src.rade_ml_pt.training.reports` imported in `base.py._finalise` but missing | Silent skip of markdown training report. |
| T4 | `member_summary.json` lacks `ccy`, `desk`, `product`, `trained_at` | Governance Member Registry table cannot be built from summary alone. |
| T5 | No drift baselines saved at training time | Monitoring tab cannot compare production vs training distributions. |
| T6 | No persistent graph layout | Trade Graph tab has no coordinates. |
| T7 | No data-quality snapshot at training time | Data Quality tab has no training-time baseline. |

### 2.2 Eval pipeline (`EnsembleEvalPipeline`)

**What it writes today**, under `{artifacts_dir}/ensemble/{ens_version}/evaluation/`:
`manifest.json`, `cluster_attributes.json`, `trade_cluster_map.json`, `graph_stats.json`, `ensemble_metrics_{split}.json`, `per_member_metrics_{split}.json`, `portfolio_summary/{split}.npz`, `cluster_summary/{split}.npz`, `members/{cid}/predictions/{split}.npz`, `trade_metrics/{split}.json`, `group_correlations/{split}.json`.
Optional per-member eval plots under `{artifacts_dir}/members/{cid}/evaluation/{member_version}/{period}/`.

**Confirmed gaps**

| ID | Missing artifact | Tab unblocked |
|---|---|---|
| E1 | `portfolio_timeseries_{split}.parquet` | Overview, Eval Portfolio |
| E2 | `portfolio_residuals_{split}.parquet` | Eval Portfolio |
| E3 | `group_metrics_{split}.parquet` (long, column `group_type ∈ {portfolio,desk,product,ccy,cluster}`) | Eval By X |
| E4 | `group_timeseries_{split}.parquet` | Eval By X |
| E5 | `group_residuals_{split}.parquet` | Eval By X |
| E6 | `cross_cluster/correlation_{split}.parquet` | Cross-Cluster |
| E7 | `cluster_residuals_{split}.parquet` | Cross-Cluster, Cluster Deep Dive |
| E8 | `graph/{cluster}/layout.parquet` + `edges.parquet` | Trade Graph |
| E9 | `graph/stats.json` vs today's `graph_stats.json` at evaluation root | Trade Graph |
| E10 | `cluster_health.json` | Overview |
| E11 | `monitoring/*` | Model Monitoring |
| E12 | `data_quality/*` | Data Quality |
| E13 | `prism.sqlite` (alerts, audit_log, activity_log, anomalies) | Overview, Monitoring, Governance, AI Anomaly |

### 2.3 FastAPI layer (`ensemble_analytics_db/api/`)

33 routes exist today, grouped across `metadata`, `metrics`, `portfolio`, `clusters`, `trades`, `groups`, `governance`, `registry`, `admin`. The audit found:

| ID | Issue | Fix |
|---|---|---|
| A1 | `/admin/reload` uses `manifest.get("version")`; manifest has no `version` key | Use `cache.ensemble_version`. |
| A2 | `ArtifactCache.get_registry_versions()` never called | Wire into `/governance/versions` or delete. |
| A3 | `/registry/graph-data/{cid}` falls through to raw Python objects for non-tensor values | Tighten serialisation. |
| A4 | `/portfolio/timeseries` returns parallel lists without `x` | Add `scenario_index`. |
| A5 | `/groups/timeseries` nested dict is Retool-unfriendly | Provide Retool-shaped alias under `/prism/v1/*`. |
| A6 | `CorrelationMatrix` has `columns` + `values` but no `rows` | Add `rows`. |
| A7 | `/registry/elementary-pnl` returns row-major `data: List[List]` | Offer `orient="records"` option. |
| A8 | `/overview/*`, `/monitoring/*`, `/quality/*`, `/anomalies/*`, `/cross-cluster/*`, `/graph/layout`, `/graph/node/{id}`, per-cluster deep-dive aggregations are missing | Implemented under `/prism/v1/*`. |

---

## 3a. Dual-UI strategy — Dash prototype + Retool production

PRISM is delivered through **two UIs sitting on the same contract**:

```
                          ┌─────────────────────────────┐
                          │    ArtifactCache (memory)   │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │   FastAPI /prism/v1/*       │
                          │   (retool_adapters + models)│
                          └──────┬───────────────┬──────┘
                                 │               │
                ┌────────────────▼──┐        ┌───▼──────────────┐
                │ ui/apps/prism_dash│        │ ui/prism_retool  │
                │  (reference impl) │        │  (production)    │
                └───────────────────┘        └──────────────────┘
```

### Principles

1. **One contract, two consumers**. Every tile/chart/table is driven by exactly one `/prism/v1/*` endpoint. If it renders correctly in Dash, it renders correctly in Retool.
2. **Page parity**. Each PRISM mock PNG maps to:
   - one Retool page
   - one Dash page file under `ui/apps/prism_dash/pages/`
   - one design spec under `ui/prism_retool/design/page_*.md`
   Same name, same route slug, same widget list, same endpoint list.
3. **Shape-first rendering**. The endpoint JSON shape is authoritative. Both UIs bind directly to it with **zero transformation**. If transformation is needed, it goes in `retool_adapters.py` server-side, not in the client.
4. **No shared code paths between `ensemble_analytics_db` (legacy) and `prism_dash` (new)**. They coexist during the transition; the new app is a clean build against `/prism/v1/*`, not a refactor of the old one.
5. **Dash-first migration**. When a new endpoint is added it must be proven in Dash before it ships to Retool — Dash is the contract test harness. Dash being "working" is the green light to wire Retool.

### `ui/apps/prism_dash/` folder

```
ui/apps/prism_dash/
├── app.py                          # entry point, Dash(server=app, ...)
├── config.py                       # api_base_url, cache TTLs, feature flags
├── layout.py                       # sidebar + tab container
├── theme.py                        # PRISM dark theme tokens (colors, fonts, spacing)
├── pages/                          # ONE file per PRISM mock
│   ├── overview.py
│   ├── model_monitoring.py
│   ├── governance.py
│   ├── eval_portfolio.py
│   ├── eval_by_group.py            # desk/product/ccy/cluster in one page, dropdown-switched
│   ├── data_quality.py
│   ├── cross_cluster.py
│   ├── trade_graph.py
│   ├── cluster_deep_dive.py
│   └── ai_anomaly.py
├── components/                     # reusable widgets that mirror Retool components
│   ├── kpi_card.py                 # renders {value, delta, direction, as_of}
│   ├── status_badge.py             # renders {status, label, color}
│   ├── data_table.py               # binds array-of-objects → dash_table.DataTable
│   ├── line_chart.py               # binds [{x,y}, ...] → plotly.express.line
│   ├── multi_line_chart.py         # aligned-x and ragged variants
│   ├── bar_chart.py
│   ├── heatmap.py                  # binds {rows, cols, values}
│   ├── histogram.py                # binds [{bin, count}]
│   └── graph_view.py               # binds {nodes, edges} via dash-cytoscape
├── data/
│   ├── prism_client.py             # thin wrapper around requests; one method per endpoint; typed returns
│   └── cache.py                    # optional dcc.Store / lru_cache for endpoints under heavy polling
└── callbacks/                      # one module per page; callbacks are thin
    ├── overview_cb.py
    ├── ...
```

Key conventions:

- **No direct artifact access**: `prism_dash` never touches `ArtifactCache` or disk. It only speaks HTTP to `/prism/v1/*`. This is what makes it a true contract consumer.
- **One endpoint per callback**: exactly like a Retool query. Callbacks are 5–10 lines: fetch → bind.
- **Components mirror Retool widgets**: `components/kpi_card.py` takes the KPI JSON shape and returns a styled `html.Div`; its Retool equivalent is a KPI widget bound to the same JSON. Names match.
- **Theme tokens in one file**: `theme.py` holds the PRISM color/spacing tokens. Retool's theme config is exported to the same names so both UIs look identical.
- **Routing**: Dash `dcc.Location` + a lookup table from URL slug → page module. Same slugs as Retool app paths.

### Ownership of `ensemble_analytics_db`

- **Phases 0–7**: leave it running. It continues to serve as the production data-viz tool through the migration.
- **Phase 8**: `prism_dash` comes online page-by-page. When a PRISM page reaches feature parity with (or exceeds) the matching `ensemble_analytics_db` tab, that tab is marked legacy.
- **Phase 10**: once all pages are parity, `ensemble_analytics_db` is archived (moved to `src/ui/apps/legacy/ensemble_analytics_db/`). Its `/api/v1/*` routes can stay mounted indefinitely or be removed if nothing depends on them.

---

## 3. File format policy

### 3.1 NPZ vs Parquet

Both formats stay in the pipeline. The rules are:

| Data pattern | Format | Why |
|---|---|---|
| Wide numeric 2-D matrix (n_scenarios × n_trades), loaded wholesale into `ArtifactCache` at startup | **NPZ** | Zero-copy `mmap`, no deserialisation overhead, compact for float32, matches existing consumers. Examples: `members/{cid}/predictions/{split}.npz`, `portfolio_summary/{split}.npz`, `cluster_summary/{split}.npz`. |
| Long-format tabular data with multiple categorical columns, filtered/projected by the API | **Parquet** | Column projection (`columns=[...]`), predicate pushdown (`filters=[...]`), stable schema, native to pandas/Arrow/DuckDB. Examples: `group_residuals_{split}.parquet`, `cluster_residuals_{split}.parquet`, `monitoring/drift_{split}.parquet`. |
| Small structured metadata | **JSON** | Human-readable, git-diff-able. Examples: `manifest.json`, `cluster_health.json`, `ensemble_metrics_{split}.json`. |
| Event store (alerts, audit, activity, anomalies) | **SQLite** | Queryable, atomic appends, single file. `prism.sqlite`. |
| PNGs and other binary reports | **PNG** | As today. |

### 3.2 Consequences

- No existing NPZ/JSON file is removed. New artifacts are added alongside.
- `ArtifactCache` grows new attributes for each new artifact kind. One cache, two UIs (Dash + Retool).
- Where long-format parquet supersedes an in-memory computation (e.g. `_group_aggregations`), the API keeps the old route shape and internally switches to reading parquet, so callers notice no change.

---

## 4. Phased implementation plan

The plan is split into two tracks:

- **Foundation track (Phases 0–6)** — pipeline outputs. Build nothing UI-shaped until every artifact Retool and `prism_dash` will ever need is emitted, validated, and cacheable. This is the current focus.
- **Delivery track (Phases 7–10)** — API layer, `prism_dash`, `prism_retool`, legacy sunset. Gated on Foundation track completion (or at minimum, per-page gated on the relevant Foundation phase).

Each phase has the same structure:

- **Goal** — one-line outcome.
- **Scope** — what is in and explicitly out.
- **Deliverables** — code, artifacts, tests.
- **Dash compatibility** — can existing UI consume this now?
- **Implementation notes** — filled in when the phase is actively picked up.
- **Status** — pending / in-progress / done.

### Foundation track

### Phase 0 — Priority-1 bug fixes (no new data)

- **Goal**: unblock convergence PNG, governance member table, admin reload version.
- **Scope — in**: T1 (training plot rename), T4 (enrich `member_summary.json`), A1 (admin reload version), A3 (graph serialisation fallback). Optionally T2, T3 if cheap.
- **Scope — out**: any new artifact.
- **Deliverables**
  - Patch `HybridGnnRnnTrainPipeline._post_train_plots` to write `training_plots.png` (or have `ArtifactCache` try both names — TBD preference).
  - Extend `member_summary.json` writer in `EnsembleTrainPipeline._register_ensemble` to include `ccy`, `desk`, `product`, `trained_at`, `model_version` from `cluster_key_values` + registry timestamp.
  - Change `/admin/reload` to use `cache.ensemble_version`.
  - Harden `registry.get_graph_data` serialisation for arbitrary objects (convert unknown types to `repr` or drop with warning).
- **Dash compatibility**: immediate — the existing convergence PNG path starts working, and any existing governance code reading `member_summary.json` keeps working (keys are additive).
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 1 — Eval output restatements (B-series parquets)

- **Goal**: restate data the eval pipeline already computes as long-format parquet, so Overview + Portfolio + Eval By X + Cross-Cluster tabs have a queryable source.
- **Scope — in**: B1–B8 in §7. No new analytics; pure re-shaping.
- **Scope — out**: cluster health, monitoring, data quality, graph layouts.
- **Deliverables**
  - New method `EnsembleEvalPipeline._save_retool_artifacts(self, split)` called after `_save_all_artifacts` when `save_retool_artifacts` config flag is true (default true; can be turned off).
  - Parquet emitters for:
    - `portfolio_timeseries_{split}.parquet` — `scenario_index, predicted, target, residual`.
    - `portfolio_residuals_{split}.parquet` — `scenario_index, residual, abs_error, sign`.
    - `cluster_residuals_{split}.parquet` — `cluster_id, scenario_index, predicted, target, residual`.
    - `group_metrics_{split}.parquet` — `group_type, group_value, mae, rmse, p95_ae, p99_ae, n_trades, n_scenarios`.
    - `group_timeseries_{split}.parquet` — `group_type, group_value, scenario_index, predicted, target`.
    - `group_residuals_{split}.parquet` — `group_type, group_value, scenario_index, residual, abs_error`.
    - `cross_cluster/correlation_{split}.parquet` — `cluster_a, cluster_b, pearson, spearman`.
    - `cross_cluster/size_vs_perf_{split}.parquet` — `cluster_id, n_trades, n_scenarios, mae, rmse, sharpe, health_score` (health from Phase 2; null for now).
  - New keys on `ArtifactCache` + loader methods.
  - Unit tests that round-trip a synthetic ensemble through the emitter and check schemas.
- **Dash compatibility**: Yes, immediately. Existing routes that compute `_group_aggregations` in memory can switch to reading parquet internally (response shape unchanged). Dash tabs for By Desk/Ccy/Product/Cluster get faster and more memory-efficient.
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 2 — Overview aggregates + cluster health (C-series)

- **Goal**: produce the data that drives the Overview tab.
- **Scope — in**: C1 `cluster_health.json`, C2 `overview_summary.json`, C3 `attention_required.json`.
- **Deliverables**
  - `src/rade_ml_pt/ensemble/health.py` — `compute_cluster_health(metrics, trade_counts, freshness) -> Dict[cid, HealthScore]`. Components: accuracy (from MAE / P95 AE), stability (residual std over split), coverage (n_trades ratio), freshness (evaluated_at age).
  - Grading: `A/B/C/D/F` from a weighted sum with tunable weights (default in code).
  - `overview_summary.json` — headline KPIs (portfolio MAE, n_clusters, n_trades, last eval timestamp, version, active alerts count).
  - `attention_required.json` — list of `{cluster_id, reason, severity, metric, value}` produced by rules defined in `health.py`.
  - Update `size_vs_perf_{split}.parquet` (from Phase 1) to fill in `health_score`.
- **Dash compatibility**: Yes — adds a fast Overview data source the Dash Overview tab can also use.
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 3 — Trade graph layouts (D-series)

- **Goal**: precompute node-link layouts so both UIs can render the Trade Graph tab without recomputing on every request.
- **Scope — in**: D1 rename `graph_stats.json` → `graph/stats.json`; D2 `layout.parquet`; D3 `edges.parquet`; D4 `graph/all/layout.parquet`; D5 `node_analytics.parquet`.
- **Deliverables**
  - New module `src/rade_ml_pt/ensemble/graph_layout.py` with:
    - `build_cluster_layout(cluster_id, graph_results, trade_universe, seed) -> (nodes_df, edges_df)` using `networkx.spring_layout` for small graphs, `umap-learn` on encoder embeddings for large graphs. Deterministic seed. Output columns listed in §7.
    - `build_global_layout(cluster_layouts) -> nodes_df` — stitched global coordinates.
    - `compute_node_analytics(graph) -> df` — degree, betweenness, local clustering, eigenvector centrality.
  - Post-eval step in `EnsembleEvalPipeline._save_retool_artifacts` to call the builders and write parquet under `evaluation/graph/`.
  - Optional standalone script `scripts/build_graph_layouts.py` for reprocessing without full eval.
- **Dash compatibility**: Yes — Trade Graph tab in Dash can stop spring-layouting at render time and read the parquet instead.
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 4 — Monitoring artifacts (E-series)

- **Goal**: drift, feature shift, residual stability, version history.
- **Scope — in**:
  - **Training side**: E1 `feature_distributions.parquet`, E2 `residual_distribution.parquet` baselines, saved once per member at training completion.
  - **Eval side**: E3 `drift_{split}.parquet` (PSI + JS), E4 `feature_shift_{split}.parquet`, E5 `residual_stability_{split}.parquet`, E6 `version_history.parquet` (appended across runs), E7 `drift_heatmap_{split}.parquet`.
- **Deliverables**
  - New module `src/rade_ml_pt/monitoring/` with:
    - `baselines.py` — compute per-feature histograms and residual distributions at training time; write parquet.
    - `drift.py` — PSI, JS divergence, population drift tests; compare current features and residuals to baselines.
    - `stability.py` — rolling mean/std of residuals per cluster + MAE.
    - `version_history.py` — append-only writer, with file lock.
  - Hook into `HybridGnnRnnTrainPipeline.post_train` (baselines) and `EnsembleEvalPipeline._save_retool_artifacts` (drift + stability + history).
- **Dash compatibility**: Yes — Dash gets a new Monitoring tab fed by the same parquets.
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 5 — Data quality artifacts (F-series)

- **Goal**: feed the Data Quality tab.
- **Scope — in**: F1–F5.
- **Deliverables**
  - `src/rade_ml_pt/quality/` module:
    - `pipeline_status.py` — status dict captured at each stage of training/eval (start, end, rows in/out, errors).
    - `completeness.py` — null rate, distinct count, dtype per feature.
    - `freshness.py` — source latency.
    - `feature_summary.py` — `count, mean, std, p1, p50, p99, min, max`.
    - `outliers.py` — sqlite writer for outlier log.
  - Invoked both at training (baseline DQ snapshot) and at eval (current DQ snapshot).
- **Dash compatibility**: Yes — new Dash tab or additions to existing tabs.
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 6 — Event store (G-series)

- **Goal**: single `prism.sqlite` with `activity_log`, `audit_log`, `alerts`, `anomalies`.
- **Scope — in**: G1; writers in each appropriate pipeline stage.
- **Deliverables**
  - Schema DDL at `src/rade_ml_pt/ensemble/prism_schema.sql`.
  - `src/rade_ml_pt/ensemble/event_store.py` — thin writer / reader helpers using sqlite3.
  - Ingest hooks: eval completion → `activity_log`; monitoring rules → `alerts`; governance/admin operations → `audit_log`; anomaly detector (future) → `anomalies`.
- **Dash compatibility**: Yes — Dash tabs that surface alerts/activity can read sqlite via the same helpers.
- **Implementation notes**: *TBD*
- **Status**: pending

### Delivery track

### Phase 7 — FastAPI `/prism/v1/*` layer + ArtifactCache extensions

- **Goal**: expose the new artifacts to Retool (and Dash via a second `DataBackend` variant if wanted).
- **Scope — in**: new routers under `src/ui/apps/ensemble_analytics_db/api/routers/prism/`, Retool-shaped Pydantic models under `api/models/prism/`, adapters in `api/services/retool_adapters.py`, `api/services/health_scorer.py`. Fix existing issues A1–A7 on `/api/v1/*` where useful to Dash.
- **Deliverables**
  - Routers: `overview.py`, `monitoring.py`, `governance.py`, `eval.py`, `quality.py`, `cross_cluster.py`, `graph.py`, `cluster.py`, `anomalies.py`. Each endpoint maps 1:1 to the catalogue in §7.
  - Extended `ArtifactCache` with loaders for every artifact added in Phases 1–6.
  - Contract tests covering every `/prism/v1/*` response shape.
- **Dash compatibility**: Yes, optional — a future `PrismApiBackend` can front the Dash UI onto these endpoints, or Dash can keep using `/api/v1/*`.
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 8 — `ui/apps/prism_dash/` reference UI (Dash prototype)

- **Goal**: a clean-slate Dash app that mirrors the PRISM mocks 1:1 and consumes `/prism/v1/*` over HTTP only. Serves as the always-available prototype and as the contract-test harness for Retool.
- **Scope — in**: folder layout in §3a; theme tokens; shared components; one page module per PRISM mock; `prism_client.py` HTTP wrapper; routing.
- **Scope — out**: modifying `ensemble_analytics_db`; direct artifact access from the new app.
- **Deliverables**
  - `ui/apps/prism_dash/app.py` + `layout.py` + `theme.py` + `config.py`.
  - Pages: `overview`, `model_monitoring`, `governance`, `eval_portfolio`, `eval_by_group`, `data_quality`, `cross_cluster`, `trade_graph`, `cluster_deep_dive`, `ai_anomaly` — one file each under `pages/`.
  - Shared components library under `components/` that renders the Retool JSON shape conventions from §7.3 (KPI card, status badge, data table, line/bar/multi-line charts, heatmap, histogram, graph view).
  - `data/prism_client.py` — typed HTTP wrapper with one method per endpoint in §7.2, returning Pydantic models (re-use `api/models/prism/*`).
  - Per-page callback module under `callbacks/`; each callback makes exactly one client call and binds the result.
  - Launcher script `examples/rade_ml_pt/hybrid_gnn_rnn/11_run_prism_dash.py` pointing Dash at `http://localhost:8000/prism/v1`.
  - Smoke test that hits every page, asserts the page loads and at least one widget is non-empty against a fixtures-backed API.
- **Dash compatibility**: this *is* the new Dash. Legacy `ensemble_analytics_db` remains untouched.
- **Page-by-page incremental delivery**: ship pages in the same order data lands (Overview, Eval Portfolio, Eval By Group, Cross-Cluster, Cluster Deep Dive, Governance, Trade Graph, Model Monitoring, Data Quality, AI Anomaly). Each page is one PR.
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 9 — `ui/prism_retool/*` scaffolding + first Retool pages

- **Goal**: the Retool contract + app export + page specs, reusing the Dash pages as the reference.
- **Scope — in**: folder layout below; contract tests generated from Pydantic models; fixture capture script; first Retool app export covering pages that have already landed in `prism_dash`.
- **Folder**
  ```
  ui/prism_retool/
  ├── README.md
  ├── contract/
  │   ├── openapi_prism.json
  │   ├── test_prism_contract.py     # asserts prism_dash + Retool see the same shapes
  │   └── schemas/
  ├── retool_apps/
  │   ├── prism_v1.json
  │   ├── prism_v1_queries.json
  │   └── CHANGELOG.md
  ├── design/
  │   ├── page_overview.md           # each file points at pages/overview.py in prism_dash
  │   ├── page_monitoring.md
  │   ├── page_governance.md
  │   ├── page_eval_portfolio.md
  │   ├── page_eval_by_desk.md
  │   ├── page_data_quality.md
  │   ├── page_cross_cluster.md
  │   ├── page_trade_graph.md
  │   ├── page_cluster_deep_dive.md
  │   └── page_ai_anomaly.md
  ├── fixtures/                      # captured JSON per endpoint for Retool preview mode
  └── scripts/
      ├── dump_openapi.py
      ├── capture_fixtures.py
      └── validate_retool_export.py
  ```
- **Implementation notes**: *TBD*
- **Status**: pending

### Phase 10 — Sunset `ensemble_analytics_db`

- **Goal**: retire the legacy Dash app once `prism_dash` reaches feature parity.
- **Scope — in**: parity checklist (one row per legacy tab, mapped to a `prism_dash` page), archive move to `src/ui/apps/legacy/ensemble_analytics_db/`, decision on whether to keep `/api/v1/*` routes mounted.
- **Deliverables**: parity checklist; PR that moves the legacy app; removal of launcher scripts that target it; announcement in `UPDATE.md`.
- **Implementation notes**: *TBD*
- **Status**: pending

---

## 5. Execution order (recommended)

Two guiding rules:

- **Data before UI.** Phases 0–6 land the artifacts. Phase 7 exposes them. Only then do we build UIs.
- **Vertical slices after that.** From Phase 7 onwards, we go page-by-page: for each PRISM mock, ship endpoint → Pydantic model → `prism_dash` page → Retool page, in that order. One PR per page.

Full order:

1. **Phase 0** — priority-1 bug fixes. Low risk, unblocks convergence/governance.
2. **Phase 1** — B-series parquets. No new analytics; pure restatement. Unblocks Overview, Portfolio, Eval By X, Cross-Cluster data.
3. **Phase 2** — cluster health + overview aggregates. Small new analytics.
4. **Phase 3** — graph layouts. Independent of 2.
5. **Phase 4** — monitoring artifacts (needs a training rerun for E1/E2 baselines).
6. **Phase 5** — data quality.
7. **Phase 6** — event store.
8. **Phase 7** — `/prism/v1/*` FastAPI layer + `ArtifactCache` loaders. After this the contract is ready for both UIs.
9. **Phase 8** — `ui/apps/prism_dash/` page-by-page. Page order (each a PR):
   1. Overview
   2. Eval Portfolio
   3. Eval By Group
   4. Cross-Cluster
   5. Cluster Deep Dive
   6. Governance
   7. Trade Graph
   8. Model Monitoring
   9. Data Quality
   10. AI Anomaly
   Same order used in Phase 9 for Retool pages.
10. **Phase 9** — `ui/prism_retool/*` scaffolding + Retool app export for pages already delivered in Phase 8.
11. **Phase 10** — sunset `ensemble_analytics_db`.

Phases 1–6 can all be validated independently with a small synthetic ensemble; the smoke test that comes with Phase 1 runs in CI and grows with each subsequent phase.

---

## 6. Cross-cutting concerns

- **Backwards compatibility**: no existing file or route is removed in Phases 0–7. All changes are additive.
- **Config flags**: each new phase is guarded by a flag on `EnsembleConfig` (e.g. `save_retool_artifacts`, `save_monitoring_artifacts`, `save_quality_artifacts`). Default ON once proven.
- **Determinism**: all graph layouts and sampling-based analytics take a seed.
- **CI**: add a `tests/rade_ml_pt/pipelines/ensemble/test_eval_retool_artifacts.py` that runs the pipeline on a toy ensemble and asserts every new file exists with the expected schema.
- **Performance budget**: the Retool dashboard targets < 300 ms p95 per endpoint on a warm `ArtifactCache`. New parquet reads should hit this with `pyarrow.dataset` + column projection.

---

## 7. Output catalogue

Legend: `S` = ✅ exists today, ⚠️ exists but needs shape change, ❌ new.

### 7.1 Paths (all under `{artifacts_dir}/ensemble/{ens_version}/evaluation/` unless noted)

| ID | Path | Format | S | Owner |
|---|---|---|---|---|
| A1 | `manifest.json` | json | ✅ | eval |
| A2 | `cluster_attributes.json` | json | ✅ | eval |
| A3 | `trade_cluster_map.json` | json | ✅ | eval |
| A4 | `ensemble_metrics_{split}.json` | json | ✅ | eval |
| A5 | `per_member_metrics_{split}.json` | json | ✅ | eval |
| A6 | `portfolio_summary/{split}.npz` | npz | ✅ | eval |
| A7 | `cluster_summary/{split}.npz` | npz | ✅ | eval |
| A8 | `members/{cid}/predictions/{split}.npz` | npz | ✅ | eval |
| A9 | `trade_metrics/{split}.json` | json | ✅ | eval |
| A10 | `group_correlations/{split}.json` | json | ✅ | eval |
| B1 | `portfolio_timeseries_{split}.parquet` | parquet | ❌ | eval |
| B2 | `portfolio_residuals_{split}.parquet` | parquet | ❌ | eval |
| B3 | `cluster_residuals_{split}.parquet` | parquet | ❌ | eval |
| B4 | `group_metrics_{split}.parquet` | parquet | ❌ | eval |
| B5 | `group_timeseries_{split}.parquet` | parquet | ❌ | eval |
| B6 | `group_residuals_{split}.parquet` | parquet | ❌ | eval |
| B7 | `cross_cluster/correlation_{split}.parquet` | parquet | ❌ | eval |
| B8 | `cross_cluster/size_vs_perf_{split}.parquet` | parquet | ❌ | eval |
| C1 | `cluster_health.json` | json | ❌ | eval |
| C2 | `overview_summary.json` | json | ❌ | eval |
| C3 | `attention_required.json` | json | ❌ | eval |
| D1 | `graph/stats.json` (rename of today's `graph_stats.json`) | json | ⚠️ | eval |
| D2 | `graph/{cid}/layout.parquet` | parquet | ❌ | eval |
| D3 | `graph/{cid}/edges.parquet` | parquet | ❌ | eval |
| D4 | `graph/all/layout.parquet` | parquet | ❌ | eval |
| D5 | `graph/{cid}/node_analytics.parquet` | parquet | ❌ | eval |
| E1 | `monitoring/baselines/feature_distributions.parquet` | parquet | ❌ | train |
| E2 | `monitoring/baselines/residual_distribution.parquet` | parquet | ❌ | train |
| E3 | `monitoring/drift_{split}.parquet` | parquet | ❌ | eval |
| E4 | `monitoring/feature_shift_{split}.parquet` | parquet | ❌ | eval |
| E5 | `monitoring/residual_stability_{split}.parquet` | parquet | ❌ | eval |
| E6 | `monitoring/version_history.parquet` (append) | parquet | ❌ | eval |
| E7 | `monitoring/drift_heatmap_{split}.parquet` | parquet | ❌ | eval |
| F1 | `data_quality/pipeline_status.json` | json | ❌ | train+eval |
| F2 | `data_quality/completeness.parquet` | parquet | ❌ | train+eval |
| F3 | `data_quality/freshness.parquet` | parquet | ❌ | eval |
| F4 | `data_quality/feature_summary.parquet` | parquet | ❌ | train+eval |
| F5 | `data_quality/outliers.sqlite` | sqlite | ❌ | eval |
| G1 | `prism.sqlite` | sqlite | ❌ | eval + governance + monitoring |

Plus the training-side `{artifacts_dir}/training/{member_version}/training_plots.png` (T1 — rename from `training_analytics.png`) and the enriched `{registry_dir}/ensemble/{ens_version}/member_summary.json` (T4).

### 7.2 Endpoint ↔ artifact ↔ Retool JSON shape

All endpoints under `/prism/v1`.

#### Overview

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /overview/summary` | C2, C1, G1 | `{version, evaluated_at, kpis:[{label,value,delta,direction}], splits_available:[...]}` |
| `GET /overview/pnl-timeseries?split=test` | B1 | `[{scenario_index, predicted, target, residual}, ...]` |
| `GET /overview/cluster-health` | C1 | `[{cluster_id, score, grade, n_trades, mae, desk, ccy}, ...]` |
| `GET /overview/attention-required` | C3, G1 | `[{cluster_id, reason, severity, metric, value}, ...]` |
| `GET /overview/recent-activity?limit=20` | G1 | `[{ts, kind, message}, ...]` |

#### Model Monitoring

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /monitoring/drift-psi?split=test` | E3 | `[{feature, psi, severity}, ...]` |
| `GET /monitoring/feature-shift?feature=X` | E1, E4 | `{baseline:[{bin,count}], current:[{bin,count}], z_shift}` |
| `GET /monitoring/residual-stability?cluster_id=...&split=test` | E5 | `[{window, mean_residual, residual_std, rolling_mae}, ...]` |
| `GET /monitoring/version-history` | E6 | `[{version, evaluated_at, mae, rmse, drift_max, n_alerts}, ...]` |
| `GET /monitoring/alerts?resolved=false` | G1 | `[{id, ts, severity, cluster_id, feature, message}, ...]` |
| `GET /monitoring/drift-heatmap?split=test` | E7 | `{rows:[cluster...], cols:[feature...], values:[[psi...]]}` |

#### Governance

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /governance/manifest` | A1 + `ensemble_config.json` | `{version, trained_at, evaluated_at, git_sha, n_clusters, n_trades, n_scenarios, splits}` |
| `GET /governance/members` | T4-enriched `member_summary.json` | `[{cluster_id, ccy, desk, product, n_trades, best_val_loss, trained_at, model_version}, ...]` |
| `GET /governance/version-compare?base=a&target=b` | existing compare logic | `[{metric, base, target, delta, direction}, ...]` |
| `GET /governance/config/{cluster_id}` | `member_configs` | `{cluster_id, config}` |
| `GET /governance/audit?limit=50` | G1 | `[{ts, actor, action, target}, ...]` |

#### Evaluation — Portfolio

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /eval/portfolio/metrics?split=test` | A4 | `[{metric, value}, ...]` |
| `GET /eval/portfolio/timeseries?split=test` | B1 | `[{scenario_index, predicted, target, residual}, ...]` |
| `GET /eval/portfolio/residuals?split=test` | B2 | `[{scenario_index, residual, abs_error, sign}, ...]` |
| `GET /eval/portfolio/percentiles?split=test` | computed | `[{percentile, predicted, target, abs_error}, ...]` |
| `GET /eval/portfolio/worst-scenarios?split=test&n=20` | computed | `[{scenario_index, predicted, target, abs_error}, ...]` |

#### Evaluation — By Desk / Product / Ccy / Cluster

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /eval/group/{group_type}/metrics?split=test` | B4 | `[{group_value, mae, rmse, p95_ae, p99_ae, n_trades, n_scenarios}, ...]` |
| `GET /eval/group/{group_type}/timeseries?split=test&group_value=...` | B5 | `[{scenario_index, predicted, target}, ...]` |
| `GET /eval/group/{group_type}/residuals?split=test&group_value=...` | B6 | `[{scenario_index, residual, abs_error}, ...]` |

#### Data Quality

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /quality/pipeline-status` | F1 | `[{name, status, started_at, finished_at, rows_in, rows_out, errors}, ...]` |
| `GET /quality/completeness` | F2 | `[{feature, null_rate, distinct_count, dtype}, ...]` |
| `GET /quality/freshness` | F3 | `[{source, as_of, lag_seconds, status}, ...]` |
| `GET /quality/feature-summary` | F4 | `[{feature, count, mean, std, p1, p50, p99, min, max}, ...]` |
| `GET /quality/outliers?limit=100` | F5 | `[{trade_id, feature, value, z_score, observed_at}, ...]` |

#### Cross-Cluster

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /cross-cluster/metrics?split=test` | A5 | `[{cluster_id, mae, rmse, p95_ae, n_trades}, ...]` |
| `GET /cross-cluster/correlation?split=test` | B7 | `{rows:[cluster...], cols:[cluster...], values:[[...]]}` |
| `GET /cross-cluster/residuals?split=test` | B3 | `[{cluster_id, scenario_index, residual}, ...]` |
| `GET /cross-cluster/size-vs-performance?split=test` | B8 | `[{cluster_id, n_trades, mae, health_score, desk}, ...]` |

#### Trade Graph

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /graph/stats` | D1 | `[{cluster_id, n_nodes, n_edges, density, mean_weight}, ...]` |
| `GET /graph/layout?cluster_id=...` | D2 | `{nodes:[{node_id,x,y,degree,betweenness,ccy,desk}], meta:{...}}` |
| `GET /graph/edges?cluster_id=...&min_weight=0.1` | D3 | `[{source, target, weight}, ...]` |
| `GET /graph/all/layout` | D4 | `{nodes:[...], meta:{...}}` |
| `GET /graph/node/{cluster_id}/{node_id}` | D2 + D5 + trade catalogue | `{node_id, ccy, desk, neighbours:[...], analytics:{...}, trade_attributes:{...}}` |

#### Cluster Deep Dive

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /cluster/{cid}/summary?split=test` | A4, A5, C1 | `{cluster_id, kpis:[{label,value}], attrs:{desk,ccy,product}}` |
| `GET /cluster/{cid}/training-curve` | `training_plots.png` (T1) | `{png_base64, epochs:[...], train_loss:[...], val_loss:[...]}` |
| `GET /cluster/{cid}/predictions?split=test&mode=summary` | A8 | `[{scenario_index, predicted, target}, ...]` or `[{trade_id, scenario_index, predicted, target}, ...]` |
| `GET /cluster/{cid}/elementary-pnl` | `elementary_pnl.parquet` | `{trades:[...], stats:[{trade_id, mean, std, min, max}, ...], series:[{scenario_index, trade_id, pnl}, ...]}` |
| `GET /cluster/{cid}/member-config` | `member_configs[cid]` | `{cluster_id, config}` |
| `GET /cluster/{cid}/residuals?split=test` | B3 filtered | `[{scenario_index, residual, abs_error}, ...]` |

#### AI Anomaly Investigation

| Endpoint | Reads | JSON shape |
|---|---|---|
| `GET /anomalies?limit=50&severity=...` | G1 | `[{id, ts, trade_id, cluster_id, score, summary}, ...]` |
| `GET /anomalies/{id}/context` | G1 + trade + cluster lookups | `{anomaly:{...}, trade:{...}, cluster:{...}, nearest_neighbours:[...], suggested_checks:[...]}` |
| `POST /anomalies/{id}/ack` | writes G1 `audit_log` | `{ok: true, id}` |

### 7.3 Retool JSON shape conventions (cheat sheet)

| UI element | Shape |
|---|---|
| Table | `[{col1:val, col2:val}, ...]` |
| Line chart (single series) | `[{x, y}, ...]` |
| Line chart (multi-series, aligned x) | `[{x, seriesA, seriesB, seriesC}, ...]` |
| Line chart (multi-series, ragged) | `{x:[...], series:[{name, y:[...]}]}` |
| Bar chart | `[{category, value}, ...]` |
| Heatmap | `{rows:[...], cols:[...], values:[[...]]}` |
| KPI | `{value, delta, direction, as_of}` |
| Status badge | `{status, label, color}` |
| Histogram bin | `[{bin, count}, ...]` |

---

## 8. Open questions

| # | Question | Candidate answers |
|---|---|---|
| Q1 | For T1, rename the trainer output to `training_plots.png`, or make `ArtifactCache` accept both names? | Prefer rename (single name in codebase). TBD. |
| Q2 | Where should `ensemble_manifest.json` (with `trained_at`, `git_sha`, etc.) live — next to `ensemble_config.json` under registry, or under `artifacts_dir/ensemble/{ver}/`? | TBD. |
| Q3 | Who owns `prism.sqlite` creation and migration — eval pipeline or a standalone `prism_init` script? | TBD. |
| Q4 | Do we keep JSON for `cluster_health.json` + `overview_summary.json` or move to parquet once stable? | Prefer JSON for readability. |
| Q5 | Which encoder is the source of UMAP coordinates for the graph layout — the trained encoder outputs (`encoder_results.joblib`), or raw features? | TBD; default to encoder outputs. |
| Q6 | Which existing `/api/v1/*` routes do we want to quietly upgrade (e.g. switch to reading parquet internally) vs leave untouched? | Default: leave untouched; add `/prism/v1/*` alongside. |
| Q7 | ~~Should Dash migrate to `/prism/v1/*` via a new `PrismApiBackend`?~~ | **Resolved (2026-04-17)**: new app `ui/apps/prism_dash/` is built clean against `/prism/v1/*`; legacy `ensemble_analytics_db` stays on `/api/v1/*` until Phase 10 sunset. |
| Q8 | `prism_dash` launcher: mount under the existing FastAPI process (Dash at `/dash` path), or run as a standalone Dash server that calls FastAPI? | Prefer standalone (simpler, same as current apps). TBD. |
| Q9 | Graph rendering in Dash — `dash-cytoscape` or `plotly` scattergl? | Cytoscape for interactivity on medium graphs; plotly fallback for huge graphs. TBD per page. |
| Q10 | Theme source of truth — hand-author `theme.py` tokens and mirror into Retool, or export from Retool into Dash? | Start hand-authored; revisit once Retool is live. |

---

## 9. Change log

| Date | Author | Change |
|---|---|---|
| 2026-04-17 | assistant + joesstevens | Initial document; audit findings, phased plan, output catalogue. |
| 2026-04-17 | assistant + joesstevens | Added §3a dual-UI strategy. Introduced `ui/apps/prism_dash/` as PRISM reference implementation. Restructured Phase 8 (prism_dash), Phase 9 (prism_retool scaffolding), Phase 10 (legacy sunset). Resolved Q7; added Q8–Q10. |
| 2026-04-17 | assistant + joesstevens | Split phases into Foundation track (0–6, pipeline outputs) and Delivery track (7–10, API + UIs). Foundation is the active focus. |
