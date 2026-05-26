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

## Appendix A — Phase M.1: Monitoring drift primitives (copy-paste sync)

> **Status**: Five files below land in the repo under
> `src/rade_ml_pt/monitoring/` (production code) and
> `tests/rade_ml_pt/monitoring/` (tests). 53 unit + smoke tests pass,
> lint-clean. This appendix is the single source of truth for M.1 —
> copy verbatim into work env.
>
> **Scope**: Pure NumPy / pandas. No I/O beyond `pd.read_parquet`. No
> dependency on the inference pipeline. M.2 (the
> `EnsembleMonitoringPipeline` that wires drift into the artifact
> layout) consumes only the public symbols re-exported from
> `monitoring/__init__.py` — keep that surface stable.
>
> **What the M.1 primitives compute** (per cluster, per feature in the
> training-time scaled space):
> * `psi` — Population Stability Index against the persisted histogram
>   (51 edges + 50 counts) saved by `monitoring.baselines.save_feature_baseline`.
>   Outlier handling: today's values are clipped into the outermost
>   bins so an unseen tail value can't blow PSI up to infinity.
> * `js_divergence` — Jensen-Shannon divergence (log base 2 so bounded
>   `[0, 1]`) on the same paired histograms.
> * `mean_shift` — `(μ_today - μ_train) / max(σ_train, ε)`, i.e. a
>   z-score in training-std units (good for traders who want
>   "how many sigmas off are we?").
> * `std_ratio` — `σ_today / max(σ_train, ε)`. Volatility regime
>   indicator; >1 → wider, <1 → tighter than calibration.
> * `severity` — `info` (PSI < 0.10) / `warn` (0.10 ≤ PSI < 0.25) /
>   `critical` (PSI ≥ 0.25) / `no_data` (NaN or unscoreable). Industry
>   defaults; consumed by every downstream colour scale.
>
> **What the portfolio aggregator returns** (consumed by the M.2
> manifest writer and the M.5 health-strip): clusters seen, severity
> counts, mean/max PSI, worst cluster + feature, portfolio severity.

### File 1: `src/rade_ml_pt/monitoring/drift.py` (full source)

```python
"""Drift metrics: PSI, JS divergence, per-feature drift table.

Pure numerical helpers — no I/O.  Callers (eval pipeline, inference
pipeline, monitoring pipeline) are responsible for loading baseline
parquets and current feature matrices.

Schema and thresholds follow the design in
``docs/platform_designs/prism_retool_migration.md`` §11.15 / Phase 4
(E-series artifacts).  See ``monitoring.baselines`` for the writer side
(training-time histograms with persisted edges) and
``monitoring.loaders.load_baseline`` for decoding those parquets back
into NumPy-ready DataFrames.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


SCHEMA_VERSION: int = 1

# Industry-standard PSI severity thresholds.  Single source of truth
# for this module + every downstream consumer (UI colour scales, alert
# rules, etc.).  Boundaries are inclusive on the upper side
# (``psi == 0.10`` → ``"warn"``;  ``psi == 0.25`` → ``"critical"``).
PSI_WARN_THRESHOLD:     float = 0.10
PSI_CRITICAL_THRESHOLD: float = 0.25

# Sentinel labels returned by ``classify_severity`` so the UI can
# distinguish missing-data cells from genuinely-stable features.
SEVERITY_INFO:     str = "info"
SEVERITY_WARN:     str = "warn"
SEVERITY_CRITICAL: str = "critical"
SEVERITY_NO_DATA:  str = "no_data"


# ═════════════════════════════════════════════════════════════════════
# Single-feature primitives — PSI + JSD
# ═════════════════════════════════════════════════════════════════════

def population_stability_index(
    baseline_counts: np.ndarray,
    baseline_edges:  np.ndarray,
    current_values:  np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between a baseline histogram and current obs.

    .. math::

        \\text{PSI} = \\sum_i (p_{\\text{curr},i} - p_{\\text{base},i})
                     \\times \\ln\\!\\left(\\frac{p_{\\text{curr},i}}
                                                 {p_{\\text{base},i}}\\right)

    Each bin proportion is smoothed by ``epsilon`` to avoid div-by-zero
    and ``log(0)`` when the baseline or current has empty bins.  Current
    observations that fall outside
    ``[baseline_edges[0], baseline_edges[-1]]`` are clipped into the
    outermost bin — this matches the PSI convention of "lump outliers
    into the tail" and prevents PSI from spuriously spiking just
    because today saw a value the baseline never did.

    Non-finite ``current_values`` (NaN / inf) are dropped before
    binning, so callers don't have to do that themselves.

    Parameters
    ----------
    baseline_counts
        Histogram counts at training time (length ``n_bins``).
    baseline_edges
        Histogram edges (length ``n_bins + 1``).
    current_values
        Today's raw observations.  No prior binning expected.
    epsilon
        Bin proportion floor to prevent log(0) / divide-by-zero.

    Returns
    -------
    float
        PSI value.  Returns ``np.nan`` when either side is degenerate
        (all-zero counts, mismatched shapes, empty current observations).
    """
    counts = np.asarray(baseline_counts, dtype=np.float64)
    edges  = np.asarray(baseline_edges,  dtype=np.float64)
    cur    = np.asarray(current_values,  dtype=np.float64)
    cur    = cur[np.isfinite(cur)]

    if counts.size == 0 or edges.size != counts.size + 1:
        return float("nan")
    if cur.size == 0:
        return float("nan")
    if counts.sum() <= 0:
        return float("nan")

    cur_clipped   = np.clip(cur, edges[0], edges[-1])
    cur_counts, _ = np.histogram(cur_clipped, bins=edges)
    if cur_counts.sum() <= 0:
        return float("nan")

    p_base = counts     / counts.sum()
    p_curr = cur_counts / cur_counts.sum()

    p_base = p_base + epsilon
    p_curr = p_curr + epsilon

    return float(np.sum((p_curr - p_base) * np.log(p_curr / p_base)))


def js_divergence(
    baseline_counts: np.ndarray,
    current_counts:  np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Jensen-Shannon divergence between two histograms (log base 2).

    .. math::

        \\text{JSD}(P \\| Q) = \\tfrac{1}{2}\\text{KL}(P \\| M)
                             + \\tfrac{1}{2}\\text{KL}(Q \\| M),
        \\quad M = \\tfrac{1}{2}(P + Q)

    Bounded ``[0, 1]`` when log base 2 is used (which is the convention
    used here); symmetric in ``P``, ``Q``.  Both inputs are normalised
    to probability mass functions and smoothed by ``epsilon`` to avoid
    ``log(0)`` on empty bins.

    Parameters
    ----------
    baseline_counts, current_counts
        Histogram counts.  Must share the same bin layout (same
        length) — the caller is responsible for ensuring they were
        binned against identical edges.
    epsilon
        Probability floor to prevent ``log(0)``.

    Returns
    -------
    float
        JSD value in ``[0, 1]``.  Returns ``np.nan`` on shape mismatch
        or degenerate inputs (either array all-zero).
    """
    p = np.asarray(baseline_counts, dtype=np.float64)
    q = np.asarray(current_counts,  dtype=np.float64)
    if p.shape != q.shape or p.size == 0:
        return float("nan")
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")

    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    p_safe = p + epsilon
    q_safe = q + epsilon
    m_safe = m + epsilon

    kl_pm = float(np.sum(p_safe * np.log2(p_safe / m_safe)))
    kl_qm = float(np.sum(q_safe * np.log2(q_safe / m_safe)))
    return 0.5 * kl_pm + 0.5 * kl_qm


# ═════════════════════════════════════════════════════════════════════
# Severity classifier
# ═════════════════════════════════════════════════════════════════════

def classify_severity(psi: Optional[float]) -> str:
    """Map a PSI value to ``info`` / ``warn`` / ``critical`` / ``no_data``.

    ====================  ================  ============
    PSI range             Severity           UI colour
    ====================  ================  ============
    ``[0, 0.10)``         ``info``           green
    ``[0.10, 0.25)``      ``warn``           amber
    ``[0.25, ∞)``         ``critical``       red
    ``NaN / None / <0``   ``no_data``        grey
    ====================  ================  ============

    Non-numeric or otherwise unparseable inputs return ``no_data``
    rather than raising, so the classifier is safe to apply
    row-by-row across a DataFrame containing transient nulls.
    """
    if psi is None:
        return SEVERITY_NO_DATA
    try:
        v = float(psi)
    except (TypeError, ValueError):
        return SEVERITY_NO_DATA
    if not np.isfinite(v) or v < 0:
        return SEVERITY_NO_DATA
    if v < PSI_WARN_THRESHOLD:
        return SEVERITY_INFO
    if v < PSI_CRITICAL_THRESHOLD:
        return SEVERITY_WARN
    return SEVERITY_CRITICAL


# ═════════════════════════════════════════════════════════════════════
# Per-cluster drift table builder
# ═════════════════════════════════════════════════════════════════════

_DRIFT_TABLE_COLUMNS: Sequence[str] = (
    "cluster_id",
    "feature_name",
    "psi",
    "js_divergence",
    "mean_shift",
    "std_ratio",
    "severity",
)

_DRIFT_TABLE_DTYPES: Dict[str, Any] = {
    "cluster_id":    "string",
    "feature_name":  "string",
    "psi":           np.float32,
    "js_divergence": np.float32,
    "mean_shift":    np.float32,
    "std_ratio":     np.float32,
    "severity":      "string",
}


def build_drift_table(
    baseline_df:      pd.DataFrame,
    current_features: pd.DataFrame,
    *,
    cluster_id:       str,
    epsilon:          float = 1e-6,
) -> pd.DataFrame:
    """Compute drift for every feature in ``baseline_df`` vs ``current_features``.

    The baseline is the **anchor** — every feature it tracks gets a row
    in the output:

    * Features present in **both** baseline and current → PSI + JSD +
      mean_shift + std_ratio computed, severity classified.
    * Features in baseline but **missing / all-NaN** in current → row
      emitted with NaN metrics and ``severity = "no_data"`` so the UI
      heatmap shows a grey cell rather than collapsing the column.
    * Features in current but **not** in baseline → silently ignored
      (we cannot score drift without a baseline anchor).

    Output schema (long-format, one row per baseline feature):

    | column          | dtype     | meaning                              |
    |-----------------|-----------|--------------------------------------|
    | ``cluster_id``  | string    | owning cluster                       |
    | ``feature_name``| string    | feature column name                  |
    | ``psi``         | float32   | population stability index           |
    | ``js_divergence`` | float32 | Jensen-Shannon divergence ``[0, 1]`` |
    | ``mean_shift``  | float32   | ``(μ_curr - μ_base) / max(σ_base, ε)`` (z-score units) |
    | ``std_ratio``   | float32   | ``σ_curr / max(σ_base, ε)`` — vol regime indicator |
    | ``severity``    | string    | ``info``/``warn``/``critical``/``no_data`` |

    Parameters
    ----------
    baseline_df
        ``load_baseline``-decoded DataFrame.  Must have ``feature_name``,
        ``mean``, ``std``, ``hist_edges`` (ndarray) and ``hist_counts``
        (ndarray) columns.
    current_features
        Today's features in the **same coordinate system** the baseline
        was built in (i.e. already pushed through the frozen training
        scaler).  Rows = observations, cols = feature names.
    cluster_id
        Stamped on every output row so multi-cluster tables can be
        ``pd.concat``-ed freely.
    epsilon
        Numerical floor used by PSI / std-ratio to avoid div-by-zero
        on degenerate near-constant baselines.

    Returns
    -------
    pd.DataFrame
        Long-format drift table (one row per baseline feature).
    """
    if baseline_df is None or baseline_df.empty:
        return _empty_drift_table()

    required = {"feature_name", "mean", "std", "hist_edges", "hist_counts"}
    missing  = required - set(baseline_df.columns)
    if missing:
        raise ValueError(
            f"baseline_df is missing required columns: {sorted(missing)}.  "
            f"Did you forget to call monitoring.loaders.load_baseline?"
        )

    rows: List[Dict[str, Any]] = []
    for _, brow in baseline_df.iterrows():
        feature_name    = str(brow["feature_name"])
        baseline_edges  = np.asarray(brow["hist_edges"],  dtype=np.float64)
        baseline_counts = np.asarray(brow["hist_counts"], dtype=np.float64)
        baseline_mean   = float(brow["mean"]) if pd.notna(brow["mean"]) else np.nan
        baseline_std    = float(brow["std"])  if pd.notna(brow["std"])  else np.nan

        if feature_name not in current_features.columns:
            rows.append(_no_data_row(cluster_id, feature_name))
            continue

        current_raw   = current_features[feature_name].to_numpy(dtype=np.float64)
        current_valid = current_raw[np.isfinite(current_raw)]
        if current_valid.size == 0 or baseline_counts.size == 0:
            rows.append(_no_data_row(cluster_id, feature_name))
            continue

        psi = population_stability_index(
            baseline_counts, baseline_edges, current_valid, epsilon=epsilon,
        )

        # JSD needs paired histograms on the same edges — bin today's
        # values with the persisted baseline edges first.  We reuse
        # the same outlier-clipping convention as PSI so the two
        # numbers tell a consistent story.
        clipped = np.clip(current_valid, baseline_edges[0], baseline_edges[-1])
        current_counts, _ = np.histogram(clipped, bins=baseline_edges)
        jsd = js_divergence(baseline_counts, current_counts, epsilon=epsilon)

        # Mean shift in baseline-std units; std ratio in raw scale.
        # ``epsilon`` guards against degenerate near-constant baselines
        # (std ≈ 0) — the resulting numbers are noisy by definition in
        # that regime, but better than divide-by-zero.
        std_floor = (
            max(baseline_std, epsilon)
            if np.isfinite(baseline_std) and baseline_std > 0
            else epsilon
        )
        current_mean = float(np.mean(current_valid))
        current_std  = float(np.std(current_valid, ddof=0))
        mean_shift   = (
            (current_mean - baseline_mean) / std_floor
            if np.isfinite(baseline_mean) else np.nan
        )
        std_ratio = current_std / std_floor

        rows.append({
            "cluster_id":    cluster_id,
            "feature_name":  feature_name,
            "psi":           psi,
            "js_divergence": jsd,
            "mean_shift":    mean_shift,
            "std_ratio":     std_ratio,
            "severity":      classify_severity(psi),
        })

    return _to_drift_dataframe(rows)


def _empty_drift_table() -> pd.DataFrame:
    """Empty long-format drift table with the canonical dtype schema."""
    return _to_drift_dataframe([])


def _no_data_row(cluster_id: str, feature_name: str) -> Dict[str, Any]:
    """Drift-table row for a feature we cannot score (missing / all-NaN)."""
    return {
        "cluster_id":    cluster_id,
        "feature_name":  feature_name,
        "psi":           np.nan,
        "js_divergence": np.nan,
        "mean_shift":    np.nan,
        "std_ratio":     np.nan,
        "severity":      SEVERITY_NO_DATA,
    }


def _to_drift_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Coerce a row list to the canonical drift-table dtype schema.

    Centralised so empty + populated tables share the same dtype
    fingerprint — important for ``pd.concat`` to work without dtype
    promotion warnings when callers stitch per-cluster tables together.
    """
    df = pd.DataFrame(rows, columns=list(_DRIFT_TABLE_COLUMNS))
    return df.astype(_DRIFT_TABLE_DTYPES)


# ═════════════════════════════════════════════════════════════════════
# Portfolio-level summary
# ═════════════════════════════════════════════════════════════════════

def build_portfolio_drift_summary(
    drift_tables: Sequence[pd.DataFrame],
) -> Dict[str, Any]:
    """Aggregate per-cluster drift tables into portfolio-level KPIs.

    Consumed by the monitoring run manifest (``drift_summary.json``) and
    the Monitoring tab health-strip KPIs.  Severity is classified off
    ``mean_psi`` using the same thresholds applied to individual
    features, so the portfolio's "warn" line matches the per-feature
    severity boundary the user already sees in the heatmap.

    Returns a dict with:

    ===================  =================================================
    key                  meaning
    ===================  =================================================
    ``n_clusters``        clusters represented across the tables
    ``n_features_total``  scoreable features (excluding ``no_data`` rows)
    ``n_features_info``   features with severity ``info``
    ``n_features_warn``   features with severity ``warn``
    ``n_features_crit``   features with severity ``critical``
    ``n_features_nodata`` features that could not be scored
    ``mean_psi``          mean PSI across all scoreable features
    ``max_psi``           max PSI across all scoreable features
    ``worst_cluster``     ``cluster_id`` holding the max-PSI feature
    ``worst_feature``     ``feature_name`` holding the max PSI
    ``severity``          portfolio-level severity, classified off ``mean_psi``
    ===================  =================================================

    Tolerates empty / ``None`` inputs by returning an empty-state dict
    with zero counts and ``severity == "no_data"`` so callers don't have
    to gate on row counts.
    """
    if drift_tables is None or len(drift_tables) == 0:
        return _empty_summary()

    nonempty = [t for t in drift_tables if t is not None and not t.empty]
    if not nonempty:
        return _empty_summary()

    combined   = pd.concat(nonempty, ignore_index=True)
    n_clusters = int(combined["cluster_id"].nunique())

    sev_counts = combined["severity"].value_counts().to_dict()
    n_info   = int(sev_counts.get(SEVERITY_INFO,     0))
    n_warn   = int(sev_counts.get(SEVERITY_WARN,     0))
    n_crit   = int(sev_counts.get(SEVERITY_CRITICAL, 0))
    n_nodata = int(sev_counts.get(SEVERITY_NO_DATA,  0))
    n_total  = n_info + n_warn + n_crit

    scoreable = combined[combined["severity"] != SEVERITY_NO_DATA]
    if scoreable.empty:
        mean_psi:      float          = float("nan")
        max_psi:       float          = float("nan")
        worst_cluster: Optional[str]  = None
        worst_feature: Optional[str]  = None
    else:
        mean_psi      = float(scoreable["psi"].mean())
        idx_max       = scoreable["psi"].idxmax()
        max_psi       = float(scoreable.loc[idx_max, "psi"])
        worst_cluster = str(scoreable.loc[idx_max, "cluster_id"])
        worst_feature = str(scoreable.loc[idx_max, "feature_name"])

    return {
        "n_clusters":         n_clusters,
        "n_features_total":   n_total,
        "n_features_info":    n_info,
        "n_features_warn":    n_warn,
        "n_features_crit":    n_crit,
        "n_features_nodata":  n_nodata,
        "mean_psi":           mean_psi,
        "max_psi":            max_psi,
        "worst_cluster":      worst_cluster,
        "worst_feature":      worst_feature,
        "severity":           classify_severity(mean_psi),
    }


def _empty_summary() -> Dict[str, Any]:
    """Empty-state portfolio summary; keeps every key the manifest writer expects."""
    return {
        "n_clusters":         0,
        "n_features_total":   0,
        "n_features_info":    0,
        "n_features_warn":    0,
        "n_features_crit":    0,
        "n_features_nodata":  0,
        "mean_psi":           float("nan"),
        "max_psi":            float("nan"),
        "worst_cluster":      None,
        "worst_feature":      None,
        "severity":           SEVERITY_NO_DATA,
    }


__all__ = [
    # constants
    "SCHEMA_VERSION",
    "PSI_WARN_THRESHOLD", "PSI_CRITICAL_THRESHOLD",
    "SEVERITY_INFO", "SEVERITY_WARN", "SEVERITY_CRITICAL", "SEVERITY_NO_DATA",
    # primitives
    "population_stability_index",
    "js_divergence",
    "classify_severity",
    # builders
    "build_drift_table",
    "build_portfolio_drift_summary",
]
```

### File 2: `src/rade_ml_pt/monitoring/loaders.py` (full source)

```python
"""Loaders for monitoring baseline parquets.

Decode the JSON-encoded ``hist_edges_json`` / ``hist_counts_json``
columns produced by :func:`monitoring.baselines.save_feature_baseline`
into ready-to-use NumPy arrays so callers (drift computation, UI
distribution overlays) don't have to repeat the ``json.loads`` dance.

Kept deliberately small — this module is just the reader half of the
training/inference symmetry; the writer half lives in ``baselines.py``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that MUST be present in any baseline parquet we're prepared
# to consume.  Hoisted to module scope so the assertion + the error
# message stay in sync (one source of truth).
_REQUIRED_COLUMNS = (
    "feature_name",
    "mean",
    "std",
    "hist_edges_json",
    "hist_counts_json",
)


def load_baseline(parquet_path: Union[str, Path]) -> pd.DataFrame:
    """Read a ``baseline_feature_stats.parquet`` and decode its JSON columns.

    Parameters
    ----------
    parquet_path
        Path to the parquet file produced by
        :func:`monitoring.baselines.save_feature_baseline`.

    Returns
    -------
    pd.DataFrame
        The original parquet schema (``feature_name``, ``mean``, ``std``,
        ``min``, ``max``, ``p05``…``p95``, ``hist_edges_json``,
        ``hist_counts_json``) plus **two new in-memory columns**:

        * ``hist_edges``  — ``np.ndarray[float64]`` of length 51 (or
          empty when the feature was all-NaN at training time).
        * ``hist_counts`` — ``np.ndarray[int64]``  of length 50 (or
          empty in the same case).

        The two JSON-string columns are kept on the frame for
        traceability / debugging — drop them yourself if not needed
        downstream.

    Raises
    ------
    FileNotFoundError
        If ``parquet_path`` does not exist.
    ValueError
        If the parquet is missing the expected ``hist_edges_json`` /
        ``hist_counts_json`` (or any other required column).
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"baseline parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"baseline parquet at {parquet_path} is missing required columns "
            f"{sorted(missing)}.  Was it written by an older version of "
            f"monitoring.baselines.save_feature_baseline (or by something else "
            f"entirely)?"
        )

    df = df.copy()
    df["hist_edges"]  = df["hist_edges_json"].map(_decode_json_array_f64)
    df["hist_counts"] = df["hist_counts_json"].map(_decode_json_array_i64)
    return df


def _decode_json_array_f64(s: object) -> np.ndarray:
    """JSON-string-of-floats → ``np.ndarray[float64]`` (empty on error/empty).

    Defensive: tolerates ``None`` / ``NaN`` / malformed JSON by
    returning an empty array, so a single corrupt row in the baseline
    parquet can't poison an entire run.  Pair with the ``hist_edges``
    length checks downstream in drift code which already treat
    ``size == 0`` as "no_data".
    """
    if s is None:
        return np.empty(0, dtype=np.float64)
    if isinstance(s, float) and np.isnan(s):
        return np.empty(0, dtype=np.float64)
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return np.empty(0, dtype=np.float64)
    return np.asarray(parsed, dtype=np.float64)


def _decode_json_array_i64(s: object) -> np.ndarray:
    """JSON-string-of-ints → ``np.ndarray[int64]`` (empty on error/empty)."""
    if s is None:
        return np.empty(0, dtype=np.int64)
    if isinstance(s, float) and np.isnan(s):
        return np.empty(0, dtype=np.int64)
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return np.empty(0, dtype=np.int64)
    return np.asarray(parsed, dtype=np.int64)


__all__ = ["load_baseline"]
```

### File 3: `src/rade_ml_pt/monitoring/__init__.py` (replace existing)

```python
"""Drift monitoring artifacts for the PRISM Model Monitoring surface.

This package houses:

* :mod:`monitoring.baselines` — training-time histogram + summary
  statistics writer (``save_feature_baseline``).
* :mod:`monitoring.loaders`   — reader that decodes baseline parquets
  back into NumPy-ready DataFrames (``load_baseline``).
* :mod:`monitoring.drift`     — pure-NumPy drift primitives + per-cluster
  / portfolio aggregators (PSI, JSD, severity classifier).

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

__all__ = [
    # constants
    "PSI_WARN_THRESHOLD", "PSI_CRITICAL_THRESHOLD",
    "SEVERITY_INFO", "SEVERITY_WARN", "SEVERITY_CRITICAL", "SEVERITY_NO_DATA",
    # primitives
    "population_stability_index",
    "js_divergence",
    "classify_severity",
    # builders
    "build_drift_table",
    "build_portfolio_drift_summary",
    # IO
    "load_baseline",
]
```

### File 4: `tests/rade_ml_pt/monitoring/__init__.py` (empty file, but create it)

```python
```

> Empty `__init__.py` — needed only so pytest discovers the test
> package under the existing `tests/rade_ml_pt/...` layout.

### File 5: `tests/rade_ml_pt/monitoring/test_drift.py` (full source)

```python
"""Unit tests for ``rade_ml_pt.monitoring.drift``.

Three rings of coverage:
* Single-feature primitives (PSI, JSD, severity classifier).
* Per-cluster ``build_drift_table`` shape + edge cases.
* Portfolio summary aggregator.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from src.rade_ml_pt.monitoring.drift import (
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


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts


def _baseline_row(name: str, values: np.ndarray, n_bins: int = 50) -> dict:
    """Build a single baseline-DataFrame row in the same shape ``load_baseline`` produces."""
    edges  = np.linspace(values.min(), values.max(), n_bins + 1)
    counts = _hist(values, edges)
    return {
        "feature_name": name,
        "mean":         float(values.mean()),
        "std":          float(values.std(ddof=0)),
        "hist_edges":   edges,
        "hist_counts":  counts,
    }


# ═════════════════════════════════════════════════════════════════════
# population_stability_index
# ═════════════════════════════════════════════════════════════════════

class TestPSI:
    def test_identical_distributions_returns_near_zero(self):
        rng    = np.random.default_rng(0)
        x      = rng.normal(0, 1, 5000)
        edges  = np.linspace(-4, 4, 51)
        counts = _hist(x, edges)
        psi    = population_stability_index(counts, edges, x)
        assert psi < 0.01

    def test_disjoint_distributions_returns_large(self):
        rng         = np.random.default_rng(1)
        base        = rng.normal(-3, 0.5, 5000)
        cur         = rng.normal(+3, 0.5, 5000)
        edges       = np.linspace(-5, 5, 51)
        base_counts = _hist(base, edges)
        psi         = population_stability_index(base_counts, edges, cur)
        assert psi > 1.0

    def test_one_sigma_shift_lands_in_warn_or_critical(self):
        rng         = np.random.default_rng(2)
        base        = rng.normal(0, 1, 5000)
        cur         = rng.normal(1, 1, 5000)
        edges       = np.linspace(-4, 5, 51)
        base_counts = _hist(base, edges)
        psi         = population_stability_index(base_counts, edges, cur)
        assert psi > PSI_WARN_THRESHOLD
        assert psi < 2.0  # sanity ceiling for 1-sigma shift

    def test_empty_current_returns_nan(self):
        edges  = np.linspace(-1, 1, 51)
        counts = np.ones(50, dtype=np.int64)
        assert np.isnan(population_stability_index(counts, edges, np.array([])))

    def test_zero_baseline_counts_returns_nan(self):
        edges  = np.linspace(-1, 1, 51)
        counts = np.zeros(50, dtype=np.int64)
        assert np.isnan(population_stability_index(counts, edges, np.array([0.0, 0.1])))

    def test_mismatched_shapes_returns_nan(self):
        # 50-element counts paired with 50-element edges (should be 51)
        edges  = np.linspace(-1, 1, 50)
        counts = np.ones(50, dtype=np.int64)
        assert np.isnan(population_stability_index(counts, edges, np.array([0.0])))

    def test_outlier_current_values_clipped_not_dropped(self):
        rng         = np.random.default_rng(3)
        base        = rng.normal(0, 1, 5000)
        edges       = np.linspace(-3, 3, 51)
        base_counts = _hist(base, edges)
        # Current values entirely outside the baseline range — naive
        # np.histogram would silently drop them.  The clip step lumps
        # them into the outermost bin, so PSI still reflects "today
        # looks nothing like training".
        cur = rng.normal(10, 1, 5000)
        psi = population_stability_index(base_counts, edges, cur)
        assert np.isfinite(psi)
        assert psi > 1.0

    def test_non_finite_current_values_dropped(self):
        rng         = np.random.default_rng(4)
        base        = rng.normal(0, 1, 5000)
        edges       = np.linspace(-4, 4, 51)
        base_counts = _hist(base, edges)
        # Half NaN, half OK; should still return a finite (small) PSI
        cur = np.concatenate([rng.normal(0, 1, 2500), np.full(2500, np.nan)])
        psi = population_stability_index(base_counts, edges, cur)
        assert np.isfinite(psi)
        assert psi < 0.05


# ═════════════════════════════════════════════════════════════════════
# js_divergence
# ═════════════════════════════════════════════════════════════════════

class TestJSD:
    def test_identical_returns_near_zero(self):
        rng    = np.random.default_rng(5)
        edges  = np.linspace(-3, 3, 51)
        counts = _hist(rng.normal(0, 1, 5000), edges)
        assert js_divergence(counts, counts) < 1e-3

    def test_symmetric(self):
        rng   = np.random.default_rng(6)
        edges = np.linspace(-3, 3, 51)
        a     = _hist(rng.normal(0, 1, 5000), edges)
        b     = _hist(rng.normal(1, 1, 5000), edges)
        assert js_divergence(a, b) == pytest.approx(js_divergence(b, a), abs=1e-9)

    def test_bounded_in_unit_interval(self):
        rng   = np.random.default_rng(7)
        edges = np.linspace(-3, 3, 51)
        a     = _hist(rng.normal(-2, 0.5, 5000), edges)
        b     = _hist(rng.normal(+2, 0.5, 5000), edges)
        jsd   = js_divergence(a, b)
        assert 0.0 <= jsd <= 1.0

    def test_shape_mismatch_returns_nan(self):
        assert np.isnan(js_divergence(np.ones(10), np.ones(20)))

    def test_zero_inputs_return_nan(self):
        assert np.isnan(js_divergence(np.zeros(10), np.ones(10)))
        assert np.isnan(js_divergence(np.ones(10), np.zeros(10)))


# ═════════════════════════════════════════════════════════════════════
# classify_severity
# ═════════════════════════════════════════════════════════════════════

class TestSeverity:
    @pytest.mark.parametrize("psi,expected", [
        (0.0,                                 SEVERITY_INFO),
        (0.09,                                SEVERITY_INFO),
        (PSI_WARN_THRESHOLD - 1e-9,           SEVERITY_INFO),
        (PSI_WARN_THRESHOLD,                  SEVERITY_WARN),
        (0.20,                                SEVERITY_WARN),
        (PSI_CRITICAL_THRESHOLD - 1e-9,       SEVERITY_WARN),
        (PSI_CRITICAL_THRESHOLD,              SEVERITY_CRITICAL),
        (0.50,                                SEVERITY_CRITICAL),
        (10.0,                                SEVERITY_CRITICAL),
    ])
    def test_thresholds(self, psi, expected):
        assert classify_severity(psi) == expected

    @pytest.mark.parametrize("v", [
        None,
        float("nan"),
        -0.1,
        float("inf"),
        -float("inf"),
    ])
    def test_no_data_cases(self, v):
        assert classify_severity(v) == SEVERITY_NO_DATA

    def test_non_numeric_returns_no_data(self):
        assert classify_severity("not a number") == SEVERITY_NO_DATA  # type: ignore[arg-type]


# ═════════════════════════════════════════════════════════════════════
# build_drift_table
# ═════════════════════════════════════════════════════════════════════

class TestBuildDriftTable:
    @staticmethod
    def _baseline_df(rng: np.random.Generator) -> pd.DataFrame:
        rows = [
            _baseline_row("rf_a", rng.normal(0, 1, 5000)),
            _baseline_row("rf_b", rng.normal(0, 1, 5000)),
        ]
        return pd.DataFrame(rows)

    def test_shape_and_columns(self):
        rng         = np.random.default_rng(8)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(0, 1, 1000),
        })
        out = build_drift_table(baseline_df, cur, cluster_id="c0")
        assert list(out.columns) == [
            "cluster_id", "feature_name", "psi", "js_divergence",
            "mean_shift", "std_ratio", "severity",
        ]
        assert len(out) == 2
        assert set(out["feature_name"]) == {"rf_a", "rf_b"}
        assert (out["cluster_id"] == "c0").all()

    def test_stable_features_classified_info(self):
        rng         = np.random.default_rng(9)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 5000),
            "rf_b": rng.normal(0, 1, 5000),
        })
        out = build_drift_table(baseline_df, cur, cluster_id="c0")
        assert (out["severity"] == SEVERITY_INFO).all()
        assert (out["psi"] < PSI_WARN_THRESHOLD).all()

    def test_shifted_features_classified_critical(self):
        rng         = np.random.default_rng(10)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(3, 1, 5000),
            "rf_b": rng.normal(3, 1, 5000),
        })
        out = build_drift_table(baseline_df, cur, cluster_id="c0")
        assert (out["severity"] == SEVERITY_CRITICAL).all()
        assert (out["mean_shift"] > 1.0).all()

    def test_missing_feature_in_current_emits_no_data_row(self):
        rng         = np.random.default_rng(11)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({"rf_a": rng.normal(0, 1, 1000)})  # rf_b missing
        out         = build_drift_table(baseline_df, cur, cluster_id="c0")
        b_row       = out.set_index("feature_name").loc["rf_b"]
        assert b_row["severity"] == SEVERITY_NO_DATA
        assert pd.isna(b_row["psi"])

    def test_all_nan_feature_in_current_emits_no_data_row(self):
        rng         = np.random.default_rng(12)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": np.full(1000, np.nan),
        })
        out   = build_drift_table(baseline_df, cur, cluster_id="c0")
        b_row = out.set_index("feature_name").loc["rf_b"]
        assert b_row["severity"] == SEVERITY_NO_DATA

    def test_extra_feature_in_current_ignored(self):
        rng         = np.random.default_rng(13)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(0, 1, 1000),
            "rf_z": rng.normal(0, 1, 1000),   # not in baseline
        })
        out = build_drift_table(baseline_df, cur, cluster_id="c0")
        assert "rf_z" not in out["feature_name"].values

    def test_empty_baseline_returns_empty_table_with_dtype_schema(self):
        out = build_drift_table(pd.DataFrame(), pd.DataFrame(), cluster_id="c0")
        assert list(out.columns) == [
            "cluster_id", "feature_name", "psi", "js_divergence",
            "mean_shift", "std_ratio", "severity",
        ]
        assert len(out) == 0

    def test_baseline_missing_required_columns_raises(self):
        bad = pd.DataFrame([{"feature_name": "x"}])
        with pytest.raises(ValueError, match="missing required columns"):
            build_drift_table(bad, pd.DataFrame({"x": [1, 2, 3]}), cluster_id="c0")

    def test_dtypes_canonical_for_concat(self):
        """Empty + populated tables share a dtype fingerprint."""
        empty = build_drift_table(pd.DataFrame(), pd.DataFrame(), cluster_id="c0")

        rng         = np.random.default_rng(14)
        baseline_df = self._baseline_df(rng)
        cur         = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(0, 1, 1000),
        })
        populated = build_drift_table(baseline_df, cur, cluster_id="c1")
        assert empty.dtypes.to_dict() == populated.dtypes.to_dict()

        # pd.concat must work without dtype-promotion warnings
        combined = pd.concat([empty, populated], ignore_index=True)
        assert len(combined) == 2


# ═════════════════════════════════════════════════════════════════════
# build_portfolio_drift_summary
# ═════════════════════════════════════════════════════════════════════

class TestPortfolioSummary:
    @staticmethod
    def _drift(cid: str, severities: List[str], psis: List[float]) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "cluster_id":    cid,
                "feature_name":  f"rf_{i}",
                "psi":           p,
                "js_divergence": 0.0,
                "mean_shift":    0.0,
                "std_ratio":     1.0,
                "severity":      s,
            }
            for i, (s, p) in enumerate(zip(severities, psis))
        ])

    def test_aggregates_across_clusters(self):
        t1  = self._drift("c0", [SEVERITY_INFO, SEVERITY_WARN], [0.02, 0.18])
        t2  = self._drift("c1", [SEVERITY_CRITICAL], [0.40])
        out = build_portfolio_drift_summary([t1, t2])
        assert out["n_clusters"]         == 2
        assert out["n_features_info"]    == 1
        assert out["n_features_warn"]    == 1
        assert out["n_features_crit"]    == 1
        assert out["max_psi"]            == pytest.approx(0.40, abs=1e-6)
        assert out["worst_cluster"]      == "c1"
        assert out["worst_feature"]      == "rf_0"

    def test_empty_input_returns_no_data(self):
        out = build_portfolio_drift_summary([])
        assert out["n_clusters"]   == 0
        assert out["severity"]     == SEVERITY_NO_DATA
        assert np.isnan(out["mean_psi"])

    def test_none_input_returns_no_data(self):
        out = build_portfolio_drift_summary(None)  # type: ignore[arg-type]
        assert out["severity"] == SEVERITY_NO_DATA

    def test_only_no_data_rows_returns_no_data(self):
        t   = self._drift("c0", [SEVERITY_NO_DATA, SEVERITY_NO_DATA], [np.nan, np.nan])
        out = build_portfolio_drift_summary([t])
        assert out["severity"]          == SEVERITY_NO_DATA
        assert out["n_features_nodata"] == 2
        assert out["worst_cluster"]     is None

    def test_severity_classification_from_mean_psi(self):
        # mean_psi == 0.15 → portfolio severity "warn"
        t   = self._drift("c0", [SEVERITY_WARN], [0.15])
        out = build_portfolio_drift_summary([t])
        assert out["severity"] == SEVERITY_WARN

    def test_skips_empty_drift_tables_silently(self):
        t1  = self._drift("c0", [SEVERITY_INFO], [0.05])
        t2  = pd.DataFrame()  # empty
        out = build_portfolio_drift_summary([t1, t2])
        assert out["n_clusters"] == 1


# ═════════════════════════════════════════════════════════════════════
# End-to-end smoke — writer (baselines.py) → loader → drift table
# ═════════════════════════════════════════════════════════════════════
# Lives here (rather than test_loaders.py) because the assertions are
# about drift outputs, not loader I/O.  Confirms the writer/reader/
# primitive trio compose correctly with zero adapter code in between.

class TestEndToEndSmoke:
    """Full chain: write a real baseline parquet, load it back, score drift."""

    def _write_and_load_baseline(self, tmp_path, features: pd.DataFrame) -> pd.DataFrame:
        # Local import so this module imports fine even when the
        # baselines writer is unavailable (e.g. running drift tests in
        # isolation in CI).
        from src.rade_ml_pt.monitoring.baselines import save_feature_baseline
        from src.rade_ml_pt.monitoring.loaders   import load_baseline

        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")
        return load_baseline(path)

    def test_stable_features_score_info(self, tmp_path):
        rng           = np.random.default_rng(20)
        training      = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 5000),
            "rf_b": rng.normal(5, 2, 5000),
        })
        baseline_df = self._write_and_load_baseline(tmp_path, training)
        current     = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(5, 2, 1000),
        })
        out = build_drift_table(baseline_df, current, cluster_id="c0")
        assert (out["severity"] == SEVERITY_INFO).all()

    def test_drift_pipeline_detects_real_shift(self, tmp_path):
        rng         = np.random.default_rng(21)
        training    = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 5000),
            "rf_b": rng.normal(0, 1, 5000),
        })
        baseline_df = self._write_and_load_baseline(tmp_path, training)
        current     = pd.DataFrame({
            "rf_a": rng.normal(2, 1, 1000),   # +2σ shift
            "rf_b": rng.normal(0, 1, 1000),
        })
        out         = build_drift_table(baseline_df, current, cluster_id="c0")
        by_feature  = out.set_index("feature_name")
        assert by_feature.loc["rf_a", "severity"] in (SEVERITY_WARN, SEVERITY_CRITICAL)
        assert by_feature.loc["rf_b", "severity"] == SEVERITY_INFO

    def test_portfolio_summary_compiles_from_real_baseline(self, tmp_path):
        rng       = np.random.default_rng(22)
        training  = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 5000),
            "rf_b": rng.normal(0, 1, 5000),
        })
        baseline_df = self._write_and_load_baseline(tmp_path, training)
        cur_stable  = pd.DataFrame({
            "rf_a": rng.normal(0, 1, 1000),
            "rf_b": rng.normal(0, 1, 1000),
        })
        cur_shifted = pd.DataFrame({
            "rf_a": rng.normal(3, 1, 1000),
            "rf_b": rng.normal(3, 1, 1000),
        })
        # Pretend we have two clusters — same baseline shape, different
        # current data, so the portfolio mixes a clean cluster with a
        # drifted one.
        t_stable  = build_drift_table(baseline_df, cur_stable,  cluster_id="c_stable")
        t_drifted = build_drift_table(baseline_df, cur_shifted, cluster_id="c_drifted")

        summary = build_portfolio_drift_summary([t_stable, t_drifted])
        assert summary["n_clusters"]      == 2
        assert summary["worst_cluster"]   == "c_drifted"
        assert summary["severity"]        in (SEVERITY_WARN, SEVERITY_CRITICAL)
        assert summary["n_features_crit"] >= 1
```

### File 6: `tests/rade_ml_pt/monitoring/test_loaders.py` (full source)

```python
"""Unit tests for ``rade_ml_pt.monitoring.loaders``.

Round-trip ``save_feature_baseline`` → ``load_baseline`` to confirm the
JSON-encoded columns are decoded correctly into NumPy arrays — this is
the exact contract every drift consumer relies on, so we exercise the
edge cases the writer documents (all-NaN feature, constant feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rade_ml_pt.monitoring.baselines import save_feature_baseline
from src.rade_ml_pt.monitoring.loaders import load_baseline


@pytest.fixture
def features() -> pd.DataFrame:
    """Four columns that exercise every code path the writer cares about."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "rf_a":        rng.normal(0, 1, 1000),
        "rf_b":        rng.normal(5, 2, 1000),
        "rf_constant": np.full(1000, 3.14),
        "rf_all_nan":  np.full(1000, np.nan),
    })


class TestLoadBaseline:
    def test_roundtrip_preserves_stats(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")

        df    = load_baseline(path)
        rf_a  = df.set_index("feature_name").loc["rf_a"]
        assert rf_a["mean"] == pytest.approx(features["rf_a"].mean(),       rel=1e-4)
        assert rf_a["std"]  == pytest.approx(features["rf_a"].std(ddof=0), rel=1e-4)

    def test_histogram_columns_decoded_to_ndarrays(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")

        df = load_baseline(path)
        for _, row in df.iterrows():
            assert isinstance(row["hist_edges"],  np.ndarray)
            assert isinstance(row["hist_counts"], np.ndarray)

        # rf_a has data → 51 edges + 50 counts (per writer's N_HIST_BINS=50)
        rf_a = df.set_index("feature_name").loc["rf_a"]
        assert rf_a["hist_edges"].shape  == (51,)
        assert rf_a["hist_counts"].shape == (50,)

    def test_all_nan_feature_yields_empty_arrays(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")

        df     = load_baseline(path)
        rf_nan = df.set_index("feature_name").loc["rf_all_nan"]
        assert rf_nan["hist_edges"].shape  == (0,)
        assert rf_nan["hist_counts"].shape == (0,)

    def test_constant_feature_has_well_defined_edges(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")

        df       = load_baseline(path)
        rf_const = df.set_index("feature_name").loc["rf_constant"]
        assert rf_const["hist_edges"].shape  == (51,)
        assert rf_const["hist_counts"].shape == (50,)
        # save_feature_baseline widens the range with a tiny epsilon so
        # np.histogram does not error — confirm the trick worked.
        assert rf_const["hist_edges"][-1] > rf_const["hist_edges"][0]

    def test_original_columns_retained(self, features, tmp_path):
        path = tmp_path / "baseline.parquet"
        save_feature_baseline(path, features, cluster_id="c0")
        df = load_baseline(path)
        # JSON columns stay on the frame for traceability
        assert "hist_edges_json"  in df.columns
        assert "hist_counts_json" in df.columns

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline(tmp_path / "does_not_exist.parquet")

    def test_malformed_parquet_raises_value_error(self, tmp_path):
        path = tmp_path / "wrong.parquet"
        pd.DataFrame({"foo": [1, 2, 3]}).to_parquet(path)
        with pytest.raises(ValueError, match="missing required columns"):
            load_baseline(path)
```

### Verification (after copy-pasting into work env)

Run from repo root:

```bash
python -m pytest tests/rade_ml_pt/monitoring/ -x -q
```

Expected: `53 passed in ~5s`, lint-clean.

Quick sanity smoke for the public surface:

```bash
python -c "
from src.rade_ml_pt.monitoring import (
    population_stability_index, js_divergence, classify_severity,
    build_drift_table, build_portfolio_drift_summary, load_baseline,
    PSI_WARN_THRESHOLD, PSI_CRITICAL_THRESHOLD,
    SEVERITY_INFO, SEVERITY_WARN, SEVERITY_CRITICAL, SEVERITY_NO_DATA,
)
print('OK — monitoring public surface intact')
"
```

### What M.2 will consume

M.2 (`EnsembleMonitoringPipeline`) imports only from this surface:

```python
from src.rade_ml_pt.monitoring import (
    load_baseline,
    build_drift_table,
    build_portfolio_drift_summary,
    SEVERITY_CRITICAL, SEVERITY_WARN,
)
```

No other monitoring module is needed at the M.2 boundary — keep this
surface stable across the M.2 / M.3 refactor.
