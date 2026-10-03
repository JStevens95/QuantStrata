"""Plotly figure builders for the Inference Console page.

Stage 13: real builders dispatched by the three segmented controls
on the page.  Each top-level public dispatcher
(:func:`build_chart_main`, :func:`build_risk_attribution_chart`,
:func:`build_stress_tails_chart`) accepts the relevant DataFrame
plus the current mode flag and returns a ready-to-render
``plotly.graph_objects.Figure``.

Modes
-----

``chart_main`` — set via ``INFERENCE_IDS['chart_view_mode']``::

    distribution  → _pnl_distribution(portfolio_df)
    timeseries    → _pnl_timeseries(portfolio_df)
    overlay       → empty_pnl_overlay()           (reserved)

``risk_attribution_chart`` — set via
``INFERENCE_IDS['risk_attribution_breakdown']``::

    cluster       → _attribution_by_cluster(clusters_df)
    risk_factor   → _attribution_by_risk_factor(...) (Phase 1a.2:
                    Approach B — cluster-share split across the
                    cluster's intersecting RFs)
    trade_type    → _attribution_unavailable(...) (v1: no metadata)

``stress_tails_chart`` — set via
``INFERENCE_IDS['stress_tails_mode']``::

    fan           → _stress_fan(portfolio_df)
    tail          → _stress_tail(portfolio_df)
    worst         → _stress_worst(portfolio_df, n=20)

Empty-state builders (the ``empty_*`` functions) are preserved as a
public API because the callback's ``empty_returns`` tuple uses them
to paint the page before any run completes.  The dispatchers fall
back to the matching empty builder when the input frame is empty or
missing the expected column, so callbacks never have to branch on
"have data yet" — they can call ``build_*`` unconditionally.

Style
-----

Every figure goes through :func:`figures._theme.rade_layout` so the
violet → cyan / amber / rose palette and the transparent
plot_bgcolor stay consistent with the rest of the dashboard.  No
chart manages its own font / margin / colour scheme — all of that
lives in ``_theme.py``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._theme import (
    empty_figure,
    rade_layout,
    rgba,
    sort_chronologically,
)


# ─────────────────────────────────────────────────────────────────────
# Palette aliases (pinned here so chart code reads naturally; the
# canonical hex values live in ``_theme.CATEGORY_PALETTE``).
# ─────────────────────────────────────────────────────────────────────

_VIOLET:    str = "#8b5cf6"   # primary series colour
_EMERALD:   str = "#10b981"   # positive contribution
_AMBER:     str = "#f59e0b"   # VaR marker
_ROSE:      str = "#f43f5e"   # tail / CVaR / negative contribution
_ZERO_LINE: str = "#475569"   # neutral grid reference

# 95 % VaR convention — the 5th percentile of the PnL distribution
# (lower tail).  Used as a literal in builder code as well so the
# annotation label stays in sync.
_VAR_QUANTILE:    float = 0.05
_WORST_N_DEFAULT: int   = 20


# ═════════════════════════════════════════════════════════════════════
# Empty-state builders — kept as public API
# ═════════════════════════════════════════════════════════════════════

def empty_pnl_distribution() -> go.Figure:
    """Placeholder for *Predicted PnL distribution across scenarios*."""
    return empty_figure("Awaiting scenario run — distribution view")


def empty_pnl_timeseries() -> go.Figure:
    """Placeholder for *PnL / exposure vs scenario order or time*."""
    return empty_figure("Awaiting scenario run — timeseries view")


def empty_pnl_overlay() -> go.Figure:
    """Reserved third chart mode (e.g. density overlay, fan chart)."""
    return empty_figure("Third chart mode — reserved")


def empty_risk_attribution() -> go.Figure:
    """Placeholder for *P&L attribution by cluster · risk-factor · trade-type*."""
    return empty_figure("Awaiting scenario run — attribution view")


def empty_stress_tails() -> go.Figure:
    """Placeholder for *VaR / CVaR / tail-quantile* view."""
    return empty_figure("Awaiting scenario run — tail-risk view")


def empty_coverage_donut() -> go.Figure:
    """Placeholder for the Diagnostics tab's *routing-coverage* donut."""
    return empty_figure("Awaiting validation — coverage donut")


def empty_sensitivity_heatmap() -> go.Figure:
    """Placeholder for the Sensitivity tab's cluster × scenario heatmap."""
    return empty_figure("Awaiting scenario run — sensitivity heatmap")


def empty_sensitivity_tornado() -> go.Figure:
    """Placeholder for the Sensitivity tab's per-cluster volatility tornado."""
    return empty_figure("Awaiting scenario run — sensitivity tornado")


# ═════════════════════════════════════════════════════════════════════
# Sensitivity tab (Phase 1a.1) — cluster × scenario heatmap + tornado
# ═════════════════════════════════════════════════════════════════════

# Diverging colorscale used by the sensitivity heatmap.  Rose-700 at
# the most-negative end, slate-800 in the neutral middle, emerald-600
# at the most-positive end — same hue family as the bar charts so a
# user reading the heatmap and a sign-coloured tornado side-by-side
# gets a consistent mental colour mapping.
_DIVERGING_COLORSCALE: list[list] = [
    [0.0,  "#9f1239"],   # rose-800  — strong negative
    [0.25, "#f43f5e"],   # rose-500
    [0.5,  "#1e293b"],   # slate-800 — neutral
    [0.75, "#10b981"],   # emerald-500
    [1.0,  "#047857"],   # emerald-700 — strong positive
]


def build_sensitivity_heatmap(
    clusters_df: Optional[pd.DataFrame],
) -> go.Figure:
    """Cluster × Scenario heatmap of ``sum_pnl_original``.

    Pivots the long-format cluster summary into a 2-D grid (clusters
    on the y-axis, scenarios on the x-axis) and colours each cell by
    its signed PnL using a symmetric diverging palette.  Lets the
    eye spot:

    * **Vertical streaks** — a single scenario that moved many
      clusters in the same direction.  Indicates a market-wide
      shock dominating the run.
    * **Horizontal streaks** — a single cluster that moved
      consistently across most scenarios.  Indicates a structural
      bias rather than scenario-driven risk.
    * **Empty cells** — cluster × scenario combinations the data
      plane didn't carry (rendered transparent / faint).

    Sorting:

    * Clusters sorted by net signed PnL (most-positive at top,
      most-negative at bottom) so the visual gradient runs
      top-to-bottom.
    * Scenarios sorted chronologically when the labels parse as
      dates, otherwise alphabetically.

    Symmetric z-range (``zmin = -max|z|``, ``zmax = +max|z|``)
    ensures the neutral colour always sits at zero — without this
    the palette skews towards the sign of the larger tail.
    """
    if clusters_df is None or clusters_df.empty:
        return empty_sensitivity_heatmap()
    needed = {"cluster_id", "scenario_label", "sum_pnl_original"}
    if not needed.issubset(clusters_df.columns):
        return empty_sensitivity_heatmap()

    # Cluster order: signed sum ascending → most-negative at top of
    # the y-axis array.  Plotly's heatmap places y[0] at the bottom,
    # so this is the right order for "most negative at the bottom" UX.
    cluster_order = (
        clusters_df.groupby("cluster_id")["sum_pnl_original"]
        .sum()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # Scenario order: chronological if the labels parse as dates.
    # We use ``format="ISO8601"`` so we don't pay the dateutil
    # warning-and-fallback path when labels are non-date strings.
    scenario_labels = clusters_df["scenario_label"].astype(str).unique().tolist()
    parsed_dates    = pd.to_datetime(
        scenario_labels, errors="coerce", format="ISO8601",
    )
    if parsed_dates.notna().any():
        scenario_order = [
            label for label, _ in sorted(
                zip(scenario_labels, parsed_dates),
                key=lambda pair: (pd.isna(pair[1]), pair[1]),
            )
        ]
    else:
        scenario_order = sorted(scenario_labels)

    matrix = (
        clusters_df.pivot_table(
            index   = "cluster_id",
            columns = "scenario_label",
            values  = "sum_pnl_original",
            aggfunc = "sum",
        )
        .reindex(index=cluster_order, columns=scenario_order)
    )

    # Symmetric range around zero so the neutral colour lands on
    # PnL == 0 regardless of which tail is fatter.  A small epsilon
    # guards against an all-zero matrix collapsing the colorscale.
    z_abs_max = float(np.nanmax(np.abs(matrix.to_numpy()))) if matrix.size else 0.0
    z_abs_max = max(z_abs_max, 1.0)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z              = matrix.to_numpy(),
        x              = matrix.columns.tolist(),
        y              = matrix.index.tolist(),
        zmid           = 0.0,
        zmin           = -z_abs_max,
        zmax           = +z_abs_max,
        colorscale     = _DIVERGING_COLORSCALE,
        hoverongaps    = False,
        colorbar       = {
            "title":     {"text": "PnL", "font": {"color": "#94a3b8", "size": 10}},
            "tickfont":  {"color": "#94a3b8", "size": 10},
            "thickness": 10,
            "len":       0.85,
            "outlinewidth": 0,
        },
        hovertemplate  = (
            "Cluster: %{y}<br>"
            "Scenario: %{x}<br>"
            "PnL: %{z:,.2f}"
            "<extra></extra>"
        ),
    ))

    # Tame the x-axis tick density on wide runs; Plotly's auto
    # behaviour ends up overlapping labels for >~30 scenarios.
    n_scenarios = matrix.shape[1]
    n_clusters  = matrix.shape[0]
    show_x_ticks = n_scenarios <= 30
    fig.update_layout(**rade_layout(
        hovermode = "closest",
        margin    = {"l": 120, "r": 24, "t": 8, "b": 56 if show_x_ticks else 32},
        xaxis     = {
            "title":          "Scenario",
            "showticklabels": show_x_ticks,
            "tickangle":      -45,
            "automargin":     True,
        },
        yaxis     = {
            "title":     "",
            "automargin": True,
            "gridcolor": "rgba(0,0,0,0)",  # heatmap has its own grid; hide axis lines
        },
    ))
    # Annotate the cell count in the corner so the user has a sense
    # of the grid density without having to count axis ticks.
    fig.add_annotation(
        text       = f"{n_clusters} clusters × {n_scenarios} scenarios",
        xref       = "paper", yref = "paper",
        x          = 1.0, y = 1.02,
        xanchor    = "right", yanchor = "bottom",
        showarrow  = False,
        font       = {"color": "#64748b", "size": 10},
    )
    return fig


def build_sensitivity_tornado(
    clusters_df: Optional[pd.DataFrame],
    *,
    top_n:       int = 15,
) -> go.Figure:
    """Per-cluster sensitivity tornado.

    Sensitivity is approximated as the cross-scenario **standard
    deviation** of ``sum_pnl_original`` for each cluster — i.e.
    *how much* the cluster's PnL varies across scenarios, regardless
    of direction.  A cluster with low std is "insensitive" (mostly
    the same value across scenarios); high std = "highly sensitive".

    Bars are violet (high contrast, no sign semantics — every bar is
    inherently non-negative) and sorted ascending so the most
    sensitive cluster sits at the top of the chart.

    Phase 4 swaps in true RF-level elasticities (∂PnL/∂shock per RF)
    once the trade-attribute API ships and we can attribute PnL to
    individual RFs.
    """
    if clusters_df is None or clusters_df.empty:
        return empty_sensitivity_tornado()
    needed = {"cluster_id", "sum_pnl_original"}
    if not needed.issubset(clusters_df.columns):
        return empty_sensitivity_tornado()

    # ``std`` defaults to ddof=1; with a single observation per
    # cluster (n_scenarios == 1) this yields NaN.  We coerce to 0
    # so the cluster still renders rather than being silently
    # dropped from the chart.
    std_series = (
        clusters_df.groupby("cluster_id")["sum_pnl_original"]
        .std()
        .fillna(0.0)
        .astype(float)
    )

    ranked = std_series.sort_values(ascending=False)
    top    = ranked.head(top_n)
    rest   = ranked.iloc[top_n:]

    rows = list(zip(top.index.astype(str), top.values.astype(float)))
    if not rest.empty:
        rows.append((f"Other ({len(rest)})", float(rest.mean())))

    # Reverse for horizontal-bar bottom-up ordering — top of chart
    # gets the biggest value.
    rows = list(reversed(rows))
    y_labels = [str(label) for label, _ in rows]
    x_values = [float(value) for _, value in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x             = x_values,
        y             = y_labels,
        orientation   = "h",
        marker        = {"color": _VIOLET, "opacity": 0.9},
        text          = [f"{v:,.0f}" for v in x_values],
        textposition  = "outside",
        textfont      = {"size": 10, "color": "#cbd5e1"},
        hovertemplate = (
            "%{y}<br>"
            "Sensitivity (σ across scenarios): %{x:,.2f}"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=0,
        line={"color": _ZERO_LINE, "width": 1, "dash": "dot"},
    )
    fig.update_layout(**rade_layout(
        hovermode = "y",
        margin    = {"l": 140, "r": 56, "t": 8, "b": 36},
        xaxis     = {
            "title": "σ of sum_pnl_original across scenarios",
        },
        yaxis     = {"title": "", "automargin": True},
    ))
    return fig


# ═════════════════════════════════════════════════════════════════════
# Diagnostics — routing coverage donut (Phase 0.3)
# ═════════════════════════════════════════════════════════════════════

def build_coverage_donut(
    affected_count:   int,
    unaffected_count: int,
) -> go.Figure:
    """Donut: ``affected`` (violet) vs ``unaffected`` (slate) cluster counts.

    The hole is annotated with the total cluster count so the user
    gets the absolute scale at a glance and the percentages from the
    slice labels at the same time.
    """
    affected   = max(int(affected_count or 0), 0)
    unaffected = max(int(unaffected_count or 0), 0)
    total = affected + unaffected
    if total == 0:
        return empty_coverage_donut()

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels        = ["Affected", "Unaffected"],
        values        = [affected, unaffected],
        hole          = 0.55,
        marker        = {"colors": [_VIOLET, _ZERO_LINE]},
        sort          = False,
        textinfo      = "label+percent",
        textfont      = {"size": 11, "color": "#e2e8f0"},
        hovertemplate = "%{label}<br>%{value} clusters (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **rade_layout(
            show_legend = False,
            margin      = {"l": 8, "r": 8, "t": 8, "b": 8},
        ),
        annotations = [
            {
                "text":      (
                    f"<b>{total}</b><br>"
                    f"<span style='font-size:11px;color:#94a3b8'>"
                    f"cluster{'s' if total != 1 else ''}</span>"
                ),
                "showarrow": False,
                "x":         0.5, "y": 0.5,
                "font":      {"size": 20, "color": "#e2e8f0"},
            },
        ],
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
# Shared input adapters
# ═════════════════════════════════════════════════════════════════════

def _portfolio_pnl(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """Pull the canonical PnL series out of a portfolio frame.

    Returns ``None`` if the frame is missing, empty, or doesn't
    carry the expected ``sum_pnl_original`` column.  Dispatchers
    use this to fall back to the empty-state figure cleanly.
    """
    if df is None or df.empty or "sum_pnl_original" not in df.columns:
        return None
    return df["sum_pnl_original"].astype(float)


def _portfolio_labels(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """Pull the scenario-label series out of a portfolio frame."""
    if df is None or df.empty or "scenario_label" not in df.columns:
        return None
    return df["scenario_label"].astype(str)


def _var_cvar(pnl: pd.Series) -> tuple[float, float]:
    """Compute 95 % VaR / CVaR on a PnL series.

    CVaR collapses to VaR when no observation lies at or below the
    5th-percentile threshold (typical for very small N).  Both
    values are returned as plain floats so callers can drop them
    directly into annotation strings.
    """
    var_value = float(pnl.quantile(_VAR_QUANTILE))
    mask = pnl <= var_value
    cvar_value = float(pnl[mask].mean()) if mask.any() else var_value
    return var_value, cvar_value


# ═════════════════════════════════════════════════════════════════════
# 1. chart_main — distribution / timeseries / overlay
# ═════════════════════════════════════════════════════════════════════

def build_chart_main(
    df:   Optional[pd.DataFrame],
    mode: Optional[str],
) -> go.Figure:
    """Top-level dispatcher for the *Charts* tab figure.

    Falls back to the matching empty figure when ``df`` is empty
    or the requested mode is the reserved ``overlay`` slot — so the
    caller can invoke this unconditionally without branching on
    "data ready yet".
    """
    mode = (mode or "distribution").lower()
    pnl  = _portfolio_pnl(df)
    if pnl is None or pnl.empty:
        return {
            "distribution": empty_pnl_distribution(),
            "timeseries":   empty_pnl_timeseries(),
            "overlay":      empty_pnl_overlay(),
        }.get(mode, empty_pnl_distribution())

    if mode == "distribution":
        return _pnl_distribution(df)
    if mode == "timeseries":
        return _pnl_timeseries(df)
    # overlay is reserved; keep the empty-state until the contract names it.
    return empty_pnl_overlay()


def _pnl_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of portfolio PnL across scenarios + VaR / CVaR lines.

    Bin count scales with N (capped 8–40) so a 10-scenario run still
    renders meaningfully and a 1000-scenario run doesn't blow into
    400 hair-thin bars.
    """
    pnl  = _portfolio_pnl(df)
    if pnl is None or pnl.empty:
        return empty_pnl_distribution()

    var_value, cvar_value = _var_cvar(pnl)
    nbins = max(8, min(40, len(pnl) // 5 or 8))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x             = pnl,
        nbinsx        = nbins,
        marker        = {"color": rgba(_VIOLET, 0.85)},
        name          = "Portfolio PnL",
        hovertemplate = "PnL: %{x:,.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(
        x=var_value,
        line={"color": _AMBER, "width": 1.5, "dash": "dot"},
        annotation_text     = f"VaR 95 %: {var_value:,.0f}",
        annotation_position = "top right",
        annotation_font     = {"color": _AMBER, "size": 10},
    )
    fig.add_vline(
        x=cvar_value,
        line={"color": _ROSE, "width": 1.5, "dash": "dash"},
        annotation_text     = f"CVaR 95 %: {cvar_value:,.0f}",
        annotation_position = "bottom right",
        annotation_font     = {"color": _ROSE, "size": 10},
    )
    fig.update_layout(**rade_layout(
        hovermode = "x",
        xaxis     = {"title": "Predicted PnL (original units)"},
        yaxis     = {"title": "Scenarios"},
    ))
    return fig


def _pnl_timeseries(df: pd.DataFrame) -> go.Figure:
    """Line of portfolio PnL ordered by scenario label.

    Uses :func:`_theme.sort_chronologically` so date-shaped
    scenario labels render in real-time order even when the
    upstream parquet is shuffled (legacy training-time order).
    """
    pnl    = _portfolio_pnl(df)
    labels = _portfolio_labels(df)
    if pnl is None or labels is None or pnl.empty:
        return empty_pnl_timeseries()

    ordered = sort_chronologically(df.copy())
    y       = ordered["sum_pnl_original"].astype(float)
    x       = ordered["scenario_label"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x             = x,
        y             = y,
        mode          = "lines+markers",
        line          = {"color": _VIOLET, "width": 1.6},
        marker        = {"color": _VIOLET, "size": 4, "opacity": 0.85},
        name          = "Portfolio PnL",
        hovertemplate = "%{x}<br>PnL: %{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(
        y=0,
        line={"color": _ZERO_LINE, "width": 1, "dash": "dot"},
    )
    fig.update_layout(**rade_layout(
        hovermode = "x unified",
        xaxis     = {"title": "Scenario"},
        yaxis     = {"title": "PnL (original units)"},
    ))
    return fig


# ═════════════════════════════════════════════════════════════════════
# 2. risk_attribution_chart — cluster / risk_factor / trade_type
# ═════════════════════════════════════════════════════════════════════

def build_risk_attribution_chart(
    clusters_df:          Optional[pd.DataFrame],
    breakdown:            Optional[str],
    *,
    expected_cluster_ids: Optional[Sequence[str]] = None,
    cluster_decisions:    Optional[Sequence[Dict[str, Any]]] = None,
) -> go.Figure:
    """Top-level dispatcher for the *Risk attribution* tab figure.

    Phase 1a.2 swaps the *risk_factor* axis from a "v1 unavailable"
    placeholder to a real chart: Approach B — each cluster's signed
    PnL is split **evenly** across the cluster's intersecting RFs
    (pulled from ``manifest.validation.cluster_decisions``), then
    summed across clusters per RF.  Phase 1b.2 will introduce
    Approach A (trade-primary-RF) once the trade-attribute API ships.

    Trade-type breakdown still renders a descriptive empty state
    because it requires per-trade attribute metadata that the v1
    data plane doesn't expose (Phase 1b).

    ``expected_cluster_ids`` is the canonical (validation-report)
    list of cluster IDs the run *should* have covered.  When
    provided, ``_attribution_by_cluster`` left-joins the aggregate
    against this list and zero-pads any missing cluster — the chart
    then always renders every expected cluster, regardless of
    writer-side drops.  (Phase 0.2.)

    ``cluster_decisions`` is the ``manifest.validation.cluster_decisions``
    list — one dict per cluster carrying its router decision,
    intersecting-RF set, etc.  Required for the *risk_factor*
    breakdown; falls back to an explanatory empty state when None.
    """
    breakdown = (breakdown or "cluster").lower()
    if clusters_df is None or clusters_df.empty:
        return empty_risk_attribution()

    if breakdown == "cluster":
        return _attribution_by_cluster(
            clusters_df,
            expected_cluster_ids=expected_cluster_ids,
        )
    if breakdown == "risk_factor":
        if not cluster_decisions:
            return _attribution_unavailable(
                "Risk-factor breakdown needs the validation report — "
                "cluster_decisions missing from this run's manifest."
            )
        return _attribution_by_risk_factor(
            clusters_df,
            cluster_decisions=cluster_decisions,
        )
    if breakdown == "trade_type":
        return _attribution_unavailable(
            "Trade-type breakdown requires a trade-attribute join "
            "(not exposed by the v1 API)."
        )
    return empty_risk_attribution()


def _attribution_unavailable(message: str) -> go.Figure:
    """Empty-figure variant carrying a *why* explanation."""
    return empty_figure(message)


def _attribution_by_cluster(
    clusters_df:          pd.DataFrame,
    *,
    expected_cluster_ids: Optional[Sequence[str]] = None,
) -> go.Figure:
    """Total signed contribution per cluster, summed across scenarios.

    Aggregates ``sum_pnl_original`` over scenarios for each cluster
    and renders a horizontal bar chart, green/positive vs rose/negative
    so the eye lands on signs immediately.  Sorted ascending so the
    worst clusters sit at the bottom — quickest to read.

    When ``expected_cluster_ids`` is provided, the aggregate is left-
    joined against it and any cluster missing from the data plane is
    zero-padded and rendered with a desaturated slate fill.  This
    guarantees the chart shows every cluster the validation report
    classified, even if a writer-side fault dropped one between
    ``_post_infer_cluster`` and the parquet flush.
    """
    if "cluster_id" not in clusters_df.columns or "sum_pnl_original" not in clusters_df.columns:
        return empty_risk_attribution()

    agg = (
        clusters_df.groupby("cluster_id", as_index=False)["sum_pnl_original"]
        .sum()
    )

    if expected_cluster_ids:
        expected = pd.DataFrame({"cluster_id": list(expected_cluster_ids)})
        agg = expected.merge(agg, on="cluster_id", how="left")
        agg["_is_missing"] = agg["sum_pnl_original"].isna()
        agg["sum_pnl_original"] = agg["sum_pnl_original"].fillna(0.0)
    else:
        agg["_is_missing"] = False

    agg = agg.sort_values("sum_pnl_original", ascending=True).reset_index(drop=True)

    colors = [
        _ZERO_LINE if missing
        else (_EMERALD if value >= 0 else _ROSE)
        for value, missing in zip(agg["sum_pnl_original"], agg["_is_missing"])
    ]
    opacities = [0.35 if m else 0.9 for m in agg["_is_missing"]]

    hovertemplate = (
        "%{y}<br>"
        "Contribution: %{x:,.2f}"
        "%{customdata[0]}"
        "<extra></extra>"
    )
    customdata = [
        ["<br><i>(no data — zero-padded)</i>"] if m else [""]
        for m in agg["_is_missing"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x             = agg["sum_pnl_original"],
        y             = agg["cluster_id"],
        orientation   = "h",
        marker        = {"color": colors, "opacity": opacities},
        customdata    = customdata,
        hovertemplate = hovertemplate,
    ))
    fig.add_vline(
        x=0,
        line={"color": _ZERO_LINE, "width": 1, "dash": "dot"},
    )
    fig.update_layout(**rade_layout(
        hovermode = "y",
        margin    = {"l": 120, "r": 16, "t": 8, "b": 36},
        xaxis     = {"title": "Contribution to portfolio PnL (original units)"},
        yaxis     = {"title": "", "automargin": True},
    ))
    return fig


def _attribution_by_risk_factor(
    clusters_df:       pd.DataFrame,
    *,
    cluster_decisions: Sequence[Dict[str, Any]],
    top_n:             int = 15,
) -> go.Figure:
    """Risk-factor attribution via **Approach B** (cluster-intersection split).

    Approach B is the first RF attribution we can compute *without*
    new data:

    * For each cluster the router classified as "affected", read
      ``cluster_decisions[i].intersecting_risk_factors`` — i.e. the
      RFs that intersected the scenario shocks.
    * Compute that cluster's signed PnL by summing
      ``sum_pnl_original`` over all scenarios.
    * Split that signed PnL **evenly** across the cluster's
      intersecting RFs (one-Nth weight per RF).
    * Sum across clusters per RF to get the per-RF aggregate.

    Plotted as a horizontal bar, sign-coloured (emerald positive,
    rose negative), sorted by absolute magnitude.  RFs beyond
    ``top_n`` collapse into a single "Other" bar so the chart
    stays legible on portfolios with hundreds of RFs.

    Methodology caveats — surfaced as a chart annotation so the
    user understands what they're reading:

    * Even-split is a *first-order* attribution — it ignores
      per-RF shock magnitude and per-trade RF sensitivity.  Phase
      1b.2 (Approach A) will replace this with the trade-primary-RF
      attribution once the trade-attribute API ships.
    * Clusters with empty ``intersecting_risk_factors`` (unaffected
      clusters or pure history-lookup paths) are dropped from the
      aggregate — they contribute zero to every RF, by construction.
    """
    if "cluster_id" not in clusters_df.columns or "sum_pnl_original" not in clusters_df.columns:
        return empty_risk_attribution()

    # Cluster → signed PnL aggregate (single groupby, reused below).
    pnl_per_cluster = (
        clusters_df.groupby("cluster_id")["sum_pnl_original"]
        .sum()
        .to_dict()
    )

    # Distribute each cluster's signed PnL evenly across its
    # intersecting RFs and sum per RF.  ``defaultdict(float)``
    # initialises to 0.0 so we can ``+=`` without pre-seeding keys.
    rf_contrib: Dict[str, float] = defaultdict(float)
    for decision in cluster_decisions:
        if not isinstance(decision, dict):
            continue
        cid = decision.get("cluster_id")
        if cid is None:
            continue
        rfs = decision.get("intersecting_risk_factors") or []
        if not rfs:
            continue
        cluster_pnl = float(pnl_per_cluster.get(str(cid), 0.0))
        share       = cluster_pnl / float(len(rfs))
        for rf in rfs:
            rf_contrib[str(rf)] += share

    if not rf_contrib:
        return _attribution_unavailable(
            "No affected clusters in this run — nothing to attribute "
            "by risk factor (all clusters took the history-lookup path)."
        )

    # Rank by absolute magnitude descending — surfacing the
    # most-impactful RFs first, regardless of sign.
    ranked = sorted(rf_contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top    = ranked[:top_n]
    rest   = ranked[top_n:]

    rows: list[tuple[str, float]] = list(top)
    if rest:
        rows.append((
            f"Other ({len(rest)})",
            float(sum(value for _, value in rest)),
        ))

    # Reverse for horizontal-bar bottom-up ordering — biggest at top.
    rows = list(reversed(rows))
    labels = [name for name, _ in rows]
    values = [float(v) for _, v in rows]

    colors = [_EMERALD if v >= 0 else _ROSE for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x             = values,
        y             = labels,
        orientation   = "h",
        marker        = {"color": colors, "opacity": 0.9},
        text          = [f"{v:,.0f}" for v in values],
        textposition  = "outside",
        textfont      = {"size": 10, "color": "#cbd5e1"},
        hovertemplate = (
            "%{y}<br>"
            "Attributed PnL: %{x:,.2f}"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=0,
        line={"color": _ZERO_LINE, "width": 1, "dash": "dot"},
    )
    fig.update_layout(**rade_layout(
        hovermode = "y",
        margin    = {"l": 140, "r": 56, "t": 8, "b": 36},
        xaxis     = {"title": "Attributed PnL (original units, even-split across intersecting RFs)"},
        yaxis     = {"title": "", "automargin": True},
    ))
    # Methodology footnote so the user knows this is a first-order
    # attribution and not a trade-primary-RF computation.
    fig.add_annotation(
        text       = "Approach B · cluster PnL split evenly across intersecting RFs",
        xref       = "paper", yref = "paper",
        x          = 1.0, y = 1.02,
        xanchor    = "right", yanchor = "bottom",
        showarrow  = False,
        font       = {"color": "#64748b", "size": 10, "style": "italic"},
    )
    return fig


# ═════════════════════════════════════════════════════════════════════
# 3. stress_tails_chart — fan / tail / worst
# ═════════════════════════════════════════════════════════════════════

def build_stress_tails_chart(
    df:   Optional[pd.DataFrame],
    mode: Optional[str],
) -> go.Figure:
    """Top-level dispatcher for the *Stress & tails* tab figure."""
    mode = (mode or "fan").lower()
    pnl  = _portfolio_pnl(df)
    if pnl is None or pnl.empty:
        return empty_stress_tails()

    if mode == "fan":
        return _stress_fan(df)
    if mode == "tail":
        return _stress_tail(df)
    if mode == "worst":
        return _stress_worst(df)
    return empty_stress_tails()


def _stress_fan(df: pd.DataFrame) -> go.Figure:
    """Sorted scenario-PnL curve with VaR / CVaR annotations.

    A literal "percentile fan" doesn't apply when each scenario has
    a single realisation, so this mode shows the **sorted-PnL
    curve** instead.  X = percentile rank (0–100 %), Y = PnL — the
    tail shape becomes legible at a glance.  Area is filled to zero
    so positive and negative regions read clearly.
    """
    pnl = _portfolio_pnl(df)
    if pnl is None or pnl.empty:
        return empty_stress_tails()

    pnl_sorted = pnl.sort_values().reset_index(drop=True)
    n          = len(pnl_sorted)
    percentile = (np.arange(1, n + 1) / n) * 100.0

    var_value, cvar_value = _var_cvar(pnl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x             = percentile,
        y             = pnl_sorted,
        mode          = "lines",
        line          = {"color": _VIOLET, "width": 1.8},
        fill          = "tozeroy",
        fillcolor     = rgba(_VIOLET, 0.18),
        name          = "Sorted PnL",
        hovertemplate = "Percentile: %{x:.0f} %<br>PnL: %{y:,.2f}<extra></extra>",
    ))
    fig.add_vline(
        x=_VAR_QUANTILE * 100,
        line={"color": _AMBER, "width": 1.5, "dash": "dot"},
        annotation_text     = f"VaR 95 %: {var_value:,.0f}",
        annotation_position = "top right",
        annotation_font     = {"color": _AMBER, "size": 10},
    )
    fig.add_hline(
        y=cvar_value,
        line={"color": _ROSE, "width": 1.5, "dash": "dash"},
        annotation_text     = f"CVaR 95 %: {cvar_value:,.0f}",
        annotation_position = "top right",
        annotation_font     = {"color": _ROSE, "size": 10},
    )
    fig.update_layout(**rade_layout(
        hovermode = "x",
        xaxis     = {"title": "Percentile rank", "ticksuffix": " %"},
        yaxis     = {"title": "PnL (original units)"},
    ))
    return fig


def _stress_tail(df: pd.DataFrame) -> go.Figure:
    """Histogram with the tail (PnL ≤ VaR) shaded rose.

    Two overlapping histograms with shared bin width so the body
    (violet) and tail (rose) read as one continuous distribution
    that just happens to be colour-segmented at the VaR threshold.
    Tail bin count scales down to keep individual tail bars
    visible even when N is small.
    """
    pnl = _portfolio_pnl(df)
    if pnl is None or pnl.empty:
        return empty_stress_tails()

    var_value, cvar_value = _var_cvar(pnl)
    nbins_total = max(8, min(40, len(pnl) // 5 or 8))

    body = pnl[pnl >  var_value]
    tail = pnl[pnl <= var_value]

    fig = go.Figure()
    if not body.empty:
        fig.add_trace(go.Histogram(
            x             = body,
            nbinsx        = nbins_total,
            marker        = {"color": rgba(_VIOLET, 0.85)},
            name          = "Body",
            hovertemplate = "PnL: %{x:,.2f}<br>Count: %{y}<extra></extra>",
        ))
    if not tail.empty:
        fig.add_trace(go.Histogram(
            x             = tail,
            nbinsx        = max(4, nbins_total // 4),
            marker        = {"color": rgba(_ROSE, 0.9)},
            name          = "Tail (≤ VaR)",
            hovertemplate = "PnL: %{x:,.2f}<br>Count: %{y}<extra></extra>",
        ))

    fig.add_vline(
        x=var_value,
        line={"color": _AMBER, "width": 1.5, "dash": "dot"},
        annotation_text     = f"VaR 95 %: {var_value:,.0f}",
        annotation_position = "top right",
        annotation_font     = {"color": _AMBER, "size": 10},
    )
    fig.add_vline(
        x=cvar_value,
        line={"color": _ROSE, "width": 1.5, "dash": "dash"},
        annotation_text     = f"CVaR 95 %: {cvar_value:,.0f}",
        annotation_position = "bottom right",
        annotation_font     = {"color": _ROSE, "size": 10},
    )
    fig.update_layout(
        **rade_layout(
            show_legend = True,
            hovermode   = "x",
            xaxis       = {"title": "PnL (original units)"},
            yaxis       = {"title": "Scenarios"},
        ),
        barmode = "overlay",
    )
    return fig


def _stress_worst(df: pd.DataFrame, n: int = _WORST_N_DEFAULT) -> go.Figure:
    """Top-N worst-loss scenarios as a horizontal bar chart.

    Always rose-coloured because every bar in this mode is in the
    loss tail by construction.  Sorted ascending so the worst-of-
    the-worst sits at the bottom of the chart (closest to the
    user's eye line).
    """
    pnl    = _portfolio_pnl(df)
    labels = _portfolio_labels(df)
    if pnl is None or labels is None or pnl.empty:
        return empty_stress_tails()

    sub = (
        pd.DataFrame({"label": labels, "pnl": pnl})
        .nsmallest(n, "pnl")
        .sort_values("pnl", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x             = sub["pnl"],
        y             = sub["label"],
        orientation   = "h",
        marker        = {"color": rgba(_ROSE, 0.9)},
        hovertemplate = "%{y}<br>PnL: %{x:,.2f}<extra></extra>",
    ))
    fig.add_vline(
        x=0,
        line={"color": _ZERO_LINE, "width": 1, "dash": "dot"},
    )
    fig.update_layout(**rade_layout(
        hovermode = "y",
        margin    = {"l": 150, "r": 16, "t": 8, "b": 36},
        xaxis     = {"title": "PnL (original units)"},
        yaxis     = {"title": "", "automargin": True},
    ))
    return fig


__all__ = [
    # Dispatchers (Stage 13 — primary entry points)
    "build_chart_main",
    "build_risk_attribution_chart",
    "build_stress_tails_chart",
    # Diagnostics (Phase 0.3)
    "build_coverage_donut",
    "empty_coverage_donut",
    # Sensitivity (Phase 1a.1)
    "build_sensitivity_heatmap",
    "build_sensitivity_tornado",
    "empty_sensitivity_heatmap",
    "empty_sensitivity_tornado",
    # Empty-state builders (still public — callbacks use them in the
    # "no data yet" branch).
    "empty_pnl_distribution",
    "empty_pnl_overlay",
    "empty_pnl_timeseries",
    "empty_risk_attribution",
    "empty_stress_tails",
]
