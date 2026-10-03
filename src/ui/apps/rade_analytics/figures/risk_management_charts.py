"""Plotly figure builders for the Risk Management page (Phase 4.0).

Eval-data driven portfolio risk view.  Every builder reads its PnL
series via a ``pnl_column`` kwarg so the same code paths work for the
*Predicted* (``df["predictions"]``) and *Actual* (``df["targets"]``)
toggle states.  Phase 3.2's reader-side rename (``predictions_original``
→ ``predictions``, ``targets_original`` → ``targets`` when
``space="original"``) means the builders see canonical column names
regardless of which physical column is on disk.

Builders
--------
* :func:`build_pnl_distribution` — portfolio PnL histogram with VaR /
  CVaR vertical markers.  Canonical risk-management visual.
* :func:`build_cluster_waterfall` — scenario-pinned cluster
  contribution waterfall.  Defaults to the worst portfolio loss.
* :func:`build_cluster_tornado` — clusters ranked by absolute
  contribution across the split.
* :func:`build_cluster_donut` — single-ring contribution share donut
  with sign-coloured slices.

KPI / stats helpers
-------------------
* :func:`compute_tail_stats` — VaR 95 / VaR 99 / CVaR 95 / CVaR 99 /
  skew / excess-kurtosis at portfolio level.
* :func:`compute_tail_table` — top-N worst scenarios with top-cluster
  enrichment.

All figures route through :func:`figures._theme.rade_layout` so the
violet / emerald / amber / rose palette stays consistent across the
app.  ``empty_*`` builders return placeholders the layout paints at
mount time (before the hydrate callback fetches real data).
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

def empty_pnl_distribution() -> go.Figure:
    """Placeholder for the portfolio PnL distribution histogram."""
    return empty_figure("Awaiting evaluation data — PnL distribution")


def empty_cluster_waterfall() -> go.Figure:
    """Placeholder for the cluster-level P&L waterfall."""
    return empty_figure("Awaiting evaluation data — P&L decomposition")


def empty_cluster_tornado() -> go.Figure:
    """Placeholder for the cluster-level tornado chart."""
    return empty_figure("Awaiting evaluation data — cluster sensitivity")


def empty_cluster_donut() -> go.Figure:
    """Placeholder for the cluster-level contribution donut."""
    return empty_figure("Awaiting evaluation data — contribution donut")


# ═════════════════════════════════════════════════════════════════════
# Shared adapters
# ═════════════════════════════════════════════════════════════════════

def _portfolio_pnl(
    df:         Optional[pd.DataFrame],
    *,
    pnl_column: str,
) -> Optional[pd.Series]:
    """Pull the canonical portfolio-PnL series from a frame.

    The eval pipeline emits both ``predictions`` and ``targets``
    columns (after the API's ``_apply_space("original")`` rename); the
    page toggle picks one of the two via ``pnl_column``.  Returns
    ``None`` for an empty / malformed / all-NaN frame so callers can
    fall back to the matching empty figure with a targeted message.
    """
    if df is None or df.empty or pnl_column not in df.columns:
        return None
    series = df[pnl_column].astype(float)
    if series.isna().all():
        return None
    return series.dropna()


def _aggregate_cluster_contributions(
    clusters_df:    Optional[pd.DataFrame],
    *,
    pnl_column:     str,
    scenario_label: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Aggregate per-cluster signed contribution.

    Returns a frame with two columns (``cluster_id``, ``pnl_total``)
    or ``None`` when the input frame is empty / lacking the expected
    columns / entirely NaN in ``pnl_column``.

    Parameters
    ----------
    clusters_df
        Long-format cluster timeseries (one row per cluster × scenario).
    pnl_column
        Which PnL column to aggregate (``"predictions"`` or ``"targets"``).
    scenario_label
        When ``None`` (default) we sum across **every** scenario — the
        canonical "all-scenarios aggregate" view used by the tornado /
        donut headlines.  When set to a specific label, we filter the
        frame to that single scenario first, so the same builders also
        serve the page-local scenario-filter dropdown (which lets the
        user inspect cluster contributions for one event at a time).
    """
    if clusters_df is None or clusters_df.empty:
        return None
    if not {"cluster_id", pnl_column}.issubset(clusters_df.columns):
        return None
    if scenario_label is not None:
        if "scenario_label" not in clusters_df.columns:
            return None
        clusters_df = clusters_df[
            clusters_df["scenario_label"].astype(str) == str(scenario_label)
        ]
        if clusters_df.empty:
            return None
    agg = (
        clusters_df.groupby("cluster_id", as_index=False)[pnl_column]
        .sum(min_count=1)
        .rename(columns={pnl_column: "pnl_total"})
    )
    agg = agg.dropna(subset=["pnl_total"])
    if agg.empty:
        return None
    return agg


# ═════════════════════════════════════════════════════════════════════
# 1. PnL distribution — canonical risk-management visual
# ═════════════════════════════════════════════════════════════════════

def build_pnl_distribution(
    portfolio_df: Optional[pd.DataFrame],
    *,
    pnl_column:   str,
    stats:        Optional[dict] = None,
    n_bins:       int = 60,
) -> go.Figure:
    """Histogram of portfolio PnL with VaR / CVaR vertical markers.

    The leftmost bin reveals the tail at a glance; the dashed
    amber/rose verticals at VaR 95 and CVaR 95 anchor the magnitudes
    that the KPI strip prints numerically.  The left-tail region
    (PnL ≤ VaR 95) is overlaid with a translucent rose fill so the
    user sees both *where* and *how much* sits in the tail.

    Parameters
    ----------
    portfolio_df
        Portfolio-level long-format frame; one row per scenario.
        Must carry ``pnl_column`` (post-``_apply_space("original")``
        canonical name).
    pnl_column
        ``"predictions"`` or ``"targets"`` — driven by the page-local
        Predicted/Actual toggle.
    stats
        Pre-computed dict from :func:`compute_tail_stats`.  If
        ``None`` we re-compute here so the figure stays usable
        standalone (unit tests, ad-hoc notebooks).
    n_bins
        Histogram resolution.  60 reads well on a 1000-scenario test
        split without smoothing the tail away.
    """
    pnl = _portfolio_pnl(portfolio_df, pnl_column=pnl_column)
    if pnl is None or pnl.empty:
        return empty_pnl_distribution()

    if stats is None:
        stats = compute_tail_stats(portfolio_df, pnl_column=pnl_column)

    var_95  = stats.get("var_95")
    cvar_95 = stats.get("cvar_95")
    mean_v  = float(pnl.mean())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x             = pnl,
        nbinsx        = n_bins,
        marker        = {
            "color":   _VIOLET,
            "opacity": 0.55,
            "line":    {"color": "#1e1b4b", "width": 0.5},
        },
        hovertemplate = "PnL: %{x:,.0f}<br>Count: %{y}<extra></extra>",
        name          = "PnL",
    ))

    # Mean marker — thin slate vertical to anchor the centre.
    fig.add_vline(
        x         = mean_v,
        line      = {"color": _SLATE, "width": 1, "dash": "dot"},
        annotation_text     = f"μ = {mean_v:,.0f}",
        annotation_position = "top",
        annotation_font     = {"color": _SLATE, "size": 10},
    )

    # VaR / CVaR markers + tail shading — only when we have a finite
    # value (otherwise the dashed line is meaningless).
    if var_95 is not None and np.isfinite(var_95):
        fig.add_vline(
            x         = var_95,
            line      = {"color": _AMBER, "width": 1.5, "dash": "dash"},
            annotation_text     = f"VaR 95% = {var_95:,.0f}",
            annotation_position = "top",
            annotation_font     = {"color": _AMBER, "size": 10},
        )
        # Shade the tail beyond VaR with a translucent rose fill.
        x_min = float(pnl.min())
        if np.isfinite(x_min) and x_min < var_95:
            fig.add_vrect(
                x0         = x_min,
                x1         = var_95,
                fillcolor  = rgba(_ROSE, 0.10),
                line_width = 0,
                layer      = "below",
            )

    if cvar_95 is not None and np.isfinite(cvar_95):
        fig.add_vline(
            x         = cvar_95,
            line      = {"color": _ROSE, "width": 1.5, "dash": "dash"},
            annotation_text     = f"CVaR 95% = {cvar_95:,.0f}",
            annotation_position = "bottom",
            annotation_font     = {"color": _ROSE, "size": 10},
        )

    fig.update_layout(**rade_layout(
        hovermode = "x",
        margin    = {"l": 36, "r": 16, "t": 36, "b": 36},
        xaxis     = {
            "title":     "Portfolio PnL (original units)",
            "showgrid":  False,
            "zeroline":  False,
        },
        yaxis     = {
            "title":     "Scenarios",
            "rangemode": "tozero",
        },
    ))
    fig.update_layout(bargap=0.02)
    return fig


# ═════════════════════════════════════════════════════════════════════
# 2. Cluster waterfall — scenario-pinned P&L decomposition
# ═════════════════════════════════════════════════════════════════════

def build_cluster_waterfall(
    clusters_df:    Optional[pd.DataFrame],
    scenario_label: Optional[str] = None,
    *,
    pnl_column:     str,
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
        Long-format cluster timeseries (one row per cluster ×
        scenario), with at least ``cluster_id``, ``scenario_label``,
        ``pnl_column`` columns.
    scenario_label
        Scenario to decompose.  None → worst portfolio PnL.
    pnl_column
        ``"predictions"`` or ``"targets"`` — driven by the page-local
        Predicted/Actual toggle.
    top_n
        How many top-magnitude clusters keep their own bar.
    """
    if clusters_df is None or clusters_df.empty:
        return empty_cluster_waterfall()
    needed = {"cluster_id", "scenario_label", pnl_column}
    if not needed.issubset(clusters_df.columns):
        return empty_cluster_waterfall()

    # Resolve scenario_label — default to the worst portfolio loss.
    portfolio_by_scenario = (
        clusters_df.groupby("scenario_label", as_index=False)[pnl_column]
        .sum(min_count=1)
    )
    portfolio_by_scenario = portfolio_by_scenario.dropna(subset=[pnl_column])
    if portfolio_by_scenario.empty:
        return empty_cluster_waterfall()
    if scenario_label is None or scenario_label not in set(portfolio_by_scenario["scenario_label"]):
        worst_idx = portfolio_by_scenario[pnl_column].idxmin()
        scenario_label = str(portfolio_by_scenario.loc[worst_idx, "scenario_label"])

    sub = clusters_df.loc[
        clusters_df["scenario_label"].astype(str) == str(scenario_label),
        ["cluster_id", pnl_column],
    ].copy()
    sub = sub.dropna(subset=[pnl_column])
    if sub.empty:
        return empty_cluster_waterfall()
    sub[pnl_column] = sub[pnl_column].astype(float)

    # Rank by |contribution| and split top_n vs rest.
    sub = sub.assign(abs_pnl=sub[pnl_column].abs()).sort_values(
        "abs_pnl", ascending=False,
    )
    top    = sub.head(top_n)
    rest   = sub.iloc[top_n:]
    other  = float(rest[pnl_column].sum()) if not rest.empty else 0.0
    total  = float(sub[pnl_column].sum())

    # Keep top bars in their |contribution| order so the biggest
    # movers land at the LEFT (most-read position).
    cluster_labels  = list(top["cluster_id"].astype(str))
    cluster_values  = list(top[pnl_column].astype(float))
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
# 3. Cluster tornado — magnitude-ranked sensitivity bars
# ═════════════════════════════════════════════════════════════════════

def build_cluster_tornado(
    clusters_df:    Optional[pd.DataFrame],
    *,
    pnl_column:     str,
    top_n:          int = 12,
    scenario_label: Optional[str] = None,
) -> go.Figure:
    """Horizontal bar chart of clusters ranked by |contribution|.

    Same data source as the cluster donut but sorted by absolute
    value (descending), so the biggest movers — regardless of sign —
    land at the top of the chart.  Phase 4.2 swaps this for a true
    risk-factor tornado once trade-RF mappings exist.

    Parameters
    ----------
    clusters_df
        Long-format cluster timeseries (one row per cluster ×
        scenario).
    pnl_column
        ``"predictions"`` or ``"targets"`` — driven by the page-local
        Predicted/Actual toggle.
    top_n
        Cap on visible bars.  Beyond this, the rest are aggregated
        into an "Other" bar so the chart stays scannable.
    scenario_label
        ``None`` (default) aggregates contributions across every
        scenario in the split — the canonical "macro tornado" view.
        Pass a specific label to filter down to a single scenario so
        the same builder serves the page-local scenario dropdown.
    """
    agg = _aggregate_cluster_contributions(
        clusters_df, pnl_column=pnl_column, scenario_label=scenario_label,
    )
    if agg is None:
        return empty_cluster_tornado()

    agg = agg.assign(abs_pnl=agg["pnl_total"].abs()).sort_values(
        "abs_pnl", ascending=False,
    )
    top  = agg.head(top_n)
    rest = agg.iloc[top_n:]
    rows = list(top[["cluster_id", "pnl_total"]].itertuples(index=False))
    if not rest.empty:
        rows.append(("Other (%d)" % len(rest), float(rest["pnl_total"].sum())))

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
        # ``auto`` lets Plotly place labels INSIDE the bar when there's
        # room, falling back to outside only for the smallest bars.
        # This recovers the horizontal real estate the old ``outside``
        # setting reserved for value text, so the bars themselves
        # render visibly wider for the same container width.
        textposition  = "auto",
        insidetextanchor = "middle",
        textfont      = {"size": 10, "color": "#0f172a"},
        hovertemplate = "%{y}<br>Contribution: %{x:,.2f}<extra></extra>",
    ))
    fig.add_vline(
        x=0,
        line={"color": _ZERO_LINE, "width": 1, "dash": "dot"},
    )
    fig.update_layout(**rade_layout(
        hovermode = "y",
        # Margins trimmed from (l=140, r=56) → (l=96, r=20):
        # * left:  cluster_id labels are short (<= 16 chars in
        #          practice).  96px is enough for them and gives the
        #          bars +44px of plot width.
        # * right: with ``textposition="auto"`` we no longer need a
        #          generous right margin to fit outside-text labels.
        margin    = {"l": 96, "r": 20, "t": 8, "b": 36},
        xaxis     = {"title": "Contribution (original units)"},
        yaxis     = {"title": "", "automargin": True},
    ))
    return fig


# ═════════════════════════════════════════════════════════════════════
# 4. Cluster donut — single-ring contribution breakdown
# ═════════════════════════════════════════════════════════════════════

def build_cluster_donut(
    clusters_df:    Optional[pd.DataFrame],
    *,
    pnl_column:     str,
    top_n:          int = 8,
    scenario_label: Optional[str] = None,
) -> go.Figure:
    """Donut of |contribution| share per cluster, slice-coloured by sign.

    Phase 4.0 deliberately stays single-ring; Phase 4.2 upgrades to
    the full RF-group / trade-type / cluster sunburst once Phase 2.1's
    trade attribute API lands.

    Parameters
    ----------
    clusters_df
        Long-format cluster timeseries.
    pnl_column
        ``"predictions"`` or ``"targets"`` — driven by the page-local
        Predicted/Actual toggle.
    top_n
        Slice count cap.  The rest are folded into a single "Other"
        slice so the chart doesn't degenerate into a hairline pie on
        large portfolios.
    scenario_label
        ``None`` (default) sums contributions across every scenario.
        Pass a specific label to filter down to a single scenario so
        the same builder serves the page-local scenario dropdown.
    """
    agg = _aggregate_cluster_contributions(
        clusters_df, pnl_column=pnl_column, scenario_label=scenario_label,
    )
    if agg is None:
        return empty_cluster_donut()

    agg = agg.assign(abs_pnl=agg["pnl_total"].abs()).sort_values(
        "abs_pnl", ascending=False,
    )
    top  = agg.head(top_n)
    rest = agg.iloc[top_n:]

    labels:  list[str]   = list(top["cluster_id"].astype(str))
    signed:  list[float] = list(top["pnl_total"].astype(float))
    values:  list[float] = list(top["abs_pnl"].astype(float))
    if not rest.empty:
        labels.append(f"Other ({len(rest)})")
        signed.append(float(rest["pnl_total"].sum()))
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
# 5. KPI helpers (stats only — the tile DOM is built in the callback)
# ═════════════════════════════════════════════════════════════════════

def compute_tail_stats(
    portfolio_df: Optional[pd.DataFrame],
    *,
    pnl_column:   str,
) -> dict[str, Optional[float]]:
    """Compute VaR 95 / VaR 99 / CVaR 95 / CVaR 99 / skew / excess-kurtosis.

    Single-shot helper so the callback doesn't duplicate the quantile
    / mean / scipy-equivalents arithmetic.  Returns a dict keyed by
    metric name with ``None`` for any metric we can't derive from the
    input (empty frame, all-NaN ``pnl_column``, etc.).

    Quantile convention
    -------------------
    For loss distributions the *left tail* is the danger zone.  We
    therefore compute VaR as the *5th* (95%) and *1st* (99%) percentile
    — both negative numbers for a healthy P&L series.  CVaR is the
    conditional mean below that quantile.  This mirrors the industry
    convention where "VaR 95%" means "the loss is no worse than this
    with 95% confidence".
    """
    pnl = _portfolio_pnl(portfolio_df, pnl_column=pnl_column)
    if pnl is None or pnl.empty:
        return {
            "var_95":  None, "var_99":  None,
            "cvar_95": None, "cvar_99": None,
            "skew":    None, "kurt":    None,
        }

    var_95  = float(pnl.quantile(0.05))
    var_99  = float(pnl.quantile(0.01))
    cvar_95 = float(pnl[pnl <= var_95].mean()) if (pnl <= var_95).any() else var_95
    cvar_99 = float(pnl[pnl <= var_99].mean()) if (pnl <= var_99).any() else var_99
    # Pandas' ``skew`` / ``kurtosis`` use the bias-corrected
    # Fisher-Pearson estimators — same convention as scipy.stats so
    # downstream comparisons match.  Excess kurtosis ⇒ Gaussian = 0.
    skew_value = float(pnl.skew())     if len(pnl) > 2 else None
    kurt_value = float(pnl.kurtosis()) if len(pnl) > 3 else None

    return {
        "var_95":  var_95,
        "var_99":  var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "skew":    skew_value,
        "kurt":    kurt_value,
    }


def compute_tail_table(
    portfolio_df: Optional[pd.DataFrame],
    clusters_df:  Optional[pd.DataFrame],
    *,
    pnl_column:   str,
    n:            int = 10,
) -> pd.DataFrame:
    """Build the tail-conditional table — the worst-N scenarios.

    Columns (Phase 4.0 — what we can compute without trade attributes):

    * ``scenario_label``       — scenario identifier (string).
    * ``portfolio_pnl``        — total PnL for the scenario.
    * ``top_cluster``          — cluster with the largest negative contribution.
    * ``top_cluster_pnl``      — that cluster's contribution to this scenario.
    * ``share_of_tail_loss``   — this scenario's share of the displayed
      tail's total loss, as a percentage.  Defined as
      ``|portfolio_pnl| / sum(|portfolio_pnl|) × 100`` over the displayed
      rows, so the column **sums to 100%** across the table.  This
      replaces the original ``var_contribution_pct`` (which compared each
      scenario to the CVaR-95 average and could exceed 100% — confusing
      because the "%" implied a 0–100 scale).

    Phase 4.1 / 4.2 swap in ``top_rf`` and ``top_trade_type`` columns
    once trade-attribute / RF-mapping APIs ship.
    """
    pnl = _portfolio_pnl(portfolio_df, pnl_column=pnl_column)
    if (
        pnl is None
        or pnl.empty
        or portfolio_df is None
        or "scenario_label" not in portfolio_df.columns
    ):
        return pd.DataFrame(columns=[
            "scenario_label", "portfolio_pnl",
            "top_cluster", "top_cluster_pnl",
            "share_of_tail_loss",
        ])

    portfolio_sub = portfolio_df[["scenario_label", pnl_column]].copy()
    portfolio_sub.columns = ["scenario_label", "portfolio_pnl"]
    portfolio_sub = portfolio_sub.dropna(subset=["portfolio_pnl"])
    portfolio_sub = portfolio_sub.nsmallest(n, "portfolio_pnl")

    # Per-scenario worst cluster (most negative contribution).
    top_cluster_lookup: dict[str, tuple[str, float]] = {}
    if (
        clusters_df is not None
        and not clusters_df.empty
        and {"cluster_id", "scenario_label", pnl_column}.issubset(clusters_df.columns)
    ):
        cl = clusters_df[["cluster_id", "scenario_label", pnl_column]]
        cl = cl.dropna(subset=[pnl_column])
        cl = cl[cl["scenario_label"].isin(set(portfolio_sub["scenario_label"]))]
        # idxmin per scenario → cluster_id with smallest pnl_column.
        for label, grp in cl.groupby("scenario_label", sort=False):
            if grp.empty:
                continue
            row = grp.loc[grp[pnl_column].idxmin()]
            top_cluster_lookup[str(label)] = (
                str(row["cluster_id"]),
                float(row[pnl_column]),
            )

    portfolio_sub["top_cluster"] = portfolio_sub["scenario_label"].map(
        lambda lbl: top_cluster_lookup.get(str(lbl), ("—", np.nan))[0]
    )
    portfolio_sub["top_cluster_pnl"] = portfolio_sub["scenario_label"].map(
        lambda lbl: top_cluster_lookup.get(str(lbl), ("—", np.nan))[1]
    )

    # Share of the displayed tail's loss — bounded 0–100% and sums to
    # 100% across the visible rows.  Mini-bar in the UI keys off this
    # column, so it stays inside [0, 100] without needing a clip.
    abs_losses     = portfolio_sub["portfolio_pnl"].abs()
    total_tail_abs = float(abs_losses.sum())
    if total_tail_abs > 0:
        portfolio_sub["share_of_tail_loss"] = (
            abs_losses / total_tail_abs * 100.0
        )
    else:
        portfolio_sub["share_of_tail_loss"] = 0.0

    return portfolio_sub.reset_index(drop=True)


__all__ = [
    # Builders
    "build_pnl_distribution",
    "build_cluster_waterfall",
    "build_cluster_tornado",
    "build_cluster_donut",
    # Empty-state builders
    "empty_pnl_distribution",
    "empty_cluster_waterfall",
    "empty_cluster_tornado",
    "empty_cluster_donut",
    # KPI/stats helpers
    "compute_tail_stats",
    "compute_tail_table",
]
