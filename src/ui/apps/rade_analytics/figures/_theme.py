"""Shared plotly layout defaults for every Rade figure.

One import point means the font, margins, grid colours and transparent
background live in exactly one place.  If the design spec shifts (say,
we swap to a lighter card tone), updating this module updates every
chart in the app.

Also hosts the deterministic category palette used whenever a chart
needs to colour-code groups (grouped violins, grouped scatter,
cross-cluster pair colours, …).  Picking the same palette here and in
``rade.css`` keeps chart colours in sync with the chip / badge tones.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────
# Canvas / typography
# ─────────────────────────────────────────────────────────────────────

_FONT_FAMILY = "Inter, system-ui, sans-serif"
_FONT_COLOR = "#cbd5e1"
_TICK_COLOR = "#64748b"
_GRID_COLOR = "rgba(30, 41, 59, 0.6)"


# ─────────────────────────────────────────────────────────────────────
# Category palette — 8 saturated tones, cycled when more groups exist.
# ─────────────────────────────────────────────────────────────────────

CATEGORY_PALETTE: tuple[str, ...] = (
    "#8b5cf6",  # violet (primary)
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#f43f5e",  # rose
    "#38bdf8",  # sky
    "#06b6d4",  # cyan
    "#d946ef",  # fuchsia
    "#84cc16",  # lime
)


def color_for_index(i: int) -> str:
    """Deterministic palette pick; cycles past 8 groups."""
    return CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]


def rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex string (``#8b5cf6``) to rgba notation.

    Plotly's violin / scatter validators refuse 8-digit hex alphas, so
    every fillcolor in this package goes through here.  Clamps alpha
    to [0, 1] so a bad caller can't produce garbage.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"rgba() expects a 6-digit hex colour, got {hex_color!r}")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r}, {g}, {b}, {a:.3f})"


# ─────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────


def rade_layout(
    *,
    show_legend: bool = False,
    hovermode: str = "closest",
    margin: Optional[Dict[str, int]] = None,
    xaxis: Optional[Dict[str, Any]] = None,
    yaxis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Base layout every figure in the app starts from.

    Returns a kwargs dict suitable for ``fig.update_layout(**...)``.
    Callers merge in chart-specific tweaks (titles, legend position,
    shapes) on top.
    """
    base_xaxis = {
        "showgrid":       False,
        "zeroline":       False,
        "showticklabels": True,
        "tickfont":       {"color": _TICK_COLOR},
    }
    base_yaxis = {
        "gridcolor": _GRID_COLOR,
        "zeroline":  False,
        "tickfont":  {"color": _TICK_COLOR},
    }
    if xaxis:
        base_xaxis.update(xaxis)
    if yaxis:
        base_yaxis.update(yaxis)

    return {
        "template":     "plotly_dark",
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "margin":       margin or {"l": 40, "r": 16, "t": 8, "b": 36},
        "font":         {"family": _FONT_FAMILY, "color": _FONT_COLOR, "size": 11},
        "xaxis":        base_xaxis,
        "yaxis":        base_yaxis,
        "hovermode":    hovermode,
        "showlegend":   show_legend,
        "legend": {
            "orientation": "h",
            "y": 1.1, "x": 1, "xanchor": "right",
            "bgcolor": "rgba(0, 0, 0, 0)",
            "font": {"color": "#94a3b8", "size": 11},
        },
    }


def empty_figure(message: str) -> go.Figure:
    """Figure shown when a callback has no data to plot."""
    fig = go.Figure()
    fig.update_layout(
        **rade_layout(
            margin={"l": 32, "r": 16, "t": 8, "b": 32},
            xaxis={"visible": False},
            yaxis={"visible": False},
        ),
        annotations=[
            {
                "text":      message,
                "showarrow": False,
                "font":      {"color": "#64748b", "size": 13},
                "xref":      "paper",
                "yref":      "paper",
                "x":         0.5,
                "y":         0.5,
            }
        ],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# PnL-space presentation helpers (Phase 3.4)
# ─────────────────────────────────────────────────────────────────────
#
# Every space-aware chart goes through these three helpers so the
# Scaled / Original toggle reads consistently across the app.  Keeping
# them in ``_theme.py`` means a future relabel (e.g. "original" →
# "currency units") updates every chart in one place.

# Human-friendly labels for axis titles / legend annotations.  The
# server-side ``space=`` API contract still uses ``scaled`` / ``original``
# everywhere — these are presentation-only.
PNL_SPACE_LABELS: Dict[str, str] = {
    "scaled":   "scaled",
    "original": "notional",
}


def pnl_axis_title(space: str, *, prefix: str = "PnL") -> str:
    """Return a Plotly axis-title string keyed on the PnL space.

    Example: ``pnl_axis_title("original")`` → ``"PnL (notional)"``.
    Used wherever a chart's y-axis or trace label depends on the
    Scaled / Original toggle so the toggle's effect is unambiguous.
    """
    label = PNL_SPACE_LABELS.get(space, space)
    return f"{prefix} ({label})"


def pnl_hover_format(space: str) -> str:
    """Return a Plotly numeric format spec keyed on the PnL space.

    Scaled values are dimensionless and typically tiny — ``:.4f`` keeps
    enough significant figures to distinguish 1e-3 differences.  Original
    values are notional PnL in currency units — large integers read best
    with thousands separators and no decimals (``:,.0f``).
    """
    return ":,.0f" if space == "original" else ":.4f"


def is_all_nan(series: Any) -> bool:
    """True when every entry in ``series`` is missing (NaN / None).

    The Phase 3.2 API converts NaN positions to JSON ``null`` for
    original-space requests on runs without scaler coverage; this helper
    is the canonical detector for "user toggled to Original but no data
    is available" → callers swap in an :func:`empty_figure` annotation
    rather than render a flat-zero chart.
    """
    if series is None:
        return True
    try:
        if len(series) == 0:
            return True
    except TypeError:
        return False
    return bool(pd.isna(series).all())


# ─────────────────────────────────────────────────────────────────────
# Chronological ordering for split-shuffled timeseries frames
# ─────────────────────────────────────────────────────────────────────


def sort_chronologically(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return a copy of ``df`` ordered by real time, not training order.

    The eval pipeline preserves *training-time* row order in the
    portfolio / cluster timeseries parquets.  For the *train* split,
    that order is shuffled — which means a naive ``sort_values
    ('scenario_idx')`` produces a line chart that zig-zags chronologically
    despite the X-axis carrying real dates.  For the held-out splits
    (``val`` / ``test``) the order is usually contiguous already, so
    this helper is a no-op on those.

    Resolution order, descending preference:

    1. ``scenario_label`` parsed as ISO-8601 date / datetime.  Used
       when ≥ 1 label parses successfully.  Unparseable labels (NaT)
       sort to the end of the frame, which keeps them visible while
       the parseable bulk reads correctly.
    2. ``scenario_label`` parsed as a number — covers the
       positional-fallback case in
       :func:`rade_ml_pt.pipelines.ensemble.eval._save_portfolio_timeseries_parquet`
       (legacy runs without trade-universe scenario labels).
    3. ``scenario_idx`` ascending.  Last-resort positional sort.
    4. Original frame, untouched.

    Returning a frame untouched is the right default — passing a
    frame in already-chronological order through this helper is
    idempotent.
    """
    if df is None or df.empty:
        return df

    if "scenario_label" in df.columns:
        labels = df["scenario_label"].astype(str)

        dt = pd.to_datetime(labels, errors="coerce", format="ISO8601")
        if dt.notna().any():
            return (
                df.assign(_chron_sort_key=dt)
                .sort_values("_chron_sort_key", kind="stable", na_position="last")
                .drop(columns="_chron_sort_key")
            )

        num = pd.to_numeric(labels, errors="coerce")
        if num.notna().any():
            return (
                df.assign(_chron_sort_key=num)
                .sort_values("_chron_sort_key", kind="stable", na_position="last")
                .drop(columns="_chron_sort_key")
            )

    if "scenario_idx" in df.columns:
        return df.sort_values("scenario_idx", kind="stable")

    return df


__all__ = [
    "CATEGORY_PALETTE",
    "color_for_index",
    "empty_figure",
    "rade_layout",
    "rgba",
    "sort_chronologically",
]
