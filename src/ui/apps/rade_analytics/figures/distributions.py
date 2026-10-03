"""Residual distribution figures — violin plots (aggregate + grouped).

Design contract
---------------
* **Aggregate** (``group_values`` is empty or all equal) — one violin
  across the whole dataset, plus a small μ/σ/skew/kurt annotation in
  the top-right corner.
* **Grouped** — one violin per unique group value, side-by-side.  No
  annotation (would clutter); per-group stats surface in the hover.

Both states use :class:`plotly.graph_objects.Violin` so the chart
shape-shifts smoothly when the user toggles the group-by control —
no flicker between two different chart types.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout, rgba


def residual_violin(
    residuals:     Sequence[float],
    *,
    group_values:  Optional[Sequence[str]] = None,
    group_order:   Optional[Iterable[str]] = None,
    group_label:   Optional[str] = None,
    y_axis_title:  str = "Residual (pred − actual)",
) -> go.Figure:
    """Residual distribution violin.

    Parameters
    ----------
    residuals
        1-D array-like of residual values.  Length must equal
        ``group_values`` when grouped.
    group_values
        Optional per-observation group key (e.g. ``["rates", "fx",
        "rates", ...]``).  When ``None`` or all values identical the
        chart renders as a single aggregate violin.
    group_order
        Optional iterable pinning the left-to-right violin order.
        Values not in ``group_values`` are skipped; values missing from
        ``group_order`` are appended alphabetically.
    group_label
        Short caption for the grouped dimension (e.g. ``"Desk"``) used
        in the x-axis title.  Ignored when aggregate.
    y_axis_title
        Y-axis caption — override when residuals use a non-default unit
        (e.g. ``"Residual (bps)"`` after a scale conversion).
    """
    arr = np.asarray(list(residuals), dtype=float)
    if arr.size == 0:
        return empty_figure("No residuals to plot.")

    grouped = group_values is not None and len(group_values) == arr.size

    if not grouped:
        return _aggregate_violin(arr, y_axis_title=y_axis_title)

    groups = _resolve_group_order(group_values, group_order)
    if not groups:
        return _aggregate_violin(arr, y_axis_title=y_axis_title)

    group_arr = np.asarray(group_values, dtype=object)
    fig = go.Figure()
    for idx, group in enumerate(groups):
        mask = group_arr == group
        values = arr[mask]
        if values.size == 0:
            continue
        fig.add_trace(
            go.Violin(
                y=values,
                name=str(group),
                x=[str(group)] * values.size,
                line_color=color_for_index(idx),
                fillcolor=rgba(color_for_index(idx), 0.2),
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                hoveron="violins",
                hovertemplate=(
                    f"<b>{group}</b><br>"
                    "median: %{median:.4f}<br>"
                    "Q1: %{q1:.4f} / Q3: %{q3:.4f}<br>"
                    "min: %{lowerfence:.4f} / max: %{upperfence:.4f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        **rade_layout(
            show_legend=False,
            xaxis={
                "title": {
                    "text": group_label or "",
                    "font": {"color": "#94a3b8", "size": 11},
                },
                "showgrid": False,
            },
            yaxis={"title": {"text": y_axis_title, "font": {"color": "#94a3b8"}}},
        ),
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.4)",
        line_width=1,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _aggregate_violin(
    residuals: np.ndarray,
    *,
    y_axis_title: str,
) -> go.Figure:
    """Single-violin chart + stats annotation."""
    mu    = float(residuals.mean())
    sigma = float(residuals.std())
    # Guard against divide-by-zero on a constant distribution.
    if sigma > 0:
        pct_1s = float(np.mean(np.abs(residuals - mu) <= sigma) * 100)
        pct_2s = float(np.mean(np.abs(residuals - mu) <= 2 * sigma) * 100)
        skew, kurt = _skew_kurt(residuals)
    else:
        pct_1s = pct_2s = 100.0
        skew = kurt = 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=residuals,
            x=[""] * residuals.size,
            name="Residuals",
            line_color=color_for_index(0),
            fillcolor=rgba(color_for_index(0), 0.2),
            box_visible=True,
            meanline_visible=True,
            points="outliers",
            hoveron="violins",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(148, 163, 184, 0.4)",
        line_width=1,
    )

    annotation_text = (
        f"μ={mu:.4f}  σ={sigma:.4f}<br>"
        f"skew={skew:.2f}  kurt={kurt:.2f}<br>"
        f"±1σ: {pct_1s:.1f}%  ±2σ: {pct_2s:.1f}%  "
        f"n={residuals.size:,}"
    )

    fig.update_layout(
        **rade_layout(
            xaxis={"visible": False},
            yaxis={"title": {"text": y_axis_title, "font": {"color": "#94a3b8"}}},
        ),
        annotations=[
            {
                "text":      annotation_text,
                "showarrow": False,
                "xref":      "paper",
                "yref":      "paper",
                "x":         0.98,
                "y":         0.98,
                "xanchor":   "right",
                "yanchor":   "top",
                "bgcolor":   "rgba(15, 23, 42, 0.85)",
                "bordercolor": "rgba(148, 163, 184, 0.3)",
                "borderwidth": 1,
                "borderpad":   6,
                "font": {"size": 11, "color": "#cbd5e1", "family": "Inter, system-ui"},
                "align":     "right",
            }
        ],
    )
    return fig


def _skew_kurt(residuals: np.ndarray) -> tuple[float, float]:
    """Skew + excess kurtosis.  scipy is pulled in lazily so the
    rade_analytics package doesn't hard-depend on scipy.stats just to
    render an annotation line."""
    try:
        from scipy import stats as _stats  # noqa: WPS433 — lazy import by design
        return float(_stats.skew(residuals)), float(_stats.kurtosis(residuals))
    except Exception:  # noqa: BLE001
        # Manual fallback — cheap third/fourth standardised moment.
        mu = residuals.mean()
        sigma = residuals.std()
        if sigma <= 0:
            return 0.0, 0.0
        z = (residuals - mu) / sigma
        return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def _resolve_group_order(
    group_values: Optional[Sequence[str]],
    group_order:  Optional[Iterable[str]],
) -> list[str]:
    if not group_values:
        return []
    uniq = sorted({str(g) for g in group_values if g is not None})
    if group_order is None:
        return uniq

    ordered: list[str] = []
    seen: set[str] = set()
    for g in group_order:
        key = str(g)
        if key in uniq and key not in seen:
            ordered.append(key)
            seen.add(key)
    for key in uniq:
        if key not in seen:
            ordered.append(key)
    return ordered


__all__ = ["residual_violin"]
