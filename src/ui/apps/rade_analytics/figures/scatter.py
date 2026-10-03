"""Predicted-vs-actual scatter figure.

Three modes share one entry point (:func:`pred_actual_scatter`), picked
by the arguments the caller passes:

* **Aggregate** — ``group_values`` is ``None`` → single-colour scatter.
* **Grouped**   — one colour per unique value in ``group_values``.
* **Focused**   — grouped scatter filtered to ``focus_group`` only
  (the implementation simply masks the input before plotting).

Each point's ``customdata`` carries its group label (aggregate mode
uses an empty string).  Callbacks read
``clickData["points"][0]["customdata"]`` to enter focus mode on click.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from ._theme import color_for_index, empty_figure, rade_layout


def pred_actual_scatter(
    predicted:     Sequence[float],
    actual:        Sequence[float],
    *,
    group_values:  Optional[Sequence[str]] = None,
    group_order:   Optional[Iterable[str]] = None,
    focus_group:   Optional[str] = None,
    hover_labels:  Optional[Sequence[str]] = None,
    x_axis_title:  str = "Predicted",
    y_axis_title:  str = "Actual",
) -> go.Figure:
    """Pred-vs-actual scatter with a 45° identity reference line.

    Parameters
    ----------
    predicted, actual
        Equal-length 1-D arrays.
    group_values
        Optional per-point group key — e.g.
        ``["rates", "fx", "rates", ...]``.  ``None`` → aggregate mode.
    group_order
        Optional iterable pinning the legend / colour order.
    focus_group
        When set *and* :paramref:`group_values` is given, only points
        whose group equals ``focus_group`` are rendered.  Identity line
        always stays; the title (when set externally) should flag the
        focus mode to the user.
    hover_labels
        Optional per-point caption shown in the hover tooltip
        (``"Cluster 07"``, ``"2025-11-14 rates swap"``, …).  When
        ``None`` the tooltip falls back to the group label.
    x_axis_title, y_axis_title
        Axis captions.
    """
    pred = np.asarray(list(predicted), dtype=float)
    act  = np.asarray(list(actual),    dtype=float)
    if pred.size == 0 or pred.shape != act.shape:
        return empty_figure("No prediction-actual pairs to plot.")

    # Per-axis tight ranges (5 % pad each side).  Decoupling x and y
    # keeps the data filling the plot box even when predicted and
    # actual span very different magnitudes — at the cost that the
    # identity line is no longer a literal 45° in screen space.  The
    # diagonal still passes through the (0, 0) → (max, max) reference
    # so it remains a useful bias indicator.
    def _padded_range(values: np.ndarray) -> tuple[float, float]:
        v_lo, v_hi = float(np.min(values)), float(np.max(values))
        v_pad = (v_hi - v_lo) * 0.05 if v_hi > v_lo else 1.0
        return v_lo - v_pad, v_hi + v_pad

    span_x = _padded_range(pred)
    span_y = _padded_range(act)
    # Identity line drawn across the union of the two ranges so the
    # reference is visible regardless of which axis dominates.
    identity_span = (
        min(span_x[0], span_y[0]),
        max(span_x[1], span_y[1]),
    )

    grouped = group_values is not None and len(group_values) == pred.size

    fig = go.Figure()

    if not grouped:
        _trace_aggregate(
            fig, pred, act,
            hover_labels=hover_labels,
            color=color_for_index(0),
        )
    else:
        groups = _resolve_group_order(group_values, group_order)
        group_arr = np.asarray(group_values, dtype=object)

        for idx, group in enumerate(groups):
            if focus_group is not None and group != focus_group:
                continue
            mask = group_arr == group
            if not mask.any():
                continue
            labels_here = (
                [hover_labels[i] for i in np.flatnonzero(mask)]
                if hover_labels is not None
                else None
            )
            _trace_one_group(
                fig,
                pred[mask], act[mask],
                group=str(group),
                color=color_for_index(idx),
                hover_labels=labels_here,
            )

    _add_identity_line(fig, identity_span)

    fig.update_layout(
        **rade_layout(
            show_legend=grouped and focus_group is None,
            xaxis={
                "title": {"text": x_axis_title, "font": {"color": "#94a3b8"}},
                "range": list(span_x),
            },
            yaxis={
                "title": {"text": y_axis_title, "font": {"color": "#94a3b8"}},
                "range": list(span_y),
            },
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Trace builders
# ─────────────────────────────────────────────────────────────────────


def _trace_aggregate(
    fig: go.Figure,
    pred: np.ndarray,
    act: np.ndarray,
    *,
    hover_labels: Optional[Sequence[str]],
    color: str,
) -> None:
    customdata = _build_customdata(group="", labels=hover_labels, n=pred.size)
    hovertemplate = (
        "pred: %{x:.4f}<br>actual: %{y:.4f}"
        "<br>%{customdata[1]}<extra></extra>"
        if hover_labels is not None else
        "pred: %{x:.4f}<br>actual: %{y:.4f}<extra></extra>"
    )
    fig.add_trace(
        go.Scattergl(
            x=pred, y=act,
            mode="markers",
            marker={
                "size":    6,
                "color":   color,
                "opacity": 0.7,
                "line":    {"width": 0},
            },
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="All",
            showlegend=False,
        )
    )


def _trace_one_group(
    fig: go.Figure,
    pred: np.ndarray,
    act: np.ndarray,
    *,
    group: str,
    color: str,
    hover_labels: Optional[Sequence[str]],
) -> None:
    customdata = _build_customdata(group=group, labels=hover_labels, n=pred.size)
    if hover_labels is not None:
        hovertemplate = (
            f"<b>{group}</b><br>"
            "pred: %{x:.4f}<br>actual: %{y:.4f}<br>"
            "%{customdata[1]}"
            "<extra>Click to focus</extra>"
        )
    else:
        hovertemplate = (
            f"<b>{group}</b><br>"
            "pred: %{x:.4f}<br>actual: %{y:.4f}"
            "<extra>Click to focus</extra>"
        )
    fig.add_trace(
        go.Scattergl(
            x=pred, y=act,
            mode="markers",
            marker={
                "size":    6,
                "color":   color,
                "opacity": 0.75,
                "line":    {"width": 0},
            },
            name=group,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )


def _add_identity_line(fig: go.Figure, span: tuple[float, float]) -> None:
    fig.add_trace(
        go.Scatter(
            x=list(span), y=list(span),
            mode="lines",
            line={"color": "rgba(148, 163, 184, 0.55)", "dash": "dash", "width": 1},
            hoverinfo="skip",
            showlegend=False,
            name="Identity",
        )
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _build_customdata(
    *,
    group: str,
    labels: Optional[Sequence[str]],
    n: int,
) -> list[list[str]]:
    """2-col customdata: ``[group, extra_label]`` per point.

    Column 0 is always the group label — this is what the click-to-
    focus callback reads.  Column 1 is the optional per-point caption
    surfaced in the hovertemplate (empty string when absent).
    """
    if labels is None:
        return [[group, ""] for _ in range(n)]
    return [[group, str(labels[i]) if labels[i] is not None else ""] for i in range(n)]


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


__all__ = ["pred_actual_scatter"]
