"""Single KPI card — label + value + optional delta chip + optional icon.

Used by the landing overview, governance, monitoring and the "headline
row" of every long page.  Pairs with :class:`Loading` 's ``kpi_strip``
variant so the loading skeleton matches the final shape.

The visual chrome lives in ``rade.css`` (B.1) via ``rade-card-compact``
+ ``rade-kpi-label`` + ``rade-kpi-value`` classes; this module is a
thin composition layer.

Design spec anchors
-------------------
* §6 Components — KPI card is the "compact card" variant.
* §7 Colour tones — deltas use emerald-400 / rose-400 / slate-400.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence

from dash import dcc, html
from dash_iconify import DashIconify


DeltaTone = Literal["positive", "negative", "neutral"]


_DELTA_CLASS: dict[DeltaTone, str] = {
    "positive": "text-emerald-400",
    "negative": "text-rose-400",
    "neutral":  "text-slate-400",
}


# Default colour for the sparkline trace.  Slate-300 reads on the
# dark card background and matches the rest of the typography tones
# without competing with the KPI value.
_SPARKLINE_COLOUR = "#94a3b8"


def _sparkline_figure(
    data: Sequence[float],
    *,
    colour: str = _SPARKLINE_COLOUR,
) -> dict[str, Any]:
    """Build a tiny zero-chrome line trace for the bottom of a KpiCard.

    The figure is intentionally cheap — no markers, no axes, no
    hovers, no padding.  Only purpose is conveying the *shape* of the
    metric distribution across the cluster's trades, so the user can
    spot bimodality / heavy tails at a glance without an extra click
    into the per-trade table.
    """
    if not data:
        return {
            "data": [],
            "layout": {
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor":  "rgba(0,0,0,0)",
                "showlegend":    False,
                "height":        36,
            },
        }
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x":    list(range(len(data))),
                "y":    list(data),
                "line": {"color": colour, "width": 1.4},
                "hoverinfo": "skip",
            }
        ],
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor":  "rgba(0,0,0,0)",
            "showlegend":    False,
            "height":        36,
        },
    }


def KpiCard(
    *,
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_tone: DeltaTone = "neutral",
    icon: Optional[str] = None,
    card_id: Optional[str] = None,
    value_id: Optional[str] = None,
    sparkline_data:    Optional[Sequence[float]] = None,
    sparkline_id:      Optional[str] = None,
    sparkline_colour:  str = _SPARKLINE_COLOUR,
) -> html.Div:
    """A single-KPI card.

    Parameters
    ----------
    label
        Short uppercase caption (e.g. ``"MAE"``, ``"ACTIVE VERSION"``).
    value
        Main figure already formatted as a string (``"12.4 bps"``,
        ``"v2024.09.22"``).  Numeric formatting is the caller's job —
        this component doesn't try to interpret the value.
    delta
        Optional secondary line under the value, e.g. ``"+1.2% vs prior"``.
    delta_tone
        ``"positive"`` (emerald), ``"negative"`` (rose) or
        ``"neutral"`` (slate).  Picks the colour of the ``delta`` text.
    icon
        Optional ``tabler:*`` identifier rendered top-right at 18 px —
        useful on monitoring dashboards to cue metric category.
    card_id
        Optional DOM id for the outer card.  Named ``card_id`` (not
        ``id``) to avoid shadowing the Python built-in — every primitive
        in this package follows the same ``{component}_id`` convention.
    value_id
        Optional DOM id for the value span.  Set this so a callback
        can update the value without re-rendering the whole card.
    sparkline_data
        Optional sequence of numeric values that will be rendered as a
        tiny line chart beneath the value (~36 px tall, no axes / no
        legend).  Useful on dense cluster-metric strips where the
        single KPI number hides distribution shape.  ``None`` (the
        default) hides the sparkline entirely.
    sparkline_id
        Optional DOM id for the sparkline ``dcc.Graph``.  Set this if
        you need a callback to swap the sparkline data without
        re-rendering the whole card.  Ignored when ``sparkline_data``
        is ``None`` and no id-bearing slot is needed.
    sparkline_colour
        Override stroke colour for the sparkline line trace; defaults
        to slate-300 to match the surrounding typography.
    """
    label_row_children: List[Any] = [
        html.Div(label, className="rade-kpi-label"),
    ]
    if icon:
        label_row_children.append(
            DashIconify(icon=icon, width=18, className="text-slate-500")
        )

    # Dash rejects ``id=None`` on component props — include the kwarg
    # only when the caller supplied a real id.
    value_id_kwargs = {"id": value_id} if value_id is not None else {}
    card_id_kwargs = {"id": card_id} if card_id is not None else {}

    body: List[Any] = [
        html.Div(
            className="flex items-center justify-between",
            children=label_row_children,
        ),
        html.Div(
            value,
            className="rade-kpi-value",
            **value_id_kwargs,
        ),
    ]
    if delta:
        body.append(
            html.Div(
                delta,
                className=f"text-xs mt-1 {_DELTA_CLASS[delta_tone]}",
            )
        )

    # Sparkline slot — always emitted when ``sparkline_id`` is given
    # so a render callback can later swap the figure in without
    # re-creating the surrounding card.  When the caller didn't pass
    # an id and didn't pass data either, we omit the slot entirely
    # (no empty 36-px gap on cards that don't want a sparkline).
    if sparkline_data is not None or sparkline_id is not None:
        spark_id_kwargs = (
            {"id": sparkline_id} if sparkline_id is not None else {}
        )
        body.append(
            dcc.Graph(
                figure=_sparkline_figure(
                    sparkline_data or [],
                    colour=sparkline_colour,
                ),
                config={"displayModeBar": False, "staticPlot": True},
                style={"height": "36px"},
                className="rade-kpi-sparkline mt-1",
                **spark_id_kwargs,
            )
        )

    return html.Div(
        className="rade-card-compact flex flex-col gap-1",
        children=body,
        **card_id_kwargs,
    )


__all__ = ["DeltaTone", "KpiCard"]
