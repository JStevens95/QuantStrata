"""Card chrome around a Plotly ``dcc.Graph``.

Every chart in the app — portfolio P&L, residual density, convergence
curves, etc. — goes through this wrapper so headers, toolbar position,
card spacing and Plotly modebar config never drift.

Design spec anchors
-------------------
* §7 Chart defaults — hide Plotly logo, strip lasso/select buttons,
  Inter as the default font.
* §9 State handling — pair with :func:`Loading` ``variant="chart"`` so
  the skeleton card has the same dimensions as the real container.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import dcc, html


# Plotly modebar buttons to strip from every chart.  Keeps the toolbar
# focused on "save image + reset view" for most use-cases; callers can
# override via ``config=``.
_DEFAULT_GRAPH_CONFIG: Dict[str, Any] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "toggleSpikelines",
    ],
    "responsive": True,
}


def ChartContainer(
    *,
    title: str,
    graph_id: str,
    figure: Optional[Any] = None,
    subtitle: Optional[str] = None,
    height: int = 320,
    actions: Optional[List[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    className: str = "",
    container_id: Optional[str] = None,
    style: Optional[Dict[str, Any]] = None,
) -> html.Div:
    """Return a card-wrapped Plotly chart with title strip + toolbar slot.

    Parameters
    ----------
    title
        Main headline shown at the top-left of the card.
    graph_id
        DOM id of the inner ``dcc.Graph``.  Callbacks target this id
        to inject figures — not the outer card id.
    figure
        Optional initial Plotly figure.  Defaults to an empty dict so
        the graph renders a blank frame until a callback fires.
    subtitle
        Optional secondary caption under the title (e.g. date range).
    height
        Inner graph height in pixels.  Default 320 matches the design
        spec's standard chart height.
    actions
        Extra components rendered top-right of the card header, e.g.
        segmented controls, download buttons.  Each item is passed
        through ``dmc.Group(gap="xs", children=actions)``.
    config
        Overrides merged on top of the default Plotly config.  Pass
        this to enable e.g. ``"scrollZoom": True`` on a time-series.
    className
        Extra Tailwind classes appended to the outer card.
    container_id
        Optional DOM id on the outer card (useful for scroll anchors).
        Named ``container_id`` (not ``id``) to avoid shadowing the
        Python built-in — every primitive in this package follows the
        same ``{component}_id`` convention.  ``graph_id`` is the inner
        ``dcc.Graph``'s id and is required separately.
    style
        Optional inline-style dict applied to the outer card.  The
        Cluster Deep-Dive page uses this to mount the elementary-PnL
        chart card with ``style={"display": "none"}`` so the empty-
        state placeholder shows by default — a callback then flips
        it visible once the user picks ≥ 1 elementary trade.  Avoids
        wrapping the card in a redundant ``html.Div`` just to carry
        a style toggle.
    """
    header_left = [
        html.Div(title, className="text-sm font-semibold text-slate-200"),
    ]
    if subtitle:
        header_left.append(
            html.Div(subtitle, className="text-xs text-slate-500")
        )

    header = dmc.Group(
        justify="space-between",
        align="center",
        children=[
            html.Div(className="flex flex-col", children=header_left),
            dmc.Group(gap="xs", children=actions or []),
        ],
    )

    merged_config = {**_DEFAULT_GRAPH_CONFIG, **(config or {})}

    graph = dcc.Graph(
        id=graph_id,
        figure=figure if figure is not None else {},
        style={"height": f"{height}px"},
        config=merged_config,
    )

    # Dash rejects ``id=None`` and treats ``style=None`` differently
    # from omission — only include each kwarg when the caller set it.
    id_kwargs = {"id": container_id} if container_id is not None else {}
    style_kwargs = {"style": style} if style is not None else {}

    return html.Div(
        className=f"rade-card flex flex-col gap-3 {className}".strip(),
        children=[header, graph],
        **id_kwargs,
        **style_kwargs,
    )


__all__ = ["ChartContainer"]
