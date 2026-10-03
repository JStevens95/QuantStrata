"""Dynamic Cytoscape stylesheet + legend body generators for the
Trade-Graph sub-tab.

The header-band's "Color by" select drives node colouring.  Five modes
are supported:

* ``trade_type``  — target=amber, elementary=violet (the safe default
  that ships in the layout's static stylesheet).
* ``residual``    — gradient from sub-zero teal → over-zero magenta,
  keyed on each node's residual metric.
* ``asset_class`` — categorical palette mapped from the distinct values
  observed in the cluster's nodes.
* ``currency``    — categorical palette.
* ``product``     — categorical palette.

The two helpers in this module produce, for a given (mode, node-list)
pair:

1. :func:`build_stylesheet` — a Cytoscape stylesheet ready to be
   handed to ``dash_cytoscape.Cytoscape.stylesheet``.  Always
   includes the four base selectors (``node``, ``node:selected``,
   ``node[trade_type='target']`` for size, ``edge``) — colour-by
   modes layer additional selectors on top.
2. :func:`build_legend_body` — a Dash component tree that the
   ``_register_render_legend`` callback drops into the legend card's
   ``legend_body`` slot.  Mirrors the active stylesheet so users
   never have to guess what a colour means.

Why a dedicated module
----------------------
The legend body and the stylesheet must agree on every palette,
threshold and label — keeping them next to each other (one builds
the visual rules, the other builds the legend that explains them)
makes drift impossible.  Both consume the same ``nodes_payload``
shape that the callback module stashes in ``store_graph``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dash import html


# ─────────────────────────────────────────────────────────────────────
# Palettes
# ─────────────────────────────────────────────────────────────────────

# Eight-tone categorical palette — colour-blind-friendly tones picked
# from the Tailwind 400 band so they render evenly on the dark theme.
# Cycled if the cluster carries more distinct values than this length.
_CATEGORICAL_PALETTE: Tuple[str, ...] = (
    "#8b5cf6",   # violet-400
    "#f59e0b",   # amber-400
    "#10b981",   # emerald-400
    "#60a5fa",   # blue-400
    "#f472b6",   # pink-400
    "#facc15",   # yellow-400
    "#22d3ee",   # cyan-400
    "#a3e635",   # lime-400
)

# Residual gradient endpoints — teal for negative, slate for ~zero,
# magenta for positive.  Cytoscape supports ``mapData(...)`` linear
# interpolation between two colours; we lay down two selectors so
# the gradient pivots through ~zero.
_RESIDUAL_NEGATIVE_COLOR = "#0ea5e9"   # sky-500
_RESIDUAL_ZERO_COLOR     = "#475569"   # slate-600
_RESIDUAL_POSITIVE_COLOR = "#ec4899"   # pink-500


# Map color-by mode → node-payload column name.  ``None`` for
# ``trade_type`` because that mode reads the ``trade_type`` selector
# directly without needing a numeric / categorical column.
_NODE_COLUMN_FOR_MODE: Dict[str, Optional[str]] = {
    "trade_type":  None,
    "residual":    "residual",
    "asset_class": "asset_class",
    "currency":    "currency_code",
    "product":     "product_code",
}


# ─────────────────────────────────────────────────────────────────────
# Stylesheet builder
# ─────────────────────────────────────────────────────────────────────


def _base_stylesheet() -> List[Dict[str, Any]]:
    """Stylesheet rules that apply regardless of color-by mode.

    These cover sizing, edge rendering and the selection ring.  Mode-
    specific colour rules are layered on top by :func:`build_stylesheet`.
    """
    return [
        {
            "selector": "node",
            "style": {
                "label":            "",
                "width":            12,
                "height":           12,
                "background-color": "#8b5cf6",
                "border-color":     "#0f172a",
                "border-width":     1,
                "transition-property": "background-color, width, height, border-color",
                "transition-duration": "150ms",
            },
        },
        {
            "selector": "node[trade_type = 'target']",
            "style": {"width": 18, "height": 18},
        },
        {
            "selector": "node:selected",
            "style": {
                "border-color":  "#10b981",
                "border-width":  3,
                "width":         22,
                "height":        22,
            },
        },
        {
            "selector": "edge",
            "style": {
                "width":           "mapData(weight, 0, 1, 0.5, 3)",
                "line-color":      "rgba(148, 163, 184, 0.35)",
                "curve-style":     "haystack",
                "haystack-radius": 0.5,
            },
        },
    ]


def _trade_type_rules() -> List[Dict[str, Any]]:
    """Restore the legacy amber / violet trade-type colouring."""
    return [
        {
            "selector": "node",
            "style": {"background-color": "#8b5cf6"},   # elementary
        },
        {
            "selector": "node[trade_type = 'target']",
            "style": {"background-color": "#f59e0b"},   # target
        },
    ]


def _residual_rules(values: Sequence[float]) -> List[Dict[str, Any]]:
    """Two-stop gradient keyed on the per-node ``residual`` data field.

    Cytoscape's ``mapData(field, min, max, color_lo, color_hi)`` does a
    linear interpolation; to get a teal → grey → magenta diverging
    palette we lay down two rules — one for the negative half, one for
    the positive half — split at zero.
    """
    if not values:
        return []

    v_min = float(min(values))
    v_max = float(max(values))

    # If every residual is zero (or NaN-only after filtering), fall
    # back to the trade-type colouring so we never paint the whole
    # graph the same flat tone.
    if v_min == v_max:
        return _trade_type_rules()

    # Anchor the diverging palette at zero so positive and negative
    # residuals are visually distinct.  When the data is one-sided
    # (e.g. all positive), the negative selector simply matches no
    # nodes — Cytoscape silently no-ops.
    return [
        {
            "selector": f"node[residual <= 0]",
            "style": {
                "background-color": (
                    f"mapData(residual, {min(v_min, 0.0)}, 0, "
                    f"{_RESIDUAL_NEGATIVE_COLOR}, {_RESIDUAL_ZERO_COLOR})"
                ),
            },
        },
        {
            "selector": f"node[residual > 0]",
            "style": {
                "background-color": (
                    f"mapData(residual, 0, {max(v_max, 1e-9)}, "
                    f"{_RESIDUAL_ZERO_COLOR}, {_RESIDUAL_POSITIVE_COLOR})"
                ),
            },
        },
    ]


def _categorical_rules(
    values: Sequence[str],
    *,
    column_name: str,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """Per-distinct-value selector → palette colour.

    Returns the rules **and** the (value, colour) pairs the legend
    builder needs — so we walk the unique values once.
    """
    distinct = sorted({v for v in values if v is not None and v != ""})
    pairs: List[Tuple[str, str]] = []
    rules: List[Dict[str, Any]] = []

    for idx, value in enumerate(distinct):
        colour = _CATEGORICAL_PALETTE[idx % len(_CATEGORICAL_PALETTE)]
        pairs.append((value, colour))
        # Cytoscape selectors need single-quoted string values; we
        # escape any embedded apostrophes defensively.
        safe = str(value).replace("'", "\\'")
        rules.append(
            {
                "selector": f"node[{column_name} = '{safe}']",
                "style": {"background-color": colour},
            }
        )

    return rules, pairs


def build_stylesheet(
    color_by: str,
    *,
    nodes_payload: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[List[Tuple[str, str]]]]:
    """Build the Cytoscape stylesheet for the given color-by mode.

    Parameters
    ----------
    color_by
        One of :data:`session.EVALUATION_TRADE_GRAPH_COLOR_BY`.  When
        the mode is unknown (e.g. user hand-edited the store), falls
        back to ``trade_type`` and the legend reflects that.
    nodes_payload
        The same dict payload the render callback stashed in
        ``store_graph["nodes"]`` — each item is
        ``{"data": {"id": ..., "trade_type": ..., "residual": ...,
        "asset_class": ..., "currency_code": ..., "product_code": ...}}``.

    Returns
    -------
    (stylesheet, legend_pairs)
        * ``stylesheet`` — the rule list, ready for
          ``Cytoscape.stylesheet``.
        * ``legend_pairs`` — for the categorical / trade_type modes,
          a list of ``(label, colour)`` tuples the legend builder
          uses to reconstruct the swatches.  ``None`` for
          ``residual`` (the legend renders a gradient bar instead).

    Notes
    -----
    The base sizing / edge rules come first so mode-specific colour
    rules can override the default ``background-color`` simply by
    appearing later in the list (Cytoscape applies rules in order).
    """
    rules = _base_stylesheet()

    if color_by == "residual":
        residuals = [
            float(n.get("data", {}).get("residual"))
            for n in nodes_payload
            if n.get("data", {}).get("residual") is not None
        ]
        rules.extend(_residual_rules(residuals))
        return rules, None

    column = _NODE_COLUMN_FOR_MODE.get(color_by)
    if column is None or color_by == "trade_type":
        # ``trade_type`` mode (default + fallback for unknown mode).
        rules.extend(_trade_type_rules())
        return rules, [("Target", "#f59e0b"), ("Elementary", "#8b5cf6")]

    values = [
        str(n.get("data", {}).get(column))
        for n in nodes_payload
        if n.get("data", {}).get(column) is not None
    ]
    cat_rules, pairs = _categorical_rules(values, column_name=column)
    rules.extend(cat_rules)

    # Edge case — cluster nodes carry no entry under this column
    # (e.g. residual-only metadata).  Fall back to trade-type so we
    # never paint a transparent / broken graph.
    if not pairs:
        rules.extend(_trade_type_rules())
        return rules, [("Target", "#f59e0b"), ("Elementary", "#8b5cf6")]

    return rules, pairs


# ─────────────────────────────────────────────────────────────────────
# Legend body builder
# ─────────────────────────────────────────────────────────────────────


def _swatch(color: str) -> html.Div:
    return html.Div(
        className="w-3 h-3 rounded-full flex-shrink-0",
        style={"backgroundColor": color},
    )


def _legend_row(color: str, label: str, sublabel: Optional[str] = None) -> html.Div:
    children: List[Any] = [_swatch(color)]
    text_block_children: List[Any] = [
        html.Span(label, className="text-xs text-slate-200"),
    ]
    if sublabel:
        text_block_children.append(
            html.Span(sublabel, className="text-[10px] text-slate-500"),
        )
    children.append(
        html.Div(
            className="flex flex-col leading-tight",
            children=text_block_children,
        ),
    )
    return html.Div(
        className="flex items-center gap-2",
        children=children,
    )


def _residual_gradient_bar() -> html.Div:
    """Horizontal gradient bar with min / mid / max tick labels.

    The renderer doesn't know the exact residual extrema (we pass the
    raw bar; the user reads "negative ← → positive" as semantic, not
    quantitative).  A future enhancement can drop the per-cluster
    ``[min, max]`` numbers in here once we surface them through the
    payload's metadata.
    """
    gradient = (
        f"linear-gradient(to right, "
        f"{_RESIDUAL_NEGATIVE_COLOR}, "
        f"{_RESIDUAL_ZERO_COLOR}, "
        f"{_RESIDUAL_POSITIVE_COLOR})"
    )
    return html.Div(
        className="flex flex-col gap-1",
        children=[
            html.Div(
                className="h-2 rounded-full w-full",
                style={"background": gradient},
            ),
            html.Div(
                className="flex justify-between text-[10px] text-slate-500",
                children=[
                    html.Span("Negative"),
                    html.Span("0"),
                    html.Span("Positive"),
                ],
            ),
        ],
    )


def build_legend_body(
    color_by: str,
    *,
    pairs: Optional[List[Tuple[str, str]]],
) -> List[Any]:
    """Build the ``legend_body``'s children for the given mode.

    The legend card is a fixed-height container (140 px); for modes
    with many distinct categorical values, the legend wraps onto a
    second / third row.  We don't paginate or scroll the legend —
    if a cluster has 30 distinct currencies, the user has bigger
    problems than legend layout.

    Parameters
    ----------
    color_by
        One of :data:`session.EVALUATION_TRADE_GRAPH_COLOR_BY`.
    pairs
        From :func:`build_stylesheet`'s return tuple.  ``None`` for
        ``residual`` (gradient bar); a list of ``(label, colour)``
        for ``trade_type`` and the categoricals.
    """
    if color_by == "residual":
        return [_residual_gradient_bar()]

    if not pairs:
        return [
            html.Span(
                "No data for legend.",
                className="text-xs text-slate-500 italic",
            ),
        ]

    # Two-column flex-wrap so up to ~8 categoricals stay legible
    # within the 140-px card without overflowing.  Beyond 8, the
    # palette cycles (callers see this in `_categorical_rules`) and
    # the swatch repeats — visually the user understands "more
    # categories than colours".
    return [
        html.Div(
            className="grid grid-cols-2 gap-x-3 gap-y-1",
            children=[_legend_row(color, label) for label, color in pairs],
        ),
    ]


__all__ = [
    "build_legend_body",
    "build_stylesheet",
]
