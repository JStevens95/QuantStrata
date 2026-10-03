"""Evaluation page — global filter bar primitive.

Renders as:

    ┌─ collapsed (default) ───────────────────────────────────────┐
    │ [🔍 Filters]    2 active  [Desk: EUR ×] [CCY: USD EUR ×]   │
    │                                              [Clear all]   │
    ├─ expanded (user clicked Filters) ──────────────────────────┤
    │ Asset Class ▾  CCY ▾  Desk ▾  Product ▾  Date ▾   [Reset] │
    └─────────────────────────────────────────────────────────────┘

Design contract
---------------
* Filters are the **single global WHERE clause** for every Evaluation
  sub-tab.  State lives in :class:`EvaluationFilters` on the session;
  this primitive is pure layout.
* **Initial values come from session at build time** (Page Contract
  §3 Rule L1) — the builder takes ``initial_filters`` /
  ``initial_open`` and bakes them into every component's ``value`` /
  ``opened`` prop, so the page is fully usable before any callback
  fires.  No hydration callback exists.
* Chips reflect the live input values.  A clientside callback in
  ``assets/js/evaluation.js::update_filter_ui`` re-renders them
  whenever a value changes — no server round-trip.  This module
  provides :func:`render_filter_chips` for the **initial** server-side
  render only; the two paths produce DOM with identical structure /
  classes so CSS handles all visual styling.
* Dropdown ``data`` is populated by a bootstrap callback once per
  version (fetched from the metadata API in E.1+).  The skeleton
  renders with empty ``data=[]`` so the shell is testable today.

Design spec anchors
-------------------
* §6 Components — MultiSelect + DatePickerInput + chip pill.
* §9 State handling — filters persist across sub-tabs.
* §10 Accessibility — toggle button has ``aria-expanded`` / ``aria-controls``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

from ..data.session import EvaluationFilters


# Every dynamic id the filter bar exposes.  Callbacks in
# ``callbacks/evaluation_cb.py`` import this dict rather than hardcoding
# strings.  Keep in sync with the skeleton's CSS classes in rade.css.
EVAL_FILTER_IDS = {
    # Top-level containers — useful as selector targets for tests +
    # scroll anchors.
    "root":         "eval-filter-bar-root",

    # Row 1: toggle + chip strip + clear-all.
    "toggle_btn":   "eval-filter-toggle-btn",
    "toggle_label": "eval-filter-toggle-label",     # "2 active" | hidden
    "chips":        "eval-filter-chips",             # html.Div holding chip list
    "clear_all":    "eval-filter-clear-all",

    # Row 2: dmc.Collapse wrapping the dropdown row.
    "collapse":     "eval-filter-collapse",
    "asset_class":  "eval-filter-asset-class",
    "currency":     "eval-filter-currency",
    "desk":         "eval-filter-desk",
    "product":      "eval-filter-product",
    "date_range":   "eval-filter-date-range",
    "reset_btn":    "eval-filter-reset-btn",
}


# Human-readable labels for the chip strip.  Keep the casing compact —
# chips render at xs size.  **Mirror this dict in
# assets/js/evaluation.js::CHIP_LABELS**; both code paths use it.
_CHIP_LABELS = {
    "asset_class": "Asset",
    "currency":    "CCY",
    "desk":        "Desk",
    "product":     "Product",
    "date":        "Date",
}


# Visibility styles for the Clear-all button.  Constants so callbacks +
# server-side initial render read by intent rather than inlining.
_CLEAR_ALL_VISIBLE: Dict[str, Any] = {}                  # default (visible)
_CLEAR_ALL_HIDDEN:  Dict[str, Any] = {"display": "none"}


# ─────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────


# Hidden sentinel option, used when no real options have been fetched
# yet.  Mantine v7's MultiSelect crashes its render path when ``data``
# is an empty array (an internal ``.map(...)`` over derived state on
# first mount); keeping at least one disabled item in the list avoids
# that path entirely.  The ``disabled`` flag means the user can't pick
# it, and matching CSS in ``rade.css`` (``[data-value="__rade_no_options__"]``)
# hides it from the dropdown so the user sees a clean empty state.
# A bootstrap callback (Phase E.1) replaces the whole array with real
# options once they're fetched from the metadata endpoint.
_MS_SENTINEL_VALUE = "__rade_no_options__"
_MS_DATA_PLACEHOLDER: List[Dict[str, Any]] = [
    {"value": _MS_SENTINEL_VALUE, "label": "—", "disabled": True},
]


def _safe_options(
    options: Optional[List[Dict[str, str]]],
) -> List[Dict[str, Any]]:
    """Return ``options`` if non-empty, else a single hidden sentinel.

    See :data:`_MS_DATA_PLACEHOLDER` for the rationale — empty data
    arrays trigger a render crash in Mantine v7's MultiSelect.
    """
    if options:
        return list(options)
    return list(_MS_DATA_PLACEHOLDER)


def _has_real_options(
    options: Optional[List[Dict[str, str]]],
) -> bool:
    """Whether ``options`` contains any real (non-sentinel) entries."""
    return bool(options)


def _ms_state_props(
    options: Optional[List[Dict[str, str]]],
) -> Dict[str, Any]:
    """State-dependent MultiSelect props.

    When ``options`` is empty (sentinel-only data), the dropdown is
    visually disabled with a "Loading…" placeholder so it reads as
    "not ready yet" rather than "broken / no matches".  Once the
    bootstrap callback writes real options into ``data``, the
    *callback* is responsible for flipping ``disabled=False`` and
    restoring the regular placeholder — this helper only governs the
    initial server-side render.
    """
    if _has_real_options(options):
        return {"placeholder": "Any", "disabled": False}
    return {"placeholder": "Loading…", "disabled": True}


def build_evaluation_filter_bar(
    *,
    initial_filters: Optional[EvaluationFilters] = None,
    initial_open: bool = False,
    asset_class_options: Optional[List[Dict[str, str]]] = None,
    currency_options:    Optional[List[Dict[str, str]]] = None,
    desk_options:        Optional[List[Dict[str, str]]] = None,
    product_options:     Optional[List[Dict[str, str]]] = None,
) -> html.Div:
    """Return the complete filter-bar tree.

    Parameters
    ----------
    initial_filters
        Filter values already held in session — baked into every
        ``value`` prop at build time so the user sees their previous
        selections without a hydration callback.  ``None`` is treated
        as "no filters" (the default state on a fresh session).
    initial_open
        Whether the dropdown drawer starts expanded.  Sourced from
        ``session.evaluation.filter_bar_open`` so the user's previous
        preference survives navigation.
    asset_class_options, currency_options, desk_options, product_options
        Optional pre-populated dropdown options.  A bootstrap callback
        normally fills these once per version; passing them in keeps
        the preview script self-contained.
    """
    f = initial_filters or EvaluationFilters()
    initial_chips = render_filter_chips(f.to_dict())
    initial_label = _count_label(f.active_chip_count())
    initial_clear_style = _CLEAR_ALL_VISIBLE if not f.is_empty() else _CLEAR_ALL_HIDDEN
    initial_date_value = (
        [f.date_from, f.date_to]
        if (f.date_from or f.date_to)
        else [None, None]
    )

    return html.Div(
        id=EVAL_FILTER_IDS["root"],
        className="rade-filter-bar",
        children=[
            _row_one(
                initial_chips=initial_chips,
                initial_label=initial_label,
                initial_clear_style=initial_clear_style,
                initial_open=initial_open,
            ),
            dmc.Collapse(
                id=EVAL_FILTER_IDS["collapse"],
                opened=initial_open,
                children=_row_two(
                    initial_filters=f,
                    initial_date_value=initial_date_value,
                    asset_class_options=_safe_options(asset_class_options),
                    currency_options=_safe_options(currency_options),
                    desk_options=_safe_options(desk_options),
                    product_options=_safe_options(product_options),
                ),
            ),
        ],
    )


def render_filter_chips(filters_dict: Dict[str, Any]) -> List[Any]:
    """Render the chip list for the given filter snapshot.

    ``filters_dict`` is the :meth:`EvaluationFilters.to_dict` payload —
    callbacks import this helper rather than re-implementing chip
    rendering on the server side.

    The returned tree **must match** the structure produced by
    ``assets/js/evaluation.js::makeChip`` so the initial server-side
    render and the post-mount clientside updates are visually
    identical.  Each chip is::

        <div class="rade-filter-chip">
          <span class="rade-filter-chip-label">{label}: {values}</span>
          <button id={"type":"eval-filter-chip-close","dimension":dim}
                  class="rade-filter-chip-close"
                  aria-label="Remove {dim} filter">×</button>
        </div>
    """
    chips: List[Any] = []

    for dim in ("asset_class", "currency", "desk", "product"):
        values = filters_dict.get(dim) or []
        if not values:
            continue
        label = _CHIP_LABELS[dim]
        value_text = ", ".join(str(v) for v in values[:3])
        if len(values) > 3:
            value_text += f" +{len(values) - 3}"
        chips.append(_chip(dim, f"{label}: {value_text}"))

    date_from = filters_dict.get("date_from")
    date_to   = filters_dict.get("date_to")
    if date_from or date_to:
        rng = "\u2013".join(filter(None, [date_from, date_to])) or "any"
        chips.append(_chip("date", f"Date: {rng}"))

    return chips


# ─────────────────────────────────────────────────────────────────────
# Row builders — private
# ─────────────────────────────────────────────────────────────────────


def _row_one(
    *,
    initial_chips: List[Any],
    initial_label: str,
    initial_clear_style: Dict[str, Any],
    initial_open: bool,
) -> html.Div:
    """Slim row: toggle button, chip strip, clear-all button.

    Initial children come from the caller (``build_evaluation_filter_bar``
    derives them from ``initial_filters``) so the server-side render is
    already correct before any callback fires.
    """
    return html.Div(
        className="rade-filter-row rade-filter-row--one",
        children=[
            html.Div(
                className="rade-filter-left",
                children=[
                    dmc.Button(
                        id=EVAL_FILTER_IDS["toggle_btn"],
                        children="Filters",
                        leftSection=DashIconify(icon="tabler:filter", width=14),
                        rightSection=DashIconify(icon="tabler:chevron-down", width=12),
                        size="xs",
                        variant="default",
                        radius="sm",
                        **{
                            "aria-expanded": "true" if initial_open else "false",
                            "aria-controls": EVAL_FILTER_IDS["collapse"],
                        },
                    ),
                    html.Span(
                        id=EVAL_FILTER_IDS["toggle_label"],
                        className="rade-filter-count",
                        children=initial_label,
                    ),
                ],
            ),

            html.Div(
                className="rade-filter-right",
                children=[
                    html.Div(
                        id=EVAL_FILTER_IDS["chips"],
                        className="rade-filter-chips",
                        children=initial_chips,
                    ),
                    dmc.Button(
                        id=EVAL_FILTER_IDS["clear_all"],
                        children="Clear all",
                        leftSection=DashIconify(icon="tabler:x", width=12),
                        size="xs",
                        variant="subtle",
                        color="gray",
                        style=initial_clear_style,
                    ),
                ],
            ),
        ],
    )


def _row_two(
    *,
    initial_filters: EvaluationFilters,
    initial_date_value: List[Optional[str]],
    asset_class_options: List[Dict[str, str]],
    currency_options:    List[Dict[str, str]],
    desk_options:        List[Dict[str, str]],
    product_options:     List[Dict[str, str]],
) -> html.Div:
    """Expanded drawer row: five dropdowns + a reset button.

    Each control's ``value`` is initialised from ``initial_filters`` so
    the user sees their session-held selections on first paint — no
    post-mount hydration callback needed.

    Until a Phase E.1 bootstrap callback fetches real ``data=`` from
    the metadata endpoint, the four MultiSelects are seeded with a
    single hidden sentinel option (see :func:`_safe_options`) and the
    helper :func:`_ms_props` overrides ``placeholder``/``disabled`` so
    the user sees a clearly-disabled control rather than a clickable
    dropdown that opens to nothing.
    """
    # Bare-minimum MultiSelect props that work on the empty-data
    # initial render.  The advanced behaviours (search, picked-options
    # hiding, nothing-found messaging) re-enter the codebase via the
    # Phase E.1 bootstrap callback once real options are available.
    ms_defaults = dict(
        size="xs",
        radius="sm",
        clearable=True,
        className="rade-filter-dropdown",
    )

    return html.Div(
        className="rade-filter-row rade-filter-row--two",
        children=[
            dmc.MultiSelect(
                id=EVAL_FILTER_IDS["asset_class"],
                label="Asset Class",
                data=asset_class_options,
                value=list(initial_filters.asset_class),
                **_ms_state_props(asset_class_options),
                **ms_defaults,
            ),
            dmc.MultiSelect(
                id=EVAL_FILTER_IDS["currency"],
                label="Currency",
                data=currency_options,
                value=list(initial_filters.currency),
                **_ms_state_props(currency_options),
                **ms_defaults,
            ),
            dmc.MultiSelect(
                id=EVAL_FILTER_IDS["desk"],
                label="Desk",
                data=desk_options,
                value=list(initial_filters.desk),
                **_ms_state_props(desk_options),
                **ms_defaults,
            ),
            dmc.MultiSelect(
                id=EVAL_FILTER_IDS["product"],
                label="Product",
                data=product_options,
                value=list(initial_filters.product),
                **_ms_state_props(product_options),
                **ms_defaults,
            ),
            dmc.DatePickerInput(
                id=EVAL_FILTER_IDS["date_range"],
                label="Date",
                placeholder="Any",
                type="range",
                allowSingleDateInRange=True,
                value=initial_date_value,
                size="xs",
                radius="sm",
                className="rade-filter-dropdown",
                leftSection=DashIconify(icon="tabler:calendar", width=14),
            ),

            html.Div(
                className="rade-filter-reset-wrap",
                children=dmc.Button(
                    id=EVAL_FILTER_IDS["reset_btn"],
                    children="Reset",
                    leftSection=DashIconify(icon="tabler:refresh", width=14),
                    size="xs",
                    variant="light",
                    color="gray",
                ),
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Internal primitives
# ─────────────────────────────────────────────────────────────────────


def _chip(dimension: str, text: str) -> html.Div:
    """One filter chip — plain HTML so the initial server render and
    the clientside ``update_filter_ui`` callback produce identical
    DOM trees.  Visual styling lives in rade.css under
    ``.rade-filter-chip*``.

    Each chip carries a pattern-matching id on its close button so a
    single callback can handle removal for every dimension::

        Input({"type": "eval-filter-chip-close", "dimension": ALL},
              "n_clicks")

    See ``callbacks/evaluation_cb.py`` for the matching capture
    callback.
    """
    return html.Div(
        className="rade-filter-chip",
        children=[
            html.Span(
                className="rade-filter-chip-label",
                children=text,
            ),
            html.Button(
                "\u00D7",  # multiplication-sign ×
                id={"type": "eval-filter-chip-close", "dimension": dimension},
                className="rade-filter-chip-close",
                **{"aria-label": f"Remove {dimension} filter"},
            ),
        ],
    )


def _count_label(n: int) -> str:
    """Render the toggle button's "n active" sidecar label.

    Mirrors ``assets/js/evaluation.js`` so the initial server render
    and the clientside update produce the same string.
    """
    if n <= 0:
        return ""
    return f"{n} active"


__all__ = [
    "EVAL_FILTER_IDS",
    "build_evaluation_filter_bar",
    "render_filter_chips",
]
