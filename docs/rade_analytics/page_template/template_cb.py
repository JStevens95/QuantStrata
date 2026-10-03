"""TEMPLATE — Callback module skeleton for a new Rade Analytics page.

DO NOT IMPORT THIS FILE.  Copy it to
``src/ui/apps/rade_analytics/callbacks/<your_page>_cb.py`` and replace
every ``# TODO:`` marker.  Pair with ``template_layout.py`` from the
same folder.  See ``docs/rade_analytics/page_template/README.md`` for
the 15-minute add-a-page workflow.

Page Contract structure
-----------------------
The public surface is a single :func:`register` that delegates to two
section helpers, matching Page Contract §2 (capture / render split):

* :func:`_register_capture` — user-input gestures → :class:`Session`
  writes (no UI side-effects, no backend access).
* :func:`_register_render`  — state → DOM updates (no Session writes,
  except for the narrow ``_register_bootstrap`` capture-edge that
  handles URL deep-links + fresh-user defaults).

Capture (1 callback in this template)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``_sync_context`` — context-picker ``value`` → session (the field
  your page persists, e.g. ``deep_dive_cluster_id``).

Render (2 callbacks in this template)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``_bootstrap`` — mount_signal-triggered fetch of the context picker
  ``Select.data`` option list (version-keyed metadata that genuinely
  needs a backend round-trip).  Also handles two narrow override paths
  in the same return tuple — URL deep-link, fresh-user default — so
  the bootstrap is the single capture-edge in the render section.
* ``_render_main`` — session → KPI value + main chart figure.

Initial UI state (no value-side hydration)
------------------------------------------
The context picker's ``value`` prop is seeded from session at layout
build time (Page Contract §3 Rule L1) — see ``template_layout.py``'s
``build_template(*, session)``.  This module never writes
``context_select.value`` except in the bootstrap's two narrow override
cases.  Eliminating value-side hydration means the page paints the
user's previously-chosen state on first frame rather than after a
callback round-trip.

Why pathname-gating
-------------------
Page Contract §4 Rule C2 — every render callback gates on
``pathname == _<PAGE>_PATH`` to avoid wasted compute on cross-page
re-renders.  Cheap, idempotent, prevents stale-DOM warnings.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

# TODO: replace template_layout with the layout module you cp'd from
# the template (the import path will be ``..layouts.<your_page>``).
from ..layouts.template_layout import TEMPLATE_IDS
from ..layouts.shell import SHELL_IDS

from ..data.result_helpers import figure_with_fallback
from ..data.session import Session

# TODO: import the figure helpers your render callback needs.  Examples:
#   from ..figures import portfolio_pnl, error_over_time
# from ..figures import empty_figure  # only if you need it outside the helpers

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants — page route + initial values
# ─────────────────────────────────────────────────────────────────────

# TODO: set this to the route the router maps to your page (e.g.
# ``/evaluation/cross-cluster``, ``/data-quality``).  Every render
# callback gates on ``pathname == _TEMPLATE_PATH``.
_TEMPLATE_PATH = "/template"

# TODO: replace with your page's domain-appropriate empty placeholder
# (e.g. ``"—"`` for KPI values, ``[]`` for AgGrid rowData).
_PLACEHOLDER = "—"


# ═════════════════════════════════════════════════════════════════════
# Public surface
# ═════════════════════════════════════════════════════════════════════


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every callback for this page to ``app``.

    The two section helpers are the only top-level symbols a reader
    should need to scan to understand the page's wiring.

    Parameters
    ----------
    app
        The Dash app returned by :func:`rade_analytics.app.create_app`.
    backend
        Shared :class:`RadeBackend` — all data fetches go through here.
    """
    _register_capture(app)
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# Section dispatchers — capture / render split (Page Contract §2)
# ─────────────────────────────────────────────────────────────────────


def _register_capture(app: "Dash") -> None:
    """Attach capture-side callbacks: input gestures → session writes.

    Capture callbacks are forbidden from doing UI rendering and from
    touching ``backend`` (Page Contract §4 Rule C1).  They write only
    to the session-store; render callbacks pick up the writes via the
    session-store ``Input``.
    """
    _register_sync_context(app)


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Attach render-side callbacks: state → DOM, no session writes.

    Render callbacks consume URL + session-store as Inputs / States,
    do backend lookups via ``backend``, and emit values + figures into
    the page's components.  They never write to the session-store
    (which would cascade into other pages' render callbacks).

    The single exception is the bootstrap callback's two narrow
    override paths — URL deep-link, fresh-user default — which write
    both the ``Select.value`` and the session-store in the same return
    tuple to avoid a second capture round-trip.
    """
    _register_bootstrap(app, backend)
    _register_render_main(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. Capture — context picker → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_context(app: "Dash") -> None:
    """Persist the context picker's ``value`` into session."""

    @app.callback(
        Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
        Input(TEMPLATE_IDS["context_select"], "value"),
        State(SHELL_IDS["session_store"], "data"),
        # Page Contract §4 Rule C5 — capture callbacks default to
        # prevent_initial_call=True; we only want this to fire on a
        # genuine user pick, not on the fresh page paint.
        prevent_initial_call=True,
    )
    def _sync(
        new_context_id: Optional[str],
        session_data:   Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        # TODO: replace with the field on EvaluationFilters / Session
        # your page persists.  Example:
        #   session.evaluation.deep_dive_cluster_id = new_context_id
        if new_context_id is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        # session.evaluation.<your_field> = new_context_id  # TODO
        return session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 2. Render — bootstrap (mount_signal → option list + override edges)
# ═════════════════════════════════════════════════════════════════════


def _register_bootstrap(app: "Dash", backend: "RadeBackend") -> None:
    """Fetch the context picker's option list once per fresh mount.

    The mount tripwire pattern (Page Contract §3 Rule L4) — this
    callback is the *only* render callback in the module that's
    allowed to write to the session-store, and only along two narrow
    edges:

    * **URL deep-link** — when ``?ctx=<id>`` differs from
      ``session.<field>``, copy URL → session + ``Select.value``.
    * **Fresh-user default** — when neither URL nor session has a
      value, pick the first option from the fetched list, write it to
      both ``Select.value`` and session.

    Both edges are handled in the same return tuple so the bootstrap
    is the single capture-edge in the render section; we avoid a
    second round-trip for these initial-state cases.
    """

    @app.callback(
        Output(TEMPLATE_IDS["context_select"],   "data"),
        Output(TEMPLATE_IDS["context_select"],   "value"),
        Output(SHELL_IDS["session_store"],       "data", allow_duplicate=True),
        Input(TEMPLATE_IDS["mount_signal"],      "data"),
        State(SHELL_IDS["url"],                  "search"),
        State(SHELL_IDS["session_store"],        "data"),
        # Page Contract §4 Rule C5 — explicit ``initial_duplicate``
        # because we share an Output with the capture callback.  The
        # mount_signal Input only fires once per fresh mount, so we
        # pay this cost exactly when we need the bootstrap to run.
        prevent_initial_call="initial_duplicate",
    )
    def _bootstrap(
        _trigger:     Any,
        url_search:   Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Any, Any]:
        # TODO: backend lookup.  The fetch should be cheap (cache hit
        # after the first call) and version-keyed if your option list
        # depends on the active ensemble version.
        #
        # Example:
        #   res = backend.context_options()
        #   if not res.ok or not res.data:
        #       return [], no_update, no_update
        #   options = [{"value": c.id, "label": c.name} for c in res.data]
        options: List[Dict[str, str]] = []  # TODO

        session = Session.from_store(session_data)
        # TODO: replace with the field your page persists.
        session_value: Optional[str] = None  # session.evaluation.<your_field>

        # Override-edge 1 — URL deep-link.  TODO: wire up if your page
        # supports ``?ctx=<id>`` style deep-links.  Otherwise drop.
        url_value: Optional[str] = None
        # if url_search:
        #     parsed = parse_qs(url_search.lstrip("?"))
        #     url_value = (parsed.get("ctx") or [None])[0]

        if url_value and url_value != session_value:
            # session.evaluation.<your_field> = url_value  # TODO
            return options, url_value, session.to_store()

        # Override-edge 2 — fresh-user default.
        if not session_value and options:
            default_value = options[0]["value"]
            # session.evaluation.<your_field> = default_value  # TODO
            return options, default_value, session.to_store()

        # Steady state — just publish the option list, leave value +
        # session alone.
        return options, no_update, no_update


# ═════════════════════════════════════════════════════════════════════
# 3. Render — main KPI + chart
# ═════════════════════════════════════════════════════════════════════


def _register_render_main(app: "Dash", backend: "RadeBackend") -> None:
    """Render the headline KPI value + the time-series chart.

    Demonstrates:

    * **Mount-tripwire trigger** — Page Contract §4 Rule C7.
      ``Input(mount_signal, "data")`` is the *only* trigger that's
      guaranteed to fire after the layout chunk has been mounted by
      the router; ``Input(pathname)`` would race the router's
      content swap and write to outputs whose IDs aren't in the DOM
      yet (Anti-pattern A8).
    * **Pathname gate** — Page Contract §4 Rule C4. The URL is a
      ``State``, never an ``Input``: it gates the callback, never
      triggers it.
    * **Tri-state via ``figure_with_fallback``** — Page Contract §5.1.
    * **uirevision keyed on the data domain** — Page Contract §6.
    """

    @app.callback(
        Output(TEMPLATE_IDS["kpi_value"],   "children"),
        Output(TEMPLATE_IDS["main_chart"],  "figure"),
        # Trigger Inputs — Rule C7.  ``mount_signal`` lives inside the
        # layout chunk the router swaps in, so it can't race the swap;
        # ``session_store`` re-fires when capture callbacks write.
        Input(TEMPLATE_IDS["mount_signal"], "data"),
        Input(SHELL_IDS["session_store"],   "data"),
        # Gate State — Rule C4.  Pathname is *never* an Input.
        State(SHELL_IDS["url"],             "pathname"),
        # Rule C5 — render callbacks driven by ``mount_signal`` use
        # ``"initial_duplicate"`` so the mount fires the initial call.
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:       Any,
        session_data: Optional[Dict[str, Any]],
        pathname:     Optional[str],
    ) -> Tuple[Any, Any]:
        if pathname != _TEMPLATE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        split = session.split

        # TODO: read the page's persisted context id from session.
        context_id: Optional[str] = None  # session.evaluation.<your_field>

        if not context_id:
            # Pre-fetch guard — no BackendResult to classify yet.
            from ..figures import empty_figure
            return _PLACEHOLDER, empty_figure(
                "Pick a context above to see this page's metrics."
            )

        # TODO: backend lookup.  Returns a BackendResult[<DataFrame or DTO>].
        #   res = backend.<your_method>(split, context_id=context_id)
        from ..data.backend import BackendResult
        import pandas as pd  # TODO: drop if you don't need the placeholder
        res: BackendResult[pd.DataFrame] = BackendResult.success(pd.DataFrame())

        # KPI value — demonstrates the manual tri-state branch when
        # the helper doesn't fit (multiple Outputs of different types).
        if not res.ok:
            kpi_text = _PLACEHOLDER
        elif res.data is None or res.data.empty:
            kpi_text = _PLACEHOLDER
        else:
            # TODO: compute the headline metric from res.data.
            kpi_text = "0.0000"

        # Figure — uses figure_with_fallback to collapse tri-state.
        # TODO: replace the on_ok lambda with your figure helper, e.g.
        #   on_ok=lambda df: portfolio_pnl(df, uirevision_key=split),
        ui_key = f"{split}::{context_id}"
        from ..figures import empty_figure  # TODO: drop after you wire on_ok
        fig = figure_with_fallback(
            res,
            on_ok=lambda df: empty_figure(  # TODO: replace
                "TODO: render your time-series figure here."
            ).update_layout(uirevision=ui_key) or empty_figure("TODO"),
            empty_msg="No data for the active selection.",
        )

        return kpi_text, fig
