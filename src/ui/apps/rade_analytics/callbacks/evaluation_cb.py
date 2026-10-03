"""Evaluation page callbacks.

Page Contract reference
-----------------------
This module is the **reference implementation** of the contract in
``docs/rade_analytics/page_contract.md``.  It is split into four
clearly-named groups so each rule is auditable in isolation:

* :func:`_register_routing`   — URL ↔ ``dmc.Tabs`` ↔ content slot.
* :func:`_register_capture`   — user input → :class:`Session` writes.
                                 Only ever writes ``session-store.data``
                                 (always-mounted, root layout) and the
                                 three input ``value`` props (chip × /
                                 clear-all clear).  No render output
                                 that can unmount mid-callback, so the
                                 "nonexistent object in Output" warning
                                 this module historically produced is
                                 structurally impossible.
* :func:`_register_render`    — input values → chips + label +
                                 clear-all visibility.  Implemented as
                                 a **clientside callback** (Lever P2) so
                                 there's zero server round-trip for
                                 trivial UI derivation, and any
                                 transient unmounts during page
                                 transitions are silently absorbed.
* :func:`_register_bootstrap` — first-mount metadata → MultiSelect
                                 ``data`` / ``disabled`` / ``placeholder``
                                 props.  Fires when the user enters
                                 ``/evaluation/*`` and replaces the
                                 layout's sentinel placeholder data with
                                 the real cluster-attribute values for
                                 the active ensemble version.

What this module deliberately does **not** do
---------------------------------------------
* No URL-driven *hydration* callback.  Initial filter values are
  baked into the layout by :func:`build_evaluation_filter_bar`
  reading ``session.evaluation.filters`` at build time
  (Rule L1).  Eliminating the hydration callback removes the
  pathname → input.value → render chain that produced both the
  "nonexistent object" warning and the visible flicker on
  Evaluation entry.
* No server-side chip rendering after first paint.  The clientside
  ``update_filter_ui`` function in ``assets/js/evaluation.js``
  re-derives the chip list, "n active" label, and clear-all
  visibility every time a filter value changes.
* No layout-time metadata fetch.  Layouts must remain pure (Page
  Contract §3 Rule L1, §1 layer separation: layouts cannot import
  ``data/backend``).  The bootstrap callback below is the layered
  way to inject backend-derived dropdown options into the rendered
  filter bar.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from dash import ALL, ClientsideFunction, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from ..components.evaluation_filter_bar import EVAL_FILTER_IDS
from ..data.session import EvaluationFilters, Session
from ..layouts.evaluation.shell import (
    EVALUATION_IDS,
    active_subtab_from_path,
    build_subtab_content,
    path_for_subtab,
)
from ..layouts.shell import SHELL_IDS


if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Evaluation callback to ``app``.

    ``backend`` is the only data dependency — the routing / capture
    / clientside-render groups are pure UI plumbing and do not touch
    the server, so only the bootstrap group gets a reference to it.
    """
    _register_routing(app)
    _register_capture(app)
    _register_render(app)
    _register_bootstrap(app, backend)


# ─────────────────────────────────────────────────────────────────────
# 1. Routing — URL ↔ dmc.Tabs ↔ content slot
# ─────────────────────────────────────────────────────────────────────


def _register_routing(app: "Dash") -> None:
    """URL → tab / content + tab click → URL.

    The router (``..router._route``) rebuilds the entire Evaluation
    tree on cross-top-level navigation, so this pair only fires for
    *within-Evaluation* sub-tab transitions.  Both callbacks bail with
    :class:`PreventUpdate` when the URL is outside ``/evaluation/*``.
    """

    # ── URL → tab value + content slot ───────────────────────────
    @app.callback(
        Output(EVALUATION_IDS["tabs"],    "value"),
        Output(EVALUATION_IDS["content"], "children"),
        Input(SHELL_IDS["url"], "pathname"),
        # Page Contract §3 Rule L1 — session is read as State (not
        # Input) so that filter-bar / split-toggle writes don't
        # rebuild the entire sub-tab content tree.  We only re-mount
        # on URL change; within-tab reactivity is each sub-tab's
        # render callback's responsibility.
        State(SHELL_IDS["session_store"], "data"),
        # Page Contract §4 Rule C5 — explicit opt-in.  When the user
        # lands directly on ``/evaluation/<sub>`` (URL share, browser
        # refresh) the router builds the chrome but the Tabs default
        # to the layout's seeded ``value``.  We rely on the implicit
        # initial pathname callback to flip the tab to the URL's
        # actual sub-tab and inflate ``content``.  The pathname guard
        # inside makes non-Evaluation hits a no-op.
        prevent_initial_call=False,
    )
    def _sync_from_url(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, Any]:
        # Outside Evaluation the tab + content slot don't exist in the
        # DOM — bailing is the only safe action; otherwise Dash logs
        # "nonexistent object in Output".
        if not pathname or not pathname.startswith("/evaluation"):
            raise PreventUpdate
        slug = active_subtab_from_path(pathname)
        # Forward live session into the sub-tab build so initial UI
        # state (group-by Select, cluster-pin, etc.) is seeded from
        # session at build time — the mechanism that replaces the
        # historic ``_hydrate_*`` callbacks (Page Contract §3 Rule L1).
        session = Session.from_store(session_data)
        return slug, build_subtab_content(slug, session=session)

    # ── Tab click → URL push ─────────────────────────────────────
    @app.callback(
        Output(SHELL_IDS["url"], "pathname", allow_duplicate=True),
        Input(EVALUATION_IDS["tabs"], "value"),
        State(SHELL_IDS["url"], "pathname"),
        prevent_initial_call=True,
    )
    def _push_url_from_tab(
        selected: Optional[str],
        current_pathname: Optional[str],
    ) -> str:
        if not selected:
            raise PreventUpdate
        new_path = path_for_subtab(selected)
        # Avoid an infinite loop with ``_sync_from_url`` — if the URL
        # is already where we'd push to, don't bounce.
        if current_pathname == new_path:
            raise PreventUpdate
        return new_path


# ─────────────────────────────────────────────────────────────────────
# 2. Capture — user input → session writes (server-side)
# ─────────────────────────────────────────────────────────────────────


def _register_capture(app: "Dash") -> None:
    """Wire every user-input gesture to its :class:`Session` slice.

    Capture callbacks share three rules:

    1. Outputs are limited to **always-mounted** stores
       (``session-store.data``) plus the filter inputs themselves
       (clear / chip × write back to the input ``value``).  No
       output can unmount mid-callback, so the historic "nonexistent
       object" warning is structurally impossible.
    2. Every callback takes ``url.pathname`` as :class:`State` and
       short-circuits with :class:`PreventUpdate` when the user is
       no longer on Evaluation — this protects against late ``value``
       events firing during page transitions.
    3. No UI rendering happens here.  Chip / label / clear-all
       visibility are derived clientside in :func:`_register_render`.
    """

    # ── Drawer toggle: click → Collapse.opened + session ─────────
    #
    # Single writer for ``Collapse.opened``.  The component's
    # initial ``opened`` is set in the layout from
    # ``session.evaluation.filter_bar_open`` (Rule L1), so no
    # hydration callback is needed.
    @app.callback(
        Output(EVAL_FILTER_IDS["collapse"],   "opened"),
        Output(SHELL_IDS["session_store"],    "data", allow_duplicate=True),
        Input(EVAL_FILTER_IDS["toggle_btn"],  "n_clicks"),
        State(EVAL_FILTER_IDS["collapse"],    "opened"),
        State(SHELL_IDS["session_store"],     "data"),
        State(SHELL_IDS["url"],               "pathname"),
        prevent_initial_call=True,
    )
    def _capture_drawer_toggle(
        n_clicks: Optional[int],
        currently_open: Optional[bool],
        session_data: Optional[Dict[str, Any]],
        pathname: Optional[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate
        if not pathname or not pathname.startswith("/evaluation"):
            raise PreventUpdate

        new_open = not bool(currently_open)
        session = Session.from_store(session_data)
        session.evaluation.filter_bar_open = new_open
        return new_open, session.to_store()

    # ── Filter values change → session ──────────────────────────
    #
    # Listens to all five inputs.  Rebuilds the full
    # :class:`EvaluationFilters` snapshot from the live values and
    # writes it back to session.  This is the **only** writer of
    # ``session.evaluation.filters`` other than the chip-× /
    # clear-all clears, which write the *input values* and rely on
    # this callback firing again to project that into session.
    @app.callback(
        Output(SHELL_IDS["session_store"],       "data", allow_duplicate=True),
        Input(EVAL_FILTER_IDS["asset_class"],    "value"),
        Input(EVAL_FILTER_IDS["currency"],       "value"),
        Input(EVAL_FILTER_IDS["desk"],           "value"),
        Input(EVAL_FILTER_IDS["product"],        "value"),
        Input(EVAL_FILTER_IDS["date_range"],     "value"),
        State(SHELL_IDS["session_store"],        "data"),
        State(SHELL_IDS["url"],                  "pathname"),
        prevent_initial_call=True,
    )
    def _capture_filter_values(
        asset_class: Optional[List[str]],
        currency:    Optional[List[str]],
        desk:        Optional[List[str]],
        product:     Optional[List[str]],
        date_range:  Optional[List[Optional[str]]],
        session_data: Optional[Dict[str, Any]],
        pathname:     Optional[str],
    ) -> Dict[str, Any]:
        if not pathname or not pathname.startswith("/evaluation"):
            raise PreventUpdate

        df, dt = _parse_date_range(date_range)
        filters = EvaluationFilters(
            asset_class=list(asset_class or []),
            currency=list(currency or []),
            desk=list(desk or []),
            product=list(product or []),
            date_from=df,
            date_to=dt,
        )

        session = Session.from_store(session_data)
        # No-op if filters haven't actually changed (avoids spurious
        # session-store writes that would re-trigger every callback
        # taking session-store as Input — none today, but cheap to
        # guard anyway).
        if session.evaluation.filters == filters:
            raise PreventUpdate

        session.evaluation.filters = filters
        return session.to_store()

    # ── Chip × — clear a single dimension ───────────────────────
    #
    # Pattern-matching callback writes the input's ``value`` to its
    # cleared form.  ``_capture_filter_values`` then picks up the
    # value change and writes the cleared :class:`EvaluationFilters`
    # to session.  Two-step rather than one-step because:
    # - the user expects the dropdown's selected pill to clear
    #   *visually*, which only happens by writing ``value``;
    # - having a single source of truth for "value → session"
    #   prevents drift between this callback's idea of session and
    #   what's actually live.
    @app.callback(
        Output(EVAL_FILTER_IDS["asset_class"], "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["currency"],    "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["desk"],        "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["product"],     "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["date_range"],  "value", allow_duplicate=True),
        Input({"type": "eval-filter-chip-close", "dimension": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _capture_chip_close(
        n_clicks_list: List[Optional[int]],
    ) -> Tuple[Any, Any, Any, Any, Any]:
        # Pattern-matching callbacks fire on *every* matching id when
        # the layout first renders; bail until a real click lands.
        if not any(n_clicks_list or []):
            raise PreventUpdate

        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        dim = triggered.get("dimension")

        # Default every Output to no_update so only the targeted
        # dropdown's value is rewritten — _capture_filter_values
        # then projects the change into session.
        if dim == "asset_class":
            return [], no_update, no_update, no_update, no_update
        if dim == "currency":
            return no_update, [], no_update, no_update, no_update
        if dim == "desk":
            return no_update, no_update, [], no_update, no_update
        if dim == "product":
            return no_update, no_update, no_update, [], no_update
        if dim == "date":
            return no_update, no_update, no_update, no_update, [None, None]

        raise PreventUpdate

    # ── Clear all / Reset — both empty every dimension ──────────
    @app.callback(
        Output(EVAL_FILTER_IDS["asset_class"], "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["currency"],    "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["desk"],        "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["product"],     "value", allow_duplicate=True),
        Output(EVAL_FILTER_IDS["date_range"],  "value", allow_duplicate=True),
        Input(EVAL_FILTER_IDS["clear_all"],    "n_clicks"),
        Input(EVAL_FILTER_IDS["reset_btn"],    "n_clicks"),
        prevent_initial_call=True,
    )
    def _capture_clear_all(
        clear_clicks: Optional[int],
        reset_clicks: Optional[int],
    ) -> Tuple[List[str], List[str], List[str], List[str], List[Any]]:
        if not (clear_clicks or reset_clicks):
            raise PreventUpdate
        return [], [], [], [], [None, None]


# ─────────────────────────────────────────────────────────────────────
# 3. Render — derived UI (clientside, zero round-trip)
# ─────────────────────────────────────────────────────────────────────


def _register_render(app: "Dash") -> None:
    """Wire the chip strip, "n active" label, and clear-all visibility
    to a single clientside callback (Lever P2).

    Implementation lives in ``assets/js/evaluation.js`` under the
    ``evaluation`` namespace.  Everything the function needs is
    already in the browser (the five filter values), so doing this
    server-side would be three needless round-trips per keystroke
    in the dropdown search box.

    The clientside path also makes Evaluation page transitions
    immune to the historical "nonexistent object in Output" warning:
    when the filter bar unmounts, the JS callback's outputs vanish
    along with its inputs, and the next mount fires the callback
    once with the freshly-rendered values.
    """
    app.clientside_callback(
        ClientsideFunction(
            namespace="evaluation",
            function_name="update_filter_ui",
        ),
        Output(EVAL_FILTER_IDS["chips"],         "children"),
        Output(EVAL_FILTER_IDS["toggle_label"],  "children"),
        Output(EVAL_FILTER_IDS["clear_all"],     "style"),
        Input(EVAL_FILTER_IDS["asset_class"],    "value"),
        Input(EVAL_FILTER_IDS["currency"],       "value"),
        Input(EVAL_FILTER_IDS["desk"],           "value"),
        Input(EVAL_FILTER_IDS["product"],        "value"),
        Input(EVAL_FILTER_IDS["date_range"],     "value"),
    )


# ─────────────────────────────────────────────────────────────────────
# 4. Bootstrap — metadata → MultiSelect data / disabled / placeholder
# ─────────────────────────────────────────────────────────────────────


# Placeholder strings + disabled flags per UI state.  Constants so the
# bootstrap callback (and any future contributor) reads by intent
# rather than inlining raw strings (Page Contract §3 Rule L3 cousin —
# behavioural state, not just visual).
_MS_PLACEHOLDER_READY:   str = "Any"
_MS_PLACEHOLDER_EMPTY:   str = "No options"
_MS_PLACEHOLDER_LOADING: str = "Loading…"


def _register_bootstrap(app: "Dash", backend: "RadeBackend") -> None:
    """Replace the layout's sentinel ``data`` arrays with real options.

    Trigger
    -------
    ``top_level_store.data`` (Input).  The router writes this store
    *together with* ``app-host.children`` whenever the user crosses
    between top-level routes (splash ↔ overview ↔ evaluation ↔ …).
    Dash schedules dependent callbacks after the router's outputs are
    applied to the DOM, so by the time this callback fires the
    Evaluation chrome (and the four MultiSelect IDs we write into) is
    *guaranteed* to be mounted.

    Why not ``url.pathname``
    ------------------------
    Pathname is the router's own Input — using it here too creates a
    mount-race: this callback fires concurrently with the router on
    pathname change, but the chrome hasn't been delivered to the
    browser yet, so our outputs target IDs that don't exist and Dash
    silently drops them.  The bug only resolved itself on the *next*
    pathname change (i.e. clicking another sub-tab), which made the
    filter bar populate only after a tab click.  Triggering on
    ``top_level_store`` waits for the chrome.

    Why not ``session-store.data``
    ------------------------------
    Session-store fires on every filter keystroke; we'd re-serialise
    12 outputs every keypress.  ``top_level_store`` only changes on
    cross-top-level nav (a handful of times per session) and stays
    constant on within-page sub-tab nav, which is exactly the
    cadence the dropdown options should refresh at.

    A future Phase E.5 (topbar version picker) will add a derived
    ``active-version-store`` Input alongside this one so switching
    versions refreshes options without leaving the page.

    ``prevent_initial_call=True`` rationale
    ---------------------------------------
    On initial app boot ``top_level_store.data`` starts as ``None``;
    the router's first run flips it to the active route.  We want to
    fire on that flip, not on the initial ``None``, so suppression of
    the initial call is correct.  Page Contract §4 Rule C5 satisfied
    — the opt-in is explicit.
    """

    @app.callback(
        # data = real option list (ready to drop into MultiSelect)
        Output(EVAL_FILTER_IDS["asset_class"], "data"),
        Output(EVAL_FILTER_IDS["currency"],    "data"),
        Output(EVAL_FILTER_IDS["desk"],        "data"),
        Output(EVAL_FILTER_IDS["product"],     "data"),
        # disabled flips false once we have real data; stays true if
        # the active ensemble has no values for that dimension.
        Output(EVAL_FILTER_IDS["asset_class"], "disabled"),
        Output(EVAL_FILTER_IDS["currency"],    "disabled"),
        Output(EVAL_FILTER_IDS["desk"],        "disabled"),
        Output(EVAL_FILTER_IDS["product"],     "disabled"),
        # placeholder swaps from "Loading…" → "Any" (or "No options"
        # when an ensemble omits a dimension entirely).
        Output(EVAL_FILTER_IDS["asset_class"], "placeholder"),
        Output(EVAL_FILTER_IDS["currency"],    "placeholder"),
        Output(EVAL_FILTER_IDS["desk"],        "placeholder"),
        Output(EVAL_FILTER_IDS["product"],     "placeholder"),
        Input(SHELL_IDS["top_level_store"], "data"),
        State(SHELL_IDS["url"],             "pathname"),
        State(SHELL_IDS["session_store"],   "data"),
        prevent_initial_call=True,  # see docstring §"prevent_initial_call=True rationale"
    )
    def _bootstrap_filter_options(
        top_level:    Optional[str],
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        # Rule C4 — page-scoped gate.  Both checks are belt-and-braces:
        # ``top_level`` is the value the router just committed; pathname
        # is read as State for parity with other Evaluation callbacks
        # so a future change to either gate doesn't desync the filter
        # bar from the rest of the page.
        if top_level != "/evaluation":
            raise PreventUpdate
        if not pathname or not pathname.startswith("/evaluation"):
            raise PreventUpdate

        # No version → splash hasn't resolved yet.  Leave the dropdowns
        # in the sentinel "Loading…" state; a later session-store write
        # by splash will (in Phase E.5) propagate here once we add a
        # derived version-store Input.
        session = Session.from_store(session_data)
        if not session.active_version:
            raise PreventUpdate

        res = backend.evaluation_filter_options()
        if not res.ok:
            # Tri-state Rule §5.1 — fail visibly: keep the dropdowns
            # disabled with a "Loading…" placeholder so the user sees
            # "not ready" rather than "broken".  A toast layer in
            # Phase D will surface the actual error message.
            raise PreventUpdate

        opts = res.data or {}

        ac_data, ac_disabled, ac_ph = _resolve_dropdown_state(opts.get("asset_class"))
        cc_data, cc_disabled, cc_ph = _resolve_dropdown_state(opts.get("currency"))
        dk_data, dk_disabled, dk_ph = _resolve_dropdown_state(opts.get("desk"))
        pr_data, pr_disabled, pr_ph = _resolve_dropdown_state(opts.get("product"))

        return (
            ac_data,     cc_data,     dk_data,     pr_data,
            ac_disabled, cc_disabled, dk_disabled, pr_disabled,
            ac_ph,       cc_ph,       dk_ph,       pr_ph,
        )


def _resolve_dropdown_state(
    items: Optional[List[Dict[str, str]]],
) -> Tuple[List[Dict[str, str]], bool, str]:
    """Map a dimension's option list to ``(data, disabled, placeholder)``.

    * Real options →  data=items, disabled=False, placeholder="Any".
    * Empty list   →  data=[],    disabled=True,  placeholder="No options".

    The "Loading…" state is owned by the layout (sentinel ``data`` +
    ``disabled=True``) and is never written by the bootstrap — once
    we're in a position to write outputs we already have either
    real options or an authoritative "this ensemble has none".
    """
    if items:
        return list(items), False, _MS_PLACEHOLDER_READY
    return [], True, _MS_PLACEHOLDER_EMPTY


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _parse_date_range(
    value: Optional[List[Optional[str]]],
) -> Tuple[Optional[str], Optional[str]]:
    """Normalise ``dmc.DatePickerInput(type='range').value`` into (from, to).

    DMC returns ``[None, None]`` on a fresh / cleared picker, ``[iso]``
    on a first-date-only partial selection and ``[iso_from, iso_to]``
    on a full range.  Normalise each of those into a tidy 2-tuple.
    """
    if not value or not isinstance(value, (list, tuple)):
        return None, None
    df = value[0] if len(value) > 0 else None
    dt = value[1] if len(value) > 1 else None
    return (df or None), (dt or None)


__all__ = ["register"]
