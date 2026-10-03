"""Splash page callbacks — bootstrap on arrival + navigate on Enter.

Two callbacks live here:

* **Bootstrap** (URL → splash state)
    Fires whenever ``dcc.Location.pathname`` changes.  Early-returns for
    any path that isn't the splash page.  Splash is reachable at *both*
    ``/`` (the front-door) and ``/splash`` (legacy alias), so the gate
    accepts both.  On a match it calls :meth:`RadeBackend.health` +
    :meth:`RadeBackend.versions` and populates the status strip, the
    read-only active-version pill, and enables the CTA.

* **Enter** (button → session + navigation)
    When the user clicks the primary CTA we stamp the API's active
    version into the session-level ``dcc.Store`` and push
    ``pathname="/overview"`` onto ``dcc.Location``, which the router
    then handles.

There is no version *switcher* on splash — see ``layouts/splash.py``
for the rationale.  Active version is sourced from
``backend.versions().active`` (cached) on Enter so the session always
reflects the API's actual binding rather than a user choice that
couldn't actually be honoured.

Both callbacks capture the shared :class:`RadeBackend` via closure, so
no module-level globals are introduced and tests can inject their own
backend by calling :func:`register` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from ..data.session import Session
from ..layouts.shell import SHELL_IDS
from ..layouts.splash import SPLASH_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


# ─────────────────────────────────────────────────────────────────────
# Styling constants — kept module-private so callbacks read them by
# intent rather than inlining Tailwind strings.
# ─────────────────────────────────────────────────────────────────────

# Class strings for the status dot + error banner.  All visual styling
# lives in rade.css (see ``.rade-status-dot--*`` and
# ``.rade-splash-error-banner--visible``); keeping these as module
# constants means the callback reads by intent instead of string-shuffling.
_DOT_OK    = "rade-status-dot rade-status-dot--ok"
_DOT_ERR   = "rade-status-dot rade-status-dot--err"
_DOT_BOOT  = "rade-status-dot rade-status-dot--booting"

_BANNER_HIDDEN = "rade-splash-error-banner"
_BANNER_ERROR  = "rade-splash-error-banner rade-splash-error-banner--visible"

# Order of the bootstrap callback's Outputs — declared as a module
# constant so the callback signature and the returned tuple never drift
# out of sync.
_BOOTSTRAP_OUTPUTS = (
    Output(SPLASH_IDS["status_dot"],     "className"),
    Output(SPLASH_IDS["status_label"],   "children"),
    Output(SPLASH_IDS["api_url"],        "children"),
    Output(SPLASH_IDS["active_version"], "children"),
    Output(SPLASH_IDS["enter_btn"],      "disabled"),
    Output(SPLASH_IDS["error_banner"],   "children"),
    Output(SPLASH_IDS["error_banner"],   "className"),
)


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach splash callbacks to ``app``.

    Called by :func:`..callbacks.register_all` during app initialisation;
    not meant to be invoked elsewhere.
    """

    # ── Bootstrap ───────────────────────────────────────────────────

    @app.callback(
        *_BOOTSTRAP_OUTPUTS,
        Input(SHELL_IDS["url"], "pathname"),
        # Page Contract §4 Rule C5 — explicit opt-in.  Splash is the
        # landing page; we *want* to fire on the implicit initial
        # pathname callback so the status strip + version pill paint
        # before the user can click the CTA.  The pathname gate inside
        # the body keeps non-splash URLs cheap.
        prevent_initial_call=False,
    )
    def _bootstrap(pathname: Optional[str]) -> Tuple[Any, ...]:
        # Cheap gate — every non-splash URL change still fires this
        # callback because Inputs can't be scoped to a path, so bail
        # fast for anything we don't care about.  Splash is mounted at
        # both ``/`` (front door) and ``/splash`` (legacy alias).
        if pathname not in ("/", "/splash"):
            raise PreventUpdate

        health = backend.health()
        versions = backend.versions()

        # ── API URL (always shown, regardless of success) ───────
        api_url = (
            health.data.artifacts_dir if health.ok and health.data is not None
            else "—"
        )

        # ── Versions failed: we can't show an active version ────
        if not versions.ok:
            err_msg = _format_error(
                "Could not reach the Rade API.",
                versions.error,
                versions.status_code,
            )
            return (
                _DOT_ERR,
                "API unavailable",
                api_url,
                "—",
                True,       # enter btn disabled
                err_msg,
                _BANNER_ERROR,
            )

        assert versions.data is not None  # narrowed by .ok
        active = versions.data.active

        # ── Health failed but versions succeeded ────────────────
        # Rare but possible (health probe disabled, versions still
        # readable from registry).  Show a softer yellow dot.
        if not health.ok:
            return (
                _DOT_ERR,
                "API degraded",
                "—",
                active,
                False,       # enter btn enabled
                "",
                _BANNER_HIDDEN,
            )

        # ── Happy path ──────────────────────────────────────────
        return (
            _DOT_OK,
            "Live",
            api_url,
            active,
            False,
            "",
            _BANNER_HIDDEN,
        )

    # ── Enter CTA ───────────────────────────────────────────────────

    @app.callback(
        Output(SHELL_IDS["url"],           "pathname", allow_duplicate=True),
        Output(SHELL_IDS["session_store"], "data",     allow_duplicate=True),
        Input(SPLASH_IDS["enter_btn"], "n_clicks"),
        State(SHELL_IDS["session_store"],   "data"),
        prevent_initial_call=True,
    )
    def _on_enter(
        n_clicks: Optional[int],
        current_session: Optional[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate

        # Hydrate the session defensively — if the store is empty or
        # stale-schema, from_store() resets to defaults.
        session = Session.from_store(current_session)

        # The API server is bound to one version at startup, so the
        # session's "active version" is whatever the API reports —
        # not a user selection.  ``backend.versions()`` is cached, so
        # this is a near-free dict lookup on the second hit.
        versions = backend.versions()
        if versions.ok and versions.data is not None:
            session.active_version = versions.data.active

        return "/overview", session.to_store()


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _format_error(
    headline: str,
    detail: Optional[str],
    status_code: Optional[int],
) -> str:
    """Compose a one-line diagnostic string for the banner."""
    pieces = [headline]
    if status_code is not None:
        pieces.append(f"(HTTP {status_code})")
    if detail:
        pieces.append(f"— {detail}")
    return " ".join(pieces)


__all__ = ["register"]
