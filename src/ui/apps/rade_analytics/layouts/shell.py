"""App shell — outer ``dmc.AppShell`` with sidebar, topbar and content slot.

Architecture
------------
``app.layout`` is a *thin* root: just ``MantineProvider`` + ``dcc.Location``
+ the persistent ``dcc.Store``\\s + a single ``html.Div(id="app-host")``.
The router writes the entire visible page tree into ``app-host.children``,
which is one of two things:

* **Splash route** (``/`` or ``/splash``) — :func:`..layouts.splash.build_splash`
  is dropped in directly with no shell wrapper, so the splash page gets the
  full-viewport "front-door" treatment its layout was designed for.
* **Every other route** — :func:`build_chrome` wraps the page builder's
  output in the ``dmc.AppShell`` (sidebar + topbar + content slot).

This keeps splash genuinely chrome-free without juggling CSS class toggles
or Mantine internals.

Within-page navigation preserves the chrome
-------------------------------------------
The router returns ``no_update`` for ``app-host.children`` when the user
navigates *within* a top-level page (e.g. between Evaluation sub-tabs).
That means the chrome stays mounted, the page-level callbacks update the
inner sub-tab content, and any ephemeral state (filter bar values, scroll
position) is preserved.

The chrome is only re-mounted on cross-top-level navigation (Overview ↔
Evaluation ↔ Monitoring …), or when leaving / arriving at splash.
"""

from __future__ import annotations

from typing import Any, Optional

import dash_mantine_components as dmc
from dash import dcc, html

from ..components.sidebar import build_sidebar_content
from ..components.topbar import build_topbar
from ..data.session import Session


# Every dynamic id in the shell lives here so callbacks never hardcode
# strings.
SHELL_IDS = {
    "mantine_root":    "mantine-root",
    "app_shell":       "rade-app-shell",
    "url":             "url",
    "session_store":   "session-store",
    "toast_store":     "toast-store",
    # Router bookkeeping — remembers the last *top-level* path
    # (spec.sidebar_path or spec.path) so the router can skip
    # ``app-host`` rebuilds when the user navigates within a page
    # (e.g. Evaluation sub-tab clicks).  Memory-scoped so a full
    # refresh always starts fresh.
    "top_level_store": "top-level-store",
    # Top-level slot the router writes into.  Holds either the splash
    # layout (no chrome) or the chrome wrapping a page (sidebar +
    # topbar + ``page-content``).
    "app_host":        "app-host",
    # Inner slot inside the chrome that page-level callbacks use to
    # target the page body (only present when chrome is mounted).
    "page_content":    "page-content",
}


def build_root() -> dmc.MantineProvider:
    """Build the *minimal* ``app.layout`` tree.

    The persistent state (``dcc.Location`` + the three ``dcc.Store``\\s)
    sits at the top so it survives every router-driven rebuild beneath
    it.  Everything visible is mounted into ``app-host`` by the router.

    The shell is locked to ``forceColorScheme="dark"`` per the design
    spec (§2); the theme toggle in the topbar is wired in a later phase
    when/if we add a light variant.
    """
    empty_session = Session().to_store()

    return dmc.MantineProvider(
        id=SHELL_IDS["mantine_root"],
        forceColorScheme="dark",
        children=[
            dcc.Location(id=SHELL_IDS["url"], refresh=False),
            dcc.Store(
                id=SHELL_IDS["session_store"],
                storage_type="session",
                data=empty_session,
            ),
            dcc.Store(
                id=SHELL_IDS["toast_store"],
                storage_type="memory",
                data=[],
            ),
            dcc.Store(
                id=SHELL_IDS["top_level_store"],
                storage_type="memory",
                data=None,
            ),
            html.Div(id=SHELL_IDS["app_host"]),
        ],
    )


def build_chrome(
    page_content: Any,
    *,
    breadcrumb_title:  str = "",
    active_path:       str = "/overview",
    current_split:     str = "test",
    current_pnl_space: str = "scaled",
) -> dmc.AppShell:
    """Wrap ``page_content`` with the standard sidebar + topbar AppShell.

    Parameters
    ----------
    page_content
        The page builder's output to drop into ``AppShellMain``.
    breadcrumb_title
        Second segment of the topbar breadcrumb (after "Dashboard ›").
    active_path
        Top-level path used by :func:`build_sidebar_content` to mark the
        matching nav item with ``rade-nav-item--active``.  Sub-routes
        (e.g. ``/evaluation/portfolio``) should pass their parent path
        (``/evaluation``) here.
    current_split
        Initial value for the topbar split toggle.  Sourced from
        ``session.split`` by the router so the topbar reflects the
        user's last choice across chrome rebuilds.
    current_pnl_space
        Initial value for the topbar PnL-space toggle (Phase 3.3).
        Sourced from ``session.pnl_space`` by the router for the same
        survive-rebuilds reason as ``current_split``.
    """
    return dmc.AppShell(
        id=SHELL_IDS["app_shell"],
        header={"height": 60},
        navbar={"width": 220, "breakpoint": "sm"},
        padding="md",
        children=[
            dmc.AppShellHeader(
                children=build_topbar(
                    breadcrumb_title=breadcrumb_title,
                    current_split=current_split,
                    current_pnl_space=current_pnl_space,
                ),
                className=(
                    "bg-slate-950/80 border-b border-slate-800 "
                    "backdrop-blur-sm"
                ),
            ),
            dmc.AppShellNavbar(
                children=build_sidebar_content(active_path=active_path),
                className="bg-slate-900 border-r border-slate-800",
            ),
            dmc.AppShellMain(
                children=html.Div(
                    id=SHELL_IDS["page_content"],
                    className="min-h-full",
                    children=page_content,
                ),
                className="bg-slate-950",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Backwards-compat shim
# ─────────────────────────────────────────────────────────────────────


def build_shell() -> dmc.MantineProvider:
    """Deprecated — use :func:`build_root` instead.

    Kept around so external scripts and the preview examples that
    haven't been updated yet still import without an ``ImportError``.
    Internally identical to :func:`build_root`.
    """
    return build_root()


__all__ = [
    "SHELL_IDS",
    "build_chrome",
    "build_root",
    "build_shell",
]
