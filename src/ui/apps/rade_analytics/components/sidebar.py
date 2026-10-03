"""Sidebar navigation — stateless content for ``dmc.AppShellNavbar``.

This module is deliberately pure layout: every nav item is a
``dcc.Link`` so clicks update ``dcc.Location`` without a full page
reload, and there are no callbacks registered here.  The router that
B.6 builds will watch ``url.pathname`` and, in the same callback that
swaps page content, apply the ``rade-nav-item--active`` class to the
matching link.

Design spec anchors
-------------------
* Layout (§5): 220 px fixed-width sidebar, brand header, nav list,
  footer build-info strip.
* Nav taxonomy (§11): the nine top-level pages matching the 20 mock
  screens in ``docs/platform_designs/rade_*.png``.
* Styling (§6): ``rade-nav-item`` and ``rade-nav-item--active``
  classes from ``assets/rade.css``; icons via ``dash-iconify``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from dash import dcc, html
from dash_iconify import DashIconify

from .brand import Brand


# ─────────────────────────────────────────────────────────────────────
# Nav taxonomy
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NavItem:
    """Declarative description of a single sidebar link.

    ``path`` is the *identity* of the nav item: it doubles as a stable
    key for :func:`nav_item_id` and as the prefix the router matches
    against ``dcc.Location.pathname`` when toggling the active class.

    ``href`` is the *target* the user navigates to on click.  For most
    pages this is identical to ``path`` and defaults accordingly, but
    pages with sub-routes (e.g. Evaluation) point ``href`` at their
    canonical default sub-tab so a sidebar click lands on something
    concrete instead of a bare `/evaluation` that would need a redirect.

    ``icon`` is a Tabler / Iconify id string resolved by DashIconify.
    """

    path: str
    label: str
    icon: str
    href: Optional[str] = None

    @property
    def target_href(self) -> str:
        """Effective link target — falls back to :attr:`path`."""
        return self.href if self.href is not None else self.path


NAV_ITEMS: List[NavItem] = [
    # ``/`` is reserved for the splash / version-picker page, which has
    # no sidebar entry by design.  Overview lives at ``/overview`` so
    # the splash → overview transition is an explicit URL change.
    NavItem("/overview", "Overview", "tabler:layout-dashboard"),
    # Evaluation lands on Portfolio by default — keeps URLs canonical
    # without a bounce-through redirect callback.
    NavItem(
        "/evaluation",
        "Evaluation",
        "tabler:chart-line",
        href="/evaluation/portfolio",
    ),
    NavItem("/monitoring", "Monitoring", "tabler:activity"),
    # Icons below use only "core" Tabler glyphs (no -check / -cog
    # suffixes) because some newer Tabler variants aren't in the
    # Iconify CDN bundle that dash-iconify fetches from — missing
    # icons render as nothing, not a fallback glyph.
    NavItem("/governance", "Governance", "tabler:shield"),
    NavItem("/inference", "Inference", "tabler:target-arrow"),
    NavItem("/risk-management", "Risk Management", "tabler:shield-half"),
    NavItem("/scenario-lab", "Scenario Lab", "tabler:flask"),
    NavItem("/report-builder", "Report Builder", "tabler:file-text"),
    NavItem("/data-quality", "Data Quality", "tabler:database"),
    NavItem("/assistant", "AI Assistant", "tabler:sparkles"),
]


# Every dynamic id in the sidebar lives here so callbacks never
# hardcode strings.  The router in B.6 uses ``nav_item_id(path)`` to
# look up the current link and toggle the active class.
def nav_item_id(path: str) -> str:
    """Deterministic DOM id for a given nav path."""
    slug = path.strip("/") or "overview"
    return f"sidebar-nav-{slug}"


SIDEBAR_IDS = {
    "brand": "sidebar-brand",
    "nav_list": "sidebar-nav-list",
    "footer": "sidebar-footer",
}


# ─────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────


_ACTIVE_NAV_CLASS = "rade-nav-item rade-nav-item--active"
_IDLE_NAV_CLASS = "rade-nav-item"


def _is_active_nav(item_path: str, active_path: str) -> bool:
    """Should the nav item at ``item_path`` highlight for ``active_path``?

    Exact-match leaves (``/overview``, ``/monitoring`` …) compare directly.
    Parent paths with sub-routes (``/evaluation``) also light up when the
    current path is a sub-route (``/evaluation/portfolio``).
    """
    if item_path == active_path:
        return True
    return active_path.startswith(item_path + "/")


def _nav_link(item: NavItem, *, active_path: str) -> dcc.Link:
    is_active = _is_active_nav(item.path, active_path)
    return dcc.Link(
        id=nav_item_id(item.path),
        href=item.target_href,
        className=_ACTIVE_NAV_CLASS if is_active else _IDLE_NAV_CLASS,
        children=[
            DashIconify(icon=item.icon, width=18),
            html.Span(item.label),
        ],
        refresh=False,
    )


def build_sidebar_content(active_path: str = "/overview") -> html.Div:
    """Build the sidebar's inner layout: brand header + nav + footer.

    Parameters
    ----------
    active_path
        The current top-level path.  The matching nav item is rendered
        with the ``rade-nav-item--active`` class baked in at build time
        so the router doesn't need a separate Output per nav item.
        Sub-routes (e.g. ``/evaluation/portfolio``) should pass their
        parent path (``/evaluation``) here.

    Returned as a plain ``html.Div`` so the caller (``build_chrome``)
    can drop it straight into ``dmc.AppShellNavbar(children=...)``.
    """
    return html.Div(
        className="flex flex-col h-full",
        children=[
            html.Div(
                id=SIDEBAR_IDS["brand"],
                className="px-4 py-5 border-b border-slate-800",
                children=Brand(size="md", href="/overview"),
            ),
            html.Nav(
                id=SIDEBAR_IDS["nav_list"],
                className="flex-1 flex flex-col gap-1 p-3 overflow-y-auto",
                children=[
                    _nav_link(item, active_path=active_path)
                    for item in NAV_ITEMS
                ],
            ),
            html.Div(
                id=SIDEBAR_IDS["footer"],
                className=(
                    "px-4 py-3 border-t border-slate-800 "
                    "text-xs text-slate-500 font-mono"
                ),
                children="Rade UI v0.1 · dev",
            ),
        ],
    )


__all__ = [
    "NAV_ITEMS",
    "NavItem",
    "SIDEBAR_IDS",
    "build_sidebar_content",
    "nav_item_id",
]
