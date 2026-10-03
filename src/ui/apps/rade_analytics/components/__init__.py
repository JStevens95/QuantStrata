"""Stateless presentational primitives used across every page.

Populated in Phase B (design system primitives).

Sub-modules shipped so far
--------------------------
* :mod:`.brand`            — wordmark + logo (B.5).
* :mod:`.sidebar`          — navbar content for ``dmc.AppShellNavbar`` (B.3).
* :mod:`.topbar`           — header content for ``dmc.AppShellHeader`` (B.3).
* :mod:`.state_wrappers`   — ``Loading`` / ``Empty`` / ``Error`` /
  ``AuthGate`` (B.4).
* :mod:`.kpi_card`         — single KPI card primitive (B.5).
* :mod:`.chart_container`  — card chrome around ``dcc.Graph`` (B.5).
* :mod:`.ag_grid_table`    — themed ``dash-ag-grid`` wrapper (B.5).
"""

from .ag_grid_table import AgGridTable, columns_from_dataframe
from .brand import Brand, BrandSize
from .chart_container import ChartContainer
from .kpi_card import DeltaTone, KpiCard
from .sidebar import (
    NAV_ITEMS,
    SIDEBAR_IDS,
    NavItem,
    build_sidebar_content,
    nav_item_id,
)
from .state_wrappers import (
    AuthGate,
    Empty,
    Error,
    Loading,
    LoadingVariant,
)
from .topbar import TOPBAR_IDS, build_topbar

__all__ = [
    # State wrappers (B.4)
    "AuthGate",
    "Empty",
    "Error",
    "Loading",
    "LoadingVariant",
    # Shell content (B.3)
    "NAV_ITEMS",
    "NavItem",
    "SIDEBAR_IDS",
    "TOPBAR_IDS",
    "build_sidebar_content",
    "build_topbar",
    "nav_item_id",
    # Primitives (B.5)
    "AgGridTable",
    "Brand",
    "BrandSize",
    "ChartContainer",
    "DeltaTone",
    "KpiCard",
    "columns_from_dataframe",
]
