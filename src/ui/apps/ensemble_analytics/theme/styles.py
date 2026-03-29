"""
Reusable CSS-in-Python style dictionaries for Dash components.

Import individual style dicts in layout modules.  All values reference
color tokens from ``colors.py`` — never hard-code hex here.
"""
from src.ui.apps.ensemble_analytics.theme.colors import (
    BG_PRIMARY,
    BG_CARD,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

CONTAINER_STYLE: dict = {
    "backgroundColor": BG_PRIMARY,
    "minHeight": "100vh",
    "padding": "0",
    "fontFamily": '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
    "color": TEXT_PRIMARY,
}

NAVBAR_STYLE: dict = {
    "backgroundColor": BG_CARD,
    "borderBottom": f"1px solid {BORDER}",
    "padding": "10px 20px",
}

CARD_STYLE: dict = {
    "backgroundColor": BG_CARD,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "20px",
    "marginBottom": "16px",
}

CARD_HEADER_STYLE: dict = {
    "fontSize": "12px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "color": TEXT_SECONDARY,
    "marginBottom": "8px",
}

KPI_VALUE_STYLE: dict = {
    "fontSize": "28px",
    "fontWeight": "700",
    "color": TEXT_PRIMARY,
    "lineHeight": "1.2",
}

TABLE_STYLE: dict = {
    "backgroundColor": BG_CARD,
    "color": TEXT_PRIMARY,
    "fontSize": "13px",
}

SECTION_TITLE_STYLE: dict = {
    "fontSize": "16px",
    "fontWeight": "600",
    "color": TEXT_PRIMARY,
    "marginBottom": "12px",
    "marginTop": "24px",
}
