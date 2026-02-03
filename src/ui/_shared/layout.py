"""
Common layout wrapper for QuantStrata Dash apps.

Provides a consistent shell: navbar (brand + optional app title), main content area,
and footer. Each app supplies its own main_content and optional app_title.
"""

from __future__ import annotations

try:
    from dash import html
except ImportError as e:
    raise ImportError("Dash is required. Install with: pip install dash") from e

from src.ui._shared.styles import FOOTER_STYLES, LAYOUT_STYLES, NAVBAR_STYLES


def make_app_layout(
    main_content: list,
    *,
    app_title: str | None = None,
    brand: str = "QuantStrata",
) -> html.Div:
    """
    Build the standard app layout: navbar, main content, footer.

    Parameters
    ----------
    main_content : list
        List of Dash components (e.g. [html.Div(...), ...]) for the main area.
    app_title : str, optional
        Subtitle or app name shown next to the brand in the navbar.
    brand : str
        Brand text in the navbar (default "QuantStrata").

    Returns
    -------
    html.Div
        Full-page layout div.
    """
    navbar_content = [html.Span(brand, style=NAVBAR_STYLES["brand"])]
    if app_title:
        navbar_content.append(html.Span(app_title, style=NAVBAR_STYLES["appTitle"]))
    navbar = html.Div(
        navbar_content,
        style=NAVBAR_STYLES["navbar"],
    )
    main_div = html.Div(main_content, style=LAYOUT_STYLES["main"])
    footer = html.Div(
        "QuantStrata — Quantitative Finance Library",
        style=FOOTER_STYLES["footer"],
    )
    return html.Div(
        [navbar, main_div, footer],
        style=LAYOUT_STYLES["container"],
    )


__all__ = ["make_app_layout"]
