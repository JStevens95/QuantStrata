"""
Shared UI building blocks for QuantStrata Dash apps.

- layout: common page layout (navbar, footer, main container).
- styles: CSS / style constants for consistent look.
- components: reusable Dash components (input rows, cards, result blocks).
"""

from src.ui._shared.layout import make_app_layout
from src.ui._shared.styles import LAYOUT_STYLES

__all__ = ["LAYOUT_STYLES", "make_app_layout"]
