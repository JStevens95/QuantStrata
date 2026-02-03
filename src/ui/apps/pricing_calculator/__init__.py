"""
FX Vanilla Option Pricing Calculator — Dash app.

Run: python -m src.ui.run pricing_calculator
Or:  from src.ui.apps.pricing_calculator.app import create_app
"""

from src.ui.apps.pricing_calculator.app import create_app

__all__ = ["create_app"]
