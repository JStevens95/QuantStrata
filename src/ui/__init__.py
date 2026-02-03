"""
QuantStrata UI package: Dash-based interactive apps.

Run a Dash app from the project root, e.g.:

    python -m src.ui.run pricing_calculator

Requires: pip install dash (or use requirements-ui.txt).
"""

__all__ = ["create_pricing_calculator_app"]


def create_pricing_calculator_app():
    """Create the FX vanilla pricing calculator Dash app (requires dash)."""
    from src.ui.apps.pricing_calculator import create_app
    return create_app()
