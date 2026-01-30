# src/models/analytic/__init__.py
"""
Analytic Pricing Models.

This package contains closed-form pricing models:
- Black-Scholes-Merton (BSM): Spot-based with cost-of-carry
- Black76: Forward-based for futures/forward options
- Bachelier: Normal model for negative rates/spreads

All models expose pure functions for maximum composability.
"""
from src.models.analytic import black_scholes_merton
from src.models.analytic import black76
from src.models.analytic import bachelier

__all__ = [
    "black_scholes_merton",
    "black76",
    "bachelier",
]
