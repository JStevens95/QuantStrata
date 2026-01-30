# src/models/analytic/bachelier/__init__.py
"""
Bachelier (Normal) Model.

Normal distribution model for options where the underlying can be negative.

Exports
-------
All functions from base.py for convenient access.
"""
from src.models.analytic.bachelier.base import (
    # Types
    OptionType,
    GreekName,
    # Validation
    validate_inputs,
    # Core helpers
    d_moneyness,
    intrinsic,
    # Vanilla
    vanilla_price,
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
    vanilla_rho,
    vanilla_greeks,
)

__all__ = [
    # Types
    "OptionType",
    "GreekName",
    # Validation
    "validate_inputs",
    # Core helpers
    "d_moneyness",
    "intrinsic",
    # Vanilla
    "vanilla_price",
    "vanilla_delta",
    "vanilla_gamma",
    "vanilla_vega",
    "vanilla_theta",
    "vanilla_rho",
    "vanilla_greeks",
]
