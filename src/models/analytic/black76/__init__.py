# src/models/analytic/black76/__init__.py
"""
Black76 Model.

Forward-based pricing model for options on futures, forwards, and forward rates.

Exports
-------
All functions from base.py for convenient access.
"""
from src.models.analytic.black76.base import (
    # Types
    OptionType,
    GreekName,
    # Validation
    validate_inputs,
    # Core helpers
    d1_d2,
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
    "d1_d2",
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
