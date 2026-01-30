# src/models/analytic/black_scholes_merton/__init__.py
"""
Black-Scholes-Merton Model.

Generalized BSM with cost-of-carry for spot-based pricing.

Exports
-------
All functions from base.py for convenient access.
"""
from src.models.analytic.black_scholes_merton.base import (
    # Types
    OptionType,
    GreekName,
    # Validation
    validate_inputs,
    # Core helpers
    d1_d2,
    forward_factor,
    discount_factor,
    intrinsic,
    # Vanilla
    vanilla_price,
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
    vanilla_rho_discount,
    vanilla_rho_carry,
    vanilla_greeks,
    # Digital cash
    digital_cash_price,
    digital_cash_delta,
    digital_cash_gamma,
    digital_cash_vega,
    digital_cash_theta,
    digital_cash_greeks,
    # Digital asset
    digital_asset_price,
    digital_asset_delta,
    digital_asset_gamma,
    digital_asset_vega,
    digital_asset_theta,
    digital_asset_greeks,
)

__all__ = [
    # Types
    "OptionType",
    "GreekName",
    # Validation
    "validate_inputs",
    # Core helpers
    "d1_d2",
    "forward_factor",
    "discount_factor",
    "intrinsic",
    # Vanilla
    "vanilla_price",
    "vanilla_delta",
    "vanilla_gamma",
    "vanilla_vega",
    "vanilla_theta",
    "vanilla_rho_discount",
    "vanilla_rho_carry",
    "vanilla_greeks",
    # Digital cash
    "digital_cash_price",
    "digital_cash_delta",
    "digital_cash_gamma",
    "digital_cash_vega",
    "digital_cash_theta",
    "digital_cash_greeks",
    # Digital asset
    "digital_asset_price",
    "digital_asset_delta",
    "digital_asset_gamma",
    "digital_asset_vega",
    "digital_asset_theta",
    "digital_asset_greeks",
]
