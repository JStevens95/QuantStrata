"""
Market Data Surfaces Module

This module provides volatility surface classes for different asset classes:

FX/Equity Surfaces:
- FlatVolSurface: Constant volatility surface
- GridVolSurface: 2D surface (expiry x strike)
- LocalVolSurface: 2D local vol surface σ(S, t)

Interest Rate Surfaces:
- SwaptionVolCube: 3D surface (expiry x tenor x strike)
- CapFloorVolSurface: 2D surface (expiry x strike)
- FlatSwaptionVolCube: Constant swaption vol
- FlatCapFloorVolSurface: Constant cap/floor vol

Author: QuantStrata
"""
from src.marketdata.surfaces.vol_surface import (
    FlatVolSurface,
    GridVolSurface,
    # IR Surfaces
    SwaptionVolCube,
    FlatSwaptionVolCube,
    CapFloorVolSurface,
    FlatCapFloorVolSurface,
    create_atm_swaption_vol_cube,
    create_cap_vol_surface_from_term_structure,
)

from src.marketdata.surfaces.local_vol_surface import (
    LocalVolSurface,
    FlatLocalVolSurface,
)

__all__ = [
    # FX/Equity
    "FlatVolSurface",
    "GridVolSurface",
    "LocalVolSurface",
    "FlatLocalVolSurface",
    # Interest Rates
    "SwaptionVolCube",
    "FlatSwaptionVolCube",
    "CapFloorVolSurface",
    "FlatCapFloorVolSurface",
    "create_atm_swaption_vol_cube",
    "create_cap_vol_surface_from_term_structure",
]
