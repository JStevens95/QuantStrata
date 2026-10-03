"""
Asset market data objects — self-contained packages per risk factor.

Each asset class (FX, IR, EQ, CR) has a concrete subclass that loads,
holds, and exposes market data (spot, curves, surfaces), dependent
assets, and shock scenarios.

Separation from instruments/:
  - instruments/ defines WHAT can be traded (product specs, pricing, greeks)
  - assets/ defines WHAT the market looks like (data, curves, shocks)

Instruments consume assets:
  instrument.price(asset)  →  uses asset.spot, asset.vol_surface, etc.
"""
from .asset import Asset
from .types import (
    AssetConfig,
    Curve,
    FXShocks,
    IRShocks,
    VolCube,
    VolSurface,
)
from .fx import FXAsset
from .rates import IRAsset

__all__ = [
    # Base
    "Asset",
    "AssetConfig",
    # Market data containers
    "Curve",
    "VolSurface",
    "VolCube",
    # Shock types
    "FXShocks",
    "IRShocks",
    # Concrete assets
    "FXAsset",
    "IRAsset",
]
