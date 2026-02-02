"""
Multi-Asset Instruments.

This module provides instrument definitions for multi-asset derivatives:
- Basket options
- Spread options
- Rainbow options (best-of, worst-of)
"""

from src.instruments.multi_asset.basket import MultiAssetBasketEuropeanOption

from src.instruments.multi_asset.spread import (
    MultiAssetSpreadEuropeanOption,
    MultiAssetExchangeEuropeanOption,
)

from src.instruments.multi_asset.rainbow import (
    MultiAssetBestOfEuropeanOption,
    MultiAssetWorstOfEuropeanOption,
)

__all__ = [
    "MultiAssetBasketEuropeanOption",
    "MultiAssetSpreadEuropeanOption",
    "MultiAssetExchangeEuropeanOption",
    "MultiAssetBestOfEuropeanOption",
    "MultiAssetWorstOfEuropeanOption",
]
