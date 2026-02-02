"""
Multi-Asset Option Pricers.

This module provides pricer classes for multi-asset derivatives:
- Basket options (Monte Carlo)
- Spread options (Monte Carlo, Kirk's approximation)
- Exchange options (Margrabe's formula)
- Rainbow options (Monte Carlo)
"""

from src.pricers.multi_asset.basket_european_mc import (
    MultiAssetBasketEuropeanOptionMcPricer,
    MultiAssetBasketEuropeanOptionMcSimulation,
)

from src.pricers.multi_asset.spread_european_mc import (
    MultiAssetSpreadEuropeanOptionMcPricer,
    MultiAssetSpreadEuropeanOptionMcSimulation,
    MultiAssetSpreadEuropeanOptionKirkPricer,
    MultiAssetExchangeEuropeanOptionMargrabePricer,
)

from src.pricers.multi_asset.rainbow_european_mc import (
    MultiAssetBestOfEuropeanOptionMcPricer,
    MultiAssetBestOfEuropeanOptionMcSimulation,
    MultiAssetWorstOfEuropeanOptionMcPricer,
    MultiAssetWorstOfEuropeanOptionMcSimulation,
)

__all__ = [
    # Basket
    "MultiAssetBasketEuropeanOptionMcPricer",
    "MultiAssetBasketEuropeanOptionMcSimulation",
    # Spread
    "MultiAssetSpreadEuropeanOptionMcPricer",
    "MultiAssetSpreadEuropeanOptionMcSimulation",
    "MultiAssetSpreadEuropeanOptionKirkPricer",
    # Exchange
    "MultiAssetExchangeEuropeanOptionMargrabePricer",
    # Rainbow
    "MultiAssetBestOfEuropeanOptionMcPricer",
    "MultiAssetBestOfEuropeanOptionMcSimulation",
    "MultiAssetWorstOfEuropeanOptionMcPricer",
    "MultiAssetWorstOfEuropeanOptionMcSimulation",
]
