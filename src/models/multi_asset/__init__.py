"""
Multi-Asset Models and Products.

This module provides simulation and pricing for multi-asset derivatives:
- Basket options
- Spread options  
- Best-of / Worst-of options
- Rainbow options
"""

from src.models.multi_asset.simulation import (
    CorrelationMatrix,
    MultiAssetGBM,
    MultiAssetSimulation,
)

from src.models.multi_asset.basket import (
    BasketParameters,
    basket_call_mc,
    basket_put_mc,
)

from src.models.multi_asset.spread import (
    SpreadParameters,
    spread_call_mc,
    spread_put_mc,
    kirk_spread_call,
    kirk_spread_put,
)

from src.models.multi_asset.rainbow import (
    RainbowParameters,
    best_of_call_mc,
    best_of_put_mc,
    worst_of_call_mc,
    worst_of_put_mc,
)

__all__ = [
    # Simulation
    "CorrelationMatrix",
    "MultiAssetGBM",
    "MultiAssetSimulation",
    # Basket
    "BasketParameters",
    "basket_call_mc",
    "basket_put_mc",
    # Spread
    "SpreadParameters",
    "spread_call_mc",
    "spread_put_mc",
    "kirk_spread_call",
    "kirk_spread_put",
    # Rainbow
    "RainbowParameters",
    "best_of_call_mc",
    "best_of_put_mc",
    "worst_of_call_mc",
    "worst_of_put_mc",
]
