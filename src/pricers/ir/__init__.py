# src/pricers/ir/__init__.py
"""
Interest Rate Pricers.

This module contains pricers for interest rate derivatives:
- FRA and IRS linear pricers
- Cap/Floor Black76 pricers
- Swaption Bachelier pricers
"""
from src.pricers.ir.swap import IrSwapPricer, IrSwapPricerSimple
from src.pricers.ir.fra import IrFraPricer, IrFraPricerSimple
from src.pricers.ir.european_b76 import (
    # Simple pricers (actual class names from european_b76.py)
    IrCapletEuropeanOptionB76PricerSimple,
    IrFloorletEuropeanOptionB76PricerSimple,
    IrCapEuropeanOptionB76PricerSimple,
    IrFloorEuropeanOptionB76PricerSimple,
    # Market data pricers
    IrCapletEuropeanOptionB76Pricer,
    IrFloorletEuropeanOptionB76Pricer,
    IrCapEuropeanOptionB76Pricer,
    IrFloorEuropeanOptionB76Pricer,
)
from src.pricers.ir.european_bch import (
    IrSwaptionEuropeanOptionBchPricer,
    IrSwaptionEuropeanOptionBchPricerSimple,
)

__all__ = [
    # Linear pricers
    "IrFraPricer",
    "IrFraPricerSimple",
    "IrSwapPricer",
    "IrSwapPricerSimple",
    # Black76 pricers
    "IrCapletEuropeanOptionB76PricerSimple",
    "IrFloorletEuropeanOptionB76PricerSimple",
    "IrCapEuropeanOptionB76PricerSimple",
    "IrFloorEuropeanOptionB76PricerSimple",
    "IrCapletEuropeanOptionB76Pricer",
    "IrFloorletEuropeanOptionB76Pricer",
    "IrCapEuropeanOptionB76Pricer",
    "IrFloorEuropeanOptionB76Pricer",
    # Bachelier pricers
    "IrSwaptionEuropeanOptionBchPricer",
    "IrSwaptionEuropeanOptionBchPricerSimple",
]
