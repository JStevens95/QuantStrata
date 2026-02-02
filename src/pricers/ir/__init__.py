# src/pricers/ir/__init__.py
"""
Interest Rate Pricers.

This module contains pricers for interest rate derivatives:
- FRA and IRS linear pricers
- Bond linear pricers (zero coupon, fixed rate)
- Cap/Floor Black76 pricers
- Bond option Black76 pricers
- Swaption Bachelier pricers
"""
from src.pricers.ir.swap import IrSwapPricer, IrSwapPricerSimple
from src.pricers.ir.fra import IrFraPricer, IrFraPricerSimple
from src.pricers.ir.bond import (
    IrBondZeroCouponPricer,
    IrBondZeroCouponPricerSimple,
    IrBondFixedRatePricer,
    IrBondFixedRatePricerSimple,
)
from src.pricers.ir.european_b76 import (
    # Caplet/Floorlet simple pricers
    IrCapletEuropeanOptionB76PricerSimple,
    IrFloorletEuropeanOptionB76PricerSimple,
    IrCapEuropeanOptionB76PricerSimple,
    IrFloorEuropeanOptionB76PricerSimple,
    # Caplet/Floorlet market data pricers
    IrCapletEuropeanOptionB76Pricer,
    IrFloorletEuropeanOptionB76Pricer,
    IrCapEuropeanOptionB76Pricer,
    IrFloorEuropeanOptionB76Pricer,
    # Bond option pricers
    IrBondEuropeanOptionB76Pricer,
    IrBondEuropeanOptionB76PricerSimple,
)
from src.pricers.ir.european_bch import (
    IrSwaptionEuropeanOptionBchPricer,
    IrSwaptionEuropeanOptionBchPricerSimple,
)

__all__ = [
    # Linear pricers - FRA/Swap
    "IrFraPricer",
    "IrFraPricerSimple",
    "IrSwapPricer",
    "IrSwapPricerSimple",
    # Linear pricers - Bonds
    "IrBondZeroCouponPricer",
    "IrBondZeroCouponPricerSimple",
    "IrBondFixedRatePricer",
    "IrBondFixedRatePricerSimple",
    # Black76 pricers - Caps/Floors
    "IrCapletEuropeanOptionB76PricerSimple",
    "IrFloorletEuropeanOptionB76PricerSimple",
    "IrCapEuropeanOptionB76PricerSimple",
    "IrFloorEuropeanOptionB76PricerSimple",
    "IrCapletEuropeanOptionB76Pricer",
    "IrFloorletEuropeanOptionB76Pricer",
    "IrCapEuropeanOptionB76Pricer",
    "IrFloorEuropeanOptionB76Pricer",
    # Black76 pricers - Bond Options
    "IrBondEuropeanOptionB76Pricer",
    "IrBondEuropeanOptionB76PricerSimple",
    # Bachelier pricers
    "IrSwaptionEuropeanOptionBchPricer",
    "IrSwaptionEuropeanOptionBchPricerSimple",
]
