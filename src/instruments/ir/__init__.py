# src/instruments/ir/__init__.py
"""
Interest Rate Instruments.

This module contains instruments for the interest rate asset class:
- Forward Rate Agreements (FRAs) - linear
- Interest Rate Swaps (IRS) - linear
- Caplets and Floorlets (single-period options)
- Caps and Floors (multi-period options)
"""
from src.instruments.ir.linear.fra import (
    ForwardRateAgreement,
    ForwardRateAgreementSimple,
)
from src.instruments.ir.linear.swap import (
    InterestRateSwap,
    InterestRateSwapSimple,
    SwapLeg,
    FixedLeg,
    FloatingLeg,
)
from src.instruments.ir.options.capfloor import (
    Caplet,
    CapletSimple,
    Floorlet,
    FloorletSimple,
    Cap,
    CapSimple,
    Floor,
    FloorSimple,
)

__all__ = [
    # Linear instruments
    "ForwardRateAgreement",
    "ForwardRateAgreementSimple",
    "InterestRateSwap",
    "InterestRateSwapSimple",
    "SwapLeg",
    "FixedLeg",
    "FloatingLeg",
    # Options
    "Caplet",
    "CapletSimple",
    "Floorlet",
    "FloorletSimple",
    "Cap",
    "CapSimple",
    "Floor",
    "FloorSimple",
]
