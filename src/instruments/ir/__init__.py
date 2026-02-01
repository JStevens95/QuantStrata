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
    IrForwardRateAgreement, IrForwardRateAgreementSimple,
)
from src.instruments.ir.linear.swap import (
    IrSwap, IrSwapSimple, SwapLeg, FixedLeg, FloatingLeg,
)
from src.instruments.ir.options.capfloor import (
    IrCapletEuropeanOption, IrCapletEuropeanOptionSimple, IrFloorletEuropeanOption, IrFloorletEuropeanOptionSimple,
    IrCapEuropeanOption, IrCapEuropeanOptionSimple, IrFloorEuropeanOption, IrFloorEuropeanOptionSimple,
)

__all__ = [
    # Linear instruments
    "IrForwardRateAgreement",
    "IrForwardRateAgreementSimple",
    "IrSwap",
    "IrSwapSimple",
    "SwapLeg",
    "FixedLeg",
    "FloatingLeg",
    # Options
    "IrCapletEuropeanOption",
    "IrCapletEuropeanOptionSimple",
    "IrFloorletEuropeanOption",
    "IrFloorletEuropeanOptionSimple",
    "IrCapEuropeanOption",
    "IrCapEuropeanOptionSimple",
    "IrFloorEuropeanOption",
    "IrFloorEuropeanOptionSimple",
]
