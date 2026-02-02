# src/instruments/ir/linear/__init__.py
"""
Linear Interest Rate Instruments.

This module contains linear (non-optional) interest rate instruments:
- Forward Rate Agreements (FRAs)
- Interest Rate Swaps (IRS)
- Bonds (Zero Coupon, Fixed Rate)
"""
from src.instruments.ir.linear.fra import (
    IrForwardRateAgreement,
    IrForwardRateAgreementSimple,
)
from src.instruments.ir.linear.swap import (
    IrSwap,
    IrSwapSimple,
    SwapLeg,
    FixedLeg,
    FloatingLeg,
)
from src.instruments.ir.linear.bond import (
    IrBondZeroCoupon,
    IrBondZeroCouponSimple,
    IrBondFixedRate,
    IrBondFixedRateSimple,
)

__all__ = [
    # FRA
    "IrForwardRateAgreement",
    "IrForwardRateAgreementSimple",
    # Swap
    "IrSwap",
    "IrSwapSimple",
    "SwapLeg",
    "FixedLeg",
    "FloatingLeg",
    # Bonds
    "IrBondZeroCoupon",
    "IrBondZeroCouponSimple",
    "IrBondFixedRate",
    "IrBondFixedRateSimple",
]
