# src/instruments/ir/linear/__init__.py
"""
Linear Interest Rate Instruments.

This module contains linear (non-optional) interest rate instruments:
- Forward Rate Agreements (FRAs)
- Interest Rate Swaps (IRS)
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

__all__ = [
    # FRA
    "ForwardRateAgreement",
    "ForwardRateAgreementSimple",
    # Swap
    "InterestRateSwap",
    "InterestRateSwapSimple",
    "SwapLeg",
    "FixedLeg",
    "FloatingLeg",
]
