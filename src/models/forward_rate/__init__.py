"""
Forward Rate Models Module

This module provides forward rate models for interest rate derivatives:

Models:
- LIBOR Market Model (LMM): Multi-factor log-normal forward rate model

The LMM (also known as BGM model) is the industry standard for pricing
complex interest rate derivatives like caps, floors, and swaptions with
smile/skew effects.

Key Features:
- Log-normal forward rates (always positive)
- Multi-factor correlation structure
- Drift correction for no-arbitrage
- Monte Carlo simulation

Author: QuantStrata
Phase: 3.8 - LIBOR Market Model
"""
from src.models.forward_rate.lmm import (
    LMMParameters,
    LMMCorrelation,
    LMMDynamics,
    LMMSimulation,
)

__all__ = [
    "LMMParameters",
    "LMMCorrelation",
    "LMMDynamics",
    "LMMSimulation",
]
