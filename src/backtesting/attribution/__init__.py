"""
P&L Attribution Module for Backtesting.

This module provides P&L decomposition and attribution:
- Greeks-based attribution (delta, gamma, theta, vega)
- Factor decomposition
- Daily/weekly/monthly aggregation
"""

from src.backtesting.attribution.pnl import (
    PnLAttribution,
    PnLBreakdown,
    attribute_pnl_to_greeks,
    aggregate_attribution,
)

__all__ = [
    "PnLAttribution",
    "PnLBreakdown",
    "attribute_pnl_to_greeks",
    "aggregate_attribution",
]
