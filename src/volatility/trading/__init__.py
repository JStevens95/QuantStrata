"""
Volatility Trading Instruments and Strategies.

Provides:
- Variance swap pricing and hedging
- Dispersion trading
- Volatility arbitrage tools

Usage:
    from src.volatility.trading import VarianceSwap, VarianceSwapPricer
"""

from src.volatility.trading.variance_swap import (
    VarianceSwap,
    VarianceSwapPricer,
    VarianceSwapResult,
)
from src.volatility.trading.dispersion import (
    DispersionTrader,
    DispersionAnalysis,
    DispersionConfig,
)

__all__ = [
    "VarianceSwap",
    "VarianceSwapPricer",
    "VarianceSwapResult",
    "DispersionTrader",
    "DispersionAnalysis",
    "DispersionConfig",
]
