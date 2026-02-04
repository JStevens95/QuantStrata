"""
Volatility Trading and Analytics Module.

Provides tools for:
- Variance swap pricing and hedging
- Dispersion trading strategies
- Volatility-of-volatility metrics
- Volatility surface analytics

Research Foundation:
- Carr & Madan (1998) "Towards a theory of volatility trading"
- Demeterfi et al. (1999) "A Guide to Volatility and Variance Swaps"

Example:
    from src.volatility import VarianceSwap, VarianceSwapPricer, DispersionTrader
    
    # Price variance swap
    swap = VarianceSwap(strike_var=0.04, maturity=0.5, notional=100_000)
    result = VarianceSwapPricer().price(swap, market_data)
    
    # Dispersion strategy
    trader = DispersionTrader(underlyings=["AAPL", "MSFT", "GOOGL"])
    signal = trader.analyze_dispersion(index_vol=0.18, stock_vols=[0.25, 0.22, 0.28])
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
from src.volatility.analytics.vol_of_vol import (
    VolOfVolAnalyzer,
    VolOfVolMetrics,
)

__all__ = [
    # Variance Swaps
    "VarianceSwap",
    "VarianceSwapPricer",
    "VarianceSwapResult",
    # Dispersion
    "DispersionTrader",
    "DispersionAnalysis",
    "DispersionConfig",
    # Vol-of-Vol
    "VolOfVolAnalyzer",
    "VolOfVolMetrics",
]
