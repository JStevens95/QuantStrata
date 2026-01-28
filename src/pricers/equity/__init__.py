"""
Equity Pricers Module

Pricing engines for equity derivatives:
- BSM (Black-Scholes-Merton) analytic pricing
- Monte Carlo simulation
- Finite Difference (PDE) methods
- American options with early exercise

Author: QuantStrata Team
"""

from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer
from src.pricers.equity.european_mc import EquityEuropeanVanillaMcPricer
from src.pricers.equity.european_fde import EquityEuropeanVanillaFdPricer
from src.pricers.equity.american_fde import EquityAmericanVanillaFdPricer
from src.pricers.equity.forward import EquityForwardPricer
from src.pricers.equity.spot import EquitySpotPricer

__all__ = [
    "EquityEuropeanVanillaBsmPricer",
    "EquityEuropeanVanillaMcPricer",
    "EquityEuropeanVanillaFdPricer",
    "EquityAmericanVanillaFdPricer",
    "EquityForwardPricer",
    "EquitySpotPricer",
]
