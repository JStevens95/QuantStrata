"""
Equity Instruments Module

Complete suite of equity derivative instruments:
- Linear: Spot, Forward
- Vanilla: European and American options
- Exotic: Barrier, Digital, Asian, Lookback

Author: QuantStrata Team
"""

# Linear instruments
from src.instruments.equity.linear.spot import EquitySpot
from src.instruments.equity.linear.forward import EquityForward

# Vanilla options
from src.instruments.equity.options.vanilla import (
    EuropeanEquityVanillaOption,
    AmericanEquityVanillaOption,
)

# Exotic options
from src.instruments.equity.options.barrier import EuropeanEquityBarrierOption
from src.instruments.equity.options.digital import EuropeanEquityDigitalOption
from src.instruments.equity.options.asian import EuropeanEquityAsianOption
from src.instruments.equity.options.lookback import EuropeanEquityLookbackOption

__all__ = [
    # Linear
    "EquitySpot",
    "EquityForward",
    # Vanilla
    "EuropeanEquityVanillaOption",
    "AmericanEquityVanillaOption",
    # Exotic
    "EuropeanEquityBarrierOption",
    "EuropeanEquityDigitalOption",
    "EuropeanEquityAsianOption",
    "EuropeanEquityLookbackOption",
]
