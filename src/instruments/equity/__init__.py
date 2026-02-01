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
    EquityVanillaEuropeanOption, EquityVanillaAmericanOption,
)

# Exotic options
from src.instruments.equity.options.barrier import EquityBarrierEuropeanOption
from src.instruments.equity.options.digital import EquityDigitalEuropeanOption
from src.instruments.equity.options.asian import EquityAsianEuropeanOption
from src.instruments.equity.options.lookback import EquityLookbackEuropeanOption

__all__ = [
    # Linear
    "EquitySpot",
    "EquityForward",
    # Vanilla
    "EquityVanillaEuropeanOption",
    "EquityVanillaAmericanOption",
    # Exotic
    "EquityBarrierEuropeanOption",
    "EquityDigitalEuropeanOption",
    "EquityAsianEuropeanOption",
    "EquityLookbackEuropeanOption",
]
