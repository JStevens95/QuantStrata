"""
Equity Options Module

Option instruments for equities:
- EuropeanEquityVanillaOption: European call/put
- AmericanEquityVanillaOption: American call/put (early exercise)
- EuropeanEquityBarrierOption: Single-barrier options
- EuropeanEquityDigitalOption: Cash/asset-or-nothing
- EuropeanEquityAsianOption: Average price options
- EuropeanEquityLookbackOption: Fixed/floating strike lookback

Author: QuantStrata Team
"""

from src.instruments.equity.options.vanilla import (
    EquityVanillaEuropeanOption, EquityVanillaAmericanOption,
)
from src.instruments.equity.options.barrier import EquityBarrierEuropeanOption
from src.instruments.equity.options.digital import EquityDigitalEuropeanOption
from src.instruments.equity.options.asian import EquityAsianEuropeanOption
from src.instruments.equity.options.lookback import EquityLookbackEuropeanOption

__all__ = [
    # Vanilla
    "EquityVanillaEuropeanOption",
    "EquityVanillaAmericanOption",
    # Exotic
    "EquityBarrierEuropeanOption",
    "EquityDigitalEuropeanOption",
    "EquityAsianEuropeanOption",
    "EquityLookbackEuropeanOption",
]
