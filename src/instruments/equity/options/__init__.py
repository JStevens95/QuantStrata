"""
Equity Options Module

Option instruments for equities:
- EuropeanEquityVanillaOption: European call/put
- AmericanEquityVanillaOption: American call/put (early exercise)

Author: QuantStrata Team
"""

from src.instruments.equity.options.vanilla import (
    EuropeanEquityVanillaOption,
    AmericanEquityVanillaOption,
)

__all__ = [
    "EuropeanEquityVanillaOption",
    "AmericanEquityVanillaOption",
]
