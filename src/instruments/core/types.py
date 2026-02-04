from __future__ import annotations

from enum import Enum
from typing import Literal


# -----------------------------------------------------------------------------
# Currency (for settlement, notional, multi-currency instruments)
# -----------------------------------------------------------------------------


class Currency(str, Enum):
    """ISO 4217-style currency codes used in instruments (settlement, notional)."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"

# -----------------------------------------------------------------------------
# Core instrument definitions (type aliases)
# -----------------------------------------------------------------------------

# Standard option type (call or put).
OptionType = Literal["call", "put"]

# Define averaging type for Asian options
AsianAveragingType = Literal["arithmetic", "geometric"]

# Define lookback type: floating strike or fixed strike
LookbackType = Literal["floating_strike", "fixed_strike"]

# Touch style: one-touch pays if hit, no-touch pays if NOT hit.
TouchStyle = Literal["one_touch", "no_touch"]

# Digital option payout type
DigitalType = Literal["cash", "asset"]

# Day count convention type.
DayCountConvention = Literal["ACT/360", "ACT/365", "30/360"]

# Type for FRA direction.
FRADirection = Literal["payer", "receiver"]

# Type for swap direction.
SwapDirection = Literal["payer", "receiver"]

# Swaption type (payer or receiver).
SwaptionType = Literal["payer", "receiver"]

# Settlement style.
SettlementStyle = Literal["cash", "physical"]