"""
Controlled vocabularies used across the library.

String-valued enums so they serialise cleanly into configs, manifests, and
artifact metadata while still giving type-safety and a single definition site.
"""
from __future__ import annotations

from enum import Enum


class AssetClass(str, Enum):
    """Built-in asset classes (extend by registering a new plugin)."""
    FX = "fx"
    RATES = "rates"


class PayoffType(str, Enum):
    """Elementary-trade payoff types."""
    CALL = "call"
    PUT = "put"
    DIGITAL_CALL = "digital_call"
    DIGITAL_PUT = "digital_put"
    FORWARD = "forward"
    PAYER = "payer"
    RECEIVER = "receiver"
    SWAP = "swap"


class StrikeConvention(str, Enum):
    """How a vol surface's strike axis is expressed."""
    ABSOLUTE = "absolute"
    MONEYNESS = "moneyness"   # K / forward
    DELTA = "delta"


class VolType(str, Enum):
    LOGNORMAL = "lognormal"   # Black / Garman-Kohlhagen
    NORMAL = "normal"         # Bachelier (bp vol)


class DayCount(str, Enum):
    ACT_365 = "act/365"
    ACT_360 = "act/360"


class Interpolation(str, Enum):
    """Curve interpolation rules."""
    LINEAR_ZERO = "linear_zero"          # linear on the zero rate
    LOG_LINEAR_DF = "log_linear_df"      # log-linear on discount factors (flat fwd)


class ShockMode(str, Enum):
    """How a scenario shock maps onto the COB base level."""
    ABSOLUTE = "absolute"          # scenario value IS the level
    ADDITIVE = "additive"          # level = base + shock
    RELATIVE = "relative"          # level = base * (1 + shock)
    LOG_RELATIVE = "log_relative"  # level = base * exp(shock)
