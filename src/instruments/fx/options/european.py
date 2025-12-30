from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

from src.marketdata.ids import MarketId

# define option type.
OptionType = Literal["call", "put"]


@dataclass(frozen=True, slots=True)
class EuropeanFxOption:
    """
    European FX option (Garman–Kohlhagen / Black–Scholes style).

    Parameters
    ----------
    option_type:
        "call" or "put".
    notional:
        Notional in domestic currency units (i.e. PV currency).
    strike:
        Strike expressed in FX quote convention (domestic per 1 foreign).
    expiry:
        Time to expiry in year fractions (>= 0).
    spot_id:
        MarketId for spot FX (kind="SPOT").
    vol_id:
        MarketId for implied vol surface (kind="VOL").
    domestic_curve_id:
        MarketId for domestic discount curve (kind="CURVE").
    foreign_curve_id:
        MarketId for foreign discount curve (kind="CURVE").

    Notes
    -----
    The pricer interprets:
      - S: spot FX (domestic per foreign)
      - r_d: domestic continuously-compounded rate
      - r_f: foreign continuously-compounded rate
      - sigma: implied vol

    The Garman–Kohlhagen PV (domestic currency) is:
      Call:  N * ( S*exp(-r_f T)*N(d1) - K*exp(-r_d T)*N(d2) )
      Put :  N * ( K*exp(-r_d T)*N(-d2) - S*exp(-r_f T)*N(-d1) )
    """
    # define option contract specs.
    option_type: OptionType
    notional: float
    strike: float
    expiry: float

    # define option market data.
    spot_id: MarketId
    vol_id: MarketId
    domestic_curve_id: MarketId
    foreign_curve_id: MarketId

    def __post_init__(self) -> None:
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'.")
        if not np.isfinite(self.notional) or self.notional <= 0.0:
            raise ValueError("notional must be finite and > 0.")
        if not np.isfinite(self.strike) or self.strike <= 0.0:
            raise ValueError("strike must be finite and > 0.")
        if not np.isfinite(self.expiry) or self.expiry < 0.0:
            raise ValueError("expiry must be finite and >= 0.")
