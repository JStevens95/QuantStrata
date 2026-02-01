from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType, DigitalPayoff


@dataclass(frozen=True, slots=True)
class FxDigitalEuropeanOption:
    """
    European FX digital option with explicit payoff type.

    Payoff styles
    -------------
    cash:
        Pays `payout_amount` in *domestic currency* if in-the-money at expiry.
        Call: pays if S_T > K
        Put : pays if S_T < K

    asset:
        Pays `payout_amount` in *foreign currency units* if in-the-money at expiry.
        (i.e. "asset-or-nothing" where the "asset" is 1 unit of foreign currency)

    PV currency
    -----------
    Always returned in *domestic* currency.

    Notes
    -----
    Keeping this as a separate instrument type avoids an explosion of flags in a single class.
    """

    option_type: OptionType
    payoff: DigitalPayoff

    # notional: float               # <- NOT REQUIRED FOR DIGITAL
    payout_amount: float          # payoff per 1 unit notional

    strike: float
    expiry: float
    spot_id: MarketId
    vol_id: MarketId
    domestic_curve_id: MarketId
    foreign_curve_id: MarketId

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if self.payoff not in ("cash", "asset"):
            raise ValueError("payoff must be 'cash' or 'asset'.")
        if float(self.payout_amount) == 0.0:
            raise ValueError("payout_amount must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")