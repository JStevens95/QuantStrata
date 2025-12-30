from __future__ import annotations

from typing import Dict
from dataclasses import dataclass

from src.instruments.fx.linear.spot import FxSpot


@dataclass(frozen=True, slots=True)
class LinearFxSpotPricer:
    """
    Linear pricer for `FxSpot`.

    Pricing
    -------
    PV per 1 unit of FxSpot exposure:
        pv = contract_multiplier * spot

    Portfolio sizing is applied outside this pricer:
        position_pv = Position.quantity * pv

    Greeks (per 1 instrument unit)
    ------------------------------
    - delta = contract_multiplier
    - gamma = 0
    - vega  = 0

    Notes
    -----
    This is intentionally minimal but upgrade-friendly:
    - You can later add currency conversion, funding, or settlement conventions
      without changing the PortfolioPricer contract.
    """

    def price(self, instrument: FxSpot, market) -> float:  # noqa: ANN001, ANN401
        """
        Return PV per 1 instrument unit.

        Parameters
        ----------
        instrument:
            FxSpot (one-unit exposure).
        market:
            Market snapshot providing quote(spot_id).

        Returns
        -------
        float
            contract_multiplier * spot_quote
        """
        spot_value = float(market.quote(instrument.spot_id))
        return float(instrument.contract_multiplier) * spot_value

    def greeks(self, instrument: FxSpot, market) -> Dict[str, float]:  # noqa: ANN001, ANN401
        """
        Return greeks per 1 instrument unit.

        Parameters
        ----------
        instrument:
            FxSpot instrument.
        market:
            Market snapshot (unused for linear greeks, included for signature consistency).

        Returns
        -------
        Dict[str, float]
            Keys: delta, gamma, vega
        """
        _ = market  # unused
        m = float(instrument.contract_multiplier)
        return {"delta": m, "gamma": 0.0, "vega": 0.0}