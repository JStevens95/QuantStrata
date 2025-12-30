from __future__ import annotations

from typing import Dict
from dataclasses import dataclass

from src.instruments.fx.linear.forward import FxForward


@dataclass(frozen=True, slots=True)
class LinearFxForwardPricer:
    """
    Linear pricer for deliverable FX forwards.

    PV (domestic currency)
    ----------------------
        pv = Nf * ( S * df_f(T) - K * df_d(T) )

    Greeks (per 1 contract)
    -----------------------
    We report a minimal set that is stable and immediately useful:

    - delta_spot = dPV/dS = Nf * df_f(T)
    - gamma_spot = 0
    - vega = 0

    Notes
    -----
    - This is intentionally minimal but upgrade-friendly:
      Later you can add curve sensitivities (rho_domestic/rho_foreign) without
      changing the portfolio pricing interface.
    """

    def price(self, instrument: FxForward, market) -> float:  # noqa: ANN001, ANN401
        """
        Return PV in domestic currency.
        """
        S = float(market.quote(instrument.spot_id))

        df_d = float(market.curve(instrument.domestic_curve_id).df(float(instrument.expiry)))
        df_f = float(market.curve(instrument.foreign_curve_id).df(float(instrument.expiry)))

        Nf = float(instrument.notional_foreign)
        K = float(instrument.strike)

        return float(Nf * (S * df_f - K * df_d))

    def greeks(self, instrument: FxForward, market) -> Dict[str, float]:  # noqa: ANN001, ANN401
        """
        Return simple greeks for the forward.

        We use distinct key names to avoid confusion with option delta:
        - delta_spot is always meaningful for linear forwards/spot.
        """
        _ = market  # included for interface consistency

        df_f = float(market.curve(instrument.foreign_curve_id).df(float(instrument.expiry)))
        Nf = float(instrument.notional_foreign)

        return {
            "delta_spot": float(Nf * df_f),
            "gamma_spot": 0.0,
            "vega": 0.0,
        }