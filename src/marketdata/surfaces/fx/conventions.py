from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.core.math.normal import std_normal_ppf

DeltaType = Literal["forward", "spot"]
AtmType = Literal["forward"]


@dataclass(frozen=True, slots=True)
class FxDeltaConvention:
    """
    FX delta convention for mapping (delta, vol) -> strike.

    V1 supported:
      - delta_type="forward": Delta = exp(-rf*T) * N(d1) for calls; -exp(-rf*T)*N(-d1) for puts
      - delta_type="spot"   : Delta = N(d1) for calls; -N(-d1) for puts

    premium_adjusted:
      - Not implemented in V1 (would require PA delta formula & iterative solve).
    """
    delta_type: DeltaType = "forward"
    premium_adjusted: bool = False

    def __post_init__(self) -> None:
        if self.delta_type not in ("forward", "spot"):
            raise ValueError("FxDeltaConvention.delta_type must be 'forward' or 'spot'.")
        if self.premium_adjusted:
            raise NotImplementedError("premium-adjusted delta not implemented in V1.")

    def _delta_scale(self, *, df_for: float) -> float:
        """
        V1 supported (non-premium-adjusted):
          - delta_type="forward": |Δ_call| = N(d1), |Δ_put| = N(-d1)
          - delta_type="spot"   : |Δ_call| = DF_for(T) * N(d1), |Δ_put| = DF_for(T) * N(-d1)

        premium_adjusted:
          - Not implemented in V1.
        """
        # forward delta: Δ = N(d1)  => scale=1
        # spot delta:    Δ = DF_for * N(d1) => scale=DF_for
        return float(df_for) if self.delta_type == "spot" else 1.0

    def strike_from_abs_delta(
        self,
        *,
        option_type: Literal["call", "put"],
        abs_delta: float,
        spot: float,
        df_dom: float,
        df_for: float,
        vol: float,
        expiry: float,
    ) -> float:
        """
        Convert abs-delta quote into an absolute strike K using BSM FX relations.

        Uses:
          F = S * df_for / df_dom
          d1 = (ln(F/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))

        For forward-delta:
          |Δ_call| = exp(-rfT) N(d1)
          |Δ_put|  = exp(-rfT) N(-d1)
        """
        t = float(expiry)
        if t <= 0.0:
            return float(spot)

        sigma = float(vol)
        if sigma <= 0.0:
            raise ValueError("vol must be > 0.")

        s = float(spot)
        if s <= 0.0:
            raise ValueError("spot must be > 0.")

        df_d = float(df_dom)
        df_f = float(df_for)
        if df_d <= 0.0 or df_f <= 0.0:
            raise ValueError("discount factors must be > 0.")

        fwd = s * df_f / df_d

        scale = self._delta_scale(df_for=df_f)
        target = abs_delta / scale
        target = min(max(target, 1e-12), 1.0 - 1e-12)

        if option_type == "call":
            d1 = float(std_normal_ppf(np.array([target], dtype=np.float64))[0])
        elif option_type == "put":
            d1 = -float(std_normal_ppf(np.array([target], dtype=np.float64))[0])
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

        st = sigma * math.sqrt(t)
        ln_f_over_k = d1 * st - 0.5 * sigma * sigma * t
        k = fwd * math.exp(-ln_f_over_k)  # K = F * exp(-ln(F/K))
        return float(k)


@dataclass(frozen=True, slots=True)
class FxAtmConvention:
    """
    ATM convention (V1): ATM-forward only (K = F(T)).
    """
    atm_type: AtmType = "forward"

    def __post_init__(self) -> None:
        if self.atm_type != "forward":
            raise ValueError("FxAtmConvention.atm_type must be 'forward' in V1.")