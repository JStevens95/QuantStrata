from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

DeltaType = Literal["forward", "spot"]
AtmType = Literal["forward"]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (Acklam approximation), accurate enough for calibration/routing.

    Raises
    ------
    ValueError if p not in (0,1).
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")

    # Coefficients from Peter J. Acklam's approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


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
            d1 = _norm_ppf(target)  # N(d1) = target
        elif option_type == "put":
            d1 = -_norm_ppf(target)  # N(-d1)=target
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