from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.models.payoffs.base import Payoff1D, _as_float_array
from src.models.payoffs.types import BarrierDirection, BarrierKnock, RebateTiming


@dataclass(frozen=True, slots=True)
class BarrierSpec:
    """
    Barrier definition (1D).

    direction:
      - 'up'   barrier is breached if S >= barrier
      - 'down' barrier is breached if S <= barrier

    knock:
      - 'out' knocks the option out on breach
      - 'in'  knocks the option in on breach
    """
    barrier: float
    direction: BarrierDirection
    knock: BarrierKnock
    rebate: float = 0.0
    rebate_timing: RebateTiming = "at_expiry"

    def __post_init__(self) -> None:
        if float(self.barrier) <= 0.0:
            raise ValueError("barrier must be > 0.")
        if self.direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'.")
        if self.knock not in ("in", "out"):
            raise ValueError("knock must be 'in' or 'out'.")
        if self.rebate_timing not in ("at_hit", "at_expiry"):
            raise ValueError("rebate_timing must be 'at_hit' or 'at_expiry'.")
        if not np.isfinite(float(self.rebate)):
            raise ValueError("rebate must be finite.")


@dataclass(frozen=True, slots=True)
class BarrierPayoff:
    """
    Wrapper that makes a base payoff into a barrier product.

    This object is path-dependent and therefore cannot be evaluated from S_T alone.
    Pricers should:
      - determine whether the option is alive at expiry (or knocked-in),
      - then call `finalize(spot_T, alive_mask, knocked_mask, hit_time_mask, ...)`.

    V1 behaviour:
      - We support rebate paid at expiry (common for simple barrier payoffs).
      - Rebate at hit is supported only if the pricer supplies a `hit_mask` (per path / per node).
    """
    base: Payoff1D
    barrier: BarrierSpec

    @property
    def is_path_dependent(self) -> bool:
        return True

    def terminal(self, spot: np.ndarray) -> np.ndarray:
        # Not meaningful alone for barriers; keep for convenience but document.
        return self.base.terminal(spot)

    def intrinsic(self, spot: np.ndarray) -> np.ndarray:
        # For American-barrier combos (rare in FX), this would need exercise + barrier logic.
        return self.base.intrinsic(spot)

    def finalize(
        self,
        *,
        spot_T: np.ndarray,
        is_hit: np.ndarray,
        cash_rebate_if_hit: bool = True,
    ) -> np.ndarray:
        """
        Convert base payoff into barrier payoff given barrier hit information.

        Parameters
        ----------
        spot_T:
            Terminal spot(s).
        is_hit:
            Boolean array: True if barrier was breached at any time (path monitoring result).
        cash_rebate_if_hit:
            If True and rebate_timing='at_expiry', pay rebate on hit.

        Returns
        -------
        payoff:
            Per-unit terminal payoff respecting knock-in/knock-out + rebate (at expiry).
        """
        sT = _as_float_array(spot_T)
        hit = np.asarray(is_hit, dtype=bool)

        base_pay = np.asarray(self.base.terminal(sT), dtype=np.float64)

        if self.barrier.knock == "out":
            alive = ~hit
        else:
            alive = hit

        payoff = base_pay * alive.astype(np.float64)

        if float(self.barrier.rebate) != 0.0 and cash_rebate_if_hit and self.barrier.rebate_timing == "at_expiry":
            payoff = payoff + float(self.barrier.rebate) * hit.astype(np.float64)

        return payoff