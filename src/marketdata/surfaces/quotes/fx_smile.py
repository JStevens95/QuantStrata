from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from src.marketdata.surfaces.conventions.fx_vol import FxAtmConvention, FxDeltaConvention

@dataclass(frozen=True, slots=True)
class FxSmileSliceQuotes:
    """
    One-expiry FX smile quotes in ATM / RR / BF space.

    Stored as:
      - atm_vol: float
      - rr_by_delta: {0.25: RR25, 0.10: RR10, ...}  (call - put)
      - bf_by_delta: {0.25: BF25, 0.10: BF10, ...}  (butterfly add-on)

    Conversion (market standard):
      sigma_call(Δ) = atm + bf(Δ) + 0.5 * rr(Δ)
      sigma_put (Δ) = atm + bf(Δ) - 0.5 * rr(Δ)
    """
    expiry: float
    atm_vol: float
    rr_by_delta: Dict[float, float]
    bf_by_delta: Dict[float, float]

    delta_convention: FxDeltaConvention = FxDeltaConvention()
    atm_convention: FxAtmConvention = FxAtmConvention()

    surface_id: Optional[str] = None

    def deltas(self) -> List[float]:
        ds = sorted(set(self.rr_by_delta.keys()) | set(self.bf_by_delta.keys()))
        return [float(d) for d in ds]

    def vol_call(self, delta: float) -> float:
        d = float(delta)
        rr = float(self.rr_by_delta.get(d, 0.0))
        bf = float(self.bf_by_delta.get(d, 0.0))
        return float(self.atm_vol + bf + 0.5 * rr)

    def vol_put(self, delta: float) -> float:
        d = float(delta)
        rr = float(self.rr_by_delta.get(d, 0.0))
        bf = float(self.bf_by_delta.get(d, 0.0))
        return float(self.atm_vol + bf - 0.5 * rr)


@dataclass(frozen=True, slots=True)
class FxSmileQuotes:
    """
    Multi-expiry FX smile quotes (ATM/RR/BF) for one underlying.
    """
    slices: List[FxSmileSliceQuotes]

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("FxSmileQuotes.slices must not be empty.")
        # Sort by expiry for deterministic calibration
        object.__setattr__(self, "slices", sorted(self.slices, key=lambda s: float(s.expiry)))

    def expiries(self) -> List[float]:
        return [float(s.expiry) for s in self.slices]

    def __iter__(self) -> Iterable[FxSmileSliceQuotes]:
        return iter(self.slices)