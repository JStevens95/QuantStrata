from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import numpy as np

from src.marketdata.ids import MarketId
from src.marketdata.providers.synthetic.specs import SpotGbmSpec, CurveZeroSpec, VolGridSmileSpec

@dataclass(frozen=True, slots=True)
class SyntheticProviderConfig:
    """
    Bundle of generator specs, with optional per-MarketId overrides.

    Design
    ------
    - Defaults live here (not as dozens of provider fields).
    - You can override per MarketId via exact key match.
    """
    spot: SpotGbmSpec = SpotGbmSpec(initial_level=1.10)
    curve: CurveZeroSpec = CurveZeroSpec(tenors=np.array([0.25, 0.5, 1.0, 2.0, 5.0]))
    vol: VolGridSmileSpec = VolGridSmileSpec(
        expiries=np.array([0.25, 0.5, 1.0, 2.0]),
        strikes=np.array([0.90, 1.00, 1.10, 1.20]),  # absolute strikes (V1)
    )

    spot_overrides: Mapping[MarketId, SpotGbmSpec] = field(default_factory=dict)
    curve_overrides: Mapping[MarketId, CurveZeroSpec] = field(default_factory=dict)
    vol_overrides: Mapping[MarketId, VolGridSmileSpec] = field(default_factory=dict)

    def spot_spec(self, mid: MarketId) -> SpotGbmSpec:
        return self.spot_overrides.get(mid, self.spot)

    def curve_spec(self, mid: MarketId) -> CurveZeroSpec:
        return self.curve_overrides.get(mid, self.curve)

    def vol_spec(self, mid: MarketId) -> VolGridSmileSpec:
        return self.vol_overrides.get(mid, self.vol)