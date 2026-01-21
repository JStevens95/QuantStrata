from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Mapping

from src.marketdata.core.ids import MarketId
from src.marketdata.core.types import CurveMethod
from src.marketdata.providers.synthetic.specs import (
    SpotGbmSpec,
    CurveZeroSpec,
    CurveBootstrapSpec,
    VolGridSmileSpec,
)


@dataclass(frozen=True, slots=True)
class SyntheticProviderConfig:
    """
    Bundle of synthetic generator specs, with optional per-MarketId overrides.

    Curve generation
    ---------------
    curve_method selects how CURVE MarketIds are produced:
      - "zeros"     : generate a smooth zero curve directly (CurveZeroSpec)
      - "bootstrap" : generate deposit+swap quotes then bootstrap (CurveBootstrapSpec)

    Per-id overrides
    ----------------
    You can override *both* method and spec per MarketId by exact key match.
    """
    # ---- Defaults ----
    spot: SpotGbmSpec = SpotGbmSpec(initial_level=1.10)

    curve_zero: CurveZeroSpec = CurveZeroSpec(
        tenors=np.array([0.25, 0.5, 1.0, 2.0, 5.0], dtype=float)
    )
    curve_bootstrap: CurveBootstrapSpec = CurveBootstrapSpec()

    curve_method: CurveMethod = "zeros"  # "zeros" | "bootstrap"

    vol: VolGridSmileSpec = VolGridSmileSpec(
        expiries=np.array([0.25, 0.5, 1.0, 2.0], dtype=float),
        strikes=np.array([0.90, 1.00, 1.10, 1.20], dtype=float),
    )

    # ---- Overrides ----
    spot_overrides: Mapping[MarketId, SpotGbmSpec] = field(default_factory=dict)

    curve_zero_overrides: Mapping[MarketId, CurveZeroSpec] = field(default_factory=dict)
    curve_bootstrap_overrides: Mapping[MarketId, CurveBootstrapSpec] = field(default_factory=dict)
    curve_method_overrides: Mapping[MarketId, CurveMethod] = field(default_factory=dict)

    vol_overrides: Mapping[MarketId, VolGridSmileSpec] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors (centralize override logic)
    # ------------------------------------------------------------------

    def spot_spec(self, mid: MarketId) -> SpotGbmSpec:
        return self.spot_overrides.get(mid, self.spot)

    def curve_zero_spec(self, mid: MarketId) -> CurveZeroSpec:
        return self.curve_zero_overrides.get(mid, self.curve_zero)

    def curve_bootstrap_spec(self, mid: MarketId) -> CurveBootstrapSpec:
        return self.curve_bootstrap_overrides.get(mid, self.curve_bootstrap)

    def curve_method_for(self, mid: MarketId) -> str:
        method = str(self.curve_method_overrides.get(mid, self.curve_method)).strip().lower()
        if method not in {"zeros", "bootstrap"}:
            raise ValueError(f"Invalid curve_method={method!r} for MarketId={mid.key()}. Expected 'zeros'|'bootstrap'.")
        return method

    def vol_spec(self, mid: MarketId) -> VolGridSmileSpec:
        return self.vol_overrides.get(mid, self.vol)