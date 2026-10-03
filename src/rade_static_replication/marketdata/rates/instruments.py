"""
Rates market instrument — the swaption volatility cube.

Rates-specific in shape and convention: a 3-D cube in (option-expiry, swap-tenor, strike)
holding **normal** (bp) vols for Bachelier pricing — distinct from the 2-D lognormal FX
surface in ``marketdata/fx/instruments.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.rade_static_replication.domain.enums import VolType
from src.rade_static_replication.marketdata.base import as_1d, require_shape


@dataclass(frozen=True)
class VolCube:
    """Swaption vols ``vols[expiry, swap_tenor, strike]`` (normal/bp by default)."""
    expiries: np.ndarray
    swap_tenors: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    vol_type: VolType = VolType.NORMAL
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        e = as_1d("cube expiries", self.expiries)
        s = as_1d("cube swap_tenors", self.swap_tenors)
        k = as_1d("cube strikes", self.strikes)
        v = np.asarray(self.vols, dtype=np.float64)
        require_shape("cube vols", v, (e.size, s.size, k.size))
        object.__setattr__(self, "expiries", e)
        object.__setattr__(self, "swap_tenors", s)
        object.__setattr__(self, "strikes", k)
        object.__setattr__(self, "vols", v)
        object.__setattr__(self, "vol_type", VolType(self.vol_type))

    def vol(self, expiry: float, swap_tenor: float, strike: float) -> float:
        """Trilinear interpolation at ``(expiry, swap_tenor, strike)``."""
        ie = np.interp(expiry, self.expiries, np.arange(self.expiries.size))
        it = np.interp(swap_tenor, self.swap_tenors, np.arange(self.swap_tenors.size))
        ik = np.interp(strike, self.strikes, np.arange(self.strikes.size))

        def _lerp(arr: np.ndarray, idx: float) -> np.ndarray:
            lo = int(np.floor(idx))
            hi = min(lo + 1, arr.shape[0] - 1)
            w = idx - lo
            return (1.0 - w) * arr[lo] + w * arr[hi]

        return float(_lerp(_lerp(_lerp(self.vols, ie), it), ik))

    def validate(self) -> None:
        return None

    def summary(self) -> dict:
        return {
            "type": "VolCube",
            "shape": [int(self.expiries.size), int(self.swap_tenors.size), int(self.strikes.size)],
            "vol_type": self.vol_type.value,
        }
