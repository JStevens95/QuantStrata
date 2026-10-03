"""
FX market instruments — spot quote and the FX implied-vol surface.

These are FX-specific in both shape and convention: the surface is 2-D in
(expiry, strike-axis) with an FX strike convention (moneyness / delta / absolute) and
lognormal vols. Rates' instrument (the 3-D normal-vol cube) lives in
``marketdata/rates/instruments.py`` — different dimensions, different convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.rade_static_replication.domain.enums import StrikeConvention, VolType
from src.rade_static_replication.marketdata.base import (
    as_1d,
    require_increasing,
    require_positive,
    require_shape,
    total_variance_interp,
)


@dataclass(frozen=True)
class Spot:
    """FX spot, quoted domestic-per-foreign for an ``XXXYYY`` pair."""
    pair: str
    value: float

    def __post_init__(self) -> None:
        require_positive(f"{self.pair} spot", self.value)

    @property
    def label(self) -> str:
        return f"FX.SPOT.{self.pair}"

    def validate(self) -> None:
        return None

    def summary(self) -> dict:
        return {"type": "Spot", "pair": self.pair, "value": float(self.value)}


@dataclass(frozen=True)
class VolSurface:
    """FX implied-vol surface ``vols[expiry, strike]``.

    Interpolates linearly in the strike axis and in **total variance** along expiry (the
    no-arbitrage-leaning default). A SABR/SVI upgrade slots in behind :meth:`vol`.

    Parameters
    ----------
    expiries : np.ndarray
        Expiry tenors (years), strictly increasing.
    strikes : np.ndarray
        Strike axis (units per ``strike_convention``), strictly increasing.
    vols : np.ndarray
        Implied vols, shape ``(n_exp, n_k)``.
    strike_convention : StrikeConvention
    vol_type : VolType
    """
    expiries: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    strike_convention: StrikeConvention = StrikeConvention.MONEYNESS
    vol_type: VolType = VolType.LOGNORMAL
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        exp = as_1d("vol expiries", self.expiries)
        k = as_1d("vol strikes", self.strikes)
        v = np.asarray(self.vols, dtype=np.float64)
        require_shape("vols", v, (exp.size, k.size))
        require_increasing("vol expiries", exp)
        object.__setattr__(self, "expiries", exp)
        object.__setattr__(self, "strikes", k)
        object.__setattr__(self, "vols", v)
        object.__setattr__(self, "strike_convention", StrikeConvention(self.strike_convention))
        object.__setattr__(self, "vol_type", VolType(self.vol_type))

    def vol(self, strike: float, expiry: float) -> float:
        """Interpolated vol at ``(strike, expiry)`` (strike units per convention)."""
        vols_by_exp = np.array([
            np.interp(strike, self.strikes, self.vols[i]) for i in range(self.expiries.size)
        ])
        return total_variance_interp(expiry, self.strikes, vols_by_exp, self.expiries)

    def validate(self) -> None:
        return None

    def summary(self) -> dict:
        return {
            "type": "VolSurface",
            "n_expiries": int(self.expiries.size), "n_strikes": int(self.strikes.size),
            "strike_convention": self.strike_convention.value, "vol_type": self.vol_type.value,
        }
