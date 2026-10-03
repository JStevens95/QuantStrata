"""
Raw client payloads — the numeric wire format clients return.

These are deliberately *thin* (arrays + axis labels), with no behaviour: clients
produce payloads, the asset builders turn them into validated market objects. This
keeps client wiring trivial and the validation in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CurvePayload:
    """Zero-rate curve pillars."""
    tenors: np.ndarray
    zero_rates: np.ndarray


@dataclass(frozen=True)
class SurfacePayload:
    """FX vol surface grid."""
    expiries: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    strike_convention: str = "moneyness"
    vol_type: str = "lognormal"


@dataclass(frozen=True)
class CubePayload:
    """Swaption vol cube grid (normal vols)."""
    expiries: np.ndarray
    swap_tenors: np.ndarray
    strikes: np.ndarray
    vols: np.ndarray
    vol_type: str = "normal"


@dataclass(frozen=True)
class FXShockPayload:
    """Resolved (absolute) FX scenario states aligned to scenario ids."""
    scenario_ids: np.ndarray
    spot: np.ndarray
    vol: np.ndarray
    vol_expiries: np.ndarray
    vol_strikes: np.ndarray
    domestic_rate: np.ndarray
    domestic_tenors: np.ndarray
    foreign_rate: np.ndarray
    foreign_tenors: np.ndarray


@dataclass(frozen=True)
class RatesShockPayload:
    """Resolved (absolute) rates scenario states."""
    scenario_ids: np.ndarray
    curve: np.ndarray
    curve_tenors: np.ndarray
    vol: Optional[np.ndarray] = None
    vol_expiries: Optional[np.ndarray] = None
    vol_swap_tenors: Optional[np.ndarray] = None
    vol_strikes: Optional[np.ndarray] = None
