"""
Asset data types - pure dataclasses for market data, configs and shocks.

These are lightweight containers used by Asset subclasses and pricing kernels.

Organised into foud groups:
    1. Configuration
    2. Market data contains.
    3. FX Shocks
    4. IR Shocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AssetConfig:
    """
    Immutable configuration for one asset.

    Captures what the asset *is* (identity, dates, calc settings).
    Does not hold market data - that is for Asset instance after load().
    """

    # defined parameters.
    asset_class: str
    asset_name: str
    asset_id: str
    cob_date: Optional[pd.Timestamp] = None
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    calc_type: str = "MAXSVAR"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Curve:
    """A term-structured curve (discount, projection, credit spread, etc)."""

    # define parameters.
    tenors: np.ndarray
    values: np.ndarray
    currency: str = ""
    curve_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_pillars(self) -> int:
        """Number of pillars on the curve."""
        return len(self.tenors)

    def rate_at(self, tenor: float) -> float:
        """Get the curve value at a specific tenor, using linear interpolation."""
        return float(np.interp(tenor, self.tenors, self.values))


@dataclass
class VolSurface:
    """A 2D volatility surface (tenors x strike)."""

    # define parameters.
    tenors: np.ndarray
    strikes: np.ndarray
    values: np.ndarray
    strike_type: str = "delta"
    vol_type: str = "lognormal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        """Shape of the vol surface (tenors x strike)."""
        return self.values.shape

    def vol_at(self, tenor: float, strike: float) -> float:
        """
        Get the vol value at a specific tenor and strike using bilinear interpolation.

        :param tenor:
        :param strike:
        :return:
        """
        # 1. interpolate across strikes at each bracketing tenor.
        t_idx = np.searchsorted(self.tenors, tenor)
        t_idx = int(np.clip(t_idx, 1, len(self.tenors) - 1))
        t0, t1 = self.tenors[t_idx - 1], self.tenors[t_idx]

        v0 = float(np.interp(strike, self.strikes, self.values[t_idx - 1]))
        v1 = float(np.interp(strike, self.strikes, self.values[t_idx]))

        # 2. linear interpolation between two tenors.
        if t1 == t0:
            return v0
        w = (tenor - t0) / (t1 - t0)
        return v0 + w * (v1 - v0)

    @property
    def atm_vol(self):
        """ATM vol at each tenor (mid/strike columns)"""
        mid_idx = self.values.shape[1] // 2
        return self.values[:, mid_idx]


@dataclass
class VolCube:
    """A 3D volatility cube for IR swaptions (expiry x tenor x strikes)."""

    # define parameters.
    expiries: np.ndarray
    tenors: np.ndarray
    strikes: np.ndarray
    values: np.ndarray
    vol_type: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        """Shape of the vol cube."""
        return self.values.shape


@dataclass
class FxShocks:
    """
    Scenario shocks for one FX risk factor.

    Each field carries its own axis labels so the arrays are self-documenting. Grids should align with the base market
    data on the FxAsset - the Asset's ``validate()`` checks this.
    """

    # define parameters.
    spot: np.ndarray

    vol_surface: np.ndarray
    vol_expiries: np.ndarray
    vol_strikes: np.ndarray

    domestic_rate: np.ndarray
    domestic_tenors: np.ndarray

    foreign_rate: np.ndarray
    foreign_tenors: np.ndarray

    forward_points: Optional[np.ndarray] = None
    forward_tenors: Optional[np.ndarray] = None

    @property
    def n_scenarios(self) -> int:
        """Return the number of scenarios."""
        return self.spot.shape[0]

    def validate_shapes(self) -> List[str]:
        """Return a list of shape inconsistency messages (empty=OK)."""
        errors: List[str] = []
        n = self.n_scenarios

        if self.vol_surface.shape[0] != n:
            errors.append(f"vol_surface scenarios {self.vol_surface.shape[0]} != {n}")
        if self.vol_surface.shape[1] != len(self.vol_expiries):
            errors.append(f"vol_surface expiries {self.vol_surface.shape[1]} != vol_expiries {len(self.vol_expiries)}")
        if self.vol_surface.shape[2] != len(self.vol_strikes):
            errors.append(f"vol_surface strikes {self.vol_surface.shape[2]} != vol_strikes {len(self.vol_strikes)}")

        if self.domestic_rate.shape != (n, len(self.domestic_tenors)):
            errors.append(f"domestic_rate shape {self.domestic_rate.shape} != ({n}, {len(self.domestic_tenors)})")
        if self.foreign_rate.shape != (n, len(self.foreign_tenors)):
            errors.append(f"foreign_rate shape {self.domestic_rate.shape} != ({n}, {len(self.foreign_tenors)})")

        if self.forward_points is not None and self.forward_tenors is not None:
            if self.forward_points.shape != (n, len(self.forward_tenors)):
                errors.append(f"forward_points shape {self.forward_points.shape} != ({n}, {len(self.forward_tenors)})")
        return errors


@dataclass
class IrShocks:
    """
    Scenario shocks for one IR risk factor.

    Each field carries its own axis labels so the arrays are self-documenting. Grid should align with the base market
    data on the IR asset.
    """

    # define parameters.
    curve: np.ndarray
    curve_tenors: np.ndarray

    vol_surface: Optional[np.ndarray] = None
    vol_expiries: Optional[np.ndarray] = None
    vol_tenors: Optional[np.ndarray] = None
    vol_strikes: Optional[np.ndarray] = None

    @property
    def n_scenarios(self) -> int:
        """Return the number of scenarios."""
        return self.curve.shape[0]

    def validate_shapes(self) -> List[str]:
        """Return a list of shape inconsistency messages (empty=OK)."""
        errors: List[str] = []
        n = self.n_scenarios

        if self.curve.shape != (n, len(self.curve_tenors)):
            errors.append(f"curve shape {self.curve.shape} != ({n}, {len(self.curve_tenors)})")

        if self.vol_surface is not None:
            if self.vol_expiries is None or self.vol_tenors is None:
                errors.append("vol surface provided but vol_expiries or vol_tenors is None")
            elif self.vol_strikes is None:
                errors.append("vol strikes provided but vol_strikes is None")
            else:
                expected = (n, len(self.vol_expiries), len(self.vol_tenors), len(self.vol_strikes))
                if self.vol_surface.shape != expected:
                    errors.append(f"vol surface shape {self.vol_surface.shape} != {expected}")
        return errors
