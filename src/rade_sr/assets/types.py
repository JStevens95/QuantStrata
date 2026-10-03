"""
Asset data types — pure dataclasses for market data, configs, and shocks.

No logic, no ABC, no API client references.  These are lightweight
containers used by Asset subclasses and pricing kernels.

Organised into four groups:
  1. Configuration (AssetConfig)
  2. Market data containers (Curve, VolSurface, VolCube)
  3. FX shocks (FXShocks)
  4. IR shocks (IRShocks)

Each shock dataclass carries its own axis labels (tenors, strikes, etc.)
so the arrays are self-documenting.  The Asset's validate() method
checks that shock grids align with the base market data grids.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Configuration (immutable, created once)
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AssetConfig:
    """Immutable configuration for one asset.

    Captures what the asset *is* (identity, dates, calc settings).
    Does not hold market data — that lives on the Asset instance
    after load().

    Parameters
    ----------
    asset_class : str
        Asset class key: ``"fx"``, ``"rates"``, ``"eq"``, ``"cr"``.
    asset_name : str
        Canonical risk factor name (e.g. ``"EURUSD"``, ``"EUR_6M"``).
    asset_id : int
        Numeric identifier (from portfolio/API).
    cob_date : pd.Timestamp or None
        Close-of-business date for market data snapshot.
    start_date : pd.Timestamp or None
        Start of historical observation window.
    end_date : pd.Timestamp or None
        End of historical observation window.
    calc_type : str
        Shock calculation methodology (e.g. ``"MAXSVAR"``, ``"HISTORICAL"``).
    extra : dict
        Catch-all for asset-class-specific config that doesn't warrant
        a dedicated field (e.g. ``epic``, ``shock_generation`` params).
    """
    asset_class: str
    asset_name: str
    asset_id: int = 0
    cob_date: Optional[pd.Timestamp] = None
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    calc_type: str = "MAXSVAR"
    extra: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Market data containers
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Curve:
    """A term-structure curve (discount, projection, credit spread, etc.).

    Parameters
    ----------
    tenors : np.ndarray
        Pillar tenors in year fractions, shape ``(n_pillars,)``.
    values : np.ndarray
        Curve values at each pillar, shape ``(n_pillars,)``.
    currency : str
        Currency or issuer identifier.
    curve_type : str
        ``"discount"``, ``"projection"``, ``"credit_spread"``, etc.
    metadata : dict
        Day count, interpolation method, source, etc.
    """
    tenors: np.ndarray
    values: np.ndarray
    currency: str = ""
    curve_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_pillars(self) -> int:
        return len(self.tenors)

    def rate_at(self, tenor: float) -> float:
        """Linear interpolation at an arbitrary tenor.

        For production use, replace with your house interpolation
        (log-linear for discount factors, etc.).
        """
        return float(np.interp(tenor, self.tenors, self.values))


@dataclass
class VolSurface:
    """A 2D volatility surface (tenors x strikes).

    Parameters
    ----------
    tenors : np.ndarray
        Expiry tenors in year fractions, shape ``(n_tenors,)``.
    strikes : np.ndarray
        Strike axis (absolute, delta, or moneyness), shape ``(n_strikes,)``.
    values : np.ndarray
        Vol values, shape ``(n_tenors, n_strikes)``.
    strike_type : str
        ``"delta"``, ``"absolute"``, ``"moneyness"``.
    vol_type : str
        ``"lognormal"``, ``"normal"``, ``"shifted_lognormal"``.
    metadata : dict
        Source, interpolation method, business date, etc.
    """
    tenors: np.ndarray
    strikes: np.ndarray
    values: np.ndarray
    strike_type: str = "delta"
    vol_type: str = "lognormal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return self.values.shape

    def vol_at(self, tenor: float, strike: float) -> float:
        """Bilinear interpolation at an arbitrary (tenor, strike) point.

        For production, replace with SABR or your house interpolator.
        """
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.tenors, self.strikes), self.values,
            method="linear", bounds_error=False, fill_value=None,
        )
        return float(np.asarray(interp([[tenor, strike]])).ravel()[0])

    @property
    def atm_vols(self) -> np.ndarray:
        """ATM vol at each tenor (mid-strike column)."""
        mid_idx = self.values.shape[1] // 2
        return self.values[:, mid_idx]


@dataclass
class VolCube:
    """A 3D volatility cube for IR swaptions (expiry x tenor x strike).

    Parameters
    ----------
    expiries : np.ndarray
        Option expiries, shape ``(n_expiries,)``.
    swap_tenors : np.ndarray
        Underlying swap tenors, shape ``(n_tenors,)``.
    strikes : np.ndarray
        Strike axis, shape ``(n_strikes,)``.
    values : np.ndarray
        Vol values, shape ``(n_expiries, n_tenors, n_strikes)``.
    vol_type : str
        ``"normal"`` (bp vol) or ``"lognormal"``.
    """
    expiries: np.ndarray
    swap_tenors: np.ndarray
    strikes: np.ndarray
    values: np.ndarray
    vol_type: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return self.values.shape


# ─────────────────────────────────────────────────────────────────────
# FX Shocks
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FXShocks:
    """Scenario shocks for one FX risk factor.

    Each field carries its own axis labels so the arrays are
    self-documenting.  Grids should align with the base market data
    on the FXAsset — the Asset's ``validate()`` checks this.

    Parameters
    ----------
    spot : np.ndarray
        Shocked spot levels, shape ``(n_scenarios,)``.
    vol_surface : np.ndarray
        Shocked vol surface, shape ``(n_scenarios, n_expiries, n_strikes)``.
    vol_expiries : np.ndarray
        Expiry axis for vol shocks in year fractions, shape ``(n_expiries,)``.
    vol_strikes : np.ndarray
        Strike axis for vol shocks (delta or absolute), shape ``(n_strikes,)``.
    domestic_rate : np.ndarray
        Shocked domestic IR curve, shape ``(n_scenarios, n_dom_tenors)``.
    domestic_tenors : np.ndarray
        Tenor axis for domestic rate shocks, shape ``(n_dom_tenors,)``.
    foreign_rate : np.ndarray
        Shocked foreign IR curve, shape ``(n_scenarios, n_for_tenors)``.
    foreign_tenors : np.ndarray
        Tenor axis for foreign rate shocks, shape ``(n_for_tenors,)``.
    forward_points : np.ndarray or None
        Shocked forward points, shape ``(n_scenarios, n_fwd_tenors)``.
        Optional — not all shock methodologies produce these.
    forward_tenors : np.ndarray or None
        Tenor axis for forward point shocks, shape ``(n_fwd_tenors,)``.
    """
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
        return self.spot.shape[0]

    def validate_shapes(self) -> List[str]:
        """Return a list of shape inconsistency messages (empty = OK)."""
        errors: List[str] = []
        n = self.n_scenarios

        if self.vol_surface.shape[0] != n:
            errors.append(
                f"vol_surface scenarios {self.vol_surface.shape[0]} != spot scenarios {n}"
            )
        if self.vol_surface.shape[1] != len(self.vol_expiries):
            errors.append(
                f"vol_surface expiries {self.vol_surface.shape[1]} != "
                f"vol_expiries {len(self.vol_expiries)}"
            )
        if self.vol_surface.shape[2] != len(self.vol_strikes):
            errors.append(
                f"vol_surface strikes {self.vol_surface.shape[2]} != "
                f"vol_strikes {len(self.vol_strikes)}"
            )
        if self.domestic_rate.shape != (n, len(self.domestic_tenors)):
            errors.append(
                f"domestic_rate shape {self.domestic_rate.shape} != "
                f"({n}, {len(self.domestic_tenors)})"
            )
        if self.foreign_rate.shape != (n, len(self.foreign_tenors)):
            errors.append(
                f"foreign_rate shape {self.foreign_rate.shape} != "
                f"({n}, {len(self.foreign_tenors)})"
            )
        if self.forward_points is not None and self.forward_tenors is not None:
            if self.forward_points.shape != (n, len(self.forward_tenors)):
                errors.append(
                    f"forward_points shape {self.forward_points.shape} != "
                    f"({n}, {len(self.forward_tenors)})"
                )
        return errors


# ─────────────────────────────────────────────────────────────────────
# IR Shocks
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IRShocks:
    """Scenario shocks for one IR risk factor.

    Each field carries its own axis labels.  Grids should align with
    the base market data on the IRAsset.

    Parameters
    ----------
    curve : np.ndarray
        Shocked curve rates, shape ``(n_scenarios, n_pillars)``.
    curve_tenors : np.ndarray
        Tenor axis for curve shocks, shape ``(n_pillars,)``.
    vol : np.ndarray or None
        Shocked swaption/cap vol,
        shape ``(n_scenarios, n_expiries, n_swap_tenors, n_strikes)``.
    vol_expiries : np.ndarray or None
        Expiry axis for vol shocks, shape ``(n_expiries,)``.
    vol_swap_tenors : np.ndarray or None
        Swap tenor axis for vol shocks, shape ``(n_swap_tenors,)``.
    vol_strikes : np.ndarray or None
        Strike axis for vol shocks, shape ``(n_strikes,)``.
    """
    curve: np.ndarray
    curve_tenors: np.ndarray
    vol: Optional[np.ndarray] = None
    vol_expiries: Optional[np.ndarray] = None
    vol_swap_tenors: Optional[np.ndarray] = None
    vol_strikes: Optional[np.ndarray] = None

    @property
    def n_scenarios(self) -> int:
        return self.curve.shape[0]

    def validate_shapes(self) -> List[str]:
        """Return a list of shape inconsistency messages (empty = OK)."""
        errors: List[str] = []
        n = self.n_scenarios

        if self.curve.shape != (n, len(self.curve_tenors)):
            errors.append(
                f"curve shape {self.curve.shape} != ({n}, {len(self.curve_tenors)})"
            )
        if self.vol is not None:
            if self.vol_expiries is None or self.vol_swap_tenors is None:
                errors.append("vol provided but vol_expiries or vol_swap_tenors is None")
            elif self.vol_strikes is None:
                errors.append("vol provided but vol_strikes is None")
            else:
                expected = (
                    n,
                    len(self.vol_expiries),
                    len(self.vol_swap_tenors),
                    len(self.vol_strikes),
                )
                if self.vol.shape != expected:
                    errors.append(f"vol shape {self.vol.shape} != {expected}")
        return errors
