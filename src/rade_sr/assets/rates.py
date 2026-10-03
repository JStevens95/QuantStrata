"""
IR asset — self-contained market data + shocks for one interest rate risk factor.

One IRAsset = one curve = one risk factor (e.g. "EUR_DISC").
IR is a leaf in the dependency graph.

After load():
    ir.spot                   # par swap rate at reference tenor
    ir.spot_series            # pd.Series (reference tenor column from history)
    ir.curve                  # Curve object
    ir.rate_history           # pd.DataFrame (dates × tenors)
    ir.vol_cube               # VolCube (expiry × swap_tenor × strike)
    ir.vol_surface            # VolSurface (ATM slice of vol_cube)
    ir.shocks                 # IRShocks (with labelled axes)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .asset import Asset
from .types import AssetConfig, Curve, IRShocks, VolCube, VolSurface

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

_TENOR_MAP = {"D": 1 / 365, "W": 7 / 365, "M": 1 / 12, "Y": 1.0}


def _tenor_to_years(label: str) -> float:
    """Convert tenor label like '3M', '1Y', '2W' to year fraction."""
    for suffix, factor in _TENOR_MAP.items():
        if label.upper().endswith(suffix):
            return float(label[: -len(suffix)]) * factor
    return float(label)


# ─────────────────────────────────────────────────────────────────────────
# IR-specific configuration
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IRConfig:
    """Typed IR-specific settings extracted from AssetConfig.extra.

    Only holds identity and methodology — grids (pillar tenors, swaption
    expiries, strikes) are data-driven and discovered at load time.

    Parameters
    ----------
    currency : str
        Curve currency (e.g. ``"EUR"``, ``"USD"``).
    curve_type : str
        ``"ois"``, ``"libor"``, ``"sofr"``, etc.
    vol_type : str
        ``"normal"`` (bp vol) or ``"lognormal"``.
    reference_tenor : float
        Tenor (years) used as the "spot" reference rate (e.g. 0.5 = 6M).
    """

    currency: str
    curve_type: str = "ois"
    vol_type: str = "normal"
    reference_tenor: float = 0.5

    @classmethod
    def from_asset_config(cls, config: AssetConfig) -> IRConfig:
        """Build IRConfig from the generic AssetConfig.extra dict."""
        e = config.extra
        return cls(
            currency=e.get(
                "currency",
                config.asset_name.split("_")[0],
            ),
            curve_type=e.get("curve_type", "ois"),
            vol_type=e.get("vol_type", "normal"),
            reference_tenor=e.get("reference_tenor", 0.5),
        )


# ─────────────────────────────────────────────────────────────────────────
# IRAsset
# ─────────────────────────────────────────────────────────────────────────


class IRAsset(Asset):
    """Interest rate risk factor: curve, vol cube, and typed shocks.

    Fields set by _load (in addition to base-class spot / spot_series):
        curve        — Curve (single discount/projection curve)
        rate_history — pd.DataFrame (dates × tenor year-fractions)
        vol_cube     — VolCube (expiry × swap_tenor × strike)
        vol_surface  — VolSurface (ATM slice of vol_cube)
    """

    # ── Construction ─────────────────────────────────────────────────

    def __init__(self, config: AssetConfig) -> None:
        super().__init__(config)

        self.ir_config = IRConfig.from_asset_config(config)

        # IR-specific market data — populated by _load
        self.curve: Optional[Curve] = None
        self.rate_history: Optional[pd.DataFrame] = None
        self.vol_cube: Optional[VolCube] = None
        self.vol_surface: Optional[VolSurface] = None

    # ── Config convenience accessors ─────────────────────────────────

    @property
    def currency(self) -> str:
        return self.ir_config.currency

    @property
    def curve_type(self) -> str:
        return self.ir_config.curve_type

    @property
    def vol_type(self) -> str:
        return self.ir_config.vol_type

    @property
    def reference_tenor(self) -> float:
        return self.ir_config.reference_tenor

    # ── Typed shocks accessor ────────────────────────────────────────

    @property
    def shocks(self) -> Optional[IRShocks]:
        """Narrow return type for IR shocks."""
        return self._shocks

    # ══════════════════════════════════════════════════════════════════
    # Abstract method implementations
    # ══════════════════════════════════════════════════════════════════

    def _load_dependencies(self, api_client: Any) -> Dict[str, Asset]:
        """IR has no dependencies — it is a leaf node."""
        return {}

    def _load(self, api_client: Any) -> None:
        """Load all IR market data.

        Orchestration:
            1. Fetch current curve  → build self.curve
            2. Compute self.spot    = par_swap_rate(reference_tenor)
            3. Fetch history panel  → build self.rate_history, self.spot_series
            4. Fetch ATM + smile    → build self.vol_cube, self.vol_surface
        """
        # 1. Current curve (cob snapshot)
        curve_data = self._fetch_curve(api_client)
        self._build_curve(curve_data)

        # 2. Spot = par swap rate at the reference tenor
        self.spot = self.par_swap_rate(self.ir_config.reference_tenor)

        # 3. Historical curve panel
        history_data = self._fetch_history(api_client)
        self._build_history(history_data)

        # 4. Swaption volatility
        atm_data = self._fetch_atm_vol(api_client)
        smile_data = self._fetch_smile_vol(api_client)
        self._build_vol_cube(atm_data, smile_data)
        self._extract_atm_surface()

    def _load_shocks(self, api_client: Any) -> None:
        """Load IR shock scenarios into self._shocks (IRShocks)."""
        shock_data = self._fetch_shocks(api_client)
        self._build_shocks(shock_data)

    def validate(self) -> bool:
        """Check that all IR market data is present and internally consistent."""
        ok = True

        # Presence checks
        for check, msg in [
            (self.spot is not None, "spot not set"),
            (self.spot_series is not None, "spot_series not set"),
            (self.curve is not None, "curve not loaded"),
            (self.rate_history is not None, "rate_history not loaded"),
            (self._shocks is not None, "shocks not loaded"),
        ]:
            if not check:
                logger.warning("%s: %s", self.asset_name, msg)
                ok = False

        # Shock shape consistency
        if isinstance(self._shocks, IRShocks):
            for err in self._shocks.validate_shapes():
                logger.warning("%s: %s", self.asset_name, err)
                ok = False

            # Shock grids must align with base market data
            if self.curve is not None:
                if not np.array_equal(self._shocks.curve_tenors, self.curve.tenors):
                    logger.warning(
                        "%s: shock curve_tenors != curve tenors", self.asset_name,
                    )
                    ok = False

            if self.vol_cube is not None and self._shocks.vol is not None:
                for shock_ax, cube_ax, label in [
                    (self._shocks.vol_expiries, self.vol_cube.expiries, "vol_expiries"),
                    (self._shocks.vol_swap_tenors, self.vol_cube.swap_tenors, "vol_swap_tenors"),
                    (self._shocks.vol_strikes, self.vol_cube.strikes, "vol_strikes"),
                ]:
                    if shock_ax is not None and not np.array_equal(shock_ax, cube_ax):
                        logger.warning(
                            "%s: shock %s != vol_cube %s", self.asset_name, label, label,
                        )
                        ok = False

        return ok

    # ══════════════════════════════════════════════════════════════════
    # Fetch methods — wire these to your internal API
    # ══════════════════════════════════════════════════════════════════

    def _fetch_curve(self, api_client: Any) -> Dict[str, Any]:
        """Fetch current curve snapshot.

        Delegates to ``MarketDataClient.get_ir_curve``; returns::

            {
                "tenors": ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"],
                "rates":  [0.038, 0.039, 0.040, 0.041, 0.043, 0.044, 0.045],
            }
        """
        return api_client.get_ir_curve(self.asset_name, self.config.cob_date)

    def _fetch_history(self, api_client: Any) -> Dict[str, Any]:
        """Fetch historical curve panel.

        Delegates to ``MarketDataClient.get_ir_history``; returns::

            {
                "dates":  ["2024-01-02", "2024-01-03", ...],
                "tenors": ["3M", "6M", "1Y", "2Y", ...],
                "values": [[0.037, 0.038, ...], ...],   # (n_dates, n_tenors)
            }
        """
        return api_client.get_ir_history(self.asset_name, self.config.cob_date)

    def _fetch_atm_vol(self, api_client: Any) -> Dict[str, Any]:
        """Fetch ATM swaption vol surface.

        Delegates to ``MarketDataClient.get_ir_atm_vol``; returns::

            {
                "expiries":    ["3M", "6M", "1Y", "2Y", "5Y", "10Y"],
                "swap_tenors": ["1Y", "2Y", "5Y", "10Y", "30Y"],
                "values":      [[0.58, 0.62, ...], ...],   # (n_exp, n_ten)
            }
        """
        return api_client.get_ir_atm_vol(self.asset_name, self.config.cob_date)

    def _fetch_smile_vol(self, api_client: Any) -> Dict[str, Any]:
        """Fetch smile swaption vol cube.

        Delegates to ``MarketDataClient.get_ir_smile_vol``; returns::

            {
                "expiries":    ["3M", "6M", "1Y", "2Y", "5Y", "10Y"],
                "swap_tenors": ["1Y", "2Y", "5Y", "10Y", "30Y"],
                "strikes":     [-200, -100, -50, 0, 50, 100, 200],
                "values":      [[[...]]],   # (n_exp, n_ten, n_strikes)
            }
        """
        return api_client.get_ir_smile_vol(self.asset_name, self.config.cob_date)

    def _fetch_shocks(self, api_client: Any) -> Dict[str, Any]:
        """Fetch IR shock scenarios.

        Delegates to ``MarketDataClient.get_ir_shocks``; returns::

            {
                "curve_shocks":     [[...], ...],     # (n_scenarios, n_pillars)
                "curve_tenors":     ["3M", "6M", ...],
                "vol_shocks":       [[[[...]]]],      # (n_scen, n_exp, n_ten, n_strikes) or None
                "vol_expiries":     ["3M", "6M", ...],
                "vol_swap_tenors":  ["1Y", "2Y", ...],
                "vol_strikes":      [-200, -100, ...],
            }
        """
        return api_client.get_ir_shocks(self.asset_name, self.config.cob_date)

    # ══════════════════════════════════════════════════════════════════
    # Build methods — fully implemented
    # ══════════════════════════════════════════════════════════════════

    def _build_curve(self, data: Dict[str, Any]) -> None:
        """Construct self.curve from raw API response."""
        self.curve = Curve(
            tenors=np.array(
                [_tenor_to_years(t) for t in data["tenors"]], dtype=np.float64,
            ),
            values=np.array(data["rates"], dtype=np.float64),
            currency=self.ir_config.currency,
            curve_type=self.ir_config.curve_type,
        )

    def _build_history(self, data: Dict[str, Any]) -> None:
        """Construct self.rate_history and self.spot_series from raw API response."""
        tenor_years = [_tenor_to_years(t) for t in data["tenors"]]

        self.rate_history = pd.DataFrame(
            data=np.array(data["values"], dtype=np.float64),
            index=pd.to_datetime(data["dates"]),
            columns=tenor_years,
        ).sort_index()

        # spot_series = the column closest to the reference tenor
        nearest = min(
            self.rate_history.columns,
            key=lambda t: abs(t - self.ir_config.reference_tenor),
        )
        self.spot_series = self.rate_history[nearest].rename(self.asset_name)

    def _build_vol_cube(
        self,
        atm_data: Dict[str, Any],
        smile_data: Dict[str, Any],
    ) -> None:
        """Construct self.vol_cube from separate ATM and smile API responses."""
        expiries = np.array(
            [_tenor_to_years(t) for t in atm_data["expiries"]], dtype=np.float64,
        )
        swap_tenors = np.array(
            [_tenor_to_years(t) for t in atm_data["swap_tenors"]], dtype=np.float64,
        )
        strikes = np.array(smile_data["strikes"], dtype=np.float64)
        values = np.array(smile_data["values"], dtype=np.float64)

        self.vol_cube = VolCube(
            expiries=expiries,
            swap_tenors=swap_tenors,
            strikes=strikes,
            values=values,
            vol_type=self.ir_config.vol_type,
            metadata={
                "currency": self.ir_config.currency,
                "cob_date": str(self.config.cob_date),
                "expiry_labels": atm_data["expiries"],
                "swap_tenor_labels": atm_data["swap_tenors"],
            },
        )

    def _extract_atm_surface(self) -> None:
        """Extract the ATM (strike ≈ 0) slice from vol_cube as a VolSurface."""
        if self.vol_cube is None:
            return

        atm_idx = int(np.argmin(np.abs(self.vol_cube.strikes)))

        self.vol_surface = VolSurface(
            tenors=self.vol_cube.expiries,
            strikes=self.vol_cube.swap_tenors,
            values=self.vol_cube.values[:, :, atm_idx],
            strike_type="absolute",
            vol_type=self.ir_config.vol_type,
            metadata={"slice": "ATM", **self.vol_cube.metadata},
        )

    def _build_shocks(self, data: Dict[str, Any]) -> None:
        """Construct self._shocks (IRShocks) from raw API response."""
        curve_shocks = np.array(data["curve_shocks"], dtype=np.float64)
        curve_tenors = np.array(
            [_tenor_to_years(t) for t in data["curve_tenors"]], dtype=np.float64,
        )

        vol = vol_exp = vol_ten = vol_k = None
        if data.get("vol_shocks") is not None:
            vol = np.array(data["vol_shocks"], dtype=np.float64)
            vol_exp = np.array(
                [_tenor_to_years(t) for t in data["vol_expiries"]], dtype=np.float64,
            )
            vol_ten = np.array(
                [_tenor_to_years(t) for t in data["vol_swap_tenors"]], dtype=np.float64,
            )
            vol_k = np.array(data["vol_strikes"], dtype=np.float64)

        self._shocks = IRShocks(
            curve=curve_shocks,
            curve_tenors=curve_tenors,
            vol=vol,
            vol_expiries=vol_exp,
            vol_swap_tenors=vol_ten,
            vol_strikes=vol_k,
        )

    # ══════════════════════════════════════════════════════════════════
    # Rate computation — fully implemented
    # ══════════════════════════════════════════════════════════════════

    def par_swap_rate(self, maturity: float, freq: float = 0.5) -> float:
        """Par swap rate S such that S × annuity = 1 − DF(T).

        Parameters
        ----------
        maturity : float
            Swap maturity in years.
        freq : float
            Payment frequency in years (0.5 = semi-annual).
        """
        if self.curve is None:
            raise ValueError("curve not loaded")

        schedule = np.arange(freq, maturity + 1e-9, freq)
        dfs = np.array([self._df(t) for t in schedule])
        annuity = float(np.sum(dfs * freq))

        if annuity < 1e-12:
            return 0.0
        return float((1.0 - dfs[-1]) / annuity)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Simply-compounded forward rate F(t1, t2) = (DF1/DF2 − 1) / τ.

        Parameters
        ----------
        t1 : float
            Start of the forward period (years).
        t2 : float
            End of the forward period (years).  Must be > t1.
        """
        if self.curve is None:
            raise ValueError("curve not loaded")
        if t2 <= t1:
            raise ValueError(f"t2 ({t2}) must be > t1 ({t1})")

        return float((self._df(t1) / self._df(t2) - 1.0) / (t2 - t1))

    def _df(self, t: float) -> float:
        """Discount factor DF(t) = exp(−r(t) × t) via continuous compounding."""
        r = self.curve.rate_at(t)
        return float(np.exp(-r * t))
