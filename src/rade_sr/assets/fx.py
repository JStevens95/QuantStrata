"""
FX asset — self-contained market data + shocks for one FX risk factor.

One FXAsset = one currency pair (e.g. "EURUSD.SPOT").
Depends on two IRAsset instances (domestic + foreign).

After load():
    fx.spot                   # 1.0842
    fx.spot_series            # pd.Series (historical spot)
    fx.vol_surface            # VolSurface (tenors × delta-strikes)
    fx.forward_points         # pd.Series (tenor-indexed)
    fx.domestic_ir            # IRAsset (base currency, e.g. EUR)
    fx.foreign_ir             # IRAsset (quote currency, e.g. USD)
    fx.shocks                 # FXShocks (with labelled axes)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from .asset import Asset
from .types import AssetConfig, FXShocks, VolSurface

if TYPE_CHECKING:
    from .rates import IRAsset

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
# FX-specific configuration
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FXConfig:
    """Typed FX-specific settings extracted from AssetConfig.extra.

    Parameters
    ----------
    pair : str
        Currency pair without suffix (e.g. ``"EURUSD"``).
    base_ccy : str
        Base (domestic) currency (e.g. ``"EUR"``).
    quote_ccy : str
        Quote (foreign) currency (e.g. ``"USD"``).
    domestic_ir_extra : dict
        Extra config passed through to the domestic IRAsset's AssetConfig.
    foreign_ir_extra : dict
        Extra config passed through to the foreign IRAsset's AssetConfig.
    smile_strikes : list of str
        Delta convention labels for the smile pillars
        (e.g. ``["10DP", "25DP", "ATM", "25DC", "10DC"]``).
    """

    pair: str
    base_ccy: str
    quote_ccy: str
    domestic_ir_extra: Dict[str, Any] = field(default_factory=dict)
    foreign_ir_extra: Dict[str, Any] = field(default_factory=dict)
    smile_strikes: List[str] = field(default_factory=lambda: [
        "10DP", "25DP", "ATM", "25DC", "10DC",
    ])

    @classmethod
    def from_asset_config(cls, config: AssetConfig) -> FXConfig:
        """Build FXConfig from the generic AssetConfig.extra dict."""
        e = config.extra
        pair = config.asset_name.split(".")[0]
        return cls(
            pair=pair,
            base_ccy=e.get("base_ccy", pair[:3]),
            quote_ccy=e.get("quote_ccy", pair[3:6]),
            domestic_ir_extra=e.get("domestic_ir", {}),
            foreign_ir_extra=e.get("foreign_ir", {}),
            smile_strikes=e.get("smile_strikes", [
                "10DP", "25DP", "ATM", "25DC", "10DC",
            ]),
        )


# ─────────────────────────────────────────────────────────────────────────
# FXAsset
# ─────────────────────────────────────────────────────────────────────────


class FXAsset(Asset):
    """FX risk factor: spot, vol surface, IR curves, and typed shocks.

    Dependency graph::

        FXAsset("EURUSD.SPOT")
          ├── domestic_ir : IRAsset("EUR_DISC")
          └── foreign_ir  : IRAsset("USD_DISC")

    Fields set by _load (in addition to base-class spot / spot_series):
        vol_surface    — VolSurface (tenors × delta-strikes)
        forward_points — pd.Series (tenor year-fractions as index)
    """

    # ── Construction ─────────────────────────────────────────────────

    def __init__(self, config: AssetConfig) -> None:
        super().__init__(config)

        self.fx_config = FXConfig.from_asset_config(config)

        # FX-specific market data — populated by _load
        self.vol_surface: Optional[VolSurface] = None
        self.forward_points: Optional[pd.Series] = None

    # ── Config convenience accessors ─────────────────────────────────

    @property
    def pair(self) -> str:
        return self.fx_config.pair

    @property
    def base_ccy(self) -> str:
        return self.fx_config.base_ccy

    @property
    def quote_ccy(self) -> str:
        return self.fx_config.quote_ccy

    # ── Typed shocks accessor ────────────────────────────────────────

    @property
    def shocks(self) -> Optional[FXShocks]:
        """Narrow return type for FX shocks."""
        return self._shocks

    # ── Convenience accessors for dependencies ───────────────────────

    @property
    def domestic_ir(self) -> IRAsset:
        """Domestic (base currency) IR asset."""
        return self._dependencies["domestic_ir"]

    @property
    def foreign_ir(self) -> IRAsset:
        """Foreign (quote currency) IR asset."""
        return self._dependencies["foreign_ir"]

    # ══════════════════════════════════════════════════════════════════
    # Abstract method implementations
    # ══════════════════════════════════════════════════════════════════

    def _load_dependencies(self, api_client: Any) -> Dict[str, Asset]:
        """Load domestic and foreign IR curves as fully loaded IRAsset instances."""
        from .rates import IRAsset

        cfg = self.fx_config

        dom = IRAsset(AssetConfig(
            asset_class="rates",
            asset_name=f"{cfg.base_ccy}_DISC",
            cob_date=self.config.cob_date,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            calc_type=self.config.calc_type,
            extra=cfg.domestic_ir_extra,
        ))
        dom.load(api_client)

        fgn = IRAsset(AssetConfig(
            asset_class="rates",
            asset_name=f"{cfg.quote_ccy}_DISC",
            cob_date=self.config.cob_date,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            calc_type=self.config.calc_type,
            extra=cfg.foreign_ir_extra,
        ))
        fgn.load(api_client)

        return {"domestic_ir": dom, "foreign_ir": fgn}

    def _load(self, api_client: Any) -> None:
        """Load all FX market data.

        Orchestration:
            1. Fetch spot + history  → build self.spot, self.spot_series
            2. Fetch forward points  → build self.forward_points
            3. Fetch ATM + smile vol → build self.vol_surface
        """
        # 1. Spot + history
        spot_data = self._fetch_spot(api_client)
        self._build_spot(spot_data)

        # 2. Forward points
        fwd_data = self._fetch_forward_points(api_client)
        self._build_forward_points(fwd_data)

        # 3. Volatility surface
        atm_data = self._fetch_atm_vol(api_client)
        smile_data = self._fetch_smile_vol(api_client)
        self._build_vol_surface(atm_data, smile_data)

    def _load_shocks(self, api_client: Any) -> None:
        """Load FX shock scenarios into self._shocks (FXShocks)."""
        shock_data = self._fetch_shocks(api_client)
        self._build_shocks(shock_data)

    def validate(self) -> bool:
        """Check that all FX market data is present and internally consistent."""
        ok = True

        # Presence checks
        for check, msg in [
            (self.spot is not None, "spot not set"),
            (self.spot_series is not None, "spot_series not set"),
            (self.vol_surface is not None, "vol_surface not built"),
            ("domestic_ir" in self._dependencies, "domestic_ir missing"),
            ("foreign_ir" in self._dependencies, "foreign_ir missing"),
            (self._shocks is not None, "shocks not loaded"),
        ]:
            if not check:
                logger.warning("%s: %s", self.asset_name, msg)
                ok = False

        # Shock shape consistency
        if isinstance(self._shocks, FXShocks):
            for err in self._shocks.validate_shapes():
                logger.warning("%s: %s", self.asset_name, err)
                ok = False

            # Shock grids must align with base market data
            if self.vol_surface is not None:
                if not np.array_equal(self._shocks.vol_expiries, self.vol_surface.tenors):
                    logger.warning(
                        "%s: shock vol_expiries != vol_surface tenors", self.asset_name,
                    )
                    ok = False
                if not np.array_equal(self._shocks.vol_strikes, self.vol_surface.strikes):
                    logger.warning(
                        "%s: shock vol_strikes != vol_surface strikes", self.asset_name,
                    )
                    ok = False

        return ok

    # ══════════════════════════════════════════════════════════════════
    # Fetch methods — wire these to your internal API
    # ══════════════════════════════════════════════════════════════════

    def _fetch_spot(self, api_client: Any) -> Dict[str, Any]:
        """Fetch spot rate and historical time series.

        Delegates to ``MarketDataClient.get_fx_spot``; returns::

            {
                "spot":   1.0842,
                "dates":  ["2024-01-02", "2024-01-03", ...],
                "values": [1.0810, 1.0825, ...],
            }
        """
        return api_client.get_fx_spot(self.pair, self.config.cob_date)

    def _fetch_forward_points(self, api_client: Any) -> Dict[str, Any]:
        """Fetch forward points by tenor.

        Delegates to ``MarketDataClient.get_fx_forward_points``; returns::

            {
                "tenors": ["1W", "1M", "3M", "6M", "1Y"],
                "points": [0.00012, 0.00035, 0.00105, ...],
            }
        """
        return api_client.get_fx_forward_points(self.pair, self.config.cob_date)

    def _fetch_atm_vol(self, api_client: Any) -> Dict[str, float]:
        """Fetch ATM vol curve (tenor → vol).

        Delegates to ``MarketDataClient.get_fx_atm_vol``; returns::

            {"1W": 0.082, "1M": 0.091, "3M": 0.098, "6M": 0.102, "1Y": 0.108}
        """
        return api_client.get_fx_atm_vol(self.pair, self.config.cob_date)

    def _fetch_smile_vol(self, api_client: Any) -> Dict[str, Dict[str, float]]:
        """Fetch smile vol grid (tenor → delta → vol).

        Delegates to ``MarketDataClient.get_fx_smile_vol``; returns::

            {
                "1W": {"10DP": 0.092, "25DP": 0.086, "25DC": 0.078, "10DC": 0.074},
                "1M": {"10DP": 0.101, ...},
                ...
            }
        """
        return api_client.get_fx_smile_vol(self.pair, self.config.cob_date)

    def _fetch_shocks(self, api_client: Any) -> Dict[str, Any]:
        """Fetch FX shock scenarios.

        Delegates to ``MarketDataClient.get_fx_shocks``; returns::

            {
                "spot_shocks":            [...],          # (n_scenarios,)
                "vol_shocks":             [[[...]]],      # (n_scenarios, n_exp, n_strikes)
                "vol_expiries":           ["1W", "1M", ...],
                "vol_strikes":            ["10DP", "25DP", "ATM", "25DC", "10DC"],
                "domestic_rate_shocks":   [[...]],        # (n_scenarios, n_dom_pillars)
                "domestic_tenors":        ["3M", "6M", ...],
                "foreign_rate_shocks":    [[...]],        # (n_scenarios, n_for_pillars)
                "foreign_tenors":         ["3M", "6M", ...],
                "forward_point_shocks":   [[...]],        # optional
                "forward_tenors":         ["1W", ...],    # optional
            }
        """
        return api_client.get_fx_shocks(self.pair, self.config.cob_date)

    # ══════════════════════════════════════════════════════════════════
    # Build methods — fully implemented
    # ══════════════════════════════════════════════════════════════════

    def _build_spot(self, data: Dict[str, Any]) -> None:
        """Construct self.spot and self.spot_series from raw API response."""
        self.spot = float(data["spot"])
        self.spot_series = pd.Series(
            data=np.array(data["values"], dtype=np.float64),
            index=pd.to_datetime(data["dates"]),
            name=self.asset_name,
        ).sort_index()

    def _build_forward_points(self, data: Dict[str, Any]) -> None:
        """Construct self.forward_points from raw API response."""
        self.forward_points = pd.Series(
            data=np.array(data["points"], dtype=np.float64),
            index=[_tenor_to_years(t) for t in data["tenors"]],
            name=f"{self.pair}.FWD_PTS",
        )

    def _build_vol_surface(
        self,
        atm_data: Dict[str, float],
        smile_data: Dict[str, Dict[str, float]],
    ) -> None:
        """Merge ATM + smile into a single VolSurface (tenors × delta-strikes).

        ATM vol is placed in the centre column; missing smile deltas
        fall back to ATM.
        """
        tenors_sorted = sorted(atm_data.keys(), key=_tenor_to_years)
        tenor_years = np.array([_tenor_to_years(t) for t in tenors_sorted])
        strikes = self.fx_config.smile_strikes
        n_t, n_k = len(tenors_sorted), len(strikes)

        values = np.empty((n_t, n_k), dtype=np.float64)
        for i, tenor in enumerate(tenors_sorted):
            atm_vol = atm_data[tenor]
            row = smile_data.get(tenor, {})
            for j, delta in enumerate(strikes):
                if delta == "ATM":
                    values[i, j] = atm_vol
                else:
                    values[i, j] = row.get(delta, atm_vol)

        self.vol_surface = VolSurface(
            tenors=tenor_years,
            strikes=np.arange(n_k, dtype=np.float64),
            values=values,
            strike_type="delta",
            vol_type="lognormal",
            metadata={
                "tenor_labels": tenors_sorted,
                "strike_labels": strikes,
                "cob_date": str(self.config.cob_date),
            },
        )

    def _build_shocks(self, data: Dict[str, Any]) -> None:
        """Construct self._shocks (FXShocks) from raw API response."""
        fwd_pts = fwd_tenors = None
        if data.get("forward_point_shocks") is not None:
            fwd_pts = np.array(data["forward_point_shocks"], dtype=np.float64)
            fwd_tenors = np.array(
                [_tenor_to_years(t) for t in data["forward_tenors"]], dtype=np.float64,
            )

        self._shocks = FXShocks(
            spot=np.array(data["spot_shocks"], dtype=np.float64),
            vol_surface=np.array(data["vol_shocks"], dtype=np.float64),
            vol_expiries=np.array(
                [_tenor_to_years(t) for t in data["vol_expiries"]], dtype=np.float64,
            ),
            vol_strikes=np.arange(len(data["vol_strikes"]), dtype=np.float64),
            domestic_rate=np.array(data["domestic_rate_shocks"], dtype=np.float64),
            domestic_tenors=np.array(
                [_tenor_to_years(t) for t in data["domestic_tenors"]], dtype=np.float64,
            ),
            foreign_rate=np.array(data["foreign_rate_shocks"], dtype=np.float64),
            foreign_tenors=np.array(
                [_tenor_to_years(t) for t in data["foreign_tenors"]], dtype=np.float64,
            ),
            forward_points=fwd_pts,
            forward_tenors=fwd_tenors,
        )
