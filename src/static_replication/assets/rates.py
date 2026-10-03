"""
IR asset - market data + shocks for a single interest rate risk factor.

A loaded IrAsset contains:
    - Spot (par rate at reference tenor) + historical IrCurve time-series.
    - Current discount curve.
    - Vol cube (expiry, tenor, strike) for swaptions - optional.\
    - Shocks: IrShocks with ir_spot_d_{tenor} and ir_spot_f_{tenor} columns.
"""
from __future__ import annotations

import logging
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.static_replication.assets.asset import Asset
from src.static_replication.assets.types import AssetConfig, Curve, VolCube, VolSurface
from src.static_replication.core.exceptions import MarketDataError, ValidationError
from src.static_replication.market_data.rates import IrCurve, IrAtmVol
from src.static_replication.market_data.shocks import IrShocks

# define module level logging.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IrConfig:
    """Additional IR-specific settings, extracted from AssetConfig.extra."""

    # define parameters.
    currency: str = ""
    curve_type: str = "ois"
    vol_type: str = "normal"
    reference_tenor: float = 0.5

    @classmethod
    def from_asset_config(cls, config: AssetConfig) -> IrConfig:
        """Build an IrConfig from the generic AssetConfig.extra dictionary."""
        e = config.extra
        name = config.asset_name
        # derive currency from asset_name if not in extra.
        parts = name.split("_")
        default_ccy = parts[1] if len(parts) >= 2 else ""
        return cls(
            currency=e.get("currency", default_ccy),
            curve_type=e.get("curve_type", "ois"),
            vol_type=e.get("vol_type", "normal"),
            reference_tenor=e.get("reference_tenor", 0.5),
        )


class IrAsset(Asset):
    """
    Interest Rate risk factor: curves, vol cube and shocks.

    No dependencies - IR is a building block.  FxAsset depends on IrAsset
    """

    def __init__(self, config: AssetConfig) -> None:
        """Initiate IrAsset instance from AssetConfig."""
        super().__init__(config)

        # define required variables.
        self.ir_config: IrConfig = IrConfig.from_asset_config(config)

        # market data objects (populated by _load())
        self.curve_series: Optional[IrCurve] = None  # full time-series.
        self.atm_vol_series: Optional[IrAtmVol] = None # atm vol time-series.

        # legacy fields kept for backward compat.
        self.curve: Optional[Curve] = None
        self.vol: Optional[VolCube] = None
        self.vol_surface: Optional[VolSurface] = None

    @property
    def shocks(self) -> Optional[IrShocks]:
        """Typed IR shock object with labeled axes."""
        return self._shocks  # type: ignore[return-value]

    @property
    def currency(self) -> str:
        """Currency code (e.g. GBP, USD)."""
        return self.ir_config.currency

    def _load_dependencies(self, api_client: Any) -> Dict[str, Asset]:
        """IR has not dependencies yet."""
        return {}

    def _load(self, api_client: Any) -> None:
        """
        Load all IR market data.

        Steps:
            1. Fetch curve time series -> build self.curve_series (IrCurve)
            2. Set self.spot -> par swap rate at reference_tenor on cob date.
            3. Fetch ATM vol if available -> build self.atm_vol_surface (IrAtmVol)

        :param self:
        :param api_client:
        :return:
        """
        rf_id = self.asset_name  # e.g. IR_CURVE_RFR_GBP.SONIA
        config_dict = {
            "cob_date": self.config.cob_date, "start_date": self.config.start_date, "end_date": self.config.end_date,
        }

        # 1. curve time series.
        try:
            curve_df = api_client.get_market_data(rf_id, config_dict)
            self.curve_series = IrCurve.from_dataframe(curve_df, rf_id=rf_id)
            self.curve_series.validate()
            logger.debug(" IrCurve loaded: %s %s", rf_id, self.curve_series.rates.shape)
        except Exception as exc:
            raise MarketDataError(f"Failed to load IR curve for {rf_id}: {exc}")

        # 2. spot = par rate at reference tenor on latest date.
        self.spot = self.curve_series.rate_at(date_idx=1, tenor=self.ir_config.reference_tenor)
        logger.debug(" IrAsset spot (par@%.2f): %.6f", self.ir_config.reference_tenor, self.spot)

        # 3. ATM vol (optional - swaptions).
        vol_rf_id = rf_id.replace("IR_CURVE_RFR", "IR_VOLATILITY_ATM")
        try:
            vol_df = api_client.get_market_data(vol_rf_id, config_dict)
            self.atm_vol_series = IrAtmVol.from_dataframe(vol_df, rf_id=rf_id)
            self.atm_vol_series.validate()
            logger.debug(" IrAsset loaded: %s %s", vol_rf_id, self.atm_vol_series.rates.shape)
        except Exception as exc:
            logger.debug(" IR ATM vol not available for %s: %s", rf_id, self.atm_vol_series.rates.shape)

    def _load_shocks(self, api_client: Any) -> None:
        """Load and format IR shocks scenarios into self._shocks (IrShocks)."""
        rf_id = self.asset_name
        config_dict = {
            "cob_date": self.config.cob_date, "start_date": self.config.start_date, "end_date": self.config.end_date,
        }
        try:
            shocks_df = api_client.get_market_data(rf_id, config_dict)
            self._shocks = IrShocks.from_dataframe(shocks_df, rf_id=rf_id)
            logger.debug(" IrShocks loaded: %s (%d scenarios)", rf_id, self.shocks.n_scenarios)
        except Exception as exc:
            raise MarketDataError(f"Failed to load IR shocks for {rf_id}: {exc}")

    def validate(self) -> None:
        """Check that all IR market data is present and internally consistent."""
        if self.curve_series is None:
            raise ValidationError(f"{self.asset_name}: curve_series not loaded.")
        if self.spot is None or np.isnan(self.spot):
            raise ValidationError(f"{self.asset_name}: spot (par rate) is None or NaN.")
        if self._shocks is None:
            raise ValidationError(f"{self.asset_name}: shocks not loaded.")
        if not isinstance(self._shocks, IrShocks):
            raise ValidationError(f"{self.asset_name}: shocks must be IrShocks, got {type(self._shocks)}.")
        logger.debug(" IrAsset validation passed: %s", self.asset_name)
