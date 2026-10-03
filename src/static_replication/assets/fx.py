"""
FX asset - self-contained market data + shocks for one FX risk factor.

Depends on two IR Asset instances for domestic and foreign curves.
"""
from __future__ import annotations

import logging
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.static_replication.assets.asset import Asset
from src.static_replication.assets.types import AssetConfig
from src.static_replication.core.exceptions import MarketDataError, ValidationError
from src.static_replication.market_data.fx import (
    FxSpot, FxAtmVol, FxSmileVol, FxVolSurface
)
from src.static_replication.market_data.shocks import FxShocks

if TYPE_CHECKING:
    from src.static_replication.assets.rates import IrAsset

# define module level logging.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FxConfig:
    """
    Types FX-specific configuration, extract from AssetConfig.extra

    Expected keys in AssetConfig.exta:
        pair - e.g. "GBPUSD", "GBPEUR"
        base_ccy - e.g. "GBP", "EUR"
        quote_ccy - e.g. "USD"
        domestic_ir_config - dict of IR config for base currency.
        foreign_ir_config - dict of IR config for foreign currency.
    """

    # define parameters.
    pair: str
    base_ccy: str
    quote_ccy: str
    domestic_ir_config: Dict[str, Any] = field(default_factory=dict)
    foreign_ir_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_asset_config(cls, config: AssetConfig) -> FxConfig:
        """Build FxConfig from the generic AssetConfig.extra dictionary."""

        e = config.extra
        name = config.asset_name
        # derive pair / currencies from asset_name if not explicitly in extra
        parts = name.split("_")
        default_quote = parts[1] if len(parts) >= 3 else None
        default_base = parts[2] if len(parts) >= 4 else None
        return cls(
            pair=e.get("pair", f"{default_base}{default_quote}"),
            base_ccy=e.get("base_ccy", default_base),
            quote_ccy=e.get("quote_ccy", default_base),
            domestic_ir_config=e.get("domestic_ir_config", {}),
            foreign_ir_config=e.get("foreign_ir_config", {}),
        )


class FxAsset(Asset):
    """FX risk factor: spot, vol surface, IR curves and typed shocks."""

    def __init__(self, config: AssetConfig) -> None:
        """Initiate FxAsset instance with AssetConfig."""
        super().__init__(config)

        # define required variables.
        self.fx_config: FxConfig = FxConfig.from_asset_config(config)

        # market data objects (populated by _load())
        self.spot_series: Optional[FxSpot] = None
        self.atm_vol: Optional[FxAtmVol] = None
        self.smile_vol: Optional[FxSmileVol] = None

        # unified vol surface - built from atm_vol + smile_vol (if available) after both are loaded.
        self.vol_surface: Optional[FxVolSurface] = None

        # use private backing vars for domestic/forieng ir to avoid showing the @propery methods below.
        self._domestic_ir: Optional[IrAsset] = None
        self._foreign_ir: Optional[IrAsset] = None

    @property
    def pair(self) -> str:
        """Return currency pair."""
        return self.fx_config.pair

    @property
    def base_ccy(self) -> str:
        """Return currency base."""
        return self.fx_config.base_ccy

    @property
    def quote_ccy(self) -> str:
        """Return currency quote."""
        return self.fx_config.quote_ccy

    @property
    def shocks(self) -> Optional[FxShocks]:
        """Return typed FX shock object with labeled axes."""
        return self._shocks  # type: ignore[return-value]

    @property
    def domestic_ir(self) -> Optional[IrAsset]:
        """Return (base ccy) IR asset."""
        return self.dependencies.get("domestic_ir")

    @property
    def foreign_ir(self) -> Optional[IrAsset]:
        """Return (quote ccy) IR asset."""
        return self.dependencies.get("foreign_ir")

    def _load_dependencies(self, api_client: Any) -> Dict[str, Asset]:
        """Load domestic and foreign IR curves as fully loaded IrAsset instances."""
        from src.static_replication.assets.rates import IrAsset

        cfg = self.fx_config
        deps: Dict[str, Asset] = {}

        for key, ir_cfg_dict, label in [
            ("domestic_ir", cfg.domestic_ir_config, "domestic"),
            ("foreign_ir", cfg.foreign_ir_config, "foreign")
        ]:
            if not ir_cfg_dict:
                logger.debug(" no %s IR config provided for %s - skipping", label, self.asset_name)
                continue

            # construct IR asset configuration.
            if_config = AssetConfig(
                asset_class="IR",
                asset_name=ir_cfg_dict.get("asset_name", ""),
                cob_date=self.config.cob_date,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                calc_type=self.config.calc_type,
                extra=ir_cfg_dict.get("extra", {})
            )

            # build ir asset
            ir_asset = IrAsset(config=if_config)
            ir_asset.load(api_client)
            deps[key] = ir_asset

        return deps

    def _load(self, api_client: Any) -> None:
        """
        Load all FX market data.

        Steps:
            1. Fetch spot time-series -> build self.spot_series (FxSpot)
            2. Fetch ATM vol -> build self.atm_vol (FxAtmVol)
            3. Fetch smile vol -> build self.smile_vol (FxSmileVol) if available.
            4. Build self.vol_surface (FxVolSurface) - smile if available, ATM-only otherwise.
            5. Set self.spot from latest value in spot_series.

        :param api_client:
        :return:
        """
        rf_id = self.asset_name     # e.g. FX_SPOT.USD.GBP
        config_dict = {
            "cob_date": self.config.cob_date, "start_date": self.config.start_date, "end_date": self.config.end_date,
        }

        # 1. spot time-series.
        try:
            spot_df = api_client.get_market_data(rf_id, config_dict)
            self.spot_series = FxSpot.from_dataframe(spot_df, rf_id=rf_id)
            self.spot_series.validate()
            self.spot = float(self.spot_series.spot[-1])  # <- check this is correct or should we use cob dat from config
            logger.debug(" FxSpot loaded: %s (cob data spot=%.4f)", rf_id, self.spot)
        except Exception as exc:
            raise MarketDataError(f"Failed to load FxSpot for {rf_id}: {exc}") from exc

        # 2. ATM vol surface.
        atm_rf_id = rf_id.replace("FX_SPOT", "FX_VOLATILTIY_ATM")
        # swap pair order: FX_SPOT.USD.GBP --> FX_VOLATILITY_ATM.GBP.USD
        parts = atm_rf_id.split(".")
        if len(parts) == 3:
            atm_rf_id = f"{parts[0]}.{parts[1]}.{parts[2]}"
        try:
            atm_df = api_client.get_market_data(atm_rf_id, config_dict)
            self.atm_vol = FxAtmVol.from_dataframe(atm_df, rf_id=rf_id)
            self.atm_vol.validate()
            logger.debug(" FxAtmVol loaded: %s %s", rf_id, self.atm_vol.vols.shape)
        except Exception as exc:
            raise MarketDataError(f"Failed to load FxAtmVol for {rf_id}: {exc}") from exc

        # 3. Smile vol surface (optional - do not raise if absent).
        smile_rf_id = rf_id.replace("FX_SPOT", "FX_VOLATILITY_SMILE")
        parts_smile = smile_rf_id.split(".")
        if len(parts_smile) == 3:
            smile_rf_id = f"{parts[0]}.{parts[2]}.{parts[1]}"
        try:
            smile_df = api_client.get_market_data(smile_rf_id, config_dict)
            # FxSmileVol.from_dataframe parse tenor/moneyness from column names.
            self.smile_vol = FxSmileVol.from_dataframe(smile_df, rf_id=rf_id)
            self.smile_vol.validate()
            logger.debug(" FxSmileVol loaded: %s %s", smile_rf_id, self.smile_vol.vols.shape)
        except Exception as exc:
            logger.warning(f" FxSmileVol not available for %s: %s", rf_id, exc)
            self.smile_vol = None

        # 4. build unified vol surface - always present if ATM vol loaded.
        if self.atm_vol is not None:
            self.vol_surface = FxVolSurface.build(atm_vol=self.atm_vol, smile_vol=self.smile_vol)
            logger.debug(" FxVolSurface built for %s (smile=%s)", rf_id, self.smile_vol is not None)

    def _load_shocks(self, api_client: Any) -> None:
        """Load FX shock scenarios into self._shocks (FxShocks)."""
        rf_id = self.asset_name
        config_dict = {
            "cob_date": self.config.cob_date, "start_date": self.config.start_date, "end_date": self.config.end_date
        }
        try:
            shocks_df = api_client.get_shocks(rf_id, config_dict)
            self._shocks = FxShocks.from_dataframe(shocks_df, rf_id=rf_id)
            logger.debug(" FxShocks loaded: %s (%d scenarios)", rf_id, self._shocks.n_scenarios)
        except Exception as exc:
            raise MarketDataError(f"Failed to load FX shocks for {rf_id}: {exc}") from exc

    def validate(self) -> None:
        """Check that all FX market data is present and internally consistent."""
        if self.spot_series is None:
            raise ValidationError(f"{self.asset_name}: spot_series not loaded")
        if self.spot is None or np.isnan(self.spot):
            raise ValidationError(f"{self.asset_name}: spot is None or NaN")
        if self._shocks is None:
            raise ValidationError(f"{self.asset_name}: shocks not loaded")
        if not isinstance(self._shocks, FxShocks):
            raise ValidationError(f"{self.asset_name}: shocks must be FxShocks, got {type(self._shocks)}")
        logger.debug(" FxAsset validation passed: %s", self.asset_name)
