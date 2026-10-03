"""
RiskFactorBuilder implementations for FX | IR asset classes.

Each builder satisfies the RiskFactorBuilder Protocol (core/protocol.py):
    - class-level ``asset_class`` string attribute (registry key)
    - ``build(factor_id, factor_config) -> Asset``

Usage (register at pipeline startup):
    registry = RiskFactorRegistry()
    registry.register(FxRiskFactorBuilder(start_client))
    registry.register(IrRiskFactorBuilder(start_client))
"""
from __future__ import annotations

import logging
import pandas as pd

from typing import Any, Dict, Optional

from src.static_replication.assets.types import AssetConfig
from src.static_replication.assets.fx import FxAsset
from src.static_replication.assets.rates import IrAsset
from src.static_replication.core.exceptions import ConfigurationError

# define module level logging.
logger = logging.getLogger(__name__)


def parse_dates(value: Any) -> Optional[pd.Timestamp]:
    """Coerce a date-like value to pd.Timestamp or return None."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def _make_asset_config(factor_id: str, factor_config: Dict[str, Any]) -> AssetConfig:
    """Build an AssetConfig from the factor_config dict supplied by the orchestrator."""
    known_keys = {"asset_class", "asset_name", "cob_date", "start_date", "end_date", "calc_type", "extra"}
    extra = {k: v for k, v in factor_config.items() if k not in known_keys}
    extra.update(factor_config.get("extra", {}))

    return AssetConfig(
        asset_class=factor_config.get("asset_class", ""),
        asset_name=factor_config.get("asset_name", ""),
        asset_id=factor_id,
        cob_date=parse_dates(factor_config.get("cob_date")),
        start_date=parse_dates(factor_config.get("start_date")),
        end_date=parse_dates(factor_config.get("end_date")),
        calc_type=factor_config.get("calc_type", "MAXSVAR"),
        extra=extra,
    )


class FxRiskFactorBuilder:
    """
    Builds a fully loaded FxAsset for one FX risk factor.
    Registered under asset_class = "FX" in the RiskFactorRegistry.
    """

    # define asset class
    asset_class: str = "FX"

    def __init__(self, api_client: Any) -> None:
        """Initialize the FxRiskFactorBuilder class."""

        # initiate required variables.
        self._api_client = api_client

    def build(self, factor_id: str, factor_config: Dict[str, Any]) -> FxAsset:
        """Build a FxAsset from the factor_config dict."""
        if factor_config.get("asset_class") != "FX":
            raise ConfigurationError(
                f"FxRiskFactorBuilder: unexpected asset_class {factor_config.get('asset_class')!r} for factor "
                f"{factor_id!r}"
            )
        config = _make_asset_config(factor_id, {**factor_config})
        asset = FxAsset(config=config)
        logger.info("FxRiskFactorBuidler: loading %s", factor_id)
        asset.load(self._api_client)
        return asset


class IrRiskFactorBuilder:
    """
    Builds a fully loaded IrAsset for one IR risk factor.
    Registered under asset class = "IR" in the RiskFactorRegistry.
    """

    # define asset class
    asset_class: str = "IR"

    def __init__(self, api_client: Any) -> None:
        """Initialize the IrRiskFactorBuilder class."""

        # initiate required variables.
        self._api_client = api_client

    def build(self, factor_id: str, factor_config: Dict[str, Any]) -> IrAsset:
        """Build an IrAsset from the factor_config dict."""
        if factor_config.get("asset_class") != "IR":
            raise ConfigurationError(
                f"IrRiskFactorBuilder: unexpected asset_class {factor_config.get('asset_class')!r} for factor "
                f"{factor_id!r}"
            )
        config = _make_asset_config(factor_id, {**factor_config})
        asset = IrAsset(config=config)
        logger.info("IrRiskFactorBuidler: loading %s", factor_id)
        asset.load(self._api_client)
        return asset
