"""
Asset-class-specific RiskFactorBuilder implementations.

IMPORTANT: Importing this module is the registration act.  The pipeline
triggers this import in ``_load_all_assets()``:

    import src.rade_sr.replication.builders

Each decorated class is automatically registered with the RiskFactorRegistry.
Adding a new asset class = one new decorated class here.  The pipeline itself
does not change.

In the portfolio-first architecture these builders are called once per unique
factor across the whole portfolio (not per-cluster).  They create and return
Asset objects from the ``assets/`` package.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from src.rade_sr.replication.registry import RiskFactorRegistry

logger = logging.getLogger(__name__)


# Reserved key in factor_config used to inject the market-data client.
MARKET_DATA_CLIENT_KEY = "market_data_client"


def _resolve_client(factor_config: Dict[str, Any]) -> Any:
    """Return the injected MarketDataClient, or a default real client stub.

    The orchestrator injects the client under ``MARKET_DATA_CLIENT_KEY``. Tests
    or ad-hoc callers may instead pass ``api_base_url`` for the real client.
    """
    client = factor_config.get(MARKET_DATA_CLIENT_KEY)
    if client is not None:
        return client
    from src.rade_sr.api.client import InternalAPIClient
    return InternalAPIClient(base_url=factor_config.get("api_base_url", ""))


def _asset_extra(factor_config: Dict[str, Any]) -> Dict[str, Any]:
    """Strip injected/transport keys before they reach AssetConfig.extra."""
    drop = {MARKET_DATA_CLIENT_KEY, "api_base_url"}
    return {k: v for k, v in factor_config.items() if k not in drop}


@RiskFactorRegistry.register
class FXRiskFactorBuilder:
    """Builds an FX asset: spot + vol surface + forward curve + IR curves + scenarios.

    Creates an ``FXAsset`` instance, calls ``load()`` with the injected
    market-data client, and returns it. The client is any
    :class:`~rade_sr.sources.market_data.MarketDataClient` implementation.
    """
    asset_class = "fx"

    def build(
        self,
        factor_id: str,
        factor_config: Dict[str, Any],
    ) -> Any:
        """Load an FXAsset for one risk factor.

        Parameters
        ----------
        factor_id : str
            Currency pair identifier (e.g. ``"EURUSD"``).
        factor_config : dict
            Per-factor config: ``pair``, ``base_ccy``, ``quote_ccy``, plus the
            injected ``market_data_client`` (added by the orchestrator).
        """
        from src.rade_sr.assets.fx import FXAsset
        from src.rade_sr.assets.types import AssetConfig

        client = _resolve_client(factor_config)
        config = AssetConfig(
            asset_class="fx",
            asset_name=factor_id,
            extra=_asset_extra(factor_config),
        )
        asset = FXAsset(config)
        asset.load(client)

        logger.info("Built FX asset: %s", factor_id)
        return asset


@RiskFactorRegistry.register
class RatesRiskFactorBuilder:
    """Builds an IR asset: yield curves + swaption vol + scenarios.

    Creates an ``IRAsset`` instance, calls ``load()`` with the injected
    market-data client, and returns it.
    """
    asset_class = "rates"

    def build(
        self,
        factor_id: str,
        factor_config: Dict[str, Any],
    ) -> Any:
        """Load an IRAsset for one risk factor.

        Parameters
        ----------
        factor_id : str
            IR factor identifier (e.g. ``"EUR_6M"``, ``"USD_OIS"``).
        factor_config : dict
            Per-factor config: ``currency``, ``curve_type``, plus the injected
            ``market_data_client`` (added by the orchestrator).
        """
        from src.rade_sr.assets.rates import IRAsset
        from src.rade_sr.assets.types import AssetConfig

        client = _resolve_client(factor_config)
        config = AssetConfig(
            asset_class="rates",
            asset_name=factor_id,
            extra=_asset_extra(factor_config),
        )
        asset = IRAsset(config)
        asset.load(client)

        logger.info("Built IR asset: %s", factor_id)
        return asset


# ── Templates for new asset classes ──────────────────────────────────────

# @RiskFactorRegistry.register
# class EqRiskFactorBuilder:
#     """Builds an EQ asset: spot + dividends + vol surface + scenarios."""
#     asset_class = "eq"
#
#     def build(self, factor_id, factor_config) -> Any:
#         # TODO(wire): Implement for equity.
#         ...

# @RiskFactorRegistry.register
# class CreditRiskFactorBuilder:
#     """Builds a CR asset: credit spread curve + recovery + scenarios."""
#     asset_class = "cr"
#
#     def build(self, factor_id, factor_config) -> Any:
#         # TODO(wire): Implement for credit.
#         ...
