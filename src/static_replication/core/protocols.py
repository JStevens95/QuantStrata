"""
Protocol definitions for all pipeline stage contracts.

Every cross-layer dependency points at these abstractions, not at concrete classes. The pipeline depends on Protocol
types only - zero imports from instruments/, market_data/, or assets/

When adding a new stage, define its Protocol here first, then implement.
"""
from __future__ import annotations

import pandas as pd

from typing import Any, Dict, List, Protocol, runtime_checkable

from src.static_replication.core.types import (
    ClusterPaths, ElementaryTrade, ElementaryTradePnL
)


@runtime_checkable
class SageClient(Protocol):
    """
    Wraps SageConnector for portfolio attributes and PnL retrieval.

    Calls are sequential - no threading. SageConnector thread-safety is not documented.
    """

    def get_portfolio_attributes(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch portfolio trade attributes from SAGE.

        Returns Dataframe with columns defined by api_config.yaml
        :param config:
        :return:
        """
        ...

    def get_portfolio_pnl(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch portfolio trade pnl from SAGE.

        Returns Dataframe with dates as columns and trade key as columns.

        :param config:
        :return:
        """
        ...


@runtime_checkable
class StarClient(Protocol):
    """
    Wraps TimeseriesRepositorySTAR and ScenarioRepositorySTAR for market data and shocks.
    """

    def get_market_data(self, rf_id: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch historical time-series for one risk factor from STAR.

        Return raw STAR DataFrame.

        :param rf_id:
        :param config:
        :return:
        """
        ...

    def get_shocks(self, rf_id: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch shock scenario DataFrame for one risk factor from STAR.

        Returns DataFrame with shock columns.

        :param rf_id:
        :param config:
        :return:
        """
        ...


@runtime_checkable
class ClusterResolver(Protocol):
    """
    Resolve user configuration into a flat list of cluster IDs.

    Wraps whatever clustering strategy is in use (k-means, user-defined keys, hierachical, etc).
    """

    def resolve(self, config: Dict[str, Any]) -> List[str]:
        """Return ordered list of cluster ID strings from the given config."""
        ...


@runtime_checkable
class PathResolver(Protocol):
    """
    Derives artifact filesystem paths for a given cluster.

    Different path layouts (convention-based, database-backed, cloud storage) can be swapped by providing a
    different PathResolver implementation
    """

    def resolve(self, cluster_id: str, path_config: Dict[str, Any]) -> ClusterPaths:
        """Return the four canonical artifact paths for one cluster."""
        ...


@runtime_checkable
class RiskFactorBuilder(Protocol):
    """
    Builds a fully loaded Asset for one risk factor ID.

    Each asset class has its own implementation registered via RiskFactorRegistry. The pipeline dispatches to the
    correct builder using RiskFactorRegistry.get(asset_class).

    Implementation must define asset_class as a class-level string attribute - this is the registry key.
    """

    # define parameters.
    asset_class: str

    def build(self, factor_id: str, factor_config: Dict[str, Any]) -> Any:
        """Load market data + scenarios and return a self-contained typed Asset."""
        ...


@runtime_checkable
class TradeGenerator(Protocol):
    """
    Generate elementary trades by interrogating assembled risk factors.

    The generator examines each risk factor's market data (vol surfaces, rate curves, spot levels) to produce
    contextually appropriate trades with appropriate strikes, tenors and payoff types.
    """

    def generate(self, assets:Dict[str, Any], trade_config: Dict[str, Any]) -> Dict[str, List[ElementaryTrade]]:
        """Generate and deduplicate elementary trades for the portfolio."""
        ...


@runtime_checkable
class PnLEngine(Protocol):
    """
    Computes PnL for a batch of elementary trades across their risk factors.

    Operates at portfolio level to avoid redundant pricing of shared risk factors across clusters.
    Groups trade by factor_id and makes one vectorised pricer call per unique risk factor group.
    """

    def compute_batch(
            self, trades: Dict[str, List[ElementaryTrade]], risk_factors: Dict[str, Any]
    ) -> Dict[str, Dict[str, ElementaryTradePnL]]:
        """
        Compute PnL for all elementary trades across the portfolio.

        Receives trades grouped by risk factor, makes one vectorised pricer call per factor group.

        :param trades:
        :param risk_factors:
        :return:
        """
        ...
