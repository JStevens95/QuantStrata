"""
Protocol definitions for all pipeline stage contracts.

Every cross-layer dependency points at these abstractions, not at
concrete classes.  The PreprocessingPipeline depends on Protocol types
only — it has zero imports from instruments/, pricing/, or
replication/ stage implementations.

Portfolio-first contracts:
  - Assets are loaded once per risk factor (not per cluster).
  - Trades are generated portfolio-wide.
  - PnL is computed portfolio-wide.
  - Clusters are resolved *after* all data is ready.  Each cluster
    maps to one or more risk factors; the pipeline populates it
    by slicing PortfolioData with those keys.

When adding a new stage, define its Protocol here FIRST, then implement.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from src.rade_sr.core.types import (
    ClusterPaths,
    ElementaryTrade,
    ElementaryTradePnL,
)


@runtime_checkable
class ClusterResolver(Protocol):
    """Resolves the target portfolio into clusters.

    The clustering method (simple key grouping, k-means, etc.)
    determines which target trades go into each cluster.  Each
    cluster's target trades imply a set of risk factors — those
    risk factors are the keys used to slice PortfolioData.
    """

    def resolve(
        self, config: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """Resolve clustering configuration into cluster → risk factors.

        Parameters
        ----------
        config : dict
            Clustering configuration.  Typical keys:
            ``cluster_key``, ``cluster_key_values``, ``method``, etc.

        Returns
        -------
        dict[str, list[str]]
            ``cluster_id → [risk_factor_names]``.  Each risk factor
            name is a key into ``PortfolioData.assets``,
            ``.elementary_trades``, and ``.trade_pnls``.

        Examples
        --------
        Simple 1:1 grouping by ccy_pair::

            {"EURUSD": ["EURUSD"], "GBPUSD": ["GBPUSD"]}

        K-means grouping with multiple factors per cluster::

            {"cluster_0": ["EURUSD", "GBPUSD"], "cluster_1": ["USDJPY"]}
        """
        ...


@runtime_checkable
class PathResolver(Protocol):
    """Derives artifact filesystem paths for a given cluster."""

    def resolve(self, cluster_id: str, path_config: Dict[str, Any]) -> ClusterPaths:
        """Return the four canonical artifact paths for one cluster.

        Parameters
        ----------
        cluster_id : str
            The cluster to resolve paths for.
        path_config : dict
            Path configuration (root directory, filename conventions).
        """
        ...


@runtime_checkable
class RiskFactorBuilder(Protocol):
    """Builds a fully loaded, self-contained Asset for one risk factor.

    The Asset loads its own dependencies internally (e.g. FXAsset
    loads domestic/foreign IR curves).

    Implementations must define ``asset_class`` as a class-level string
    attribute — this is the registry key.
    """

    asset_class: str

    def build(
        self,
        factor_id: str,
        factor_config: Dict[str, Any],
    ) -> Any:
        """Load market data + scenarios and return a self-contained Asset.

        Parameters
        ----------
        factor_id : str
            Unique risk factor name (e.g. ``"EURUSD"``).
        factor_config : dict
            Per-factor configuration: ``asset_class`` plus asset-specific
            keys (``pair``, ``currency``, etc.).

        Returns
        -------
        Any
            A loaded Asset instance (FXAsset, IRAsset, etc.).
        """
        ...


@runtime_checkable
class TradeGenerator(Protocol):
    """Generates elementary trades across the entire portfolio.

    Operates on the full set of loaded assets, not on a per-cluster
    basis.
    """

    def generate(
        self,
        assets: Dict[str, Any],
        trade_config: Dict[str, Any],
    ) -> Dict[str, List[ElementaryTrade]]:
        """Generate and deduplicate elementary trades for the portfolio.

        Parameters
        ----------
        assets : dict[str, Any]
            ``risk_factor → loaded Asset``.
        trade_config : dict
            Trade generation parameters.

        Returns
        -------
        dict[str, list[ElementaryTrade]]
            ``risk_factor → [trades]``.  Mirrors the assets dict.
        """
        ...


@runtime_checkable
class PnLEngine(Protocol):
    """Computes PnL for all elementary trades across the portfolio.

    Receives trades already grouped by risk factor, makes one
    vectorised pricer call per factor group.
    """

    def compute_batch(
        self,
        trades: Dict[str, List[ElementaryTrade]],
        risk_factors: Dict[str, Any],
    ) -> Dict[str, Dict[str, ElementaryTradePnL]]:
        """Compute PnL for all trades, grouped by risk factor.

        Parameters
        ----------
        trades : dict[str, list[ElementaryTrade]]
            ``risk_factor → [trades]``.
        risk_factors : dict[str, Any]
            Loaded Asset instances keyed by risk factor name.

        Returns
        -------
        dict[str, dict[str, ElementaryTradePnL]]
            ``risk_factor → {trade_id → PnL}``.
        """
        ...
