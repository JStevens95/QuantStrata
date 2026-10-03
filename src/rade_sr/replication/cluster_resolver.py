"""
Cluster resolver stage — resolves the target portfolio into clusters.

Takes the user's clustering configuration and returns a mapping of
cluster_id → risk_factor_names.  The clustering method determines how
target trades are grouped:

- Simple key grouping (e.g. by ccy_pair): 1 cluster = 1 risk factor
- K-means / algorithmic: 1 cluster = potentially many risk factors

TODO(wire): Replace the internal calls with your existing ClusterManager API.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ClusterResolver:
    """Wraps the existing cluster manager to satisfy the Protocol.

    Parameters
    ----------
    cluster_manager : Any
        Your existing ClusterManager instance.

        TODO(wire): Type-hint with your actual ClusterManager class.
    """

    def __init__(self, cluster_manager: Any) -> None:
        self._cm = cluster_manager

    def resolve(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Resolve clustering configuration into cluster → risk factors.

        The cluster_id is the cluster's own identifier (from the
        clustering method).  The risk factor names are the keys used
        to look up pre-computed data from PortfolioData.  These are
        two separate concepts — cluster_id is NOT a risk factor name
        (though they may coincide in the 1:1 key-grouping case).

        All risk factors in a cluster must share the same asset class.

        Parameters
        ----------
        config : dict
            Clustering configuration.  Typical keys:

            - ``cluster_key`` (str): Attribute used for partitioning
              (e.g. ``"ccy_pair"``, ``"product_type"``).
            - ``cluster_key_values`` (list[str]): Values to partition by.
            - ``method`` (str): Clustering method (``"key_grouping"``,
              ``"kmeans"``, etc.).

        Returns
        -------
        dict[str, list[str]]
            ``cluster_id → [risk_factor_names]``.

        Examples
        --------
        Key grouping by ccy_pair (cluster_id happens to match factor)::

            {"EURUSD": ["EURUSD"], "GBPUSD": ["GBPUSD"]}

        K-means (cluster_id is algorithm-assigned, factors are looked up)::

            {"cluster_0": ["EURUSD", "GBPUSD"], "cluster_1": ["USDJPY"]}

        TODO(wire): Replace the body with your actual clustering call.
        """
        raise NotImplementedError(
            "Wire to your ClusterManager or equivalent"
        )


class KeyGroupingClusterResolver:
    """Config-driven resolver that needs no external ClusterManager.

    Supports two methods so the module is runnable out of the box:

    - ``"one_to_one"`` (default): one cluster per risk factor
      (``{"EURUSD": ["EURUSD"], ...}``). Trivially asset-class homogeneous.
    - ``"explicit"``: use a caller-supplied ``cluster_map`` from config
      (``config["cluster_map"] = {"G10_FX": ["EURUSD", "GBPUSD"], ...}``).

    Parameters
    ----------
    risk_factors : list[str]
        The portfolio's risk factors (from Step 0 ingestion). Used by the
        ``one_to_one`` method.
    """

    def __init__(self, risk_factors: List[str]) -> None:
        self._risk_factors = list(risk_factors)

    def resolve(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        method = config.get("method", "one_to_one")
        if method == "explicit":
            cluster_map = config.get("cluster_map")
            if not cluster_map:
                raise ValueError("method='explicit' requires config['cluster_map']")
            return {cid: list(rfs) for cid, rfs in cluster_map.items()}
        if method == "one_to_one":
            return {rf: [rf] for rf in self._risk_factors}
        raise ValueError(f"Unknown clustering method {method!r}")
