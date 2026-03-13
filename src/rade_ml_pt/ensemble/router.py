"""
Trade-to-cluster routing for ensemble inference.

``TradeRouter`` maps each trade ID to its responsible cluster.  For known
trades this is a simple dictionary lookup.  For unseen trades the router
falls back to a configurable strategy (nearest-cluster by trade attributes,
or a default cluster).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TradeRouter:
    """
    Directs trades to their cluster's model.

    Parameters
    ----------
    cluster_mapping : dict
        ``{cluster_id: [trade_id, ...]}`` as stored in ``EnsembleConfig``.
    default_cluster : str or None
        Fallback cluster for trades not found in any mapping.
        If ``None``, unknown trades raise ``KeyError``.
    """

    def __init__(
        self,
        cluster_mapping: Dict[str, List[str]],
        default_cluster: Optional[str] = None,
    ) -> None:
        self._cluster_mapping = cluster_mapping
        self._default_cluster = default_cluster

        # Build the reverse index: trade_id -> cluster_id.
        self._trade_to_cluster: Dict[str, str] = {}
        for cid, tids in cluster_mapping.items():
            for tid in tids:
                if tid in self._trade_to_cluster:
                    logger.warning(
                        "Trade '%s' appears in multiple clusters: '%s' and '%s'. "
                        "Last one wins.",
                        tid, self._trade_to_cluster[tid], cid,
                    )
                self._trade_to_cluster[tid] = cid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cluster_ids(self) -> List[str]:
        return sorted(self._cluster_mapping.keys())

    @property
    def n_trades(self) -> int:
        return len(self._trade_to_cluster)

    def get_cluster_for_trade(self, trade_id: str) -> str:
        """Return the cluster responsible for *trade_id*."""
        cid = self._trade_to_cluster.get(trade_id)
        if cid is not None:
            return cid
        if self._default_cluster is not None:
            return self._default_cluster
        raise KeyError(
            f"Trade '{trade_id}' is not assigned to any cluster and no "
            f"default_cluster is set."
        )

    def route(self, trade_ids: List[str]) -> Dict[str, List[str]]:
        """
        Partition *trade_ids* by cluster.

        Returns ``{cluster_id: [trade_ids_for_that_cluster]}``.
        """
        routed: Dict[str, List[str]] = {cid: [] for cid in self.cluster_ids}
        for tid in trade_ids:
            cid = self.get_cluster_for_trade(tid)
            routed.setdefault(cid, []).append(tid)
        return {cid: tids for cid, tids in routed.items() if tids}

    def get_trades_for_cluster(self, cluster_id: str) -> List[str]:
        """Return all trade IDs assigned to *cluster_id*."""
        return list(self._cluster_mapping.get(cluster_id, []))

    def assign_new_trade(
        self,
        trade_attribs: Dict[str, Any],
        cluster_centroids: Optional[Dict[str, np.ndarray]] = None,
    ) -> str:
        """
        Assign an unseen trade to the nearest cluster.

        If *cluster_centroids* is provided the trade is assigned to the
        cluster whose centroid is closest in Euclidean distance.  Otherwise
        the *default_cluster* is returned.

        Parameters
        ----------
        trade_attribs : dict
            Encoded feature vector (or raw attributes) for the new trade.
        cluster_centroids : dict or None
            ``{cluster_id: centroid_array}`` — pre-computed cluster centroids.

        Returns
        -------
        str
            The assigned cluster ID.
        """
        if cluster_centroids is not None:
            feat = np.asarray(trade_attribs.get("features", trade_attribs.get("encoded", [])))
            if feat.size > 0:
                best_cid, best_dist = None, float("inf")
                for cid, centroid in cluster_centroids.items():
                    dist = float(np.linalg.norm(feat - centroid))
                    if dist < best_dist:
                        best_cid, best_dist = cid, dist
                if best_cid is not None:
                    logger.info("New trade assigned to cluster '%s' (dist=%.4f)", best_cid, best_dist)
                    return best_cid

        if self._default_cluster is not None:
            return self._default_cluster

        raise ValueError(
            "Cannot assign new trade: no cluster_centroids supplied and "
            "no default_cluster configured."
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_trade_cluster_map(self) -> Dict[str, str]:
        """Return ``{trade_id: cluster_id}`` for all known trades."""
        return dict(self._trade_to_cluster)
