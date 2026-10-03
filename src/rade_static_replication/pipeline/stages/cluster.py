"""
Stage 8 — resolve clusters.

Slices the portfolio into single-asset-class clusters keyed by the configured
attribute columns (``AssetClass`` is always the first key, guaranteeing no
cross-asset clusters). Each cluster carries its target trade ids and the union of
their primary risk factors.
"""
from __future__ import annotations

import logging
import re
from typing import List

from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    ClusterSet,
    ClusterSpec,
    Portfolio,
    RiskFactorUniverse,
)

logger = logging.getLogger(__name__)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-")


def resolve_clusters(
    portfolio: Portfolio, universe: RiskFactorUniverse, config: OrchestratorConfig,
) -> ClusterSet:
    """Group trades into single-asset-class clusters by the configured keys."""
    keys = config.clustering.keys
    if "AssetClass" not in keys:
        keys = ["AssetClass"] + list(keys)

    attr = portfolio.attributes
    missing = [k for k in keys if k not in attr.columns]
    if missing:
        logger.warning("clustering keys not in attributes, ignored: %s", missing)
        keys = [k for k in keys if k in attr.columns]

    clusters: List[ClusterSpec] = []
    for key_tuple, group in attr.groupby(keys, sort=True):
        values = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
        key_map = {k: str(v) for k, v in zip(keys, values)}
        trade_ids = [str(t) for t in group.index]

        factor_ids: List[str] = []
        for t in trade_ids:
            for fid in universe.factors_by_trade.get(t, []):
                if fid not in factor_ids:
                    factor_ids.append(fid)

        cluster_id = "__".join(_slug(v) for v in values)
        clusters.append(ClusterSpec(
            cluster_id=cluster_id, key=key_map, asset_class=key_map.get("AssetClass", ""),
            risk_factor_ids=factor_ids, target_trade_ids=trade_ids,
        ))

    logger.info("Resolved %d clusters", len(clusters))
    return ClusterSet(clusters=clusters)
