"""
ClusterResolver implementations.

A ClusterResolver maps a portfolio_attributes DataFrame into a dictionary of:
    cluster_id -> List[factor_id]

This drives PortfolioData.slice() in the orchestrator so each ReplicationJob receives only the assets the trades
relevant to its cluster.

Implementation provided:
    AttributeClusterResolver - groups trades by one or more portfolio attribute columns.
"""
from __future__ import annotations

import logging
import pandas as pd

from typing import Dict, List, Optional, Union

from src.static_replication.core.exceptions import ConfigurationError

# define module level logging.
logger = logging.getLogger(__name__)

# Separator used when joining composite cluster key clues into a cluster ID.
_COMPOSITE_SEP = "|"


def _make_cluster_id(row: pd.Series, keys: List[str]) -> str:
    """Build a cluster ID string from one or more column value in a DataFrame row."""
    parts = [str(row[k]) for k in keys if k in row.index]
    return _COMPOSITE_SEP.join(parts) if parts else "ALL"


class AttributeClusterResolver:
    """
    Groups risk factors into clusters by matching portfolio attribute column values against loaded factor IDs

    Composite key example
    ---------------------
    cluster_key=["AssetClass", "Desk", "Currency", "Product"]
        -> cluster_ids: "FX|FX OPTIONS|GBP|FX EUROPEAN"

    Resolution logic
    ----------------
    1. Validate that all keys in cluster_key exist in portfolio_attributes.
    2. If a ``factor_id_col`` column exists, use it direcrly to assign factor IDs to clusters.
    3. Otherwise fall back to a name-matching heuristic: each unique cluster key value is compared.
    4. If no portfolio_attribute are supplied, return a single "ALL" cluster.
    """

    def resolve(
            self, portfolio_attributes: pd.DataFrame, available_factor_ids: List[str],
            cluster_key: Union[List[str], str], factor_id_col: Optional[str] = "RiskFactorId"
    ) -> Dict[str, List[str]]:
        """Resolve cluster_id -> List[factor_id] from portfolio attributes."""

        # define output result\
        result: Dict[str, List[str]] = {}

        if portfolio_attributes.empty:
            logger.warning("AttributeClusterResolver: portfolio attributes is empty - returning single ALL cluster.")
            return {"ALL": list(available_factor_ids)}

        #
        keys: List[str] = [cluster_key] if isinstance(cluster_key, str) else list(cluster_key)
        missing_keys = [k for k in keys if k not in portfolio_attributes.columns]
        if missing_keys:
            raise ConfigurationError(
                f"AttributeClusterResolver: cluster_key column(s) {missing_keys!r} not found in portfolio_attributes."
            )

        if factor_id_col and factor_id_col in portfolio_attributes.columns:
            df = portfolio_attributes.copy()
            df["_cluster_id"] = df.apply(lambda row: _make_cluster_id(row, keys), axis=1)

            for cluster_id, group in df.groupby(["_cluster_id"]):
                factor_ids = [fid for fid in group[factor_id_col].dropna().unique().tolist()]
                if factor_ids:
                    result[str(cluster_id)] = factor_ids
                    logger.debug("AttributeClusterResolver: cluster %r --> %d factors", cluster_id, len(factor_ids))
        else:
            logger.warning("AttributeClusterResolver: no factor ID column in portfolio_attributes.")
            df = portfolio_attributes.copy()
            df["_cluster_id"] = df.apply(lambda row: _make_cluster_id(row, keys), axis=1)

            for cluster_id, group in df.groupby("_cluster_id"):
                key_value = set()
                for k in keys:
                    key_value.update(str(v).upper() for v in group[k].dropna().unique())
                    matched = [fid for fid in available_factor_ids if any(v in fid.upper() for v in key_value)]
                    if matched:
                        result[str(cluster_id)] = matched
        if not result:
            logger.warning("AttributeClusterResolver: no clusters resolved - falling back to single ALL cluster")
            result = {"ALL": list(available_factor_ids)}
        else:
            logger.info(
                "AttributeClusterResolver: resolved %d clusters frp, %d available factors (key=%r)",
                len(result), len(available_factor_ids), keys
            )
        return result
