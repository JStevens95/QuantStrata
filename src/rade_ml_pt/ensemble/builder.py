"""
Ensemble builder: load member models from the registry and assemble an
``EnsembleModel`` with validated trade coverage.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch.nn as nn

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.model import EnsembleModel
from src.rade_ml_pt.ensemble.router import TradeRouter

if TYPE_CHECKING:
    from src.rade_ml_pt.registry.store import ModelRegistry

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """
    Construct an ``EnsembleModel`` from an ``EnsembleConfig``.

    Loads each member from the model registry, validates that every target
    trade is assigned to exactly one member (for disjoint mode), and wires
    up the ``TradeRouter`` and aggregation strategy.
    """

    def __init__(self, registry: "ModelRegistry") -> None:
        self.registry = registry

    def build(
        self,
        config: EnsembleConfig,
        member_versions: Optional[Dict[str, str]] = None,
    ) -> EnsembleModel:
        """
        Build the ensemble.

        Parameters
        ----------
        config : EnsembleConfig
        member_versions : dict or None
            ``{cluster_id: registry_version_string}``.
            If ``None``, the latest version for each cluster is loaded
            (assumed to be tagged ``"{cluster_id}_latest"``).

        Returns
        -------
        EnsembleModel
        """
        member_versions = member_versions or {}

        self._validate_coverage(config)

        members = self._load_members(config, member_versions)
        router = TradeRouter(
            config.cluster_mapping,
            cluster_keys=config.get_cluster_keys_for_router(),
        )

        cluster_trade_indices = self._build_cluster_trade_indices(config)
        n_total = len(config.all_trade_ids)

        ensemble = EnsembleModel(
            members=members,
            router=router,
            aggregation=config.aggregation,
            weights=config.weights,
            cluster_trade_indices=cluster_trade_indices,
            n_total_targets=n_total,
        )

        logger.info(
            "Ensemble built: %d members, %d total targets, aggregation='%s'",
            config.n_members, n_total, config.aggregation,
        )
        return ensemble

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_coverage(self, config: EnsembleConfig) -> None:
        """
        Ensure every trade appears in exactly one cluster (for concat mode)
        or at least one cluster (for weighted_mean mode).
        """
        seen: Dict[str, str] = {}
        duplicates: List[str] = []

        for cid in config.cluster_ids:
            for tid in config.cluster_mapping[cid]:
                if tid in seen:
                    duplicates.append(f"'{tid}' in both '{seen[tid]}' and '{cid}'")
                seen[tid] = cid

        if config.aggregation == "concat" and duplicates:
            raise ValueError(
                f"Concat aggregation requires disjoint clusters but found "
                f"overlapping trades: {duplicates[:5]}"
                + (f" ... and {len(duplicates) - 5} more" if len(duplicates) > 5 else "")
            )
        elif duplicates:
            logger.warning(
                "Overlapping trades found across clusters (%d); weighted "
                "aggregation will be used.", len(duplicates),
            )

        if not seen:
            raise ValueError("No trades found in cluster_mapping.")

        logger.info(
            "Coverage validated: %d trades across %d clusters.",
            len(seen), config.n_members,
        )

    # ------------------------------------------------------------------
    # Member loading
    # ------------------------------------------------------------------

    def _load_members(
        self,
        config: EnsembleConfig,
        member_versions: Dict[str, str],
    ) -> Dict[str, nn.Module]:
        """Load each member model from the registry."""
        members: Dict[str, nn.Module] = {}

        for cid in config.cluster_ids:
            version_or_tag = member_versions.get(cid, f"{cid}_latest")
            try:
                model, entry = self.registry.load(version_or_tag)
                members[cid] = model
                logger.info(
                    "Loaded member '%s' from version '%s' (val_loss=%.6f)",
                    cid, entry.version,
                    entry.metrics.get("best_val_loss", float("nan")),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load member model for cluster '{cid}' "
                    f"(version_or_tag='{version_or_tag}'): {exc}"
                ) from exc

        return members

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cluster_trade_indices(
        config: EnsembleConfig,
    ) -> Dict[str, List[int]]:
        """
        Map each cluster's trades to their column position in the combined
        output array.

        The global ordering is defined by iterating ``config.all_trade_ids``
        (sorted by cluster, then by order within cluster).
        """
        all_ids = config.all_trade_ids
        id_to_global = {tid: i for i, tid in enumerate(all_ids)}

        indices: Dict[str, List[int]] = {}
        for cid in config.cluster_ids:
            indices[cid] = [
                id_to_global[tid]
                for tid in config.cluster_mapping[cid]
                if tid in id_to_global
            ]
        return indices
