"""
Ensemble configuration dataclasses.

``EnsembleConfig`` aggregates per-cluster member configs, trade-to-cluster
mapping, aggregation strategy, and infrastructure paths.  It is the single
config object consumed by all ensemble pipelines and the EnsembleBuilder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.rade_ml_pt.pipelines.config import PipelineConfig


@dataclass
class EnsembleConfig:
    """
    Top-level configuration for an ensemble of model members.

    Attributes
    ----------
    member_configs : dict
        ``{cluster_id: PipelineConfig_dict}`` — per-cluster pipeline config.
    pipeline_class : dict
        ``{cluster_id: dotpath_str}`` — pipeline class for each cluster.
        Omit or set to ``None`` for a cluster to use the default
        (``HybridGnnRnnTrainPipeline``).
    cluster_mapping : dict
        ``{cluster_id: [trade_id, ...]}`` — assigns every target trade to
        exactly one cluster.
    cluster_keys : dict or None
        ``{cluster_id: {attr_name: value, ...}}`` — optional attribute-based
        routing for new trades. Can be set directly or derived from
        ``cluster_key`` + ``cluster_key_values`` via ``get_cluster_keys_for_router()``.
    cluster_key : list or None
        Shared list of attribute names that define routing, e.g. ``["ccy", "desk", "product"]``.
        Used with ``cluster_key_values`` to build the key dict per cluster.
    cluster_key_values : dict or None
        ``{cluster_id: [value_ccy, value_desk, value_product, ...]}`` — per-cluster values
        in the same order as ``cluster_key``. E.g. cluster_0 = ``["GBP", "FLOW_RATES", "EUROPEAN"]``
        means cluster_0 has ccy=GBP, desk=FLOW_RATES, product=EUROPEAN.
    aggregation : str
        Aggregation strategy: ``"concat"`` (disjoint clusters) or
        ``"weighted_mean"`` (overlapping clusters).
    weights : dict or None
        ``{cluster_id: float}`` — member weights for weighted-mean
        aggregation.  Ignored when *aggregation* is ``"concat"``.
    execution_strategy : str
        How to execute per-member pipelines.  ``"sequential"`` runs one at
        a time (default).  Future values: ``"process_pool"`` (multi-CPU),
        ``"gpu_parallel"`` (multi-GPU), ``"distributed"`` (Ray / cloud).
    max_workers : int or None
        Maximum number of parallel workers for ``process_pool`` and
        ``gpu_parallel``.  ``None`` means one worker per cluster.
    gpu_device_ids : list of int or None
        Explicit GPU device IDs for ``gpu_parallel`` strategy.  ``None``
        means use all available GPUs via ``torch.cuda.device_count()``.
        Ignored by other strategies.
    registry_dir : str or None
        Root directory for ensemble and member registries.
    artifacts_dir : str or None
        Root directory for ensemble artifacts (plots, metrics, predictions).
    metadata : dict
        Arbitrary key-value pairs forwarded into run records.
    """

    member_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_class: Dict[str, Optional[str]] = field(default_factory=dict)
    cluster_mapping: Dict[str, List[str]] = field(default_factory=dict)
    cluster_keys: Optional[Dict[str, Dict[str, Any]]] = None
    cluster_key: Optional[List[str]] = None
    cluster_key_values: Optional[Dict[str, List[Any]]] = None
    aggregation: str = "concat"
    weights: Optional[Dict[str, float]] = None
    execution_strategy: str = "sequential"
    max_workers: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    registry_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def cluster_ids(self) -> List[str]:
        """Ordered list of cluster identifiers."""
        return sorted(self.cluster_mapping.keys())

    @property
    def n_members(self) -> int:
        return len(self.cluster_mapping)

    @property
    def all_trade_ids(self) -> List[str]:
        """Flat list of every trade ID across all clusters."""
        ids: List[str] = []
        for cid in self.cluster_ids:
            ids.extend(self.cluster_mapping[cid])
        return ids

    def get_cluster_keys_for_router(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Return ``{cluster_id: {attr_name: value, ...}}`` for TradeRouter.

        If ``cluster_keys`` is set, return it. Otherwise if ``cluster_key`` and
        ``cluster_key_values`` are set, build the dict from the shared attribute
        names and per-cluster value lists (same order). E.g. cluster_key =
        ["ccy", "desk", "product"], cluster_0_keys = ["GBP", "FLOW_RATES", "EUROPEAN"]
        -> cluster_0 key = {"ccy": "GBP", "desk": "FLOW_RATES", "product": "EUROPEAN"}.
        """
        if self.cluster_keys is not None:
            return self.cluster_keys
        if self.cluster_key is not None and self.cluster_key_values is not None:
            return {
                cid: dict(zip(self.cluster_key, values))
                for cid, values in self.cluster_key_values.items()
            }
        return None

    def get_member_pipeline_config(self, cluster_id: str) -> PipelineConfig:
        """Build a ``PipelineConfig`` for one member from its dict representation."""
        raw = self.member_configs.get(cluster_id, {})
        return PipelineConfig(
            training_config=raw.get("training_config"),
            data_config=raw.get("data_config"),
            model_config=raw.get("model_config"),
            registry_dir=self.registry_dir,
            tracking_dir=raw.get("tracking_dir"),
            artifacts_dir=self.artifacts_dir,
            version_or_tag=raw.get("version_or_tag", "latest"),
            metadata={
                **raw.get("metadata", {}),
                "cluster_id": cluster_id,
                "trade_ids": self.cluster_mapping.get(cluster_id, []),
            },
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnsembleConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EnsembleConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EnsembleConfig":
        """Load from a YAML file (requires ``pyyaml``)."""
        import yaml
        from src.rade_ml_pt.core.config import sanitize_yaml_values
        with open(path, "r") as f:
            return cls.from_dict(sanitize_yaml_values(yaml.safe_load(f)))
