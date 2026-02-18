"""
Feature schemas: define what features exist and how they are computed.

A schema describes a feature set (e.g. pricing, GNN) with names, dtypes,
and optional transform identifiers. Used by both training and inference
to ensure identical feature computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureSchema:
    """
    Schema for a set of features used by a model.

    Attributes
    ----------
    name : str
        Identifier for this feature set (e.g. "pricing", "gnn").
    feature_names : List[str]
        Ordered names of features in the output array.
    dtypes : List[str], optional
        Expected dtypes per feature ("float32", "int32", etc.).
    transform_ids : List[str], optional
        Transform identifier per feature (e.g. "zscore", "none", "onehot").
    metadata : Dict[str, Any], optional
        Extra metadata (e.g. source columns, window sizes).
    """

    name: str
    feature_names: List[str]
    dtypes: Optional[List[str]] = None
    transform_ids: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        if self.dtypes is None:
            self.dtypes = ["float32"] * n
        elif len(self.dtypes) != n:
            raise ValueError(f"dtypes length {len(self.dtypes)} != feature_names {n}")
        if self.transform_ids is None:
            self.transform_ids = ["none"] * n
        elif len(self.transform_ids) != n:
            raise ValueError(f"transform_ids length != feature_names {n}")

    def n_features(self) -> int:
        """Return the number of features in this schema."""
        return len(self.feature_names)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema for storage or logging."""
        return {
            "name": self.name,
            "feature_names": self.feature_names,
            "dtypes": self.dtypes,
            "transform_ids": self.transform_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSchema":
        """Deserialize schema from dict."""
        return cls(
            name=d["name"],
            feature_names=d["feature_names"],
            dtypes=d.get("dtypes"),
            transform_ids=d.get("transform_ids"),
            metadata=d.get("metadata", {}),
        )


def PricingFeatureSchema(include_rate_foreign: bool = False) -> FeatureSchema:
    """
    Standard schema for option pricing features.

    Features: spot, strike, vol, rate, expiry, option_type [, rate_foreign].
    By default, vol and rate use zscore; others use none.

    Parameters
    ----------
    include_rate_foreign : bool, default False
        If True, add rate_foreign as the last feature.

    Returns
    -------
    FeatureSchema
    """
    names = ["spot", "strike", "vol", "rate", "expiry", "option_type"]
    # vol, rate typically benefit from zscore; spot/strike often kept raw or minmax
    transforms = ["none", "none", "zscore", "zscore", "none", "none"]
    if include_rate_foreign:
        names.append("rate_foreign")
        transforms.append("zscore")
    return FeatureSchema(
        name="pricing",
        feature_names=names,
        transform_ids=transforms,
        metadata={"include_rate_foreign": include_rate_foreign},
    )


def GnnFeatureSchema() -> FeatureSchema:
    """
    Schema for GNN-RNN hybrid model feature set.

    Includes numeric trade attributes (moneyness, delta, vega, time_to_maturity),
    optional categorical embeddings, and graph structure (adjacency, indices).
    Used for portfolio P&amp;L prediction.

    Returns
    -------
    FeatureSchema
    """
    return FeatureSchema(
        name="gnn",
        feature_names=[
            "moneyness",
            "time_to_maturity",
            "normalised_delta",
            "normalised_vega",
            "product_type_embedding",
            "product_subtype_embedding",
            "underlying_risk_factors_embedding",
        ],
        transform_ids=["zscore", "zscore", "zscore", "zscore", "onehot", "onehot", "multilabel"],
        metadata={"model_type": "hybrid_gnn_rnn"},
    )
