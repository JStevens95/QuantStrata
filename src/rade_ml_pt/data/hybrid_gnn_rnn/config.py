"""
Data pipeline configuration for HybridGnnRnn model.
"""
from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Literal
from dataclasses import dataclass, field, asdict

from src.rade_ml_pt.core.config import DataPipelineConfig

# define module level logging.
logger = logging.getLogger(__name__)


@dataclass
class FolderEnvironmentConfig:
    """Configuration for model folder environment."""

    root_folder: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FolderEnvironmentConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BasisSelectionConfig:
    """Configuration for basis selection dimensionality reduction model."""

    var_threshold: float = 0.9999
    weight_tail: float = 1.0
    method: Literal['svd', 'pca', 'svds'] = "pca"
    max_components: int = 200

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasisSelectionConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GridConstructionConfig:
    """Configuration for grid construction dimensionality reduction model."""

    corr_threshold: float = 0.75

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridConstructionConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DimensionalityConfig:
    """Configuration for dimensionality reduction model."""
    # general parameters.
    reduction_mode: str = 'basis_selection'

    # basis selection parameters.
    basis_selection: Optional[BasisSelectionConfig] = None

    # grid construction parameters.
    grid_construction: Optional[GridConstructionConfig] = None


    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionalityConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GraphBuilderConfig:
    """Configuration for trade graph builder."""
    include_quota: bool = False
    distance_metric: str = 'euclidean'
    k: int = 2
    alpha_moneyness: Optional[float] = 1.0
    alpha_maturity: Optional[float] = 1.0
    alpha_prod_type: Optional[float] = 1.0
    alpha_prod_subtype: Optional[float] = 1.0
    alpha_delta: Optional[float] = 1.0
    alpha_vega: Optional[float] = 1.0
    alpha_underlying: Optional[float] = 1.0
    alpha_underlying_rf: Optional[float] = 1.0
    p_min_elementary: Optional[int] = 1
    q_min_target: Optional[int] = 1

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphBuilderConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AttributeEncoderConfig:
    """Configuration for trade attribute encoder."""
    normalise_features: bool = True
    use_decay_terms: bool = True
    num_decay_terms: int = 5
    numeric_keys: List[str] = field(default_factory=lambda: ["moneyness", "yrs_to_maturity", "delta", "vega"])
    categorical_keys: List[str] = field(default_factory=lambda: ["product_type", "product_subtype", "trade_type"])
    multi_label_keys: List[str] = field(default_factory=lambda: ["underlying_risk_factors"])

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttributeEncoderConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class HybridGnnRnnDataConfig(DataPipelineConfig):
    """
    Data configuration for the Hybrid GNN-RNN model.

    Inherits universal pipeline settings (batch size, shuffle, cache, etc.) from DataPipelineConfig and add GNN-RNN
    specific settings.
    """
    # root folder environment.
    folders: FolderEnvironmentConfig = field(default_factory=FolderEnvironmentConfig)

    # dimensionality reduction parameters.
    dimensionality: DimensionalityConfig = field(default_factory=DimensionalityConfig)

    # graph builder parameters.
    graph_builder: GraphBuilderConfig = field(default_factory=GraphBuilderConfig)

    # attribute encoding parameters.
    attribute_encoder: AttributeEncoderConfig = field(default_factory=AttributeEncoderConfig)

    # analytics / plotting parameters.
    save_intermediate_files: bool = False
    plot_trade_graph: bool = False
    plot_pnl_distribution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "folders": self.folders.to_dict(),
            "dimensionality": self.dimensionality.to_dict() if self.dimensionality else None,
            "graph_builder": self.graph_builder.to_dict() if self.graph_builder else None,
            "attribute_encoder": self.attribute_encoder.to_dict() if self.attribute_encoder else None,
            "save_intermediate_files": self.save_intermediate_files,
            "plot_trade_graph": self.plot_trade_graph,
            "plot_pnl_distribution": self.plot_pnl_distribution
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HybridGnnRnnDataConfig":
        """Create from dictionary."""
        return cls(
            folders=FolderEnvironmentConfig.from_dict(d["folders"]) if d.get("folders") else None,
            dimensionality=DimensionalityConfig.from_dict(d["dimensionality"]) if d.get("dimensionality") else None,
            graph_builder=GraphBuilderConfig.from_dict(d["graph_builder"]) if d.get("graph_builder") else None,
            attribute_encoder=AttributeEncoderConfig.from_dict(d["attribute_encoder"]) if d.get("attribute_encoder") else None,
            save_intermediate_files=d.get("save_intermediate_files"),
            plot_trade_graph=d.get("plot_trade_graph"),
            plot_pnl_distribution=d.get("plot_pnl_distribution")
        )

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to json file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "HybridGnnRnnDataConfig":
        """Load configuration from json file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
