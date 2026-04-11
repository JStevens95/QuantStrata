"""
Configuration scheme and defaults for HybridGnnRnn model.

Dataclass-based config with to_dict/from_dict for serialization and
construction from dict/YAML. Aligns with rade_ml core.config patterns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Union

from pathlib import Path


# -----------------------------------------------------------------------------
# Layer sub-configs: general + parameters per layer type
# -----------------------------------------------------------------------------


@dataclass
class GnnGeneralConfig:
    architecture: str = "default"
    layers: int = 2
    layer_type: str = "mixed_graph_sage"
    dropout_rate: float = 0.1
    use_bias: bool = True
    use_residual: bool = True
    batch_norm: bool = True
    aggregator_op: str = "mean"
    # Graphormer-specific (ignored by GraphSAGE / MixedGraphSAGE).
    num_heads: int = 4
    k_nbrs: int = 50
    max_degree: int = 512

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GnnGeneralConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GnnParametersConfig:
    units: int = 128
    activation: str = "relu"
    kernel_initializer: str = "glorot_uniform"
    bias_initializer: str = "zeros"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GnnParametersConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GnnLayerConfig:
    """GNN block config: general (structure) + parameters (weights)."""
    general: GnnGeneralConfig = field(default_factory=GnnGeneralConfig)
    parameters: GnnParametersConfig = field(default_factory=GnnParametersConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"general": self.general.to_dict(), "parameters": self.parameters.to_dict()}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GnnLayerConfig":
        if not d:
            return cls()
        return cls(
            general=GnnGeneralConfig.from_dict(d.get("general")),
            parameters=GnnParametersConfig.from_dict(d.get("parameters")),
        )


# -----------------------------------------------------------------------------
# RNN Layer
# -----------------------------------------------------------------------------


@dataclass
class RnnGeneralConfig:
    architecture: str = "default"
    layers: int = 2
    layer_type: str = "lstm"
    dropout_rate: float = 0.1
    use_bias: bool = True
    # TCN-specific (ignored by LSTM/GRU/Dense).
    kernel_size: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RnnGeneralConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RnnParametersConfig:
    units: int = 128
    activation: str = "relu"
    recurrent_activation: str = "sigmoid"
    kernel_initializer: str = "glorot_uniform"
    recurrent_initializer: str = "orthogonal"
    bias_initializer: str = "zeros"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RnnParametersConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RnnLayerConfig:
    general: RnnGeneralConfig = field(default_factory=RnnGeneralConfig)
    parameters: RnnParametersConfig = field(default_factory=RnnParametersConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"general": self.general.to_dict(), "parameters": self.parameters.to_dict()}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RnnLayerConfig":
        if not d:
            return cls()
        return cls(
            general=RnnGeneralConfig.from_dict(d.get("general")),
            parameters=RnnParametersConfig.from_dict(d.get("parameters")),
        )


# -----------------------------------------------------------------------------
# Fusion Layer
# -----------------------------------------------------------------------------


@dataclass
class FusionGeneralConfig:
    fusion_mode: str = "gate"
    dropout_rate: float = 0.1
    num_heads: int = 1
    k_nbrs: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FusionGeneralConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FusionParametersConfig:
    units: int = 64
    activation: str = "sigmoid"
    kernel_initializer: str = "he_uniform"
    bias_initializer: str = "zeros"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FusionParametersConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FusionLayerConfig:
    general: FusionGeneralConfig = field(default_factory=FusionGeneralConfig)
    parameters: FusionParametersConfig = field(default_factory=FusionParametersConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"general": self.general.to_dict(), "parameters": self.parameters.to_dict()}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "FusionLayerConfig":
        if not d:
            return cls()
        return cls(
            general=FusionGeneralConfig.from_dict(d.get("general")),
            parameters=FusionParametersConfig.from_dict(d.get("parameters")),
        )


# -----------------------------------------------------------------------------
# Attention Layer
# -----------------------------------------------------------------------------


@dataclass
class AttentionGeneralConfig:
    layer_type: str = "standard"
    use_residual: bool = True
    use_layer_norm: bool = True
    attention_mode: bool = True
    num_heads: int = 1
    dropout_rate: float = 0.1
    k_nbrs: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AttentionGeneralConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AttentionParametersConfig:
    units: int = 32
    activation: str = "tanh"
    kernel_initializer: str = "he_uniform"
    bias_initializer: str = "zeros"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AttentionParametersConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AttentionLayerConfig:
    general: AttentionGeneralConfig = field(default_factory=AttentionGeneralConfig)
    parameters: AttentionParametersConfig = field(default_factory=AttentionParametersConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"general": self.general.to_dict(), "parameters": self.parameters.to_dict()}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AttentionLayerConfig":
        if not d:
            return cls()
        return cls(
            general=AttentionGeneralConfig.from_dict(d.get("general")),
            parameters=AttentionParametersConfig.from_dict(d.get("parameters")),
        )


# -----------------------------------------------------------------------------
# Projection Layer
# -----------------------------------------------------------------------------


@dataclass
class ProjectionGeneralConfig:
    dropout_rate: float = 0.1
    baseline_new_mode: str = "output_mix"
    use_baseline_norm: bool = True
    use_attn_scale_new: bool = False
    use_attn_bias_new: bool = False
    knn_k: int = 5
    knn_mode: str = "cosine_softmax"
    knn_temperature: float = 5.0
    knn_power: float = 2.0
    residual_new_damp: float = 1.0
    baseline_trade_count: Optional[int] = None
    attn_dim: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ProjectionGeneralConfig":
        if not d:
            return cls()
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for int_field in ("baseline_trade_count", "attn_dim"):
            if int_field in filtered and filtered[int_field] is not None:
                filtered[int_field] = int(filtered[int_field])
        return cls(**filtered)


@dataclass
class ProjectionParametersConfig:
    units: int = 32
    activation: str = "gelu"
    kernel_initializer: str = "glorot_uniform"
    bias_initializer: str = "zeros"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ProjectionParametersConfig":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProjectionLayerConfig:
    general: ProjectionGeneralConfig = field(default_factory=ProjectionGeneralConfig)
    parameters: ProjectionParametersConfig = field(default_factory=ProjectionParametersConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {"general": self.general.to_dict(), "parameters": self.parameters.to_dict()}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ProjectionLayerConfig":
        if not d:
            return cls()
        return cls(
            general=ProjectionGeneralConfig.from_dict(d.get("general")),
            parameters=ProjectionParametersConfig.from_dict(d.get("parameters")),
        )


# -----------------------------------------------------------------------------
# Top-level model config
# -----------------------------------------------------------------------------


@dataclass
class HybridGnnRnnModelConfig:
    """
    Full model configuration for HybridGnnRnn.

    Use from_dict() / from_yaml() to build from partial overrides; missing keys
    get defaults. to_dict() produces the exact nested dict expected by the model.
    """
    general: Dict[str, Any] = field(default_factory=dict)
    gnn_layer: GnnLayerConfig = field(default_factory=GnnLayerConfig)
    rnn_layer: RnnLayerConfig = field(default_factory=RnnLayerConfig)
    fusion_layer: FusionLayerConfig = field(default_factory=FusionLayerConfig)
    attention_layer: AttentionLayerConfig = field(default_factory=AttentionLayerConfig)
    projection_layer: ProjectionLayerConfig = field(default_factory=ProjectionLayerConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Produce the dict structure expected by HybridGnnRnn and its layers."""
        return {
            "general": self.general,
            "gnn_layer": self.gnn_layer.to_dict(),
            "rnn_layer": self.rnn_layer.to_dict(),
            "fusion_layer": self.fusion_layer.to_dict(),
            "attention_layer": self.attention_layer.to_dict(),
            "projection_layer": self.projection_layer.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "HybridGnnRnnModelConfig":
        """Build from dict with partial overrides; missing keys use defaults."""
        if not d:
            return cls()
        return cls(
            general=d.get("general", {}),
            gnn_layer=GnnLayerConfig.from_dict(d.get("gnn_layer")),
            rnn_layer=RnnLayerConfig.from_dict(d.get("rnn_layer")),
            fusion_layer=FusionLayerConfig.from_dict(d.get("fusion_layer")),
            attention_layer=AttentionLayerConfig.from_dict(d.get("attention_layer")),
            projection_layer=ProjectionLayerConfig.from_dict(d.get("projection_layer")),
        )

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "HybridGnnRnnModelConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "HybridGnnRnnModelConfig":
        """Load configuration from YAML file (requires PyYAML)."""
        try:
            import yaml
        except ImportError:
            raise ImportError("from_yaml requires PyYAML. Install with: pip install pyyaml")
        from src.rade_ml_pt.core.config import sanitize_yaml_values
        with open(path, "r") as f:
            data = sanitize_yaml_values(yaml.safe_load(f))
        return cls.from_dict(data)


# -----------------------------------------------------------------------------
# Backward-compatible default
# -----------------------------------------------------------------------------


def default_model_config() -> Dict[str, Any]:
    """
    Returns a minimal valid model configuration for HybridGnnRnn.

    For new code, prefer HybridGnnRnnModelConfig() or from_dict/from_yaml.
    This function remains for backward compatibility.
    """
    return HybridGnnRnnModelConfig().to_dict()
