"""
Configuration schema — pure, validated dataclasses.

The structure mirrors the pipeline stages so a reader maps config block -> stage at
a glance. Parsing lives in :mod:`config.loader`; this module is data only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rade_static_replication.domain.errors import ConfigurationError


@dataclass(frozen=True)
class AssetFactorRule:
    """How to resolve risk factors for one asset class.

    Field names mirror the work-env ``factor_config`` block:

    * ``source`` — ``"file"`` / ``"mapping"`` (look a key up in a CSV) or
      ``"attribute"`` (factor ids already on the trade row).
    * ``attrs_key`` — the **portfolio** column holding the lookup key (file source).
    * ``mapping_key`` — the **key column in the mapping CSV** (file source). Decoupled
      from ``attrs_key`` so the portfolio and the CSV may name the key differently
      (e.g. ``CurrencyCode`` vs ``currency``); blank falls back to ``attrs_key``.
    * ``factor_cols`` — columns whose values are the **primary** risk factors.
    * ``dependency_cols`` — columns whose values are **dependency** factors (built but
      not priced), e.g. the IR curves an FX factor needs.
    * ``asset_class_of`` — optional per-column override of the produced factor's asset
      class. When a column is absent here, primaries take this rule's ``asset_class`` and
      dependencies are inferred from the factor-id prefix (see ``rules.infer_asset_class``).
    """
    asset_class: str
    source: str
    factor_cols: List[str]
    dependency_cols: List[str] = field(default_factory=list)
    attrs_key: str = ""
    mapping_key: str = ""
    mapping_file: Optional[str] = None
    asset_class_of: Dict[str, str] = field(default_factory=dict)

    @property
    def is_file_source(self) -> bool:
        """True when factors come from a mapping CSV rather than the trade row."""
        return self.source in ("file", "mapping")

    @property
    def mapping_key_col(self) -> str:
        """Key column to index the CSV by (defaults to the portfolio key column)."""
        return self.mapping_key or self.attrs_key

    def __post_init__(self) -> None:
        if self.source not in ("file", "mapping", "attribute"):
            raise ConfigurationError(
                f"[{self.asset_class}] source must be 'file', 'mapping' or 'attribute', got {self.source!r}"
            )
        if self.is_file_source:
            if not self.mapping_file:
                raise ConfigurationError(f"[{self.asset_class}] file source needs 'mapping_file'")
            if not self.attrs_key:
                raise ConfigurationError(f"[{self.asset_class}] file source needs 'attrs_key'")
        if not self.factor_cols:
            raise ConfigurationError(f"[{self.asset_class}] needs at least one factor_col")


@dataclass(frozen=True)
class FactorResolutionConfig:
    rules: Dict[str, AssetFactorRule] = field(default_factory=dict)


@dataclass(frozen=True)
class ElementaryConfig:
    """Per-asset-class elementary-grid parameters (opaque dict per class)."""
    grids: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineConfig:
    max_workers: int = 1
    use_numba: bool = True


@dataclass(frozen=True)
class ClusteringConfig:
    """Attribute columns whose value-tuples define clusters (AssetClass implied first)."""
    keys: List[str] = field(default_factory=lambda: ["AssetClass"])


@dataclass(frozen=True)
class OutputConfig:
    """Artifact-store location."""
    root: Path = Path("data/rade_static_replication")
    run_id: str = "run"


@dataclass(frozen=True)
class OrchestratorConfig:
    """Top-level run recipe (one per invocation)."""
    cob_date: str
    factors: FactorResolutionConfig = field(default_factory=FactorResolutionConfig)
    elementary: ElementaryConfig = field(default_factory=ElementaryConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
