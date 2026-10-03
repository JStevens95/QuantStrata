"""
Risk-factor resolution schema — mirrors work-env ``factor_config`` blocks.

Parsing turns YAML sub-keys (``fx_config``, ``ir_config``) into registry-aligned
asset classes (``FX``, ``IR``) and validated :class:`AssetFactorRule` instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.static_replication.core.exceptions import ConfigurationError

# Registry keys used by FxRiskFactorBuilder / IrRiskFactorBuilder.
_REGISTRY_ALIASES: Dict[str, str] = {
    "fx": "FX",
    "ir": "IR",
    "rates": "IR",
    "eq": "EQ",
    "cr": "CR",
}


def asset_class_from_block_key(block_key: str) -> str:
    """``fx_config`` -> ``FX``, ``ir_config`` -> ``IR``."""
    key = str(block_key).strip().lower()
    if key.endswith("_config"):
        key = key[: -len("_config")]
    return normalize_asset_class(key)


def normalize_asset_class(value: str) -> str:
    """Map portfolio / config labels to registry keys (``fx`` -> ``FX``)."""
    text = str(value).strip()
    if not text:
        return ""
    return _REGISTRY_ALIASES.get(text.lower(), text.upper())


@dataclass(frozen=True)
class AssetFactorRule:
    """How to resolve risk factors for one asset class."""

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
        return self.source in ("file", "mapping")

    @property
    def mapping_key_col(self) -> str:
        return self.mapping_key or self.attrs_key

    def __post_init__(self) -> None:
        if self.source not in ("file", "mapping", "attribute"):
            raise ConfigurationError(
                f"[{self.asset_class}] source must be 'file', 'mapping' or 'attribute', "
                f"got {self.source!r}"
            )
        if self.is_file_source:
            if not self.mapping_file:
                raise ConfigurationError(
                    f"[{self.asset_class}] file source needs 'mapping_file'"
                )
            if not self.attrs_key:
                raise ConfigurationError(
                    f"[{self.asset_class}] file source needs 'attrs_key'"
                )
        if not self.factor_cols:
            raise ConfigurationError(
                f"[{self.asset_class}] needs at least one factor_col"
            )


def parse_factor_rules(
    factor_config: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, AssetFactorRule]:
    """
    Build ``asset_class -> AssetFactorRule`` from orchestrator ``factor_config``.

    Accepts the work-env shape (``fx_config`` / ``ir_config`` sub-dicts).
    """
    if not factor_config:
        raise ConfigurationError("factor_config is empty")

    rules: Dict[str, AssetFactorRule] = {}
    for block_key, block in factor_config.items():
        if not isinstance(block, dict):
            raise ConfigurationError(
                f"factor_config[{block_key!r}] must be a mapping, got {type(block).__name__}"
            )
        asset_class = asset_class_from_block_key(block_key)

        mapping_file = block.get("mapping_file")
        if mapping_file and base_dir is not None and not Path(mapping_file).is_absolute():
            mapping_file = str((base_dir / mapping_file).resolve())

        asset_class_of = {
            col: _REGISTRY_ALIASES.get(str(ac).lower(), str(ac).upper())
            for col, ac in dict(block.get("asset_class_of", {})).items()
        }

        rules[asset_class] = AssetFactorRule(
            asset_class=asset_class,
            source=str(block["source"]),
            factor_cols=list(block.get("factor_cols", [])),
            dependency_cols=list(block.get("dependency_cols", [])),
            attrs_key=str(block.get("attrs_key", "")),
            mapping_key=str(block.get("mapping_key", "")),
            mapping_file=mapping_file,
            asset_class_of=asset_class_of,
        )
    return rules
