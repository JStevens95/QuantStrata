"""
Configuration loading — YAML/dict -> validated :class:`OrchestratorConfig`.

Relative ``mapping_file`` entries resolve against the YAML file's directory so a
config is portable. The resolution block may be named ``factor_config`` (work-env
convention) or ``factors``; each sub-key (e.g. ``fx_config``) yields one asset class
with a trailing ``_config`` stripped (``fx_config`` -> ``fx``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.rade_static_replication.config.schema import (
    AssetFactorRule,
    ClusteringConfig,
    ElementaryConfig,
    EngineConfig,
    FactorResolutionConfig,
    OrchestratorConfig,
    OutputConfig,
)
from src.rade_static_replication.domain.errors import ConfigurationError


def _asset_class_from_key(block_key: str) -> str:
    """Derive the asset class from a resolution-block key (``fx_config`` -> ``fx``)."""
    key = str(block_key).strip().lower()
    return key[: -len("_config")] if key.endswith("_config") else key


def load_config(path: str | Path) -> OrchestratorConfig:
    """Load and validate an :class:`OrchestratorConfig` from a YAML file."""
    import yaml

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ConfigurationError(f"config file not found: {path}") from exc
    return config_from_dict(raw, base_dir=path.parent)


def config_from_dict(raw: Dict[str, Any], base_dir: Optional[Path] = None) -> OrchestratorConfig:
    """Build a config from an already-parsed dict."""
    if "cob_date" not in raw:
        raise ConfigurationError("config missing required 'cob_date'")

    # Accept the work-env key name (factor_config) or the shorter alias (factors).
    factor_block = raw.get("factor_config") or raw.get("factors") or {}

    rules: Dict[str, AssetFactorRule] = {}
    for block_key, block in factor_block.items():
        asset_class = _asset_class_from_key(block_key)

        # Resolve a relative mapping file against the config's own directory.
        mapping_file = block.get("mapping_file")
        if mapping_file and base_dir is not None and not Path(mapping_file).is_absolute():
            mapping_file = str((base_dir / mapping_file).resolve())

        rules[asset_class] = AssetFactorRule(
            asset_class=asset_class,
            source=block["source"],
            factor_cols=list(block.get("factor_cols", [])),
            dependency_cols=list(block.get("dependency_cols", [])),
            attrs_key=block.get("attrs_key", ""),
            mapping_key=block.get("mapping_key", ""),
            mapping_file=mapping_file,
            asset_class_of=dict(block.get("asset_class_of", {})),
        )

    out_block = raw.get("output", {}) or {}
    eng = raw.get("engine", {}) or {}
    clus = raw.get("clustering", {}) or {}
    return OrchestratorConfig(
        cob_date=str(raw["cob_date"]),
        factors=FactorResolutionConfig(rules=rules),
        elementary=ElementaryConfig(grids=dict(raw.get("elementary", {}) or {})),
        engine=EngineConfig(
            max_workers=int(eng.get("max_workers", 1)),
            use_numba=bool(eng.get("use_numba", True)),
        ),
        clustering=ClusteringConfig(keys=list(clus.get("keys", ["AssetClass"]))),
        output=OutputConfig(
            root=Path(out_block.get("root", "data/rade_static_replication")),
            run_id=out_block.get("run_id", "run"),
        ),
    )
