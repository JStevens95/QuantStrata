"""
Configuration loader — reads YAML/CSV configs and validates against dataclasses.

Provides a single entry point for loading the pipeline's configuration
from disk, validating required keys, and constructing PipelineConfig.

TODO(wire): Adapt the YAML keys and validation rules to match your
actual configuration files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.rade_sr.core.exceptions import ConfigurationError
from src.rade_sr.replication.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates pipeline configuration from YAML files.

    Parameters
    ----------
    config_dir : str or Path
        Directory containing configuration files.

    Expected files:
        - ``config_model.yaml``: Pipeline, clustering, and trade generation config.
        - ``config_api.yaml``: API endpoints and authentication.
        - ``mapping.csv``: Asset class / factor ID mappings (optional).
    """

    def __init__(self, config_dir: Union[str, Path]) -> None:
        self._dir = Path(config_dir)

    def load_pipeline_config(
        self,
        model_config_file: str = "config_model.yaml",
        api_config_file: str = "config_api.yaml",
    ) -> PipelineConfig:
        """Load and validate the full pipeline configuration.

        Returns
        -------
        PipelineConfig
            Validated configuration ready for PreprocessingPipeline.run().

        Raises
        ------
        ConfigurationError
            If required keys are missing or values are invalid.

        TODO(wire): Adjust the YAML parsing below to match your actual
        config file structure.
        """
        model_cfg = self._load_yaml(self._dir / model_config_file)
        api_cfg = self._load_yaml(self._dir / api_config_file)

        self._validate_model_config(model_cfg)

        asset_config = self._build_asset_config(model_cfg, api_cfg)

        return PipelineConfig(
            cluster_config=model_cfg.get("clustering", {}),
            path_config=model_cfg.get("paths", {}),
            asset_config=asset_config,
            trade_config=model_cfg.get("trade_generation", {}),
            fail_fast=model_cfg.get("fail_fast", True),
        )

    def load_factor_mapping(
        self,
        mapping_file: str = "mapping.csv",
    ) -> Dict[str, Dict[str, str]]:
        """Load factor ID to asset class mapping from CSV.

        Returns
        -------
        dict
            ``{factor_id: {asset_class, pair/currency, ...}}``.

        TODO(wire): Adapt CSV columns to your mapping file format.
        """
        import pandas as pd
        path = self._dir / mapping_file
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        return {
            row["factor_id"]: row.to_dict()
            for _, row in df.iterrows()
        }

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        import yaml
        if not path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config file must be a dict: {path}")
        return data

    @staticmethod
    def _validate_model_config(cfg: Dict[str, Any]) -> None:
        """Validate required top-level keys in model config.

        TODO(wire): Add validation rules specific to your config schema.
        """
        required = ["clustering", "paths"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ConfigurationError(
                f"Missing required config keys: {missing}"
            )

    @staticmethod
    def _build_asset_config(
        model_cfg: Dict[str, Any],
        api_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge model and API config into per-factor asset config.

        TODO(wire): Adapt to your config structure. This should produce
        the asset_config dict expected by PipelineConfig, where each
        factor_id maps to its asset_class and loading parameters.
        """
        factors = model_cfg.get("risk_factors", {})
        api_base = api_cfg.get("base_url", "")

        asset_config: Dict[str, Any] = {}
        for factor_id, factor_cfg in factors.items():
            merged = dict(factor_cfg)
            merged["api_base_url"] = api_base
            asset_config[factor_id] = merged

        return asset_config
