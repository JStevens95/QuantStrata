"""YAML configuration loader and writer for the rade_static_replication pipeline."""
from __future__ import annotations

import yaml
import logging

from pathlib import Path
from typing import Any, Dict, Union, List

from src.static_replication.core.exceptions import ConfigurationError

# define module level logging.
logger = logging.getLogger(__name__)

# Type alias - anything that names a YAML file on disk.
PathLike: Union[str, Path]

# Top-level sections that must be present in api_config.yaml
_API_REQUIRED_SECTIONS = ("sage_trade_attribute_query", "sage_trade_pnl_query")

# Top-level sections that must be present in pipeline_config.yaml
_PIPELINE_REQUIRED_SECTIONS = (
    "root_folder", "portfolio_config", "asset_config", "factor_config", "trade_config", "cluster_config",
    "execution_backend"
)

# Sub-keys required inside the attribute query block.
_ATTRIBUTE_REQUIRED_KEYS = ("metrics", "fields", "filter", "pivot")

# Sub-keys required inside the pnl query block.
_PNL_REQUIRED_KEYS = ("metrics", "fields", "filter")

# ------------------------------------------------------------------------------------
# Generic YAML I/O
# ------------------------------------------------------------------------------------

def load_yaml(file_path: PathLike) -> Dict[str, Any]:
    """
    Read a YAML file from disk and return it as a dictionary.

    No validation is preformed - the file is parsed and returned as is.

    :param file_path: path to YAML file.
    :return:
    """
    path = Path(file_path)
    if not path.exists():
        raise ConfigurationError(f"YAML file not found: {path}")

    try:
        # safe load avoids executing arbitrary tags.
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML {path}: {exc}") from exc
    return data if data is not None else {}


def save_yaml(data: Dict[str, Any], save_path: PathLike, file_name: str) -> Path:
    """
    Persist a dict t ``{save_path}/{file_name}.yaml`` to disk.

    :param data:
    :param save_path:
    :param file_name:
    :return:
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    full_path = save_dir / f"{file_name}.yaml"
    try:
        with full_path.open("w", encoding="utf-8") as stream:
            # default flow style=False keeps the output human readable.
            yaml.safe_dump(data, stream, default_flow_style=False, sort_keys=False)
    except OSError as exc:
        raise ConfigurationError(f"Failed to write YAML {full_path}: {exc}") from exc

    logger.info("Saved YAML config to %s", full_path)
    return full_path

# ------------------------------------------------------------------------------------
# API configuration loader.
# ------------------------------------------------------------------------------------

def load_api_config(config_file: Union[Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load and structurally validate the SAGE API config.

    The input may be either a path to ``api_config.yaml`` or an already loaded dict
    (useful for tests that build config inline).

    :param config_file:
    :return:
    """
    # Accepts either a path or a pre-loaded dict so callers can inject test data.
    if isinstance(config_file, dict):
        config = config_file
    elif isinstance(config_file, (str, Path)):
        config = load_yaml(config_file)
    else:
        # Anything else is a programmer error.
        raise ConfigurationError(f"load_api_config expects a path or dict, got {type(config_file).__name__}")
    _validate_api_config(config)
    return config


def _validate_api_config(config: Dict[str, Any]) -> None:
    """
    Structural validation for the API config.

    Raises ConfigurationError with a clear, actionable message if anything is missing or wrong. Kept private -
    callers should go through ``load_api_congig``.

    :param config:
    :return:
    """
    if not isinstance(config, dict):
        raise ConfigurationError(f"API config must be a mapping, got {type(config).__name__}")

    # 1. both required top-level sections must exist.
    missing_sections = [s for s in _API_REQUIRED_SECTIONS if s not in config]
    if missing_sections:
        raise ConfigurationError(f"API config missing required section(s): {missing_sections}")

    # 2. attribute query block validation.
    _validate_query_block(
        config["sage_trade_attribute_query"], required_keys=_ATTRIBUTE_REQUIRED_KEYS,
        section_name="sage_trade_attribute_query"
    )

    # 3. pnl query block validation
    _validate_query_block(
        config["sage_trade_pnl_query"], required_keys=_PNL_REQUIRED_KEYS, section_name="sage_trade_pnl_query"
    )


def _validate_query_block(block: Any, required_keys: tuple, section_name: str) -> None:
    """Validate a single SAGE query block (attribute or PNL)."""
    if not isinstance(block, dict):
        raise ConfigurationError(f"'{section_name}' must be a mapping, got {type(block).__name__}")

    missing = [k for k in required_keys if k not in block]
    if missing:
        raise ConfigurationError(f"'{section_name}' missing required key(s): {missing}")

    # filter must be a dict.
    if not isinstance(block["filter"], dict):
        raise ConfigurationError(f"'{section_name}' must be a mapping, got {type(block["filter"]).__name__}")

# ------------------------------------------------------------------------------------
# Preprocessing configuration loader.
# ------------------------------------------------------------------------------------

def load_model_config(config_file: Union[Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load and structurally validate the Rade Static Replication pipeline config.

    The input may either be a path to ``pipeline_config.yaml`` or an already-loaded dict
    (useful for tests that build the config inline).

    :param config_file:
    :return:
    """
    # Accepts either a path or a pre-loaded dict so callers can inject test data.
    if isinstance(config_file, dict):
        config = config_file
    elif isinstance(config_file, (str, Path)):
        config = load_yaml(config_file)
    else:
        # Anything else is a programmer error.
        raise ConfigurationError(f"load_model_config expects a path or dict, got {type(config_file).__name__}")
    _validate_model_config(config)
    return config


def _validate_model_config(config: Dict[str, Any]) -> None:
    """
    Structural validation for the Rade Static Replication pipeline.

    Raises ConfigurationError with a clear, actionable message if anything is missing or wrong.
    :param config:
    :return:
    """
    if not isinstance(config, dict):
        raise ConfigurationError(f"Static replication pipeline config must be a mapping, got {type(config).__name__}")

    # 1. both required top-level sections must exist.
    missing_sections = [s for s in _PIPELINE_REQUIRED_SECTIONS if s not in config]
    if missing_sections:
        raise ConfigurationError(f"Static replication pipeline config missing required sections: {missing_sections}")
