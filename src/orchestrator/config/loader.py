"""
Config loader.

Supports:
- JSON (always available)
- YAML (optional, requires PyYAML)

We keep the loader strict: config root must be a mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.orchestrator.config.schemas import IOConfig, RunConfig
from src.orchestrator.config.validate import validate_run_config
from src.orchestrator.core.errors import ConfigError


def load_run_config(path: str | Path) -> RunConfig:
    """
    Load a RunConfig from a JSON or YAML file.

    Parameters
    ----------
    path:
        Path to .json/.yaml/.yml config file.

    Returns
    -------
    RunConfig
        Validated typed configuration.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        raw_obj = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        raw_obj = _load_yaml(raw_text, path=str(config_path))
    else:
        raise ConfigError(f"Unsupported config extension: {suffix}. Use .json or .yaml/.yml")

    if not isinstance(raw_obj, dict):
        raise ConfigError("Config root must be a mapping/object.")

    cfg = _parse_run_config(raw_obj)
    return validate_run_config(cfg)


def _parse_run_config(data: Dict[str, Any]) -> RunConfig:
    """
    Convert a raw mapping into a RunConfig dataclass.

    Notes
    -----
    We keep parsing explicit and defensive so config errors are obvious.
    """
    io_raw = dict(data.get("io") or {})
    io_cfg = IOConfig(
        workdir=str(io_raw.get("workdir", "./.runs")),
        artifacts_dir=str(io_raw.get("artifacts_dir", "artifacts")),
        logs_dir=str(io_raw.get("logs_dir", "logs")),
    )

    return RunConfig(
        pipeline=str(data.get("pipeline", "")).strip(),
        only=data.get("only"),
        skip=data.get("skip"),
        resume_from=data.get("resume_from"),
        dry_run=bool(data.get("dry_run", False)),
        io=io_cfg,
        params=dict(data.get("params") or {}),
    )


def _load_yaml(text: str, *, path: str) -> Dict[str, Any]:
    """
    Load YAML safely if PyYAML is installed; otherwise raise a clear error.
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            "YAML config requested but PyYAML is not installed.\n"
            f"Config file: {path}\n"
            "Install with: pip install pyyaml"
        ) from exc

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ConfigError("YAML root must be a mapping/object.")
    return loaded