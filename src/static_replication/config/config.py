"""
OrchestratorConfig - typed pipeline configuration

Loaded form a YAML file and passed to Orchestrator.from_config().
Eliminates the untyped Dict[str, Any] that run() previously accepted.

Minimal pipeline_config.yaml example
------------------------------------
# -------------------------------------------------------------------------------------------------
# Top level --> STATIC REPLICATION PREPROCESSING CONFIGURATION.
# -------------------------------------------------------------------------------------------------
root_folder: "<ROOT_FOLDER>"

# -------------------------------------------------------------------------------------------------
# Defaults applied to portfolio construction.
# -------------------------------------------------------------------------------------------------
portfolio_defaults:
  trade_id_key: ["AssetClassCode", "DeskName", "UnderlyingOption", "TRBookKey", "ProductCode", "TradeCode"]
  currency_field: "UnderlyingOption"

# -------------------------------------------------------------------------------------------------
# Risk factor resolution configuration.
# -------------------------------------------------------------------------------------------------
factor_config:
  fx_config:
    source: "file"
    mapping_file: "<PATH_TO_FX_MAPPING_FILE>"
    mapping_key: "currency"
    attrs_key: "UnderlyingOption"
    factor_cols: ["fx_risk_factor_1", "fx_risk_factor_2"]
    dependency_cols: ["ir_risk_factor_1", "ir_risk_factor_2"]
  ir_config:
    source: "attribute"
    factor_cols: ["CurveCode"]

# -------------------------------------------------------------------------------------------------
# Defaults applied to every Asset built. Per-factor overrides go under ``factor_configs``.
# -------------------------------------------------------------------------------------------------
asset_defaults:
  cob_date: "2026-03-31"
  start_date: "2024-01-01"
  end_date: "2026-03-31"
  calc_type: "MAXSVAR"

# -------------------------------------------------------------------------------------------------
# Elementary trade generator configuration. fx_config / ir_config let each asset class
# declare its own strike / tenor grid; top level keys apply to both if no sub dict is given
# -------------------------------------------------------------------------------------------------
trade_config:
  fx_config:
    payoff_types: ["EC", "EP", "DC", "DP", "QC", "QP"]
    strike_values: [0.9, 1.0, 1.1]
    tenor_values: [0.083, 0.25, 0.5, 1.0]
    strike_convention: "relative"
    plot_option_surface: False
  ir_config:
    payoff_types: ["IRS"]
    strike_values: [0.0]
    tenor_values: [0.083, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    strike_convention: "absolute"
    plot_option_surface: False

# -------------------------------------------------------------------------------------------------
# Slicing strategy for the final ReplicationJobs.
# Use a list for composite keys e.g. ["AssetClass", "Desk"]
# -------------------------------------------------------------------------------------------------
cluster_config:
  normalise_target_notional: 1
  cluster_key: ["AssetClassCode", "UnderlyingOption", "ProductCode"]

# PnL computation backend - "serial" | "threading" | "multiprocessing"
execution_backend: "serial"

# Base label for this run; a UTC datetime + hash suffix is appended at load time.
run_id: "mock_run"
"""
from __future__ import annotations

import hashlib
import logging

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union
from dataclasses import dataclass, field

# define module level logging.
logger = logging.getLogger(__name__)

# External fx risk factor mapping path.
_FX_MAPPING_PATH: Path = Path(__file__).parents[1] / "configs" / "fx_mapping.csv"


def resolve_run_id(base: str, *, at: datetime | None = None) -> str:
    """
    Turn a YAML ``run_id`` label into a unique, filesystem-safe run folder name.

    Appends a UTC datetime stamp and a short hash so repeated runs with the same
    base label (e.g. ``mock_run``) never overwrite prior artifacts.

    Example
    -------
    >>> resolve_run_id("mock_run")  # doctest: +SKIP
    'mock_run_20260303_143022_a1b2c3d4'

    Parameters
    ----------
    base :
        User-provided label from ``pipeline_config.yaml`` (``run_id`` key).
    at :
        Optional timestamp for tests; defaults to current UTC time.
    """
    label = str(base).strip().rstrip("_") or "run"
    moment = at or datetime.now(timezone.utc)
    # Human-readable UTC stamp for audit trails.
    stamp = moment.strftime("%Y%m%d_%H%M%S")
    # Short hash of ISO time — extra uniqueness if two runs start in the same second.
    digest = hashlib.sha256(moment.isoformat().encode()).hexdigest()[:8]
    return f"{label}_{stamp}_{digest}"


@dataclass
class OrchestratorConfig:
    """
    Typed configuration for the rade_static_replication pipeline.

    All field have safe defaults so only the fields that differ from the defaults need to appear in the
    pipeline_config.yaml file.
    """

    # root_folder:
    root_folder: str = field(default_factory=str)

    # run_id: base label from YAML; resolved unique id computed at load time.
    run_id_base: str = "run"
    run_id: str = field(default_factory=str)

    # portfolio_defaults:
    portfolio_config: Dict[str, Any] = field(default_factory=lambda : {
        "trade_id_key": ["AssetClassCode", "DeskName", "UnderlyingOption", "TRBookKey", "ProductCode", "TradeCode"],
        "currency_field": "UnderlyingOption"
    })

    # factor_config:
    factor_config: Dict[str, Any] = field(default_factory=lambda : {
        "fx_config": {
            "source": "file", "mapping_file": str(_FX_MAPPING_PATH), "mapping_key": "currency",
            "attrs_key": "UnderlyingOption", "factor_cols": ["fx_risk_factor_1", "fx_risk_factor_2"],
            "dependency_cols": ["ir_risk_factor_1", "ir_risk_factor_2"],
            "asset_class_of": {
                "ir_risk_factor_1": "IR",
                "ir_risk_factor_2": "IR",
            },
        },
        "ir_config": {
            "source": "attribute", "factor_cols": ["CurveCode"]
        }
    })

    # asset_defaults:
    asset_config: Dict[str, Any] = field(default_factory=lambda : {"calc_type": "MAXSVAR"})

    # trade_config:
    trade_config: Dict[str, Any] = field(default_factory=lambda : {
        "fx_config": {
            "payoff_types": ["EC", "EP"], "strike_values": [0.9, 1.0, 1.1], "tenor_values": [0.083, 0.25, 0.5, 1.0],
            "strike_convention": "relative",
        },
        "ir_config": {
            "payoff_types": ["IRS"], "strike_values": [0.0], "tenor_values": [0.083, 0.25, 0.5, 1.0],
            "strike_convention": "absolute",
        }
    })

    # cluster_config:
    cluster_config: Dict[str, Any] = field(default_factory=lambda : {
        "normalise_target_notional": 1, "cluster_key": ["AssetClassCode", "UnderlyingOption", "ProductCode"],
    })

    # execution_backend:
    execution_backend: str = "serial"

    # folder config placeholder:
    folder_config: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure ``run_id`` is always the resolved (unique) value."""
        if not self.run_id:
            self.run_id = resolve_run_id(self.run_id_base)

    # ------------------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "OrchestratorConfig":
        """
        Load config from a YAML file.

        Unknown keys are silently ignored. Missing keys fall back to dataclass defaults so a minimal YAML only needs
        to specify what differs.

        :param path:
        :return:
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required to load OrchestratorConfig --> pip install pyyaml") from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(F"OrchestratorConfig: config file not found: {path}")

        with path.open("r", encoding="utf-8") as stream:
            raw: Dict[str, Any] = yaml.safe_load(stream) or {}

        logger.info("OrchestratorConfig: loaded from %s", path)
        return cls._from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "OrchestratorConfig":
        """
        Build a OrchestratorConfig from a plain dict (e.g. for tests).

        :param raw:
        :return:
        """
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "OrchestratorConfig":
        """Construct a OrchestratorConfig from a plain dict (e.g. for tests)."""

        # get all dataclass defaults.
        defaults = cls()

        # root_folder: artifacts path.
        root_folder = raw.get("root_folder", defaults.root_folder)

        # run_id: YAML supplies the base label; we store the unique resolved id.
        run_id_base = str(raw.get("run_id", defaults.run_id_base))
        run_id = resolve_run_id(run_id_base)

        # portfolio_config:
        portfolio_config = {**defaults.portfolio_config, **raw.get("portfolio_config", {})}

        # factor config:
        factor_config = dict(defaults.factor_config)
        raw_factor = raw.get("factor_config", {})
        for key in ("fx_config", "ir_config"):
            if key in raw_factor:
                factor_config[key] = {**(defaults.factor_config.get(key, {})), **raw_factor[key]}

        # asset_config:
        asset_config = {**defaults.asset_config, **raw.get("asset_config", {})}

        # trade_config:
        trade_config = dict(defaults.trade_config)
        raw_trade = raw.get("trade_config", {})
        for key in ("fx_config", "ir_config"):
            if key in raw_trade:
                trade_config[key] = {**(defaults.trade_config.get(key, {})), **raw_trade[key]}

        # cluster_config:
        cluster_config = {**defaults.cluster_config, **raw.get("cluster_config", {})}

        # execution_backend:
        execution_backend = raw.get("execution_backend", defaults.execution_backend)

        return cls(
            root_folder=root_folder,
            run_id_base=run_id_base,
            run_id=run_id,
            portfolio_config=portfolio_config,
            factor_config=factor_config,
            asset_config=asset_config,
            trade_config=trade_config,
            cluster_config=cluster_config,
            execution_backend=execution_backend,
        )

    # ------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------
