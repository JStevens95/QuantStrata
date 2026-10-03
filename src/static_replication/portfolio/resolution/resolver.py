"""
Risk-factor resolution (pipeline stage 2).

Walks portfolio attributes, applies each asset class's :class:`AssetFactorRule`,
and returns a de-duplicated :class:`RiskFactorUniverse`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.static_replication.config.factor_schema import normalize_asset_class, parse_factor_rules
from src.static_replication.core.exceptions import FactorResolutionError
from src.static_replication.core.types import RiskFactorSpec, RiskFactorUniverse
from src.static_replication.portfolio.resolution.rules import (
    MappingCache,
    USD_SOFR_FACTOR_ID,
    primary_dependency_ids,
    resolve_row,
)

logger = logging.getLogger(__name__)

_DEFAULT_ASSET_CLASS_COLUMN = "AssetClassCode"
_FALLBACK_ASSET_CLASS_COLUMN = "AssetClass"


def resolve_factors(
    attributes: pd.DataFrame,
    factor_config: Dict[str, Any],
    *,
    base_dir: Optional[Path] = None,
    asset_class_column: Optional[str] = None,
) -> RiskFactorUniverse:
    """
    Resolve the full risk-factor universe from formatted portfolio attributes.

    Parameters
    ----------
    attributes :
        One row per trade (index = trade id or RangeIndex with id in columns).
    factor_config :
        Work-env ``factor_config`` block (``fx_config``, ``ir_config``, …).
    base_dir :
        Directory for resolving relative ``mapping_file`` paths (YAML parent).
    asset_class_column :
        Column used to pick the rule (default ``AssetClassCode``, else ``AssetClass``).
    """
    if attributes is None or attributes.empty:
        raise FactorResolutionError("attributes DataFrame is empty")

    rules = parse_factor_rules(factor_config, base_dir=base_dir)
    ac_col = asset_class_column or _resolve_asset_class_column(attributes)

    specs: Dict[str, RiskFactorSpec] = {}
    factors_by_trade: Dict[str, list[str]] = {}
    mappings = MappingCache()

    for trade_id, row in attributes.iterrows():
        trade_key = _trade_key(trade_id, row)
        asset_class = _row_asset_class(row, ac_col)
        rule = rules.get(asset_class)
        if rule is None:
            logger.warning(
                "no resolution rule for asset class %r (trade %s); skipped",
                asset_class,
                trade_key,
            )
            factors_by_trade[trade_key] = []
            continue

        primaries, dependencies = resolve_row(rule, row, mappings)

        for dep in dependencies:
            specs.setdefault(dep.factor_id, dep)

        if rule.asset_class == "FX" and USD_SOFR_FACTOR_ID not in specs:
            specs[USD_SOFR_FACTOR_ID] = RiskFactorSpec(
                factor_id=USD_SOFR_FACTOR_ID,
                asset_class="IR",
                is_primary=False,
                meta={"field": "usd_sofr", "synthetic": True},
            )

        primary_ids: list[str] = []
        for prim in primaries:
            if rule.asset_class == "FX":
                dep_ids = primary_dependency_ids(rule, prim, dependencies)
            else:
                dep_ids = tuple(d.factor_id for d in dependencies)

            spec = RiskFactorSpec(
                factor_id=prim.factor_id,
                asset_class=prim.asset_class,
                dependencies=dep_ids,
                is_primary=True,
                meta=prim.meta,
            )
            existing = specs.get(prim.factor_id)
            if existing is None or not existing.is_primary:
                specs[prim.factor_id] = spec
            primary_ids.append(prim.factor_id)

        factors_by_trade[trade_key] = primary_ids

    if not specs:
        raise FactorResolutionError("resolution produced no risk factors")

    universe = RiskFactorUniverse(specs=specs, factors_by_trade=factors_by_trade)
    logger.info(
        "Resolved %d risk factors (%d primary, %d dependency-only)",
        len(specs),
        len(universe.primary_ids),
        len(specs) - len(universe.primary_ids),
    )
    return universe


def _resolve_asset_class_column(attributes: pd.DataFrame) -> str:
    if _DEFAULT_ASSET_CLASS_COLUMN in attributes.columns:
        return _DEFAULT_ASSET_CLASS_COLUMN
    if _FALLBACK_ASSET_CLASS_COLUMN in attributes.columns:
        return _FALLBACK_ASSET_CLASS_COLUMN
    raise FactorResolutionError(
        f"attributes missing asset-class column "
        f"({_DEFAULT_ASSET_CLASS_COLUMN!r} or {_FALLBACK_ASSET_CLASS_COLUMN!r})"
    )


def _row_asset_class(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return normalize_asset_class(str(value))


def _trade_key(trade_id: Any, row: pd.Series) -> str:
    if isinstance(trade_id, str) and trade_id and trade_id != "nan":
        return trade_id
    for col in ("TradeCode", "trade_id", "PTSDealNumber"):
        if col in row.index and pd.notna(row.get(col)):
            return str(row[col]).strip()
    return str(trade_id)
