"""
Rule application helpers for risk-factor resolution.

Pure functions: one trade row + :class:`AssetFactorRule` -> primary and
dependency :class:`RiskFactorSpec` lists. Mapping CSVs are cached; trivial
``FX.SPOT.USD.USD`` legs are dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.static_replication.config.factor_schema import AssetFactorRule
from src.static_replication.core.exceptions import FactorResolutionError
from src.static_replication.core.types import RiskFactorSpec

# USD discount curve attached to every FX primary (USD-quoted leg).
USD_SOFR_FACTOR_ID = "IR_CURVE.RFR.USD.SOFR"


def is_fx_identity(factor_id: str) -> bool:
    """True for the trivial USD-per-USD FX leg."""
    return factor_id.upper().endswith(".USD.USD")


def infer_asset_class(factor_id: str) -> str:
    """Best-effort asset class from a factor-id prefix (dependency factors)."""
    head = factor_id.split(".")[0].upper()
    if head.startswith("FX"):
        return "FX"
    if head.startswith("IR"):
        return "IR"
    return head


def _valid(value) -> bool:
    return value is not None and pd.notna(value) and str(value).strip() != ""


class MappingCache:
    """Lazily loads and indexes mapping CSVs, keyed by path."""

    def __init__(self) -> None:
        self._cache: Dict[str, pd.DataFrame] = {}

    def get(self, path: str, key_field: str) -> pd.DataFrame:
        if path not in self._cache:
            df = pd.read_csv(Path(path))
            key_col = key_field if key_field in df.columns else df.columns[0]
            self._cache[path] = df.set_index(df[key_col].astype(str))
        return self._cache[path]


def resolve_row(
    rule: AssetFactorRule,
    row: pd.Series,
    mappings: MappingCache,
) -> Tuple[List[RiskFactorSpec], List[RiskFactorSpec]]:
    """Resolve one trade row into (primary specs, dependency specs)."""
    all_cols = rule.factor_cols + rule.dependency_cols
    if rule.is_file_source:
        key = row.get(rule.attrs_key)
        if not _valid(key):
            raise FactorResolutionError(
                f"[{rule.asset_class}] trade missing lookup key {rule.attrs_key!r}"
            )
        table = mappings.get(rule.mapping_file, rule.mapping_key_col)
        key_str = str(key).strip()
        if key_str not in table.index:
            raise FactorResolutionError(
                f"[{rule.asset_class}] key {key!r} not found in mapping {rule.mapping_file}"
            )
        lookup = table.loc[key_str]
        if isinstance(lookup, pd.DataFrame):
            lookup = lookup.iloc[0]
        source_row = {col: lookup.get(col) for col in all_cols}
        meta_key = {"key": key_str}
    else:
        source_row = {col: row.get(col) for col in all_cols}
        meta_key = {}

    primaries = _make_specs(
        rule, rule.factor_cols, source_row, is_primary=True, meta_key=meta_key,
    )
    dependencies = _make_specs(
        rule, rule.dependency_cols, source_row, is_primary=False, meta_key=meta_key,
    )
    return primaries, dependencies


def _make_specs(
    rule: AssetFactorRule,
    columns: List[str],
    source_row: Dict,
    *,
    is_primary: bool,
    meta_key: Dict,
) -> List[RiskFactorSpec]:
    specs: List[RiskFactorSpec] = []
    seen: set[str] = set()
    for column in columns:
        factor_id = source_row.get(column)
        if not _valid(factor_id):
            continue
        factor_id = str(factor_id).strip()
        if is_fx_identity(factor_id) or factor_id in seen:
            continue
        seen.add(factor_id)
        specs.append(
            RiskFactorSpec(
                factor_id=factor_id,
                asset_class=_asset_class_for(rule, column, factor_id, is_primary),
                is_primary=is_primary,
                meta={"field": column, **meta_key},
            )
        )
    return specs


def _asset_class_for(
    rule: AssetFactorRule, column: str, factor_id: str, is_primary: bool,
) -> str:
    if column in rule.asset_class_of:
        return rule.asset_class_of[column]
    return rule.asset_class if is_primary else infer_asset_class(factor_id)


def primary_dependency_ids(
    rule: AssetFactorRule,
    prim: RiskFactorSpec,
    dependencies: List[RiskFactorSpec],
) -> Tuple[str, ...]:
    """
    IR factor ids for one FX primary: ``dependency_cols[i]`` for ``factor_cols[i]``, plus SOFR.

    Cross pairs (e.g. EURCNH) get (CCY1_IR, SOFR) and (CCY2_IR, SOFR), not both IR curves on each leg.
    """
    dep_by_field = {
        d.meta["field"]: d.factor_id
        for d in dependencies
        if d.meta.get("field")
    }

    field = prim.meta.get("field", "")
    try:
        idx = rule.factor_cols.index(field)
    except ValueError:
        idx = 0

    ids: List[str] = []

    if idx < len(rule.dependency_cols):
        ir_column = rule.dependency_cols[idx]
        local_id = dep_by_field.get(ir_column)
        if local_id and local_id != USD_SOFR_FACTOR_ID:
            ids.append(local_id)

    if USD_SOFR_FACTOR_ID not in ids:
        ids.append(USD_SOFR_FACTOR_ID)

    return tuple(ids)
