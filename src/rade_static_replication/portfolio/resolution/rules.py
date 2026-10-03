"""
Rule application helpers for risk-factor resolution.

Pure functions that turn one trade row + an :class:`AssetFactorRule` into primary and
dependency :class:`RiskFactorSpec`s. Mapping CSVs are cached. The trivial FX
numeraire identity (``FX.SPOT.USD.USD``) is dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.rade_static_replication.config.schema import AssetFactorRule
from src.rade_static_replication.domain.contracts import RiskFactorSpec
from src.rade_static_replication.domain.errors import FactorResolutionError


def is_fx_identity(factor_id: str) -> bool:
    """True for the trivial USD-per-USD FX leg."""
    return factor_id.upper().endswith(".USD.USD")


def infer_asset_class(factor_id: str) -> str:
    """Best-effort asset class from a factor-id prefix (for dependency factors).

    Used only when a dependency column is not pinned via ``asset_class_of``. The default
    mapping matches this library's registry keys (``fx`` / ``rates``); if your plugins
    use different class names (e.g. ``ir``), pin them with ``asset_class_of`` in config.
    """
    head = factor_id.split(".")[0].upper()
    if head.startswith("FX"):
        return "fx"
    if head.startswith("IR"):
        return "rates"
    return head.lower()


def _valid(value) -> bool:
    """A field value is usable when it is present, non-NaN and non-blank."""
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
    rule: AssetFactorRule, row: pd.Series, mappings: MappingCache,
) -> Tuple[List[RiskFactorSpec], List[RiskFactorSpec]]:
    """Resolve one trade row into (primary specs, dependency specs)."""
    all_cols = rule.factor_cols + rule.dependency_cols
    if rule.is_file_source:
        # File source: read the lookup key from the trade, then pull factor ids from the CSV.
        key = row.get(rule.attrs_key)
        if not _valid(key):
            raise FactorResolutionError(f"[{rule.asset_class}] trade missing lookup key {rule.attrs_key!r}")
        table = mappings.get(rule.mapping_file, rule.mapping_key_col)
        if str(key) not in table.index:
            raise FactorResolutionError(
                f"[{rule.asset_class}] key {key!r} not found in mapping {rule.mapping_file}"
            )
        lookup = table.loc[str(key)]
        source_row = {col: lookup.get(col) for col in all_cols}
        meta_key = {"key": str(key)}
    else:
        # Attribute source: factor ids already live on the trade row.
        source_row = {col: row.get(col) for col in all_cols}
        meta_key = {}

    primaries = _make_specs(rule, rule.factor_cols, source_row, is_primary=True, meta_key=meta_key)
    dependencies = _make_specs(rule, rule.dependency_cols, source_row, is_primary=False, meta_key=meta_key)
    return primaries, dependencies


def _make_specs(
    rule: AssetFactorRule, columns: List[str], source_row: Dict, *, is_primary: bool, meta_key: Dict,
) -> List[RiskFactorSpec]:
    """Turn the named columns of one resolved row into de-duplicated factor specs."""
    specs: List[RiskFactorSpec] = []
    seen: set[str] = set()
    for column in columns:
        factor_id = source_row.get(column)
        if not _valid(factor_id):
            continue
        factor_id = str(factor_id).strip()
        # Drop the trivial USD/USD leg and any intra-row repeats.
        if is_fx_identity(factor_id) or factor_id in seen:
            continue
        seen.add(factor_id)
        specs.append(RiskFactorSpec(
            factor_id=factor_id,
            asset_class=_asset_class_for(rule, column, factor_id, is_primary),
            is_primary=is_primary,
            meta={"field": column, **meta_key},
        ))
    return specs


def _asset_class_for(rule: AssetFactorRule, column: str, factor_id: str, is_primary: bool) -> str:
    """Asset class of a produced factor: explicit override -> rule class (primary) -> inferred (dep)."""
    if column in rule.asset_class_of:
        return rule.asset_class_of[column]
    return rule.asset_class if is_primary else infer_asset_class(factor_id)
