"""
Resolve the unique risk-factor ids required by a portfolio.

Config (YAML) + optional mapping CSV per asset class, or ids already on the
trade row. Returns a sorted deduplicated list — nothing else.

::

    rules = load_rules("src/rade_sr/configs/risk_factor_resolution.yaml")
    ids = factor_universe(attributes, rules)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.rade_sr.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class FactorRule:
    """One asset class: how to collect factor ids from trades."""

    asset_class: str
    source: str  # "mapping" | "attribute"
    factor_cols: tuple[str, ...]
    dependency_cols: tuple[str, ...] = ()
    mapping: Optional[Union[str, Path, pd.DataFrame]] = None
    mapping_key_col: str = "ccy"
    key_field: Optional[str] = None
    normalize_keys: bool = True

    def collect_columns(self, *, include_dependencies: bool) -> List[str]:
        cols = list(self.factor_cols)
        if include_dependencies:
            cols.extend(self.dependency_cols)
        return cols

    @property
    def trade_columns(self) -> List[str]:
        if self.source == "attribute":
            return list(dict.fromkeys([*self.factor_cols, *self.dependency_cols]))
        return [self.key_field] if self.key_field else []


def factor_universe(
    attributes: pd.DataFrame,
    rules: Dict[str, FactorRule],
    *,
    asset_class_col: str = "AssetClass",
    include_dependencies: bool = True,
) -> List[str]:
    """Sorted, deduplicated risk-factor ids for the portfolio."""
    found: set[str] = set()
    for rule in rules.values():
        cols = rule.collect_columns(include_dependencies=include_dependencies)
        found.update(_resolve(attributes, rule, asset_class_col, cols))
    return sorted(found)


def load_rules(
    yaml_path: Union[str, Path],
    *,
    section: str = "risk_factor_resolution",
) -> Dict[str, FactorRule]:
    """Load rules from YAML; ``mapping`` paths are relative to the YAML file."""
    import yaml

    path = Path(yaml_path)
    with path.open() as f:
        data = yaml.safe_load(f)
    if section not in data:
        raise ConfigurationError(f"{path}: missing {section!r}")
    return build_rules(data[section], base_dir=path.parent)


def build_rules(
    config: Dict[str, Dict[str, Any]],
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, FactorRule]:
    """Build rules from a dict — one block per asset class, each with ``source``."""
    return {name: _parse_rule(name, dict(spec), base_dir) for name, spec in config.items()}


def _parse_rule(
    name: str,
    spec: Dict[str, Any],
    base_dir: Optional[Union[str, Path]],
) -> FactorRule:
    if "source" not in spec:
        raise ConfigurationError(f"[{name}] must declare 'source: mapping' or 'source: attribute'")
    source = str(spec.pop("source")).lower()
    if source not in ("mapping", "attribute"):
        raise ConfigurationError(f"[{name}] source must be 'mapping' or 'attribute'")

    factor_cols = tuple(spec.pop("factor_cols"))
    dependency_cols = tuple(spec.pop("dependency_cols", ()))
    normalize_keys = bool(spec.pop("normalize_keys", True))

    if source == "attribute":
        if spec:
            raise ConfigurationError(f"[{name}] unexpected keys: {sorted(spec)}")
        return FactorRule(
            asset_class=name,
            source=source,
            factor_cols=factor_cols,
            dependency_cols=dependency_cols,
            normalize_keys=normalize_keys,
        )

    mapping = spec.pop("mapping")
    mapping_key_col = spec.pop("mapping_key_col", "ccy")
    key_field = spec.pop("key_field")
    if not key_field:
        raise ConfigurationError(f"[{name}] source=mapping requires 'key_field'")

    if not isinstance(mapping, pd.DataFrame):
        path = Path(mapping)
        if base_dir is not None and not path.is_absolute():
            mapping = Path(base_dir) / path

    if spec:
        raise ConfigurationError(f"[{name}] unexpected keys: {sorted(spec)}")

    return FactorRule(
        asset_class=name,
        source=source,
        factor_cols=factor_cols,
        dependency_cols=dependency_cols,
        mapping=mapping,
        mapping_key_col=mapping_key_col,
        key_field=key_field,
        normalize_keys=normalize_keys,
    )


def _resolve(
    attributes: pd.DataFrame,
    rule: FactorRule,
    asset_class_col: str,
    columns: Sequence[str],
) -> set[str]:
    mask = attributes[asset_class_col].astype(str).str.lower() == rule.asset_class.lower()
    sub = attributes.loc[mask]
    if sub.empty:
        return set()

    for col in rule.trade_columns:
        if col not in sub.columns:
            raise ConfigurationError(
                f"[{rule.asset_class}] attributes missing {col!r}; has {list(sub.columns)}"
            )

    if rule.source == "attribute":
        return _collect(sub, columns)

    keys = sub.reset_index(names="trade_id")[["trade_id", rule.key_field]].copy()
    keys["key"] = keys[rule.key_field].astype(str).str.strip()
    if rule.normalize_keys:
        keys["key"] = keys["key"].str.upper()
    keys = keys.loc[keys["key"] != "", ["trade_id", "key"]]

    mapping = _read_mapping(rule, columns)
    merged = keys.merge(mapping, how="left", left_on="key", right_on=rule.mapping_key_col)
    matched = merged[merged[rule.mapping_key_col].notna()]
    return _collect(matched, columns)


def _read_mapping(rule: FactorRule, columns: Sequence[str]) -> pd.DataFrame:
    if isinstance(rule.mapping, pd.DataFrame):
        df = rule.mapping.copy()
    else:
        path = Path(rule.mapping)  # type: ignore[arg-type]
        df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)

    missing = [c for c in {rule.mapping_key_col, *columns} if c not in df.columns]
    if missing:
        raise ConfigurationError(f"[{rule.asset_class}] mapping missing {missing}")

    if rule.normalize_keys:
        df = df.copy()
        df[rule.mapping_key_col] = df[rule.mapping_key_col].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=[rule.mapping_key_col], keep="first")


def _collect(frame: pd.DataFrame, columns: Sequence[str]) -> set[str]:
    stacked = frame[list(columns)].stack(future_stack=True)
    values = stacked.dropna().astype(str).str.strip()
    return set(values[values != ""].tolist())
