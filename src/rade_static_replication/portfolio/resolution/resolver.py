"""
Risk-factor resolution (stage 3).

Walks the portfolio, applies each asset class's :class:`AssetFactorRule`, and returns
a de-duplicated :class:`RiskFactorUniverse`: every primary factor, every dependency
factor needed to build them, and the trade -> primary-factors map used downstream.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    Portfolio,
    RiskFactorSpec,
    RiskFactorUniverse,
)
from src.rade_static_replication.domain.errors import FactorResolutionError
from src.rade_static_replication.portfolio.resolution.rules import MappingCache, resolve_row

logger = logging.getLogger(__name__)


def resolve(portfolio: Portfolio, config: OrchestratorConfig) -> RiskFactorUniverse:
    """Resolve the full risk-factor universe for the portfolio."""
    rules = {k.lower(): v for k, v in config.factors.rules.items()}
    if not rules:
        raise FactorResolutionError("no factor-resolution rules configured")

    specs: Dict[str, RiskFactorSpec] = {}
    factors_by_trade: Dict[str, List[str]] = {}
    mappings = MappingCache()

    for trade_id, row in portfolio.attributes.iterrows():
        asset_class = str(row.get("AssetClass", "")).lower()
        rule = rules.get(asset_class)
        if rule is None:
            logger.warning("no resolution rule for asset class %r (trade %s); skipped", asset_class, trade_id)
            factors_by_trade[str(trade_id)] = []
            continue

        primaries, dependencies = resolve_row(rule, row, mappings)
        dep_ids = tuple(d.factor_id for d in dependencies)

        for dep in dependencies:
            specs.setdefault(dep.factor_id, dep)

        primary_ids: List[str] = []
        for prim in primaries:
            spec = RiskFactorSpec(
                factor_id=prim.factor_id, asset_class=prim.asset_class,
                dependencies=dep_ids, is_primary=True, meta=prim.meta,
            )
            existing = specs.get(prim.factor_id)
            if existing is None or not existing.is_primary:
                specs[prim.factor_id] = spec
            primary_ids.append(prim.factor_id)

        factors_by_trade[str(trade_id)] = primary_ids

    if not specs:
        raise FactorResolutionError("resolution produced no risk factors")

    universe = RiskFactorUniverse(specs=specs, factors_by_trade=factors_by_trade)
    logger.info(
        "Resolved %d risk factors (%d primary, %d dependency-only)",
        len(specs), len(universe.primary_ids), len(specs) - len(universe.primary_ids),
    )
    return universe
