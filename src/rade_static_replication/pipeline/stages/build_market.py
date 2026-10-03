"""
Stage 4 — build risk-factor market data.

Builds each factor in dependency-respecting order (so an FX factor receives its IR
dependency curves), routing to the registered asset-class builder.
"""
from __future__ import annotations

import logging
from typing import Dict

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.clients.base import MarketDataClient
from src.rade_static_replication.domain.contracts import (
    FactorDataSet,
    RiskFactorData,
    RiskFactorUniverse,
)

logger = logging.getLogger(__name__)


def build_factor_data(
    universe: RiskFactorUniverse, registry: Registry, client: MarketDataClient, cob_date: str,
) -> FactorDataSet:
    """Build every risk factor's snapshot + scenarios in dependency order."""
    built: Dict[str, RiskFactorData] = {}
    for fid in universe.build_order():
        spec = universe.specs[fid]
        plugin = registry.get(spec.asset_class)
        deps = {dep: built[dep] for dep in spec.dependencies if dep in built}
        built[fid] = plugin.builder.build(spec, deps, client, cob_date)
        logger.debug("built factor %s (%s)", fid, spec.asset_class)
    logger.info("Built %d risk-factor market objects", len(built))
    return FactorDataSet(factors=built)
