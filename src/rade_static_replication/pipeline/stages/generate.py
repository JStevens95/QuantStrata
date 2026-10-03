"""
Stage 5 — generate the elementary-trade universe.

Builds a replicating-basis grid for each *primary* factor (dependency-only factors are
market data, not replication targets), using that asset class's generator.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.config.schema import OrchestratorConfig
from src.rade_static_replication.domain.contracts import (
    ElementaryUniverse,
    FactorDataSet,
    RiskFactorUniverse,
)
from src.rade_static_replication.domain.instruments import ElementaryTrade

logger = logging.getLogger(__name__)


def generate_elementary(
    universe: RiskFactorUniverse,
    factor_data: FactorDataSet,
    registry: Registry,
    config: OrchestratorConfig,
) -> ElementaryUniverse:
    """Emit the elementary universe for each primary risk factor."""
    by_factor: Dict[str, List[ElementaryTrade]] = {}
    for fid in universe.primary_ids:
        rf = factor_data.factors[fid]
        plugin = registry.get(rf.spec.asset_class)
        grid = config.elementary.grids.get(rf.spec.asset_class, {})
        by_factor[fid] = plugin.generator.generate(rf, grid)

    universe_out = ElementaryUniverse(by_factor=by_factor)
    logger.info(
        "Generated %d elementary trades across %d factors", universe_out.n_trades, len(by_factor)
    )
    return universe_out
