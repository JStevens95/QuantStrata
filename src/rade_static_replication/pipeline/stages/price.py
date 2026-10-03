"""Stage 6 — price the elementary universe at COB (base PVs)."""
from __future__ import annotations

from typing import Dict

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.domain.contracts import (
    BasePriceSet,
    ElementaryUniverse,
    FactorDataSet,
)


def price_base(
    elementary: ElementaryUniverse, factor_data: FactorDataSet, registry: Registry,
) -> BasePriceSet:
    """Compute COB present values for every elementary trade."""
    by_factor: Dict[str, Dict[str, float]] = {}
    for fid, trades in elementary.by_factor.items():
        rf = factor_data.factors[fid]
        plugin = registry.get(rf.spec.asset_class)
        by_factor[fid] = plugin.pricer.base_prices(trades, rf)
    return BasePriceSet(by_factor=by_factor)
