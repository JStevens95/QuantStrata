"""
PnL engine.

Computes each factor's elementary scenario-PnL matrix by routing to its pricer's
vectorised ``scenario_pnl``. Per-factor work is independent, so factors fan out across a
thread pool (the heavy numerics run in compiled kernels that release the GIL when Numba
is present; otherwise it degrades to serial-equivalent throughput).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.domain.contracts import (
    BasePriceSet,
    ElementaryUniverse,
    FactorDataSet,
    FactorPnL,
    PnLResult,
)
from src.rade_static_replication.domain.errors import PnLError

logger = logging.getLogger(__name__)


def _price_one_factor(factor_id, trades, factor_data, base_prices, registry) -> FactorPnL:
    """Price one factor's elementary trades across all scenarios."""
    plugin = registry.get(factor_data.spec.asset_class)
    pnl_matrix = plugin.pricer.scenario_pnl(trades, factor_data, base_prices)
    return FactorPnL(
        factor_id=factor_id,
        trade_ids=[trade.trade_id for trade in trades],
        scenario_ids=factor_data.scenarios.scenario_ids,
        pnl=pnl_matrix,
    )


def run_pnl_engine(
    elementary: ElementaryUniverse,
    factor_data: FactorDataSet,
    base: BasePriceSet,
    registry: Registry,
    max_workers: int = 1,
) -> PnLResult:
    """Compute the elementary scenario PnL for every primary factor."""
    # Only factors that actually produced elementary trades need pricing.
    factor_ids = [fid for fid in elementary.by_factor if elementary.by_factor[fid]]

    def _compute(factor_id: str) -> FactorPnL:
        try:
            return _price_one_factor(
                factor_id, elementary.by_factor[factor_id],
                factor_data.factors[factor_id], base.by_factor[factor_id], registry,
            )
        except Exception as exc:  # attach the offending factor for a clear failure
            raise PnLError(f"PnL failed for factor {factor_id}: {exc}") from exc

    results: Dict[str, FactorPnL] = {}
    if max_workers > 1 and len(factor_ids) > 1:
        # Factors are independent -> fan out across threads.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for factor_pnl in pool.map(_compute, factor_ids):
                results[factor_pnl.factor_id] = factor_pnl
    else:
        for factor_id in factor_ids:
            factor_pnl = _compute(factor_id)
            results[factor_pnl.factor_id] = factor_pnl

    logger.info("Computed PnL for %d factors", len(results))
    return PnLResult(by_factor=results)
