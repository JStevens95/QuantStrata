"""Stage 7 — scenario PnL (delegates to the engine)."""
from __future__ import annotations

from src.rade_static_replication.assets.base import Registry
from src.rade_static_replication.config.schema import EngineConfig
from src.rade_static_replication.domain.contracts import (
    BasePriceSet,
    ElementaryUniverse,
    FactorDataSet,
    PnLResult,
)
from src.rade_static_replication.pipeline.engine.pnl_engine import run_pnl_engine


def compute_pnl(
    elementary: ElementaryUniverse,
    factor_data: FactorDataSet,
    base: BasePriceSet,
    registry: Registry,
    engine: EngineConfig,
) -> PnLResult:
    """Compute per-factor elementary scenario PnL matrices."""
    return run_pnl_engine(elementary, factor_data, base, registry, max_workers=engine.max_workers)
