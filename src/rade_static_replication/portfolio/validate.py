"""
Portfolio validation (stage 2, second half).

Fail-fast on the invariants the rest of the pipeline assumes; return non-fatal
warnings for the caller/log.
"""
from __future__ import annotations

import logging
from typing import List

from src.rade_static_replication.domain.contracts import Portfolio
from src.rade_static_replication.domain.errors import PortfolioError
from src.rade_static_replication.portfolio.normalise import ASSET_CLASS_COL

logger = logging.getLogger(__name__)


def validate(portfolio: Portfolio) -> List[str]:
    """Raise :class:`PortfolioError` on any breach; return warnings list."""
    attr = portfolio.attributes
    warnings: List[str] = []

    if attr.empty:
        raise PortfolioError("portfolio has no trades after normalisation")
    if not attr.index.is_unique:
        dupes = attr.index[attr.index.duplicated()].unique().tolist()
        raise PortfolioError(f"duplicate trade ids after normalisation: {dupes[:5]}")
    if ASSET_CLASS_COL not in attr.columns:
        raise PortfolioError(f"normalised attributes missing {ASSET_CLASS_COL!r}")
    missing_ac = attr.index[attr[ASSET_CLASS_COL].isna()].tolist()
    if missing_ac:
        raise PortfolioError(f"{len(missing_ac)} trades have no resolved AssetClass: {missing_ac[:5]}")

    missing_pnl = [t for t in portfolio.trade_ids if t not in portfolio.target_pnl.index]
    if missing_pnl:
        warnings.append(f"{len(missing_pnl)} trades have no target PnL row")
    if portfolio.scenario_ids.size == 0:
        raise PortfolioError("target PnL has no scenario columns")

    for w in warnings:
        logger.warning("portfolio validation: %s", w)
    return warnings
