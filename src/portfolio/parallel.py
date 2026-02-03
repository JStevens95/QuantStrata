"""Parallel portfolio pricing via ThreadPoolExecutor."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.portfolio.core import Portfolio, PortfolioResult, PortfolioTotals, Position


@dataclass(frozen=True, slots=True)
class ParallelPortfolioPricer:
    """
    Wrapper that prices portfolio positions in parallel using a thread pool.

    Assumes the underlying pricer and market are read-only during price(instrument, market);
    do not use with pricers or markets that mutate shared state.

    Parameters
    ----------
    portfolio_pricer : PortfolioPricer-like
        Must have price(portfolio, market, *, pricer_id) -> PortfolioResult.
    max_workers : int or None
        Max threads. None = use cpu_count or 4. 1 = sequential. >1 = parallel.
    """

    portfolio_pricer: Any  # PortfolioPricer
    max_workers: Optional[int] = None

    def price(
        self,
        portfolio: Portfolio,
        market: Any,  # noqa: ANN401
        *,
        pricer_id: Optional[str] = None,
    ) -> PortfolioResult:
        """Price every position (in parallel if max_workers > 1) and return same result as sequential."""
        positions = list(portfolio.positions)
        if not positions:
            return self.portfolio_pricer.price(portfolio, market, pricer_id=pricer_id)

        n_workers = self.max_workers
        if n_workers is None:
            n_workers = getattr(os, "cpu_count", lambda: 4)() or 4
        if n_workers < 2:
            return self.portfolio_pricer.price(portfolio, market, pricer_id=pricer_id)

        def price_one_position(position: Position) -> tuple[int, PortfolioResult]:
            single = Portfolio(positions=[position])
            result = self.portfolio_pricer.price(single, market, pricer_id=pricer_id)
            idx = next(i for i, p in enumerate(positions) if p is position)
            return (idx, result)

        with ThreadPoolExecutor(max_workers=min(n_workers, len(positions))) as executor:
            future_to_pos = {executor.submit(price_one_position, pos): pos for pos in positions}
            idx_to_result: Dict[int, PortfolioResult] = {}
            for future in as_completed(future_to_pos):
                idx, res = future.result()
                idx_to_result[idx] = res

        per_position = [idx_to_result[i].per_position[0] for i in range(len(positions))]
        total_pv = sum(r.pv for r in per_position)
        total_greeks: Dict[str, float] = {}
        for r in per_position:
            for k, v in r.greeks.items():
                total_greeks[k] = total_greeks.get(k, 0.0) + v

        return PortfolioResult(
            per_position=per_position,
            totals=PortfolioTotals(pv=float(total_pv), greeks=total_greeks),
        )
