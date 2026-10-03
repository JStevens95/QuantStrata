"""
PnL engine stage — portfolio-wide vectorised PnL computation.

Groups all elementary trades by factor_id, makes one pricer call
per unique factor group, and returns a flat dict of per-trade PnL
vectors.  Factor groups are computed in parallel.

This runs in Phase 1 (portfolio-wide) so that all PnL is computed once.
Clusters receive pre-computed PnL slices in Phase 2 — no re-pricing.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

from src.rade_sr.core.types import ElementaryTrade, ElementaryTradePnL
from src.rade_sr.core.exceptions import PnLComputationError

logger = logging.getLogger(__name__)


class VectorisedPnLEngine:
    """Portfolio-wide PnL computation engine.

    Groups trades by factor_id, then computes PnL for each group in
    a single pricer call.  Factor groups can be processed in parallel
    via ProcessPoolExecutor.

    Parameters
    ----------
    pricer : OptionPricer
        The batch pricer instance. Must implement
        ``price_trades(trades, market_data) -> np.ndarray``.
    max_workers : int
        Number of parallel workers for factor group processing.
    """

    def __init__(self, pricer: Any, max_workers: int = 4) -> None:
        self._pricer = pricer
        self._max_workers = max_workers

    def compute_batch(
        self,
        trades: Dict[str, List[ElementaryTrade]],
        risk_factors: Dict[str, Any],
    ) -> Dict[str, Dict[str, ElementaryTradePnL]]:
        """Compute PnL for all trades, grouped by risk factor.

        Parameters
        ----------
        trades : dict[str, list[ElementaryTrade]]
            ``factor_id → [trades]``.  Already grouped — mirrors the
            ``assets`` dict produced by Phase 1.
        risk_factors : dict[str, Any]
            Loaded Asset instances keyed by factor_id.

        Returns
        -------
        dict[str, dict[str, ElementaryTradePnL]]
            ``factor_id → {trade_id → PnL}``.  Mirrors the input
            trades dict so cluster slicing is a dict subset.
        """
        n_trades = sum(len(ts) for ts in trades.values())
        results: Dict[str, Dict[str, ElementaryTradePnL]] = {}

        logger.info(
            "PnL engine: %d trades across %d factor groups, %d workers",
            n_trades, len(trades), self._max_workers,
        )

        if self._max_workers <= 1:
            for factor_id, group_trades in trades.items():
                if factor_id not in risk_factors:
                    continue
                results[factor_id] = self._price_group(
                    factor_id, group_trades, risk_factors[factor_id],
                )
        else:
            with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_factor = {
                    executor.submit(
                        self._price_group,
                        factor_id,
                        group_trades,
                        risk_factors[factor_id],
                    ): factor_id
                    for factor_id, group_trades in trades.items()
                    if factor_id in risk_factors
                }
                for future in as_completed(future_to_factor):
                    factor_id = future_to_factor[future]
                    try:
                        results[factor_id] = future.result()
                        logger.debug("Priced factor group: %s", factor_id)
                    except Exception as exc:
                        raise PnLComputationError(
                            f"PnL computation failed for factor {factor_id!r}"
                        ) from exc

        n_total = sum(len(pnls) for pnls in results.values())
        logger.info("PnL engine: computed %d trade PnLs", n_total)
        return results

    def _price_group(
        self,
        factor_id: str,
        trades: List[ElementaryTrade],
        asset: Any,
    ) -> Dict[str, ElementaryTradePnL]:
        """Price all trades for one risk factor group.

        Delegates to ``OptionPricer.price_trades()`` which handles
        shocks conversion, payoff grouping, and kernel dispatch.
        """
        pnl_matrix = self._pricer.price_trades(trades, asset)

        return {
            trade.trade_id: ElementaryTradePnL(
                trade_id=trade.trade_id,
                factor_id=factor_id,
                pnl_vector=pnl_matrix[i],
            )
            for i, trade in enumerate(trades)
        }
