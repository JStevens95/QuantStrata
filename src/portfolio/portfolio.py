# src/portfolio/portfolio.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.portfolio.core import Portfolio, PortfolioResult, PortfolioTotals, PositionResult
from src.pricers.registry import InstrumentPricer, PricerRegistry


@dataclass(frozen=True, slots=True)
class PortfolioPricer:
    pricer_registry: PricerRegistry

    def price(
        self,
        portfolio: Portfolio,
        market: Any,  # noqa: ANN401
        *,
        pricer_id: Optional[str] = None,
    ) -> PortfolioResult:
        """
        Price every position in the portfolio and return per-position + totals.

        Routing
        -------
        - If position.pricer_id is set, it takes precedence.
        - Else, if a global `pricer_id` is provided, it is used.
        - Else, registry default routing is used.
        """
        per_position_results: list[PositionResult] = []

        total_pv: float = 0.0
        total_greeks: Dict[str, float] = {}

        for position in portfolio.positions:
            effective_pricer_id = position.pricer_id or pricer_id
            pricer = self.pricer_registry.resolve(position.instrument, pricer_id=effective_pricer_id)

            instrument_pv = float(pricer.price(position.instrument, market))
            if not math.isfinite(instrument_pv):
                raise ValueError(f"Non-finite instrument PV {instrument_pv}")

            position_pv = float(position.quantity) * instrument_pv

            instrument_greeks = self._safe_greeks(pricer=pricer, instrument=position.instrument, market=market)

            position_greeks: Dict[str, float] = {}
            for greek_name, greek_value in instrument_greeks.items():
                if not math.isfinite(float(greek_value)):
                    raise ValueError(f"Non-finite greek value {float(greek_value)}")

                scaled_value = float(position.quantity) * float(greek_value)
                position_greeks[greek_name] = scaled_value
                total_greeks[greek_name] = total_greeks.get(greek_name, 0.0) + scaled_value

            total_pv += position_pv

            per_position_results.append(
                PositionResult(
                    position_id=position.position_id,
                    instrument_type=type(position.instrument).__name__,
                    quantity=float(position.quantity),
                    pv=float(position_pv),
                    greeks=position_greeks,
                )
            )

        totals = PortfolioTotals(pv=float(total_pv), greeks=total_greeks)
        return PortfolioResult(per_position=per_position_results, totals=totals)

    @staticmethod
    def _safe_greeks(pricer: InstrumentPricer, instrument: Any, market: Any) -> Dict[str, float]:  # noqa: ANN401
        greeks_fn = getattr(pricer, "greeks", None)
        if callable(greeks_fn):
            greeks = greeks_fn(instrument, market)
            return {str(k): float(v) for k, v in dict(greeks).items()}
        return {}