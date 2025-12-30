from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from src.portfolio.core import Portfolio, PortfolioResult, PortfolioTotals, PositionResult
from src.pricers.registry import InstrumentPricer, PricerRegistry


@dataclass(frozen=True, slots=True)
class PortfolioPricer:
    """
    Prices a mixed portfolio using a PricerRegistry.

    Example
    -------
    registry = DefaultPricerRegistry().build()   # returns PricerRegistry
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    result = portfolio_pricer.price(portfolio, market)
    print(result.totals.pv)
    """

    pricer_registry: PricerRegistry

    def price(self, portfolio: Portfolio, market: Any) -> PortfolioResult:  # noqa: ANN401
        """
        Price every position in the portfolio and return per-position + totals.

        Notes
        -----
        - Each position PV is scaled by quantity:
            position_pv = quantity * instrument_pv
        - Greeks (if supported) are also scaled by quantity and aggregated by key.
        """
        per_position_results: list[PositionResult] = []

        total_pv: float = 0.0
        total_greeks: Dict[str, float] = {}

        for position in portfolio.positions:
            # Resolve a pricer based on the instrument type (delegated to registry).
            pricer = self.pricer_registry.resolve(position.instrument)

            # Price the instrument PV and scale by position quantity.
            instrument_pv = float(pricer.price(position.instrument, market))
            position_pv = float(position.quantity) * instrument_pv

            # Compute greeks if the pricer supports it.
            instrument_greeks = self._safe_greeks(pricer=pricer, instrument=position.instrument, market=market)

            # Scale greeks by quantity and accumulate.
            position_greeks: Dict[str, float] = {}
            for greek_name, greek_value in instrument_greeks.items():
                if not math.isfinite(float(greek_value)):
                    raise ValueError(f"Non-finite greek value {float(greek_value)}")
                scaled_value = float(position.quantity) * float(greek_value)
                position_greeks[greek_name] = scaled_value
                total_greeks[greek_name] = total_greeks.get(greek_name, 0.0) + scaled_value

            # Accumulate portfolio PV.
            total_pv += position_pv

            per_position_results.append(
                PositionResult(
                    position_id=position.position_id,
                    instrument_type=type(position.instrument).__name__,
                    quantity=float(position.quantity),
                    pv=position_pv,
                    greeks=position_greeks,
                )
            )

        totals = PortfolioTotals(pv=float(total_pv), greeks=total_greeks)
        return PortfolioResult(per_position=per_position_results, totals=totals)

    @staticmethod
    def _safe_greeks(pricer: InstrumentPricer, instrument: Any, market: Any) -> Dict[str, float]:  # noqa: ANN401
        """
        Call pricer.greeks() if present; otherwise return an empty dict.

        This allows you to include pricers that only support PV in early iterations.
        """
        greeks_fn = getattr(pricer, "greeks", None)
        if callable(greeks_fn):
            greeks = greeks_fn(instrument, market)
            return {str(k): float(v) for k, v in dict(greeks).items()}
        return {}