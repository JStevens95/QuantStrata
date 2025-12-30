from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Type

from ..portfolio.core import Portfolio, PortfolioResult, PortfolioTotals, PositionResult


class InstrumentPricer(Protocol):
    """
    Minimal protocol for instrument pricers used by PortfolioPricer.

    Any pricer placed in the registry must implement:
      - price(instrument, market) -> float
    and may optionally implement:
      - greeks(instrument, market) -> Dict[str, float]
    """

    def price(self, instrument: Any, market: Any) -> float:  # noqa: ANN401 (Any is intentional)
        ...

    # Optional by convention; PortfolioPricer will detect at runtime.
    def greeks(self, instrument: Any, market: Any) -> Dict[str, float]:  # noqa: ANN401
        ...


class UnsupportedInstrumentError(TypeError):
    """Raised when PortfolioPricer cannot route an instrument to a registered pricer."""


@dataclass(frozen=True, slots=True)
class PortfolioPricer:
    """
    Prices a mixed portfolio using a registry mapping instrument types -> pricers.

    Example
    -------
    registry = {EuropeanFxOption: BlackScholesPricer()}
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    result = portfolio_pricer.price(portfolio, market)
    print(result.totals.pv)
    """

    pricer_registry: Mapping[Type[Any], InstrumentPricer]

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
            # Resolve a pricer based on the instrument type.
            pricer = self._resolve_pricer(position.instrument)

            # Price the instrument PV and scale by position quantity.
            instrument_pv = float(pricer.price(position.instrument, market))
            position_pv = float(position.quantity) * instrument_pv

            # Compute greeks if the pricer supports it.
            instrument_greeks = self._safe_greeks(pricer=pricer, instrument=position.instrument, market=market)

            # Scale greeks by quantity and accumulate.
            position_greeks: Dict[str, float] = {}
            for greek_name, greek_value in instrument_greeks.items():
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

    def _resolve_pricer(self, instrument: Any) -> InstrumentPricer:  # noqa: ANN401
        """
        Resolve the best pricer for an instrument.

        Resolution rules
        ---------------
        1) Exact type match in registry (fast path).
        2) Otherwise, select the *most specific* registered base class such that:
              isinstance(instrument, registered_type) is True
           using MRO distance to choose the closest match.

        Raises
        ------
        UnsupportedInstrumentError
            If no pricer can be found for the instrument type.
        """
        instrument_type = type(instrument)

        # Fast path: exact type registered.
        exact = self.pricer_registry.get(instrument_type)
        if exact is not None:
            return exact

        # Otherwise, look for compatible base classes in the registry.
        best: Optional[Tuple[int, Type[Any], InstrumentPricer]] = None
        mro = instrument_type.mro()

        for registered_type, pricer in self.pricer_registry.items():
            # Skip non-types defensively (should not happen in well-typed code).
            if not isinstance(registered_type, type):
                continue

            if isinstance(instrument, registered_type):
                # Compute how "close" the registered type is in the instrument's MRO.
                # Smaller index = more specific match.
                try:
                    distance = mro.index(registered_type)
                except ValueError:
                    # Should be unlikely if isinstance() is True, but keep safe.
                    distance = 10**9

                candidate = (distance, registered_type, pricer)
                if best is None or candidate[0] < best[0]:
                    best = candidate

        if best is not None:
            return best[2]

        raise UnsupportedInstrumentError(
            f"No pricer registered for instrument type: {instrument_type.__name__}. "
            f"Registered types: {[t.__name__ for t in self.pricer_registry.keys()]}"
        )

    @staticmethod
    def _safe_greeks(pricer: InstrumentPricer, instrument: Any, market: Any) -> Dict[str, float]:  # noqa: ANN401
        """
        Call pricer.greeks() if present; otherwise return an empty dict.

        This allows you to include pricers that only support PV in early iterations.
        """
        greeks_fn = getattr(pricer, "greeks", None)
        if callable(greeks_fn):
            greeks = greeks_fn(instrument, market)
            # Ensure we always return a plain dict[str,float].
            return {str(k): float(v) for k, v in dict(greeks).items()}
        return {}