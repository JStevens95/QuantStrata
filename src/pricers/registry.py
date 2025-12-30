from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Protocol, Type

# ---- instruments ----
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption

# ---- pricers ----
from src.pricers.fx.spot import FxSpotPricer
from src.pricers.fx.forward import FxForwardPricer
from src.pricers.fx.european import FxEuropeanVanillaBsmPricer


class InstrumentPricer(Protocol):
    """
    Minimal protocol for instrument pricers.

    Required
    --------
    price(instrument, market) -> float

    Optional (if implemented, PortfolioPricer will use it)
    ------------------------------------------------------
    greeks(instrument, market) -> dict[str, float]
    """

    def price(self, instrument: Any, market: Any) -> float:  # noqa: ANN401
        ...

    def greeks(self, instrument: Any, market: Any) -> Dict[str, float]:  # noqa: ANN401
        ...


class UnsupportedInstrumentError(TypeError):
    """Raised when no registered pricer can be resolved for an instrument."""


@dataclass(slots=True)
class PricerRegistry:
    """
    Registry that resolves instrument -> pricer using MRO-first lookup.

    Resolution rules
    ---------------
    1) Exact type match
    2) Walk instrument_type.__mro__ and return the first registered base
    3) Optional fallback: isinstance(...) scan (for ABC virtual subclass cases)
    """

    _by_type: MutableMapping[Type[Any], InstrumentPricer] = field(default_factory=dict)
    _cache: MutableMapping[Type[Any], InstrumentPricer] = field(default_factory=dict)

    def register(self, instrument_type: Type[Any], pricer: InstrumentPricer, *, overwrite: bool = False) -> None:
        if not isinstance(instrument_type, type):
            raise TypeError("instrument_type must be a class/type.")

        if instrument_type in self._by_type and not overwrite:
            raise ValueError(
                f"Pricer already registered for {instrument_type.__name__}. "
                f"Pass overwrite=True to replace."
            )

        self._by_type[instrument_type] = pricer
        self._cache.clear()  # conservative: registrations can change resolution paths

    def unregister(self, instrument_type: Type[Any]) -> None:
        self._by_type.pop(instrument_type, None)
        self._cache.clear()

    def get(self, instrument_type: Type[Any]) -> Optional[InstrumentPricer]:
        return self._by_type.get(instrument_type)

    def as_mapping(self) -> Mapping[Type[Any], InstrumentPricer]:
        return dict(self._by_type)

    def resolve(self, instrument: Any) -> InstrumentPricer:  # noqa: ANN401
        instrument_type = type(instrument)

        cached = self._cache.get(instrument_type)
        if cached is not None:
            return cached

        # 1) Exact match
        pricer = self._by_type.get(instrument_type)
        if pricer is not None:
            self._cache[instrument_type] = pricer
            return pricer

        # 2) MRO walk (most-specific base class wins)
        for base in instrument_type.__mro__[1:]:
            pricer = self._by_type.get(base)
            if pricer is not None:
                self._cache[instrument_type] = pricer
                return pricer

        # 3) Fallback for ABC / virtual subclass registrations (rare, but safe)
        for registered_type, pricer in self._by_type.items():
            if isinstance(registered_type, type) and isinstance(instrument, registered_type):
                self._cache[instrument_type] = pricer
                return pricer

        raise UnsupportedInstrumentError(
            f"No pricer registered for instrument type: {instrument_type.__name__}. "
            f"Registered types: {[t.__name__ for t in self._by_type.keys()]}"
        )


@dataclass(slots=True)
class DefaultPricerRegistry:
    """
    Default registry factory (V1).

    This is intentionally small. V2/V3 just adds registrations.
    """

    def build(self) -> PricerRegistry:
        reg = PricerRegistry()
        reg.register(FxSpot, FxSpotPricer())
        reg.register(FxForward, FxForwardPricer())
        reg.register(EuropeanFxVanillaOption, FxEuropeanVanillaBsmPricer())
        return reg