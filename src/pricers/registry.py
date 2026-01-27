from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Protocol, Type

# import instruments
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.options.digital import EuropeanFxDigitalOption
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.instruments.fx.options.barrier import EuropeanFxBarrierOption
from src.instruments.fx.options.asian import EuropeanFxAsianOption
from src.instruments.fx.options.lookback import EuropeanFxLookbackOption
from src.instruments.fx.options.vanilla import AmericanFxVanillaOption


# import pricer (linear + non-linear)
from src.pricers.fx.spot import FxSpotPricer
from src.pricers.fx.forward import FxForwardPricer
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer
from src.pricers.fx.european_bsm import FxEuropeanDigitalBsmPricer
from src.pricers.fx.european_mc import FxEuropeanBarrierMcPricer, FxEuropeanVanillaMcPricer, FxEuropeanDigitalMcPricer, FxEuropeanAsianMcPricer, FxEuropeanLookbackMcPricer
from src.pricers.fx.european_fde import FxEuropeanVanillaFdPricer, FxEuropeanDigitalFdPricer
from src.pricers.fx.american_fde import FxAmericanVanillaFdPricer


class InstrumentPricer(Protocol):
    """
    Minimal protocol for instrument pricers.

    Required
    --------
    price(instrument, market) -> float

    Optional
    --------
    greeks(instrument, market) -> dict[str, float]
    (PortfolioPricer checks for this via getattr/callable.)
    """
    def price(self, instrument: Any, market: Any) -> float:  # noqa: ANN401
        ...


class UnsupportedInstrumentError(TypeError):
    """Raised when no registered pricer can be resolved for an instrument."""


def _normalize_pricer_id(pricer_id: Optional[str]) -> Optional[str]:
    """
    Normalize pricer_id for stable routing and caching.

    - None stays None (default routing)
    - strings are stripped
    - empty/whitespace-only becomes invalid (raises)
    """
    if pricer_id is None:
        return None
    if not isinstance(pricer_id, str):
        raise TypeError("pricer_id must be a string or None.")
    pid = pricer_id.strip()
    if not pid:
        raise ValueError("pricer_id must be a non-empty string when provided.")
    return pid


@dataclass(slots=True)
class PricerRegistry:
    """
    Registry that resolves instrument -> pricer using MRO-first lookup.

    Supports:
      - default pricer per instrument type (pricer_id=None)
      - optional named pricers per instrument type (pricer_id="...")

    Resolution rules
    ---------------
    1) Exact type match
    2) Walk instrument_type.__mro__ and return the first registered base
    3) Optional fallback: isinstance(...) scan (for ABC virtual subclass cases)
    """

    _by_type: MutableMapping[Type[Any], InstrumentPricer] = field(default_factory=dict)
    _by_type_and_id: MutableMapping[tuple[Type[Any], str], InstrumentPricer] = field(default_factory=dict)
    _cache: MutableMapping[tuple[Type[Any], Optional[str]], InstrumentPricer] = field(default_factory=dict)

    def register(
        self,
        instrument_type: Type[Any],
        pricer: InstrumentPricer,
        *,
        pricer_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(instrument_type, type):
            raise TypeError("instrument_type must be a class/type.")

        pid = _normalize_pricer_id(pricer_id)

        if pid is None:
            if instrument_type in self._by_type and not overwrite:
                raise ValueError(
                    f"Default pricer already registered for {instrument_type.__name__}. "
                    f"Pass overwrite=True to replace."
                )
            self._by_type[instrument_type] = pricer
        else:
            key = (instrument_type, pid)
            if key in self._by_type_and_id and not overwrite:
                raise ValueError(
                    f"Pricer already registered for {instrument_type.__name__} (pricer_id={pid!r}). "
                    f"Pass overwrite=True to replace."
                )
            self._by_type_and_id[key] = pricer

        self._cache.clear()

    def unregister(self, instrument_type: Type[Any], *, pricer_id: Optional[str] = None) -> None:
        pid = _normalize_pricer_id(pricer_id)
        if pid is None:
            self._by_type.pop(instrument_type, None)
        else:
            self._by_type_and_id.pop((instrument_type, pid), None)
        self._cache.clear()

    def get(self, instrument_type: Type[Any], *, pricer_id: Optional[str] = None) -> Optional[InstrumentPricer]:
        pid = _normalize_pricer_id(pricer_id)
        if pid is None:
            return self._by_type.get(instrument_type)
        return self._by_type_and_id.get((instrument_type, pid))

    def as_mapping(self) -> Mapping[Type[Any], InstrumentPricer]:
        # Default mapping only (named pricers intentionally excluded here)
        return dict(self._by_type)

    def resolve(self, instrument: Any, *, pricer_id: Optional[str] = None) -> InstrumentPricer:  # noqa: ANN401
        instrument_type = type(instrument)
        pid = _normalize_pricer_id(pricer_id)
        cache_key = (instrument_type, pid)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if pid is None:
            # ---- default resolution ----
            pricer = self._by_type.get(instrument_type)
            if pricer is not None:
                self._cache[cache_key] = pricer
                return pricer

            for base in instrument_type.__mro__[1:]:
                pricer = self._by_type.get(base)
                if pricer is not None:
                    self._cache[cache_key] = pricer
                    return pricer

            for registered_type, pricer in self._by_type.items():
                if isinstance(instrument, registered_type):
                    self._cache[cache_key] = pricer
                    return pricer

            raise UnsupportedInstrumentError(
                f"No pricer registered for instrument type: {instrument_type.__name__}. "
                f"Registered types: {[t.__name__ for t in self._by_type.keys()]}"
            )

        # ---- named resolution ----
        pricer = self._by_type_and_id.get((instrument_type, pid))
        if pricer is not None:
            self._cache[cache_key] = pricer
            return pricer

        for base in instrument_type.__mro__[1:]:
            pricer = self._by_type_and_id.get((base, pid))
            if pricer is not None:
                self._cache[cache_key] = pricer
                return pricer

        for (registered_type, registered_pid), pricer in self._by_type_and_id.items():
            if registered_pid != pid:
                continue
            if isinstance(instrument, registered_type):
                self._cache[cache_key] = pricer
                return pricer

        raise UnsupportedInstrumentError(
            f"No pricer registered for instrument type: {instrument_type.__name__} with pricer_id={pid!r}. "
            f"Registered named types: {[t.__name__ for (t, p) in self._by_type_and_id.keys() if p == pid]}"
        )


@dataclass(slots=True)
class DefaultPricerRegistry:
    """
    Default registry factory (V1).

    This is intentionally small. V2/V3 just adds registrations.
    """

    def build(self) -> PricerRegistry:
        reg = PricerRegistry()

        # ---- instantiate once (so default + alias share the same config) ----
        spot = FxSpotPricer()
        fwd = FxForwardPricer()

        # Analytic Black Scholes Merton Pricer - European.
        eur_van_bsm = FxEuropeanVanillaBsmPricer()
        eur_dig_bsm = FxEuropeanDigitalBsmPricer()

        # Numerical Monte Carlo Pricer - European.
        eur_van_mc = FxEuropeanVanillaMcPricer()
        eur_dig_mc = FxEuropeanDigitalMcPricer()
        eur_bar_mc = FxEuropeanBarrierMcPricer()

        # Path-dependent Monte Carlo Pricers - European.
        eur_asian_mc = FxEuropeanAsianMcPricer()
        eur_lookback_mc = FxEuropeanLookbackMcPricer()

        # Numerical Finite Difference Pricer - European.
        eur_van_fd = FxEuropeanVanillaFdPricer()
        eur_dig_fd = FxEuropeanDigitalFdPricer()

        # Numerical Finite Difference Pricer - American.
        am_van_fd = FxAmericanVanillaFdPricer()



        # ---- defaults ----
        reg.register(FxSpot, spot)
        reg.register(FxForward, fwd)
        reg.register(EuropeanFxVanillaOption, eur_van_bsm)
        reg.register(EuropeanFxDigitalOption, eur_dig_bsm)
        reg.register(EuropeanFxBarrierOption, eur_bar_mc)
        reg.register(AmericanFxVanillaOption, am_van_fd)
        reg.register(EuropeanFxAsianOption, eur_asian_mc)
        reg.register(EuropeanFxLookbackOption, eur_lookback_mc)

        # ---- named aliases ----
        reg.register(EuropeanFxVanillaOption, eur_van_mc, pricer_id="mc")
        reg.register(EuropeanFxDigitalOption, eur_dig_mc, pricer_id="mc")
        reg.register(EuropeanFxBarrierOption, eur_bar_mc, pricer_id="mc")
        reg.register(EuropeanFxAsianOption, eur_asian_mc, pricer_id="mc")
        reg.register(EuropeanFxLookbackOption, eur_lookback_mc, pricer_id="mc")

        reg.register(EuropeanFxVanillaOption, eur_van_fd, pricer_id="fd")
        reg.register(EuropeanFxDigitalOption, eur_dig_fd, pricer_id="fd")
        reg.register(AmericanFxVanillaOption, am_van_fd, pricer_id="fd")

        return reg