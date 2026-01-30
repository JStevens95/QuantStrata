from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Protocol, Type

# --- import FX instruments ---
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.options.digital import EuropeanFxDigitalOption
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption, AmericanFxVanillaOption
from src.instruments.fx.options.barrier import EuropeanFxBarrierOption
from src.instruments.fx.options.double_barrier import EuropeanFxDoubleBarrierOption
from src.instruments.fx.options.asian import EuropeanFxAsianOption
from src.instruments.fx.options.lookback import EuropeanFxLookbackOption
from src.instruments.fx.options.touch import EuropeanFxTouchOption

# --- import equity instruments ---
from src.instruments.equity.linear.spot import EquitySpot
from src.instruments.equity.linear.forward import EquityForward
from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption, AmericanEquityVanillaOption
from src.instruments.equity.options.digital import EuropeanEquityDigitalOption
from src.instruments.equity.options.barrier import EuropeanEquityBarrierOption
from src.instruments.equity.options.asian import EuropeanEquityAsianOption
from src.instruments.equity.options.lookback import EuropeanEquityLookbackOption


# --- import FX analytic pricers (linear + non-linear) ---
from src.pricers.fx.spot import FxSpotPricer
from src.pricers.fx.forward import FxForwardPricer
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer, FxEuropeanDigitalBsmPricer

# --- import Equity analytic pricers (linear + non-linear) ---
from src.pricers.equity.spot import EquitySpotPricer
from src.pricers.equity.forward import EquityForwardPricer
from src.pricers.equity.european_bsm import EquityEuropeanVanillaBsmPricer, EquityEuropeanDigitalBsmPricer

# --- import FX numerical pricers ---
from src.pricers.fx.european_mc import (
    FxEuropeanBarrierMcPricer, FxEuropeanVanillaMcPricer, FxEuropeanDigitalMcPricer, FxEuropeanAsianMcPricer,
    FxEuropeanLookbackMcPricer, FxEuropeanDoubleBarrierMcPricer, FxEuropeanTouchMcPricer
    )
from src.pricers.fx.european_fde import (
    FxEuropeanVanillaFdPricer, FxEuropeanDigitalFdPricer
    )
from src.pricers.fx.american_fde import FxAmericanVanillaFdPricer

# --- import Equity numerical pricers ---
from src.pricers.equity.european_mc import (
    EquityEuropeanVanillaMcPricer, EquityEuropeanAsianMcPricer, EquityEuropeanBarrierMcPricer,
    EquityEuropeanLookbackMcPricer
)
from src.pricers.equity.european_fde import (
    EquityEuropeanVanillaFdPricer
)
from src.pricers.equity.american_fde import EquityAmericanVanillaFdPricer


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

        # ==== 1. FX pricers instantiate once (so default + alias share the same config) ====

        # ---- 1.1. linear pricers ----
        fx_spot = FxSpotPricer()
        fx_fwd = FxForwardPricer()

        # ---- 1.2. non-linear analytic pricers ----
        fx_eur_van_bsm = FxEuropeanVanillaBsmPricer()
        fx_eur_dig_bsm = FxEuropeanDigitalBsmPricer()

        # ---- 1.3 non-linear numerical pricers - Monte Carlo. ----
        # non-path dependent instruments.
        fx_eur_van_mc = FxEuropeanVanillaMcPricer()
        fx_eur_dig_mc = FxEuropeanDigitalMcPricer()

        # path dependent instruments.
        fx_eur_asian_mc = FxEuropeanAsianMcPricer()
        fx_eur_lookback_mc = FxEuropeanLookbackMcPricer()
        fx_eur_bar_mc = FxEuropeanBarrierMcPricer()
        fx_eur_double_barrier_mc = FxEuropeanDoubleBarrierMcPricer()
        fx_eur_touch_mc = FxEuropeanTouchMcPricer()

        # ---- 1.4 non-linear numerical pricers - Finite Difference. ----
        # non-path dependent instruments.
        fx_eur_van_fd = FxEuropeanVanillaFdPricer()
        fx_eur_dig_fd = FxEuropeanDigitalFdPricer()

        # path dependent instruments.
        fx_am_van_fd = FxAmericanVanillaFdPricer()

        # ==== 2. Equity pricers instantiate once (so default + alias share the same config) ====

        # ---- 2.1. linear pricers ----
        eq_spot = EquitySpotPricer()
        eq_fwd = EquityForwardPricer()

        # ---- 2.2. non-linear analytic pricers ----
        eq_eur_van_bsm = EquityEuropeanVanillaBsmPricer()
        eq_eur_dig_bsm = EquityEuropeanDigitalBsmPricer()

        # ---- 2.3 non-linear numerical pricers - Monte Carlo. ----
        # non-path dependent instruments.
        eq_eur_van_mc = EquityEuropeanVanillaMcPricer()
        # eq_eur_dig_mc = EquityEuropeanDigitalMcPricer() <--- TODO: need to implement digital mc pricer.

        # path dependent instruments.
        eq_eur_asian_mc = EquityEuropeanAsianMcPricer()
        eq_eur_lookback_mc = EquityEuropeanLookbackMcPricer()
        eq_eur_bar_mc = EquityEuropeanBarrierMcPricer()
        # eq_eur_double_barrier_mc = EquityEuropeanDoubleBarrierMcPricer()  <--- TODO: need to implement in mc pricer.
        # eq_eur_touch_mc = EquityEuropeanTouchMcPricer()   <--- TODO: need to implement in mc pricer.

        # ---- 2.4 non-linear numerical pricers - Finite Difference. ----
        # non-path dependent instruments.
        eq_eur_van_fd = EquityEuropeanVanillaFdPricer()
        # eq_eur_dig_fd = EquityEuropeanDigitalFdPricer()   <--- TODO: need to implement in fd pricer.

        # path dependent instruments.
        eq_am_van_fd = EquityAmericanVanillaFdPricer()

        # ============================================================================================================ #
        # ============================================================================================================ #
        # ============================================================================================================ #

        # ---- Register FX defaults ----
        reg.register(FxSpot, fx_spot)
        reg.register(FxForward, fx_fwd)
        reg.register(EuropeanFxVanillaOption, fx_eur_van_bsm)
        reg.register(EuropeanFxDigitalOption, fx_eur_dig_bsm)
        reg.register(EuropeanFxBarrierOption, fx_eur_bar_mc)
        reg.register(AmericanFxVanillaOption, fx_am_van_fd)
        reg.register(EuropeanFxAsianOption, fx_eur_asian_mc)
        reg.register(EuropeanFxLookbackOption, fx_eur_lookback_mc)
        reg.register(EuropeanFxDoubleBarrierOption, fx_eur_double_barrier_mc)
        reg.register(EuropeanFxTouchOption, fx_eur_touch_mc)

        # ---- Register Equity defaults ----
        reg.register(EquitySpot, eq_spot)
        reg.register(EquityForward, eq_fwd)
        reg.register(EuropeanEquityVanillaOption, eq_eur_van_bsm)
        reg.register(EuropeanEquityDigitalOption, eq_eur_dig_bsm)
        reg.register(EuropeanEquityBarrierOption, eq_eur_bar_mc)
        reg.register(AmericanEquityVanillaOption, eq_am_van_fd)
        reg.register(EuropeanEquityAsianOption, eq_eur_asian_mc)
        reg.register(EuropeanEquityLookbackOption, eq_eur_lookback_mc)

        # ============================================================================================================ #
        # ============================================================================================================ #
        # ============================================================================================================ #

        # ---- named aliases - FX ----
        reg.register(EuropeanFxVanillaOption, fx_eur_van_mc, pricer_id="mc")
        reg.register(EuropeanFxDigitalOption, fx_eur_dig_mc, pricer_id="mc")
        reg.register(EuropeanFxBarrierOption, fx_eur_bar_mc, pricer_id="mc")
        reg.register(EuropeanFxAsianOption, fx_eur_asian_mc, pricer_id="mc")
        reg.register(EuropeanFxLookbackOption, fx_eur_lookback_mc, pricer_id="mc")
        reg.register(EuropeanFxDoubleBarrierOption, fx_eur_double_barrier_mc, pricer_id="mc")
        reg.register(EuropeanFxTouchOption, fx_eur_touch_mc, pricer_id="mc")

        reg.register(EuropeanFxVanillaOption, fx_eur_van_fd, pricer_id="fd")
        reg.register(EuropeanFxDigitalOption, fx_eur_dig_fd, pricer_id="fd")
        reg.register(AmericanFxVanillaOption, fx_am_van_fd, pricer_id="fd")

        # ---- named aliases - Equity ----
        reg.register(EuropeanEquityVanillaOption, eq_eur_van_mc, pricer_id="mc")
        # reg.register(EuropeanEquityDigitalOption, fx_eur_dig_mc, pricer_id="mc") <--- TODO: need to implement in mc pricer.
        reg.register(EuropeanEquityBarrierOption, eq_eur_bar_mc, pricer_id="mc")
        reg.register(EuropeanEquityAsianOption, eq_eur_asian_mc, pricer_id="mc")
        reg.register(EuropeanEquityLookbackOption, eq_eur_lookback_mc, pricer_id="mc")

        reg.register(EuropeanEquityVanillaOption, eq_eur_van_fd, pricer_id="fd")
        # reg.register(EuropeanEquityDigitalOption, fx_eur_dig_fd, pricer_id="fd") <--- TODO: need to implement in fd pricer.
        reg.register(AmericanEquityVanillaOption, eq_am_van_fd, pricer_id="fd")

        return reg