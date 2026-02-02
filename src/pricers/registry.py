from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Protocol, Type

# --- import FX instruments ---
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.options.digital import FxDigitalEuropeanOption
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption, FxVanillaAmericanOption
from src.instruments.fx.options.barrier import FxBarrierEuropeanOption
from src.instruments.fx.options.double_barrier import FxDoubleBarrierEuropeanOption
from src.instruments.fx.options.asian import FxAsianEuropeanOption
from src.instruments.fx.options.lookback import FxLookbackEuropeanOption
from src.instruments.fx.options.touch import FxTouchEuropeanOption
from src.instruments.fx.options.spread import FxSpreadEuropeanOption, FxSpreadEuropeanOptionSimple
from src.instruments.fx.options.forward import FxForwardEuropeanOption, FxForwardEuropeanOptionSimple

# --- import equity instruments ---
from src.instruments.equity.linear.spot import EquitySpot
from src.instruments.equity.linear.forward import EquityForward
from src.instruments.equity.options.vanilla import EquityVanillaEuropeanOption, EquityVanillaAmericanOption
from src.instruments.equity.options.digital import EquityDigitalEuropeanOption
from src.instruments.equity.options.barrier import EquityBarrierEuropeanOption
from src.instruments.equity.options.asian import EquityAsianEuropeanOption
from src.instruments.equity.options.lookback import EquityLookbackEuropeanOption
from src.instruments.equity.options.spread import EquitySpreadEuropeanOption, EquitySpreadEuropeanOptionSimple
from src.instruments.equity.options.futures import EquityFuturesEuropeanOption, EquityFuturesEuropeanOptionSimple

# --- import ir instruments ---
from src.instruments.ir.linear.swap import IrSwap, IrSwapSimple
from src.instruments.ir.linear.fra import IrForwardRateAgreement, IrForwardRateAgreementSimple
from src.instruments.ir.options.swaption import IrSwaptionEuropeanOption, IrSwaptionEuropeanOptionSimple
from src.instruments.ir.options.capfloor import (
    IrCapletEuropeanOption, IrCapletEuropeanOptionSimple, IrCapEuropeanOption, IrCapEuropeanOptionSimple,
    IrFloorletEuropeanOption, IrFloorletEuropeanOptionSimple, IrFloorEuropeanOption, IrFloorEuropeanOptionSimple
)


# --- import FX analytic pricers (linear + non-linear) ---
from src.pricers.fx.spot import FxSpotPricer
from src.pricers.fx.forward import FxForwardPricer
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer, FxDigitalEuropeanOptionBsmPricer
from src.pricers.fx.european_b76 import FxForwardEuropeanOptionB76Pricer, FxForwardEuropeanOptionB76PricerSimple
from src.pricers.fx.european_bch import FxSpreadEuropeanOptionBchPricer, FxSpreadEuropeanOptionBchPricerSimple

# --- import Equity analytic pricers (linear + non-linear) ---
from src.pricers.equity.spot import EquitySpotPricer
from src.pricers.equity.forward import EquityForwardPricer
from src.pricers.equity.european_bsm import EquityVanillaEuropeanOptionBsmPricer, EquityDigitalEuropeanOptionBsmPricer
from src.pricers.equity.european_b76 import (
    EquityFuturesEuropeanOptionB76Pricer, EquityFuturesEuropeanOptionB76PricerSimple
)
from src.pricers.equity.european_bch import (
    EquitySpreadEuropeanOptionBchPricer, EquitySpreadEuropeanOptionBchPricerSimple
)

# --- import ir analytic pricers (linear + non-linear)
from src.pricers.ir.swap import IrSwapPricer, IrSwapPricerSimple
from src.pricers.ir.fra import IrFraPricer, IrFraPricerSimple
from src.pricers.ir.european_b76 import (
    IrCapletEuropeanOptionB76Pricer, IrCapletEuropeanOptionB76PricerSimple,
    IrFloorletEuropeanOptionB76Pricer, IrFloorletEuropeanOptionB76PricerSimple,
    IrCapEuropeanOptionB76Pricer, IrCapEuropeanOptionB76PricerSimple,
    IrFloorEuropeanOptionB76Pricer, IrFloorEuropeanOptionB76PricerSimple,
)
from src.pricers.ir.european_bch import (
    IrSwaptionEuropeanOptionBchPricer, IrSwaptionEuropeanOptionBchPricerSimple,
)

# --- import FX numerical pricers ---
from src.pricers.fx.european_mc import (
    FxVanillaEuropeanOptionMcPricer, FxDigitalEuropeanOptionMcPricer, FxBarrierEuropeanOptionMcPricer,
    FxAsianEuropeanOptionMcPricer, FxLookbackEuropeanOptionMcPricer, FxDoubleBarrierEuropeanOptionMcPricer,
    FxTouchEuropeanOptionMcPricer
    )
from src.pricers.fx.european_fde import (
    FxVanillaEuropeanOptionFdPricer, FxDigitalEuropeanOptionFdPricer
    )
from src.pricers.fx.american_fde import FxVanillaAmericanOptionFdPricer

# --- import Equity numerical pricers ---
from src.pricers.equity.european_mc import (
    EquityVanillaEuropeanOptionMcPricer, EquityAsianEuropeanOptionMcPricer, EquityBarrierEuropeanOptionMcPricer,
    EquityLookbackEuropeanOptionMcPricer
)
from src.pricers.equity.european_fde import (
    EquityVanillaEuropeanOptionFdPricer
)
from src.pricers.equity.american_fde import EquityVanillaAmericanOptionFdPricer


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
        fx_van_eur_bsm = FxVanillaEuropeanOptionBsmPricer()
        fx_dig_eur_bsm = FxDigitalEuropeanOptionBsmPricer()
        fx_fwd_eur_b76 = FxForwardEuropeanOptionB76Pricer()
        fx_fwd_eur_b76_s = FxForwardEuropeanOptionB76PricerSimple()
        fx_sprd_eur_bch = FxSpreadEuropeanOptionBchPricer()
        fx_sprd_eur_bch_s = FxSpreadEuropeanOptionBchPricerSimple()

        # ---- 1.3 non-linear numerical pricers - Monte Carlo. ----
        # non-path dependent instruments.
        fx_van_eur_mc = FxVanillaEuropeanOptionMcPricer()
        fx_dig_eur_mc = FxDigitalEuropeanOptionMcPricer()

        # path dependent instruments.
        fx_asian_eur_mc = FxAsianEuropeanOptionMcPricer()
        fx_lookback_eur_mc = FxLookbackEuropeanOptionMcPricer()
        fx_bar_eur_mc = FxBarrierEuropeanOptionMcPricer()
        fx_double_bar_eur_mc = FxDoubleBarrierEuropeanOptionMcPricer()
        fx_touch_eur_mc = FxTouchEuropeanOptionMcPricer()

        # ---- 1.4 non-linear numerical pricers - Finite Difference. ----
        # non-path dependent instruments.
        fx_van_eur_fd = FxVanillaEuropeanOptionFdPricer()
        fx_dig_eur_fd = FxDigitalEuropeanOptionFdPricer()

        # path dependent instruments.
        fx_van_am_fd = FxVanillaAmericanOptionFdPricer()

        # ==== 2. Equity pricers instantiate once (so default + alias share the same config) ====

        # ---- 2.1. linear pricers ----
        eq_spot = EquitySpotPricer()
        eq_fwd = EquityForwardPricer()

        # ---- 2.2. non-linear analytic pricers ----
        eq_van_eur_bsm = EquityVanillaEuropeanOptionBsmPricer()
        eq_dig_eur_bsm = EquityDigitalEuropeanOptionBsmPricer()
        eq_fut_eur_b76 = EquityFuturesEuropeanOptionB76Pricer()
        eq_fut_eur_b76_s = EquityFuturesEuropeanOptionB76PricerSimple()
        eq_sprd_eur_bch = EquitySpreadEuropeanOptionBchPricer()
        eq_sprd_eur_bch_s = EquitySpreadEuropeanOptionBchPricerSimple()

        # ---- 2.3 non-linear numerical pricers - Monte Carlo. ----
        # non-path dependent instruments.
        eq_van_eur_mc = EquityVanillaEuropeanOptionMcPricer()
        # eq_dig_eur_mc = EquityEuropeanDigitalMcPricer() <--- TODO: need to implement digital mc pricer.

        # path dependent instruments.
        eq_asian_eur_mc = EquityAsianEuropeanOptionMcPricer()
        eq_lookback_eur_mc = EquityLookbackEuropeanOptionMcPricer()
        eq_bar_eur_mc = EquityBarrierEuropeanOptionMcPricer()
        # eq_double_bar_eur_mc = EquityDoubleBarrierEuropeanOptionMcPricer()  <--- TODO: need to implement in mc pricer.
        # eq_touch_eur_mc = EquityTouchEuropeanOptionMcPricer()   <--- TODO: need to implement in mc pricer.

        # ---- 2.4 non-linear numerical pricers - Finite Difference. ----
        # non-path dependent instruments.
        eq_van_eur_fd = EquityVanillaEuropeanOptionFdPricer()
        # eq_dig_eur_fd = EquityDigitalEuropeanOptionFdPricer()   <--- TODO: need to implement in fd pricer.

        # path dependent instruments.
        eq_van_am_fd = EquityVanillaAmericanOptionFdPricer()


        # ==== 3. Interest Rate pricers instantiate once (so default + alias share the same config) ====

        # ---- 3.1. linear pricers ----
        ir_fra = IrFraPricer()
        ir_fra_s = IrFraPricerSimple()
        ir_swp = IrSwapPricer()
        ir_swp_s = IrSwapPricerSimple()

        # ---- 3.2. non-linear analytic pricers (Black76) ----
        ir_caplet_b76 = IrCapletEuropeanOptionB76Pricer()
        ir_caplet_b76_s = IrCapletEuropeanOptionB76PricerSimple()
        ir_floorlet_b76 = IrFloorletEuropeanOptionB76Pricer()
        ir_floorlet_b76_s = IrFloorletEuropeanOptionB76PricerSimple()
        ir_cap_b76 = IrCapEuropeanOptionB76Pricer()
        ir_cap_b76_s = IrCapEuropeanOptionB76PricerSimple()
        ir_floor_b76 = IrFloorEuropeanOptionB76Pricer()
        ir_floor_b76_s = IrFloorEuropeanOptionB76PricerSimple()

        # ---- 3.3. non-linear analytic pricers (Bachelier) ----
        ir_swaption_bch = IrSwaptionEuropeanOptionBchPricer()
        ir_swaption_bch_s = IrSwaptionEuropeanOptionBchPricerSimple()

        # ============================================================================================================ #
        # ============================================================================================================ #
        # ============================================================================================================ #

        # ---- Register FX defaults ----
        reg.register(FxSpot, fx_spot)
        reg.register(FxForward, fx_fwd)
        reg.register(FxVanillaEuropeanOption, fx_van_eur_bsm)
        reg.register(FxDigitalEuropeanOption, fx_dig_eur_bsm)
        reg.register(FxForwardEuropeanOption, fx_fwd_eur_b76)
        reg.register(FxForwardEuropeanOptionSimple, fx_fwd_eur_b76_s)
        reg.register(FxSpreadEuropeanOption, fx_sprd_eur_bch)
        reg.register(FxSpreadEuropeanOptionSimple, fx_sprd_eur_bch_s)
        reg.register(FxBarrierEuropeanOption, fx_bar_eur_mc)
        reg.register(FxVanillaAmericanOption, fx_van_am_fd)
        reg.register(FxAsianEuropeanOption, fx_asian_eur_mc)
        reg.register(FxLookbackEuropeanOption, fx_lookback_eur_mc)
        reg.register(FxDoubleBarrierEuropeanOption, fx_double_bar_eur_mc)
        reg.register(FxTouchEuropeanOption, fx_touch_eur_mc)

        # ---- Register Equity defaults ----
        reg.register(EquitySpot, eq_spot)
        reg.register(EquityForward, eq_fwd)
        reg.register(EquityVanillaEuropeanOption, eq_van_eur_bsm)
        reg.register(EquityDigitalEuropeanOption, eq_dig_eur_bsm)
        reg.register(EquityFuturesEuropeanOption, eq_fut_eur_b76)
        reg.register(EquityFuturesEuropeanOptionSimple, eq_fut_eur_b76_s)
        reg.register(EquitySpreadEuropeanOption, eq_sprd_eur_bch)
        reg.register(EquitySpreadEuropeanOptionSimple, eq_sprd_eur_bch_s)
        reg.register(EquityBarrierEuropeanOption, eq_bar_eur_mc)
        reg.register(EquityVanillaAmericanOption, eq_van_am_fd)
        reg.register(EquityAsianEuropeanOption, eq_asian_eur_mc)
        reg.register(EquityLookbackEuropeanOption, eq_lookback_eur_mc)

        # ---- Register IR defaults ----
        # Linear pricers
        reg.register(IrForwardRateAgreement, ir_fra)
        reg.register(IrForwardRateAgreementSimple, ir_fra_s)
        reg.register(IrSwap, ir_swp)
        reg.register(IrSwapSimple, ir_swp_s)

        # Black76 pricers (caps/floors)
        reg.register(IrCapletEuropeanOption, ir_caplet_b76)
        reg.register(IrCapletEuropeanOptionSimple, ir_caplet_b76_s)
        reg.register(IrFloorletEuropeanOption, ir_floorlet_b76)
        reg.register(IrFloorletEuropeanOptionSimple, ir_floorlet_b76_s)
        reg.register(IrCapEuropeanOption, ir_cap_b76)
        reg.register(IrCapEuropeanOptionSimple, ir_cap_b76_s)
        reg.register(IrFloorEuropeanOption, ir_floor_b76)
        reg.register(IrFloorEuropeanOptionSimple, ir_floor_b76_s)

        # Bachelier pricers (swaptions)
        reg.register(IrSwaptionEuropeanOption, ir_swaption_bch)
        reg.register(IrSwaptionEuropeanOptionSimple, ir_swaption_bch_s)

        # ============================================================================================================ #
        # ============================================================================================================ #
        # ============================================================================================================ #

        # ---- named aliases - FX ----
        reg.register(FxVanillaEuropeanOption, fx_van_eur_mc, pricer_id="mc")
        reg.register(FxDigitalEuropeanOption, fx_dig_eur_mc, pricer_id="mc")
        reg.register(FxBarrierEuropeanOption, fx_bar_eur_mc, pricer_id="mc")
        reg.register(FxAsianEuropeanOption, fx_asian_eur_mc, pricer_id="mc")
        reg.register(FxLookbackEuropeanOption, fx_lookback_eur_mc, pricer_id="mc")
        reg.register(FxDoubleBarrierEuropeanOption, fx_double_bar_eur_mc, pricer_id="mc")
        reg.register(FxTouchEuropeanOption, fx_touch_eur_mc, pricer_id="mc")

        reg.register(FxVanillaEuropeanOption, fx_van_eur_fd, pricer_id="fd")
        reg.register(FxDigitalEuropeanOption, fx_dig_eur_fd, pricer_id="fd")
        reg.register(FxVanillaAmericanOption, fx_van_am_fd, pricer_id="fd")

        # ---- named aliases - Equity ----
        reg.register(EquityVanillaEuropeanOption, eq_van_eur_mc, pricer_id="mc")
        # reg.register(EquityDigitalEuropeanOption, eq_dig_eur_mc, pricer_id="mc") <--- TODO: need to implement in mc pricer.
        reg.register(EquityBarrierEuropeanOption, eq_bar_eur_mc, pricer_id="mc")
        reg.register(EquityAsianEuropeanOption, eq_asian_eur_mc, pricer_id="mc")
        reg.register(EquityLookbackEuropeanOption, eq_lookback_eur_mc, pricer_id="mc")

        reg.register(EquityVanillaEuropeanOption, eq_van_eur_fd, pricer_id="fd")
        # reg.register(EquityDigitalEuropeanOption, eq_dig_eur_fd, pricer_id="fd") <--- TODO: need to implement in fd pricer.
        reg.register(EquityVanillaAmericanOption, eq_van_am_fd, pricer_id="fd")

        return reg