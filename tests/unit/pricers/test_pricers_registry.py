from __future__ import annotations

import pytest

from src.pricers.registry import DefaultPricerRegistry, UnsupportedInstrumentError
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.instruments.fx.options.digital import FxDigitalEuropeanOption
from src.instruments.fx.options.barrier import FxBarrierEuropeanOption
from src.instruments.fx.options.vanilla import FxVanillaAmericanOption


def test_registry_resolves_default_pricers() -> None:
    reg = DefaultPricerRegistry().build()

    # Bare instances are enough for routing tests
    assert reg.resolve(FxVanillaEuropeanOption.__new__(FxVanillaEuropeanOption)) is not None
    assert reg.resolve(FxDigitalEuropeanOption.__new__(FxDigitalEuropeanOption)) is not None
    assert reg.resolve(FxBarrierEuropeanOption.__new__(FxBarrierEuropeanOption)) is not None
    assert reg.resolve(FxVanillaAmericanOption.__new__(FxVanillaAmericanOption)) is not None


def test_registry_named_pricers_exist_for_vanilla_and_digital() -> None:
    reg = DefaultPricerRegistry().build()

    reg.resolve(FxVanillaEuropeanOption.__new__(FxVanillaEuropeanOption), pricer_id="mc")
    reg.resolve(FxDigitalEuropeanOption.__new__(FxDigitalEuropeanOption), pricer_id="mc")
    reg.resolve(FxVanillaEuropeanOption.__new__(FxVanillaEuropeanOption), pricer_id="fd")
    reg.resolve(FxDigitalEuropeanOption.__new__(FxDigitalEuropeanOption), pricer_id="fd")


def test_registry_global_override_mixed_book_expected_behaviour() -> None:
    """
    This test will FAIL until you add aliases:
      - Barrier: pricer_id="mc"
      - American: pricer_id="fd"
    """
    reg = DefaultPricerRegistry().build()

    # Barrier should be routable under "mc" if you want portfolio-wide mc override.
    reg.resolve(FxBarrierEuropeanOption.__new__(FxBarrierEuropeanOption), pricer_id="mc")

    # American should be routable under "fd" if you want portfolio-wide fd override.
    reg.resolve(FxVanillaAmericanOption.__new__(FxVanillaAmericanOption), pricer_id="fd")


def test_registry_unknown_named_pricer_raises() -> None:
    reg = DefaultPricerRegistry().build()
    with pytest.raises(UnsupportedInstrumentError):
        reg.resolve(FxVanillaEuropeanOption.__new__(FxVanillaEuropeanOption), pricer_id="does_not_exist")