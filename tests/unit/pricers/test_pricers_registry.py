from __future__ import annotations

import pytest

from src.pricers.registry import DefaultPricerRegistry, UnsupportedInstrumentError
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.instruments.fx.options.digital import EuropeanFxDigitalOption
from src.instruments.fx.options.barrier import EuropeanFxBarrierOption
from src.instruments.fx.options.vanilla import AmericanFxVanillaOption


def test_registry_resolves_default_pricers() -> None:
    reg = DefaultPricerRegistry().build()

    # Bare instances are enough for routing tests
    assert reg.resolve(EuropeanFxVanillaOption.__new__(EuropeanFxVanillaOption)) is not None
    assert reg.resolve(EuropeanFxDigitalOption.__new__(EuropeanFxDigitalOption)) is not None
    assert reg.resolve(EuropeanFxBarrierOption.__new__(EuropeanFxBarrierOption)) is not None
    assert reg.resolve(AmericanFxVanillaOption.__new__(AmericanFxVanillaOption)) is not None


def test_registry_named_pricers_exist_for_vanilla_and_digital() -> None:
    reg = DefaultPricerRegistry().build()

    reg.resolve(EuropeanFxVanillaOption.__new__(EuropeanFxVanillaOption), pricer_id="mc")
    reg.resolve(EuropeanFxDigitalOption.__new__(EuropeanFxDigitalOption), pricer_id="mc")
    reg.resolve(EuropeanFxVanillaOption.__new__(EuropeanFxVanillaOption), pricer_id="fd")
    reg.resolve(EuropeanFxDigitalOption.__new__(EuropeanFxDigitalOption), pricer_id="fd")


def test_registry_global_override_mixed_book_expected_behaviour() -> None:
    """
    This test will FAIL until you add aliases:
      - Barrier: pricer_id="mc"
      - American: pricer_id="fd"
    """
    reg = DefaultPricerRegistry().build()

    # Barrier should be routable under "mc" if you want portfolio-wide mc override.
    reg.resolve(EuropeanFxBarrierOption.__new__(EuropeanFxBarrierOption), pricer_id="mc")

    # American should be routable under "fd" if you want portfolio-wide fd override.
    reg.resolve(AmericanFxVanillaOption.__new__(AmericanFxVanillaOption), pricer_id="fd")


def test_registry_unknown_named_pricer_raises() -> None:
    reg = DefaultPricerRegistry().build()
    with pytest.raises(UnsupportedInstrumentError):
        reg.resolve(EuropeanFxVanillaOption.__new__(EuropeanFxVanillaOption), pricer_id="does_not_exist")