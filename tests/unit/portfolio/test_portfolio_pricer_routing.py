from __future__ import annotations

import pytest

from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry, UnsupportedInstrumentError


# ---- minimal fake instruments ----
class InstA: ...
class InstB: ...


# ---- minimal fake pricers ----
class PricerA_Default:
    def price(self, instrument, market) -> float:
        return 10.0


class PricerA_MC:
    def price(self, instrument, market) -> float:
        return 12.5


class PricerB_Default:
    def price(self, instrument, market) -> float:
        return 3.0


def test_portfolio_pricer_uses_default_pricers():
    reg = PricerRegistry()
    reg.register(InstA, PricerA_Default())
    reg.register(InstB, PricerB_Default())

    pricer = PortfolioPricer(pricer_registry=reg)

    pf = Portfolio(
        positions=[
            Position(position_id="p1", instrument=InstA(), quantity=2.0),
            Position(position_id="p2", instrument=InstB(), quantity=-1.0),
        ]
    )

    res = pricer.price(pf, market={})
    assert res.totals.pv == pytest.approx(2.0 * 10.0 + (-1.0) * 3.0)


def test_position_pricer_id_overrides_default():
    reg = PricerRegistry()
    reg.register(InstA, PricerA_Default())
    reg.register(InstA, PricerA_MC(), pricer_id="mc")

    pricer = PortfolioPricer(pricer_registry=reg)

    pf = Portfolio(
        positions=[
            Position(position_id="p1", instrument=InstA(), quantity=1.0, pricer_id="mc"),
        ]
    )

    res = pricer.price(pf, market={})
    assert res.totals.pv == pytest.approx(12.5)


def test_global_pricer_id_applies_when_position_pricer_id_missing():
    reg = PricerRegistry()
    reg.register(InstA, PricerA_Default())
    reg.register(InstA, PricerA_MC(), pricer_id="mc")

    pricer = PortfolioPricer(pricer_registry=reg)

    pf = Portfolio(
        positions=[
            Position(position_id="p1", instrument=InstA(), quantity=2.0),
        ]
    )

    res = pricer.price(pf, market={}, pricer_id="mc")
    assert res.totals.pv == pytest.approx(2.0 * 12.5)


def test_position_pricer_id_beats_global_pricer_id():
    reg = PricerRegistry()
    reg.register(InstA, PricerA_Default())
    reg.register(InstA, PricerA_MC(), pricer_id="mc")

    pricer = PortfolioPricer(pricer_registry=reg)

    pf = Portfolio(
        positions=[
            Position(position_id="p1", instrument=InstA(), quantity=1.0, pricer_id=None),
            Position(position_id="p2", instrument=InstA(), quantity=1.0, pricer_id="mc"),
        ]
    )

    res = pricer.price(pf, market={}, pricer_id=None)
    assert res.per_position[0].pv == pytest.approx(10.0)
    assert res.per_position[1].pv == pytest.approx(12.5)


def test_unknown_named_pricer_raises():
    reg = PricerRegistry()
    reg.register(InstA, PricerA_Default())

    pricer = PortfolioPricer(pricer_registry=reg)

    pf = Portfolio(positions=[Position(position_id="p1", instrument=InstA(), quantity=1.0, pricer_id="mc")])

    with pytest.raises(UnsupportedInstrumentError):
        pricer.price(pf, market={})