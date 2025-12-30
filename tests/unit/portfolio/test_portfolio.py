# tests/unit/portfolio/test_portfolio.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import PricerRegistry


# -------------------------
# Dummy instruments & market
# -------------------------
@dataclass(frozen=True, slots=True)
class _BaseInstrument:
    pass


@dataclass(frozen=True, slots=True)
class _DerivedInstrument(_BaseInstrument):
    pass


@dataclass(frozen=True, slots=True)
class _OtherInstrument:
    pass


@dataclass(frozen=True, slots=True)
class _DummyMarket:
    tag: str = "mkt"


# -------------------------
# Dummy pricers
# -------------------------
@dataclass(frozen=True, slots=True)
class _PvAndGreeksPricer:
    pv: float
    greeks_map: Dict[str, float]

    def price(self, instrument: Any, market: Any) -> float:  # noqa: ANN401
        return float(self.pv)

    def greeks(self, instrument: Any, market: Any) -> Dict[str, float]:  # noqa: ANN401
        return dict(self.greeks_map)


@dataclass(frozen=True, slots=True)
class _PriceOnlyPricer:
    pv: float

    def price(self, instrument: Any, market: Any) -> float:  # noqa: ANN401
        return float(self.pv)


@dataclass(frozen=True, slots=True)
class _BadGreeksPricer:
    def price(self, instrument: Any, market: Any) -> float:  # noqa: ANN401
        return 1.0

    def greeks(self, instrument: Any, market: Any) -> Dict[str, float]:  # noqa: ANN401
        return {"delta": float("nan")}


def test_portfolio_pricer_scales_pv_and_greeks_and_aggregates() -> None:
    registry = PricerRegistry()
    registry.register(_BaseInstrument, _PvAndGreeksPricer(pv=10.0, greeks_map={"delta": 2.0, "gamma": 3.0}))

    pricer = PortfolioPricer(pricer_registry=registry)

    portfolio = Portfolio(
        positions=[
            Position(position_id="pos1", instrument=_BaseInstrument(), quantity=2.0),
            Position(position_id="pos2", instrument=_BaseInstrument(), quantity=-1.0),
        ]
    )

    result = pricer.price(portfolio, _DummyMarket())

    assert result.totals.pv == pytest.approx(10.0)
    assert result.totals.greeks["delta"] == pytest.approx(2.0)
    assert result.totals.greeks["gamma"] == pytest.approx(3.0)

    by_id = {r.position_id: r for r in result.per_position}
    assert by_id["pos1"].pv == pytest.approx(20.0)
    assert by_id["pos2"].pv == pytest.approx(-10.0)
    assert by_id["pos1"].greeks["delta"] == pytest.approx(4.0)
    assert by_id["pos2"].greeks["delta"] == pytest.approx(-2.0)


def test_portfolio_pricer_resolves_mro_base_class_pricer() -> None:
    registry = PricerRegistry()
    registry.register(_BaseInstrument, _PvAndGreeksPricer(pv=7.0, greeks_map={"delta": 1.0}))

    pricer = PortfolioPricer(pricer_registry=registry)

    portfolio = Portfolio(positions=[Position(position_id="p1", instrument=_DerivedInstrument(), quantity=1.0)])
    result = pricer.price(portfolio, _DummyMarket())

    assert result.totals.pv == pytest.approx(7.0)
    assert result.totals.greeks["delta"] == pytest.approx(1.0)


def test_portfolio_pricer_supports_price_only_pricers() -> None:
    registry = PricerRegistry()
    registry.register(_OtherInstrument, _PriceOnlyPricer(pv=5.0))

    pricer = PortfolioPricer(pricer_registry=registry)

    portfolio = Portfolio(positions=[Position(position_id="p1", instrument=_OtherInstrument(), quantity=3.0)])
    result = pricer.price(portfolio, _DummyMarket())

    assert result.totals.pv == pytest.approx(15.0)
    assert result.totals.greeks == {}


def test_portfolio_pricer_raises_for_unsupported_instrument() -> None:
    registry = PricerRegistry()
    registry.register(_BaseInstrument, _PriceOnlyPricer(pv=1.0))

    pricer = PortfolioPricer(pricer_registry=registry)

    portfolio = Portfolio(positions=[Position(position_id="p1", instrument=_OtherInstrument(), quantity=1.0)])

    # Registry raises a TypeError-derived exception; PortfolioPricer will propagate it.
    with pytest.raises(TypeError, match="No pricer registered for instrument type"):
        pricer.price(portfolio, _DummyMarket())


def test_portfolio_pricer_raises_on_non_finite_greek() -> None:
    registry = PricerRegistry()
    registry.register(_BaseInstrument, _BadGreeksPricer())

    pricer = PortfolioPricer(pricer_registry=registry)
    portfolio = Portfolio(positions=[Position(position_id="p1", instrument=_BaseInstrument(), quantity=1.0)])

    with pytest.raises(ValueError, match="Non-finite greek value"):
        pricer.price(portfolio, _DummyMarket())