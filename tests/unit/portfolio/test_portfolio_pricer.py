# tests/unit/pricers/test_portfolio_pricer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

import pytest

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.pricers.fx.european import FxEuropeanVanillaBsmPricer


@dataclass(frozen=True, slots=True)
class _Registry:
    """
    Minimal pricer registry compatible with PortfolioPricer (Option B),
    i.e. it provides `.resolve(instrument)`.

    Resolution:
      1) exact type match
      2) MRO-closest base-class match
    """
    mapping: Mapping[Type[Any], Any]

    def resolve(self, instrument: Any) -> Any:
        instrument_type = type(instrument)

        # 1) exact match
        exact = self.mapping.get(instrument_type)
        if exact is not None:
            return exact

        # 2) closest base class by MRO index
        best: Optional[Tuple[int, Type[Any], Any]] = None
        mro = instrument_type.mro()

        for registered_type, pricer in self.mapping.items():
            if not isinstance(registered_type, type):
                continue
            if isinstance(instrument, registered_type):
                try:
                    distance = mro.index(registered_type)
                except ValueError:
                    distance = 10**9
                cand = (distance, registered_type, pricer)
                if best is None or cand[0] < best[0]:
                    best = cand

        if best is not None:
            return best[2]

        raise TypeError(f"No pricer registered for instrument type: {instrument_type.__name__}")


@pytest.fixture(scope="module")
def fx_market_and_ids():
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD.VOL")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([spot_id, vol_id, rd_id, rf_id]),
        )
    )
    return market, spot_id, vol_id, rd_id, rf_id


def test_portfolio_pv_equals_sum_of_position_pvs(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids

    fx_pricer = FxEuropeanVanillaBsmPricer()
    registry = _Registry(mapping={EuropeanFxVanillaOption: fx_pricer})
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    S = float(market.quote(spot_id))

    positions = [
        Position(
            position_id="POS_1",
            instrument=EuropeanFxVanillaOption(
                option_type="call",
                notional=1_000_000.0,
                strike=S,
                expiry=1.0,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=1.0,
        ),
        Position(
            position_id="POS_2",
            instrument=EuropeanFxVanillaOption(
                option_type="put",
                notional=500_000.0,
                strike=0.98 * S,
                expiry=0.5,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=2.0,
        ),
        Position(
            position_id="POS_3",
            instrument=EuropeanFxVanillaOption(
                option_type="call",
                notional=750_000.0,
                strike=1.02 * S,
                expiry=2.0,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=-0.5,
        ),
    ]

    portfolio = Portfolio(positions=positions)

    result = portfolio_pricer.price(portfolio, market)

    expected_total = 0.0
    for p in positions:
        expected_total += float(p.quantity) * float(fx_pricer.price(p.instrument, market))

    assert result.totals.pv == pytest.approx(expected_total, rel=1e-12, abs=1e-6)


def test_portfolio_greeks_equal_sum_of_position_greeks(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids

    fx_pricer = FxEuropeanVanillaBsmPricer()
    registry = _Registry(mapping={EuropeanFxVanillaOption: fx_pricer})
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    S = float(market.quote(spot_id))

    positions = [
        Position(
            position_id="POS_A",
            instrument=EuropeanFxVanillaOption(
                option_type="call",
                notional=1_000_000.0,
                strike=S,
                expiry=1.0,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=1.0,
        ),
        Position(
            position_id="POS_B",
            instrument=EuropeanFxVanillaOption(
                option_type="put",
                notional=1_000_000.0,
                strike=S,
                expiry=1.0,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=1.0,
        ),
    ]

    portfolio = Portfolio(positions=positions)
    result = portfolio_pricer.price(portfolio, market)

    expected: Dict[str, float] = {}
    for p in positions:
        g = fx_pricer.greeks(p.instrument, market)
        for k, v in g.items():
            expected[k] = expected.get(k, 0.0) + float(p.quantity) * float(v)

    assert set(result.totals.greeks.keys()) == set(expected.keys())
    for k in expected.keys():
        assert float(result.totals.greeks[k]) == pytest.approx(float(expected[k]), rel=1e-10, abs=1e-6)