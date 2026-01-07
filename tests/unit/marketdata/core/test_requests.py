# tests/unit/marketdata/core/test_requests.py

import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import Universe, MarketRequest, TimeseriesRequest


def test_universe_empty_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Universe(ids=[])


def test_universe_duplicate_keys_raises() -> None:
    a = MarketId("FX", "SPOT", "EURUSD")
    b = MarketId("fx", "spot", "EURUSD")  # same key after normalization

    with pytest.raises(ValueError, match="duplicate"):
        Universe(ids=[a, b])


def test_market_request_normalizes_asof() -> None:
    u = Universe([MarketId("FX", "SPOT", "EURUSD")])
    req = MarketRequest(asof="2026-01-07", universe=u, scenario=None)
    assert req.asof == "2026-01-07"


def test_market_request_negative_scenario_raises() -> None:
    u = Universe([MarketId("FX", "SPOT", "EURUSD")])
    with pytest.raises(ValueError, match="must be >= 0"):
        MarketRequest(asof="2026-01-07", universe=u, scenario=-1)


def test_timeseries_request_date_order_guard() -> None:
    u = Universe([MarketId("FX", "SPOT", "EURUSD")])
    with pytest.raises(ValueError, match="end must be >= start"):
        TimeseriesRequest(start="2026-01-10", end="2026-01-07", freq="D", universe=u)


def test_timeseries_request_freq_empty_raises() -> None:
    u = Universe([MarketId("FX", "SPOT", "EURUSD")])
    with pytest.raises(ValueError, match="freq must not be empty"):
        TimeseriesRequest(start="2026-01-01", end="2026-01-07", freq="  ", universe=u)


def test_timeseries_request_scenarios_must_be_ge_1() -> None:
    u = Universe([MarketId("FX", "SPOT", "EURUSD")])
    with pytest.raises(ValueError, match="must be >= 1"):
        TimeseriesRequest(start="2026-01-01", end="2026-01-07", freq="D", universe=u, scenarios=0)