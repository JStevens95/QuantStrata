# tests/unit/pricers/test_portfolio_pricer.py
from __future__ import annotations

from typing import Dict

import pytest

# ---- market-data plumbing used by the pricer ----
from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import MarketRequest, Universe
from src.marketdata.providers.synthetic.provider import SyntheticProvider

# ---- portfolio domain objects ----
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer

# ---- instrument + pricer under test ----
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer

# ---- production registry used for instrument -> pricer routing ----
from src.pricers.registry import PricerRegistry


@pytest.fixture(scope="module")
def fx_market_and_ids():
    """
    Build a synthetic Market snapshot plus the MarketIds required by FX vanilla pricing.

    We return both:
      - `market`: concrete market snapshot with quotes/curves/vol, etc.
      - the MarketId handles used by instruments/pricers to locate market data
    """
    # Spot quote identifier: used to fetch EURUSD spot from Market.quote(...)
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    # Vol surface identifier: used to fetch EURUSD vol surface from Market.vol(...)
    vol_id = MarketId("FX", "VOL", "EURUSD.VOL")

    # Domestic discount curve identifier (USD curve)
    rd_id = MarketId("IR", "CURVE", "USD.OIS")

    # Foreign discount curve identifier (EUR curve)
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    # SyntheticProvider generates deterministic fake market data (seeded RNG)
    provider = SyntheticProvider(seed=123)

    # Build one market snapshot at a given asof date containing the required data
    market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            # Universe tells the provider exactly which market objects to populate
            universe=Universe([spot_id, vol_id, rd_id, rf_id]),
        )
    )

    # Return everything the tests need (market + IDs used by instruments)
    return market, spot_id, vol_id, rd_id, rf_id


def test_portfolio_pv_equals_sum_of_position_pvs(fx_market_and_ids) -> None:
    """
    Sanity check:
    Portfolio PV should equal the sum of each position PV, where
      position PV = quantity * instrument PV.
    """
    # Unpack the shared fixture outputs
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids

    # Create the concrete instrument pricer (BSM European FX vanilla)
    fx_pricer = FxVanillaEuropeanOptionBsmPricer()

    # Use the *production* registry so test matches real routing behaviour
    registry = PricerRegistry()

    # Register our pricer as the default for EuropeanFxVanillaOption instruments
    registry.register(FxVanillaEuropeanOption, fx_pricer)

    # PortfolioPricer uses the registry to resolve pricers per instrument
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    # Pull the actual spot from market so strikes are consistent with current market
    S = float(market.quote(spot_id))

    # Build a few test positions with different strikes/expiries/quantities
    positions = [
        Position(
            position_id="POS_1",  # unique id used for reporting/debugging
            instrument=FxVanillaEuropeanOption(
                option_type="call",
                notional=1_000_000.0,
                strike=S,  # ATM
                expiry=1.0,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=1.0,  # long 1x
        ),
        Position(
            position_id="POS_2",
            instrument=FxVanillaEuropeanOption(
                option_type="put",
                notional=500_000.0,
                strike=0.98 * S,  # slightly ITM/OTM depending on spot definition
                expiry=0.5,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=2.0,  # long 2x
        ),
        Position(
            position_id="POS_3",
            instrument=FxVanillaEuropeanOption(
                option_type="call",
                notional=750_000.0,
                strike=1.02 * S,  # slightly OTM
                expiry=2.0,
                spot_id=spot_id,
                vol_id=vol_id,
                domestic_curve_id=rd_id,
                foreign_curve_id=rf_id,
            ),
            quantity=-0.5,  # short half
        ),
    ]

    # Wrap positions into a Portfolio object (validates unique position IDs)
    portfolio = Portfolio(positions=positions)

    # Price the whole portfolio using the PortfolioPricer (and registry routing)
    result = portfolio_pricer.price(portfolio, market)

    # Compute the expected total PV explicitly: sum(quantity * instrument PV)
    expected_total = 0.0
    for p in positions:
        # Price each instrument directly with the same pricer used in the registry
        instrument_pv = float(fx_pricer.price(p.instrument, market))
        # Scale by quantity to get position PV contribution
        expected_total += float(p.quantity) * instrument_pv

    # Compare registry-based portfolio pricing vs explicit manual sum
    assert result.totals.pv == pytest.approx(expected_total, rel=1e-12, abs=1e-6)


def test_portfolio_greeks_equal_sum_of_position_greeks(fx_market_and_ids) -> None:
    """
    Sanity check:
    Portfolio Greeks should equal the sum of position Greeks, where each position greek is
      quantity * instrument greek.
    """
    # Unpack the shared fixture outputs
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids

    # Concrete pricer under test
    fx_pricer = FxVanillaEuropeanOptionBsmPricer()

    # Production registry (same routing contract as the real system)
    registry = PricerRegistry()
    registry.register(FxVanillaEuropeanOption, fx_pricer)

    # PortfolioPricer aggregates greeks across positions
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    # Use spot to set ATM strikes consistently
    S = float(market.quote(spot_id))

    # Two positions: call + put, both 1x quantity
    positions = [
        Position(
            position_id="POS_A",
            instrument=FxVanillaEuropeanOption(
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
            instrument=FxVanillaEuropeanOption(
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

    # Build the portfolio
    portfolio = Portfolio(positions=positions)

    # Price portfolio and aggregate greeks through PortfolioPricer
    result = portfolio_pricer.price(portfolio, market)

    # Manually compute expected greek totals by summing (quantity * greek)
    expected: Dict[str, float] = {}
    for p in positions:
        # Get instrument greeks from the pricer directly
        g = fx_pricer.greeks(p.instrument, market)
        for k, v in g.items():
            # Accumulate per-greek-key (delta, vega, etc.) across positions
            expected[k] = expected.get(k, 0.0) + float(p.quantity) * float(v)

    # The set of greek keys should match exactly (no missing / extra)
    assert set(result.totals.greeks.keys()) == set(expected.keys())

    # Each greek value should match the manual aggregation within tolerance
    for k in expected.keys():
        assert float(result.totals.greeks[k]) == pytest.approx(float(expected[k]), rel=1e-10, abs=1e-6)