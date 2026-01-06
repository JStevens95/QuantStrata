from __future__ import annotations

import pytest

from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.scenarios.base import ScenarioPack
from src.marketdata.scenarios.shocks import SpotShock
from src.marketdata.providers.synthetic.provider import SyntheticProvider
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry
from src.risk.validation.greeks_vs_scenarios import GreeksVsScenarioConfig, validate_greeks_vs_scenarios


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    """Deterministic provider for stable tests."""
    return SyntheticProvider(seed=123)


@pytest.fixture(scope="module")
def portfolio_pricer() -> PortfolioPricer:
    """Default portfolio pricer with the default registry."""
    return PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())


# =============================================================================
# Helpers (keep tests short)
# =============================================================================

def _market(provider: SyntheticProvider, ids: list[MarketId]):
    """Build a Market snapshot containing exactly the requested IDs."""
    req = MarketRequest(asof="2025-12-29", universe=Universe(ids))
    return provider.get_market(req)


def _spot_portfolio(spot_id: MarketId, *, quantity: float = 1.0) -> Portfolio:
    """One-position linear spot portfolio."""
    return Portfolio(
        positions=[Position(position_id="SPOT", instrument=FxSpot(spot_id=spot_id), quantity=float(quantity))]
    )


def _atm_call_portfolio(
    *,
    market,
    spot_id: MarketId,
    vol_id: MarketId,
    domestic_curve_id: MarketId,
    foreign_curve_id: MarketId,
    notional: float = 1_000_000.0,
    expiry: float = 1.0,
) -> Portfolio:
    """Single ATM-ish EURUSD call (strike = current spot)."""
    spot = float(market.quote(spot_id))
    instrument = EuropeanFxVanillaOption(
        option_type="call",
        notional=float(notional),
        strike=float(spot),
        expiry=float(expiry),
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=domestic_curve_id,
        foreign_curve_id=foreign_curve_id,
    )
    return Portfolio(positions=[Position(position_id="CALL", instrument=instrument, quantity=1.0)])


def _spot_only_report_rel_error(
    *,
    portfolio: Portfolio,
    market,
    portfolio_pricer: PortfolioPricer,
    spot_id: MarketId,
    rel_bump: float,
    include_gamma: bool,
) -> float:
    """
    Run greeks-vs-scenarios for a single spot scenario and return the rel_error for the shocked row.

    Assumes validate_greeks_vs_scenarios returns rows including BASE then the scenario.
    """
    scenario_name = f"spot_up_{rel_bump:g}"
    pack = ScenarioPack(
        scenarios={
            scenario_name: SpotShock(
                name=scenario_name,
                spot_id=spot_id,
                bump=float(rel_bump),
                bump_mode="relative",
            )
        }
    )

    report = validate_greeks_vs_scenarios(
        portfolio=portfolio,
        base_market=market,
        portfolio_pricer=portfolio_pricer,
        scenarios=pack,
        config=GreeksVsScenarioConfig(include_gamma_for_spot=include_gamma),
    )

    assert len(report.rows) >= 2
    return float(report.rows[1].rel_error)


# =============================================================================
# Delta-only tests
# =============================================================================

@pytest.mark.parametrize("rel_bump", [1e-4, 5e-4, 1e-3])  # 1bp, 5bp, 10bp
def test_delta_only_linear_spot_is_exact(
    provider: SyntheticProvider,
    portfolio_pricer: PortfolioPricer,
    rel_bump: float,
) -> None:
    """
    Linear spot should match delta-only prediction essentially exactly across small bumps.
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    market = _market(provider, [spot_id])
    portfolio = _spot_portfolio(spot_id, quantity=1.0)

    rel_error = _spot_only_report_rel_error(
        portfolio=portfolio,
        market=market,
        portfolio_pricer=portfolio_pricer,
        spot_id=spot_id,
        rel_bump=rel_bump,
        include_gamma=False,  # delta-only
    )

    assert rel_error < 1e-10


@pytest.mark.parametrize("rel_bump", [1e-5, 5e-5, 1e-4])  # 0.1bp, 0.5bp, 1bp
def test_delta_only_option_tiny_bumps_are_close(
    provider: SyntheticProvider,
    portfolio_pricer: PortfolioPricer,
    rel_bump: float,
) -> None:
    """
    For very tiny bumps, delta-only should explain spot PnL for options very well.
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    market = _market(provider, [spot_id, vol_id, rd_id, rf_id])
    portfolio = _atm_call_portfolio(
        market=market,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    rel_error = _spot_only_report_rel_error(
        portfolio=portfolio,
        market=market,
        portfolio_pricer=portfolio_pricer,
        spot_id=spot_id,
        rel_bump=rel_bump,
        include_gamma=False,  # delta-only
    )

    # Safe tolerance for tiny bumps (delta-only should be very close)
    assert rel_error < 5e-3


# =============================================================================
# Gamma+Delta tests
# =============================================================================

@pytest.mark.parametrize("rel_bump", [2e-3, 5e-3, 1e-2])  # 20bp, 50bp, 100bp
def test_gamma_plus_delta_is_not_worse_than_delta_only(
    provider: SyntheticProvider,
    portfolio_pricer: PortfolioPricer,
    rel_bump: float,
) -> None:
    """
    For moderate bumps, adding gamma should reduce (or at least not increase) error.
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    market = _market(provider, [spot_id, vol_id, rd_id, rf_id])
    portfolio = _atm_call_portfolio(
        market=market,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    err_delta_only = _spot_only_report_rel_error(
        portfolio=portfolio,
        market=market,
        portfolio_pricer=portfolio_pricer,
        spot_id=spot_id,
        rel_bump=rel_bump,
        include_gamma=False,
    )
    err_delta_gamma = _spot_only_report_rel_error(
        portfolio=portfolio,
        market=market,
        portfolio_pricer=portfolio_pricer,
        spot_id=spot_id,
        rel_bump=rel_bump,
        include_gamma=True,
    )

    assert err_delta_gamma <= err_delta_only + 1e-12


@pytest.mark.parametrize("rel_bump", [5e-3])  # 50bp: gamma effect is typically visible
def test_gamma_plus_delta_expected_to_improve_for_option(
    provider: SyntheticProvider,
    portfolio_pricer: PortfolioPricer,
    rel_bump: float,
) -> None:
    """
    Stronger assertion for a representative bump: gamma should usually improve.
    If this ever flakes due to model/market parameters, relax to 'not worse' test only.
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    market = _market(provider, [spot_id, vol_id, rd_id, rf_id])
    portfolio = _atm_call_portfolio(
        market=market,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    err_delta_only = _spot_only_report_rel_error(
        portfolio=portfolio,
        market=market,
        portfolio_pricer=portfolio_pricer,
        spot_id=spot_id,
        rel_bump=rel_bump,
        include_gamma=False,
    )
    err_delta_gamma = _spot_only_report_rel_error(
        portfolio=portfolio,
        market=market,
        portfolio_pricer=portfolio_pricer,
        spot_id=spot_id,
        rel_bump=rel_bump,
        include_gamma=True,
    )

    assert err_delta_gamma < err_delta_only