from __future__ import annotations

import pytest

from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.linear.spot import FxSpot
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry

from src.risk.sensitivities.config import SensitivitiesBumps, SensitivitiesConfig
from src.risk.sensitivities.engine import compute_sensitivities


@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    return SyntheticProvider(seed=123)


@pytest.fixture(scope="module")
def pricer() -> PortfolioPricer:
    return PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())


def _market(provider: SyntheticProvider, ids: list[MarketId]):
    return provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe(ids)))


def test_fd_delta_matches_analytic_for_linear_spot(provider: SyntheticProvider, pricer: PortfolioPricer) -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    market = _market(provider, [spot_id])

    portfolio = Portfolio(positions=[Position("SPOT", FxSpot(spot_id=spot_id), quantity=1.0)])

    # Analytic delta
    analytic = compute_sensitivities(
        portfolio,
        market,
        pricer,
        config=SensitivitiesConfig(method="analytic"),
        requested_greeks=("delta",),
    )
    delta_a = analytic.rows[0].value

    # FD delta
    fd = compute_sensitivities(
        portfolio,
        market,
        pricer,
        config=SensitivitiesConfig(method="fd_central", bumps=SensitivitiesBumps(spot_rel=1e-4)),
        requested_greeks=("delta",),
    )
    delta_fd = fd.rows[0].value

    assert delta_fd == pytest.approx(delta_a, rel=1e-10, abs=1e-12)


def test_fd_rho_matches_analytic_for_forward(provider: SyntheticProvider, pricer: PortfolioPricer) -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    market = _market(provider, [spot_id, rd_id, rf_id])

    S0 = float(market.quote(spot_id))
    T = 1.0

    df_d = float(market.curve(rd_id).df(T))
    df_f = float(market.curve(rf_id).df(T))
    F0 = S0 * df_f / df_d  # fair strike

    inst = FxForward(
        notional=1_000_000.0,
        strike=F0,
        expiry=T,
        spot_id=spot_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )
    portfolio = Portfolio(positions=[Position("FWD", inst, quantity=1.0)])

    # Analytic rhos from pricer
    analytic = compute_sensitivities(
        portfolio,
        market,
        pricer,
        config=SensitivitiesConfig(
            method="analytic",
            rho_key_by_curve_id={rd_id: "rho_domestic", rf_id: "rho_foreign"},
        ),
        requested_greeks=("rho_domestic", "rho_foreign"),
    )
    rho_d_a = next(r.value for r in analytic.rows if r.key.greek == "rho_domestic")
    rho_f_a = next(r.value for r in analytic.rows if r.key.greek == "rho_foreign")

    # FD rhos
    fd = compute_sensitivities(
        portfolio,
        market,
        pricer,
        config=SensitivitiesConfig(
            method="fd_central",
            bumps=SensitivitiesBumps(rate_abs=1e-5),  # 1bp
            rho_key_by_curve_id={rd_id: "rho_domestic", rf_id: "rho_foreign"},
        ),
        requested_greeks=("rho_domestic", "rho_foreign"),
    )
    rho_d_fd = next(r.value for r in fd.rows if r.key.greek == "rho_domestic")
    rho_f_fd = next(r.value for r in fd.rows if r.key.greek == "rho_foreign")

    assert rho_d_fd == pytest.approx(rho_d_a, rel=5e-6, abs=1e-6)
    assert rho_f_fd == pytest.approx(rho_f_a, rel=5e-6, abs=1e-6)