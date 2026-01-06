from __future__ import annotations

import pytest

# ---- instrument objects ----
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption

# ---- marketdata objects ----
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.providers.synthetic.provider import SyntheticProvider
from src.marketdata.scenarios.shocks import SpotShock

# ---- portfolio objects ----
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer

# ---- pricer objects ----
from src.pricers.registry import DefaultPricerRegistry
from src.pricers.fx.european_fde import FxEuropeanVanillaFdPricer

# ---- sensitivity objects ----
from src.risk.sensitivities.config import SensitivitiesBumps, SensitivitiesConfig
from src.risk.sensitivities.engine import compute_sensitivities


@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    return SyntheticProvider(seed=123)


@pytest.fixture(scope="module")
def pricer() -> PortfolioPricer:
    return PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())


@pytest.fixture(scope="module")
def pricer_fd() -> PortfolioPricer:
    registry = DefaultPricerRegistry().build()

    # Ensure the named FD pricer is registered (instance, not class).
    registry.register(EuropeanFxVanillaOption, FxEuropeanVanillaFdPricer(), pricer_id="fd", overwrite=True)

    return PortfolioPricer(pricer_registry=registry)


def _market(provider: SyntheticProvider, ids: list[MarketId]):
    return provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe(ids)))


def _central_gamma_from_pv(
    *,
    portfolio: Portfolio,
    market,
    pricer: PortfolioPricer,
    spot_id: MarketId,
    spot_rel_bump: float,
) -> float:


    s0 = float(market.quote(spot_id))
    h = float(s0 * spot_rel_bump)

    shock_up = SpotShock(name="up", spot_id=spot_id, bump=spot_rel_bump, bump_mode="relative")
    shock_dn = SpotShock(name="dn", spot_id=spot_id, bump=-spot_rel_bump, bump_mode="relative")

    pv0 = float(pricer.price(portfolio, market).totals.pv)
    pv_up = float(pricer.price(portfolio, shock_up.apply(market)).totals.pv)
    pv_dn = float(pricer.price(portfolio, shock_dn.apply(market)).totals.pv)

    return float((pv_up - 2.0 * pv0 + pv_dn) / (h * h))


def test_unknown_greek_returns_empty(provider: SyntheticProvider, pricer: PortfolioPricer) -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    market = _market(provider, [spot_id])
    portfolio = Portfolio(positions=[Position("SPOT", FxSpot(spot_id=spot_id), quantity=1.0)])

    res = compute_sensitivities(
        portfolio, market, pricer,
        config=SensitivitiesConfig(method="fd_central"),
        requested_greeks=("not_a_greek",),
    )
    assert res.rows == []


def test_invalid_method_raises(provider: SyntheticProvider, pricer: PortfolioPricer) -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    market = _market(provider, [spot_id])
    portfolio = Portfolio(positions=[Position("SPOT", FxSpot(spot_id=spot_id), quantity=1.0)])

    with pytest.raises(ValueError):
        compute_sensitivities(
            portfolio, market, pricer,
            config=SensitivitiesConfig(method="nope"),
            requested_greeks=("delta",),
        )


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


def test_fd_delta_for_vanilla_option_matches_analytic_bsm(
    provider: SyntheticProvider,
    pricer: PortfolioPricer,
    pricer_fd: PortfolioPricer,
) -> None:
    """
    Validate that bump-and-reprice delta for a European vanilla option
    (priced via the FD/PDE pricer) matches the analytic BSM delta (default pricer).

    Routing:
      - analytic greeks come from default pricer (BSM)
      - FD bump PVs come from FD/PDE pricer via Position.pricer_id="fd"
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")
    vol_id = MarketId("FX", "VOL", "EURUSD")

    market = _market(provider, [spot_id, rd_id, rf_id, vol_id])

    S0 = float(market.quote(spot_id))
    T = 1.0
    K = S0  # ATM
    notional = 1_000_000.0

    opt = EuropeanFxVanillaOption(
        notional=notional,
        strike=K,
        expiry=T,
        spot_id=spot_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
        vol_id=vol_id,
        option_type="call",
    )

    # Analytic delta from BSM pricer (default routing).
    portfolio_bsm = Portfolio(positions=[Position("OPT", opt, quantity=1.0)])
    delta_a = compute_sensitivities(
        portfolio_bsm,
        market,
        pricer,
        config=SensitivitiesConfig(method="analytic"),
        requested_greeks=("delta",),
    ).rows[0].value

    # FD delta from bump-and-reprice, but PVs computed via FD/PDE pricer.
    portfolio_fd = Portfolio(positions=[Position("OPT", opt, quantity=1.0, pricer_id="fd")])
    delta_fd = compute_sensitivities(
        portfolio_fd,
        market,
        pricer_fd,
        config=SensitivitiesConfig(method="fd_central", bumps=SensitivitiesBumps(spot_rel=1e-4)),
        requested_greeks=("delta",),
    ).rows[0].value

    # Tolerance: FD grid discretization can introduce small error.
    assert delta_fd == pytest.approx(delta_a, rel=5e-4, abs=1e-4)


def test_fd_gamma_matches_analytic_for_vanilla_option_using_bsm_pv(
    provider: SyntheticProvider,
    pricer: PortfolioPricer,
) -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")
    vol_id = MarketId("FX", "VOL", "EURUSD")

    market = _market(provider, [spot_id, rd_id, rf_id, vol_id])

    S0 = float(market.quote(spot_id))
    T = 1.0
    K = S0
    notional = 1_000_000.0

    opt = EuropeanFxVanillaOption(
        notional=notional,
        strike=K,
        expiry=T,
        spot_id=spot_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
        vol_id=vol_id,
        option_type="call",
    )
    portfolio = Portfolio(positions=[Position("OPT", opt, quantity=1.0)])

    gamma_a = compute_sensitivities(
        portfolio,
        market,
        pricer,
        config=SensitivitiesConfig(method="analytic"),
        requested_greeks=("gamma",),
    ).rows[0].value

    gamma_fd = compute_sensitivities(
        portfolio,
        market,
        pricer,  # <--- BSM PVs
        config=SensitivitiesConfig(method="fd_central", bumps=SensitivitiesBumps(spot_rel=1e-4)),
        requested_greeks=("gamma",),
    ).rows[0].value

    assert gamma_fd == pytest.approx(gamma_a, rel=1e-4, abs=1e-6)


def test_fd_gamma_uses_fd_pricer_routing_and_matches_manual_central_diff(
    provider: SyntheticProvider,
    pricer_fd: PortfolioPricer,
) -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")
    vol_id = MarketId("FX", "VOL", "EURUSD")

    market = _market(provider, [spot_id, rd_id, rf_id, vol_id])

    S0 = float(market.quote(spot_id))
    T = 1.0
    K = S0
    notional = 1_000_000.0

    opt = EuropeanFxVanillaOption(
        notional=notional,
        strike=K,
        expiry=T,
        spot_id=spot_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
        vol_id=vol_id,
        option_type="call",
    )

    # Force FD/PDE pricer usage via Position.pricer_id="fd"
    portfolio_fd = Portfolio(positions=[Position("OPT", opt, quantity=1.0, pricer_id="fd")])

    spot_rel = 1e-4

    gamma_engine = compute_sensitivities(
        portfolio_fd,
        market,
        pricer_fd,
        config=SensitivitiesConfig(method="fd_central", bumps=SensitivitiesBumps(spot_rel=spot_rel)),
        requested_greeks=("gamma",),
    ).rows[0].value

    gamma_manual = _central_gamma_from_pv(
        portfolio=portfolio_fd,
        market=market,
        pricer=pricer_fd,
        spot_id=spot_id,
        spot_rel_bump=spot_rel,
    )

    # This should be extremely tight: same PV engine, same shocks, same formula.
    assert gamma_engine == pytest.approx(gamma_manual, rel=1e-12, abs=1e-9)