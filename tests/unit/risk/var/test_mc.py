"""Unit tests for Monte Carlo VaR."""

from __future__ import annotations

import numpy as np
import pytest

from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import MarketRequest, Universe
from src.marketdata.providers.synthetic.provider import SyntheticProvider
from src.marketdata.scenarios.shocks import SpotShock
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry
from src.risk.sensitivities.result import SensitivityKey
from src.risk.var.config import VarConfig, VarResult
from src.risk.var.mc import DiagonalFactorModel, mc_var


@pytest.fixture(scope="module")
def provider() -> SyntheticProvider:
    return SyntheticProvider(seed=42)


@pytest.fixture(scope="module")
def pricer() -> PortfolioPricer:
    return PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())


@pytest.fixture(scope="module")
def spot_id() -> MarketId:
    return MarketId.parse("FX.SPOT.EURUSD")


@pytest.fixture(scope="module")
def portfolio(spot_id: MarketId) -> Portfolio:
    opt = FxVanillaEuropeanOption(
        option_type="call",
        notional=1_000_000.0,
        strike=1.10,
        expiry=1.0,
        spot_id=spot_id,
        domestic_curve_id=MarketId.parse("IR.CURVE.USD.OIS"),
        foreign_curve_id=MarketId.parse("IR.CURVE.EUR.OIS"),
        vol_id=MarketId.parse("FX.VOL.EURUSD"),
    )
    return Portfolio(positions=[Position("opt1", opt, quantity=1.0)])


@pytest.fixture(scope="module")
def market(provider: SyntheticProvider, spot_id: MarketId):
    ids = [
        spot_id,
        MarketId.parse("IR.CURVE.USD.OIS"),
        MarketId.parse("IR.CURVE.EUR.OIS"),
        MarketId.parse("FX.VOL.EURUSD"),
    ]
    return provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe(ids)))


def test_mc_var_basic(
    portfolio: Portfolio,
    market,
    pricer: PortfolioPricer,
    spot_id: MarketId,
) -> None:
    delta_key = SensitivityKey(greek="delta", market_id=spot_id)
    factor_order = [delta_key]
    factor_volatilities = {delta_key: 0.01}
    shock_builders = {
        delta_key: lambda v: SpotShock(
            name="mc_spot",
            spot_id=spot_id,
            bump=v,
            bump_mode="relative",
        ),
    }
    factor_model = DiagonalFactorModel(
        factor_order=factor_order,
        factor_volatilities=factor_volatilities,
        shock_builders=shock_builders,
    )
    config = VarConfig(confidence=0.99, horizon_days=1, method="mc")
    result = mc_var(
        portfolio,
        market,
        pricer,
        factor_model,
        config,
        n_paths=500,
        seed=123,
    )
    assert isinstance(result, VarResult)
    assert result.method == "mc"
    assert result.var > 0
    assert result.cvar is not None
    assert result.metadata["n_simulations"] == 500


def test_mc_var_wrong_method_raises(
    portfolio: Portfolio,
    market,
    pricer: PortfolioPricer,
    spot_id: MarketId,
) -> None:
    delta_key = SensitivityKey(greek="delta", market_id=spot_id)
    factor_model = DiagonalFactorModel(
        factor_order=[delta_key],
        factor_volatilities={delta_key: 0.01},
        shock_builders={
            delta_key: lambda v: SpotShock("s", spot_id=spot_id, bump=v, bump_mode="relative"),
        },
    )
    config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
    with pytest.raises(ValueError, match="method must be 'mc'"):
        mc_var(portfolio, market, pricer, factor_model, config, n_paths=10)
