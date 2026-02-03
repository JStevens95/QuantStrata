"""Unit tests for JAX-based FX vanilla MC pricer (optional; skipped when JAX not installed)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytest

from src.core.performance.backend import jax_available
from src.marketdata.core.ids import MarketId
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer


# Minimal market for FX vanilla (same interface as test_fx_european_mc_pricer)
@dataclass(frozen=True, slots=True)
class _FlatCurve:
    rate: float

    def df(self, t: float) -> float:
        if t < 0.0:
            raise ValueError("t must be >= 0.")
        return float(math.exp(-float(self.rate) * t))


@dataclass(frozen=True, slots=True)
class _FlatVolSurface:
    sigma: float

    def vol(self, *, expiry: float, strike: float) -> float:
        if expiry < 0.0:
            raise ValueError("expiry must be >= 0.")
        return float(self.sigma)


@dataclass(frozen=True, slots=True)
class _DummyMarket:
    spot: float
    rd: float
    rf: float
    sigma: float
    spot_id: MarketId
    vol_id: MarketId
    rd_id: MarketId
    rf_id: MarketId

    def quote(self, market_id: MarketId) -> float:
        if market_id != self.spot_id:
            raise KeyError(f"Unknown quote id: {market_id}")
        return float(self.spot)

    def curve(self, market_id: MarketId):
        if market_id == self.rd_id:
            return _FlatCurve(rate=self.rd)
        if market_id == self.rf_id:
            return _FlatCurve(rate=self.rf)
        raise KeyError(f"Unknown curve id: {market_id}")

    def vol_surface(self, market_id: MarketId):
        if market_id != self.vol_id:
            raise KeyError(f"Unknown vol id: {market_id}")
        return _FlatVolSurface(sigma=self.sigma)


@pytest.fixture(scope="module")
def ids() -> Dict[str, MarketId]:
    return {
        "spot": MarketId("FX", "SPOT", "EURUSD"),
        "vol": MarketId("FX", "VOL", "EURUSD.VOL"),
        "rd": MarketId("IR", "CURVE", "USD.OIS"),
        "rf": MarketId("IR", "CURVE", "EUR.OIS"),
    }


@pytest.fixture(scope="module")
def base_params() -> Dict[str, float]:
    return {
        "spot": 1.25,
        "strike": 1.25,
        "t": 1.0,
        "rd": 0.03,
        "rf": 0.01,
        "sigma": 0.20,
        "notional": 1_000_000.0,
    }


@pytest.fixture()
def market(ids: Dict[str, MarketId], base_params: Dict[str, float]) -> _DummyMarket:
    return _DummyMarket(
        spot=float(base_params["spot"]),
        rd=float(base_params["rd"]),
        rf=float(base_params["rf"]),
        sigma=float(base_params["sigma"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )


@pytest.mark.skipif(not jax_available(), reason="JAX not installed; skip JAX MC pricer tests")
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_jax_mc_price_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    from src.pricers.fx.european_bsm_jax_mc import FxVanillaEuropeanJaxMcPricer

    trade = FxVanillaEuropeanOption(
        option_type=option_type,
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    bsm = FxVanillaEuropeanOptionBsmPricer()
    jax_mc = FxVanillaEuropeanJaxMcPricer(n_paths=100_000, seed=7)

    pv_bsm = float(bsm.price(trade, market))
    pv_jax = float(jax_mc.price(trade, market))

    assert pv_jax == pytest.approx(pv_bsm, rel=0.03, abs=1e-2)


@pytest.mark.skipif(not jax_available(), reason="JAX not installed; skip JAX MC pricer tests")
def test_jax_mc_run_returns_simulation_artifact(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    from src.pricers.fx.european_bsm_jax_mc import FxVanillaEuropeanJaxMcPricer
    from src.pricers.fx.european_bsm_mc import FxVanillaOptionMcSimulation

    trade = FxVanillaEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    jax_mc = FxVanillaEuropeanJaxMcPricer(n_paths=10_000, seed=42)
    sim = jax_mc.run(trade, market, store_paths=False)

    assert isinstance(sim, FxVanillaOptionMcSimulation)
    assert sim.terminal_spots.shape == (10_000,)
    assert sim.discounted_payoffs.shape == (10_000,)
    assert np.isfinite(sim.discounted_payoffs).all()
    assert float(sim.discounted_payoffs.mean()) == pytest.approx(
        jax_mc.price(trade, market), rel=1e-5
    )
