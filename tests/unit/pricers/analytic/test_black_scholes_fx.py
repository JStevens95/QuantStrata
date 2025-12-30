from __future__ import annotations

import pytest
from dataclasses import dataclass

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider
from src.instruments.fx.options.european import EuropeanFxOption
from src.pricers.analytic.black_scholes import BlackScholesPricer


# -----------------------------------------------------------------------------
# Small test-only wrappers (duck-typed) to override one market input cleanly.
# These wrappers keep your production Market class untouched.
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SpotOverrideMarket:
    """Market wrapper that overrides a single SPOT quote and delegates everything else."""
    base_market: object
    spot_id: MarketId
    overridden_spot: float

    def quote(self, market_id: MarketId) -> float:
        if market_id == self.spot_id:
            return float(self.overridden_spot)
        return float(self.base_market.quote(market_id))

    def curve(self, market_id: MarketId):
        return self.base_market.curve(market_id)

    def vol_surface(self, market_id: MarketId):
        return self.base_market.vol_surface(market_id)


@dataclass(frozen=True, slots=True)
class _FlatVolSurfaceOverride:
    """VolSurface wrapper that always returns a bumped vol, ignoring (expiry,strike)."""
    bumped_sigma: float

    def vol(self, expiry: float, strike: float) -> float:
        return float(self.bumped_sigma)


@dataclass(frozen=True, slots=True)
class _VolOverrideMarket:
    """Market wrapper that overrides a single VOL surface and delegates everything else."""
    base_market: object
    vol_id: MarketId
    bumped_sigma: float

    def quote(self, market_id: MarketId) -> float:
        return float(self.base_market.quote(market_id))

    def curve(self, market_id: MarketId):
        return self.base_market.curve(market_id)

    def vol_surface(self, market_id: MarketId):
        if market_id == self.vol_id:
            return _FlatVolSurfaceOverride(bumped_sigma=self.bumped_sigma)
        return self.base_market.vol_surface(market_id)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fx_market_and_ids():
    """
    Build a deterministic Market snapshot suitable for FX vanilla tests.

    We include:
    - FX spot
    - FX vol
    - domestic + foreign curves (both flat in V1)
    """
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    rd_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS")
    rf_id = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([spot_id, vol_id, rd_id, rf_id]),
        )
    )
    return market, spot_id, vol_id, rd_id, rf_id


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_intrinsic_value_at_expiry(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = BlackScholesPricer()

    S = market.quote(spot_id)
    K = S * 0.98
    N = 1_000_000.0

    call = EuropeanFxOption("call", N, strike=K, expiry=0.0, spot_id=spot_id, vol_id=vol_id, domestic_curve_id=rd_id, foreign_curve_id=rf_id)
    put = EuropeanFxOption("put", N, strike=K, expiry=0.0, spot_id=spot_id, vol_id=vol_id, domestic_curve_id=rd_id, foreign_curve_id=rf_id)

    call_pv = pricer.price(call, market)
    put_pv = pricer.price(put, market)

    assert call_pv == pytest.approx(N * max(S - K, 0.0))
    assert put_pv == pytest.approx(N * max(K - S, 0.0))


def test_put_call_parity(fx_market_and_ids) -> None:
    """
    FX put-call parity under Garman–Kohlhagen:
        C - P = N * ( S*df_f(T) - K*df_d(T) )
    """
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = BlackScholesPricer()

    T = 1.0
    N = 2_000_000.0

    S = market.quote(spot_id)
    K = S  # ATM-ish makes parity robust numerically

    call = EuropeanFxOption("call", N, K, T, spot_id, vol_id, rd_id, rf_id)
    put = EuropeanFxOption("put", N, K, T, spot_id, vol_id, rd_id, rf_id)

    C = pricer.price(call, market)
    P = pricer.price(put, market)

    df_d = market.curve(rd_id).df(T)
    df_f = market.curve(rf_id).df(T)

    rhs = N * (S * df_f - K * df_d)
    assert (C - P) == pytest.approx(rhs, rel=1e-10, abs=1e-7)


def test_greek_signs_and_ranges(fx_market_and_ids) -> None:
    """
    Basic sanity checks:
    - call delta should be positive
    - put delta should be negative (usually)
    - gamma and vega should be positive for vanilla European
    """
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = BlackScholesPricer()

    S = market.quote(spot_id)
    T = 1.0
    K = S
    N = 1_000_000.0

    call = EuropeanFxOption("call", N, K, T, spot_id, vol_id, rd_id, rf_id)
    put = EuropeanFxOption("put", N, K, T, spot_id, vol_id, rd_id, rf_id)

    g_call = pricer.greeks(call, market)
    g_put = pricer.greeks(put, market)

    assert g_call["delta"] > 0.0
    assert g_put["delta"] < 0.0

    assert g_call["gamma"] > 0.0
    assert g_put["gamma"] > 0.0

    assert g_call["vega"] > 0.0
    assert g_put["vega"] > 0.0

    # Delta should be bounded by notional * df_f in magnitude for FX vanilla.
    df_f = market.curve(rf_id).df(T)
    delta_bound = N * df_f
    assert abs(g_call["delta"]) <= delta_bound + 1e-6
    assert abs(g_put["delta"]) <= delta_bound + 1e-6


def test_delta_matches_finite_difference(fx_market_and_ids) -> None:
    """
    Validate analytic delta against a central finite difference in spot.

    We do NOT mutate Market; we wrap it with a spot override.
    """
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = BlackScholesPricer()

    S0 = market.quote(spot_id)
    T = 1.0
    K = S0
    N = 1_000_000.0

    trade = EuropeanFxOption("call", N, K, T, spot_id, vol_id, rd_id, rf_id)

    # Analytic delta
    delta = pricer.greeks(trade, market)["delta"]

    # Spot bump (relative bump keeps scale sensible across assets)
    bump = 1e-4  # 1bp relative bump
    S_up = S0 * (1.0 + bump)
    S_dn = S0 * (1.0 - bump)

    m_up = _SpotOverrideMarket(base_market=market, spot_id=spot_id, overridden_spot=S_up)
    m_dn = _SpotOverrideMarket(base_market=market, spot_id=spot_id, overridden_spot=S_dn)

    pv_up = pricer.price(trade, m_up)
    pv_dn = pricer.price(trade, m_dn)

    # Central difference approximation of dPV/dS
    delta_fd = (pv_up - pv_dn) / (S_up - S_dn)

    # Tolerance: analytic vs FD should match closely for smooth vanilla pricing
    assert delta == pytest.approx(delta_fd, rel=5e-4, abs=1e-2)


def test_vega_matches_finite_difference(fx_market_and_ids) -> None:
    """
    Validate analytic vega against a central finite difference in implied vol.

    We wrap the market vol surface to return a bumped flat sigma.
    """
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = BlackScholesPricer()

    S0 = market.quote(spot_id)
    T = 1.0
    K = S0
    N = 1_000_000.0

    trade = EuropeanFxOption("call", N, K, T, spot_id, vol_id, rd_id, rf_id)

    # Base sigma from market surface
    sigma0 = market.vol_surface(vol_id).vol(expiry=T, strike=K)

    # Analytic vega (per 1.00 vol)
    vega = pricer.greeks(trade, market)["vega"]

    # Vol bump (absolute bump in sigma)
    eps = 1e-4
    sigma_up = sigma0 + eps
    sigma_dn = max(sigma0 - eps, 1e-8)

    m_up = _VolOverrideMarket(base_market=market, vol_id=vol_id, bumped_sigma=sigma_up)
    m_dn = _VolOverrideMarket(base_market=market, vol_id=vol_id, bumped_sigma=sigma_dn)

    pv_up = pricer.price(trade, m_up)
    pv_dn = pricer.price(trade, m_dn)

    vega_fd = (pv_up - pv_dn) / (sigma_up - sigma_dn)

    assert vega == pytest.approx(vega_fd, rel=5e-4, abs=1e-2)
