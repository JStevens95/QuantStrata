"""
Mock market-data client.

Deterministic, internally-consistent FX/Rates market data + shocks for offline
development and tests. Seeded per lookup key so runs are reproducible. Replace with the
STAR adapter in production — the **array shapes returned here are the contract** your
adapter must satisfy (diff your real payloads against these).
"""
from __future__ import annotations

import numpy as np

from src.rade_static_replication.clients.payloads import (
    CubePayload,
    CurvePayload,
    FXShockPayload,
    RatesShockPayload,
    SurfacePayload,
)

# Shared grids (the axis labels every payload is built on).
CURVE_TENORS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
VOL_MONEYNESS = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
VOL_EXPIRIES = np.array([0.25, 0.5, 1.0, 2.0])
CUBE_EXPIRIES = np.array([1.0, 2.0, 5.0])
CUBE_TENORS = np.array([2.0, 5.0, 10.0])
CUBE_STRIKES = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
N_SCENARIOS = 60

# Plausible COB levels keyed by parser output (foreign+domestic) / currency.
_BASE_SPOT = {"EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 150.0, "AUDUSD": 0.66}
_BASE_RATE = {"USD": 0.045, "EUR": 0.032, "GBP": 0.047, "JPY": 0.004}


def _seeded_rng(*key) -> np.random.Generator:
    """A reproducible RNG derived from a lookup key (so a pair/ccy always yields the same data)."""
    return np.random.default_rng(abs(hash(key)) % (2**32))


class MockMarketDataClient:
    """Generates seeded, consistent market data + shocks."""

    def __init__(self, n_scenarios: int = N_SCENARIOS) -> None:
        self.n_scenarios = n_scenarios
        self.scenario_ids = np.array([f"scen_{i}" for i in range(self.n_scenarios)])

    # ---- FX ----

    def fx_spot(self, pair: str, cob_date: str) -> float:
        return _BASE_SPOT.get(pair.upper(), 1.0)

    def fx_vol_surface(self, pair: str, cob_date: str) -> SurfacePayload:
        rng = _seeded_rng("fxvol", pair)
        atm_vol = 0.08 + 0.04 * rng.random()
        smile = 0.02 * (VOL_MONEYNESS - 1.0) ** 2 / 0.01      # convex in moneyness
        term_factor = 1.0 + 0.1 * np.log1p(VOL_EXPIRIES)      # mild upward term structure
        vols = atm_vol * np.outer(term_factor, np.ones_like(VOL_MONEYNESS)) + smile[None, :]
        return SurfacePayload(VOL_EXPIRIES, VOL_MONEYNESS, vols, "moneyness", "lognormal")

    def fx_shocks(self, pair: str, cob_date: str) -> FXShockPayload:
        rng = _seeded_rng("fxshock", pair)
        n = self.n_scenarios
        spot = self.fx_spot(pair, cob_date) * np.exp(rng.normal(0, 0.01, n))          # lognormal spot moves
        base_surface = self.fx_vol_surface(pair, cob_date).vols
        vol = base_surface[None, :, :] * np.exp(rng.normal(0, 0.05, (n, *base_surface.shape)))

        foreign_ccy = pair[:3].upper()
        domestic_curve = self.discount_curve("USD", cob_date)
        foreign_curve = self.discount_curve(foreign_ccy, cob_date)
        domestic_rate = domestic_curve.zero_rates[None, :] + rng.normal(0, 5e-4, (n, CURVE_TENORS.size))
        foreign_rate = foreign_curve.zero_rates[None, :] + rng.normal(0, 5e-4, (n, CURVE_TENORS.size))
        return FXShockPayload(
            scenario_ids=self.scenario_ids, spot=spot, vol=vol,
            vol_expiries=VOL_EXPIRIES, vol_strikes=VOL_MONEYNESS,
            domestic_rate=domestic_rate, domestic_tenors=CURVE_TENORS,
            foreign_rate=foreign_rate, foreign_tenors=CURVE_TENORS,
        )

    # ---- Rates / shared ----

    def discount_curve(self, currency: str, cob_date: str) -> CurvePayload:
        rng = _seeded_rng("curve", currency)
        level = _BASE_RATE.get(currency.upper(), 0.03)
        slope = 0.004 * rng.random()
        zero_rates = level + slope * np.log1p(CURVE_TENORS)   # gently upward-sloping curve
        return CurvePayload(CURVE_TENORS, zero_rates)

    def rates_vol_cube(self, factor_id: str, cob_date: str) -> CubePayload:
        rng = _seeded_rng("cube", factor_id)
        atm_normal_vol = 0.006 + 0.003 * rng.random()
        skew = 0.0005 * (CUBE_STRIKES - 0.04) / 0.02
        vols = atm_normal_vol + skew[None, None, :] + np.zeros(
            (CUBE_EXPIRIES.size, CUBE_TENORS.size, CUBE_STRIKES.size)
        )
        return CubePayload(CUBE_EXPIRIES, CUBE_TENORS, CUBE_STRIKES, np.abs(vols), "normal")

    def rates_shocks(self, factor_id: str, cob_date: str) -> RatesShockPayload:
        rng = _seeded_rng("rshock", factor_id)
        currency = factor_id.split(".")[-1]
        base_curve = self.discount_curve(currency, cob_date).zero_rates
        n = self.n_scenarios
        curve = base_curve[None, :] + rng.normal(0, 8e-4, (n, base_curve.size))       # additive rate shocks
        base_cube = self.rates_vol_cube(factor_id, cob_date).vols
        vol = np.abs(base_cube[None, ...] + rng.normal(0, 5e-4, (n, *base_cube.shape)))
        return RatesShockPayload(
            scenario_ids=self.scenario_ids, curve=curve, curve_tenors=CURVE_TENORS,
            vol=vol, vol_expiries=CUBE_EXPIRIES, vol_swap_tenors=CUBE_TENORS, vol_strikes=CUBE_STRIKES,
        )
