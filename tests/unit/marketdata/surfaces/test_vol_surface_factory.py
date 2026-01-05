from __future__ import annotations

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict

from src.marketdata.ids import MarketId
from src.marketdata.scenarios.shocks import VolShock
from src.marketdata.surfaces.factories import FlatVolFactory, GridVolFactory
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface


# =============================================================================
# Test doubles (minimal MarketView to exercise shocks without needing full Market)
# =============================================================================

@dataclass(frozen=True, slots=True)
class _DummyMarketView:
    """
    Minimal MarketView implementation for testing vol shocks.

    We intentionally keep this tiny: VolShock should only call vol_surface(),
    and delegate quote()/curve() if asked.
    """
    vol_surfaces: Dict[MarketId, object]
    quotes: Dict[MarketId, float] | None = None
    curves: Dict[MarketId, object] | None = None

    def quote(self, market_id: MarketId) -> float:
        if self.quotes is None:
            raise KeyError(f"No quotes configured for market_id={market_id}")
        return float(self.quotes[market_id])

    def curve(self, market_id: MarketId):
        if self.curves is None:
            raise KeyError(f"No curves configured for market_id={market_id}")
        return self.curves[market_id]

    def vol_surface(self, market_id: MarketId):
        return self.vol_surfaces[market_id]


# =============================================================================
# Existing surface + factory tests
# =============================================================================

def test_flat_vol_surface_returns_constant_vol() -> None:
    surface = FlatVolSurface(sigma=0.12)

    assert surface.vol(expiry=0.0, strike=1.0) == pytest.approx(0.12)
    assert surface.vol(expiry=1.0, strike=0.8) == pytest.approx(0.12)
    assert surface.vol(expiry=2.0, strike=1.5) == pytest.approx(0.12)

    with pytest.raises(ValueError):
        surface.vol(expiry=-0.1, strike=1.0)


def test_flat_vol_surface_validation() -> None:
    with pytest.raises(ValueError):
        FlatVolSurface(sigma=0.0)
    with pytest.raises(ValueError):
        FlatVolSurface(sigma=-0.2)
    with pytest.raises(ValueError):
        FlatVolSurface(sigma=float("nan"))


def test_flat_vol_factory_accepts_scalar_formats() -> None:
    factory = FlatVolFactory()

    s1 = factory.build(np.asarray(0.25))
    assert s1.vol(expiry=1.0, strike=1.0) == pytest.approx(0.25)

    s2 = factory.build(np.asarray([0.18]))
    assert s2.vol(expiry=3.0, strike=2.0) == pytest.approx(0.18)

    with pytest.raises(ValueError):
        factory.build(np.asarray([0.1, 0.2]))


def test_grid_vol_surface_exact_on_grid_points() -> None:
    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([90.0, 100.0, 110.0], dtype=float)

    vols = np.array(
        [
            [0.20, 0.21, 0.22],
            [0.25, 0.26, 0.27],
        ],
        dtype=float,
    )

    surf = GridVolSurface(expiries=expiries, strikes=strikes, implied_vols=vols)

    assert surf.vol(0.5, 90.0) == pytest.approx(0.20)
    assert surf.vol(0.5, 100.0) == pytest.approx(0.21)
    assert surf.vol(1.0, 110.0) == pytest.approx(0.27)


def test_grid_vol_surface_bilinear_interpolation_is_correct_for_plane() -> None:
    """
    Bilinear interpolation is exact for functions that are affine in each variable.

    Use: vol(t, k) = t + 0.001*k
    """
    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([100.0, 110.0], dtype=float)

    # Build grid by formula
    vols = np.array([[t + 0.001 * k for k in strikes] for t in expiries], dtype=float)

    surf = GridVolSurface(expiries=expiries, strikes=strikes, implied_vols=vols)

    t_q = 0.75
    k_q = 105.0
    expected = t_q + 0.001 * k_q

    assert surf.vol(t_q, k_q) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_grid_vol_surface_flat_extrapolation_clamps_to_edges() -> None:
    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([90.0, 100.0, 110.0], dtype=float)

    vols = np.array(
        [
            [0.20, 0.21, 0.22],
            [0.25, 0.26, 0.27],
        ],
        dtype=float,
    )

    surf = GridVolSurface(expiries=expiries, strikes=strikes, implied_vols=vols, extrapolation="flat")

    # expiry below min -> clamp to 0.5 row
    assert surf.vol(0.25, 100.0) == pytest.approx(0.21)

    # strike above max -> clamp to 110 col; expiry below min -> clamp to first row
    assert surf.vol(0.25, 200.0) == pytest.approx(0.22)

    # expiry above max -> clamp to 1.0 row; strike below min -> clamp to 90 col
    assert surf.vol(5.0, 50.0) == pytest.approx(0.25)


def test_grid_vol_factory_builds_surface_from_flat_params() -> None:
    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([100.0, 110.0], dtype=float)

    # params are vols flattened row-major:
    # row0: [0.20, 0.21], row1: [0.25, 0.26]
    params = np.array([0.20, 0.21, 0.25, 0.26], dtype=float)

    factory = GridVolFactory(expiries=expiries, strikes=strikes, extrapolation="flat")
    surf = factory.build(params)

    assert surf.vol(0.5, 100.0) == pytest.approx(0.20)
    assert surf.vol(1.0, 110.0) == pytest.approx(0.26)


# =============================================================================
# NEW: VolShock unit tests (compatible with FlatVolSurface and GridVolSurface)
# =============================================================================

def test_vol_shock_relative_bumps_flat_surface() -> None:
    vol_id = MarketId("FX", "VOL", "EURUSD")
    base_market = _DummyMarketView(vol_surfaces={vol_id: FlatVolSurface(sigma=0.20)})

    shocked_market = VolShock(
        name="vol_up_10pct",
        vol_id=vol_id,
        bump=0.10,
        bump_mode="relative",
        vol_floor=1e-8,
    ).apply(base_market)

    sigma = shocked_market.vol_surface(vol_id).vol(expiry=1.0, strike=100.0)
    assert sigma == pytest.approx(0.20 * 1.10, rel=1e-12, abs=1e-12)


def test_vol_shock_absolute_bumps_flat_surface() -> None:
    vol_id = MarketId("FX", "VOL", "EURUSD")
    base_market = _DummyMarketView(vol_surfaces={vol_id: FlatVolSurface(sigma=0.20)})

    shocked_market = VolShock(
        name="vol_up_1pt",
        vol_id=vol_id,
        bump=0.01,
        bump_mode="absolute",
        vol_floor=1e-8,
    ).apply(base_market)

    sigma = shocked_market.vol_surface(vol_id).vol(expiry=2.0, strike=123.0)
    assert sigma == pytest.approx(0.21, rel=1e-12, abs=1e-12)


def test_vol_shock_applies_to_grid_surface_and_leaves_other_surfaces_unchanged() -> None:
    vol_id_a = MarketId("FX", "VOL", "EURUSD")
    vol_id_b = MarketId("FX", "VOL", "GBPUSD")

    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([100.0, 110.0], dtype=float)
    vols_a = np.array([[0.20, 0.21], [0.25, 0.26]], dtype=float)

    base_market = _DummyMarketView(
        vol_surfaces={
            vol_id_a: GridVolSurface(expiries=expiries, strikes=strikes, implied_vols=vols_a),
            vol_id_b: FlatVolSurface(sigma=0.30),
        }
    )

    shocked_market = VolShock(
        name="eurusd_vol_down_5pct",
        vol_id=vol_id_a,
        bump=-0.05,
        bump_mode="relative",
        vol_floor=1e-8,
    ).apply(base_market)

    # EURUSD surface is bumped (exact grid point 0.26 -> 0.247)
    sigma_a = shocked_market.vol_surface(vol_id_a).vol(expiry=1.0, strike=110.0)
    assert sigma_a == pytest.approx(0.26 * 0.95, rel=1e-12, abs=1e-12)

    # GBPUSD surface is unchanged
    sigma_b = shocked_market.vol_surface(vol_id_b).vol(expiry=1.0, strike=110.0)
    assert sigma_b == pytest.approx(0.30, rel=1e-12, abs=1e-12)


def test_vol_shock_floor_prevents_non_positive_vol() -> None:
    vol_id = MarketId("FX", "VOL", "EURUSD")
    base_market = _DummyMarketView(vol_surfaces={vol_id: FlatVolSurface(sigma=0.02)})

    shocked_market = VolShock(
        name="vol_down_extreme",
        vol_id=vol_id,
        bump=-10.0,           # extreme relative shock -> would go negative without floor
        bump_mode="relative",
        vol_floor=1e-6,
    ).apply(base_market)

    sigma = shocked_market.vol_surface(vol_id).vol(expiry=1.0, strike=100.0)
    assert sigma == pytest.approx(1e-6, rel=0.0, abs=0.0)