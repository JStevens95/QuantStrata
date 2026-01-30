"""
Unit tests for Dupire local volatility calibration with equity parameters.

These tests verify that the generic Dupire infrastructure works correctly
for equity-style inputs (spot price, risk-free rate, dividend yield).

Author: QuantStrata Team
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.volatility_surface.dupire import (
    DupireCalibrator,
    DupireConfig,
)
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface
from src.marketdata.surfaces.local_vol_surface import LocalVolSurface


# =============================================================================
# Tests: Flat Implied Vol (should give constant local vol)
# =============================================================================

def test_dupire_flat_implied_vol_gives_constant_local_vol() -> None:
    """
    For a flat implied vol surface, local vol should be constant and equal
    to implied vol (this is the Black-Scholes case).
    """
    implied_vol = 0.20  # 20% flat vol
    implied_surface = FlatVolSurface(sigma=implied_vol)

    calibrator = DupireCalibrator(config=DupireConfig())

    # Equity parameters
    spot = 100.0
    r = 0.05    # 5% risk-free rate
    q = 0.02    # 2% dividend yield

    # Test at various strikes and expiries
    test_points = [
        (100.0, 1.0),   # ATM, 1Y
        (90.0, 0.5),    # 10% OTM put, 6M
        (110.0, 0.5),   # 10% OTM call, 6M
        (100.0, 0.25),  # ATM, 3M
    ]

    for strike, expiry in test_points:
        local_vol = calibrator.local_vol_at_point(
            implied_surface=implied_surface,
            spot=spot,
            strike=strike,
            expiry=expiry,
            r=r,
            q=q,
        )
        # Local vol should be close to implied vol for flat surface
        assert abs(local_vol - implied_vol) < 0.02, (
            f"Expected local_vol ≈ {implied_vol}, got {local_vol} "
            f"at (K={strike}, T={expiry})"
        )


def test_dupire_calibrate_grid_produces_valid_surface() -> None:
    """DupireCalibrator.calibrate_grid should produce a valid LocalVolSurface."""
    implied_surface = FlatVolSurface(sigma=0.25)
    calibrator = DupireCalibrator(config=DupireConfig())

    # Equity parameters
    spot = 100.0
    r = 0.05
    q = 0.01

    # Define grid
    times = np.array([0.25, 0.5, 1.0], dtype=float)
    spots = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)

    local_vol_surface = calibrator.calibrate_grid(
        implied_surface=implied_surface,
        spot=spot,
        r=r,
        q=q,
        times=times,
        spots=spots,
    )

    # Verify result is a LocalVolSurface
    assert isinstance(local_vol_surface, LocalVolSurface)
    assert local_vol_surface.shape == (len(times), len(spots))

    # All local vols should be positive
    assert np.all(local_vol_surface.local_vols > 0)


# =============================================================================
# Tests: Equity Skew Surface
# =============================================================================

def test_dupire_with_equity_skew() -> None:
    """
    Test Dupire with a typical equity skew surface (higher vol at low strikes).
    """
    # Create a simple equity-style vol surface with skew
    expiries = np.array([0.25, 0.5, 1.0], dtype=float)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)

    # Build implied vol grid with negative skew (higher vol at low strikes)
    base_vol = 0.20
    skew = -0.002  # -0.2% vol per 1% away from ATM

    implied_vols = np.zeros((len(expiries), len(strikes)), dtype=float)
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            # Simple skew model: vol increases as K decreases
            moneyness = (K - 100.0) / 100.0  # Centered at ATM = 100
            implied_vols[i, j] = base_vol + skew * moneyness * 100.0

    # Ensure all vols are positive
    implied_vols = np.maximum(implied_vols, 0.05)

    implied_surface = GridVolSurface(
        expiries=expiries,
        strikes=strikes,
        implied_vols=implied_vols,
    )

    calibrator = DupireCalibrator(config=DupireConfig())

    # Equity parameters
    spot = 100.0
    r = 0.05
    q = 0.02

    # Calibrate local vol at ATM
    local_vol_atm = calibrator.local_vol_at_point(
        implied_surface=implied_surface,
        spot=spot,
        strike=100.0,
        expiry=0.5,
        r=r,
        q=q,
    )

    # Local vol should be finite and positive
    assert np.isfinite(local_vol_atm)
    assert local_vol_atm > 0


# =============================================================================
# Tests: High Dividend Yield
# =============================================================================

def test_dupire_with_high_dividend_yield() -> None:
    """
    Test Dupire calibration with high dividend yield (e.g., 5% yield).
    This is common for certain equity indices or dividend-paying stocks.
    """
    implied_surface = FlatVolSurface(sigma=0.25)
    calibrator = DupireCalibrator(config=DupireConfig())

    # High dividend yield
    spot = 100.0
    r = 0.03    # 3% risk-free rate
    q = 0.05    # 5% dividend yield (higher than r!)

    local_vol = calibrator.local_vol_at_point(
        implied_surface=implied_surface,
        spot=spot,
        strike=100.0,
        expiry=1.0,
        r=r,
        q=q,
    )

    # Should still produce valid local vol
    assert np.isfinite(local_vol)
    assert local_vol > 0
    assert abs(local_vol - 0.25) < 0.05


# =============================================================================
# Tests: Zero Dividend
# =============================================================================

def test_dupire_with_zero_dividend() -> None:
    """
    Test Dupire calibration with zero dividend yield.
    This is the simplest equity case (non-dividend paying stock).
    """
    implied_surface = FlatVolSurface(sigma=0.30)
    calibrator = DupireCalibrator(config=DupireConfig())

    spot = 100.0
    r = 0.05
    q = 0.0  # No dividends

    local_vol = calibrator.local_vol_at_point(
        implied_surface=implied_surface,
        spot=spot,
        strike=100.0,
        expiry=1.0,
        r=r,
        q=q,
    )

    assert np.isfinite(local_vol)
    assert abs(local_vol - 0.30) < 0.02


# =============================================================================
# Tests: LocalVolSurface with Equity Parameters
# =============================================================================

def test_local_vol_surface_interpolation_equity() -> None:
    """Test that LocalVolSurface interpolation works correctly for equity-style grid."""
    # Create a simple equity local vol surface
    times = np.array([0.0, 0.5, 1.0], dtype=float)
    spots = np.array([80.0, 100.0, 120.0], dtype=float)

    # Create local vol grid with some variation
    local_vols = np.array([
        [0.25, 0.20, 0.18],  # t=0.0
        [0.24, 0.19, 0.17],  # t=0.5
        [0.23, 0.18, 0.16],  # t=1.0
    ], dtype=float)

    surface = LocalVolSurface(
        times=times,
        spots=spots,
        local_vols=local_vols,
    )

    # Test interpolation at grid points
    assert surface.local_vol(spot=100.0, time=0.5) == pytest.approx(0.19, rel=1e-10)

    # Test interpolation between grid points
    vol_interp = surface.local_vol(spot=110.0, time=0.75)
    assert 0.15 < vol_interp < 0.25  # Should be reasonable


def test_local_vol_surface_equity_skew_pattern() -> None:
    """Verify equity skew pattern (higher vol at lower spots) works in LocalVolSurface."""
    times = np.array([0.5, 1.0], dtype=float)
    spots = np.array([80.0, 100.0, 120.0], dtype=float)

    # Equity skew: higher vol at lower spots
    local_vols = np.array([
        [0.30, 0.22, 0.18],  # t=0.5
        [0.28, 0.20, 0.16],  # t=1.0
    ], dtype=float)

    surface = LocalVolSurface(times=times, spots=spots, local_vols=local_vols)

    # Verify skew pattern
    vol_low_spot = surface.local_vol(spot=80.0, time=0.5)
    vol_atm = surface.local_vol(spot=100.0, time=0.5)
    vol_high_spot = surface.local_vol(spot=120.0, time=0.5)

    assert vol_low_spot > vol_atm > vol_high_spot


# =============================================================================
# Tests: Edge Cases
# =============================================================================

def test_dupire_short_expiry() -> None:
    """Dupire should handle very short expiries gracefully."""
    implied_surface = FlatVolSurface(sigma=0.20)
    config = DupireConfig(min_expiry=1.0 / 365.0)  # 1 day minimum
    calibrator = DupireCalibrator(config=config)

    spot = 100.0
    r = 0.05
    q = 0.02

    # Very short expiry (below minimum)
    local_vol = calibrator.local_vol_at_point(
        implied_surface=implied_surface,
        spot=spot,
        strike=100.0,
        expiry=0.001,  # ~0.4 days
        r=r,
        q=q,
    )

    # Should fall back to implied vol
    assert abs(local_vol - 0.20) < 0.05


def test_dupire_deep_otm() -> None:
    """Dupire should handle deep OTM options (may be numerically unstable)."""
    implied_surface = FlatVolSurface(sigma=0.20)
    calibrator = DupireCalibrator(config=DupireConfig())

    spot = 100.0
    r = 0.05
    q = 0.02

    # Deep OTM put (K = 50, spot = 100)
    local_vol = calibrator.local_vol_at_point(
        implied_surface=implied_surface,
        spot=spot,
        strike=50.0,  # 50% OTM
        expiry=1.0,
        r=r,
        q=q,
    )

    # Should produce valid result (clamped if needed)
    assert np.isfinite(local_vol)
    assert local_vol > 0
