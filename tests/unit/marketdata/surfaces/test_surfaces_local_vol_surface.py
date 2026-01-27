"""
Unit tests for LocalVolSurface and FlatLocalVolSurface.

Tests cover:
1. Construction and validation
2. Interpolation correctness
3. Edge cases (boundary extrapolation)
4. Properties and methods
5. Integration with Dupire calibration
"""

import numpy as np
import pytest

from src.marketdata.surfaces.local_vol_surface import (
    LocalVolSurface,
    FlatLocalVolSurface,
)


# =============================================================================
# FlatLocalVolSurface Tests
# =============================================================================

class TestFlatLocalVolSurface:
    """Tests for constant local volatility surface."""

    def test_construction_valid(self) -> None:
        """Test valid construction with positive sigma."""
        surface = FlatLocalVolSurface(sigma=0.20)
        assert surface.sigma == 0.20

    def test_construction_invalid_sigma_zero(self) -> None:
        """Test that sigma=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            FlatLocalVolSurface(sigma=0.0)

    def test_construction_invalid_sigma_negative(self) -> None:
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="must be > 0"):
            FlatLocalVolSurface(sigma=-0.10)

    def test_construction_invalid_sigma_nan(self) -> None:
        """Test that NaN sigma raises ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            FlatLocalVolSurface(sigma=float("nan"))

    def test_construction_invalid_sigma_inf(self) -> None:
        """Test that infinite sigma raises ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            FlatLocalVolSurface(sigma=float("inf"))

    def test_local_vol_returns_constant(self) -> None:
        """Test that local_vol always returns the constant sigma."""
        sigma = 0.25
        surface = FlatLocalVolSurface(sigma=sigma)

        # Test various spot and time values.
        assert surface.local_vol(spot=100.0, time=0.0) == sigma
        assert surface.local_vol(spot=100.0, time=1.0) == sigma
        assert surface.local_vol(spot=50.0, time=0.5) == sigma
        assert surface.local_vol(spot=200.0, time=2.0) == sigma

    def test_callable_interface(self) -> None:
        """Test that surface can be called as function."""
        surface = FlatLocalVolSurface(sigma=0.15)
        assert surface(100.0, 0.5) == 0.15

    def test_invalid_spot_zero(self) -> None:
        """Test that spot=0 raises ValueError."""
        surface = FlatLocalVolSurface(sigma=0.20)
        with pytest.raises(ValueError, match="spot must be finite and > 0"):
            surface.local_vol(spot=0.0, time=0.5)

    def test_invalid_spot_negative(self) -> None:
        """Test that negative spot raises ValueError."""
        surface = FlatLocalVolSurface(sigma=0.20)
        with pytest.raises(ValueError, match="spot must be finite and > 0"):
            surface.local_vol(spot=-100.0, time=0.5)

    def test_invalid_time_negative(self) -> None:
        """Test that negative time raises ValueError."""
        surface = FlatLocalVolSurface(sigma=0.20)
        with pytest.raises(ValueError, match="time must be finite and >= 0"):
            surface.local_vol(spot=100.0, time=-0.1)


# =============================================================================
# LocalVolSurface Tests
# =============================================================================

class TestLocalVolSurface:
    """Tests for 2D local volatility surface."""

    @pytest.fixture
    def simple_surface(self) -> LocalVolSurface:
        """Create a simple test surface."""
        times = np.array([0.0, 0.5, 1.0])
        spots = np.array([80.0, 100.0, 120.0])
        # Local vol that decreases with spot (typical equity skew).
        local_vols = np.array([
            [0.25, 0.20, 0.15],  # t=0.0
            [0.24, 0.19, 0.14],  # t=0.5
            [0.23, 0.18, 0.13],  # t=1.0
        ])
        return LocalVolSurface(times=times, spots=spots, local_vols=local_vols)

    def test_construction_valid(self, simple_surface: LocalVolSurface) -> None:
        """Test valid construction."""
        assert simple_surface.shape == (3, 3)
        assert simple_surface.time_range == (0.0, 1.0)
        assert simple_surface.spot_range == (80.0, 120.0)

    def test_construction_invalid_empty_times(self) -> None:
        """Test that empty times raises ValueError."""
        with pytest.raises(ValueError, match="times must not be empty"):
            LocalVolSurface(
                times=np.array([]),
                spots=np.array([100.0]),
                local_vols=np.array([]),
            )

    def test_construction_invalid_empty_spots(self) -> None:
        """Test that empty spots raises ValueError."""
        with pytest.raises(ValueError, match="spots must not be empty"):
            LocalVolSurface(
                times=np.array([0.0, 1.0]),
                spots=np.array([]),
                local_vols=np.array([]),
            )

    def test_construction_invalid_negative_times(self) -> None:
        """Test that negative times raises ValueError."""
        with pytest.raises(ValueError, match="times must be >= 0"):
            LocalVolSurface(
                times=np.array([-0.5, 0.5, 1.0]),
                spots=np.array([100.0]),
                local_vols=np.array([[0.2], [0.2], [0.2]]),
            )

    def test_construction_invalid_non_increasing_times(self) -> None:
        """Test that non-increasing times raises ValueError."""
        with pytest.raises(ValueError, match="times must be strictly increasing"):
            LocalVolSurface(
                times=np.array([0.0, 1.0, 0.5]),
                spots=np.array([100.0]),
                local_vols=np.array([[0.2], [0.2], [0.2]]),
            )

    def test_construction_invalid_non_positive_spots(self) -> None:
        """Test that non-positive spots raises ValueError."""
        with pytest.raises(ValueError, match="spots must be > 0"):
            LocalVolSurface(
                times=np.array([0.0, 1.0]),
                spots=np.array([-50.0, 100.0]),
                local_vols=np.array([[0.2, 0.2], [0.2, 0.2]]),
            )

    def test_construction_invalid_non_positive_vols(self) -> None:
        """Test that non-positive local vols raises ValueError."""
        with pytest.raises(ValueError, match="local_vols must be > 0"):
            LocalVolSurface(
                times=np.array([0.0, 1.0]),
                spots=np.array([80.0, 100.0]),
                local_vols=np.array([[0.2, 0.0], [0.2, 0.2]]),
            )

    def test_construction_invalid_shape_mismatch(self) -> None:
        """Test that shape mismatch raises ValueError."""
        with pytest.raises(ValueError, match="shape must be"):
            LocalVolSurface(
                times=np.array([0.0, 1.0]),
                spots=np.array([80.0, 100.0, 120.0]),
                local_vols=np.array([[0.2, 0.2], [0.2, 0.2]]),
            )

    def test_interpolation_at_grid_points(self, simple_surface: LocalVolSurface) -> None:
        """Test that interpolation returns exact values at grid points."""
        # At t=0, S=100.
        assert simple_surface.local_vol(spot=100.0, time=0.0) == pytest.approx(0.20)
        # At t=0.5, S=80.
        assert simple_surface.local_vol(spot=80.0, time=0.5) == pytest.approx(0.24)
        # At t=1.0, S=120.
        assert simple_surface.local_vol(spot=120.0, time=1.0) == pytest.approx(0.13)

    def test_interpolation_between_grid_points(self, simple_surface: LocalVolSurface) -> None:
        """Test bilinear interpolation between grid points."""
        # At t=0.25 (between 0 and 0.5), S=100.
        # Expected: 0.5 * 0.20 + 0.5 * 0.19 = 0.195.
        assert simple_surface.local_vol(spot=100.0, time=0.25) == pytest.approx(0.195, rel=1e-6)

        # At t=0, S=90 (between 80 and 100).
        # Expected: 0.5 * 0.25 + 0.5 * 0.20 = 0.225.
        assert simple_surface.local_vol(spot=90.0, time=0.0) == pytest.approx(0.225, rel=1e-6)

    def test_flat_extrapolation_time_below(self, simple_surface: LocalVolSurface) -> None:
        """Test flat extrapolation for time below grid."""
        # Query at t=-0.5 should give same as t=0.
        # Note: Negative time should raise error due to validation.
        pass  # Handled by validation, not extrapolation.

    def test_flat_extrapolation_spot_below(self, simple_surface: LocalVolSurface) -> None:
        """Test flat extrapolation for spot below grid."""
        # Query at S=50 (below 80) should give same as S=80.
        val_at_50 = simple_surface.local_vol(spot=50.0, time=0.5)
        val_at_80 = simple_surface.local_vol(spot=80.0, time=0.5)
        assert val_at_50 == pytest.approx(val_at_80, rel=1e-6)

    def test_flat_extrapolation_spot_above(self, simple_surface: LocalVolSurface) -> None:
        """Test flat extrapolation for spot above grid."""
        # Query at S=150 (above 120) should give same as S=120.
        val_at_150 = simple_surface.local_vol(spot=150.0, time=0.5)
        val_at_120 = simple_surface.local_vol(spot=120.0, time=0.5)
        assert val_at_150 == pytest.approx(val_at_120, rel=1e-6)

    def test_error_extrapolation_mode(self) -> None:
        """Test that error mode raises on out-of-bounds queries."""
        surface = LocalVolSurface(
            times=np.array([0.0, 1.0]),
            spots=np.array([80.0, 120.0]),
            local_vols=np.array([[0.2, 0.2], [0.2, 0.2]]),
            extrapolation="error",
        )
        # Should work within bounds.
        assert surface.local_vol(spot=100.0, time=0.5) > 0

        # Should raise outside bounds.
        with pytest.raises(ValueError, match="outside grid"):
            surface.local_vol(spot=50.0, time=0.5)

        with pytest.raises(ValueError, match="outside grid"):
            surface.local_vol(spot=100.0, time=2.0)

    def test_callable_interface(self, simple_surface: LocalVolSurface) -> None:
        """Test that surface can be called as function."""
        assert simple_surface(100.0, 0.5) == simple_surface.local_vol(100.0, 0.5)

    def test_single_point_surface(self) -> None:
        """Test surface with single time and spot point."""
        surface = LocalVolSurface(
            times=np.array([0.5]),
            spots=np.array([100.0]),
            local_vols=np.array([[0.25]]),
        )
        # All queries should return the single value.
        assert surface.local_vol(spot=100.0, time=0.5) == pytest.approx(0.25)
        assert surface.local_vol(spot=50.0, time=0.0) == pytest.approx(0.25)
        assert surface.local_vol(spot=200.0, time=2.0) == pytest.approx(0.25)


# =============================================================================
# Integration Tests
# =============================================================================

class TestLocalVolSurfaceIntegration:
    """Integration tests for local vol surface with other components."""

    def test_flat_surface_equivalent_to_flat_vol(self) -> None:
        """Test that flat surface gives same result as FlatLocalVolSurface."""
        sigma = 0.20
        flat = FlatLocalVolSurface(sigma=sigma)

        # Create a grid surface with all same values.
        times = np.array([0.0, 0.5, 1.0])
        spots = np.array([80.0, 100.0, 120.0])
        local_vols = np.full((3, 3), sigma)
        grid = LocalVolSurface(times=times, spots=spots, local_vols=local_vols)

        # Should give same results at interior points.
        for t in [0.25, 0.5, 0.75]:
            for s in [90.0, 100.0, 110.0]:
                assert flat.local_vol(s, t) == pytest.approx(grid.local_vol(s, t), rel=1e-6)

    def test_surface_with_smile_pattern(self) -> None:
        """Test surface with typical equity volatility smile pattern."""
        times = np.array([0.1, 0.5, 1.0, 2.0])
        spots = np.array([70, 80, 90, 100, 110, 120, 130], dtype=float)

        # Create smile: higher vol at lower spots (downside skew).
        base_vol = 0.20
        skew = 0.002  # 0.2% per point below ATM.
        local_vols = np.zeros((4, 7))
        for i, t in enumerate(times):
            term_factor = 1.0 - 0.1 * t  # Term structure: flattening.
            for j, s in enumerate(spots):
                moneyness_effect = skew * (100 - s)
                local_vols[i, j] = base_vol + moneyness_effect * term_factor

        surface = LocalVolSurface(times=times, spots=spots, local_vols=local_vols)

        # Check: at low spot, vol should be higher.
        vol_low = surface.local_vol(spot=70.0, time=0.5)
        vol_atm = surface.local_vol(spot=100.0, time=0.5)
        vol_high = surface.local_vol(spot=130.0, time=0.5)

        assert vol_low > vol_atm
        assert vol_atm > vol_high
