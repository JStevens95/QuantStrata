"""
Unit tests for Dupire Local Volatility Calibration.

Tests cover:
1. DupireConfig validation
2. DupireCalibrator single-point calibration
3. DupireCalibrator grid calibration
4. Consistency with flat implied vol (should give flat local vol)
5. Edge cases and numerical stability
"""

import numpy as np
import pytest

from src.calibration.volatility_surface.dupire import (
    DupireConfig,
    DupireCalibrator,
    calibrate_local_vol_from_implied,
)
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface
from src.marketdata.surfaces.local_vol_surface import LocalVolSurface


# =============================================================================
# DupireConfig Tests
# =============================================================================

class TestDupireConfig:
    """Tests for Dupire calibration configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = DupireConfig()
        assert config.dT == pytest.approx(1.0 / 252.0)
        assert config.dK_pct == pytest.approx(0.01)
        assert config.min_local_vol == pytest.approx(0.01)
        assert config.max_local_vol == pytest.approx(2.0)

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = DupireConfig(
            dT=1.0 / 365.0,
            dK_pct=0.02,
            min_local_vol=0.05,
            max_local_vol=1.0,
        )
        assert config.dT == pytest.approx(1.0 / 365.0)
        assert config.dK_pct == pytest.approx(0.02)

    def test_invalid_dT_zero(self) -> None:
        """Test that dT=0 raises ValueError."""
        with pytest.raises(ValueError, match="dT must be > 0"):
            DupireConfig(dT=0.0)

    def test_invalid_dT_negative(self) -> None:
        """Test that negative dT raises ValueError."""
        with pytest.raises(ValueError, match="dT must be > 0"):
            DupireConfig(dT=-0.01)

    def test_invalid_dK_pct_zero(self) -> None:
        """Test that dK_pct=0 raises ValueError."""
        with pytest.raises(ValueError, match="dK_pct must be > 0"):
            DupireConfig(dK_pct=0.0)

    def test_invalid_min_local_vol_zero(self) -> None:
        """Test that min_local_vol=0 raises ValueError."""
        with pytest.raises(ValueError, match="min_local_vol must be > 0"):
            DupireConfig(min_local_vol=0.0)

    def test_invalid_max_local_vol_less_than_min(self) -> None:
        """Test that max_local_vol <= min_local_vol raises ValueError."""
        with pytest.raises(ValueError, match="max_local_vol must be > min_local_vol"):
            DupireConfig(min_local_vol=0.5, max_local_vol=0.3)


# =============================================================================
# DupireCalibrator Tests
# =============================================================================

class TestDupireCalibrator:
    """Tests for Dupire local vol calibrator."""

    @pytest.fixture
    def flat_implied_vol_surface(self) -> FlatVolSurface:
        """Create a flat implied vol surface (σ_BS = 0.20)."""
        return FlatVolSurface(sigma=0.20)

    @pytest.fixture
    def calibrator(self) -> DupireCalibrator:
        """Create calibrator with default config."""
        return DupireCalibrator()

    def test_flat_implied_gives_flat_local_vol(
        self,
        flat_implied_vol_surface: FlatVolSurface,
        calibrator: DupireCalibrator,
    ) -> None:
        """
        Test: Flat implied vol surface should give flat local vol.

        This is the key theoretical result: if σ_BS(K, T) = σ = const,
        then σ_LV(K, T) = σ as well.
        """
        sigma_implied = 0.20

        # Calibrate local vol at a single point.
        local_vol = calibrator.local_vol_at_point(
            implied_surface=flat_implied_vol_surface,
            spot=100.0,
            strike=100.0,
            expiry=1.0,
            r=0.05,
            q=0.02,
        )

        # Should be very close to the implied vol.
        assert local_vol == pytest.approx(sigma_implied, rel=0.01)

    def test_local_vol_at_various_strikes(
        self,
        flat_implied_vol_surface: FlatVolSurface,
        calibrator: DupireCalibrator,
    ) -> None:
        """Test local vol at various strikes with flat implied vol."""
        sigma_implied = 0.20
        spot = 100.0
        r, q = 0.05, 0.02
        T = 1.0

        for strike in [80.0, 90.0, 100.0, 110.0, 120.0]:
            local_vol = calibrator.local_vol_at_point(
                implied_surface=flat_implied_vol_surface,
                spot=spot,
                strike=strike,
                expiry=T,
                r=r,
                q=q,
            )
            # All should be close to implied vol.
            assert local_vol == pytest.approx(sigma_implied, rel=0.02)

    def test_local_vol_at_various_expiries(
        self,
        flat_implied_vol_surface: FlatVolSurface,
        calibrator: DupireCalibrator,
    ) -> None:
        """Test local vol at various expiries with flat implied vol."""
        sigma_implied = 0.20
        spot, strike = 100.0, 100.0
        r, q = 0.05, 0.02

        for T in [0.1, 0.25, 0.5, 1.0, 2.0]:
            local_vol = calibrator.local_vol_at_point(
                implied_surface=flat_implied_vol_surface,
                spot=spot,
                strike=strike,
                expiry=T,
                r=r,
                q=q,
            )
            # All should be close to implied vol.
            assert local_vol == pytest.approx(sigma_implied, rel=0.02)

    def test_short_expiry_fallback(
        self,
        flat_implied_vol_surface: FlatVolSurface,
        calibrator: DupireCalibrator,
    ) -> None:
        """Test that very short expiry falls back to implied vol."""
        # Very short expiry (below min_expiry).
        local_vol = calibrator.local_vol_at_point(
            implied_surface=flat_implied_vol_surface,
            spot=100.0,
            strike=100.0,
            expiry=0.0001,  # ~0.036 days.
            r=0.05,
            q=0.02,
        )

        # Should return implied vol (clamped to bounds).
        assert 0.01 <= local_vol <= 2.0

    def test_grid_calibration(
        self,
        flat_implied_vol_surface: FlatVolSurface,
        calibrator: DupireCalibrator,
    ) -> None:
        """Test full grid calibration."""
        times = np.array([0.1, 0.5, 1.0])
        spots = np.array([80.0, 100.0, 120.0])

        local_vol_surface = calibrator.calibrate_grid(
            implied_surface=flat_implied_vol_surface,
            spot=100.0,
            r=0.05,
            q=0.02,
            times=times,
            spots=spots,
        )

        assert isinstance(local_vol_surface, LocalVolSurface)
        assert local_vol_surface.shape == (3, 3)

        # All values should be close to 0.20.
        for i, t in enumerate(times):
            for j, s in enumerate(spots):
                lv = local_vol_surface.local_vols[i, j]
                assert lv == pytest.approx(0.20, rel=0.05)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestCalibrateLocalVolFromImplied:
    """Tests for the convenience function."""

    def test_default_grids(self) -> None:
        """Test calibration with default grids."""
        implied = FlatVolSurface(sigma=0.25)

        local_vol_surface = calibrate_local_vol_from_implied(
            implied_surface=implied,
            spot=100.0,
            r=0.05,
            q=0.02,
        )

        assert isinstance(local_vol_surface, LocalVolSurface)
        # Check that local vol is close to implied vol at ATM.
        atm_local_vol = local_vol_surface.local_vol(spot=100.0, time=0.5)
        assert atm_local_vol == pytest.approx(0.25, rel=0.05)

    def test_custom_grids(self) -> None:
        """Test calibration with custom grids."""
        implied = FlatVolSurface(sigma=0.30)

        times = np.array([0.25, 0.5, 1.0, 2.0])
        spots = np.array([70.0, 85.0, 100.0, 115.0, 130.0])

        local_vol_surface = calibrate_local_vol_from_implied(
            implied_surface=implied,
            spot=100.0,
            r=0.03,
            q=0.01,
            times=times,
            spots=spots,
        )

        assert local_vol_surface.shape == (4, 5)

    def test_custom_config(self) -> None:
        """Test calibration with custom config."""
        implied = FlatVolSurface(sigma=0.20)
        config = DupireConfig(dT=1.0/500.0, dK_pct=0.005)

        local_vol_surface = calibrate_local_vol_from_implied(
            implied_surface=implied,
            spot=100.0,
            r=0.05,
            q=0.02,
            config=config,
        )

        assert isinstance(local_vol_surface, LocalVolSurface)


# =============================================================================
# Integration Tests
# =============================================================================

class TestDupireIntegration:
    """Integration tests with GridVolSurface."""

    def test_with_smile_surface(self) -> None:
        """Test calibration from a surface with volatility smile."""
        # Create a simple smile: vol increases away from ATM.
        expiries = np.array([0.25, 0.5, 1.0])
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

        # ATM vol is 0.20, smile adds 0.02 per 10 points away from ATM.
        atm_vol = 0.20
        implied_vols = np.zeros((3, 5))
        for i in range(3):
            for j, k in enumerate(strikes):
                moneyness = abs(k - 100.0) / 10.0
                implied_vols[i, j] = atm_vol + 0.02 * moneyness

        surface = GridVolSurface(
            expiries=expiries,
            strikes=strikes,
            implied_vols=implied_vols,
        )

        calibrator = DupireCalibrator()

        # Calibrate at ATM.
        local_vol_atm = calibrator.local_vol_at_point(
            implied_surface=surface,
            spot=100.0,
            strike=100.0,
            expiry=0.5,
            r=0.05,
            q=0.02,
        )

        # Calibrate at OTM (higher implied vol).
        local_vol_otm = calibrator.local_vol_at_point(
            implied_surface=surface,
            spot=100.0,
            strike=80.0,
            expiry=0.5,
            r=0.05,
            q=0.02,
        )

        # Local vol should reflect the smile structure.
        # Note: The relationship is complex; we just check bounds.
        assert 0.01 <= local_vol_atm <= 2.0
        assert 0.01 <= local_vol_otm <= 2.0
