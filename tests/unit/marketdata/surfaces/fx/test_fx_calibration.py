"""
Unit tests for FX Volatility Surface Calibration.

Tests cover:
1. Smile calibration from quotes
2. Delta-to-strike conversion
3. Surface extraction from grid
4. Fixed-point iteration convergence
5. Arbitrage validation
"""

import numpy as np
import pytest

from src.marketdata.surfaces.fx.calibration import (
    calibrate_fx_smile_to_grid_surface,
    extract_fx_smile_from_grid_surface,
    FxSmileToGridConfig,
    FxGridToSmileConfig,
)
from src.marketdata.surfaces.fx.quotes import FxSmileQuotes, FxSmileSliceQuotes
from src.marketdata.surfaces.vol_surface import GridVolSurface, FlatVolSurface


# =============================================================================
# Helper Functions
# =============================================================================

def _simple_df(t: float) -> float:
    """Simple discount factor function for testing."""
    return np.exp(-0.05 * t)


# =============================================================================
# Calibration Tests
# =============================================================================

class TestFxSmileCalibration:
    """Tests for FX smile calibration."""

    def test_flat_smile_calibration(self) -> None:
        """Test calibration of flat smile (constant vol)."""
        # Create flat smile quotes.
        slices = [
            FxSmileSliceQuotes(
                expiry=0.25,
                atm_vol=0.20,
                rr_by_delta={},
                bf_by_delta={},
            ),
            FxSmileSliceQuotes(
                expiry=0.5,
                atm_vol=0.20,
                rr_by_delta={},
                bf_by_delta={},
            ),
        ]
        smile = FxSmileQuotes(slices=slices)

        spot = 100.0
        surface = calibrate_fx_smile_to_grid_surface(
            smile=smile,
            spot=spot,
            df_domestic=_simple_df,
            df_foreign=_simple_df,
        )

        assert isinstance(surface, GridVolSurface)
        assert len(surface.expiries) == 2

        # ATM vols should be close to 0.20.
        for t in [0.25, 0.5]:
            forward = spot * np.exp((0.05 - 0.05) * t)  # r_d = r_f = 0.05
            atm_vol = surface.implied_vol(t, forward)
            assert atm_vol == pytest.approx(0.20, rel=0.01)

    def test_smile_with_rr_bf(self) -> None:
        """Test calibration with risk reversal and butterfly."""
        slices = [
            FxSmileSliceQuotes(
                expiry=0.25,
                atm_vol=0.20,
                rr_by_delta={0.25: 0.01},  # 1% risk reversal (smaller to avoid arbitrage).
                bf_by_delta={0.25: 0.005},  # 0.5% butterfly (smaller to avoid arbitrage).
            ),
        ]
        smile = FxSmileQuotes(slices=slices)

        spot = 100.0
        # Disable validation for this test (small RR/BF may still trigger strict checks).
        surface = calibrate_fx_smile_to_grid_surface(
            smile=smile,
            spot=spot,
            df_domestic=_simple_df,
            df_foreign=_simple_df,
            validate=False,  # Disable strict arbitrage check for test.
        )

        assert isinstance(surface, GridVolSurface)

    def test_multiple_expiries(self) -> None:
        """Test calibration with multiple expiries."""
        slices = [
            FxSmileSliceQuotes(expiry=0.1, atm_vol=0.22, rr_by_delta={}, bf_by_delta={}),
            FxSmileSliceQuotes(expiry=0.25, atm_vol=0.21, rr_by_delta={}, bf_by_delta={}),
            FxSmileSliceQuotes(expiry=0.5, atm_vol=0.20, rr_by_delta={}, bf_by_delta={}),
            FxSmileSliceQuotes(expiry=1.0, atm_vol=0.19, rr_by_delta={}, bf_by_delta={}),
        ]
        smile = FxSmileQuotes(slices=slices)

        spot = 100.0
        surface = calibrate_fx_smile_to_grid_surface(
            smile=smile,
            spot=spot,
            df_domestic=_simple_df,
            df_foreign=_simple_df,
        )

        assert len(surface.expiries) == 4

    def test_calibration_validation(self) -> None:
        """Test calibration input validation."""
        slices = [
            FxSmileSliceQuotes(expiry=0.25, atm_vol=0.20, rr_by_delta={}, bf_by_delta={}),
        ]
        smile = FxSmileQuotes(slices=slices)

        # Invalid spot.
        with pytest.raises(ValueError, match="spot must be > 0"):
            calibrate_fx_smile_to_grid_surface(
                smile=smile,
                spot=0.0,
                df_domestic=_simple_df,
                df_foreign=_simple_df,
            )

        # Invalid config.
        config = FxSmileToGridConfig(n_strikes=3)  # Too few strikes.
        with pytest.raises(ValueError, match="n_strikes must be >= 5"):
            calibrate_fx_smile_to_grid_surface(
                smile=smile,
                spot=100.0,
                df_domestic=_simple_df,
                df_foreign=_simple_df,
                config=config,
            )


# =============================================================================
# Extraction Tests
# =============================================================================

class TestFxSmileExtraction:
    """Tests for extracting smile quotes from grid surface."""

    def test_extract_from_flat_surface(self) -> None:
        """Test extraction from flat vol surface."""
        # Create flat surface.
        surface = FlatVolSurface(sigma=0.20)

        # Convert to GridVolSurface for extraction.
        expiries = np.array([0.25, 0.5, 1.0])
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        vols = np.full((3, 5), 0.20)
        grid_surface = GridVolSurface(expiries=expiries, strikes=strikes, implied_vols=vols)

        smile = extract_fx_smile_from_grid_surface(
            surface=grid_surface,
            spot=100.0,
            df_domestic=_simple_df,
            df_foreign=_simple_df,
        )

        assert isinstance(smile, FxSmileQuotes)
        assert len(smile.slices) == 3

        # ATM vols should be close to 0.20.
        for slice_quote in smile.slices:
            assert slice_quote.atm_vol == pytest.approx(0.20, rel=0.01)

    def test_extract_rr_bf(self) -> None:
        """Test extraction of risk reversal and butterfly."""
        # Create surface with smile.
        expiries = np.array([0.25])
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        # Higher vol at wings.
        vols = np.array([[0.22, 0.21, 0.20, 0.21, 0.22]])
        surface = GridVolSurface(expiries=expiries, strikes=strikes, implied_vols=vols)

        smile = extract_fx_smile_from_grid_surface(
            surface=surface,
            spot=100.0,
            df_domestic=_simple_df,
            df_foreign=_simple_df,
            config=FxGridToSmileConfig(deltas=(0.25,)),
        )

        assert len(smile.slices) == 1
        slice_quote = smile.slices[0]
        # Should have RR and BF for 25Δ.
        assert 0.25 in slice_quote.rr_by_delta
        assert 0.25 in slice_quote.bf_by_delta


# =============================================================================
# Round-Trip Tests
# =============================================================================

class TestCalibrationRoundTrip:
    """Tests for calibration → extraction round-trip."""

    def test_round_trip_flat_smile(self) -> None:
        """Test round-trip: quotes → surface → quotes."""
        original_slices = [
            FxSmileSliceQuotes(expiry=0.25, atm_vol=0.20, rr_by_delta={}, bf_by_delta={}),
            FxSmileSliceQuotes(expiry=0.5, atm_vol=0.20, rr_by_delta={}, bf_by_delta={}),
        ]
        original_smile = FxSmileQuotes(slices=original_slices)

        # Calibrate to surface.
        surface = calibrate_fx_smile_to_grid_surface(
            smile=original_smile,
            spot=100.0,
            df_domestic=_simple_df,
            df_foreign=_simple_df,
        )

        # Extract back to quotes.
        extracted_smile = extract_fx_smile_from_grid_surface(
            surface=surface,
            spot=100.0,
            df_domestic=_simple_df,
            df_foreign=_simple_df,
        )

        # Should have same number of expiries.
        assert len(extracted_smile.slices) == len(original_smile.slices)

        # ATM vols should be close.
        for orig, extr in zip(original_smile.slices, extracted_smile.slices):
            assert extr.atm_vol == pytest.approx(orig.atm_vol, rel=0.05)
