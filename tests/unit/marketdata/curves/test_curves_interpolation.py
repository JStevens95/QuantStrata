"""
Unit tests for Curve Interpolation Methods.

Tests cover:
1. Linear DF interpolation
2. Log-linear DF interpolation (industry standard)
3. Linear zero rate interpolation
4. Cubic spline interpolation
5. Factory function
6. Extrapolation modes
7. Utility functions

Author: QuantStrata Team
"""
import math
import numpy as np
import pytest

from src.marketdata.curves.interpolation import (
    LinearDfInterpolator,
    LogLinearDfInterpolator,
    LinearZeroInterpolator,
    CubicSplineZeroInterpolator,
    create_curve_interpolator,
    df_to_zero_rate,
    zero_rate_to_df,
    forward_rate_from_dfs,
    simple_forward_rate_from_dfs,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_curve_data():
    """Simple test curve with known values."""
    tenors = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    # 5% flat rate: DF = exp(-0.05 * T)
    dfs = np.exp(-0.05 * tenors)
    zero_rates = np.full_like(tenors, 0.05)
    return tenors, dfs, zero_rates


@pytest.fixture
def steep_curve_data():
    """Steeper curve (upward sloping)."""
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0], dtype=float)
    # Upward sloping: 2%, 2.5%, 3%, 3.5%, 4%
    zero_rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04], dtype=float)
    dfs = np.exp(-zero_rates * tenors)
    return tenors, dfs, zero_rates


# =============================================================================
# LinearDfInterpolator Tests
# =============================================================================

class TestLinearDfInterpolator:
    """Tests for linear DF interpolation."""
    
    def test_interpolate_at_nodes(self, simple_curve_data):
        """Test interpolation returns exact values at nodes."""
        tenors, dfs, _ = simple_curve_data
        interp = LinearDfInterpolator(_tenors=tenors, _dfs=dfs)
        
        for t, df in zip(tenors, dfs):
            assert interp(t) == pytest.approx(df, rel=1e-10)
    
    def test_interpolate_between_nodes(self, simple_curve_data):
        """Test linear interpolation between nodes."""
        tenors, dfs, _ = simple_curve_data
        interp = LinearDfInterpolator(_tenors=tenors, _dfs=dfs)
        
        # Midpoint between 0.5 and 1.0
        t_mid = 0.75
        expected = 0.5 * (dfs[1] + dfs[2])  # Linear average
        assert interp(t_mid) == pytest.approx(expected, rel=1e-10)
    
    def test_flat_extrapolation(self, simple_curve_data):
        """Test flat extrapolation at boundaries."""
        tenors, dfs, _ = simple_curve_data
        interp = LinearDfInterpolator(_tenors=tenors, _dfs=dfs, _extrapolation="flat")
        
        # Before first node
        assert interp(0.1) == pytest.approx(dfs[0], rel=1e-10)
        # After last node
        assert interp(3.0) == pytest.approx(dfs[-1], rel=1e-10)
    
    def test_extrapolation_disabled(self, simple_curve_data):
        """Test that extrapolation raises error when disabled."""
        tenors, dfs, _ = simple_curve_data
        interp = LinearDfInterpolator(_tenors=tenors, _dfs=dfs, _extrapolation="none")
        
        with pytest.raises(ValueError, match="Extrapolation disabled"):
            interp(0.1)
        
        with pytest.raises(ValueError, match="Extrapolation disabled"):
            interp(3.0)
    
    def test_vectorized_input(self, simple_curve_data):
        """Test vectorized interpolation."""
        tenors, dfs, _ = simple_curve_data
        interp = LinearDfInterpolator(_tenors=tenors, _dfs=dfs)
        
        t_query = np.array([0.25, 0.5, 0.75, 1.0], dtype=float)
        result = interp(t_query)
        
        assert result.shape == t_query.shape
        assert result[0] == pytest.approx(dfs[0], rel=1e-10)
        assert result[1] == pytest.approx(dfs[1], rel=1e-10)
    
    def test_validation_errors(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="non-empty"):
            LinearDfInterpolator(_tenors=np.array([]), _dfs=np.array([]))
        
        with pytest.raises(ValueError, match="same length"):
            LinearDfInterpolator(_tenors=np.array([0.5, 1.0]), _dfs=np.array([0.99]))
        
        with pytest.raises(ValueError, match="strictly increasing"):
            LinearDfInterpolator(_tenors=np.array([1.0, 0.5]), _dfs=np.array([0.99, 0.98]))
        
        with pytest.raises(ValueError, match="positive"):
            LinearDfInterpolator(_tenors=np.array([0.5, 1.0]), _dfs=np.array([0.99, -0.01]))


# =============================================================================
# LogLinearDfInterpolator Tests
# =============================================================================

class TestLogLinearDfInterpolator:
    """Tests for log-linear DF interpolation (industry standard)."""
    
    def test_interpolate_at_nodes(self, simple_curve_data):
        """Test interpolation returns exact values at nodes."""
        tenors, dfs, _ = simple_curve_data
        interp = LogLinearDfInterpolator(_tenors=tenors, _dfs=dfs)
        
        for t, df in zip(tenors, dfs):
            assert interp(t) == pytest.approx(df, rel=1e-10)
    
    def test_constant_forward_rate(self, simple_curve_data):
        """Test that log-linear produces constant forward rates between nodes."""
        tenors, dfs, _ = simple_curve_data
        interp = LogLinearDfInterpolator(_tenors=tenors, _dfs=dfs)
        
        # Forward rate between 0.5 and 1.0 should be constant
        fwd_01 = interp.forward_rate(0.5, 0.6)
        fwd_02 = interp.forward_rate(0.5, 0.8)
        fwd_03 = interp.forward_rate(0.5, 1.0)
        
        # All should be approximately equal for flat curve
        assert fwd_01 == pytest.approx(fwd_03, rel=1e-6)
        assert fwd_02 == pytest.approx(fwd_03, rel=1e-6)
    
    def test_interpolation_formula(self, steep_curve_data):
        """Test log-linear interpolation formula explicitly."""
        tenors, dfs, _ = steep_curve_data
        interp = LogLinearDfInterpolator(_tenors=tenors, _dfs=dfs)
        
        # Interpolate at 0.75 (between 0.5 and 1.0)
        t = 0.75
        t1, t2 = 0.5, 1.0
        df1, df2 = dfs[1], dfs[2]
        
        # Log-linear formula: DF(t) = DF1 * (DF2/DF1)^((t-t1)/(t2-t1))
        w = (t - t1) / (t2 - t1)  # = 0.5
        expected = df1 * (df2 / df1) ** w
        
        assert interp(t) == pytest.approx(expected, rel=1e-10)
    
    def test_forward_rate_formula(self, steep_curve_data):
        """Test forward rate calculation."""
        tenors, dfs, _ = steep_curve_data
        interp = LogLinearDfInterpolator(_tenors=tenors, _dfs=dfs)
        
        # Forward rate = -ln(DF2/DF1) / (t2 - t1)
        t1, t2 = 0.5, 1.0
        df1, df2 = float(interp(t1)), float(interp(t2))
        expected = -math.log(df2 / df1) / (t2 - t1)
        
        assert interp.forward_rate(t1, t2) == pytest.approx(expected, rel=1e-10)
    
    def test_flat_extrapolation(self, simple_curve_data):
        """Test flat extrapolation."""
        tenors, dfs, _ = simple_curve_data
        interp = LogLinearDfInterpolator(_tenors=tenors, _dfs=dfs, _extrapolation="flat")
        
        assert interp(0.1) == pytest.approx(dfs[0], rel=1e-10)
        assert interp(5.0) == pytest.approx(dfs[-1], rel=1e-10)


# =============================================================================
# LinearZeroInterpolator Tests
# =============================================================================

class TestLinearZeroInterpolator:
    """Tests for linear zero rate interpolation."""
    
    def test_interpolate_at_nodes(self, simple_curve_data):
        """Test interpolation returns exact values at nodes."""
        tenors, _, zero_rates = simple_curve_data
        interp = LinearZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates)
        
        for t, r in zip(tenors, zero_rates):
            assert interp(t) == pytest.approx(r, rel=1e-10)
    
    def test_linear_interpolation(self, steep_curve_data):
        """Test linear interpolation between nodes."""
        tenors, _, zero_rates = steep_curve_data
        interp = LinearZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates)
        
        # Midpoint between 0.5 (2.5%) and 1.0 (3.0%)
        t_mid = 0.75
        expected = 0.5 * (zero_rates[1] + zero_rates[2])  # = 0.0275
        assert interp(t_mid) == pytest.approx(expected, rel=1e-10)
    
    def test_df_conversion(self, steep_curve_data):
        """Test DF computation from zero rates."""
        tenors, dfs, zero_rates = steep_curve_data
        interp = LinearZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates)
        
        # At nodes, DF should match
        for t, df in zip(tenors, dfs):
            assert interp.df(t) == pytest.approx(df, rel=1e-10)
    
    def test_vectorized_input(self, simple_curve_data):
        """Test vectorized interpolation."""
        tenors, _, zero_rates = simple_curve_data
        interp = LinearZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates)
        
        t_query = np.array([0.25, 0.5, 1.0], dtype=float)
        result = interp(t_query)
        
        assert result.shape == t_query.shape
        np.testing.assert_allclose(result, zero_rates[:3], rtol=1e-10)


# =============================================================================
# CubicSplineZeroInterpolator Tests
# =============================================================================

class TestCubicSplineZeroInterpolator:
    """Tests for cubic spline zero rate interpolation."""
    
    def test_interpolate_at_nodes(self, simple_curve_data):
        """Test interpolation returns exact values at nodes."""
        tenors, _, zero_rates = simple_curve_data
        interp = CubicSplineZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates)
        
        for t, r in zip(tenors, zero_rates):
            assert interp(t) == pytest.approx(r, rel=1e-8)
    
    def test_smooth_interpolation(self, steep_curve_data):
        """Test that spline produces smooth curve."""
        tenors, _, zero_rates = steep_curve_data
        interp = CubicSplineZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates)
        
        # Sample many points
        t_fine = np.linspace(tenors[0], tenors[-1], 100)
        r_fine = interp(t_fine)
        
        # Check values are finite and reasonable
        assert np.all(np.isfinite(r_fine))
        assert np.all(r_fine >= 0)  # Non-negative rates (for this upward sloping curve)
    
    def test_forward_rate(self, steep_curve_data):
        """Test forward rate calculation."""
        tenors, _, zero_rates = steep_curve_data
        interp = CubicSplineZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates)
        
        # f(t1, t2) = (r2*t2 - r1*t1) / (t2 - t1)
        t1, t2 = 0.5, 1.0
        r1, r2 = float(interp(t1)), float(interp(t2))
        expected = (r2 * t2 - r1 * t1) / (t2 - t1)
        
        assert interp.forward_rate(t1, t2) == pytest.approx(expected, rel=1e-10)
    
    def test_no_arbitrage_check_passes(self, steep_curve_data):
        """Test that arbitrage check passes for good curve."""
        tenors, _, zero_rates = steep_curve_data
        interp = CubicSplineZeroInterpolator(_tenors=tenors, _zero_rates=zero_rates)
        
        # Should not raise
        interp.check_no_arbitrage(n_points=50)
    
    def test_minimum_points_required(self):
        """Test that at least 2 points are required."""
        with pytest.raises(ValueError, match="At least 2 points"):
            CubicSplineZeroInterpolator(
                _tenors=np.array([0.5]),
                _zero_rates=np.array([0.05])
            )


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateCurveInterpolator:
    """Tests for the factory function."""
    
    def test_linear_df(self, simple_curve_data):
        """Test creating linear DF interpolator."""
        tenors, dfs, _ = simple_curve_data
        interp = create_curve_interpolator(
            tenors=tenors, values=dfs, method="linear_df", value_type="df"
        )
        
        assert isinstance(interp, LinearDfInterpolator)
        assert interp(0.5) == pytest.approx(dfs[1], rel=1e-10)
    
    def test_log_linear_df(self, simple_curve_data):
        """Test creating log-linear DF interpolator."""
        tenors, dfs, _ = simple_curve_data
        interp = create_curve_interpolator(
            tenors=tenors, values=dfs, method="log_linear_df", value_type="df"
        )
        
        assert isinstance(interp, LogLinearDfInterpolator)
    
    def test_linear_zero(self, simple_curve_data):
        """Test creating linear zero rate interpolator."""
        tenors, _, zero_rates = simple_curve_data
        interp = create_curve_interpolator(
            tenors=tenors, values=zero_rates, method="linear_zero", value_type="zero_rate"
        )
        
        assert isinstance(interp, LinearZeroInterpolator)
    
    def test_cubic_spline_zero(self, simple_curve_data):
        """Test creating cubic spline zero rate interpolator."""
        tenors, _, zero_rates = simple_curve_data
        interp = create_curve_interpolator(
            tenors=tenors, values=zero_rates, method="cubic_spline_zero", value_type="zero_rate"
        )
        
        assert isinstance(interp, CubicSplineZeroInterpolator)
    
    def test_convert_zero_to_df(self, simple_curve_data):
        """Test conversion from zero rates to DFs."""
        tenors, dfs, zero_rates = simple_curve_data
        
        # Create log-linear interpolator from zero rates
        interp = create_curve_interpolator(
            tenors=tenors, values=zero_rates, method="log_linear_df", value_type="zero_rate"
        )
        
        # Should produce same DFs at nodes
        for t, df in zip(tenors, dfs):
            assert interp(t) == pytest.approx(df, rel=1e-10)
    
    def test_unknown_method_raises(self, simple_curve_data):
        """Test that unknown method raises error."""
        tenors, dfs, _ = simple_curve_data
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            create_curve_interpolator(
                tenors=tenors, values=dfs, method="unknown_method", value_type="df"
            )


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_df_to_zero_rate(self):
        """Test DF to zero rate conversion."""
        # r = -ln(DF) / t
        df = 0.95
        t = 1.0
        expected = -math.log(df) / t
        assert df_to_zero_rate(df=df, t=t) == pytest.approx(expected, rel=1e-10)
    
    def test_df_to_zero_rate_at_zero(self):
        """Test DF to zero rate at t=0."""
        assert df_to_zero_rate(df=1.0, t=0.0) == pytest.approx(0.0, abs=1e-10)
    
    def test_zero_rate_to_df(self):
        """Test zero rate to DF conversion."""
        r = 0.05
        t = 2.0
        expected = math.exp(-r * t)
        assert zero_rate_to_df(r=r, t=t) == pytest.approx(expected, rel=1e-10)
    
    def test_forward_rate_from_dfs(self):
        """Test forward rate calculation from DFs."""
        df1 = 0.98
        df2 = 0.95
        t1 = 0.5
        t2 = 1.0
        expected = -math.log(df2 / df1) / (t2 - t1)
        assert forward_rate_from_dfs(df1=df1, df2=df2, t1=t1, t2=t2) == pytest.approx(expected, rel=1e-10)
    
    def test_simple_forward_rate(self):
        """Test simple (LIBOR-style) forward rate."""
        df1 = 0.98
        df2 = 0.95
        t1 = 0.5
        t2 = 1.0
        expected = (df1 / df2 - 1.0) / (t2 - t1)
        assert simple_forward_rate_from_dfs(df1=df1, df2=df2, t1=t1, t2=t2) == pytest.approx(expected, rel=1e-10)
    
    def test_vectorized_conversions(self):
        """Test vectorized conversions."""
        dfs = np.array([0.99, 0.98, 0.95], dtype=float)
        ts = np.array([0.25, 0.5, 1.0], dtype=float)
        
        rates = df_to_zero_rate(df=dfs, t=ts)
        dfs_back = zero_rate_to_df(r=rates, t=ts)
        
        np.testing.assert_allclose(dfs_back, dfs, rtol=1e-10)
    
    def test_forward_rate_invalid_input(self):
        """Test forward rate with invalid inputs."""
        with pytest.raises(ValueError, match="t2 must be > t1"):
            forward_rate_from_dfs(df1=0.98, df2=0.95, t1=1.0, t2=0.5)
        
        with pytest.raises(ValueError, match="positive"):
            forward_rate_from_dfs(df1=-0.98, df2=0.95, t1=0.5, t2=1.0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestInterpolationIntegration:
    """Integration tests comparing different interpolation methods."""
    
    def test_all_methods_agree_at_nodes(self, steep_curve_data):
        """Test that all methods agree at node points (when converted to DFs)."""
        tenors, dfs, zero_rates = steep_curve_data
        
        # DF-based interpolators
        df_methods = ["linear_df", "log_linear_df"]
        df_interpolators = [
            create_curve_interpolator(tenors=tenors, values=dfs, method=m, value_type="df")
            for m in df_methods
        ]
        
        # Zero-rate interpolators (need to convert to DF for comparison)
        zr_methods = ["linear_zero", "cubic_spline_zero"]
        zr_interpolators = [
            create_curve_interpolator(tenors=tenors, values=zero_rates, method=m, value_type="zero_rate")
            for m in zr_methods
        ]
        
        # At nodes, all DF interpolators should agree
        for t, df in zip(tenors, dfs):
            for interp in df_interpolators:
                assert interp(t) == pytest.approx(df, rel=1e-6)
        
        # At nodes, all zero rate interpolators should agree
        for t, zr in zip(tenors, zero_rates):
            for interp in zr_interpolators:
                assert interp(t) == pytest.approx(zr, rel=1e-6)
    
    def test_log_linear_preserves_monotonicity(self, steep_curve_data):
        """Test that log-linear interpolation preserves monotonicity."""
        tenors, dfs, _ = steep_curve_data
        interp = create_curve_interpolator(
            tenors=tenors, values=dfs, method="log_linear_df", value_type="df"
        )
        
        # Sample fine grid
        t_fine = np.linspace(tenors[0], tenors[-1], 100)
        df_fine = np.array([interp(t) for t in t_fine])
        
        # DFs should be monotonically decreasing
        assert np.all(np.diff(df_fine) <= 0)
    
    def test_round_trip_zero_to_df(self, steep_curve_data):
        """Test round-trip: zero rates -> DFs -> zero rates."""
        tenors, _, zero_rates = steep_curve_data
        
        # Start with zero rates
        interp = create_curve_interpolator(
            tenors=tenors, values=zero_rates, method="linear_zero", value_type="zero_rate"
        )
        
        # Get DFs at nodes
        dfs = interp.df(tenors)
        
        # Convert back to zero rates
        rates_back = df_to_zero_rate(df=dfs, t=tenors)
        
        np.testing.assert_allclose(rates_back, zero_rates, rtol=1e-10)
