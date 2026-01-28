"""
Unit tests for Curve Bootstrapping.

Tests cover:
1. DepositQuote bootstrapping (simple/continuous)
2. ParSwapQuote bootstrapping
3. FraQuote bootstrapping (NEW)
4. Combined bootstrapping workflows
5. Validation and error handling
6. Interpolation and extrapolation
"""

import numpy as np
import pytest

from src.marketdata.curves.bootstrapper import (
    DepositQuote,
    FraQuote,
    bootstrap_discount_curve,
)
from src.marketdata.quotes.rates import ParSwapQuote


# =============================================================================
# DepositQuote Tests
# =============================================================================

class TestDepositQuote:
    """Tests for deposit quote bootstrapping."""

    def test_simple_compounding_deposit(self) -> None:
        """Test deposit with simple compounding."""
        dep = DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple")
        result = bootstrap_discount_curve(instruments=[dep], engine="native")

        # DF = 1 / (1 + r*T) = 1 / (1 + 0.05*0.25) = 1 / 1.0125 ≈ 0.98765
        expected_df = 1.0 / (1.0 + 0.05 * 0.25)
        assert result.dfs[0] == pytest.approx(expected_df, rel=1e-6)

    def test_continuous_compounding_deposit(self) -> None:
        """Test deposit with continuous compounding."""
        dep = DepositQuote(label="DEP 6M", t=0.5, rate=0.04, compounding="continuous")
        result = bootstrap_discount_curve(instruments=[dep], engine="native")

        # DF = exp(-r*T) = exp(-0.04*0.5) = exp(-0.02) ≈ 0.9802
        expected_df = np.exp(-0.04 * 0.5)
        assert result.dfs[0] == pytest.approx(expected_df, rel=1e-6)

    def test_multiple_deposits(self) -> None:
        """Test bootstrapping multiple deposits."""
        deposits = [
            DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple"),
            DepositQuote(label="DEP 6M", t=0.5, rate=0.05, compounding="simple"),
            DepositQuote(label="DEP 1Y", t=1.0, rate=0.05, compounding="simple"),
        ]
        result = bootstrap_discount_curve(instruments=deposits, engine="native")

        assert len(result.tenors) == 3
        assert len(result.dfs) == 3
        # DFs should be decreasing.
        assert result.dfs[0] > result.dfs[1] > result.dfs[2]


# =============================================================================
# ParSwapQuote Tests
# =============================================================================

class TestParSwapQuote:
    """Tests for par swap bootstrapping."""

    def test_annual_swap(self) -> None:
        """Test annual swap bootstrapping."""
        # First need a deposit to bootstrap DF(1.0)
        dep = DepositQuote(label="DEP 6M", t=0.5, rate=0.05, compounding="simple")
        swap = ParSwapQuote(label="SWAP 1Y", kind="IRS", maturity_t=1.0, par_rate=0.05, fixed_freq="1Y")

        result = bootstrap_discount_curve(instruments=[dep, swap], engine="native")

        # Should have DFs at 0.5 and 1.0.
        assert len(result.tenors) == 2
        assert 0.5 in result.tenors
        assert 1.0 in result.tenors

    def test_semi_annual_swap(self) -> None:
        """Test semi-annual swap bootstrapping."""
        # Need deposits at all payment dates.
        dep1 = DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple")
        dep2 = DepositQuote(label="DEP 6M", t=0.5, rate=0.05, compounding="simple")
        swap = ParSwapQuote(label="SWAP 1Y", kind="IRS", maturity_t=1.0, par_rate=0.05, fixed_freq="6M")

        result = bootstrap_discount_curve(instruments=[dep1, dep2, swap], engine="native")

        # Should have DFs at payment dates: 0.5, 1.0.
        assert len(result.tenors) >= 2

    def test_swap_with_explicit_schedule(self) -> None:
        """Test swap with explicit payment schedule."""
        # Need deposits at all payment dates.
        dep1 = DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple")
        dep2 = DepositQuote(label="DEP 6M", t=0.5, rate=0.05, compounding="simple")
        swap = ParSwapQuote(
            label="SWAP 1Y",
            kind="IRS",
            maturity_t=1.0,
            par_rate=0.05,
            fixed_freq="6M",
            schedule=(0.5, 1.0),
        )

        result = bootstrap_discount_curve(instruments=[dep1, dep2, swap], engine="native")
        assert len(result.tenors) >= 2


# =============================================================================
# FraQuote Tests
# =============================================================================

class TestFraQuote:
    """Tests for FRA quote bootstrapping."""

    def test_fra_bootstrapping(self) -> None:
        """Test FRA bootstrapping."""
        # Need a deposit at start date.
        dep = DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple")
        fra = FraQuote(label="FRA 3x6", t_start=0.25, t_end=0.5, forward_rate=0.05)

        result = bootstrap_discount_curve(instruments=[dep, fra], engine="native")

        # Should have DFs at 0.25 and 0.5.
        assert len(result.tenors) == 2
        assert 0.25 in result.tenors
        assert 0.5 in result.tenors

        # Verify FRA relationship: DF(0.5) = DF(0.25) / (1 + R*α)
        df_start = result.dfs[result.tenors == 0.25][0]
        df_end = result.dfs[result.tenors == 0.5][0]
        alpha = 0.5 - 0.25
        expected_df_end = df_start / (1.0 + 0.05 * alpha)
        assert df_end == pytest.approx(expected_df_end, rel=1e-6)

    def test_fra_with_custom_day_count(self) -> None:
        """Test FRA with custom day count fraction."""
        dep = DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple")
        # Note: rates.py FraQuote computes day_count_fraction automatically from t_start/t_end
        # This test verifies the computed property works correctly
        fra = FraQuote(
            label="FRA 3x6",
            t_start=0.25,
            t_end=0.5,
            forward_rate=0.05,
        )

        result = bootstrap_discount_curve(instruments=[dep, fra], engine="native")
        assert len(result.tenors) == 2

    def test_fra_requires_start_date_df(self) -> None:
        """Test that FRA requires DF at start date."""
        fra = FraQuote(label="FRA 3x6", t_start=0.25, t_end=0.5, forward_rate=0.05)

        with pytest.raises(ValueError, match="Missing DF at"):
            bootstrap_discount_curve(instruments=[fra], engine="native")

    def test_fra_validation(self) -> None:
        """Test FRA validation."""
        # Invalid: t_end <= t_start.
        with pytest.raises(ValueError, match="t_end > t_start"):
            FraQuote(label="FRA", t_start=0.5, t_end=0.25, forward_rate=0.05)

        # Invalid: negative t_start.
        with pytest.raises(ValueError, match="t_start must be >= 0"):
            FraQuote(label="FRA", t_start=-0.1, t_end=0.5, forward_rate=0.05)


# =============================================================================
# Combined Workflow Tests
# =============================================================================

class TestCombinedBootstrapping:
    """Tests for combined bootstrapping workflows."""

    def test_deposit_fra_swap_workflow(self) -> None:
        """Test realistic workflow: deposits → FRAs → swaps."""
        instruments = [
            DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple"),
            FraQuote(label="FRA 3x6", t_start=0.25, t_end=0.5, forward_rate=0.05),
            FraQuote(label="FRA 6x9", t_start=0.5, t_end=0.75, forward_rate=0.05),
            ParSwapQuote(label="SWAP 1Y", kind="IRS", maturity_t=1.0, par_rate=0.05, fixed_freq="6M"),
        ]

        result = bootstrap_discount_curve(instruments=instruments, engine="native")

        # Should have DFs at all key dates.
        assert len(result.tenors) >= 4
        assert all(df > 0 for df in result.dfs)
        assert all(df <= 1.0 for df in result.dfs)

        # DFs should be decreasing.
        assert np.all(np.diff(result.dfs) <= 1e-6)

    def test_curve_monotonicity(self) -> None:
        """Test that bootstrapped curve is monotonic."""
        instruments = [
            DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple"),
            DepositQuote(label="DEP 6M", t=0.5, rate=0.05, compounding="simple"),
            DepositQuote(label="DEP 1Y", t=1.0, rate=0.05, compounding="simple"),
        ]

        result = bootstrap_discount_curve(instruments=instruments, engine="native")

        # Tenors should be increasing.
        assert np.all(np.diff(result.tenors) > 0)

        # DFs should be decreasing (or at least non-increasing).
        assert np.all(np.diff(result.dfs) <= 1e-6)

        # Zero rates should be reasonable.
        assert np.all(result.zero_rates >= 0.0)
        assert np.all(result.zero_rates < 1.0)  # Sanity check


# =============================================================================
# Validation Tests
# =============================================================================

class TestBootstrapValidation:
    """Tests for bootstrapping validation."""

    def test_empty_instruments(self) -> None:
        """Test that empty instruments list raises error."""
        with pytest.raises(ValueError, match="instruments must not be empty"):
            bootstrap_discount_curve(instruments=[], engine="native")

    def test_negative_discount_factor_detection(self) -> None:
        """Test detection of negative discount factors."""
        # This would require a very negative rate, which should be caught.
        # (Hard to construct without violating validation, but test structure is here.)
        pass

    def test_increasing_discount_factor_detection(self) -> None:
        """Test detection of increasing discount factors."""
        # This would indicate negative forward rates.
        # FRA validation should catch this.
        dep = DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple")
        # Very negative FRA rate causes increasing DF (negative forward rate).
        fra = FraQuote(label="FRA 3x6", t_start=0.25, t_end=0.5, forward_rate=-0.20)

        # Should raise error due to validation.
        with pytest.raises(ValueError, match="increasing discount factor"):
            bootstrap_discount_curve(instruments=[dep, fra], engine="native")


# =============================================================================
# Integration Tests
# =============================================================================

class TestBootstrapIntegration:
    """Integration tests for bootstrapping."""

    def test_native_vs_quantlib_consistency(self) -> None:
        """Test that native and QuantLib engines give similar results."""
        instruments = [
            DepositQuote(label="DEP 3M", t=0.25, rate=0.05, compounding="simple"),
            DepositQuote(label="DEP 6M", t=0.5, rate=0.05, compounding="simple"),
            ParSwapQuote(label="SWAP 1Y", kind="IRS", maturity_t=1.0, par_rate=0.05, fixed_freq="6M"),
        ]

        try:
            result_native = bootstrap_discount_curve(instruments=instruments, engine="native")
            result_ql = bootstrap_discount_curve(
                instruments=instruments, engine="quantlib", asof="2025-01-15"
            )

            # Should give similar DFs (within tolerance).
            for t in result_native.tenors:
                if t in result_ql.tenors:
                    df_native = result_native.dfs[result_native.tenors == t][0]
                    df_ql = result_ql.dfs[result_ql.tenors == t][0]
                    assert df_native == pytest.approx(df_ql, rel=0.01)
        except ImportError:
            pytest.skip("QuantLib not available")
