"""
Unit tests for Lookback option payoff implementation.

These tests verify:
- Floating strike lookback payoffs (call and put)
- Fixed strike lookback payoffs (call and put)
- Key properties: floating strike always ITM, lookback >= vanilla
- Edge cases and validation
- Vectorization
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.payoffs.lookback import LookbackPayoff
from src.models.payoffs.types import OptionType


# =============================================================================
# Floating Strike Lookback Tests
# =============================================================================

class TestFloatingStrikeLookbackCall:
    """
    Test floating strike lookback call payoff: Payoff = S_T - min(S_t).

    Key property: This is ALWAYS >= 0 (always in-the-money).
    The holder always "buys at the minimum".
    """

    def test_call_payoff_basic(self) -> None:
        """
        Test floating strike call payoff computation.

        Payoff = S_T - min(S_t)
        The holder captures the difference between terminal spot and path minimum.
        """
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        # Path: S0=100, min=80, S_T=110
        # Payoff = 110 - 80 = 30
        paths = np.array([
            [100.0, 90.0, 80.0, 95.0, 110.0],  # min=80, S_T=110, payoff=30
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(30.0, abs=1e-10)

    def test_call_payoff_always_non_negative(self) -> None:
        """
        Test that floating strike call is ALWAYS non-negative.

        Since min(S_t) <= S_T by definition, payoff >= 0 always.
        """
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        # Generate random paths
        np.random.seed(42)
        n_paths = 1000
        n_steps = 50
        paths = np.cumsum(np.random.randn(n_paths, n_steps + 1) * 2, axis=1) + 100.0
        paths = np.maximum(paths, 1.0)  # Keep positive

        result = payoff.terminal_from_paths(paths)

        # All payoffs should be >= 0
        assert np.all(result >= 0.0)

    def test_call_payoff_terminal_equals_min(self) -> None:
        """
        Test when terminal equals minimum (payoff = 0).

        This is the worst-case scenario for a floating strike call.
        """
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        # Path where terminal equals minimum
        paths = np.array([
            [100.0, 90.0, 80.0, 85.0, 80.0],  # min=80, S_T=80, payoff=0
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_call_payoff_constant_path(self) -> None:
        """
        Test floating strike call with constant path.

        If path is constant, S_T = min(S_t), so payoff = 0.
        """
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        # Constant path
        paths = np.array([
            [100.0, 100.0, 100.0, 100.0, 100.0],
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)


class TestFloatingStrikeLookbackPut:
    """
    Test floating strike lookback put payoff: Payoff = max(S_t) - S_T.

    Key property: This is ALWAYS >= 0 (always in-the-money).
    The holder always "sells at the maximum".
    """

    def test_put_payoff_basic(self) -> None:
        """
        Test floating strike put payoff computation.

        Payoff = max(S_t) - S_T
        The holder captures the difference between path maximum and terminal spot.
        """
        payoff = LookbackPayoff(
            option_type="put",
            lookback_type="floating_strike",
        )

        # Path: S0=100, max=120, S_T=90
        # Payoff = 120 - 90 = 30
        paths = np.array([
            [100.0, 110.0, 120.0, 105.0, 90.0],  # max=120, S_T=90, payoff=30
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(30.0, abs=1e-10)

    def test_put_payoff_always_non_negative(self) -> None:
        """
        Test that floating strike put is ALWAYS non-negative.

        Since max(S_t) >= S_T by definition, payoff >= 0 always.
        """
        payoff = LookbackPayoff(
            option_type="put",
            lookback_type="floating_strike",
        )

        # Generate random paths
        np.random.seed(42)
        n_paths = 1000
        n_steps = 50
        paths = np.cumsum(np.random.randn(n_paths, n_steps + 1) * 2, axis=1) + 100.0
        paths = np.maximum(paths, 1.0)

        result = payoff.terminal_from_paths(paths)

        # All payoffs should be >= 0
        assert np.all(result >= 0.0)

    def test_put_payoff_terminal_equals_max(self) -> None:
        """
        Test when terminal equals maximum (payoff = 0).

        This is the worst-case scenario for a floating strike put.
        """
        payoff = LookbackPayoff(
            option_type="put",
            lookback_type="floating_strike",
        )

        # Path where terminal equals maximum
        paths = np.array([
            [100.0, 90.0, 110.0, 105.0, 110.0],  # max=110, S_T=110, payoff=0
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Fixed Strike Lookback Tests
# =============================================================================

class TestFixedStrikeLookbackCall:
    """
    Test fixed strike lookback call payoff: Payoff = max(max(S_t) - K, 0).

    This is an option on the maximum spot price over the path.
    """

    def test_call_payoff_in_the_money(self) -> None:
        """
        Test fixed strike call when max(S_t) > K.

        Payoff = max(S_t) - K
        """
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        # Path: max=120 > K=100
        # Payoff = max(120 - 100, 0) = 20
        paths = np.array([
            [100.0, 110.0, 120.0, 105.0, 90.0],  # max=120
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(20.0, abs=1e-10)

    def test_call_payoff_out_of_the_money(self) -> None:
        """
        Test fixed strike call when max(S_t) < K.

        Payoff = max(max(S_t) - K, 0) = 0
        """
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        # Path: max=95 < K=100
        # Payoff = max(95 - 100, 0) = 0
        paths = np.array([
            [90.0, 85.0, 95.0, 80.0, 85.0],  # max=95
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_call_payoff_at_the_money(self) -> None:
        """
        Test fixed strike call when max(S_t) = K.

        Payoff = max(K - K, 0) = 0
        """
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        # Path: max=100 = K
        paths = np.array([
            [90.0, 95.0, 100.0, 95.0, 90.0],  # max=100
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)


class TestFixedStrikeLookbackPut:
    """
    Test fixed strike lookback put payoff: Payoff = max(K - min(S_t), 0).

    This is an option on the minimum spot price over the path.
    """

    def test_put_payoff_in_the_money(self) -> None:
        """
        Test fixed strike put when K > min(S_t).

        Payoff = K - min(S_t)
        """
        payoff = LookbackPayoff(
            option_type="put",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        # Path: min=80 < K=100
        # Payoff = max(100 - 80, 0) = 20
        paths = np.array([
            [100.0, 90.0, 80.0, 95.0, 110.0],  # min=80
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(20.0, abs=1e-10)

    def test_put_payoff_out_of_the_money(self) -> None:
        """
        Test fixed strike put when K < min(S_t).

        Payoff = max(K - min(S_t), 0) = 0
        """
        payoff = LookbackPayoff(
            option_type="put",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        # Path: min=105 > K=100
        # Payoff = max(100 - 105, 0) = 0
        paths = np.array([
            [110.0, 115.0, 105.0, 120.0, 115.0],  # min=105
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_put_payoff_at_the_money(self) -> None:
        """
        Test fixed strike put when K = min(S_t).

        Payoff = max(K - K, 0) = 0
        """
        payoff = LookbackPayoff(
            option_type="put",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        # Path: min=100 = K
        paths = np.array([
            [110.0, 105.0, 100.0, 105.0, 110.0],  # min=100
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Validation Tests
# =============================================================================

class TestLookbackPayoffValidation:
    """Test input validation for LookbackPayoff."""

    def test_invalid_option_type(self) -> None:
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            LookbackPayoff(
                option_type="invalid",  # type: ignore[arg-type]
                lookback_type="floating_strike",
            )

    def test_invalid_lookback_type(self) -> None:
        """Test that invalid lookback type raises ValueError."""
        with pytest.raises(ValueError, match="lookback_type must be 'floating_strike' or 'fixed_strike'"):
            LookbackPayoff(
                option_type="call",
                lookback_type="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_strike_for_fixed_strike(self) -> None:
        """Test that zero/negative strike raises ValueError for fixed_strike."""
        with pytest.raises(ValueError, match="strike must be > 0 for fixed_strike"):
            LookbackPayoff(
                option_type="call",
                lookback_type="fixed_strike",
                strike=0.0,
            )

        with pytest.raises(ValueError, match="strike must be > 0 for fixed_strike"):
            LookbackPayoff(
                option_type="call",
                lookback_type="fixed_strike",
                strike=-10.0,
            )

    def test_strike_ignored_for_floating_strike(self) -> None:
        """Test that strike is ignored for floating_strike (no error)."""
        # Should not raise even with strike=0 (default)
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )
        assert payoff.strike == 0.0

    def test_invalid_paths_shape(self) -> None:
        """Test that non-2D paths raises ValueError."""
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        # 1D array
        with pytest.raises(ValueError, match="paths must be a 2D array"):
            payoff.terminal_from_paths(np.array([100.0, 110.0, 120.0]))

        # 3D array
        with pytest.raises(ValueError, match="paths must be a 2D array"):
            payoff.terminal_from_paths(np.array([[[100.0]]]))

    def test_empty_paths(self) -> None:
        """Test that empty paths raises ValueError."""
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        with pytest.raises(ValueError, match="paths must be non-empty"):
            payoff.terminal_from_paths(np.array([[]], dtype=np.float64))


# =============================================================================
# Property Tests
# =============================================================================

class TestLookbackPayoffProperties:
    """Test LookbackPayoff class properties."""

    def test_is_path_dependent(self) -> None:
        """Test that lookback payoff is path-dependent."""
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        # Lookback options require full path to compute extrema
        assert payoff.is_path_dependent is True

    def test_callable_interface(self) -> None:
        """Test that payoff can be called directly."""
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        paths = np.array([[100.0, 90.0, 110.0]], dtype=np.float64)

        # Direct call should work (alias for terminal_from_paths)
        result = payoff(paths)

        # min=90, S_T=110, payoff=20
        assert result[0] == pytest.approx(20.0, abs=1e-10)


# =============================================================================
# Vectorization Tests
# =============================================================================

class TestLookbackPayoffVectorization:
    """Test vectorization and multiple path handling."""

    def test_multiple_paths_floating_strike(self) -> None:
        """Test floating strike lookback with multiple paths."""
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        paths = np.array([
            [100.0, 90.0, 80.0, 95.0, 110.0],  # min=80, S_T=110, payoff=30
            [100.0, 100.0, 100.0, 100.0, 100.0],  # min=100, S_T=100, payoff=0
            [100.0, 110.0, 120.0, 130.0, 90.0],  # min=90, S_T=90, payoff=0
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result.shape == (3,)
        assert result[0] == pytest.approx(30.0, abs=1e-10)
        assert result[1] == pytest.approx(0.0, abs=1e-10)
        assert result[2] == pytest.approx(0.0, abs=1e-10)

    def test_multiple_paths_fixed_strike(self) -> None:
        """Test fixed strike lookback with multiple paths."""
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        paths = np.array([
            [100.0, 110.0, 120.0, 115.0, 105.0],  # max=120, payoff=20
            [90.0, 85.0, 95.0, 80.0, 85.0],  # max=95, payoff=0
            [100.0, 100.0, 100.0, 100.0, 100.0],  # max=100, payoff=0
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result.shape == (3,)
        assert result[0] == pytest.approx(20.0, abs=1e-10)
        assert result[1] == pytest.approx(0.0, abs=1e-10)
        assert result[2] == pytest.approx(0.0, abs=1e-10)

    def test_large_number_of_paths(self) -> None:
        """Test lookback payoff with large number of paths."""
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        n_paths = 10000
        n_steps = 50

        # Generate random paths
        np.random.seed(42)
        paths = np.cumsum(np.random.randn(n_paths, n_steps + 1) * 2, axis=1) + 100.0
        paths = np.maximum(paths, 1.0)

        result = payoff.terminal_from_paths(paths)

        # Verify shape
        assert result.shape == (n_paths,)

        # All payoffs should be >= 0 (floating strike property)
        assert np.all(result >= 0.0)

    def test_dtype_consistency(self) -> None:
        """Test that output dtype is float64."""
        payoff = LookbackPayoff(
            option_type="call",
            lookback_type="floating_strike",
        )

        # Input as int array
        paths = np.array([
            [100, 90, 80, 95, 110],
        ])

        result = payoff.terminal_from_paths(paths)

        assert result.dtype == np.float64


# =============================================================================
# Comparison Tests (Lookback vs Vanilla)
# =============================================================================

class TestLookbackVsVanilla:
    """
    Test that lookback options are always >= vanilla options.

    This is a fundamental property: lookback captures optimal timing.
    """

    def test_fixed_strike_call_vs_vanilla(self) -> None:
        """
        Test that fixed strike lookback call >= vanilla call.

        Lookback call: max(max(S_t) - K, 0)
        Vanilla call:  max(S_T - K, 0)

        Since max(S_t) >= S_T, lookback >= vanilla.
        """
        lookback_payoff = LookbackPayoff(
            option_type="call",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        # Generate paths
        np.random.seed(42)
        n_paths = 1000
        n_steps = 50
        paths = np.cumsum(np.random.randn(n_paths, n_steps + 1) * 2, axis=1) + 100.0
        paths = np.maximum(paths, 1.0)

        lookback_result = lookback_payoff.terminal_from_paths(paths)

        # Vanilla payoff: max(S_T - K, 0)
        terminal_spots = paths[:, -1]
        vanilla_result = np.maximum(terminal_spots - 100.0, 0.0)

        # Lookback should be >= vanilla for each path
        assert np.all(lookback_result >= vanilla_result - 1e-10)

    def test_fixed_strike_put_vs_vanilla(self) -> None:
        """
        Test that fixed strike lookback put >= vanilla put.

        Lookback put: max(K - min(S_t), 0)
        Vanilla put:  max(K - S_T, 0)

        Since min(S_t) <= S_T, lookback >= vanilla.
        """
        lookback_payoff = LookbackPayoff(
            option_type="put",
            lookback_type="fixed_strike",
            strike=100.0,
        )

        # Generate paths
        np.random.seed(42)
        n_paths = 1000
        n_steps = 50
        paths = np.cumsum(np.random.randn(n_paths, n_steps + 1) * 2, axis=1) + 100.0
        paths = np.maximum(paths, 1.0)

        lookback_result = lookback_payoff.terminal_from_paths(paths)

        # Vanilla payoff: max(K - S_T, 0)
        terminal_spots = paths[:, -1]
        vanilla_result = np.maximum(100.0 - terminal_spots, 0.0)

        # Lookback should be >= vanilla for each path
        assert np.all(lookback_result >= vanilla_result - 1e-10)
