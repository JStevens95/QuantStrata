"""
Unit tests for Asian option payoff implementation.

These tests verify:
- Correct computation of arithmetic and geometric averages
- Proper application of vanilla-style payoffs to averages
- Vectorization (handling multiple paths)
- Edge cases (single point, constant paths, etc.)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.payoffs.asian import AsianPayoff
from src.models.payoffs.types import OptionType


class TestAsianPayoffArithmetic:
    """Test arithmetic averaging Asian payoffs."""

    def test_call_payoff_arithmetic_in_the_money(self) -> None:
        """
        Test call payoff when average spot is above strike.

        For arithmetic averaging:
        - Average = (S1 + S2 + ... + Sn) / n
        - Call payoff = max(Average - K, 0)
        """
        # Create payoff: call with strike 100, arithmetic averaging
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="arithmetic")

        # Create paths where average is above strike
        # Path 1: [90, 100, 110] -> average = 100, payoff = max(100 - 100, 0) = 0
        # Path 2: [100, 110, 120] -> average = 110, payoff = max(110 - 100, 0) = 10
        paths = np.array([
            [90.0, 100.0, 110.0],  # Average = 100, payoff = 0
            [100.0, 110.0, 120.0],  # Average = 110, payoff = 10
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Verify shape: should return one payoff per path
        assert result.shape == (2,)
        assert result.dtype == np.float64

        # Verify values
        assert result[0] == pytest.approx(0.0, abs=1e-10)  # Average = 100, at-the-money
        assert result[1] == pytest.approx(10.0, abs=1e-10)  # Average = 110, payoff = 10

    def test_call_payoff_arithmetic_out_of_the_money(self) -> None:
        """Test call payoff when average spot is below strike."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="arithmetic")

        # Paths where average is below strike
        # Path: [80, 85, 90] -> average = 85, payoff = max(85 - 100, 0) = 0
        paths = np.array([
            [80.0, 85.0, 90.0],  # Average = 85, payoff = 0 (out-of-the-money)
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_put_payoff_arithmetic_in_the_money(self) -> None:
        """Test put payoff when average spot is below strike."""
        payoff = AsianPayoff(option_type="put", strike=100.0, averaging_type="arithmetic")

        # Paths where average is below strike
        # Path: [80, 85, 90] -> average = 85, payoff = max(100 - 85, 0) = 15
        paths = np.array([
            [80.0, 85.0, 90.0],  # Average = 85, payoff = 15
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(15.0, abs=1e-10)

    def test_put_payoff_arithmetic_out_of_the_money(self) -> None:
        """Test put payoff when average spot is above strike."""
        payoff = AsianPayoff(option_type="put", strike=100.0, averaging_type="arithmetic")

        # Paths where average is above strike
        # Path: [100, 110, 120] -> average = 110, payoff = max(100 - 110, 0) = 0
        paths = np.array([
            [100.0, 110.0, 120.0],  # Average = 110, payoff = 0
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_arithmetic_average_single_point(self) -> None:
        """Test arithmetic average with single monitoring point (path = [S0])."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="arithmetic")

        # Single point path: average = S0
        paths = np.array([
            [100.0],  # Average = 100, payoff = 0
            [110.0],  # Average = 110, payoff = 10
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(10.0, abs=1e-10)

    def test_arithmetic_average_vectorization(self) -> None:
        """Test that payoff correctly handles multiple paths (vectorization)."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="arithmetic")

        # Create many paths with different averages
        n_paths = 1000
        paths = np.random.rand(n_paths, 10) * 200.0  # Random paths

        result = payoff.terminal_from_paths(paths)

        # Verify shape
        assert result.shape == (n_paths,)
        assert result.dtype == np.float64

        # Verify all payoffs are non-negative (call payoff property)
        assert np.all(result >= 0.0)


class TestAsianPayoffGeometric:
    """Test geometric averaging Asian payoffs."""

    def test_call_payoff_geometric_in_the_money(self) -> None:
        """
        Test call payoff with geometric averaging.

        For geometric averaging:
        - Average = (S1 * S2 * ... * Sn)^(1/n)
        - Call payoff = max(Average - K, 0)
        """
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="geometric")

        # Path: [100, 100, 100] -> geometric mean = 100, payoff = 0
        # Path: [100, 110, 121] -> geometric mean = (100*110*121)^(1/3) = 110, payoff = 10
        paths = np.array([
            [100.0, 100.0, 100.0],  # Geometric mean = 100, payoff = 0
            [100.0, 110.0, 121.0],  # Geometric mean = 110, payoff = 10
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(10.0, abs=1e-6)  # Allow small numerical error

    def test_geometric_vs_arithmetic_difference(self) -> None:
        """
        Test that geometric average is less than or equal to arithmetic average.

        This is a fundamental property: geometric mean <= arithmetic mean (Jensen's inequality).
        """
        call_arithmetic = AsianPayoff(option_type="call", strike=0.0, averaging_type="arithmetic")
        call_geometric = AsianPayoff(option_type="call", strike=0.0, averaging_type="geometric")

        # Path with varying values
        paths = np.array([
            [100.0, 50.0, 200.0],  # Arithmetic = 116.67, Geometric = 100
        ], dtype=np.float64)

        arith_result = call_arithmetic.terminal_from_paths(paths)
        geom_result = call_geometric.terminal_from_paths(paths)

        # Geometric average should be <= arithmetic average
        assert geom_result[0] <= arith_result[0]

    def test_geometric_average_single_point(self) -> None:
        """Test geometric average with single monitoring point."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="geometric")

        paths = np.array([
            [100.0],  # Geometric mean = 100, payoff = 0
            [110.0],  # Geometric mean = 110, payoff = 10
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(10.0, abs=1e-10)


class TestAsianPayoffEdgeCases:
    """Test edge cases and validation."""

    def test_invalid_strike(self) -> None:
        """Test that negative or zero strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            AsianPayoff(option_type="call", strike=0.0, averaging_type="arithmetic")

        with pytest.raises(ValueError, match="strike must be > 0"):
            AsianPayoff(option_type="call", strike=-10.0, averaging_type="arithmetic")

    def test_invalid_option_type(self) -> None:
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            AsianPayoff(option_type="invalid", strike=100.0, averaging_type="arithmetic")  # type: ignore[arg-type]

    def test_invalid_averaging_type(self) -> None:
        """Test that invalid averaging type raises ValueError."""
        with pytest.raises(ValueError, match="averaging_type must be 'arithmetic' or 'geometric'"):
            AsianPayoff(option_type="call", strike=100.0, averaging_type="invalid")  # type: ignore[arg-type]

    def test_invalid_paths_shape(self) -> None:
        """Test that non-2D paths array raises ValueError."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="arithmetic")

        # 1D array (should be 2D)
        with pytest.raises(ValueError, match="paths must be a 2D array"):
            payoff.terminal_from_paths(np.array([100.0, 110.0, 120.0]))

        # 3D array (should be 2D)
        with pytest.raises(ValueError, match="paths must be a 2D array"):
            payoff.terminal_from_paths(np.array([[[100.0]]]))

    def test_empty_paths(self) -> None:
        """Test that empty paths array raises ValueError."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="arithmetic")

        with pytest.raises(ValueError, match="paths must be non-empty"):
            payoff.terminal_from_paths(np.array([[]], dtype=np.float64))

    def test_path_dependent_property(self) -> None:
        """Test that Asian payoff is correctly marked as path-dependent."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="arithmetic")

        # Asian options are path-dependent (need full path for average)
        assert payoff.is_path_dependent is True

    def test_constant_path_arithmetic(self) -> None:
        """Test payoff with constant path (all spots equal)."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="arithmetic")

        # Constant path: all values are 100
        paths = np.array([
            [100.0, 100.0, 100.0, 100.0],
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Average = 100, strike = 100, so payoff = 0 (at-the-money)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_constant_path_geometric(self) -> None:
        """Test geometric payoff with constant path."""
        payoff = AsianPayoff(option_type="call", strike=100.0, averaging_type="geometric")

        # Constant path: all values are 100
        paths = np.array([
            [100.0, 100.0, 100.0, 100.0],
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Geometric mean = 100, strike = 100, so payoff = 0
        assert result[0] == pytest.approx(0.0, abs=1e-10)
