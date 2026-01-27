"""
Unit tests for Vanilla option payoff implementation.

These tests verify:
- Correct computation of call and put payoffs: max(S-K, 0) and max(K-S, 0)
- Proper handling of scalar and array inputs
- Validation of input parameters (strike, option_type)
- Edge cases (at-the-money, deep in/out of the money)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.payoffs.vanilla import VanillaPayoff
from src.models.payoffs.types import OptionType


class TestVanillaPayoffCall:
    """Test vanilla call payoff: max(S - K, 0)."""

    def test_call_payoff_in_the_money(self) -> None:
        """
        Test call payoff when spot is above strike.

        For a call option:
        - Payoff = max(S - K, 0)
        - When S > K, payoff = S - K
        """
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        # Spot above strike
        spots = np.array([110.0, 120.0, 150.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # Expected payoffs: [10.0, 20.0, 50.0]
        expected = np.array([10.0, 20.0, 50.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_payoff_out_of_the_money(self) -> None:
        """
        Test call payoff when spot is below strike.

        When S < K, payoff = 0 (worthless to exercise).
        """
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        # Spot below strike
        spots = np.array([90.0, 80.0, 50.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # All payoffs should be 0
        expected = np.zeros(3, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_payoff_at_the_money(self) -> None:
        """
        Test call payoff when spot equals strike.

        At-the-money: S = K, so payoff = max(0, 0) = 0.
        """
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        spots = np.array([100.0], dtype=np.float64)
        result = payoff.terminal(spots)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_call_payoff_scalar_input(self) -> None:
        """Test call payoff with scalar input (should be converted to array)."""
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        # Scalar input
        result = payoff.terminal(110.0)

        # Should return array with single element
        assert result.shape == (1,) or result.shape == ()
        assert float(result) == pytest.approx(10.0, abs=1e-10)

    def test_call_payoff_deep_in_the_money(self) -> None:
        """Test call payoff when deeply in-the-money (large positive delta)."""
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        # Very high spot
        spots = np.array([1000.0, 10000.0], dtype=np.float64)
        result = payoff.terminal(spots)

        expected = np.array([900.0, 9900.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)


class TestVanillaPayoffPut:
    """Test vanilla put payoff: max(K - S, 0)."""

    def test_put_payoff_in_the_money(self) -> None:
        """
        Test put payoff when spot is below strike.

        For a put option:
        - Payoff = max(K - S, 0)
        - When S < K, payoff = K - S
        """
        payoff = VanillaPayoff(option_type="put", strike=100.0)

        # Spot below strike
        spots = np.array([90.0, 80.0, 50.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # Expected payoffs: [10.0, 20.0, 50.0]
        expected = np.array([10.0, 20.0, 50.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_put_payoff_out_of_the_money(self) -> None:
        """
        Test put payoff when spot is above strike.

        When S > K, payoff = 0 (worthless to exercise).
        """
        payoff = VanillaPayoff(option_type="put", strike=100.0)

        # Spot above strike
        spots = np.array([110.0, 120.0, 150.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # All payoffs should be 0
        expected = np.zeros(3, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_put_payoff_at_the_money(self) -> None:
        """
        Test put payoff when spot equals strike.

        At-the-money: S = K, so payoff = max(0, 0) = 0.
        """
        payoff = VanillaPayoff(option_type="put", strike=100.0)

        spots = np.array([100.0], dtype=np.float64)
        result = payoff.terminal(spots)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_put_payoff_scalar_input(self) -> None:
        """Test put payoff with scalar input."""
        payoff = VanillaPayoff(option_type="put", strike=100.0)

        result = payoff.terminal(90.0)

        assert float(result) == pytest.approx(10.0, abs=1e-10)

    def test_put_payoff_deep_in_the_money(self) -> None:
        """Test put payoff when deeply in-the-money."""
        payoff = VanillaPayoff(option_type="put", strike=100.0)

        # Very low spot (but positive)
        spots = np.array([10.0, 1.0], dtype=np.float64)
        result = payoff.terminal(spots)

        expected = np.array([90.0, 99.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)


class TestVanillaPayoffVectorization:
    """Test vectorization and array handling."""

    def test_large_array_call(self) -> None:
        """Test call payoff with large array."""
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        n = 10000
        spots = np.random.uniform(50.0, 150.0, n)
        result = payoff.terminal(spots)

        # Verify shape
        assert result.shape == (n,)

        # Verify all payoffs are non-negative
        assert np.all(result >= 0.0)

        # Verify call payoff formula
        expected = np.maximum(spots - 100.0, 0.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_large_array_put(self) -> None:
        """Test put payoff with large array."""
        payoff = VanillaPayoff(option_type="put", strike=100.0)

        n = 10000
        spots = np.random.uniform(50.0, 150.0, n)
        result = payoff.terminal(spots)

        # Verify shape
        assert result.shape == (n,)

        # Verify all payoffs are non-negative
        assert np.all(result >= 0.0)

        # Verify put payoff formula
        expected = np.maximum(100.0 - spots, 0.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_dtype_consistency(self) -> None:
        """Test that output dtype is float64."""
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        # Input as int array
        spots = np.array([90, 100, 110])
        result = payoff.terminal(spots)

        assert result.dtype == np.float64


class TestVanillaPayoffValidation:
    """Test input validation."""

    def test_invalid_strike_zero(self) -> None:
        """Test that zero strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            VanillaPayoff(option_type="call", strike=0.0)

    def test_invalid_strike_negative(self) -> None:
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            VanillaPayoff(option_type="call", strike=-10.0)

    def test_invalid_option_type(self) -> None:
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            VanillaPayoff(option_type="invalid", strike=100.0)  # type: ignore[arg-type]

    def test_invalid_option_type_straddle(self) -> None:
        """Test that 'straddle' (non-existent type) raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            VanillaPayoff(option_type="straddle", strike=100.0)  # type: ignore[arg-type]


class TestVanillaPayoffProperties:
    """Test payoff class properties and methods."""

    def test_is_not_path_dependent(self) -> None:
        """Test that vanilla payoff is not path-dependent."""
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        # Vanilla options only depend on terminal spot, not the path
        assert payoff.is_path_dependent is False

    def test_intrinsic_equals_terminal(self) -> None:
        """Test that intrinsic value equals terminal payoff for vanilla options."""
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        spots = np.array([90.0, 100.0, 110.0], dtype=np.float64)

        terminal = payoff.terminal(spots)
        intrinsic = payoff.intrinsic(spots)

        np.testing.assert_array_equal(terminal, intrinsic)

    def test_callable_interface(self) -> None:
        """Test that payoff can be called directly (alias for terminal)."""
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        spots = np.array([110.0], dtype=np.float64)

        # Direct call should work
        result = payoff(spots)

        assert result[0] == pytest.approx(10.0, abs=1e-10)

    def test_immutability(self) -> None:
        """Test that payoff is immutable (frozen dataclass)."""
        payoff = VanillaPayoff(option_type="call", strike=100.0)

        # Attempting to modify should raise an error
        with pytest.raises(AttributeError):
            payoff.strike = 200.0  # type: ignore[misc]


class TestVanillaPayoffPutCallParity:
    """Test put-call parity relationship."""

    def test_put_call_parity_payoff(self) -> None:
        """
        Test put-call parity at the payoff level.

        At expiry: Call(S,K) - Put(S,K) = S - K

        This is a fundamental relationship in options theory.
        """
        strike = 100.0
        call_payoff = VanillaPayoff(option_type="call", strike=strike)
        put_payoff = VanillaPayoff(option_type="put", strike=strike)

        # Test across range of spots
        spots = np.linspace(50.0, 150.0, 101)

        call_values = call_payoff.terminal(spots)
        put_values = put_payoff.terminal(spots)

        # Call - Put should equal S - K
        difference = call_values - put_values
        expected = spots - strike

        np.testing.assert_array_almost_equal(difference, expected)
