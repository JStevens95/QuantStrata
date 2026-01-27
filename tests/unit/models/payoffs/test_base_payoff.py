"""
Unit tests for base payoff classes and helper functions.

These tests verify:
- BasePayoff1D base class behavior
- BasePathPayoff1D base class behavior
- Helper functions (_as_float_array, _as_paths_array, _validate_option_type)
- Protocol compliance
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass

from src.models.payoffs.base import (
    BasePayoff1D,
    BasePathPayoff1D,
    _as_float_array,
    _as_paths_array,
    _validate_option_type,
)


# =============================================================================
# Helper Function Tests: _as_float_array
# =============================================================================

class TestAsFloatArray:
    """Test the _as_float_array helper function."""

    def test_scalar_to_array(self) -> None:
        """Test conversion of scalar to float64 array."""
        result = _as_float_array(5.0)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        # Scalar becomes 0-d array
        assert float(result) == 5.0

    def test_int_scalar_to_array(self) -> None:
        """Test conversion of integer scalar to float64 array."""
        result = _as_float_array(5)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert float(result) == 5.0

    def test_float_array_passthrough(self) -> None:
        """Test that float64 array passes through without copy."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _as_float_array(arr)

        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr)

    def test_int_array_conversion(self) -> None:
        """Test conversion of int array to float64."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = _as_float_array(arr)

        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_float32_array_conversion(self) -> None:
        """Test conversion of float32 array to float64."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _as_float_array(arr)

        assert result.dtype == np.float64

    def test_list_to_array(self) -> None:
        """Test conversion of Python list to float64 array."""
        lst = [1.0, 2.0, 3.0]
        result = _as_float_array(lst)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, lst)

    def test_preserves_shape(self) -> None:
        """Test that array shape is preserved."""
        arr_1d = np.array([1.0, 2.0, 3.0])
        arr_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

        result_1d = _as_float_array(arr_1d)
        result_2d = _as_float_array(arr_2d)

        assert result_1d.shape == (3,)
        assert result_2d.shape == (2, 2)


# =============================================================================
# Helper Function Tests: _as_paths_array
# =============================================================================

class TestAsPathsArray:
    """Test the _as_paths_array helper function."""

    def test_valid_2d_array(self) -> None:
        """Test with valid 2D paths array."""
        paths = np.array([
            [100.0, 105.0, 110.0],
            [100.0, 95.0, 90.0],
        ], dtype=np.float64)

        result = _as_paths_array(paths)

        assert result.dtype == np.float64
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, paths)

    def test_int_array_conversion(self) -> None:
        """Test conversion of int paths array to float64."""
        paths = np.array([
            [100, 105, 110],
            [100, 95, 90],
        ], dtype=np.int32)

        result = _as_paths_array(paths)

        assert result.dtype == np.float64

    def test_1d_array_raises(self) -> None:
        """Test that 1D array raises ValueError."""
        arr = np.array([100.0, 105.0, 110.0])

        with pytest.raises(ValueError, match="paths must be a 2D array"):
            _as_paths_array(arr)

    def test_3d_array_raises(self) -> None:
        """Test that 3D array raises ValueError."""
        arr = np.array([[[100.0, 105.0, 110.0]]])

        with pytest.raises(ValueError, match="paths must be a 2D array"):
            _as_paths_array(arr)

    def test_empty_array_raises(self) -> None:
        """Test that empty array raises ValueError."""
        arr = np.array([[]], dtype=np.float64)

        with pytest.raises(ValueError, match="paths must be non-empty"):
            _as_paths_array(arr)

    def test_single_path(self) -> None:
        """Test with single path (1 row)."""
        paths = np.array([[100.0, 105.0, 110.0]], dtype=np.float64)

        result = _as_paths_array(paths)

        assert result.shape == (1, 3)

    def test_single_point_paths(self) -> None:
        """Test with single-point paths (1 column)."""
        paths = np.array([[100.0], [110.0]], dtype=np.float64)

        result = _as_paths_array(paths)

        assert result.shape == (2, 1)


# =============================================================================
# Helper Function Tests: _validate_option_type
# =============================================================================

class TestValidateOptionType:
    """Test the _validate_option_type helper function."""

    def test_valid_call(self) -> None:
        """Test that 'call' is valid."""
        # Should not raise
        _validate_option_type("call")

    def test_valid_put(self) -> None:
        """Test that 'put' is valid."""
        # Should not raise
        _validate_option_type("put")

    def test_invalid_type(self) -> None:
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            _validate_option_type("straddle")  # type: ignore[arg-type]

    def test_invalid_case(self) -> None:
        """Test that wrong case raises ValueError (case-sensitive)."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            _validate_option_type("Call")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            _validate_option_type("PUT")  # type: ignore[arg-type]

    def test_empty_string(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            _validate_option_type("")  # type: ignore[arg-type]


# =============================================================================
# BasePayoff1D Tests
# =============================================================================

class TestBasePayoff1D:
    """Test BasePayoff1D base class behavior."""

    def test_terminal_not_implemented(self) -> None:
        """Test that terminal() raises NotImplementedError in base class."""
        # Create a concrete subclass that doesn't override terminal
        @dataclass(frozen=True, slots=True)
        class DummyPayoff(BasePayoff1D):
            pass

        payoff = DummyPayoff()

        with pytest.raises(NotImplementedError, match="Subclasses must implement terminal"):
            payoff.terminal(np.array([100.0]))

    def test_intrinsic_defaults_to_terminal(self) -> None:
        """Test that intrinsic() defaults to calling terminal()."""
        # Create a concrete subclass that implements terminal
        @dataclass(frozen=True, slots=True)
        class TestPayoff(BasePayoff1D):
            def terminal(self, spot: np.ndarray) -> np.ndarray:
                return spot * 2.0

        payoff = TestPayoff()
        spots = np.array([100.0, 200.0])

        # intrinsic should return same as terminal
        terminal_result = payoff.terminal(spots)
        intrinsic_result = payoff.intrinsic(spots)

        np.testing.assert_array_equal(terminal_result, intrinsic_result)

    def test_is_path_dependent_false(self) -> None:
        """Test that is_path_dependent is False for BasePayoff1D."""
        @dataclass(frozen=True, slots=True)
        class TestPayoff(BasePayoff1D):
            def terminal(self, spot: np.ndarray) -> np.ndarray:
                return spot

        payoff = TestPayoff()

        assert payoff.is_path_dependent is False

    def test_callable_interface(self) -> None:
        """Test that payoff is callable (alias for terminal)."""
        @dataclass(frozen=True, slots=True)
        class TestPayoff(BasePayoff1D):
            def terminal(self, spot: np.ndarray) -> np.ndarray:
                return spot + 10.0

        payoff = TestPayoff()
        spots = np.array([100.0])

        # Direct call should work
        result = payoff(spots)

        np.testing.assert_array_equal(result, [110.0])


# =============================================================================
# BasePathPayoff1D Tests
# =============================================================================

class TestBasePathPayoff1D:
    """Test BasePathPayoff1D base class behavior."""

    def test_terminal_from_paths_not_implemented(self) -> None:
        """Test that terminal_from_paths() raises NotImplementedError in base class."""
        @dataclass(frozen=True, slots=True)
        class DummyPathPayoff(BasePathPayoff1D):
            pass

        payoff = DummyPathPayoff()

        with pytest.raises(NotImplementedError, match="Subclasses must implement terminal_from_paths"):
            payoff.terminal_from_paths(np.array([[100.0, 110.0]]))

    def test_intrinsic_from_paths_defaults_to_terminal(self) -> None:
        """Test that intrinsic_from_paths() defaults to calling terminal_from_paths()."""
        @dataclass(frozen=True, slots=True)
        class TestPathPayoff(BasePathPayoff1D):
            def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
                # Return max of each path
                return np.max(paths, axis=1)

        payoff = TestPathPayoff()
        paths = np.array([
            [100.0, 110.0, 120.0],
            [100.0, 90.0, 80.0],
        ])

        terminal_result = payoff.terminal_from_paths(paths)
        intrinsic_result = payoff.intrinsic_from_paths(paths)

        np.testing.assert_array_equal(terminal_result, intrinsic_result)

    def test_is_path_dependent_true(self) -> None:
        """Test that is_path_dependent is True for BasePathPayoff1D."""
        @dataclass(frozen=True, slots=True)
        class TestPathPayoff(BasePathPayoff1D):
            def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
                return paths[:, -1]

        payoff = TestPathPayoff()

        assert payoff.is_path_dependent is True

    def test_callable_interface(self) -> None:
        """Test that path payoff is callable (alias for terminal_from_paths)."""
        @dataclass(frozen=True, slots=True)
        class TestPathPayoff(BasePathPayoff1D):
            def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
                return paths[:, -1]

        payoff = TestPathPayoff()
        paths = np.array([[100.0, 110.0, 120.0]])

        # Direct call should work
        result = payoff(paths)

        np.testing.assert_array_equal(result, [120.0])


# =============================================================================
# Integration Tests: Concrete Implementation Examples
# =============================================================================

class TestConcretePayoffImplementations:
    """Test that concrete payoff implementations work correctly with base classes."""

    def test_terminal_payoff_implementation(self) -> None:
        """Test a concrete terminal-only payoff implementation."""
        @dataclass(frozen=True, slots=True)
        class SimpleCallPayoff(BasePayoff1D):
            """Simple call payoff for testing."""
            strike: float

            def terminal(self, spot: np.ndarray) -> np.ndarray:
                s = _as_float_array(spot)
                return np.maximum(s - self.strike, 0.0)

        payoff = SimpleCallPayoff(strike=100.0)

        # Test terminal
        spots = np.array([90.0, 100.0, 110.0])
        result = payoff.terminal(spots)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 10.0])

        # Test intrinsic (should equal terminal)
        intrinsic = payoff.intrinsic(spots)
        np.testing.assert_array_equal(result, intrinsic)

        # Test callable
        callable_result = payoff(spots)
        np.testing.assert_array_equal(result, callable_result)

        # Test property
        assert payoff.is_path_dependent is False

    def test_path_payoff_implementation(self) -> None:
        """Test a concrete path-dependent payoff implementation."""
        @dataclass(frozen=True, slots=True)
        class MaxSpotPayoff(BasePathPayoff1D):
            """Payoff based on maximum spot during path."""
            strike: float

            def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
                p = _as_paths_array(paths)
                max_spots = np.max(p, axis=1)
                return np.maximum(max_spots - self.strike, 0.0)

        payoff = MaxSpotPayoff(strike=100.0)

        paths = np.array([
            [100.0, 90.0, 80.0],   # Max=100, payoff=0
            [100.0, 110.0, 90.0],  # Max=110, payoff=10
            [100.0, 120.0, 130.0], # Max=130, payoff=30
        ])

        # Test terminal_from_paths
        result = payoff.terminal_from_paths(paths)
        np.testing.assert_array_almost_equal(result, [0.0, 10.0, 30.0])

        # Test intrinsic (should equal terminal)
        intrinsic = payoff.intrinsic_from_paths(paths)
        np.testing.assert_array_equal(result, intrinsic)

        # Test callable
        callable_result = payoff(paths)
        np.testing.assert_array_equal(result, callable_result)

        # Test property
        assert payoff.is_path_dependent is True


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_as_float_array_with_nan(self) -> None:
        """Test that NaN values are preserved."""
        arr = np.array([1.0, np.nan, 3.0])
        result = _as_float_array(arr)

        assert np.isnan(result[1])

    def test_as_float_array_with_inf(self) -> None:
        """Test that infinite values are preserved."""
        arr = np.array([1.0, np.inf, -np.inf])
        result = _as_float_array(arr)

        assert np.isinf(result[1])
        assert np.isinf(result[2])
        assert result[1] > 0
        assert result[2] < 0

    def test_as_paths_array_large_array(self) -> None:
        """Test with large paths array."""
        n_paths = 10000
        n_steps = 252  # Daily monitoring for 1 year

        paths = np.random.randn(n_paths, n_steps) * 10 + 100
        result = _as_paths_array(paths)

        assert result.shape == (n_paths, n_steps)
        assert result.dtype == np.float64

    def test_as_float_array_0d_array(self) -> None:
        """Test with 0-dimensional array."""
        arr = np.array(5.0)  # 0-d array
        result = _as_float_array(arr)

        assert result.dtype == np.float64
        assert float(result) == 5.0
