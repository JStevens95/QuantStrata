"""
Unit tests for Barrier option payoff implementation.

These tests verify:
- SingleBarrierPayoff (path-dependent, discrete monitoring)
- Up/down barrier directions
- Knock-out/knock-in barrier styles
- Rebate handling
- Validation of input parameters
- Edge cases and path handling
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.payoffs.barrier import SingleBarrierPayoff
from src.models.payoffs.types import OptionType, BarrierDirection, BarrierStyle


# =============================================================================
# Up-and-Out Barrier Tests (barrier_direction="up", barrier_style="knock_out")
# =============================================================================

class TestUpAndOutBarrier:
    """Test up-and-out barrier payoff."""

    def test_up_out_call_barrier_not_hit(self) -> None:
        """
        Test up-and-out call when barrier is NOT hit.

        When barrier is not hit:
        - Knock-out option pays vanilla payoff
        """
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=0.0,
        )

        # Path that stays below barrier (max = 115 < 120)
        # S0=100, ..., ST=110 (in the money)
        paths = np.array([
            [100.0, 105.0, 110.0, 115.0, 110.0],  # Max=115, not hit
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Barrier not hit, so vanilla call payoff: max(110 - 100, 0) = 10
        assert result[0] == pytest.approx(10.0, abs=1e-10)

    def test_up_out_call_barrier_hit(self) -> None:
        """
        Test up-and-out call when barrier IS hit.

        When barrier is hit:
        - Knock-out option is worthless (pays rebate)
        """
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=5.0,
        )

        # Path that touches barrier (max = 120 >= 120)
        paths = np.array([
            [100.0, 110.0, 120.0, 115.0, 130.0],  # Max=130 >= 120, hit
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Barrier hit, so pays rebate
        assert result[0] == pytest.approx(5.0, abs=1e-10)

    def test_up_out_call_barrier_exactly_at_level(self) -> None:
        """
        Test up-and-out when path exactly touches barrier level.

        Barrier hit condition is: max(path) >= barrier_level (inclusive).
        """
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=0.0,
        )

        # Path that exactly touches barrier
        paths = np.array([
            [100.0, 110.0, 120.0, 115.0, 110.0],  # Max=120, exactly at barrier
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Barrier is hit (inclusive), so pays rebate (0)
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_up_out_put_barrier_not_hit(self) -> None:
        """Test up-and-out put when barrier is not hit."""
        payoff = SingleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=0.0,
        )

        # Path stays below barrier, ends below strike (put ITM)
        paths = np.array([
            [100.0, 95.0, 90.0, 85.0, 80.0],  # Max=100 < 120, ST=80
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Not hit, vanilla put payoff: max(100 - 80, 0) = 20
        assert result[0] == pytest.approx(20.0, abs=1e-10)


# =============================================================================
# Down-and-Out Barrier Tests (barrier_direction="down", barrier_style="knock_out")
# =============================================================================

class TestDownAndOutBarrier:
    """Test down-and-out barrier payoff."""

    def test_down_out_call_barrier_not_hit(self) -> None:
        """Test down-and-out call when barrier is NOT hit."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="down",
            barrier_style="knock_out",
            barrier_level=80.0,
            rebate_amount=0.0,
        )

        # Path that stays above barrier (min = 90 > 80)
        paths = np.array([
            [100.0, 95.0, 90.0, 110.0, 120.0],  # Min=90, not hit
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Not hit, vanilla call payoff: max(120 - 100, 0) = 20
        assert result[0] == pytest.approx(20.0, abs=1e-10)

    def test_down_out_call_barrier_hit(self) -> None:
        """Test down-and-out call when barrier IS hit."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="down",
            barrier_style="knock_out",
            barrier_level=80.0,
            rebate_amount=3.0,
        )

        # Path that touches barrier (min = 75 <= 80)
        paths = np.array([
            [100.0, 90.0, 75.0, 110.0, 120.0],  # Min=75 <= 80, hit
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Barrier hit, pays rebate
        assert result[0] == pytest.approx(3.0, abs=1e-10)

    def test_down_out_call_barrier_exactly_at_level(self) -> None:
        """Test down-and-out when path exactly touches barrier level."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="down",
            barrier_style="knock_out",
            barrier_level=80.0,
            rebate_amount=0.0,
        )

        # Path exactly touches barrier
        paths = np.array([
            [100.0, 90.0, 80.0, 110.0, 120.0],  # Min=80, exactly at barrier
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Barrier is hit (inclusive), pays rebate
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_down_out_put_barrier_not_hit(self) -> None:
        """Test down-and-out put when barrier is not hit."""
        payoff = SingleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_direction="down",
            barrier_style="knock_out",
            barrier_level=50.0,
            rebate_amount=0.0,
        )

        # Path stays above barrier, ends below strike (put ITM)
        paths = np.array([
            [100.0, 90.0, 80.0, 70.0, 60.0],  # Min=60 > 50, ST=60
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Not hit, vanilla put payoff: max(100 - 60, 0) = 40
        assert result[0] == pytest.approx(40.0, abs=1e-10)


# =============================================================================
# Up-and-In Barrier Tests (barrier_direction="up", barrier_style="knock_in")
# =============================================================================

class TestUpAndInBarrier:
    """Test up-and-in barrier payoff."""

    def test_up_in_call_barrier_hit(self) -> None:
        """
        Test up-and-in call when barrier IS hit.

        When barrier is hit:
        - Knock-in option becomes active and pays vanilla payoff
        """
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_in",
            barrier_level=120.0,
            rebate_amount=0.0,
        )

        # Path that touches barrier
        paths = np.array([
            [100.0, 110.0, 125.0, 115.0, 130.0],  # Max=130 >= 120, hit
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Barrier hit, so vanilla call payoff: max(130 - 100, 0) = 30
        assert result[0] == pytest.approx(30.0, abs=1e-10)

    def test_up_in_call_barrier_not_hit(self) -> None:
        """
        Test up-and-in call when barrier is NOT hit.

        When barrier is not hit:
        - Knock-in option never activates, pays rebate
        """
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_in",
            barrier_level=120.0,
            rebate_amount=2.0,
        )

        # Path that stays below barrier
        paths = np.array([
            [100.0, 105.0, 110.0, 115.0, 119.0],  # Max=119 < 120, not hit
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Not hit, pays rebate
        assert result[0] == pytest.approx(2.0, abs=1e-10)


# =============================================================================
# Down-and-In Barrier Tests (barrier_direction="down", barrier_style="knock_in")
# =============================================================================

class TestDownAndInBarrier:
    """Test down-and-in barrier payoff."""

    def test_down_in_put_barrier_hit(self) -> None:
        """Test down-and-in put when barrier IS hit."""
        payoff = SingleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_direction="down",
            barrier_style="knock_in",
            barrier_level=80.0,
            rebate_amount=0.0,
        )

        # Path that touches barrier
        paths = np.array([
            [100.0, 90.0, 75.0, 85.0, 95.0],  # Min=75 <= 80, hit
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Barrier hit, vanilla put payoff: max(100 - 95, 0) = 5
        assert result[0] == pytest.approx(5.0, abs=1e-10)

    def test_down_in_put_barrier_not_hit(self) -> None:
        """Test down-and-in put when barrier is NOT hit."""
        payoff = SingleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_direction="down",
            barrier_style="knock_in",
            barrier_level=80.0,
            rebate_amount=1.0,
        )

        # Path that stays above barrier
        paths = np.array([
            [100.0, 95.0, 90.0, 85.0, 81.0],  # Min=81 > 80, not hit
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        # Not hit, pays rebate
        assert result[0] == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# Rebate Tests
# =============================================================================

class TestBarrierRebate:
    """Test rebate handling for barrier options."""

    def test_zero_rebate(self) -> None:
        """Test barrier with zero rebate."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=0.0,
        )

        # Path that hits barrier
        paths = np.array([[100.0, 130.0, 110.0]], dtype=np.float64)
        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_positive_rebate(self) -> None:
        """Test barrier with positive rebate."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=10.0,
        )

        # Path that hits barrier
        paths = np.array([[100.0, 130.0, 110.0]], dtype=np.float64)
        result = payoff.terminal_from_paths(paths)

        assert result[0] == pytest.approx(10.0, abs=1e-10)

    def test_rebate_default_is_zero(self) -> None:
        """Test that default rebate is 0."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
        )

        assert payoff.rebate_amount == 0.0


# =============================================================================
# Validation Tests
# =============================================================================

class TestBarrierPayoffValidation:
    """Test input validation for SingleBarrierPayoff."""

    def test_invalid_strike_zero(self) -> None:
        """Test that zero strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            SingleBarrierPayoff(
                option_type="call",
                strike=0.0,
                barrier_direction="up",
                barrier_style="knock_out",
                barrier_level=120.0,
            )

    def test_invalid_strike_negative(self) -> None:
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            SingleBarrierPayoff(
                option_type="call",
                strike=-10.0,
                barrier_direction="up",
                barrier_style="knock_out",
                barrier_level=120.0,
            )

    def test_invalid_option_type(self) -> None:
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            SingleBarrierPayoff(
                option_type="invalid",  # type: ignore[arg-type]
                strike=100.0,
                barrier_direction="up",
                barrier_style="knock_out",
                barrier_level=120.0,
            )

    def test_invalid_barrier_direction(self) -> None:
        """Test that invalid barrier direction raises ValueError."""
        with pytest.raises(ValueError, match="barrier_direction must be 'up' or 'down'"):
            SingleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_direction="sideways",  # type: ignore[arg-type]
                barrier_style="knock_out",
                barrier_level=120.0,
            )

    def test_invalid_barrier_style(self) -> None:
        """Test that invalid barrier style raises ValueError."""
        with pytest.raises(ValueError, match="barrier_style must be 'knock_out' or 'knock_in'"):
            SingleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_direction="up",
                barrier_style="knock_through",  # type: ignore[arg-type]
                barrier_level=120.0,
            )

    def test_invalid_barrier_level_zero(self) -> None:
        """Test that zero barrier level raises ValueError."""
        with pytest.raises(ValueError, match="barrier_level must be > 0"):
            SingleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_direction="up",
                barrier_style="knock_out",
                barrier_level=0.0,
            )

    def test_invalid_barrier_level_negative(self) -> None:
        """Test that negative barrier level raises ValueError."""
        with pytest.raises(ValueError, match="barrier_level must be > 0"):
            SingleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_direction="up",
                barrier_style="knock_out",
                barrier_level=-10.0,
            )

    def test_invalid_rebate_infinite(self) -> None:
        """Test that infinite rebate raises ValueError."""
        with pytest.raises(ValueError, match="rebate_amount must be finite"):
            SingleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_direction="up",
                barrier_style="knock_out",
                barrier_level=120.0,
                rebate_amount=np.inf,
            )

    def test_invalid_rebate_negative(self) -> None:
        """Test that negative rebate raises ValueError."""
        with pytest.raises(ValueError, match="rebate_amount must be >= 0"):
            SingleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_direction="up",
                barrier_style="knock_out",
                barrier_level=120.0,
                rebate_amount=-1.0,
            )


# =============================================================================
# Path Handling Tests
# =============================================================================

class TestBarrierPathHandling:
    """Test path array handling for barrier payoffs."""

    def test_invalid_paths_1d(self) -> None:
        """Test that 1D paths array raises ValueError."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
        )

        with pytest.raises(ValueError, match="paths must be a 2D array"):
            payoff.terminal_from_paths(np.array([100.0, 110.0, 120.0]))

    def test_invalid_paths_3d(self) -> None:
        """Test that 3D paths array raises ValueError."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
        )

        with pytest.raises(ValueError, match="paths must be a 2D array"):
            payoff.terminal_from_paths(np.array([[[100.0, 110.0]]]))

    def test_empty_paths(self) -> None:
        """Test that empty paths raises ValueError."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
        )

        with pytest.raises(ValueError, match="paths must be non-empty"):
            payoff.terminal_from_paths(np.array([[]], dtype=np.float64))

    def test_single_point_path(self) -> None:
        """Test barrier with single-point path (just S0)."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=0.0,
        )

        # Single point path - S0 only
        paths = np.array([[110.0]], dtype=np.float64)  # S0=110, below barrier
        result = payoff.terminal_from_paths(paths)

        # Not hit, vanilla call: max(110 - 100, 0) = 10
        assert result[0] == pytest.approx(10.0, abs=1e-10)


# =============================================================================
# Vectorization Tests
# =============================================================================

class TestBarrierVectorization:
    """Test vectorization and multiple path handling."""

    def test_multiple_paths(self) -> None:
        """Test barrier payoff with multiple paths."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=5.0,
        )

        # Multiple paths with different outcomes
        paths = np.array([
            [100.0, 105.0, 110.0, 115.0, 110.0],  # Not hit, ITM: 10
            [100.0, 110.0, 125.0, 130.0, 115.0],  # Hit: rebate 5
            [100.0, 95.0, 90.0, 85.0, 80.0],      # Not hit, OTM: 0
            [100.0, 110.0, 120.0, 115.0, 125.0],  # Hit at barrier: rebate 5
        ], dtype=np.float64)

        result = payoff.terminal_from_paths(paths)

        assert result.shape == (4,)
        assert result[0] == pytest.approx(10.0, abs=1e-10)  # Not hit, call payoff
        assert result[1] == pytest.approx(5.0, abs=1e-10)   # Hit, rebate
        assert result[2] == pytest.approx(0.0, abs=1e-10)   # Not hit, OTM
        assert result[3] == pytest.approx(5.0, abs=1e-10)   # Hit at barrier

    def test_large_number_of_paths(self) -> None:
        """Test barrier payoff with large number of paths."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=150.0,
            rebate_amount=0.0,
        )

        n_paths = 10000
        n_steps = 50

        # Generate random paths
        np.random.seed(42)
        paths = np.cumsum(np.random.randn(n_paths, n_steps + 1) * 2, axis=1) + 100.0

        result = payoff.terminal_from_paths(paths)

        # Verify shape
        assert result.shape == (n_paths,)

        # Verify all payoffs are non-negative
        assert np.all(result >= 0.0)


# =============================================================================
# Properties Tests
# =============================================================================

class TestBarrierPayoffProperties:
    """Test barrier payoff class properties."""

    def test_is_path_dependent(self) -> None:
        """Test that barrier payoff is path-dependent."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
        )

        # Barrier options require full path to check if barrier was hit
        assert payoff.is_path_dependent is True

    def test_callable_interface(self) -> None:
        """Test that payoff can be called directly."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
        )

        paths = np.array([[100.0, 105.0, 110.0]], dtype=np.float64)

        # Direct call should work (alias for terminal_from_paths)
        result = payoff(paths)

        assert result[0] == pytest.approx(10.0, abs=1e-10)

    def test_intrinsic_equals_terminal(self) -> None:
        """Test intrinsic_from_paths equals terminal_from_paths."""
        payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
        )

        paths = np.array([
            [100.0, 105.0, 110.0],
            [100.0, 125.0, 110.0],
        ], dtype=np.float64)

        terminal = payoff.terminal_from_paths(paths)
        intrinsic = payoff.intrinsic_from_paths(paths)

        np.testing.assert_array_equal(terminal, intrinsic)


# =============================================================================
# Knock-Out / Knock-In Parity Tests
# =============================================================================

class TestBarrierKnockOutKnockInParity:
    """
    Test knock-out / knock-in parity.

    For zero rebate:
    KO + KI = Vanilla (same underlying vanilla payoff)
    """

    def test_up_ko_ki_parity(self) -> None:
        """Test that up-and-out + up-and-in = vanilla (zero rebate)."""
        ko_payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_out",
            barrier_level=120.0,
            rebate_amount=0.0,
        )
        ki_payoff = SingleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_direction="up",
            barrier_style="knock_in",
            barrier_level=120.0,
            rebate_amount=0.0,
        )

        # Various paths
        paths = np.array([
            [100.0, 105.0, 110.0, 115.0, 110.0],  # Not hit
            [100.0, 110.0, 125.0, 130.0, 115.0],  # Hit
            [100.0, 95.0, 90.0, 85.0, 80.0],      # Not hit, OTM
        ], dtype=np.float64)

        ko_result = ko_payoff.terminal_from_paths(paths)
        ki_result = ki_payoff.terminal_from_paths(paths)

        # Vanilla call payoffs for comparison
        terminal_spots = paths[:, -1]
        vanilla = np.maximum(terminal_spots - 100.0, 0.0)

        # KO + KI should equal vanilla
        np.testing.assert_array_almost_equal(ko_result + ki_result, vanilla)

    def test_down_ko_ki_parity(self) -> None:
        """Test that down-and-out + down-and-in = vanilla (zero rebate)."""
        ko_payoff = SingleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_direction="down",
            barrier_style="knock_out",
            barrier_level=80.0,
            rebate_amount=0.0,
        )
        ki_payoff = SingleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_direction="down",
            barrier_style="knock_in",
            barrier_level=80.0,
            rebate_amount=0.0,
        )

        # Various paths
        paths = np.array([
            [100.0, 95.0, 90.0, 85.0, 81.0],  # Not hit
            [100.0, 90.0, 75.0, 85.0, 90.0],  # Hit
            [100.0, 105.0, 110.0, 115.0, 120.0],  # Not hit, OTM
        ], dtype=np.float64)

        ko_result = ko_payoff.terminal_from_paths(paths)
        ki_result = ki_payoff.terminal_from_paths(paths)

        # Vanilla put payoffs for comparison
        terminal_spots = paths[:, -1]
        vanilla = np.maximum(100.0 - terminal_spots, 0.0)

        # KO + KI should equal vanilla
        np.testing.assert_array_almost_equal(ko_result + ki_result, vanilla)
