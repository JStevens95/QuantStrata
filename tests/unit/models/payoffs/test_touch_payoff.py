"""
Unit tests for TouchPayoff.

This module tests the touch (binary barrier) payoff implementation, including:
- One-touch behavior (pays if barrier touched)
- No-touch behavior (pays if barrier NOT touched)
- Up/Down barrier directions
- Edge cases and validation
"""

import numpy as np
import pytest

from src.models.payoffs.touch import TouchPayoff


class TestTouchPayoffConstruction:
    """Tests for TouchPayoff construction and validation."""

    def test_valid_one_touch_up(self) -> None:
        """Test valid one-touch up construction."""
        payoff = TouchPayoff(
            touch_style="one_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=1.0,
        )
        assert payoff.touch_style == "one_touch"
        assert payoff.barrier_direction == "up"
        assert payoff.barrier_level == 110.0
        assert payoff.payout_amount == 1.0

    def test_valid_no_touch_down(self) -> None:
        """Test valid no-touch down construction."""
        payoff = TouchPayoff(
            touch_style="no_touch",
            barrier_direction="down",
            barrier_level=90.0,
            payout_amount=100.0,
        )
        assert payoff.touch_style == "no_touch"
        assert payoff.barrier_direction == "down"

    def test_invalid_touch_style_raises(self) -> None:
        """Test that invalid touch style raises ValueError."""
        with pytest.raises(ValueError, match="touch_style must be"):
            TouchPayoff(
                touch_style="invalid",  # type: ignore
                barrier_direction="up",
                barrier_level=110.0,
                payout_amount=1.0,
            )

    def test_invalid_barrier_direction_raises(self) -> None:
        """Test that invalid barrier direction raises ValueError."""
        with pytest.raises(ValueError, match="barrier_direction must be"):
            TouchPayoff(
                touch_style="one_touch",
                barrier_direction="invalid",  # type: ignore
                barrier_level=110.0,
                payout_amount=1.0,
            )

    def test_invalid_barrier_level_raises(self) -> None:
        """Test that non-positive barrier level raises ValueError."""
        with pytest.raises(ValueError, match="barrier_level must be > 0"):
            TouchPayoff(
                touch_style="one_touch",
                barrier_direction="up",
                barrier_level=0.0,
                payout_amount=1.0,
            )

    def test_negative_payout_raises(self) -> None:
        """Test that negative payout raises ValueError."""
        with pytest.raises(ValueError, match="payout_amount must be >= 0"):
            TouchPayoff(
                touch_style="one_touch",
                barrier_direction="up",
                barrier_level=110.0,
                payout_amount=-1.0,
            )

    def test_zero_payout_allowed(self) -> None:
        """Test that zero payout is allowed."""
        payoff = TouchPayoff(
            touch_style="one_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=0.0,
        )
        assert payoff.payout_amount == 0.0


class TestOneTouchUp:
    """Tests for one-touch up barrier payoff."""

    @pytest.fixture
    def one_touch_up(self) -> TouchPayoff:
        """One-touch up: pays 1 if max(path) >= 110."""
        return TouchPayoff(
            touch_style="one_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=1.0,
        )

    def test_path_touches_barrier(self, one_touch_up: TouchPayoff) -> None:
        """Path touches barrier -> pays payout."""
        # Path: 100 -> 105 -> 115 -> 108 (max=115 >= 110)
        paths = np.array([[100.0, 105.0, 115.0, 108.0]])
        payoffs = one_touch_up.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [1.0])

    def test_path_touches_barrier_exactly(self, one_touch_up: TouchPayoff) -> None:
        """Path touches barrier exactly -> pays payout."""
        # Path: 100 -> 105 -> 110 -> 105 (max=110 >= 110)
        paths = np.array([[100.0, 105.0, 110.0, 105.0]])
        payoffs = one_touch_up.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [1.0])

    def test_path_does_not_touch(self, one_touch_up: TouchPayoff) -> None:
        """Path doesn't touch barrier -> pays 0."""
        # Path: 100 -> 105 -> 108 -> 105 (max=108 < 110)
        paths = np.array([[100.0, 105.0, 108.0, 105.0]])
        payoffs = one_touch_up.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])


class TestOneTouchDown:
    """Tests for one-touch down barrier payoff."""

    @pytest.fixture
    def one_touch_down(self) -> TouchPayoff:
        """One-touch down: pays 1 if min(path) <= 90."""
        return TouchPayoff(
            touch_style="one_touch",
            barrier_direction="down",
            barrier_level=90.0,
            payout_amount=1.0,
        )

    def test_path_touches_barrier(self, one_touch_down: TouchPayoff) -> None:
        """Path touches lower barrier -> pays payout."""
        # Path: 100 -> 95 -> 85 -> 92 (min=85 <= 90)
        paths = np.array([[100.0, 95.0, 85.0, 92.0]])
        payoffs = one_touch_down.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [1.0])

    def test_path_touches_barrier_exactly(self, one_touch_down: TouchPayoff) -> None:
        """Path touches barrier exactly -> pays payout."""
        # Path: 100 -> 95 -> 90 -> 95 (min=90 <= 90)
        paths = np.array([[100.0, 95.0, 90.0, 95.0]])
        payoffs = one_touch_down.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [1.0])

    def test_path_does_not_touch(self, one_touch_down: TouchPayoff) -> None:
        """Path doesn't touch lower barrier -> pays 0."""
        # Path: 100 -> 95 -> 92 -> 95 (min=92 > 90)
        paths = np.array([[100.0, 95.0, 92.0, 95.0]])
        payoffs = one_touch_down.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])


class TestNoTouchUp:
    """Tests for no-touch up barrier payoff."""

    @pytest.fixture
    def no_touch_up(self) -> TouchPayoff:
        """No-touch up: pays 1 if max(path) < 110."""
        return TouchPayoff(
            touch_style="no_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=1.0,
        )

    def test_path_stays_below_barrier(self, no_touch_up: TouchPayoff) -> None:
        """Path stays below barrier -> pays payout."""
        # Path: 100 -> 105 -> 108 -> 105 (max=108 < 110)
        paths = np.array([[100.0, 105.0, 108.0, 105.0]])
        payoffs = no_touch_up.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [1.0])

    def test_path_touches_barrier(self, no_touch_up: TouchPayoff) -> None:
        """Path touches barrier -> pays 0."""
        # Path: 100 -> 105 -> 115 -> 108 (max=115 >= 110)
        paths = np.array([[100.0, 105.0, 115.0, 108.0]])
        payoffs = no_touch_up.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])

    def test_path_touches_barrier_exactly(self, no_touch_up: TouchPayoff) -> None:
        """Path touches barrier exactly -> pays 0 (boundary is touched)."""
        # Path: 100 -> 105 -> 110 -> 105 (max=110 >= 110)
        paths = np.array([[100.0, 105.0, 110.0, 105.0]])
        payoffs = no_touch_up.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])


class TestNoTouchDown:
    """Tests for no-touch down barrier payoff."""

    @pytest.fixture
    def no_touch_down(self) -> TouchPayoff:
        """No-touch down: pays 1 if min(path) > 90."""
        return TouchPayoff(
            touch_style="no_touch",
            barrier_direction="down",
            barrier_level=90.0,
            payout_amount=1.0,
        )

    def test_path_stays_above_barrier(self, no_touch_down: TouchPayoff) -> None:
        """Path stays above barrier -> pays payout."""
        # Path: 100 -> 95 -> 92 -> 95 (min=92 > 90)
        paths = np.array([[100.0, 95.0, 92.0, 95.0]])
        payoffs = no_touch_down.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [1.0])

    def test_path_touches_barrier(self, no_touch_down: TouchPayoff) -> None:
        """Path touches lower barrier -> pays 0."""
        # Path: 100 -> 95 -> 85 -> 92 (min=85 <= 90)
        paths = np.array([[100.0, 95.0, 85.0, 92.0]])
        payoffs = no_touch_down.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])


class TestTouchPayoffParity:
    """Test that one-touch + no-touch = payout_amount (touch parity)."""

    def test_touch_parity_up(self) -> None:
        """One-touch up + No-touch up = payout for all paths."""
        one_touch = TouchPayoff(
            touch_style="one_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=1.0,
        )
        no_touch = TouchPayoff(
            touch_style="no_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=1.0,
        )

        paths = np.array([
            [100.0, 105.0, 108.0, 105.0],  # No touch
            [100.0, 105.0, 115.0, 108.0],  # Touch
            [100.0, 110.0, 105.0, 100.0],  # Touch exactly
        ])

        one_touch_payoffs = one_touch.terminal_from_paths(paths)
        no_touch_payoffs = no_touch.terminal_from_paths(paths)

        # Parity: one_touch + no_touch = payout_amount
        np.testing.assert_allclose(one_touch_payoffs + no_touch_payoffs, [1.0, 1.0, 1.0])

    def test_touch_parity_down(self) -> None:
        """One-touch down + No-touch down = payout for all paths."""
        one_touch = TouchPayoff(
            touch_style="one_touch",
            barrier_direction="down",
            barrier_level=90.0,
            payout_amount=1.0,
        )
        no_touch = TouchPayoff(
            touch_style="no_touch",
            barrier_direction="down",
            barrier_level=90.0,
            payout_amount=1.0,
        )

        paths = np.array([
            [100.0, 95.0, 92.0, 95.0],  # No touch
            [100.0, 95.0, 85.0, 92.0],  # Touch
            [100.0, 90.0, 95.0, 100.0],  # Touch exactly
        ])

        one_touch_payoffs = one_touch.terminal_from_paths(paths)
        no_touch_payoffs = no_touch.terminal_from_paths(paths)

        np.testing.assert_allclose(one_touch_payoffs + no_touch_payoffs, [1.0, 1.0, 1.0])


class TestTouchPayoffVectorization:
    """Tests for vectorized computation across multiple paths."""

    @pytest.fixture
    def one_touch_up(self) -> TouchPayoff:
        return TouchPayoff(
            touch_style="one_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=1.0,
        )

    def test_multiple_paths(self, one_touch_up: TouchPayoff) -> None:
        """Test multiple paths processed in one call."""
        paths = np.array([
            [100.0, 105.0, 115.0, 108.0],  # Touch -> 1
            [100.0, 105.0, 108.0, 105.0],  # No touch -> 0
            [100.0, 110.0, 105.0, 100.0],  # Touch exactly -> 1
            [100.0, 109.0, 109.0, 109.0],  # No touch -> 0
        ])
        payoffs = one_touch_up.terminal_from_paths(paths)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(payoffs, expected)

    def test_large_batch(self, one_touch_up: TouchPayoff) -> None:
        """Test with large number of paths."""
        n_paths = 10000
        n_steps = 100
        # Generate paths
        paths = 100 + np.random.randn(n_paths, n_steps) * 5
        
        payoffs = one_touch_up.terminal_from_paths(paths)
        
        assert payoffs.shape == (n_paths,)
        assert payoffs.dtype == np.float64
        # All payoffs should be either 0 or 1
        assert np.all((payoffs == 0.0) | (payoffs == 1.0))


class TestTouchPayoffCustomPayout:
    """Tests for custom payout amounts."""

    def test_custom_payout_one_touch(self) -> None:
        """Test one-touch with custom payout amount."""
        payoff = TouchPayoff(
            touch_style="one_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=1000.0,
        )
        paths_touch = np.array([[100.0, 105.0, 115.0, 108.0]])
        paths_no_touch = np.array([[100.0, 105.0, 108.0, 105.0]])
        
        np.testing.assert_allclose(payoff.terminal_from_paths(paths_touch), [1000.0])
        np.testing.assert_allclose(payoff.terminal_from_paths(paths_no_touch), [0.0])

    def test_custom_payout_no_touch(self) -> None:
        """Test no-touch with custom payout amount."""
        payoff = TouchPayoff(
            touch_style="no_touch",
            barrier_direction="up",
            barrier_level=110.0,
            payout_amount=500.0,
        )
        paths_touch = np.array([[100.0, 105.0, 115.0, 108.0]])
        paths_no_touch = np.array([[100.0, 105.0, 108.0, 105.0]])
        
        np.testing.assert_allclose(payoff.terminal_from_paths(paths_touch), [0.0])
        np.testing.assert_allclose(payoff.terminal_from_paths(paths_no_touch), [500.0])
