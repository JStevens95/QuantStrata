"""
Unit tests for DoubleBarrierPayoff.

This module tests the double barrier payoff implementation, including:
- Knock-out behavior (survives in corridor)
- Knock-in behavior (activates on exit)
- Call/Put payoffs
- Rebate handling
- Edge cases and validation
"""

import numpy as np
import pytest

from src.models.payoffs.double_barrier import DoubleBarrierPayoff


class TestDoubleBarrierPayoffConstruction:
    """Tests for DoubleBarrierPayoff construction and validation."""

    def test_valid_knock_out_call(self) -> None:
        """Test valid knock-out call construction."""
        payoff = DoubleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_style="knock_out",
            lower_barrier=90.0,
            upper_barrier=110.0,
        )
        assert payoff.option_type == "call"
        assert payoff.strike == 100.0
        assert payoff.barrier_style == "knock_out"
        assert payoff.lower_barrier == 90.0
        assert payoff.upper_barrier == 110.0
        assert payoff.rebate_amount == 0.0

    def test_valid_knock_in_put_with_rebate(self) -> None:
        """Test valid knock-in put with rebate."""
        payoff = DoubleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_style="knock_in",
            lower_barrier=80.0,
            upper_barrier=120.0,
            rebate_amount=5.0,
        )
        assert payoff.option_type == "put"
        assert payoff.barrier_style == "knock_in"
        assert payoff.rebate_amount == 5.0

    def test_invalid_option_type_raises(self) -> None:
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be"):
            DoubleBarrierPayoff(
                option_type="invalid",  # type: ignore
                strike=100.0,
                barrier_style="knock_out",
                lower_barrier=90.0,
                upper_barrier=110.0,
            )

    def test_invalid_strike_raises(self) -> None:
        """Test that non-positive strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            DoubleBarrierPayoff(
                option_type="call",
                strike=0.0,
                barrier_style="knock_out",
                lower_barrier=90.0,
                upper_barrier=110.0,
            )

    def test_invalid_barrier_style_raises(self) -> None:
        """Test that invalid barrier style raises ValueError."""
        with pytest.raises(ValueError, match="barrier_style must be"):
            DoubleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_style="invalid",  # type: ignore
                lower_barrier=90.0,
                upper_barrier=110.0,
            )

    def test_invalid_lower_barrier_raises(self) -> None:
        """Test that non-positive lower barrier raises ValueError."""
        with pytest.raises(ValueError, match="lower_barrier must be > 0"):
            DoubleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_style="knock_out",
                lower_barrier=0.0,
                upper_barrier=110.0,
            )

    def test_invalid_upper_barrier_raises(self) -> None:
        """Test that non-positive upper barrier raises ValueError."""
        with pytest.raises(ValueError, match="upper_barrier must be > 0"):
            DoubleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_style="knock_out",
                lower_barrier=90.0,
                upper_barrier=0.0,
            )

    def test_lower_barrier_not_less_than_upper_raises(self) -> None:
        """Test that lower_barrier >= upper_barrier raises ValueError."""
        with pytest.raises(ValueError, match="lower_barrier.*must be < upper_barrier"):
            DoubleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_style="knock_out",
                lower_barrier=110.0,  # >= upper
                upper_barrier=100.0,
            )

    def test_equal_barriers_raises(self) -> None:
        """Test that equal barriers raise ValueError."""
        with pytest.raises(ValueError, match="lower_barrier.*must be < upper_barrier"):
            DoubleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_style="knock_out",
                lower_barrier=100.0,
                upper_barrier=100.0,
            )

    def test_negative_rebate_raises(self) -> None:
        """Test that negative rebate raises ValueError."""
        with pytest.raises(ValueError, match="rebate_amount must be >= 0"):
            DoubleBarrierPayoff(
                option_type="call",
                strike=100.0,
                barrier_style="knock_out",
                lower_barrier=90.0,
                upper_barrier=110.0,
                rebate_amount=-1.0,
            )


class TestDoubleBarrierKnockOutCall:
    """Tests for knock-out call double barrier payoff."""

    @pytest.fixture
    def ko_call_payoff(self) -> DoubleBarrierPayoff:
        """Knock-out call: pays max(S_T - K, 0) if stays in [90, 110]."""
        return DoubleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_style="knock_out",
            lower_barrier=90.0,
            upper_barrier=110.0,
            rebate_amount=0.0,
        )

    def test_stays_in_corridor_itm(self, ko_call_payoff: DoubleBarrierPayoff) -> None:
        """Path stays in corridor and finishes ITM -> vanilla payoff."""
        # Path: 100 -> 102 -> 98 -> 105 (stays in [90, 110], S_T = 105)
        paths = np.array([[100.0, 102.0, 98.0, 105.0]])
        payoffs = ko_call_payoff.terminal_from_paths(paths)
        expected = max(105.0 - 100.0, 0.0)  # 5.0
        np.testing.assert_allclose(payoffs, [expected])

    def test_stays_in_corridor_otm(self, ko_call_payoff: DoubleBarrierPayoff) -> None:
        """Path stays in corridor but finishes OTM -> 0."""
        # Path: 100 -> 102 -> 98 -> 95 (stays in [90, 110], S_T = 95 < K)
        paths = np.array([[100.0, 102.0, 98.0, 95.0]])
        payoffs = ko_call_payoff.terminal_from_paths(paths)
        expected = max(95.0 - 100.0, 0.0)  # 0.0
        np.testing.assert_allclose(payoffs, [expected])

    def test_hits_lower_barrier(self, ko_call_payoff: DoubleBarrierPayoff) -> None:
        """Path hits lower barrier -> knocked out, rebate = 0."""
        # Path: 100 -> 95 -> 85 -> 105 (hits lower at 85 <= 90)
        paths = np.array([[100.0, 95.0, 85.0, 105.0]])
        payoffs = ko_call_payoff.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])  # Knocked out

    def test_hits_upper_barrier(self, ko_call_payoff: DoubleBarrierPayoff) -> None:
        """Path hits upper barrier -> knocked out, rebate = 0."""
        # Path: 100 -> 105 -> 115 -> 95 (hits upper at 115 >= 110)
        paths = np.array([[100.0, 105.0, 115.0, 95.0]])
        payoffs = ko_call_payoff.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])  # Knocked out

    def test_touches_lower_boundary_exactly(self, ko_call_payoff: DoubleBarrierPayoff) -> None:
        """Path touches lower barrier exactly -> knocked out."""
        # Path: 100 -> 95 -> 90 -> 105 (touches lower at 90 <= 90)
        paths = np.array([[100.0, 95.0, 90.0, 105.0]])
        payoffs = ko_call_payoff.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])  # Knocked out

    def test_touches_upper_boundary_exactly(self, ko_call_payoff: DoubleBarrierPayoff) -> None:
        """Path touches upper barrier exactly -> knocked out."""
        # Path: 100 -> 105 -> 110 -> 95 (touches upper at 110 >= 110)
        paths = np.array([[100.0, 105.0, 110.0, 95.0]])
        payoffs = ko_call_payoff.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])  # Knocked out


class TestDoubleBarrierKnockOutCallWithRebate:
    """Tests for knock-out call with non-zero rebate."""

    @pytest.fixture
    def ko_call_with_rebate(self) -> DoubleBarrierPayoff:
        """Knock-out call with rebate of 2.0."""
        return DoubleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_style="knock_out",
            lower_barrier=90.0,
            upper_barrier=110.0,
            rebate_amount=2.0,
        )

    def test_knocked_out_pays_rebate(self, ko_call_with_rebate: DoubleBarrierPayoff) -> None:
        """Path knocked out -> pays rebate instead of 0."""
        paths = np.array([[100.0, 95.0, 85.0, 105.0]])  # Hits lower
        payoffs = ko_call_with_rebate.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [2.0])  # Rebate

    def test_survives_pays_vanilla(self, ko_call_with_rebate: DoubleBarrierPayoff) -> None:
        """Path survives -> pays vanilla (not rebate)."""
        paths = np.array([[100.0, 102.0, 98.0, 105.0]])  # Stays in corridor
        payoffs = ko_call_with_rebate.terminal_from_paths(paths)
        expected = max(105.0 - 100.0, 0.0)  # 5.0 (not rebate)
        np.testing.assert_allclose(payoffs, [expected])


class TestDoubleBarrierKnockInCall:
    """Tests for knock-in call double barrier payoff."""

    @pytest.fixture
    def ki_call_payoff(self) -> DoubleBarrierPayoff:
        """Knock-in call: pays max(S_T - K, 0) only if exits corridor."""
        return DoubleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_style="knock_in",
            lower_barrier=90.0,
            upper_barrier=110.0,
            rebate_amount=0.0,
        )

    def test_stays_in_corridor_pays_rebate(self, ki_call_payoff: DoubleBarrierPayoff) -> None:
        """Path stays in corridor -> never activated, pays rebate = 0."""
        paths = np.array([[100.0, 102.0, 98.0, 105.0]])
        payoffs = ki_call_payoff.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])  # Not activated

    def test_hits_lower_and_finishes_itm(self, ki_call_payoff: DoubleBarrierPayoff) -> None:
        """Path hits lower barrier and finishes ITM -> vanilla payoff."""
        paths = np.array([[100.0, 95.0, 85.0, 105.0]])  # Hits lower
        payoffs = ki_call_payoff.terminal_from_paths(paths)
        expected = max(105.0 - 100.0, 0.0)  # 5.0
        np.testing.assert_allclose(payoffs, [expected])

    def test_hits_upper_and_finishes_itm(self, ki_call_payoff: DoubleBarrierPayoff) -> None:
        """Path hits upper barrier and finishes ITM -> vanilla payoff."""
        paths = np.array([[100.0, 105.0, 115.0, 105.0]])  # Hits upper at 115
        payoffs = ki_call_payoff.terminal_from_paths(paths)
        expected = max(105.0 - 100.0, 0.0)  # 5.0
        np.testing.assert_allclose(payoffs, [expected])

    def test_hits_barrier_but_finishes_otm(self, ki_call_payoff: DoubleBarrierPayoff) -> None:
        """Path hits barrier but finishes OTM -> 0 (vanilla is 0)."""
        paths = np.array([[100.0, 95.0, 85.0, 95.0]])  # Hits lower, S_T = 95 < K
        payoffs = ki_call_payoff.terminal_from_paths(paths)
        expected = max(95.0 - 100.0, 0.0)  # 0.0
        np.testing.assert_allclose(payoffs, [expected])


class TestDoubleBarrierKnockOutPut:
    """Tests for knock-out put double barrier payoff."""

    @pytest.fixture
    def ko_put_payoff(self) -> DoubleBarrierPayoff:
        """Knock-out put: pays max(K - S_T, 0) if stays in [90, 110]."""
        return DoubleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_style="knock_out",
            lower_barrier=90.0,
            upper_barrier=110.0,
        )

    def test_stays_in_corridor_itm_put(self, ko_put_payoff: DoubleBarrierPayoff) -> None:
        """Path stays in corridor and finishes ITM (S_T < K) -> vanilla put payoff."""
        paths = np.array([[100.0, 98.0, 95.0, 92.0]])  # S_T = 92 < K=100
        payoffs = ko_put_payoff.terminal_from_paths(paths)
        expected = max(100.0 - 92.0, 0.0)  # 8.0
        np.testing.assert_allclose(payoffs, [expected])

    def test_stays_in_corridor_otm_put(self, ko_put_payoff: DoubleBarrierPayoff) -> None:
        """Path stays in corridor but finishes OTM (S_T > K) -> 0."""
        paths = np.array([[100.0, 102.0, 105.0, 105.0]])  # S_T = 105 > K
        payoffs = ko_put_payoff.terminal_from_paths(paths)
        expected = max(100.0 - 105.0, 0.0)  # 0.0
        np.testing.assert_allclose(payoffs, [expected])

    def test_hits_barrier_put(self, ko_put_payoff: DoubleBarrierPayoff) -> None:
        """Path hits barrier -> knocked out, 0."""
        paths = np.array([[100.0, 95.0, 85.0, 92.0]])  # Hits lower
        payoffs = ko_put_payoff.terminal_from_paths(paths)
        np.testing.assert_allclose(payoffs, [0.0])


class TestDoubleBarrierVectorization:
    """Tests for vectorized computation across multiple paths."""

    @pytest.fixture
    def ko_call(self) -> DoubleBarrierPayoff:
        """Knock-out call for vectorization tests."""
        return DoubleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_style="knock_out",
            lower_barrier=90.0,
            upper_barrier=110.0,
        )

    def test_multiple_paths(self, ko_call: DoubleBarrierPayoff) -> None:
        """Test multiple paths processed in one call."""
        paths = np.array([
            [100.0, 102.0, 98.0, 105.0],   # Survives, ITM -> 5
            [100.0, 95.0, 85.0, 105.0],    # Hits lower -> 0
            [100.0, 105.0, 115.0, 95.0],   # Hits upper -> 0
            [100.0, 102.0, 98.0, 95.0],    # Survives, OTM -> 0
        ])
        payoffs = ko_call.terminal_from_paths(paths)
        expected = np.array([5.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(payoffs, expected)

    def test_large_batch(self, ko_call: DoubleBarrierPayoff) -> None:
        """Test with large number of paths."""
        n_paths = 10000
        n_steps = 100
        # Generate paths that stay in corridor
        paths = 100 + np.random.randn(n_paths, n_steps) * 3  # Small vol, stays in [90, 110]
        paths = np.clip(paths, 91, 109)  # Force into corridor
        
        payoffs = ko_call.terminal_from_paths(paths)
        
        assert payoffs.shape == (n_paths,)
        assert payoffs.dtype == np.float64
        # All should survive since we clipped
        assert np.all(payoffs >= 0)


class TestDoubleBarrierInOutParity:
    """Test that knock-in + knock-out = vanilla payoff (In-Out parity)."""

    def test_in_out_parity_call(self) -> None:
        """Knock-in + Knock-out = Vanilla for calls."""
        ko_call = DoubleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_style="knock_out",
            lower_barrier=90.0,
            upper_barrier=110.0,
        )
        ki_call = DoubleBarrierPayoff(
            option_type="call",
            strike=100.0,
            barrier_style="knock_in",
            lower_barrier=90.0,
            upper_barrier=110.0,
        )

        # Generate various paths
        paths = np.array([
            [100.0, 102.0, 98.0, 105.0],   # Survives
            [100.0, 95.0, 85.0, 105.0],    # Hits lower
            [100.0, 105.0, 115.0, 105.0],  # Hits upper
            [100.0, 102.0, 98.0, 95.0],    # Survives, OTM
        ])

        ko_payoffs = ko_call.terminal_from_paths(paths)
        ki_payoffs = ki_call.terminal_from_paths(paths)

        # Vanilla payoff
        terminal_spots = paths[:, -1]
        vanilla_payoffs = np.maximum(terminal_spots - 100.0, 0.0)

        # In-Out parity: KO + KI = Vanilla
        np.testing.assert_allclose(ko_payoffs + ki_payoffs, vanilla_payoffs)

    def test_in_out_parity_put(self) -> None:
        """Knock-in + Knock-out = Vanilla for puts."""
        ko_put = DoubleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_style="knock_out",
            lower_barrier=90.0,
            upper_barrier=110.0,
        )
        ki_put = DoubleBarrierPayoff(
            option_type="put",
            strike=100.0,
            barrier_style="knock_in",
            lower_barrier=90.0,
            upper_barrier=110.0,
        )

        paths = np.array([
            [100.0, 98.0, 95.0, 92.0],     # Survives, ITM
            [100.0, 95.0, 85.0, 92.0],     # Hits lower, ITM
            [100.0, 105.0, 115.0, 105.0],  # Hits upper, OTM
        ])

        ko_payoffs = ko_put.terminal_from_paths(paths)
        ki_payoffs = ki_put.terminal_from_paths(paths)

        terminal_spots = paths[:, -1]
        vanilla_payoffs = np.maximum(100.0 - terminal_spots, 0.0)

        np.testing.assert_allclose(ko_payoffs + ki_payoffs, vanilla_payoffs)
