"""
Unit tests for Digital option payoff implementations.

These tests verify:
- DigitalCashPayoff: cash-or-nothing payoffs
- DigitalAssetPayoff: asset-or-nothing payoffs
- Correct boundary conditions (S >= K for calls, S <= K for puts)
- Validation of input parameters
- Edge cases and vectorization
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.payoffs.digital import DigitalCashPayoff, DigitalAssetPayoff
from src.models.payoffs.types import OptionType


# =============================================================================
# DigitalCashPayoff Tests
# =============================================================================

class TestDigitalCashPayoffCall:
    """Test digital cash-or-nothing call payoff: cash * 1{S >= K}."""

    def test_call_payoff_in_the_money(self) -> None:
        """
        Test cash digital call when spot is above strike.

        For a cash digital call:
        - Pays fixed cash amount if S >= K at expiry
        - Otherwise pays 0
        """
        payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=10.0)

        # Spot above strike
        spots = np.array([100.1, 110.0, 150.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # All should pay the cash amount
        expected = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_payoff_out_of_the_money(self) -> None:
        """
        Test cash digital call when spot is below strike.

        When S < K, payoff = 0.
        """
        payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=10.0)

        # Spot below strike
        spots = np.array([99.9, 90.0, 50.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # All should be 0
        expected = np.zeros(3, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_payoff_at_the_money(self) -> None:
        """
        Test cash digital call when spot equals strike.

        At-the-money: S = K, condition is S >= K, so it pays.
        """
        payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=10.0)

        spots = np.array([100.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # Exactly at strike, S >= K is true
        assert result[0] == pytest.approx(10.0, abs=1e-10)

    def test_call_payoff_default_cash(self) -> None:
        """Test digital call with default cash amount (1.0)."""
        payoff = DigitalCashPayoff(option_type="call", strike=100.0)

        spots = np.array([110.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # Default cash is 1.0
        assert result[0] == pytest.approx(1.0, abs=1e-10)


class TestDigitalCashPayoffPut:
    """Test digital cash-or-nothing put payoff: cash * 1{S <= K}."""

    def test_put_payoff_in_the_money(self) -> None:
        """
        Test cash digital put when spot is below strike.

        For a cash digital put:
        - Pays fixed cash amount if S <= K at expiry
        - Otherwise pays 0
        """
        payoff = DigitalCashPayoff(option_type="put", strike=100.0, cash=10.0)

        # Spot below strike
        spots = np.array([99.9, 90.0, 50.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # All should pay the cash amount
        expected = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_put_payoff_out_of_the_money(self) -> None:
        """
        Test cash digital put when spot is above strike.

        When S > K, payoff = 0.
        """
        payoff = DigitalCashPayoff(option_type="put", strike=100.0, cash=10.0)

        # Spot above strike
        spots = np.array([100.1, 110.0, 150.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # All should be 0
        expected = np.zeros(3, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_put_payoff_at_the_money(self) -> None:
        """
        Test cash digital put when spot equals strike.

        At-the-money: S = K, condition is S <= K, so it pays.
        """
        payoff = DigitalCashPayoff(option_type="put", strike=100.0, cash=10.0)

        spots = np.array([100.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # Exactly at strike, S <= K is true
        assert result[0] == pytest.approx(10.0, abs=1e-10)


class TestDigitalCashPayoffValidation:
    """Test DigitalCashPayoff input validation."""

    def test_invalid_strike_zero(self) -> None:
        """Test that zero strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            DigitalCashPayoff(option_type="call", strike=0.0, cash=10.0)

    def test_invalid_strike_negative(self) -> None:
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            DigitalCashPayoff(option_type="call", strike=-10.0, cash=10.0)

    def test_invalid_option_type(self) -> None:
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            DigitalCashPayoff(option_type="invalid", strike=100.0, cash=10.0)  # type: ignore[arg-type]

    def test_invalid_cash_infinite(self) -> None:
        """Test that infinite cash raises ValueError."""
        with pytest.raises(ValueError, match="cash must be finite"):
            DigitalCashPayoff(option_type="call", strike=100.0, cash=np.inf)

    def test_invalid_cash_nan(self) -> None:
        """Test that NaN cash raises ValueError."""
        with pytest.raises(ValueError, match="cash must be finite"):
            DigitalCashPayoff(option_type="call", strike=100.0, cash=np.nan)

    def test_negative_cash_allowed(self) -> None:
        """Test that negative cash is allowed (short position)."""
        # This should not raise - negative cash represents a short digital
        payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=-10.0)

        spots = np.array([110.0], dtype=np.float64)
        result = payoff.terminal(spots)

        assert result[0] == pytest.approx(-10.0, abs=1e-10)


# =============================================================================
# DigitalAssetPayoff Tests
# =============================================================================

class TestDigitalAssetPayoffCall:
    """Test digital asset-or-nothing call payoff: units * S * 1{S >= K}."""

    def test_call_payoff_in_the_money(self) -> None:
        """
        Test asset digital call when spot is above strike.

        For an asset digital call:
        - Pays asset_units * S if S >= K at expiry
        - Otherwise pays 0
        """
        payoff = DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=2.0)

        # Spot above strike
        spots = np.array([100.0, 110.0, 150.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # Payoff = 2.0 * S for each spot >= strike
        expected = np.array([200.0, 220.0, 300.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_payoff_out_of_the_money(self) -> None:
        """
        Test asset digital call when spot is below strike.

        When S < K, payoff = 0.
        """
        payoff = DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=2.0)

        # Spot below strike
        spots = np.array([99.9, 90.0, 50.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # All should be 0
        expected = np.zeros(3, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_payoff_at_the_money(self) -> None:
        """
        Test asset digital call when spot equals strike.

        At-the-money: S = K = 100, condition is S >= K, so payoff = units * S.
        """
        payoff = DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=2.0)

        spots = np.array([100.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # units * S = 2.0 * 100.0 = 200.0
        assert result[0] == pytest.approx(200.0, abs=1e-10)

    def test_call_payoff_default_units(self) -> None:
        """Test asset digital call with default units (1.0)."""
        payoff = DigitalAssetPayoff(option_type="call", strike=100.0)

        spots = np.array([110.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # Default units = 1.0, so payoff = 1.0 * 110.0 = 110.0
        assert result[0] == pytest.approx(110.0, abs=1e-10)


class TestDigitalAssetPayoffPut:
    """Test digital asset-or-nothing put payoff: units * S * 1{S <= K}."""

    def test_put_payoff_in_the_money(self) -> None:
        """
        Test asset digital put when spot is below strike.

        For an asset digital put:
        - Pays asset_units * S if S <= K at expiry
        - Otherwise pays 0
        """
        payoff = DigitalAssetPayoff(option_type="put", strike=100.0, asset_units=2.0)

        # Spot below or at strike
        spots = np.array([100.0, 90.0, 50.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # Payoff = 2.0 * S for each spot <= strike
        expected = np.array([200.0, 180.0, 100.0], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_put_payoff_out_of_the_money(self) -> None:
        """
        Test asset digital put when spot is above strike.

        When S > K, payoff = 0.
        """
        payoff = DigitalAssetPayoff(option_type="put", strike=100.0, asset_units=2.0)

        # Spot above strike
        spots = np.array([100.1, 110.0, 150.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # All should be 0
        expected = np.zeros(3, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_put_payoff_at_the_money(self) -> None:
        """
        Test asset digital put when spot equals strike.

        At-the-money: S = K = 100, condition is S <= K, so payoff = units * S.
        """
        payoff = DigitalAssetPayoff(option_type="put", strike=100.0, asset_units=2.0)

        spots = np.array([100.0], dtype=np.float64)
        result = payoff.terminal(spots)

        # units * S = 2.0 * 100.0 = 200.0
        assert result[0] == pytest.approx(200.0, abs=1e-10)


class TestDigitalAssetPayoffValidation:
    """Test DigitalAssetPayoff input validation."""

    def test_invalid_strike_zero(self) -> None:
        """Test that zero strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            DigitalAssetPayoff(option_type="call", strike=0.0, asset_units=1.0)

    def test_invalid_strike_negative(self) -> None:
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            DigitalAssetPayoff(option_type="call", strike=-10.0, asset_units=1.0)

    def test_invalid_option_type(self) -> None:
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            DigitalAssetPayoff(option_type="invalid", strike=100.0, asset_units=1.0)  # type: ignore[arg-type]

    def test_invalid_asset_units_infinite(self) -> None:
        """Test that infinite asset_units raises ValueError."""
        with pytest.raises(ValueError, match="asset_units must be finite"):
            DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=np.inf)

    def test_invalid_asset_units_nan(self) -> None:
        """Test that NaN asset_units raises ValueError."""
        with pytest.raises(ValueError, match="asset_units must be finite"):
            DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=np.nan)


# =============================================================================
# Vectorization and Properties Tests
# =============================================================================

class TestDigitalPayoffVectorization:
    """Test vectorization and array handling."""

    def test_cash_digital_large_array(self) -> None:
        """Test cash digital with large array."""
        payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=5.0)

        n = 10000
        spots = np.random.uniform(50.0, 150.0, n)
        result = payoff.terminal(spots)

        # Verify shape
        assert result.shape == (n,)

        # Verify values are either 0 or cash
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
        assert set(unique_values).issubset({0.0, 5.0})

    def test_asset_digital_large_array(self) -> None:
        """Test asset digital with large array."""
        payoff = DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=1.0)

        n = 10000
        spots = np.random.uniform(50.0, 150.0, n)
        result = payoff.terminal(spots)

        # Verify shape
        assert result.shape == (n,)

        # Verify all non-zero payoffs equal spots (since units=1)
        itm_mask = spots >= 100.0
        np.testing.assert_array_almost_equal(result[itm_mask], spots[itm_mask])
        np.testing.assert_array_almost_equal(result[~itm_mask], 0.0)

    def test_dtype_consistency(self) -> None:
        """Test that output dtype is float64."""
        cash_payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=10.0)
        asset_payoff = DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=1.0)

        # Input as int array
        spots = np.array([90, 100, 110])

        cash_result = cash_payoff.terminal(spots)
        asset_result = asset_payoff.terminal(spots)

        assert cash_result.dtype == np.float64
        assert asset_result.dtype == np.float64


class TestDigitalPayoffProperties:
    """Test payoff class properties and methods."""

    def test_cash_digital_not_path_dependent(self) -> None:
        """Test that cash digital is not path-dependent."""
        payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=10.0)
        assert payoff.is_path_dependent is False

    def test_asset_digital_not_path_dependent(self) -> None:
        """Test that asset digital is not path-dependent."""
        payoff = DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=1.0)
        assert payoff.is_path_dependent is False

    def test_cash_digital_intrinsic_equals_terminal(self) -> None:
        """Test intrinsic equals terminal for cash digital."""
        payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=10.0)

        spots = np.array([90.0, 100.0, 110.0], dtype=np.float64)

        terminal = payoff.terminal(spots)
        intrinsic = payoff.intrinsic(spots)

        np.testing.assert_array_equal(terminal, intrinsic)

    def test_asset_digital_intrinsic_equals_terminal(self) -> None:
        """Test intrinsic equals terminal for asset digital."""
        payoff = DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=1.0)

        spots = np.array([90.0, 100.0, 110.0], dtype=np.float64)

        terminal = payoff.terminal(spots)
        intrinsic = payoff.intrinsic(spots)

        np.testing.assert_array_equal(terminal, intrinsic)

    def test_callable_interface(self) -> None:
        """Test that payoffs can be called directly."""
        cash_payoff = DigitalCashPayoff(option_type="call", strike=100.0, cash=10.0)
        asset_payoff = DigitalAssetPayoff(option_type="call", strike=100.0, asset_units=1.0)

        spots = np.array([110.0], dtype=np.float64)

        # Direct call should work
        cash_result = cash_payoff(spots)
        asset_result = asset_payoff(spots)

        assert cash_result[0] == pytest.approx(10.0, abs=1e-10)
        assert asset_result[0] == pytest.approx(110.0, abs=1e-10)


class TestDigitalPayoffCallPutParity:
    """Test call-put relationships for digital options."""

    def test_cash_digital_call_put_sum(self) -> None:
        """
        Test that cash digital call + put = cash (everywhere except exactly at strike).

        For cash digitals (with >= and <= conditions):
        - At S > K: call pays cash, put pays 0
        - At S < K: call pays 0, put pays cash
        - At S = K: both pay cash (so sum = 2*cash)
        """
        strike = 100.0
        cash = 10.0
        call_payoff = DigitalCashPayoff(option_type="call", strike=strike, cash=cash)
        put_payoff = DigitalCashPayoff(option_type="put", strike=strike, cash=cash)

        # Test away from strike
        spots_away = np.array([90.0, 110.0], dtype=np.float64)
        call_values = call_payoff.terminal(spots_away)
        put_values = put_payoff.terminal(spots_away)

        # Away from strike, call + put = cash
        np.testing.assert_array_almost_equal(call_values + put_values, cash)

        # At strike, both conditions are true
        spots_at_strike = np.array([100.0], dtype=np.float64)
        call_at_strike = call_payoff.terminal(spots_at_strike)
        put_at_strike = put_payoff.terminal(spots_at_strike)

        # At strike, sum = 2 * cash
        assert (call_at_strike[0] + put_at_strike[0]) == pytest.approx(2 * cash, abs=1e-10)

    def test_asset_digital_call_put_sum(self) -> None:
        """
        Test asset digital call + put relationship.

        Similar to cash: away from strike, call + put = units * S.
        At strike, both pay, so sum = 2 * units * S.
        """
        strike = 100.0
        units = 2.0
        call_payoff = DigitalAssetPayoff(option_type="call", strike=strike, asset_units=units)
        put_payoff = DigitalAssetPayoff(option_type="put", strike=strike, asset_units=units)

        # Test away from strike
        spots_away = np.array([90.0, 110.0], dtype=np.float64)
        call_values = call_payoff.terminal(spots_away)
        put_values = put_payoff.terminal(spots_away)

        # Away from strike, call + put = units * S
        expected = units * spots_away
        np.testing.assert_array_almost_equal(call_values + put_values, expected)
