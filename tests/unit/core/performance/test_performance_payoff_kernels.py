"""
Tests for payoff performance kernels.

Author: QuantStrata Team
"""
import pytest
import numpy as np

from src.core.performance.backend import numba_available
from src.core.performance.payoff_kernels import (
    vanilla_payoff,
    digital_payoff,
    barrier_payoff,
    asian_payoff,
    lookback_payoff,
    vanilla_payoff_numpy,
    digital_payoff_numpy,
    barrier_payoff_numpy,
    asian_payoff_numpy,
    lookback_payoff_numpy,
)


class TestVanillaPayoffNumpy:
    """Tests for NumPy vanilla payoff."""
    
    def test_call_payoff(self):
        """Call payoff should be max(S-K, 0)."""
        spots = np.array([90.0, 100.0, 110.0])
        strike = 100.0
        payoff = vanilla_payoff_numpy(spots, strike, "call")
        expected = np.array([0.0, 0.0, 10.0])
        assert np.allclose(payoff, expected)
    
    def test_put_payoff(self):
        """Put payoff should be max(K-S, 0)."""
        spots = np.array([90.0, 100.0, 110.0])
        strike = 100.0
        payoff = vanilla_payoff_numpy(spots, strike, "put")
        expected = np.array([10.0, 0.0, 0.0])
        assert np.allclose(payoff, expected)
    
    def test_payoff_non_negative(self):
        """Payoffs should be non-negative."""
        np.random.seed(42)
        spots = np.random.lognormal(mean=np.log(100), sigma=0.3, size=1000)
        
        call_payoff = vanilla_payoff_numpy(spots, 100.0, "call")
        put_payoff = vanilla_payoff_numpy(spots, 100.0, "put")
        
        assert np.all(call_payoff >= 0)
        assert np.all(put_payoff >= 0)


class TestDigitalPayoffNumpy:
    """Tests for NumPy digital payoff."""
    
    def test_digital_call(self):
        """Digital call should pay if S > K."""
        spots = np.array([99.0, 100.0, 101.0])
        strike = 100.0
        payoff = digital_payoff_numpy(spots, strike, "call", payout=1.0)
        expected = np.array([0.0, 0.0, 1.0])
        assert np.allclose(payoff, expected)
    
    def test_digital_put(self):
        """Digital put should pay if S < K."""
        spots = np.array([99.0, 100.0, 101.0])
        strike = 100.0
        payoff = digital_payoff_numpy(spots, strike, "put", payout=1.0)
        expected = np.array([1.0, 0.0, 0.0])
        assert np.allclose(payoff, expected)
    
    def test_custom_payout(self):
        """Should use custom payout amount."""
        spots = np.array([101.0])
        payoff = digital_payoff_numpy(spots, 100.0, "call", payout=10.0)
        assert payoff[0] == 10.0


class TestBarrierPayoffNumpy:
    """Tests for NumPy barrier payoff."""
    
    @pytest.fixture
    def paths_up(self):
        """Paths that breach an upper barrier."""
        # 10 steps, 3 paths
        paths = np.array([
            [100.0, 100.0, 100.0],
            [102.0, 101.0, 103.0],
            [104.0, 102.0, 106.0],
            [106.0, 103.0, 109.0],  # Path 0 and 2 breach 105
            [108.0, 102.0, 112.0],
            [107.0, 101.0, 115.0],
        ])
        return paths
    
    def test_up_and_out(self, paths_up):
        """Up-and-out should pay only if barrier not breached."""
        # Barrier at 105
        payoff = barrier_payoff_numpy(
            paths_up, strike=100.0, barrier=105.0,
            option_type="call", barrier_type="up_and_out"
        )
        # Path 0 and 2 breach -> 0 payoff
        # Path 1 doesn't breach -> vanilla payoff = max(101-100, 0) = 1
        assert payoff[1] > 0  # Path 1 pays
    
    def test_up_and_in(self, paths_up):
        """Up-and-in should pay only if barrier breached."""
        payoff = barrier_payoff_numpy(
            paths_up, strike=100.0, barrier=105.0,
            option_type="call", barrier_type="up_and_in"
        )
        # Path 0 and 2 breach -> vanilla payoff
        # Path 1 doesn't breach -> 0
        assert payoff[1] == 0  # Path 1 doesn't pay


class TestAsianPayoffNumpy:
    """Tests for NumPy Asian payoff."""
    
    def test_arithmetic_average(self):
        """Should compute arithmetic average correctly."""
        paths = np.array([
            [100.0, 100.0],
            [110.0, 90.0],
            [120.0, 80.0],
        ])  # 3 steps, 2 paths
        
        # Path 0: avg = (100+110+120)/3 = 110
        # Path 1: avg = (100+90+80)/3 = 90
        payoff = asian_payoff_numpy(paths, strike=100.0, option_type="call", asian_type="arithmetic")
        expected = np.array([10.0, 0.0])
        assert np.allclose(payoff, expected)
    
    def test_geometric_average(self):
        """Should compute geometric average correctly."""
        paths = np.array([
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, 100.0],
        ])
        # Geometric average of all 100s is 100
        payoff = asian_payoff_numpy(paths, strike=100.0, option_type="call", asian_type="geometric")
        assert np.allclose(payoff, 0.0)


class TestLookbackPayoffNumpy:
    """Tests for NumPy lookback payoff."""
    
    def test_floating_call(self):
        """Floating call: S_T - min(S)."""
        paths = np.array([
            [100.0],
            [90.0],   # min
            [105.0],
            [110.0],  # terminal
        ])
        payoff = lookback_payoff_numpy(paths, strike=None, option_type="call", lookback_type="floating")
        expected = 110.0 - 90.0
        assert np.isclose(payoff[0], expected)
    
    def test_floating_put(self):
        """Floating put: max(S) - S_T."""
        paths = np.array([
            [100.0],
            [120.0],  # max
            [105.0],
            [95.0],   # terminal
        ])
        payoff = lookback_payoff_numpy(paths, strike=None, option_type="put", lookback_type="floating")
        expected = 120.0 - 95.0
        assert np.isclose(payoff[0], expected)
    
    def test_fixed_call(self):
        """Fixed call: max(max(S) - K, 0)."""
        paths = np.array([
            [100.0],
            [110.0],
            [105.0],
        ])
        payoff = lookback_payoff_numpy(paths, strike=100.0, option_type="call", lookback_type="fixed")
        expected = 10.0  # max(110) - 100
        assert np.isclose(payoff[0], expected)


class TestUnifiedApi:
    """Tests for unified payoff APIs."""
    
    def test_vanilla_numpy_backend(self):
        """Vanilla payoff should work with NumPy backend."""
        spots = np.array([90.0, 100.0, 110.0])
        payoff = vanilla_payoff(spots, strike=100.0, option_type="call", backend="numpy")
        expected = np.array([0.0, 0.0, 10.0])
        assert np.allclose(payoff, expected)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_vanilla_numba_backend(self):
        """Vanilla payoff should work with Numba backend."""
        spots = np.array([90.0, 100.0, 110.0])
        payoff = vanilla_payoff(spots, strike=100.0, option_type="call", backend="numba")
        expected = np.array([0.0, 0.0, 10.0])
        assert np.allclose(payoff, expected)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_backends_agree_vanilla(self):
        """NumPy and Numba should produce same vanilla results."""
        np.random.seed(42)
        spots = np.random.lognormal(mean=np.log(100), sigma=0.3, size=10000)
        
        numpy_result = vanilla_payoff(spots, strike=100.0, option_type="call", backend="numpy")
        numba_result = vanilla_payoff(spots, strike=100.0, option_type="call", backend="numba")
        
        assert np.allclose(numpy_result, numba_result)
    
    @pytest.mark.skipif(not numba_available(), reason="Numba not installed")
    def test_backends_agree_asian(self):
        """NumPy and Numba should produce same Asian results."""
        np.random.seed(42)
        paths = np.random.lognormal(mean=np.log(100), sigma=0.2, size=(50, 1000))
        
        numpy_result = asian_payoff(paths, strike=100.0, option_type="call", asian_type="arithmetic", backend="numpy")
        numba_result = asian_payoff(paths, strike=100.0, option_type="call", asian_type="arithmetic", backend="numba")
        
        assert np.allclose(numpy_result, numba_result)
