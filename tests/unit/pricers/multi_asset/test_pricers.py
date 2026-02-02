"""
Unit tests for multi-asset pricers.

Tests cover MC pricing for baskets, spreads, and rainbow options.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from src.marketdata.core.ids import MarketId
from src.instruments.multi_asset import (
    MultiAssetBasketEuropeanOption,
    MultiAssetSpreadEuropeanOption,
    MultiAssetExchangeEuropeanOption,
    MultiAssetBestOfEuropeanOption,
    MultiAssetWorstOfEuropeanOption,
)
from src.models.numeric.monte_carlo.multi_asset import CorrelationMatrix
from src.pricers.multi_asset import (
    MultiAssetBasketEuropeanOptionMcPricer,
    MultiAssetSpreadEuropeanOptionMcPricer,
    MultiAssetSpreadEuropeanOptionKirkPricer,
    MultiAssetExchangeEuropeanOptionMargrabePricer,
    MultiAssetBestOfEuropeanOptionMcPricer,
    MultiAssetWorstOfEuropeanOptionMcPricer,
)


def make_id(name: str) -> MarketId:
    """Helper to create MarketId for tests."""
    return MarketId(asset_class="EQ", mkt_type="SPOT", name=name)


# =============================================================================
# Basket Pricer Tests
# =============================================================================

class TestMultiAssetBasketEuropeanOptionMcPricer:
    """Tests for MultiAssetBasketEuropeanOptionMcPricer."""

    @pytest.fixture
    def basket_call(self):
        """Sample basket call."""
        return MultiAssetBasketEuropeanOption.from_lists(
            option_type="call",
            underlyings=[make_id("A"), make_id("B")],
            weights=[0.5, 0.5],
            strike=100.0,
            expiry=1.0,
        )

    @pytest.fixture
    def basket_put(self):
        """Sample basket put."""
        return MultiAssetBasketEuropeanOption.from_lists(
            option_type="put",
            underlyings=[make_id("A"), make_id("B")],
            weights=[0.5, 0.5],
            strike=100.0,
            expiry=1.0,
        )

    @pytest.fixture
    def market_data(self):
        """Sample market data."""
        return {
            'spots': np.array([100.0, 100.0]),
            'r': 0.05,
            'dividends': np.array([0.02, 0.02]),
            'volatilities': np.array([0.2, 0.25]),
            'correlation': CorrelationMatrix.from_flat(0.5, n=2),
        }

    def test_basket_call_positive(self, basket_call, market_data):
        """Basket call price should be positive."""
        pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price = pricer.price(basket_call, **market_data)
        assert price > 0

    def test_basket_put_positive(self, basket_put, market_data):
        """Basket put price should be positive."""
        pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price = pricer.price(basket_put, **market_data)
        assert price > 0

    def test_price_with_std_error(self, basket_call, market_data):
        """Test price_with_std_error method."""
        pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price, std_error = pricer.price_with_std_error(basket_call, **market_data)
        assert price > 0
        assert std_error > 0
        assert std_error < price

    def test_run_returns_simulation_artifact(self, basket_call, market_data):
        """Test run method returns simulation artifact."""
        pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=50000, seed=42)
        sim = pricer.run(basket_call, **market_data)
        assert sim.n_paths_effective > 0
        assert len(sim.discounted_payoffs) == sim.n_paths_effective
        assert len(sim.basket_values) == sim.n_paths_effective

    def test_basket_put_call_parity(self, basket_call, basket_put, market_data):
        """Basket options should satisfy put-call parity."""
        pricer = MultiAssetBasketEuropeanOptionMcPricer(n_paths=200000, seed=42)
        call = pricer.price(basket_call, **market_data)
        put = pricer.price(basket_put, **market_data)

        weights = np.array(basket_call.weights)
        spots = market_data['spots']
        r = market_data['r']
        q = market_data['dividends']
        T = basket_call.expiry
        K = basket_call.strike

        forward_basket = (spots * np.exp((r - q) * T) * weights).sum()
        discount = np.exp(-r * T)
        parity = discount * (forward_basket - K)

        assert_allclose(call - put, parity, rtol=0.05)


# =============================================================================
# Spread Pricer Tests
# =============================================================================

class TestMultiAssetSpreadEuropeanOptionMcPricer:
    """Tests for MultiAssetSpreadEuropeanOptionMcPricer."""

    @pytest.fixture
    def spread_call(self):
        """Sample spread call."""
        return MultiAssetSpreadEuropeanOption(
            option_type="call",
            underlying1=make_id("A"),
            underlying2=make_id("B"),
            strike=5.0,
            expiry=0.5,
        )

    @pytest.fixture
    def spread_put(self):
        """Sample spread put."""
        return MultiAssetSpreadEuropeanOption(
            option_type="put",
            underlying1=make_id("A"),
            underlying2=make_id("B"),
            strike=5.0,
            expiry=0.5,
        )

    def test_spread_call_positive(self, spread_call):
        """Spread call price should be positive."""
        pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price = pricer.price(
            spread_call,
            spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
            sigma1=0.2, sigma2=0.25, rho=0.5
        )
        assert price > 0

    def test_spread_put_positive(self, spread_put):
        """Spread put price should be positive."""
        pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price = pricer.price(
            spread_put,
            spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
            sigma1=0.2, sigma2=0.25, rho=0.5
        )
        assert price > 0

    def test_kirk_vs_mc(self, spread_call):
        """Kirk's approximation should be close to MC."""
        mc_pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=200000, seed=42)
        kirk_pricer = MultiAssetSpreadEuropeanOptionKirkPricer()

        mc_price = mc_pricer.price(
            spread_call,
            spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
            sigma1=0.2, sigma2=0.25, rho=0.5
        )
        kirk_price = kirk_pricer.price(
            spread_call,
            spot1=100.0, spot2=95.0, r=0.05, q1=0.02, q2=0.02,
            sigma1=0.2, sigma2=0.25, rho=0.5
        )
        assert_allclose(kirk_price, mc_price, rtol=0.1)


# =============================================================================
# Exchange Option Pricer Tests
# =============================================================================

class TestMultiAssetExchangeEuropeanOptionMargrabePricer:
    """Tests for Margrabe's formula pricer."""

    def test_margrabe_vs_mc(self):
        """Margrabe should match MC for exchange option."""
        exchange = MultiAssetExchangeEuropeanOption(
            underlying1=make_id("A"),
            underlying2=make_id("B"),
            expiry=1.0,
        )

        # Create equivalent spread call for MC
        spread = MultiAssetSpreadEuropeanOption(
            option_type="call",
            underlying1=make_id("A"),
            underlying2=make_id("B"),
            strike=0.0,
            expiry=1.0,
        )

        margrabe_pricer = MultiAssetExchangeEuropeanOptionMargrabePricer()
        mc_pricer = MultiAssetSpreadEuropeanOptionMcPricer(n_paths=200000, seed=42)

        margrabe_price = margrabe_pricer.price(
            exchange,
            spot1=100.0, spot2=100.0, r=0.05, q1=0.02, q2=0.02,
            sigma1=0.2, sigma2=0.25, rho=0.5
        )
        mc_price = mc_pricer.price(
            spread,
            spot1=100.0, spot2=100.0, r=0.05, q1=0.02, q2=0.02,
            sigma1=0.2, sigma2=0.25, rho=0.5
        )

        assert_allclose(margrabe_price, mc_price, rtol=0.03)

    def test_exchange_positive(self):
        """Exchange option price should be positive."""
        exchange = MultiAssetExchangeEuropeanOption(
            underlying1=make_id("A"),
            underlying2=make_id("B"),
            expiry=1.0,
        )
        pricer = MultiAssetExchangeEuropeanOptionMargrabePricer()
        price = pricer.price(
            exchange,
            spot1=100.0, spot2=100.0, r=0.05, q1=0.02, q2=0.02,
            sigma1=0.2, sigma2=0.25, rho=0.5
        )
        assert price > 0


# =============================================================================
# Rainbow Pricer Tests
# =============================================================================

class TestMultiAssetBestOfEuropeanOptionMcPricer:
    """Tests for MultiAssetBestOfEuropeanOptionMcPricer."""

    @pytest.fixture
    def rainbow_params(self):
        """Sample rainbow parameters."""
        return {
            'spots': np.array([100.0, 100.0, 100.0]),
            'r': 0.05,
            'dividends': np.array([0.02, 0.02, 0.02]),
            'volatilities': np.array([0.2, 0.25, 0.3]),
            'correlation': CorrelationMatrix.from_flat(0.5, n=3),
        }

    def test_best_of_call_positive(self, rainbow_params):
        """Best-of call should be positive."""
        inst = MultiAssetBestOfEuropeanOption.from_list(
            option_type="call",
            underlyings=[make_id("A"), make_id("B"), make_id("C")],
            strike=100.0, expiry=1.0
        )
        pricer = MultiAssetBestOfEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price = pricer.price(inst, **rainbow_params)
        assert price > 0

    def test_best_of_put_positive(self, rainbow_params):
        """Best-of put should be positive."""
        inst = MultiAssetBestOfEuropeanOption(
            option_type="put",
            underlyings=(make_id("A"), make_id("B"), make_id("C")),
            strike=100.0, expiry=1.0
        )
        pricer = MultiAssetBestOfEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price = pricer.price(inst, **rainbow_params)
        assert price > 0

    def test_correlation_effect_best_of(self, rainbow_params):
        """Higher correlation should give lower best-of price."""
        inst = MultiAssetBestOfEuropeanOption(
            option_type="call",
            underlyings=(make_id("A"), make_id("B")),
            strike=100.0, expiry=1.0
        )

        pricer = MultiAssetBestOfEuropeanOptionMcPricer(n_paths=100000, seed=42)

        low_params = rainbow_params.copy()
        low_params['spots'] = np.array([100.0, 100.0])
        low_params['dividends'] = np.array([0.0, 0.0])
        low_params['volatilities'] = np.array([0.3, 0.3])
        low_params['correlation'] = CorrelationMatrix.from_flat(0.2, n=2)

        high_params = low_params.copy()
        high_params['correlation'] = CorrelationMatrix.from_flat(0.9, n=2)

        price_low = pricer.price(inst, **low_params)
        price_high = pricer.price(inst, **high_params)

        assert price_low > price_high


class TestMultiAssetWorstOfEuropeanOptionMcPricer:
    """Tests for MultiAssetWorstOfEuropeanOptionMcPricer."""

    @pytest.fixture
    def rainbow_params(self):
        """Sample rainbow parameters."""
        return {
            'spots': np.array([100.0, 100.0, 100.0]),
            'r': 0.05,
            'dividends': np.array([0.02, 0.02, 0.02]),
            'volatilities': np.array([0.2, 0.25, 0.3]),
            'correlation': CorrelationMatrix.from_flat(0.5, n=3),
        }

    def test_worst_of_call_positive(self, rainbow_params):
        """Worst-of call should be positive."""
        inst = MultiAssetWorstOfEuropeanOption(
            option_type="call",
            underlyings=(make_id("A"), make_id("B"), make_id("C")),
            strike=100.0, expiry=1.0
        )
        pricer = MultiAssetWorstOfEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price = pricer.price(inst, **rainbow_params)
        assert price > 0

    def test_worst_of_put_positive(self, rainbow_params):
        """Worst-of put should be positive."""
        inst = MultiAssetWorstOfEuropeanOption(
            option_type="put",
            underlyings=(make_id("A"), make_id("B"), make_id("C")),
            strike=100.0, expiry=1.0
        )
        pricer = MultiAssetWorstOfEuropeanOptionMcPricer(n_paths=50000, seed=42)
        price = pricer.price(inst, **rainbow_params)
        assert price > 0

    def test_best_of_geq_worst_of(self, rainbow_params):
        """Best-of call should be >= worst-of call."""
        best_inst = MultiAssetBestOfEuropeanOption(
            option_type="call",
            underlyings=(make_id("A"), make_id("B"), make_id("C")),
            strike=100.0, expiry=1.0
        )
        worst_inst = MultiAssetWorstOfEuropeanOption(
            option_type="call",
            underlyings=(make_id("A"), make_id("B"), make_id("C")),
            strike=100.0, expiry=1.0
        )

        best_pricer = MultiAssetBestOfEuropeanOptionMcPricer(n_paths=100000, seed=42)
        worst_pricer = MultiAssetWorstOfEuropeanOptionMcPricer(n_paths=100000, seed=42)

        best_price = best_pricer.price(best_inst, **rainbow_params)
        worst_price = worst_pricer.price(worst_inst, **rainbow_params)

        assert best_price >= worst_price

    def test_correlation_effect_worst_of(self, rainbow_params):
        """Higher correlation should give higher worst-of price."""
        inst = MultiAssetWorstOfEuropeanOption(
            option_type="call",
            underlyings=(make_id("A"), make_id("B")),
            strike=100.0, expiry=1.0
        )

        pricer = MultiAssetWorstOfEuropeanOptionMcPricer(n_paths=100000, seed=42)

        low_params = rainbow_params.copy()
        low_params['spots'] = np.array([100.0, 100.0])
        low_params['dividends'] = np.array([0.0, 0.0])
        low_params['volatilities'] = np.array([0.3, 0.3])
        low_params['correlation'] = CorrelationMatrix.from_flat(0.2, n=2)

        high_params = low_params.copy()
        high_params['correlation'] = CorrelationMatrix.from_flat(0.9, n=2)

        price_low = pricer.price(inst, **low_params)
        price_high = pricer.price(inst, **high_params)

        assert price_high > price_low
