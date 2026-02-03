"""
Unit tests for deep_hedging.core.costs module.
"""

import numpy as np
import pytest

from src.deep_hedging.core.costs import (
    ProportionalCost,
    FixedCost,
    MarketImpactCost,
    CombinedCost,
    ZeroCost,
    create_realistic_cost,
)


class TestProportionalCost:
    """Tests for ProportionalCost."""
    
    def test_compute_basic(self):
        """Test basic cost computation."""
        cost = ProportionalCost(spread_bps=10.0)  # 10bp = 0.1%
        
        # Trading 100 shares at S=100
        result = cost.compute(trade_size=100, spot=100)
        
        # Half-spread = 5bp = 0.0005
        # Cost = 0.0005 * 100 * 100 = 5.0
        assert result == pytest.approx(5.0, rel=1e-6)
    
    def test_compute_zero_trade(self):
        """Test that zero trade has zero cost."""
        cost = ProportionalCost(spread_bps=10.0)
        result = cost.compute(trade_size=0, spot=100)
        assert result == 0.0
    
    def test_compute_negative_trade(self):
        """Test that selling has same cost as buying (symmetric)."""
        cost = ProportionalCost(spread_bps=10.0)
        
        buy_cost = cost.compute(trade_size=100, spot=100)
        sell_cost = cost.compute(trade_size=-100, spot=100)
        
        assert buy_cost == sell_cost
    
    def test_compute_vectorised(self):
        """Test vectorised computation."""
        cost = ProportionalCost(spread_bps=10.0)
        
        trades = np.array([100, -50, 0])
        spots = np.array([100, 100, 100])
        
        results = cost.compute(trade_size=trades, spot=spots)
        
        assert results.shape == (3,)
        assert results[0] == pytest.approx(5.0, rel=1e-6)
        assert results[1] == pytest.approx(2.5, rel=1e-6)
        assert results[2] == pytest.approx(0.0, rel=1e-6)
    
    def test_half_spread_property(self):
        """Test half_spread property."""
        cost = ProportionalCost(spread_bps=10.0)
        assert cost.half_spread == pytest.approx(0.0005, rel=1e-6)


class TestFixedCost:
    """Tests for FixedCost."""
    
    def test_compute_with_trade(self):
        """Test that any trade incurs fixed cost."""
        cost = FixedCost(cost_per_trade=5.0)
        
        result = cost.compute(trade_size=100, spot=100)
        assert result == 5.0
    
    def test_compute_zero_trade(self):
        """Test that zero trade has zero cost."""
        cost = FixedCost(cost_per_trade=5.0)
        
        result = cost.compute(trade_size=0, spot=100)
        assert result == 0.0
    
    def test_compute_small_trade(self):
        """Test trade just above threshold."""
        cost = FixedCost(cost_per_trade=5.0, threshold=0.01)
        
        # Below threshold
        result_below = cost.compute(trade_size=0.001, spot=100)
        assert result_below == 0.0
        
        # Above threshold
        result_above = cost.compute(trade_size=0.1, spot=100)
        assert result_above == 5.0


class TestMarketImpactCost:
    """Tests for MarketImpactCost."""
    
    def test_compute_linear_impact(self):
        """Test with linear impact (exponent=1)."""
        cost = MarketImpactCost(impact_coef=0.001, impact_exp=1.0)
        
        result = cost.compute(trade_size=100, spot=100)
        # Cost = 0.001 * 100 * 100^1 = 10.0
        assert result == pytest.approx(10.0, rel=1e-6)
    
    def test_compute_square_root_impact(self):
        """Test with square-root impact (exponent=0.5)."""
        cost = MarketImpactCost(impact_coef=0.001, impact_exp=0.5)
        
        result = cost.compute(trade_size=100, spot=100)
        # Cost = 0.001 * 100 * 100^0.5 = 0.001 * 100 * 10 = 1.0
        assert result == pytest.approx(1.0, rel=1e-6)
    
    def test_compute_zero_trade(self):
        """Test that zero trade has zero cost."""
        cost = MarketImpactCost(impact_coef=0.001, impact_exp=1.5)
        
        result = cost.compute(trade_size=0, spot=100)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestCombinedCost:
    """Tests for CombinedCost."""
    
    def test_combine_two_costs(self):
        """Test combining proportional and fixed costs."""
        prop = ProportionalCost(spread_bps=10.0)
        fixed = FixedCost(cost_per_trade=1.0)
        
        combined = CombinedCost(components=[prop, fixed])
        
        result = combined.compute(trade_size=100, spot=100)
        
        expected = prop.compute(100, 100) + fixed.compute(100, 100)
        assert result == pytest.approx(expected, rel=1e-6)
    
    def test_addition_operator(self):
        """Test using + to combine costs."""
        prop = ProportionalCost(spread_bps=10.0)
        fixed = FixedCost(cost_per_trade=1.0)
        
        combined = prop + fixed
        
        assert isinstance(combined, CombinedCost)
        assert len(combined.components) == 2


class TestZeroCost:
    """Tests for ZeroCost."""
    
    def test_compute_always_zero(self):
        """Test that ZeroCost always returns zero."""
        cost = ZeroCost()
        
        assert cost.compute(trade_size=100, spot=100) == 0.0
        assert cost.compute(trade_size=-50, spot=200) == 0.0
        assert cost.compute(trade_size=0, spot=100) == 0.0


class TestCreateRealisticCost:
    """Tests for create_realistic_cost factory."""
    
    def test_creates_proportional_only(self):
        """Test creating cost with spread only."""
        cost = create_realistic_cost(spread_bps=10.0, impact_coef=0.0, fixed_cost=0.0)
        
        assert isinstance(cost, ProportionalCost)
    
    def test_creates_combined(self):
        """Test creating combined cost."""
        cost = create_realistic_cost(spread_bps=10.0, impact_coef=0.001, fixed_cost=1.0)
        
        assert isinstance(cost, CombinedCost)
        assert len(cost.components) == 3
    
    def test_creates_zero_cost(self):
        """Test creating zero cost when all params are zero."""
        cost = create_realistic_cost(spread_bps=0.0, impact_coef=0.0, fixed_cost=0.0)
        
        assert isinstance(cost, ZeroCost)
