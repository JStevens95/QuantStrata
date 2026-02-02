"""
Unit tests for P&L attribution.

Tests the P&L decomposition and attribution:
- Greeks-based attribution
- Aggregation functions
"""

import numpy as np
import pytest
from datetime import date

from src.backtesting.attribution.pnl import (
    PnLBreakdown,
    PnLAttribution,
    attribute_pnl_to_greeks,
    aggregate_attribution,
)


class TestPnLBreakdown:
    """Tests for PnLBreakdown."""
    
    def test_basic_breakdown(self):
        """Test basic breakdown creation."""
        breakdown = PnLBreakdown(
            total_pnl=1000.0,
            delta_pnl=800.0,
            gamma_pnl=50.0,
            theta_pnl=-100.0,
            vega_pnl=200.0,
            rho_pnl=10.0,
            residual=40.0,
        )
        
        assert breakdown.total_pnl == 1000.0
        assert breakdown.explained_pnl == 960.0  # Sum of components
        assert breakdown.residual == 40.0
    
    def test_explanation_ratio(self):
        """Test explanation ratio calculation."""
        breakdown = PnLBreakdown(
            total_pnl=1000.0,
            delta_pnl=900.0,
            gamma_pnl=0.0,
            theta_pnl=0.0,
            vega_pnl=0.0,
            rho_pnl=0.0,
            residual=100.0,
        )
        
        # 90% explained
        assert np.isclose(breakdown.explanation_ratio, 0.9)
    
    def test_str_representation(self):
        """Test string representation."""
        breakdown = PnLBreakdown(
            total_pnl=1000.0,
            delta_pnl=500.0,
            gamma_pnl=50.0,
            theta_pnl=-50.0,
            vega_pnl=100.0,
            rho_pnl=10.0,
            residual=390.0,
        )
        
        s = str(breakdown)
        assert "Total P&L" in s
        assert "Delta P&L" in s
        assert "Residual" in s


class TestAttributePnLToGreeks:
    """Tests for attribute_pnl_to_greeks function."""
    
    def test_delta_only(self):
        """Test pure delta attribution."""
        breakdown = attribute_pnl_to_greeks(
            pnl=500.0,
            delta=100.0,
            spot_move=5.0,
        )
        
        assert breakdown.delta_pnl == 500.0  # 100 * 5
        assert breakdown.residual == 0.0
    
    def test_gamma_contribution(self):
        """Test gamma contribution."""
        breakdown = attribute_pnl_to_greeks(
            pnl=550.0,
            delta=100.0,
            gamma=10.0,
            spot_move=5.0,
        )
        
        assert breakdown.delta_pnl == 500.0
        assert breakdown.gamma_pnl == 125.0  # 0.5 * 10 * 5^2
        # Total explained: 625, residual: -75
    
    def test_vega_contribution(self):
        """Test vega contribution."""
        breakdown = attribute_pnl_to_greeks(
            pnl=100.0,
            vega=1000.0,
            vol_move=0.01,  # 1% vol move
        )
        
        assert breakdown.vega_pnl == 10.0  # 1000 * 0.01
    
    def test_theta_contribution(self):
        """Test theta contribution."""
        breakdown = attribute_pnl_to_greeks(
            pnl=-100.0,
            theta=-100.0,  # Daily theta
            dt=1/252,
        )
        
        assert breakdown.theta_pnl == -100.0
    
    def test_full_attribution(self):
        """Test full attribution with all Greeks."""
        breakdown = attribute_pnl_to_greeks(
            pnl=1000.0,
            delta=500.0,
            gamma=10.0,
            theta=-50.0,
            vega=5000.0,
            rho=100.0,
            spot_move=2.0,
            vol_move=0.01,
            rate_move=0.001,
            dt=1/252,
        )
        
        assert breakdown.delta_pnl == 1000.0
        assert breakdown.gamma_pnl == 20.0
        assert breakdown.vega_pnl == 50.0
        assert breakdown.rho_pnl == 0.1
        assert np.isclose(breakdown.theta_pnl, -50.0)


class TestPnLAttribution:
    """Tests for PnLAttribution class."""
    
    def test_add_breakdowns(self):
        """Test adding daily breakdowns."""
        attr = PnLAttribution()
        
        attr.add(date(2024, 1, 1), PnLBreakdown(100, 80, 10, -5, 10, 5, 0))
        attr.add(date(2024, 1, 2), PnLBreakdown(200, 150, 20, -10, 30, 10, 0))
        
        assert len(attr.dates) == 2
        assert len(attr.breakdowns) == 2
    
    def test_cumulative(self):
        """Test cumulative breakdown."""
        attr = PnLAttribution()
        attr.add(date(2024, 1, 1), PnLBreakdown(100, 80, 10, -5, 10, 5, 0))
        attr.add(date(2024, 1, 2), PnLBreakdown(200, 150, 20, -10, 30, 10, 0))
        
        cum = attr.cumulative
        assert cum.total_pnl == 300
        assert cum.delta_pnl == 230
    
    def test_to_arrays(self):
        """Test conversion to arrays."""
        attr = PnLAttribution()
        attr.add(date(2024, 1, 1), PnLBreakdown(100, 80, 10, -5, 10, 5, 0))
        attr.add(date(2024, 1, 2), PnLBreakdown(200, 150, 20, -10, 30, 10, 0))
        
        arrays = attr.to_arrays()
        assert "total_pnl" in arrays
        assert len(arrays["total_pnl"]) == 2
        assert arrays["delta_pnl"][0] == 80


class TestAggregateAttribution:
    """Tests for aggregate_attribution function."""
    
    def test_weekly_aggregation(self):
        """Test weekly aggregation."""
        attr = PnLAttribution()
        # Add daily data for two weeks
        for i in range(10):
            dt = date(2024, 1, i + 1)  # Jan 1-10
            attr.add(dt, PnLBreakdown(100, 80, 10, -5, 10, 5, 0))
        
        weekly = aggregate_attribution(attr, frequency="weekly")
        
        # Should have 2 weeks
        assert len(weekly.dates) == 2
        assert weekly.cumulative.total_pnl == 1000
    
    def test_monthly_aggregation(self):
        """Test monthly aggregation."""
        attr = PnLAttribution()
        # Add daily data for two months
        for month in [1, 2]:
            for day in range(1, 11):
                dt = date(2024, month, day)
                attr.add(dt, PnLBreakdown(100, 80, 10, -5, 10, 5, 0))
        
        monthly = aggregate_attribution(attr, frequency="monthly")
        
        # Should have 2 months
        assert len(monthly.dates) == 2
        assert monthly.cumulative.total_pnl == 2000
    
    def test_empty_attribution(self):
        """Test aggregating empty attribution."""
        attr = PnLAttribution()
        weekly = aggregate_attribution(attr, frequency="weekly")
        
        assert len(weekly.dates) == 0
