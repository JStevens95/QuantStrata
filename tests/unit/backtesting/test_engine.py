"""
Unit tests for BacktestEngine.

Tests the core backtesting engine functionality:
- Portfolio tracking
- Order execution
- Transaction costs
- Result computation
"""

import numpy as np
import pytest
from dataclasses import dataclass
from datetime import date

from src.backtesting.core.engine import (
    BacktestEngine,
    BacktestResult,
    BacktestConfig,
    PortfolioState,
    Position,
    BacktestContext,
)
from src.backtesting.data.providers import DictDataProvider, SimpleMarketSnapshot


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class SimpleOrder:
    """Simple order for testing."""
    instrument_id: str
    quantity: float


@pytest.fixture
def simple_data():
    """Create simple test data."""
    return {
        date(2024, 1, 1): {"AAPL": 150.0, "GOOGL": 140.0},
        date(2024, 1, 2): {"AAPL": 152.0, "GOOGL": 142.0},
        date(2024, 1, 3): {"AAPL": 151.0, "GOOGL": 145.0},
        date(2024, 1, 4): {"AAPL": 155.0, "GOOGL": 143.0},
        date(2024, 1, 5): {"AAPL": 158.0, "GOOGL": 148.0},
    }


@pytest.fixture
def provider(simple_data):
    """Create data provider."""
    return DictDataProvider(simple_data)


# =============================================================================
# Portfolio State Tests
# =============================================================================

class TestPortfolioState:
    """Tests for PortfolioState."""
    
    def test_initial_state(self):
        """Test initial portfolio state."""
        portfolio = PortfolioState(cash=100000.0)
        
        assert portfolio.cash == 100000.0
        assert portfolio.total_value == 100000.0
        assert len(portfolio.positions) == 0
    
    def test_with_positions(self):
        """Test portfolio with positions."""
        portfolio = PortfolioState(
            cash=50000.0,
            positions={
                "AAPL": Position("AAPL", 100, 150.0, 15000.0),
                "GOOGL": Position("GOOGL", 50, 140.0, 7000.0),
            }
        )
        
        assert portfolio.total_value == 50000.0 + 15000.0 + 7000.0
    
    def test_get_quantity(self):
        """Test getting position quantity."""
        portfolio = PortfolioState(
            cash=50000.0,
            positions={"AAPL": Position("AAPL", 100, 150.0, 15000.0)}
        )
        
        assert portfolio.get_quantity("AAPL") == 100
        assert portfolio.get_quantity("GOOGL") == 0


class TestPosition:
    """Tests for Position class."""
    
    def test_update_market_value(self):
        """Test market value update."""
        pos = Position("AAPL", 100, 150.0, 15000.0)
        pos.update_market_value(160.0)
        
        assert pos.market_value == 16000.0
        assert pos.unrealized_pnl == 1000.0  # 100 * (160 - 150)
    
    def test_unrealized_pnl_loss(self):
        """Test unrealized P&L for losing position."""
        pos = Position("AAPL", 100, 150.0, 15000.0)
        pos.update_market_value(140.0)
        
        assert pos.unrealized_pnl == -1000.0


# =============================================================================
# Backtest Engine Tests
# =============================================================================

class TestBacktestEngine:
    """Tests for BacktestEngine."""
    
    def test_buy_and_hold(self, provider):
        """Test simple buy and hold strategy."""
        def buy_and_hold(market, portfolio, context):
            if context.step == 0:
                return [SimpleOrder("AAPL", 100)]
            return []
        
        engine = BacktestEngine(config=BacktestConfig())
        result = engine.run(
            strategy=buy_and_hold,
            data_provider=provider,
            initial_capital=100000.0,
        )
        
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 100000.0
        assert len(result.dates) == 5
        assert len(result.trades) == 1
    
    def test_transaction_costs(self, provider):
        """Test transaction costs are applied."""
        def buy_strategy(market, portfolio, context):
            if context.step == 0:
                return [SimpleOrder("AAPL", 100)]
            return []
        
        # Without costs
        engine_no_cost = BacktestEngine(config=BacktestConfig(transaction_cost=0.0))
        result_no_cost = engine_no_cost.run(
            strategy=buy_strategy,
            data_provider=provider,
            initial_capital=100000.0,
        )
        
        # With costs
        engine_with_cost = BacktestEngine(config=BacktestConfig(transaction_cost=0.01))
        result_with_cost = engine_with_cost.run(
            strategy=buy_strategy,
            data_provider=provider,
            initial_capital=100000.0,
        )
        
        # With costs should have lower final value
        assert result_with_cost.final_value < result_no_cost.final_value
    
    def test_no_trades_strategy(self, provider):
        """Test strategy that makes no trades."""
        def no_trades(market, portfolio, context):
            return []
        
        engine = BacktestEngine()
        result = engine.run(
            strategy=no_trades,
            data_provider=provider,
            initial_capital=100000.0,
        )
        
        assert len(result.trades) == 0
        assert result.final_value == result.initial_capital
    
    def test_result_metrics(self, provider):
        """Test that result contains valid metrics."""
        def buy_and_hold(market, portfolio, context):
            if context.step == 0:
                return [SimpleOrder("AAPL", 100)]
            return []
        
        engine = BacktestEngine()
        result = engine.run(
            strategy=buy_and_hold,
            data_provider=provider,
            initial_capital=100000.0,
        )
        
        assert result.metrics is not None
        assert np.isfinite(result.metrics.total_return)
        assert np.isfinite(result.metrics.sharpe_ratio)
    
    def test_result_arrays(self, provider):
        """Test result array properties."""
        def buy_and_hold(market, portfolio, context):
            if context.step == 0:
                return [SimpleOrder("AAPL", 100)]
            return []
        
        engine = BacktestEngine()
        result = engine.run(
            strategy=buy_and_hold,
            data_provider=provider,
            initial_capital=100000.0,
        )
        
        assert len(result.portfolio_values) == 5
        assert len(result.returns) == 5
        assert len(result.cash_series) == 5
        assert len(result.cumulative_returns) == 5
        assert len(result.drawdown_series) == 5


class TestBacktestConfig:
    """Tests for BacktestConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = BacktestConfig()
        
        assert config.transaction_cost == 0.0
        assert config.slippage == 0.0
        assert config.risk_free_rate == 0.0
        assert config.periods_per_year == 252
        assert config.allow_short is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            transaction_cost=0.001,
            slippage=0.0005,
            risk_free_rate=0.02,
            allow_short=False,
        )
        
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.allow_short is False


class TestBacktestResult:
    """Tests for BacktestResult."""
    
    def test_str_representation(self, provider):
        """Test string representation."""
        def no_trades(market, portfolio, context):
            return []
        
        engine = BacktestEngine()
        result = engine.run(
            strategy=no_trades,
            data_provider=provider,
            initial_capital=100000.0,
        )
        
        s = str(result)
        assert "BacktestResult" in s
        assert "Initial Capital" in s
        assert "Final Value" in s
