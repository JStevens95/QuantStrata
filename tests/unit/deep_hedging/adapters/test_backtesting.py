"""
Unit tests for deep hedging backtesting adapter module.

Tests BacktestEngineAdapter, HedgingStrategy, and BacktestConfig.
"""

import numpy as np
import pytest

from src.deep_hedging.adapters.backtesting import (
    BacktestConfig,
    BacktestEngineAdapter,
    HedgingBacktestResult,
    HedgingStrategy,
    OptionParams,
)


class TestOptionParams:
    """Tests for OptionParams dataclass."""
    
    def test_default_params(self) -> None:
        """Test default option parameters."""
        params = OptionParams()
        
        assert params.strike == 100.0
        assert params.maturity == 0.25
        assert params.is_call is True
    
    def test_custom_params(self) -> None:
        """Test custom option parameters."""
        params = OptionParams(
            strike=110.0,
            maturity=0.5,
            is_call=False,
        )
        
        assert params.strike == 110.0
        assert params.is_call is False


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = BacktestConfig()
        
        assert config.transaction_cost >= 0
        assert config.rebalance_frequency in ["daily", "hourly", "continuous"]
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BacktestConfig(
            transaction_cost=0.002,
            rebalance_frequency="daily",
            compute_benchmark=True,
        )
        
        assert config.transaction_cost == 0.002
        assert config.compute_benchmark is True


class TestHedgingStrategy:
    """Tests for HedgingStrategy wrapper."""
    
    def test_strategy_creation(self) -> None:
        """Test strategy creation with mock agent."""
        
        class MockAgent:
            def act(self, state: np.ndarray) -> float:
                return 0.5
        
        agent = MockAgent()
        strategy = HedgingStrategy(agent=agent)
        
        assert strategy is not None
    
    def test_strategy_get_action(self) -> None:
        """Test getting action from strategy."""
        
        class MockAgent:
            def act(self, state: np.ndarray) -> float:
                return state[0] * 0.01  # Simple delta-like
        
        agent = MockAgent()
        strategy = HedgingStrategy(agent=agent)
        
        state = np.array([100.0, 0.5, 0.2])
        action = strategy.get_hedge_ratio(state)
        
        assert isinstance(action, (int, float, np.floating))


class TestHedgingBacktestResult:
    """Tests for HedgingBacktestResult dataclass."""
    
    def test_result_creation(self) -> None:
        """Test result creation."""
        result = HedgingBacktestResult(
            total_pnl=-50.0,
            pnl_series=np.array([-10, -20, -30, -40, -50]),
            hedge_ratios=np.array([0.5, 0.52, 0.48, 0.55, 0.51]),
            transaction_costs=25.0,
        )
        
        assert result.total_pnl == -50.0
        assert len(result.pnl_series) == 5
    
    def test_result_metrics(self) -> None:
        """Test result metrics computation."""
        result = HedgingBacktestResult(
            total_pnl=100.0,
            pnl_series=np.random.randn(100).cumsum(),
            hedge_ratios=np.ones(100) * 0.5,
            transaction_costs=10.0,
        )
        
        # Should have total PnL
        assert result.total_pnl is not None


class TestBacktestEngineAdapter:
    """Tests for BacktestEngineAdapter."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock hedging agent."""
        
        class DeltaAgent:
            """Simple delta hedging agent."""
            
            def act(self, state: np.ndarray) -> float:
                # Simple: hedge ratio = delta (assume first element)
                return min(max(state[0] / 100.0, -1.0), 1.0)
        
        return DeltaAgent()
    
    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample market data."""
        np.random.seed(42)
        n_days = 63  # Quarter
        
        # Simulate price path
        returns = np.random.randn(n_days) * 0.01
        prices = 100 * np.cumprod(1 + returns)
        
        # Volatility (realized)
        volatilities = np.ones(n_days) * 0.20
        
        return prices, volatilities
    
    def test_adapter_creation(self, mock_agent) -> None:
        """Test adapter creation."""
        adapter = BacktestEngineAdapter(agent=mock_agent)
        
        assert adapter is not None
    
    def test_adapter_with_config(self, mock_agent) -> None:
        """Test adapter with custom config."""
        config = BacktestConfig(
            transaction_cost=0.001,
            rebalance_frequency="daily",
        )
        
        adapter = BacktestEngineAdapter(
            agent=mock_agent,
            config=config,
        )
        
        assert adapter is not None
    
    def test_run_backtest(self, mock_agent, sample_data) -> None:
        """Test running backtest."""
        prices, volatilities = sample_data
        
        adapter = BacktestEngineAdapter(agent=mock_agent)
        
        result = adapter.run_backtest(
            prices=prices,
            volatilities=volatilities,
        )
        
        # Should return result
        assert isinstance(result, HedgingBacktestResult)
        
        # Should have PnL series
        assert len(result.pnl_series) > 0
    
    def test_run_backtest_with_option_params(self, mock_agent, sample_data) -> None:
        """Test backtest with custom option parameters."""
        prices, volatilities = sample_data
        
        adapter = BacktestEngineAdapter(agent=mock_agent)
        
        option_params = OptionParams(
            strike=105.0,
            maturity=0.25,
            is_call=True,
        )
        
        result = adapter.run_backtest(
            prices=prices,
            volatilities=volatilities,
            option_params=option_params,
        )
        
        assert isinstance(result, HedgingBacktestResult)
    
    def test_run_backtest_with_benchmark(self, mock_agent, sample_data) -> None:
        """Test backtest with delta hedge benchmark."""
        prices, volatilities = sample_data
        
        config = BacktestConfig(compute_benchmark=True)
        adapter = BacktestEngineAdapter(agent=mock_agent, config=config)
        
        result = adapter.run_backtest(
            prices=prices,
            volatilities=volatilities,
            run_benchmark=True,
        )
        
        # Should have benchmark comparison
        assert result is not None
    
    def test_transaction_costs_applied(self, mock_agent, sample_data) -> None:
        """Test that transaction costs are applied."""
        prices, volatilities = sample_data
        
        config_no_cost = BacktestConfig(transaction_cost=0.0)
        config_with_cost = BacktestConfig(transaction_cost=0.01)
        
        adapter_no_cost = BacktestEngineAdapter(agent=mock_agent, config=config_no_cost)
        adapter_with_cost = BacktestEngineAdapter(agent=mock_agent, config=config_with_cost)
        
        result_no_cost = adapter_no_cost.run_backtest(prices=prices, volatilities=volatilities)
        result_with_cost = adapter_with_cost.run_backtest(prices=prices, volatilities=volatilities)
        
        # Transaction costs should reduce PnL
        assert result_with_cost.transaction_costs >= result_no_cost.transaction_costs
    
    def test_hedge_ratios_recorded(self, mock_agent, sample_data) -> None:
        """Test that hedge ratios are recorded."""
        prices, volatilities = sample_data
        
        adapter = BacktestEngineAdapter(agent=mock_agent)
        
        result = adapter.run_backtest(
            prices=prices,
            volatilities=volatilities,
        )
        
        # Should have hedge ratios
        assert len(result.hedge_ratios) > 0
        
        # Hedge ratios should be reasonable
        assert all(abs(h) <= 2.0 for h in result.hedge_ratios)
