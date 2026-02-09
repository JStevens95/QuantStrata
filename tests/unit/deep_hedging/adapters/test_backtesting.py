"""
Unit tests for deep hedging backtesting adapter module.

Tests BacktestEngineAdapter, HedgingStrategy, and BacktestConfig.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from src.deep_hedging.adapters.backtesting import (
    BacktestConfig,
    BacktestEngineAdapter,
    HedgingBacktestResult,
    HedgingStrategy,
    OptionParams,
)


def _default_option_params() -> OptionParams:
    """OptionParams with required strike and maturity (date)."""
    return OptionParams(
        strike=100.0,
        maturity=date.today() + timedelta(days=90),
        option_type="call",
    )


class TestOptionParams:
    """Tests for OptionParams dataclass."""
    
    def test_default_params(self) -> None:
        """Test default option parameters (required strike, maturity)."""
        params = _default_option_params()
        
        assert params.strike == 100.0
        assert params.maturity > date.today()
        assert params.option_type == "call"
    
    def test_custom_params(self) -> None:
        """Test custom option parameters."""
        params = OptionParams(
            strike=110.0,
            maturity=date.today() + timedelta(days=180),
            option_type="put",
        )
        
        assert params.strike == 110.0
        assert params.option_type == "put"


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = BacktestConfig()
        
        assert config.transaction_cost >= 0
        assert config.rehedge_frequency in ["daily", "weekly", "hourly"]
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BacktestConfig(
            transaction_cost=0.002,
            rehedge_frequency="daily",
        )
        
        assert config.transaction_cost == 0.002
        assert config.rehedge_frequency == "daily"


class TestHedgingStrategy:
    """Tests for HedgingStrategy wrapper."""
    
    def test_strategy_creation(self) -> None:
        """Test strategy creation with mock agent and required option_params, config."""
        
        class MockAgent:
            def act(self, state: np.ndarray) -> float:
                return 0.5
        
        agent = MockAgent()
        strategy = HedgingStrategy(
            agent=agent,
            option_params=_default_option_params(),
            config=BacktestConfig(),
        )
        
        assert strategy is not None
    
    def test_strategy_on_data(self) -> None:
        """Test strategy returns (trade_qty, info) from on_data."""
        
        class MockAgent:
            def act(self, state: np.ndarray) -> float:
                return state[0] * 0.01  # Simple delta-like
        
        agent = MockAgent()
        strategy = HedgingStrategy(
            agent=agent,
            option_params=_default_option_params(),
            config=BacktestConfig(),
        )
        strategy.on_start(100.0, date.today())
        
        trade_qty, info = strategy.on_data(
            spot=101.0,
            current_date=date.today() + timedelta(days=1),
            volatility=0.2,
            risk_free_rate=0.05,
        )
        
        assert isinstance(trade_qty, (int, float, np.floating))
        assert isinstance(info, dict)


class TestHedgingBacktestResult:
    """Tests for HedgingBacktestResult dataclass."""
    
    def test_result_creation(self) -> None:
        """Test result creation (API: total_pnl, hedging_pnl, option_pnl, total_cost, etc.)."""
        result = HedgingBacktestResult(
            total_pnl=-50.0,
            hedging_pnl=-30.0,
            option_pnl=-20.0,
            total_cost=10.0,
            mean_daily_pnl=-1.0,
            std_daily_pnl=2.0,
            sharpe_ratio=-0.5,
            max_drawdown=5.0,
            mean_position=0.5,
            max_position=1.0,
            pnl_history=[-10.0, -20.0, -30.0, -40.0, -50.0],
            position_history=[0.5, 0.52, 0.48, 0.55, 0.51],
        )
        
        assert result.total_pnl == -50.0
        assert len(result.pnl_history) == 5
    
    def test_result_metrics(self) -> None:
        """Test result metrics."""
        result = HedgingBacktestResult(
            total_pnl=100.0,
            hedging_pnl=90.0,
            option_pnl=20.0,
            total_cost=10.0,
            mean_daily_pnl=1.0,
            std_daily_pnl=2.0,
            sharpe_ratio=0.5,
            max_drawdown=3.0,
            mean_position=0.5,
            max_position=1.0,
            pnl_history=list(np.random.randn(100).cumsum()),
            position_history=[0.5] * 100,
        )
        
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
            rehedge_frequency="daily",
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
        
        assert isinstance(result, HedgingBacktestResult)
        assert len(result.pnl_history) > 0
    
    def test_run_backtest_with_option_params(self, mock_agent, sample_data) -> None:
        """Test backtest with custom option parameters."""
        prices, volatilities = sample_data
        n = len(prices)
        
        adapter = BacktestEngineAdapter(agent=mock_agent)
        
        option_params = OptionParams(
            strike=105.0,
            maturity=date.today() + timedelta(days=min(90, n)),
            option_type="call",
        )
        
        result = adapter.run_backtest(
            prices=prices,
            volatilities=volatilities,
            option_params=option_params,
        )
        
        assert isinstance(result, HedgingBacktestResult)
    
    def test_run_backtest_with_benchmark(self, mock_agent, sample_data) -> None:
        """Test backtest with delta hedge benchmark (run_benchmark=True)."""
        prices, volatilities = sample_data
        
        adapter = BacktestEngineAdapter(agent=mock_agent)
        
        result = adapter.run_backtest(
            prices=prices,
            volatilities=volatilities,
            run_benchmark=True,
        )
        
        assert result is not None
    
    def test_transaction_costs_applied(self, mock_agent, sample_data) -> None:
        """Test that transaction costs are applied (total_cost)."""
        prices, volatilities = sample_data
        
        config_no_cost = BacktestConfig(transaction_cost=0.0)
        config_with_cost = BacktestConfig(transaction_cost=0.01)
        
        adapter_no_cost = BacktestEngineAdapter(agent=mock_agent, config=config_no_cost)
        adapter_with_cost = BacktestEngineAdapter(agent=mock_agent, config=config_with_cost)
        
        result_no_cost = adapter_no_cost.run_backtest(prices=prices, volatilities=volatilities)
        result_with_cost = adapter_with_cost.run_backtest(prices=prices, volatilities=volatilities)
        
        assert result_with_cost.total_cost >= result_no_cost.total_cost
    
    def test_hedge_ratios_recorded(self, mock_agent, sample_data) -> None:
        """Test that position history is recorded."""
        prices, volatilities = sample_data
        
        adapter = BacktestEngineAdapter(agent=mock_agent)
        
        result = adapter.run_backtest(
            prices=prices,
            volatilities=volatilities,
        )
        
        assert len(result.position_history) > 0
        assert all(abs(h) <= 2.0 for h in result.position_history)
