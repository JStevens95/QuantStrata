"""
Unit tests for deep_hedging.environments.gbm module.
"""

import numpy as np
import pytest

from src.deep_hedging.core.types import HedgingConfig, HedgingState
from src.deep_hedging.core.costs import ProportionalCost, ZeroCost
from src.deep_hedging.environments.gbm import GBMHedgingEnv, create_gbm_env


class TestGBMHedgingEnv:
    """Tests for GBMHedgingEnv."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration."""
        return HedgingConfig(
            option_type="call",
            strike=100.0,
            maturity=0.25,
            spot_initial=100.0,
            volatility=0.2,
            risk_free_rate=0.05,
            n_steps=10,
        )
    
    @pytest.fixture
    def env(self, config):
        """Create environment with proportional cost."""
        cost = ProportionalCost(spread_bps=10.0)
        return GBMHedgingEnv(config, cost)
    
    def test_reset_returns_state(self, env):
        """Test that reset returns a HedgingState."""
        state, info = env.reset(seed=42)
        
        assert isinstance(state, HedgingState)
        assert state.spot == env.config.spot_initial
        assert state.time == 0.0
        assert state.time_to_maturity == env.config.maturity
        assert state.step == 0
    
    def test_reset_with_seed_reproducible(self, env):
        """Test that same seed gives same path."""
        state1, _ = env.reset(seed=42)
        env.step(0.5)  # Take a step
        
        state2, _ = env.reset(seed=42)
        
        assert state1.spot == state2.spot
    
    def test_step_returns_correct_tuple(self, env):
        """Test step returns (state, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        
        result = env.step(0.5)  # Action = hold 0.5 units
        
        assert len(result) == 5
        state, reward, terminated, truncated, info = result
        
        assert isinstance(state, HedgingState)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_episode_terminates_at_maturity(self, env):
        """Test that episode terminates after n_steps."""
        state, _ = env.reset(seed=42)
        
        steps = 0
        while True:
            state, reward, terminated, truncated, info = env.step(0.5)
            steps += 1
            if terminated or truncated:
                break
        
        assert steps == env.config.n_steps
        assert terminated == True
        assert "terminal_pnl" in info
    
    def test_compute_payoff_call(self, config):
        """Test call payoff computation."""
        env = GBMHedgingEnv(config, ZeroCost())
        
        # ITM call
        assert env.compute_payoff(110.0) == 10.0
        
        # ATM call
        assert env.compute_payoff(100.0) == 0.0
        
        # OTM call
        assert env.compute_payoff(90.0) == 0.0
    
    def test_compute_payoff_put(self):
        """Test put payoff computation."""
        config = HedgingConfig(
            option_type="put",
            strike=100.0,
            maturity=0.25,
            spot_initial=100.0,
            volatility=0.2,
            risk_free_rate=0.05,
            n_steps=10,
        )
        env = GBMHedgingEnv(config, ZeroCost())
        
        # ITM put
        assert env.compute_payoff(90.0) == 10.0
        
        # ATM put
        assert env.compute_payoff(100.0) == 0.0
        
        # OTM put
        assert env.compute_payoff(110.0) == 0.0
    
    def test_compute_greeks_call(self, env):
        """Test Greeks computation for call."""
        greeks = env.compute_greeks(spot=100.0, time_to_maturity=0.25)
        
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        
        # ATM call delta should be around 0.5
        assert 0.4 < greeks["delta"] < 0.7
        
        # Gamma and vega should be positive
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0
    
    def test_transaction_costs_applied(self, env):
        """Test that transaction costs are applied."""
        state, _ = env.reset(seed=42)
        
        # Make a trade
        state, reward, _, _, info = env.step(0.5)
        
        # Cost should be positive
        assert info["cost"] > 0
        assert info["trade_size"] == 0.5 - env.config.initial_position
    
    def test_no_cost_with_zero_cost_model(self, config):
        """Test that ZeroCost gives zero transaction costs."""
        env = GBMHedgingEnv(config, ZeroCost())
        
        state, _ = env.reset(seed=42)
        state, reward, _, _, info = env.step(0.5)
        
        assert info["cost"] == 0.0
    
    def test_get_episode_after_termination(self, env):
        """Test get_episode returns full record."""
        state, _ = env.reset(seed=42)
        
        while True:
            state, _, terminated, _, _ = env.step(0.5)
            if terminated:
                break
        
        episode = env.get_episode()
        
        assert episode.config == env.config
        assert len(episode.spot_path) == env.config.n_steps + 1
        assert len(episode.positions) == env.config.n_steps
        assert len(episode.costs) == env.config.n_steps
    
    def test_simulate_paths_shape(self, env):
        """Test batch path simulation shape."""
        paths = env.simulate_paths(n_paths=100, seed=42)
        
        assert paths.shape == (100, env.config.n_steps + 1)
    
    def test_simulate_paths_antithetic(self, env):
        """Test antithetic variates in path simulation."""
        paths = env.simulate_paths(n_paths=100, seed=42, antithetic=True)
        
        # First 50 and last 50 should be mirrored in log-returns
        # (not exact due to exponentiation, but returns should be negatively correlated)
        log_returns_first = np.log(paths[:50, -1] / paths[:50, 0])
        log_returns_last = np.log(paths[50:, -1] / paths[50:, 0])
        
        # Correlation should be negative
        corr = np.corrcoef(log_returns_first, log_returns_last)[0, 1]
        assert corr < -0.9


class TestCreateGbmEnv:
    """Tests for create_gbm_env factory function."""
    
    def test_creates_environment(self):
        """Test factory creates valid environment."""
        env = create_gbm_env(
            option_type="call",
            strike=100,
            maturity=0.25,
            spread_bps=10.0,
        )
        
        assert isinstance(env, GBMHedgingEnv)
        assert env.config.option_type == "call"
        assert env.config.strike == 100
    
    def test_zero_spread_gives_zero_cost(self):
        """Test that zero spread gives ZeroCost."""
        env = create_gbm_env(spread_bps=0.0)
        
        assert isinstance(env.cost_model, ZeroCost)
