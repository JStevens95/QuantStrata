"""
Trading Environment for RL agents.

Wraps the backtesting infrastructure to provide a Gymnasium-compatible
environment for training trading agents.

Example:
    from src.q_learning.environments import TradingEnvironment
    
    env = TradingEnvironment(
        data_provider=historical_data,
        initial_capital=1_000_000,
        transaction_cost=0.001,
    )
    
    state, info = env.reset()
    for _ in range(1000):
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state, info = env.reset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.q_learning.core.protocols import RLEnvironment


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""
    
    # Capital and costs
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001  # Proportional cost
    slippage: float = 0.0  # Price slippage
    
    # Episode settings
    max_steps: int = 252  # Trading days
    
    # State features
    lookback_window: int = 20  # Historical window for features
    include_position: bool = True
    include_pnl: bool = True
    include_cash: bool = True
    
    # Action space
    action_type: str = "discrete"  # "discrete" or "continuous"
    n_discrete_actions: int = 5  # For discrete: [-1, -0.5, 0, 0.5, 1] position targets
    max_position: float = 1.0  # Maximum position as fraction of capital
    
    # Reward
    reward_type: str = "pnl"  # "pnl", "sharpe", "log_return"
    reward_scale: float = 1.0


# =============================================================================
# Simple Data Provider Protocol
# =============================================================================


class SimpleDataProvider:
    """
    Simple data provider that wraps price arrays.
    
    For more sophisticated data, use src.backtesting.data providers.
    """
    
    def __init__(
        self,
        prices: np.ndarray,
        dates: Optional[Sequence[date]] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize data provider.
        
        Parameters
        ----------
        prices : ndarray
            Price series of shape (n_steps,) or (n_steps, n_assets).
        dates : sequence of date, optional
            Dates corresponding to prices.
        features : ndarray, optional
            Additional features of shape (n_steps, n_features).
        """
        self.prices = np.atleast_2d(prices)
        if self.prices.shape[0] == 1:
            self.prices = self.prices.T
        
        self.n_steps = self.prices.shape[0]
        self.n_assets = self.prices.shape[1] if self.prices.ndim > 1 else 1
        
        if dates is None:
            from datetime import timedelta
            base = date.today()
            dates = [base + timedelta(days=i) for i in range(self.n_steps)]
        self.dates = list(dates)
        
        self.features = features
    
    def get_dates(self) -> Sequence[date]:
        """Return all available dates."""
        return self.dates
    
    def get_price(self, idx: int) -> np.ndarray:
        """Get price at index."""
        return self.prices[idx]
    
    def get_window(self, start_idx: int, window_size: int) -> np.ndarray:
        """Get price window [start_idx : start_idx + window_size], shape (window_size, n_assets)."""
        end_idx = start_idx + window_size
        return self.prices[start_idx:end_idx].copy()
    
    def get_features(self, idx: int) -> Optional[np.ndarray]:
        """Get features at index."""
        if self.features is None:
            return None
        return self.features[idx]


# =============================================================================
# Trading Environment
# =============================================================================


class TradingEnvironment:
    """
    Reinforcement learning environment for trading.
    
    Implements the RLEnvironment protocol with:
    - State: price returns, position, PnL, optional features
    - Action: position target (discrete or continuous)
    - Reward: PnL, Sharpe, or log return based
    - Episode: runs through historical data window
    
    Example:
        # Create with synthetic data
        prices = np.cumprod(1 + np.random.randn(500) * 0.01)
        env = TradingEnvironment(
            data_provider=SimpleDataProvider(prices),
            config=TradingEnvConfig(max_steps=100),
        )
        
        state, info = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    """
    
    def __init__(
        self,
        data_provider: Any,
        config: Optional[TradingEnvConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize trading environment.
        
        Parameters
        ----------
        data_provider : SimpleDataProvider or compatible
            Provider of price and feature data.
        config : TradingEnvConfig, optional
            Environment configuration.
        seed : int, optional
            Random seed for episode starting points.
        """
        self.data_provider = data_provider
        self.config = config or TradingEnvConfig()
        self._rng = np.random.default_rng(seed)
        
        # State tracking
        self._current_idx: int = 0
        self._start_idx: int = 0
        self._step_count: int = 0
        
        # Portfolio tracking
        self._cash: float = 0.0
        self._position: float = 0.0  # Position value (not quantity)
        self._position_qty: float = 0.0  # Actual quantity
        self._portfolio_value: float = 0.0
        self._pnl: float = 0.0
        self._returns: List[float] = []
        
        # History for state features
        self._price_history: List[float] = []
        
        # Action space info
        if self.config.action_type == "discrete":
            self.n_actions = self.config.n_discrete_actions
            # Map discrete actions to position targets
            self._action_map = np.linspace(-1, 1, self.n_actions)
        else:
            self.n_actions = 1  # Continuous action
    
    @property
    def observation_space_dim(self) -> int:
        """Return observation space dimension."""
        dim = self.config.lookback_window  # Returns
        if self.config.include_position:
            dim += 1
        if self.config.include_pnl:
            dim += 1
        if self.config.include_cash:
            dim += 1
        return dim
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Options including 'start_idx' for specific starting point.
            
        Returns
        -------
        state : ndarray
            Initial observation.
        info : dict
            Additional information.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Determine starting index
        n_data = len(self.data_provider.get_dates())
        min_start = self.config.lookback_window
        max_start = n_data - self.config.max_steps - 1
        
        if options and "start_idx" in options:
            self._start_idx = options["start_idx"]
        else:
            if max_start > min_start:
                self._start_idx = self._rng.integers(min_start, max_start)
            else:
                self._start_idx = min_start
        
        self._current_idx = self._start_idx
        self._step_count = 0
        
        # Reset portfolio
        self._cash = self.config.initial_capital
        self._position = 0.0
        self._position_qty = 0.0
        self._portfolio_value = self._cash
        self._pnl = 0.0
        self._returns = []
        
        # Initialize price history
        self._price_history = []
        for i in range(self.config.lookback_window):
            idx = self._start_idx - self.config.lookback_window + i
            if idx >= 0:
                price = self._get_price(idx)
                self._price_history.append(float(price[0] if hasattr(price, '__len__') else price))
        
        state = self._get_state()
        info = {
            "step": 0,
            "idx": self._current_idx,
            "cash": self._cash,
            "position": self._position,
            "portfolio_value": self._portfolio_value,
        }
        
        return state, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step.
        
        Parameters
        ----------
        action : int or float
            Action (discrete index or continuous position target).
            
        Returns
        -------
        state : ndarray
            New observation.
        reward : float
            Step reward.
        terminated : bool
            Whether episode ended naturally.
        truncated : bool
            Whether episode was truncated.
        info : dict
            Additional information.
        """
        # Convert action to target position (as fraction of capital)
        if self.config.action_type == "discrete":
            a_idx = int(np.clip(action, 0, self.n_actions - 1))
            target_position_frac = self._action_map[a_idx]
        else:
            a = np.asarray(action)
            target_position_frac = float(np.clip(a.flat[0] if a.size else 0.0, -1.0, 1.0))
        
        # Scale by max position
        target_position_frac *= self.config.max_position
        
        # Get current price
        current_price = self._get_price(self._current_idx)
        current_price = float(current_price[0] if hasattr(current_price, '__len__') else current_price)
        
        # Calculate target position value and quantity
        target_position_value = target_position_frac * self._portfolio_value
        target_qty = target_position_value / current_price if current_price > 0 else 0.0
        
        # Execute trade
        trade_qty = target_qty - self._position_qty
        trade_value = abs(trade_qty * current_price)
        transaction_cost = trade_value * self.config.transaction_cost
        
        # Update portfolio
        self._cash -= trade_qty * current_price + transaction_cost
        self._position_qty = target_qty
        self._position = self._position_qty * current_price
        
        # Move to next step
        self._current_idx += 1
        self._step_count += 1
        
        # Get new price and update portfolio
        new_price = self._get_price(self._current_idx)
        new_price = float(new_price[0] if hasattr(new_price, '__len__') else new_price)
        
        # Update position value with new price
        self._position = self._position_qty * new_price
        new_portfolio_value = self._cash + self._position
        
        # Calculate return
        step_return = (new_portfolio_value - self._portfolio_value) / self._portfolio_value
        self._returns.append(step_return)
        
        # Update state
        old_pnl = self._pnl
        self._pnl = new_portfolio_value - self.config.initial_capital
        self._portfolio_value = new_portfolio_value
        
        # Update price history
        self._price_history.append(new_price)
        if len(self._price_history) > self.config.lookback_window:
            self._price_history.pop(0)
        
        # Calculate reward
        reward = self._compute_reward(step_return, self._pnl - old_pnl)
        
        # Check termination
        terminated = False
        truncated = self._step_count >= self.config.max_steps
        
        # Check for bankruptcy
        if self._portfolio_value <= 0:
            terminated = True
            reward = -10.0  # Large penalty for bankruptcy
        
        state = self._get_state()
        info = {
            "step": self._step_count,
            "idx": self._current_idx,
            "cash": self._cash,
            "position": self._position,
            "position_qty": self._position_qty,
            "portfolio_value": self._portfolio_value,
            "pnl": self._pnl,
            "return": step_return,
            "transaction_cost": transaction_cost,
            "price": new_price,
        }
        
        return state, reward, terminated, truncated, info
    
    def _get_price(self, idx: int) -> np.ndarray:
        """Get price at index."""
        return self.data_provider.get_price(idx)
    
    def _get_state(self) -> np.ndarray:
        """Build state observation."""
        features = []
        
        # Price returns over lookback window
        if len(self._price_history) >= 2:
            prices = np.array(self._price_history)
            returns = np.diff(prices) / prices[:-1]
            # Pad to lookback_window size
            if len(returns) < self.config.lookback_window:
                returns = np.pad(returns, (self.config.lookback_window - len(returns), 0))
            features.extend(returns[-self.config.lookback_window:])
        else:
            features.extend([0.0] * self.config.lookback_window)
        
        # Position as fraction of portfolio
        if self.config.include_position:
            pos_frac = self._position / self._portfolio_value if self._portfolio_value > 0 else 0.0
            features.append(pos_frac)
        
        # PnL as fraction of initial capital
        if self.config.include_pnl:
            pnl_frac = self._pnl / self.config.initial_capital
            features.append(pnl_frac)
        
        # Cash as fraction of portfolio
        if self.config.include_cash:
            cash_frac = self._cash / self._portfolio_value if self._portfolio_value > 0 else 1.0
            features.append(cash_frac)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_reward(self, step_return: float, step_pnl: float) -> float:
        """Compute reward based on configuration."""
        if self.config.reward_type == "pnl":
            reward = step_pnl / self.config.initial_capital
        elif self.config.reward_type == "log_return":
            reward = np.log(1 + step_return) if step_return > -1 else -10.0
        elif self.config.reward_type == "sharpe":
            # Running Sharpe (simplified)
            if len(self._returns) >= 2:
                ret_arr = np.array(self._returns)
                mean_ret = np.mean(ret_arr)
                std_ret = np.std(ret_arr) + 1e-8
                reward = mean_ret / std_ret
            else:
                reward = step_return
        else:
            reward = step_return
        
        return float(reward * self.config.reward_scale)
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Get current portfolio performance metrics."""
        returns = np.array(self._returns) if self._returns else np.array([0.0])
        
        metrics = {
            "total_return": (self._portfolio_value / self.config.initial_capital) - 1,
            "pnl": self._pnl,
            "sharpe_ratio": (
                np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                if len(returns) > 1 else 0.0
            ),
            "max_drawdown": self._compute_max_drawdown(),
            "n_steps": self._step_count,
        }
        
        return metrics
    
    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown."""
        if not self._returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(self._returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        
        return float(np.max(drawdowns))


def create_trading_env_from_prices(
    prices: np.ndarray,
    config: Optional[TradingEnvConfig] = None,
    **kwargs: Any,
) -> TradingEnvironment:
    """
    Create a trading environment from price data.
    
    Parameters
    ----------
    prices : ndarray
        Price series.
    config : TradingEnvConfig, optional
        Environment configuration.
    **kwargs
        Additional arguments passed to TradingEnvironment.
        
    Returns
    -------
    TradingEnvironment
        Configured environment.
    """
    provider = SimpleDataProvider(prices)
    return TradingEnvironment(provider, config, **kwargs)


__all__ = [
    "TradingEnvironment",
    "TradingEnvConfig",
    "SimpleDataProvider",
    "create_trading_env_from_prices",
]
