"""
Streaming Environment for live/paper trading with RL agents.

Wraps the streaming engine to provide a Gymnasium-compatible environment
for deploying trained agents in live or paper trading scenarios.

Example:
    from src.q_learning.environments import StreamingEnvironment
    
    env = StreamingEnvironment(
        streaming_engine=engine,
        mode="paper",
    )
    
    state, info = env.reset()
    while env.is_running:
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.q_learning.core.protocols import RLEnvironment


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingEnvConfig:
    """Configuration for streaming environment."""
    
    # Mode
    mode: str = "paper"  # "paper" or "live"
    
    # State features
    lookback_window: int = 20
    feature_names: List[str] = field(default_factory=lambda: ["price", "volume"])
    include_position: bool = True
    include_pnl: bool = True
    include_cash: bool = True
    
    # Action space
    action_type: str = "discrete"
    n_discrete_actions: int = 5
    max_position: float = 1.0
    
    # Timing
    step_interval_ms: int = 1000  # Milliseconds between steps
    max_steps: Optional[int] = None  # None = run until stopped
    
    # Capital
    initial_capital: float = 100_000.0
    
    # Risk limits
    max_drawdown: float = 0.1  # Stop if drawdown exceeds 10%
    max_loss: float = 10_000.0  # Stop if loss exceeds this
    
    # Transaction costs
    transaction_cost: float = 0.001


# =============================================================================
# Market Data Buffer
# =============================================================================


class MarketDataBuffer:
    """
    Buffer for accumulating streaming market data.
    
    Maintains a rolling window of recent data for state construction.
    """
    
    def __init__(
        self,
        window_size: int,
        feature_names: List[str],
    ) -> None:
        """
        Initialize buffer.
        
        Parameters
        ----------
        window_size : int
            Number of historical points to keep.
        feature_names : list of str
            Names of features to track.
        """
        self.window_size = window_size
        self.feature_names = feature_names
        self._data: Dict[str, List[float]] = {
            name: [] for name in feature_names
        }
        self._timestamps: List[datetime] = []
    
    def update(self, data: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """
        Add new data point.
        
        Parameters
        ----------
        data : dict
            Feature values (e.g., {"price": 100.5, "volume": 1000}).
        timestamp : datetime, optional
            Timestamp of the data point.
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for name in self.feature_names:
            if name in data:
                self._data[name].append(data[name])
                # Keep only window_size points
                if len(self._data[name]) > self.window_size:
                    self._data[name].pop(0)
        
        self._timestamps.append(timestamp)
        if len(self._timestamps) > self.window_size:
            self._timestamps.pop(0)
    
    def get_features(self) -> Dict[str, np.ndarray]:
        """Get all features as arrays."""
        return {
            name: np.array(values)
            for name, values in self._data.items()
        }
    
    def get_latest(self, feature: str) -> Optional[float]:
        """Get latest value for a feature."""
        values = self._data.get(feature, [])
        return values[-1] if values else None
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data."""
        return all(
            len(values) >= self.window_size
            for values in self._data.values()
        )
    
    def clear(self) -> None:
        """Clear all data."""
        for name in self.feature_names:
            self._data[name] = []
        self._timestamps = []


# =============================================================================
# Streaming Environment
# =============================================================================


class StreamingEnvironment:
    """
    RL environment for live/paper trading via streaming.
    
    This environment integrates with the streaming engine to:
    - Receive real-time market data
    - Execute trades through the brokerage interface
    - Track portfolio state and P&L
    
    Can be used in:
    - Paper mode: Simulated execution with real data
    - Live mode: Real execution with real data
    
    Example:
        from src.streaming import StreamingEngine
        from src.q_learning.environments import StreamingEnvironment
        
        # Create streaming engine (see src.streaming for setup)
        engine = StreamingEngine(...)
        
        # Create environment
        env = StreamingEnvironment(
            streaming_engine=engine,
            config=StreamingEnvConfig(mode="paper"),
        )
        
        # Run agent
        state, info = env.reset()
        while env.is_running:
            action = agent.select_action(state, training=False, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Episode ended. P&L: {info['pnl']}")
                break
    """
    
    def __init__(
        self,
        streaming_engine: Optional[Any] = None,
        config: Optional[StreamingEnvConfig] = None,
        data_callback: Optional[Callable[[], Dict[str, float]]] = None,
        execute_callback: Optional[Callable[[str, float], Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize streaming environment.
        
        Parameters
        ----------
        streaming_engine : StreamingEngine, optional
            Engine for receiving data and executing trades.
            If None, uses callback functions.
        config : StreamingEnvConfig, optional
            Environment configuration.
        data_callback : callable, optional
            Function () -> dict returning market data.
            Used if streaming_engine is None.
        execute_callback : callable, optional
            Function (side, quantity) -> dict for executing trades.
            Used if streaming_engine is None.
        """
        self.config = config or StreamingEnvConfig()
        self.streaming_engine = streaming_engine
        self._data_callback = data_callback
        self._execute_callback = execute_callback
        
        # Data buffer
        self._buffer = MarketDataBuffer(
            window_size=self.config.lookback_window,
            feature_names=self.config.feature_names,
        )
        
        # State tracking
        self._step_count: int = 0
        self._is_running: bool = False
        self._start_time: Optional[datetime] = None
        
        # Portfolio tracking
        self._cash: float = 0.0
        self._position: float = 0.0
        self._position_value: float = 0.0
        self._portfolio_value: float = 0.0
        self._pnl: float = 0.0
        self._peak_value: float = 0.0
        self._returns: List[float] = []
        
        # Action space
        if self.config.action_type == "discrete":
            self.n_actions = self.config.n_discrete_actions
            self._action_map = np.linspace(-1, 1, self.n_actions)
        else:
            self.n_actions = 1
    
    @property
    def is_running(self) -> bool:
        """Check if environment is running."""
        return self._is_running
    
    @property
    def observation_space_dim(self) -> int:
        """Return observation space dimension."""
        dim = self.config.lookback_window * len(self.config.feature_names)
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
        Reset environment and start streaming.
        
        Returns
        -------
        state : ndarray
            Initial observation (may be zeros if buffer not ready).
        info : dict
            Additional information.
        """
        # Reset state
        self._step_count = 0
        self._is_running = True
        self._start_time = datetime.now()
        
        # Reset portfolio
        self._cash = self.config.initial_capital
        self._position = 0.0
        self._position_value = 0.0
        self._portfolio_value = self._cash
        self._pnl = 0.0
        self._peak_value = self._cash
        self._returns = []
        
        # Clear buffer
        self._buffer.clear()
        
        # Wait for initial data
        self._wait_for_data()
        
        state = self._get_state()
        info = self._get_info()
        
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
            Trading action.
            
        Returns
        -------
        state : ndarray
            New observation.
        reward : float
            Step reward.
        terminated : bool
            Whether to stop (risk limit hit).
        truncated : bool
            Whether max steps reached.
        info : dict
            Additional information.
        """
        if not self._is_running:
            raise RuntimeError("Environment not running. Call reset() first.")
        
        # Convert action to position target
        if self.config.action_type == "discrete":
            a_idx = int(np.clip(action, 0, self.n_actions - 1))
            target_position_frac = self._action_map[a_idx]
        else:
            target_position_frac = float(np.clip(action, -1, 1))
        
        target_position_frac *= self.config.max_position
        
        # Get current price
        current_price = self._buffer.get_latest("price")
        if current_price is None or current_price <= 0:
            current_price = 1.0  # Fallback
        
        # Calculate target position
        target_position_value = target_position_frac * self._portfolio_value
        target_position = target_position_value / current_price
        
        # Execute trade
        trade_qty = target_position - self._position
        trade_result = self._execute_trade(trade_qty, current_price)
        
        # Update portfolio
        trade_value = trade_result.get("executed_value", abs(trade_qty * current_price))
        trade_cost = trade_value * self.config.transaction_cost
        
        self._cash -= trade_qty * trade_result.get("executed_price", current_price) + trade_cost
        self._position = target_position
        
        # Wait for next data point
        self._wait_for_data()
        
        # Get new price
        new_price = self._buffer.get_latest("price")
        if new_price is None or new_price <= 0:
            new_price = current_price
        
        # Update portfolio value
        self._position_value = self._position * new_price
        old_portfolio_value = self._portfolio_value
        self._portfolio_value = self._cash + self._position_value
        
        # Update P&L and returns
        step_return = (self._portfolio_value - old_portfolio_value) / old_portfolio_value
        self._returns.append(step_return)
        self._pnl = self._portfolio_value - self.config.initial_capital
        
        # Update peak for drawdown calculation
        self._peak_value = max(self._peak_value, self._portfolio_value)
        
        self._step_count += 1
        
        # Calculate reward
        reward = step_return
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Check risk limits
        drawdown = (self._peak_value - self._portfolio_value) / self._peak_value
        if drawdown > self.config.max_drawdown:
            terminated = True
            reward = -1.0  # Penalty
        
        if self._pnl < -self.config.max_loss:
            terminated = True
            reward = -1.0
        
        # Check max steps
        if self.config.max_steps and self._step_count >= self.config.max_steps:
            truncated = True
        
        if terminated or truncated:
            self._is_running = False
        
        state = self._get_state()
        info = self._get_info()
        info["trade_qty"] = trade_qty
        info["trade_cost"] = trade_cost
        info["drawdown"] = drawdown
        
        return state, reward, terminated, truncated, info
    
    def _wait_for_data(self) -> None:
        """Wait for and process new market data."""
        import time
        
        if self._data_callback is not None:
            # Use callback
            data = self._data_callback()
            self._buffer.update(data)
        elif self.streaming_engine is not None:
            # Use streaming engine
            # This would integrate with the actual streaming infrastructure
            # For now, we just check if data is available
            pass
        else:
            # Simulate waiting
            time.sleep(self.config.step_interval_ms / 1000.0)
    
    def _execute_trade(self, quantity: float, price: float) -> Dict[str, Any]:
        """Execute a trade."""
        if abs(quantity) < 1e-8:
            return {"executed_qty": 0, "executed_price": price, "executed_value": 0}
        
        side = "buy" if quantity > 0 else "sell"
        
        if self._execute_callback is not None:
            return self._execute_callback(side, abs(quantity))
        elif self.streaming_engine is not None:
            # Use streaming engine's brokerage adapter
            # This would integrate with actual execution
            pass
        
        # Simulated execution
        return {
            "executed_qty": quantity,
            "executed_price": price,
            "executed_value": abs(quantity * price),
        }
    
    def _get_state(self) -> np.ndarray:
        """Build state observation."""
        features = []
        
        # Market data features
        data = self._buffer.get_features()
        for name in self.config.feature_names:
            if name in data:
                arr = data[name]
                # Normalize and pad
                if len(arr) > 0:
                    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
                else:
                    arr = np.zeros(self.config.lookback_window)
                
                if len(arr) < self.config.lookback_window:
                    arr = np.pad(arr, (self.config.lookback_window - len(arr), 0))
                
                features.extend(arr[-self.config.lookback_window:])
            else:
                features.extend([0.0] * self.config.lookback_window)
        
        # Position
        if self.config.include_position:
            pos_frac = self._position_value / self._portfolio_value if self._portfolio_value > 0 else 0
            features.append(pos_frac)
        
        # P&L
        if self.config.include_pnl:
            pnl_frac = self._pnl / self.config.initial_capital
            features.append(pnl_frac)
        
        # Cash
        if self.config.include_cash:
            cash_frac = self._cash / self._portfolio_value if self._portfolio_value > 0 else 1
            features.append(cash_frac)
        
        return np.array(features, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary."""
        return {
            "step": self._step_count,
            "cash": self._cash,
            "position": self._position,
            "position_value": self._position_value,
            "portfolio_value": self._portfolio_value,
            "pnl": self._pnl,
            "return": self._returns[-1] if self._returns else 0.0,
            "mode": self.config.mode,
        }
    
    def stop(self) -> None:
        """Stop the environment."""
        self._is_running = False
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        returns = np.array(self._returns) if self._returns else np.array([0.0])
        
        return {
            "total_return": (self._portfolio_value / self.config.initial_capital) - 1,
            "pnl": self._pnl,
            "sharpe_ratio": (
                np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                if len(returns) > 1 else 0.0
            ),
            "max_drawdown": (self._peak_value - self._portfolio_value) / self._peak_value,
            "n_steps": self._step_count,
        }


def create_simulated_streaming_env(
    price_series: np.ndarray,
    config: Optional[StreamingEnvConfig] = None,
) -> StreamingEnvironment:
    """
    Create a streaming environment with simulated data.
    
    Useful for testing streaming environment logic without
    actual streaming infrastructure.
    
    Parameters
    ----------
    price_series : ndarray
        Simulated price series.
    config : StreamingEnvConfig, optional
        Environment configuration.
        
    Returns
    -------
    StreamingEnvironment
        Configured environment.
    """
    prices = list(price_series)
    current_idx = [0]  # Mutable to track position
    
    def data_callback() -> Dict[str, float]:
        """Return next price from series."""
        idx = current_idx[0]
        if idx < len(prices):
            current_idx[0] += 1
            return {"price": float(prices[idx]), "volume": 1000.0}
        return {"price": float(prices[-1]), "volume": 1000.0}
    
    def execute_callback(side: str, quantity: float) -> Dict[str, Any]:
        """Simulate execution."""
        idx = min(current_idx[0], len(prices) - 1)
        price = prices[idx]
        return {
            "executed_qty": quantity if side == "buy" else -quantity,
            "executed_price": price,
            "executed_value": quantity * price,
        }
    
    return StreamingEnvironment(
        config=config,
        data_callback=data_callback,
        execute_callback=execute_callback,
    )


__all__ = [
    "StreamingEnvironment",
    "StreamingEnvConfig",
    "MarketDataBuffer",
    "create_simulated_streaming_env",
]
