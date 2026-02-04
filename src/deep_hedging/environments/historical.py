"""
Historical Hedging Environment.

Environment for hedging options using historical market data,
enabling model-agnostic hedging that learns directly from data.

Example:
    from src.deep_hedging.environments import HistoricalHedgingEnv
    from src.deep_hedging.adapters import HistoricalDataAdapter
    
    # Prepare historical data
    adapter = HistoricalDataAdapter()
    market_data = adapter.from_prices(historical_prices)
    
    # Create environment
    env = HistoricalHedgingEnv(market_data=market_data)
    
    state, info = env.reset()
    for _ in range(env.episode_length):
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class HistoricalHedgingConfig:
    """Configuration for historical hedging environment."""
    
    # Option parameters
    strike_moneyness: float = 1.0  # Strike as fraction of initial spot
    maturity_days: int = 30
    option_type: str = "call"
    notional: float = 1.0
    
    # Transaction costs
    proportional_cost: float = 0.001
    fixed_cost: float = 0.0
    
    # State features
    lookback_returns: int = 10
    include_delta: bool = True
    include_gamma: bool = False
    include_position: bool = True
    include_pnl: bool = True
    include_vol: bool = True
    
    # Reward
    reward_type: str = "risk_adjusted"  # "pnl", "risk_adjusted", "sharpe"
    risk_aversion: float = 0.1


# =============================================================================
# Historical Market Data Interface
# =============================================================================


@dataclass
class HistoricalMarketDataInterface:
    """
    Interface for historical market data.
    
    Provides access to prices, volatilities, and rates at each time step.
    """
    
    prices: np.ndarray
    volatilities: np.ndarray
    rates: np.ndarray
    dates: List[date] = field(default_factory=list)
    
    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.prices)
    
    def get_price(self, idx: int) -> float:
        """Get price at index."""
        return float(self.prices[min(idx, self.n_steps - 1)])
    
    def get_volatility(self, idx: int) -> float:
        """Get volatility at index."""
        return float(self.volatilities[min(idx, len(self.volatilities) - 1)])
    
    def get_rate(self, idx: int) -> float:
        """Get rate at index."""
        return float(self.rates[min(idx, len(self.rates) - 1)])
    
    def get_returns(self, end_idx: int, lookback: int) -> np.ndarray:
        """Get log returns over lookback window."""
        start_idx = max(0, end_idx - lookback)
        if end_idx <= start_idx:
            return np.zeros(lookback)
        
        window_prices = self.prices[start_idx:end_idx + 1]
        if len(window_prices) < 2:
            return np.zeros(lookback)
        
        returns = np.log(window_prices[1:] / window_prices[:-1])
        
        # Pad to lookback size
        if len(returns) < lookback:
            returns = np.pad(returns, (lookback - len(returns), 0))
        
        return returns[-lookback:]


# =============================================================================
# Historical Hedging Environment
# =============================================================================


class HistoricalHedgingEnv:
    """
    Hedging environment using historical market data.
    
    Instead of simulating dynamics, this environment replays
    historical price paths, enabling model-agnostic hedging
    that learns directly from real market behavior.
    
    Features:
    - Learns from actual market dynamics
    - Captures real volatility clustering and fat tails
    - No model assumptions required
    - Can be used for out-of-sample evaluation
    
    Example:
        # Prepare data
        market_data = HistoricalMarketDataInterface(
            prices=historical_prices,
            volatilities=historical_vols,
            rates=historical_rates,
        )
        
        env = HistoricalHedgingEnv(
            market_data=market_data,
            config=HistoricalHedgingConfig(maturity_days=30),
        )
        
        state, info = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
    """
    
    def __init__(
        self,
        market_data: HistoricalMarketDataInterface,
        config: Optional[HistoricalHedgingConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize environment.
        
        Parameters
        ----------
        market_data : HistoricalMarketDataInterface
            Historical market data.
        config : HistoricalHedgingConfig, optional
            Environment configuration.
        seed : int, optional
            Random seed for episode start selection.
        """
        self.market_data = market_data
        self.config = config or HistoricalHedgingConfig()
        self._rng = np.random.default_rng(seed)
        
        # State tracking
        self._episode_start_idx: int = 0
        self._current_idx: int = 0
        self._step_count: int = 0
        self._initial_spot: float = 0.0
        self._strike: float = 0.0
        self._time_to_expiry: float = 0.0
        
        # Portfolio tracking
        self._position: float = 0.0
        self._cash: float = 0.0
        self._pnl: float = 0.0
        self._pnl_history: List[float] = []
    
    @property
    def episode_length(self) -> int:
        """Episode length in steps."""
        return self.config.maturity_days
    
    @property
    def observation_space_dim(self) -> int:
        """Observation space dimension."""
        dim = 2  # Normalized spot, time
        
        if self.config.include_delta:
            dim += 1
        if self.config.include_gamma:
            dim += 1
        if self.config.include_position:
            dim += 1
        if self.config.include_pnl:
            dim += 1
        if self.config.include_vol:
            dim += 1
        
        dim += self.config.lookback_returns  # Historical returns
        
        return dim
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Selects a random starting point in the historical data.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Select random start (ensure enough data for episode)
        min_start = self.config.lookback_returns
        max_start = self.market_data.n_steps - self.config.maturity_days - 1
        
        if options and "start_idx" in options:
            self._episode_start_idx = options["start_idx"]
        else:
            if max_start > min_start:
                self._episode_start_idx = self._rng.integers(min_start, max_start)
            else:
                self._episode_start_idx = min_start
        
        self._current_idx = self._episode_start_idx
        self._step_count = 0
        
        # Initialize option
        self._initial_spot = self.market_data.get_price(self._current_idx)
        self._strike = self._initial_spot * self.config.strike_moneyness
        self._time_to_expiry = self.config.maturity_days / 252.0  # Trading days
        
        # Initialize portfolio (short option, receive premium)
        initial_vol = self.market_data.get_volatility(self._current_idx)
        initial_rate = self.market_data.get_rate(self._current_idx)
        
        option_value = self._price_option(
            self._initial_spot, initial_vol, initial_rate
        )
        
        self._cash = option_value
        self._position = 0.0
        self._pnl = 0.0
        self._pnl_history = []
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(
        self,
        action: float,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute hedging step.
        
        Parameters
        ----------
        action : float
            Hedge ratio (e.g., 1.0 for delta hedge).
        
        Returns
        -------
        state, reward, terminated, truncated, info
        """
        action = float(np.clip(action, -2.0, 2.0))
        
        # Current market state
        spot = self.market_data.get_price(self._current_idx)
        vol = self.market_data.get_volatility(self._current_idx)
        rate = self.market_data.get_rate(self._current_idx)
        
        # Compute delta for target position
        delta = self._compute_delta(spot, vol, rate)
        target_position = action * abs(delta) * self.config.notional
        
        # Execute trade
        trade = target_position - self._position
        trade_cost = (
            abs(trade * spot) * self.config.proportional_cost +
            self.config.fixed_cost * (1 if abs(trade) > 1e-8 else 0)
        )
        
        self._cash -= trade * spot + trade_cost
        self._position = target_position
        
        # Move to next time step
        self._current_idx += 1
        self._step_count += 1
        self._time_to_expiry = max(0, self._time_to_expiry - 1/252.0)
        
        # Get new market state
        new_spot = self.market_data.get_price(self._current_idx)
        new_vol = self.market_data.get_volatility(self._current_idx)
        new_rate = self.market_data.get_rate(self._current_idx)
        
        # Compute new option value
        new_option_value = self._price_option(new_spot, new_vol, new_rate)
        
        # Update P&L
        # Position value change
        position_pnl = self._position * (new_spot - spot)
        
        # Total P&L: cash + position - option liability
        old_pnl = self._pnl
        self._pnl = self._cash + self._position * new_spot - new_option_value
        
        step_pnl = self._pnl - old_pnl
        self._pnl_history.append(step_pnl)
        
        # Compute reward
        reward = self._compute_reward(step_pnl)
        
        # Check termination
        terminated = self._time_to_expiry <= 0
        truncated = self._step_count >= self.config.maturity_days
        
        # Handle expiry
        if terminated:
            # Option payoff
            if self.config.option_type == "call":
                payoff = max(new_spot - self._strike, 0)
            else:
                payoff = max(self._strike - new_spot, 0)
            
            # Final settlement
            self._cash -= payoff * self.config.notional
            self._cash += self._position * new_spot
            self._pnl = self._cash
        
        state = self._get_state()
        info = self._get_info()
        info["trade"] = trade
        info["trade_cost"] = trade_cost
        info["hedge_ratio"] = action
        
        return state, float(reward), terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """Build state observation."""
        features = []
        
        spot = self.market_data.get_price(self._current_idx)
        vol = self.market_data.get_volatility(self._current_idx)
        rate = self.market_data.get_rate(self._current_idx)
        
        # Normalized spot
        features.append(spot / self._initial_spot - 1.0)
        
        # Time to expiry
        features.append(self._time_to_expiry * 252 / self.config.maturity_days)
        
        # Delta
        if self.config.include_delta:
            delta = self._compute_delta(spot, vol, rate)
            features.append(delta)
        
        # Gamma
        if self.config.include_gamma:
            gamma = self._compute_gamma(spot, vol, rate)
            features.append(gamma * spot)
        
        # Position
        if self.config.include_position:
            delta = abs(self._compute_delta(spot, vol, rate)) + 1e-8
            features.append(self._position / delta)
        
        # P&L
        if self.config.include_pnl:
            initial_option_value = self._price_option(
                self._initial_spot, vol, rate
            )
            features.append(self._pnl / (initial_option_value + 1e-8))
        
        # Volatility
        if self.config.include_vol:
            features.append(vol)
        
        # Historical returns
        returns = self.market_data.get_returns(
            self._current_idx,
            self.config.lookback_returns,
        )
        features.extend(returns)
        
        return np.array(features, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary."""
        spot = self.market_data.get_price(self._current_idx)
        vol = self.market_data.get_volatility(self._current_idx)
        rate = self.market_data.get_rate(self._current_idx)
        
        return {
            "step": self._step_count,
            "idx": self._current_idx,
            "spot": spot,
            "time_to_expiry": self._time_to_expiry,
            "delta": self._compute_delta(spot, vol, rate),
            "position": self._position,
            "cash": self._cash,
            "pnl": self._pnl,
            "volatility": vol,
        }
    
    def _price_option(
        self,
        spot: float,
        vol: float,
        rate: float,
    ) -> float:
        """Price option using Black-Scholes."""
        T = max(self._time_to_expiry, 1e-8)
        K = self._strike
        
        d1 = (np.log(spot / K) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
        from scipy.stats import norm
        
        if self.config.option_type == "call":
            price = spot * norm.cdf(d1) - K * np.exp(-rate * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return float(price * self.config.notional)
    
    def _compute_delta(
        self,
        spot: float,
        vol: float,
        rate: float,
    ) -> float:
        """Compute Black-Scholes delta."""
        T = max(self._time_to_expiry, 1e-8)
        K = self._strike
        
        d1 = (np.log(spot / K) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        
        from scipy.stats import norm
        
        if self.config.option_type == "call":
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)
    
    def _compute_gamma(
        self,
        spot: float,
        vol: float,
        rate: float,
    ) -> float:
        """Compute Black-Scholes gamma."""
        T = max(self._time_to_expiry, 1e-8)
        K = self._strike
        
        d1 = (np.log(spot / K) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        
        from scipy.stats import norm
        
        return float(norm.pdf(d1) / (spot * vol * np.sqrt(T)))
    
    def _compute_reward(self, step_pnl: float) -> float:
        """Compute reward based on configuration."""
        if self.config.reward_type == "pnl":
            return step_pnl / self._initial_spot
        
        elif self.config.reward_type == "risk_adjusted":
            if len(self._pnl_history) > 1:
                pnl_std = np.std(self._pnl_history)
                return step_pnl / self._initial_spot - self.config.risk_aversion * pnl_std / self._initial_spot
            return step_pnl / self._initial_spot
        
        elif self.config.reward_type == "sharpe":
            if len(self._pnl_history) > 1:
                pnl_arr = np.array(self._pnl_history)
                return np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-8)
            return step_pnl / self._initial_spot
        
        return step_pnl / self._initial_spot


__all__ = [
    "HistoricalHedgingEnv",
    "HistoricalHedgingConfig",
    "HistoricalMarketDataInterface",
]
