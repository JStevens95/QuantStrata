"""
Hedging Environment for RL agents.

Wraps pricers and market simulation to provide an environment for
training hedging agents (similar to deep hedging but with standard RL interface).

This complements the deep_hedging module by providing a Gymnasium-compatible
interface that can be used with any RL agent from q_learning.

Example:
    from src.q_learning.environments import HedgingEnvironment, HedgingEnvConfig
    
    env = HedgingEnvironment(
        config=HedgingEnvConfig(
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            volatility=0.2,
        ),
    )
    
    state, info = env.reset()
    for _ in range(env.config.n_steps):
        action = agent.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from src.q_learning.core.protocols import RLEnvironment


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class HedgingEnvConfig:
    """Configuration for hedging environment."""
    
    # Option parameters
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0  # Years
    volatility: float = 0.2
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    option_type: str = "call"  # "call" or "put"
    
    # Simulation parameters
    n_steps: int = 50  # Hedging intervals
    n_paths: int = 1  # Paths per episode (for variance reduction)
    
    # Transaction costs
    proportional_cost: float = 0.001  # 10 bps
    fixed_cost: float = 0.0
    
    # State features
    include_delta: bool = True
    include_gamma: bool = True
    include_vega: bool = False
    include_time: bool = True
    include_position: bool = True
    include_pnl: bool = True
    
    # Action space
    action_type: str = "continuous"  # "continuous" or "discrete"
    n_discrete_actions: int = 11  # For discrete: hedge ratios
    max_hedge_ratio: float = 2.0  # Maximum position as multiple of delta
    
    # Reward
    reward_type: str = "risk_adjusted"  # "pnl", "risk_adjusted", "sharpe"
    risk_aversion: float = 0.1  # For risk-adjusted reward
    
    # Dynamics
    drift: float = 0.0  # Real-world drift (can differ from risk-free)

    # Optional production-grade pricing (ZeroRateCurve + GridVolSurface via library pricers).
    # When all four are set, they are used instead of inline BSM (scalar vol/rate).
    price_fn: Optional[Callable[[float, float, float, str], float]] = None
    delta_fn: Optional[Callable[[float, float, float, str], float]] = None
    gamma_fn: Optional[Callable[[float, float, float], float]] = None
    vega_fn: Optional[Callable[[float, float, float], float]] = None


# =============================================================================
# Black-Scholes Utilities
# =============================================================================


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF using error function."""
    return 0.5 * (1 + np.vectorize(lambda z: np.tanh(z * 0.7978845608))(x / np.sqrt(2)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def _black_scholes_d1d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> Tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    if time_to_expiry <= 0:
        return 0.0, 0.0
    
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    
    return d1, d2


def _black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """Compute Black-Scholes option price."""
    if time_to_expiry <= 0:
        if option_type == "call":
            return max(spot - strike, 0)
        else:
            return max(strike - spot, 0)
    
    d1, d2 = _black_scholes_d1d2(
        spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield
    )
    
    df = np.exp(-risk_free_rate * time_to_expiry)
    qf = np.exp(-dividend_yield * time_to_expiry)
    
    if option_type == "call":
        return spot * qf * _norm_cdf(np.array([d1]))[0] - strike * df * _norm_cdf(np.array([d2]))[0]
    else:
        return strike * df * _norm_cdf(np.array([-d2]))[0] - spot * qf * _norm_cdf(np.array([-d1]))[0]


def _black_scholes_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """Compute Black-Scholes delta."""
    if time_to_expiry <= 0:
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0
    
    d1, _ = _black_scholes_d1d2(
        spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield
    )
    
    qf = np.exp(-dividend_yield * time_to_expiry)
    
    if option_type == "call":
        return qf * _norm_cdf(np.array([d1]))[0]
    else:
        return qf * (_norm_cdf(np.array([d1]))[0] - 1)


def _black_scholes_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    """Compute Black-Scholes gamma."""
    if time_to_expiry <= 0:
        return 0.0
    
    d1, _ = _black_scholes_d1d2(
        spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield
    )
    
    qf = np.exp(-dividend_yield * time_to_expiry)
    
    return qf * _norm_pdf(np.array([d1]))[0] / (spot * volatility * np.sqrt(time_to_expiry))


def _black_scholes_vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    """Compute Black-Scholes vega."""
    if time_to_expiry <= 0:
        return 0.0
    
    d1, _ = _black_scholes_d1d2(
        spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield
    )
    
    qf = np.exp(-dividend_yield * time_to_expiry)
    
    return spot * qf * _norm_pdf(np.array([d1]))[0] * np.sqrt(time_to_expiry) / 100


# =============================================================================
# Hedging Environment
# =============================================================================


class HedgingEnvironment:
    """
    RL environment for option hedging.
    
    Simulates GBM paths and allows an agent to dynamically hedge
    an option position. The goal is to minimize hedging P&L variance
    while controlling transaction costs.
    
    State includes:
    - Normalized spot price (S/S0)
    - Time to expiry
    - Current delta (if enabled)
    - Current position
    - Running P&L
    
    Action is the hedge ratio (position in underlying relative to option notional).
    
    Example:
        config = HedgingEnvConfig(
            spot=100.0,
            strike=100.0,
            maturity=0.25,
            volatility=0.2,
            n_steps=50,
        )
        env = HedgingEnvironment(config)
        
        state, info = env.reset()
        total_reward = 0
        while True:
            action = delta_hedge_agent(state)  # Or RL agent
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"Episode P&L: {info['pnl']:.2f}")
    """
    
    def __init__(
        self,
        config: Optional[HedgingEnvConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize hedging environment.
        
        Parameters
        ----------
        config : HedgingEnvConfig, optional
            Environment configuration.
        seed : int, optional
            Random seed.
        """
        self.config = config or HedgingEnvConfig()
        self._rng = np.random.default_rng(seed)
        
        # Time grid
        self._dt = self.config.maturity / self.config.n_steps
        
        # State tracking
        self._step_count: int = 0
        self._spot: float = 0.0
        self._time_to_expiry: float = 0.0
        
        # Portfolio tracking
        self._position: float = 0.0  # Hedge position (shares)
        self._cash: float = 0.0
        self._option_value: float = 0.0
        self._pnl: float = 0.0
        self._cumulative_cost: float = 0.0
        self._pnl_history: List[float] = []
        
        # Action space
        if self.config.action_type == "discrete":
            self.n_actions = self.config.n_discrete_actions
            self._action_map = np.linspace(
                -self.config.max_hedge_ratio,
                self.config.max_hedge_ratio,
                self.n_actions
            )
        else:
            self.n_actions = 1

        # Production-grade pricing: use callables when all four are provided
        self._use_external_pricing = all([
            self.config.price_fn is not None,
            self.config.delta_fn is not None,
            self.config.gamma_fn is not None,
            self.config.vega_fn is not None,
        ])

    def _price(self, spot: float, strike: float, tau: float, option_type: str) -> float:
        """Option value: external callable or inline BSM."""
        if self._use_external_pricing and self.config.price_fn is not None:
            return float(self.config.price_fn(spot, strike, tau, option_type))
        return _black_scholes_price(
            spot, strike, tau,
            self.config.volatility,
            self.config.risk_free_rate,
            self.config.dividend_yield,
            option_type,
        )

    def _delta(self, spot: float, strike: float, tau: float, option_type: str) -> float:
        """Option delta: external callable or inline BSM."""
        if self._use_external_pricing and self.config.delta_fn is not None:
            return float(self.config.delta_fn(spot, strike, tau, option_type))
        return _black_scholes_delta(
            spot, strike, tau,
            self.config.volatility,
            self.config.risk_free_rate,
            self.config.dividend_yield,
            option_type,
        )

    def _gamma(self, spot: float, strike: float, tau: float) -> float:
        """Option gamma: external callable or inline BSM."""
        if self._use_external_pricing and self.config.gamma_fn is not None:
            return float(self.config.gamma_fn(spot, strike, tau))
        return _black_scholes_gamma(
            spot, strike, tau,
            self.config.volatility,
            self.config.risk_free_rate,
            self.config.dividend_yield,
        )

    def _vega(self, spot: float, strike: float, tau: float) -> float:
        """Option vega: external callable or inline BSM."""
        if self._use_external_pricing and self.config.vega_fn is not None:
            return float(self.config.vega_fn(spot, strike, tau))
        return _black_scholes_vega(
            spot, strike, tau,
            self.config.volatility,
            self.config.risk_free_rate,
            self.config.dividend_yield,
        )

    @property
    def observation_space_dim(self) -> int:
        """Return observation space dimension."""
        dim = 2  # spot, time
        if self.config.include_delta:
            dim += 1
        if self.config.include_gamma:
            dim += 1
        if self.config.include_vega:
            dim += 1
        if self.config.include_position:
            dim += 1
        if self.config.include_pnl:
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
        
        Returns
        -------
        state : ndarray
            Initial observation.
        info : dict
            Additional information.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Reset state
        self._step_count = 0
        self._spot = self.config.spot
        self._time_to_expiry = self.config.maturity
        
        # Compute initial option value
        self._option_value = self._price(
            self._spot,
            self.config.strike,
            self._time_to_expiry,
            self.config.option_type,
        )
        
        # Reset portfolio (short the option, need to hedge)
        self._position = 0.0
        self._cash = self._option_value  # Received option premium
        self._pnl = 0.0
        self._cumulative_cost = 0.0
        self._pnl_history = []
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one hedging step.
        
        Parameters
        ----------
        action : float or int
            Hedge ratio (continuous) or action index (discrete).
            
        Returns
        -------
        state : ndarray
            New observation.
        reward : float
            Step reward.
        terminated : bool
            True if option has expired.
        truncated : bool
            True if max steps reached.
        info : dict
            Additional information.
        """
        # Convert action to hedge ratio
        if self.config.action_type == "discrete":
            a_idx = int(np.clip(action, 0, self.n_actions - 1))
            hedge_ratio = self._action_map[a_idx]
        else:
            hedge_ratio = float(np.clip(
                action, 
                -self.config.max_hedge_ratio, 
                self.config.max_hedge_ratio
            ))
        
        # Compute target position
        delta = self._delta(
            self._spot,
            self.config.strike,
            self._time_to_expiry,
            self.config.option_type,
        )
        
        # Target position is hedge_ratio * delta * notional (1 option)
        target_position = hedge_ratio * abs(delta)
        
        # Execute trade
        trade_size = target_position - self._position
        trade_cost = (
            abs(trade_size * self._spot) * self.config.proportional_cost
            + self.config.fixed_cost * (1 if abs(trade_size) > 1e-8 else 0)
        )
        
        self._cash -= trade_size * self._spot + trade_cost
        self._position = target_position
        self._cumulative_cost += trade_cost
        
        # Simulate spot price movement (GBM)
        drift = (self.config.drift - self.config.dividend_yield) * self._dt
        diffusion = self.config.volatility * np.sqrt(self._dt) * self._rng.standard_normal()
        self._spot = self._spot * np.exp(drift + diffusion)
        
        # Update time
        self._time_to_expiry -= self._dt
        self._step_count += 1
        
        # Compute new option value
        new_option_value = self._price(
            self._spot,
            self.config.strike,
            max(self._time_to_expiry, 0),
            self.config.option_type,
        )
        
        # Compute P&L
        # We're short the option, so we pay the change in option value
        # We're long the hedge, so we receive the change in hedge value
        hedge_pnl = self._position * (self._spot - self._spot / np.exp(drift + diffusion))
        option_pnl = -(new_option_value - self._option_value)
        step_pnl = hedge_pnl + option_pnl - trade_cost
        
        old_pnl = self._pnl
        # Total P&L: cash + position value - option liability
        self._pnl = self._cash + self._position * self._spot - new_option_value
        self._pnl_history.append(self._pnl)
        
        self._option_value = new_option_value
        
        # Compute reward
        reward = self._compute_reward(step_pnl, self._pnl - old_pnl)
        
        # Check termination
        terminated = self._time_to_expiry <= 0
        truncated = self._step_count >= self.config.n_steps
        
        # At expiry, settle the option
        if terminated:
            # Option payoff
            if self.config.option_type == "call":
                payoff = max(self._spot - self.config.strike, 0)
            else:
                payoff = max(self.config.strike - self._spot, 0)
            
            # Final P&L adjustment
            self._cash -= payoff  # Pay option payoff
            self._cash += self._position * self._spot  # Liquidate hedge
            self._pnl = self._cash
        
        state = self._get_state()
        info = self._get_info()
        info["trade_size"] = trade_size
        info["trade_cost"] = trade_cost
        info["hedge_ratio"] = hedge_ratio
        
        return state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """Build state observation."""
        features = []
        
        # Normalized spot
        features.append(self._spot / self.config.spot - 1.0)
        
        # Time to expiry (normalized)
        if self.config.include_time:
            features.append(self._time_to_expiry / self.config.maturity)
        
        # Delta
        if self.config.include_delta:
            delta = self._delta(
                self._spot,
                self.config.strike,
                max(self._time_to_expiry, 0),
                self.config.option_type,
            )
            features.append(delta)
        
        # Gamma
        if self.config.include_gamma:
            tau = max(self._time_to_expiry, 0)
            gamma = self._gamma(self._spot, self.config.strike, tau)
            features.append(gamma * self._spot)  # Dollar gamma
        
        # Vega
        if self.config.include_vega:
            tau = max(self._time_to_expiry, 0)
            vega = self._vega(self._spot, self.config.strike, tau)
            features.append(vega)
        
        # Position
        if self.config.include_position:
            features.append(self._position)
        
        # P&L (normalized)
        if self.config.include_pnl:
            features.append(self._pnl / self._option_value if self._option_value > 0 else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary."""
        delta = self._delta(
            self._spot,
            self.config.strike,
            max(self._time_to_expiry, 0),
            self.config.option_type,
        )
        
        return {
            "step": self._step_count,
            "spot": self._spot,
            "time_to_expiry": self._time_to_expiry,
            "option_value": self._option_value,
            "delta": delta,
            "position": self._position,
            "cash": self._cash,
            "pnl": self._pnl,
            "cumulative_cost": self._cumulative_cost,
        }
    
    def _compute_reward(self, step_pnl: float, delta_pnl: float) -> float:
        """Compute reward based on configuration."""
        if self.config.reward_type == "pnl":
            reward = delta_pnl / self.config.spot
        elif self.config.reward_type == "risk_adjusted":
            # Penalize variance of P&L
            if len(self._pnl_history) > 1:
                pnl_std = np.std(self._pnl_history)
                reward = delta_pnl / self.config.spot - self.config.risk_aversion * pnl_std / self.config.spot
            else:
                reward = delta_pnl / self.config.spot
        elif self.config.reward_type == "sharpe":
            if len(self._pnl_history) > 1:
                pnl_arr = np.diff(self._pnl_history)
                sharpe = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-8)
                reward = sharpe
            else:
                reward = delta_pnl / self.config.spot
        else:
            reward = delta_pnl / self.config.spot
        
        return float(reward)
    
    def get_delta_hedge_action(self) -> float:
        """
        Get action for pure delta hedging (benchmark).
        
        Returns
        -------
        float
            Hedge ratio of 1.0 (perfect delta hedge).
        """
        return 1.0


__all__ = [
    "HedgingEnvironment",
    "HedgingEnvConfig",
]
