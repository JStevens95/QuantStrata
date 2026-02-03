"""
Protocols for Deep Hedging

This module defines the abstract interfaces (protocols) for hedging environments
and agents. Any implementation that conforms to these protocols can be used with
the generic deep hedging training and evaluation infrastructure.

Design Philosophy
-----------------
Protocols are designed to be:
1. **Minimal**: Only require essential methods
2. **Flexible**: Allow different implementations (GBM, Heston, historical data)
3. **Compatible**: Work with existing RL infrastructure (RLEnvironment, RLAgent)
4. **Typed**: Full type hints for IDE support and documentation

The HedgingEnvironment protocol extends RLEnvironment with hedging-specific
methods (e.g., Greeks computation, option payoff).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

from src.deep_hedging.core.types import (
    HedgingConfig,
    HedgingState,
    HedgingEpisode,
)
from src.deep_hedging.core.costs import TransactionCostModel
from src.q_learning.core.protocols import RLEnvironment


@runtime_checkable
class HedgingEnvironment(Protocol):
    """
    Protocol for hedging simulation environments.
    
    A HedgingEnvironment simulates the evolution of the underlying asset
    and manages the hedging P&L accounting. It conforms to the RLEnvironment
    protocol and adds hedging-specific functionality.
    
    Required Methods
    ----------------
    reset(seed, options) -> (state, info)
        Reset the environment for a new episode.
        
    step(action) -> (state, reward, terminated, truncated, info)
        Execute one hedging step.
        
    compute_payoff(spot) -> float
        Compute option payoff at maturity.
        
    compute_greeks(spot, time_to_maturity) -> dict
        Compute Black-Scholes Greeks.
    
    Required Properties
    -------------------
    config : HedgingConfig
        Configuration for this environment.
        
    cost_model : TransactionCostModel
        Transaction cost model used.
    
    Example
    -------
    >>> env = GBMHedgingEnv(config=config, cost_model=cost)
    >>> state, info = env.reset(seed=42)
    >>> while True:
    ...     action = agent.select_action(state)
    ...     state, reward, terminated, truncated, info = env.step(action)
    ...     if terminated or truncated:
    ...         break
    >>> print(f"Terminal P&L: {info['terminal_pnl']:.2f}")
    """
    
    @property
    def config(self) -> HedgingConfig:
        """Configuration for this hedging environment."""
        ...
    
    @property
    def cost_model(self) -> TransactionCostModel:
        """Transaction cost model used."""
        ...
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[HedgingState, Dict[str, Any]]:
        """
        Reset the environment for a new hedging episode.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        options : dict, optional
            Additional options (e.g., specific initial spot).
        
        Returns
        -------
        state : HedgingState
            Initial state.
        info : dict
            Additional information (e.g., initial Greeks, BSM price).
        """
        ...
    
    def step(
        self,
        action: Union[float, np.ndarray],
    ) -> Tuple[HedgingState, float, bool, bool, Dict[str, Any]]:
        """
        Execute one hedging step: update position, evolve market, compute P&L.
        
        Parameters
        ----------
        action : float or ndarray
            New hedge position δ_t (or trade size, depending on action_type).
        
        Returns
        -------
        state : HedgingState
            New state after market evolution.
        reward : float
            Step reward (e.g., change in P&L minus costs).
        terminated : bool
            True if episode ended (reached maturity).
        truncated : bool
            True if episode was cut (not used for hedging).
        info : dict
            Additional information (e.g., trade size, cost, Greeks).
        
        Notes
        -----
        The state transition is:
        1. Take action (update position from δ_{t-1} to δ_t)
        2. Pay transaction cost C(δ_t - δ_{t-1})
        3. Market evolves (S_t → S_{t+1})
        4. P&L updates with hedge gains: δ_t * (S_{t+1} - S_t)
        5. If terminal: subtract option payoff
        """
        ...
    
    def compute_payoff(self, spot: float) -> float:
        """
        Compute option payoff at maturity.
        
        Parameters
        ----------
        spot : float
            Terminal spot price S_T.
        
        Returns
        -------
        float
            Option payoff max(S_T - K, 0) for call, max(K - S_T, 0) for put.
        """
        ...
    
    def compute_greeks(
        self,
        spot: float,
        time_to_maturity: float,
    ) -> Dict[str, float]:
        """
        Compute Black-Scholes Greeks at given state.
        
        Parameters
        ----------
        spot : float
            Current spot price.
        time_to_maturity : float
            Time remaining to maturity τ = T - t.
        
        Returns
        -------
        dict
            Greeks: {"delta": float, "gamma": float, "vega": float, "theta": float}.
        """
        ...
    
    def get_episode(self) -> HedgingEpisode:
        """
        Get the complete episode record after termination.
        
        Returns
        -------
        HedgingEpisode
            Full record of the hedging episode.
        
        Raises
        ------
        RuntimeError
            If called before episode termination.
        """
        ...


class BaseHedgingEnv(ABC):
    """
    Abstract base class for hedging environments.
    
    Provides common functionality for hedging simulations:
    - P&L accounting
    - State management
    - Episode recording
    - Greeks computation hooks
    
    Subclasses must implement:
    - _simulate_spot_path(): Generate spot price path
    - _compute_bsm_price(): Compute initial option price
    
    Example
    -------
    >>> class GBMHedgingEnv(BaseHedgingEnv):
    ...     def _simulate_spot_path(self, rng):
    ...         # Generate GBM path
    ...         ...
    """
    
    def __init__(
        self,
        config: HedgingConfig,
        cost_model: TransactionCostModel,
    ):
        """
        Initialize the hedging environment.
        
        Parameters
        ----------
        config : HedgingConfig
            Hedging configuration (option params, market params).
        cost_model : TransactionCostModel
            Transaction cost model.
        """
        self._config = config
        self._cost_model = cost_model
        
        # Episode state
        self._rng: Optional[np.random.Generator] = None
        self._spot_path: Optional[np.ndarray] = None
        self._current_step: int = 0
        self._current_position: float = config.initial_position
        self._current_pnl: float = 0.0
        self._positions: List[float] = []
        self._costs: List[float] = []
        self._pnl_history: List[float] = []
        self._terminated: bool = False
    
    @property
    def config(self) -> HedgingConfig:
        """Configuration for this environment."""
        return self._config
    
    @property
    def cost_model(self) -> TransactionCostModel:
        """Transaction cost model."""
        return self._cost_model
    
    @abstractmethod
    def _simulate_spot_path(self, rng: np.random.Generator) -> np.ndarray:
        """
        Simulate the spot price path for one episode.
        
        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.
        
        Returns
        -------
        ndarray, shape (n_steps + 1,)
            Spot prices S_0, S_1, ..., S_T.
        """
        pass
    
    @abstractmethod
    def _compute_bsm_price(self) -> float:
        """
        Compute Black-Scholes price for the option.
        
        Returns
        -------
        float
            BSM price at t=0.
        """
        pass
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[HedgingState, Dict[str, Any]]:
        """Reset for a new episode."""
        # Initialize RNG
        self._rng = np.random.default_rng(seed)
        
        # Simulate spot path
        self._spot_path = self._simulate_spot_path(self._rng)
        
        # Reset state
        self._current_step = 0
        self._current_position = self._config.initial_position
        
        # Initial cash = premium received
        if self._config.initial_cash is not None:
            initial_cash = self._config.initial_cash
        else:
            initial_cash = self._compute_bsm_price() * self._config.notional
        
        self._current_pnl = initial_cash
        
        # Clear episode history
        self._positions = []
        self._costs = []
        self._pnl_history = [self._current_pnl]
        self._terminated = False
        
        # Compute initial state
        spot = self._spot_path[0]
        tau = self._config.maturity
        greeks = self.compute_greeks(spot, tau)
        
        state = HedgingState(
            spot=spot,
            time=0.0,
            time_to_maturity=tau,
            position=self._current_position,
            pnl=self._current_pnl,
            step=0,
            delta_bs=greeks.get("delta"),
            gamma_bs=greeks.get("gamma"),
            vega_bs=greeks.get("vega"),
            strike=self._config.strike,
            initial_spot=self._config.spot_initial,
        )
        
        info = {
            "bsm_price": initial_cash / self._config.notional,
            "initial_delta": greeks.get("delta", 0.0),
            **greeks,
        }
        
        return state, info
    
    def step(
        self,
        action: Union[float, np.ndarray],
    ) -> Tuple[HedgingState, float, bool, bool, Dict[str, Any]]:
        """Execute one hedging step."""
        if self._terminated:
            raise RuntimeError("Episode has terminated. Call reset() first.")
        
        if self._spot_path is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Parse action (new position)
        new_position = float(np.asarray(action).item())
        trade_size = new_position - self._current_position
        
        # Current spot
        spot_now = self._spot_path[self._current_step]
        
        # Compute transaction cost
        cost = self._cost_model.compute(
            trade_size=trade_size * self._config.notional,
            spot=spot_now,
            volatility=self._config.volatility,
        )
        
        # Update position
        old_position = self._current_position
        self._current_position = new_position
        self._positions.append(new_position)
        self._costs.append(float(cost))
        
        # Market evolves
        self._current_step += 1
        spot_next = self._spot_path[self._current_step]
        
        # Hedging P&L: position * (S_{t+1} - S_t)
        hedge_pnl = new_position * (spot_next - spot_now) * self._config.notional
        
        # Update total P&L
        self._current_pnl += hedge_pnl - cost
        
        # Check if terminal
        is_terminal = (self._current_step >= self._config.n_steps)
        
        if is_terminal:
            # Subtract option payoff at maturity
            payoff = self.compute_payoff(spot_next) * self._config.notional
            self._current_pnl -= payoff
            self._terminated = True
        
        self._pnl_history.append(self._current_pnl)
        
        # Compute next state
        tau = self._config.maturity - self._current_step * self._config.dt
        tau = max(tau, 0.0)
        
        if is_terminal:
            greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        else:
            greeks = self.compute_greeks(spot_next, tau)
        
        state = HedgingState(
            spot=spot_next,
            time=self._current_step * self._config.dt,
            time_to_maturity=tau,
            position=self._current_position,
            pnl=self._current_pnl,
            step=self._current_step,
            delta_bs=greeks.get("delta"),
            gamma_bs=greeks.get("gamma"),
            vega_bs=greeks.get("vega"),
            strike=self._config.strike,
            initial_spot=self._config.spot_initial,
        )
        
        # Reward: step P&L (hedge gains - costs)
        reward = float(hedge_pnl - cost)
        if is_terminal:
            reward -= payoff
        
        info = {
            "trade_size": trade_size,
            "cost": float(cost),
            "hedge_pnl": float(hedge_pnl),
            "spot_change": float(spot_next - spot_now),
            **greeks,
        }
        
        if is_terminal:
            info["payoff"] = payoff / self._config.notional
            info["terminal_pnl"] = self._current_pnl
        
        return state, reward, is_terminal, False, info
    
    def compute_payoff(self, spot: float) -> float:
        """Compute option payoff."""
        K = self._config.strike
        if self._config.option_type == "call":
            return max(spot - K, 0.0)
        else:
            return max(K - spot, 0.0)
    
    def compute_greeks(
        self,
        spot: float,
        time_to_maturity: float,
    ) -> Dict[str, float]:
        """
        Compute Black-Scholes Greeks.
        
        Default implementation uses closed-form BSM formulas.
        Can be overridden for other models.
        """
        from src.models.analytic.black_scholes_merton.base import vanilla_greeks
        
        if time_to_maturity <= 0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        
        greeks = vanilla_greeks(
            option_type=self._config.option_type,
            spot=spot,
            strike=self._config.strike,
            expiry=time_to_maturity,
            discount_rate=self._config.risk_free_rate,
            carry=self._config.cost_of_carry,
            vol=self._config.volatility,
        )
        
        return {
            "delta": float(greeks["delta"]),
            "gamma": float(greeks["gamma"]),
            "vega": float(greeks["vega"]),
            "theta": float(greeks["theta"]),
        }
    
    def get_episode(self) -> HedgingEpisode:
        """Get episode record after termination."""
        if not self._terminated:
            raise RuntimeError("Episode has not terminated yet.")
        
        return HedgingEpisode(
            config=self._config,
            spot_path=self._spot_path.copy(),
            positions=np.array(self._positions),
            pnl_path=np.array(self._pnl_history),
            costs=np.array(self._costs),
            terminal_pnl=self._current_pnl,
            payoff=self.compute_payoff(self._spot_path[-1]),
        )


__all__ = [
    "HedgingEnvironment",
    "BaseHedgingEnv",
]
