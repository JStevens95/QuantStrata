"""
Delta Hedging Agent (Benchmark)

Classical Black-Scholes delta hedging agent used as a benchmark for
comparing deep hedging performance.

Theory
------
Under the Black-Scholes model, the replicating portfolio for a European
option requires holding Δ = ∂V/∂S units of the underlying at each instant.

For a call option:
    Δ = e^{(b-r)τ} N(d₁)

For a put option:
    Δ = e^{(b-r)τ} (N(d₁) - 1)

where:
    d₁ = [ln(S/K) + (b + σ²/2)τ] / (σ√τ)
    b = cost of carry (r - q for FX)
    τ = time to maturity

Limitations
-----------
Delta hedging is optimal only when:
1. Trading is continuous (no discrete rebalancing)
2. No transaction costs
3. Volatility is constant and known
4. Markets are complete

When these assumptions fail, delta hedging incurs hedging error and/or
excessive transaction costs. Deep hedging aims to learn a better policy.

Example
-------
>>> from src.deep_hedging.agents import DeltaHedgingAgent
>>> from src.deep_hedging.environments import GBMHedgingEnv
>>> 
>>> agent = DeltaHedgingAgent()
>>> env = GBMHedgingEnv(config, cost_model)
>>> 
>>> state, _ = env.reset()
>>> while True:
...     action = agent.select_action(state)  # Returns BSM delta
...     state, reward, done, _, _ = env.step(action)
...     if done:
...         break
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.deep_hedging.core.types import HedgingState


@dataclass
class DeltaHedgingAgent:
    """
    Black-Scholes delta hedging agent.
    
    This agent simply returns the BSM delta from the state as its action.
    It serves as a benchmark for comparing deep hedging performance.
    
    The agent conforms to the RLAgent protocol but:
    - Does not learn (update() is a no-op)
    - Has no trainable parameters
    - Action is deterministic (no exploration)
    
    Parameters
    ----------
    delta_scaling : float
        Scale factor applied to the BSM delta. Default 1.0 (exact delta).
        Can be used to test under/over-hedging.
    clip_delta : tuple of (float, float), optional
        Clip delta to this range. Default None (no clipping).
    use_gamma_adjustment : bool
        If True, add a gamma-based adjustment for discrete hedging.
        This is the Leland-style adjustment: δ_adj = δ + 0.5 * Γ * (dS)
        Default False.
    
    Attributes
    ----------
    name : str
        Agent name for logging.
    
    Example
    -------
    >>> agent = DeltaHedgingAgent(delta_scaling=1.0)
    >>> state = HedgingState(spot=100, ..., delta_bs=0.55)
    >>> action = agent.select_action(state)
    >>> print(action)  # 0.55
    """
    
    delta_scaling: float = 1.0
    clip_delta: Optional[tuple] = None
    use_gamma_adjustment: bool = False
    
    # Store recent state for potential gamma adjustment
    _last_spot: Optional[float] = None
    
    @property
    def name(self) -> str:
        """Agent name."""
        return "DeltaHedging"
    
    def select_action(
        self,
        state: Union[HedgingState, np.ndarray, Dict[str, Any]],
        *,
        training: bool = False,
        explore: bool = False,
    ) -> float:
        """
        Select action (hedge position) based on BSM delta.
        
        Parameters
        ----------
        state : HedgingState or ndarray or dict
            Current hedging state. If HedgingState, uses delta_bs attribute.
            If ndarray, assumes delta is at a known index.
            If dict, looks for 'delta_bs' key.
        training : bool
            Ignored (agent doesn't learn).
        explore : bool
            Ignored (action is deterministic).
        
        Returns
        -------
        float
            Hedge position (BSM delta, possibly scaled/adjusted).
        """
        # Extract delta from state
        if isinstance(state, HedgingState):
            delta = state.delta_bs
            gamma = state.gamma_bs
            spot = state.spot
        elif isinstance(state, dict):
            delta = state.get("delta_bs", state.get("delta", 0.0))
            gamma = state.get("gamma_bs", state.get("gamma", 0.0))
            spot = state.get("spot", 0.0)
        elif isinstance(state, np.ndarray):
            # Assume standard feature order: [log_moneyness, tau, position, pnl, delta, ...]
            # This is fragile; prefer HedgingState
            delta = state[4] if len(state) > 4 else 0.0
            gamma = state[5] if len(state) > 5 else 0.0
            spot = np.exp(state[0]) if len(state) > 0 else 0.0  # Assuming log-moneyness
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")
        
        if delta is None:
            delta = 0.0
        
        # Apply scaling
        action = self.delta_scaling * delta
        
        # Optional gamma adjustment for discrete hedging
        if self.use_gamma_adjustment and gamma is not None and self._last_spot is not None:
            dS = spot - self._last_spot
            action += 0.5 * gamma * dS
        
        self._last_spot = spot
        
        # Optional clipping
        if self.clip_delta is not None:
            action = np.clip(action, self.clip_delta[0], self.clip_delta[1])
        
        return float(action)
    
    def update(
        self,
        transitions: Optional[List[Any]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Update agent (no-op for delta hedging).
        
        Delta hedging is model-based, not learned. This method exists
        only for compatibility with the RLAgent protocol.
        
        Returns
        -------
        None
            No training metrics.
        """
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get agent parameters.
        
        Returns
        -------
        dict
            Configuration parameters (not learned weights).
        """
        return {
            "delta_scaling": self.delta_scaling,
            "clip_delta": self.clip_delta,
            "use_gamma_adjustment": self.use_gamma_adjustment,
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set agent parameters.
        
        Parameters
        ----------
        params : dict
            Parameters from get_parameters().
        """
        if "delta_scaling" in params:
            self.delta_scaling = params["delta_scaling"]
        if "clip_delta" in params:
            self.clip_delta = params["clip_delta"]
        if "use_gamma_adjustment" in params:
            self.use_gamma_adjustment = params["use_gamma_adjustment"]
    
    def reset(self) -> None:
        """Reset agent state (clear last spot for gamma adjustment)."""
        self._last_spot = None
    
    def __repr__(self) -> str:
        return f"DeltaHedgingAgent(scaling={self.delta_scaling})"


class NoHedgingAgent:
    """
    No-hedging agent (naked position).
    
    This agent always returns 0 as the action, meaning no hedge is held.
    Useful as a baseline to show the value of hedging.
    """
    
    @property
    def name(self) -> str:
        return "NoHedging"
    
    def select_action(
        self,
        state: Any,
        *,
        training: bool = False,
        explore: bool = False,
    ) -> float:
        """Return zero hedge position."""
        return 0.0
    
    def update(
        self,
        transitions: Optional[List[Any]] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        return {}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        pass


__all__ = [
    "DeltaHedgingAgent",
    "NoHedgingAgent",
]
