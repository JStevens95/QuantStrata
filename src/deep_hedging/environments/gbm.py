"""
GBM Hedging Environment

Hedging environment under Geometric Brownian Motion (Black-Scholes) dynamics.

This module wraps the existing `GbmDynamicsSimulator` from `src/models/dynamics/`
to provide an RL-compatible hedging environment. The path simulation logic is
reused; this module adds:

- RL environment interface (reset, step, reward)
- P&L accounting and transaction costs
- Position tracking and episode recording
- State management for the hedging agent

Model
-----
Under the physical measure P:

    dS_t = μ S_t dt + σ S_t dW_t

where:
- μ: drift (expected return)
- σ: volatility
- W_t: standard Brownian motion

For hedging, we typically use μ = r (risk-free rate) to be consistent with
risk-neutral pricing, but this can be configured.

Example
-------
>>> from src.deep_hedging.environments import GBMHedgingEnv
>>> from src.deep_hedging.core import HedgingConfig, ProportionalCost
>>> 
>>> config = HedgingConfig(
...     option_type="call",
...     strike=100.0,
...     maturity=0.25,
...     spot_initial=100.0,
...     volatility=0.2,
...     risk_free_rate=0.05,
...     n_steps=63,
... )
>>> cost = ProportionalCost(spread_bps=10.0)
>>> env = GBMHedgingEnv(config, cost)
>>> 
>>> state, info = env.reset(seed=42)
>>> print(f"Initial spot: {state.spot:.2f}, Delta: {state.delta_bs:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.deep_hedging.core.protocols import BaseHedgingEnv
from src.deep_hedging.core.types import HedgingConfig, HedgingState
from src.deep_hedging.core.costs import TransactionCostModel, ZeroCost
from src.models.analytic.black_scholes_merton.base import vanilla_price

# Reuse the existing GBM dynamics simulator
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator


class GBMHedgingEnv(BaseHedgingEnv):
    """
    Hedging environment with Geometric Brownian Motion dynamics.
    
    This environment wraps the existing `GbmDynamicsSimulator` for path simulation
    and adds hedging-specific functionality: P&L accounting, transaction costs,
    position tracking, and the RL interface.
    
    Architecture
    ------------
    - **Path simulation:** Delegates to `src/models/dynamics/gbm_dynamics.GbmDynamicsSimulator`
    - **Hedging logic:** Managed by `BaseHedgingEnv` (P&L, costs, state)
    - **Greeks:** Uses `src/models/analytic/black_scholes_merton/` formulas
    
    Mathematical Setup
    ------------------
    The spot price evolves as:
    
        S_{t+Δt} = S_t · exp((μ - σ²/2)Δt + σ√Δt Z)
    
    where Z ~ N(0,1). This uses the "exact" scheme from GbmDynamicsSimulator.
    
    The hedging P&L over one step is:
    
        ΔP&L = δ_t · (S_{t+Δt} - S_t) - C(Δδ, S_t)
    
    At maturity:
    
        P&L_T = V_0 + Σ δ_t (ΔS_t) - Σ C_t - Φ(S_T)
    
    where V_0 is the initial option premium and Φ(S_T) is the payoff.
    
    Parameters
    ----------
    config : HedgingConfig
        Option and market configuration.
    cost_model : TransactionCostModel
        Transaction cost model. Default: ZeroCost().
    use_risk_neutral_drift : bool
        If True, use μ = r (risk-neutral measure).
        If False, use μ = r + λσ where λ is market price of risk.
        Default: True.
    market_price_of_risk : float
        Market price of risk λ (only used if use_risk_neutral_drift=False).
        Default: 0.0.
    
    Attributes
    ----------
    config : HedgingConfig
        The configuration.
    cost_model : TransactionCostModel
        The cost model.
    drift : float
        The drift parameter μ used in simulation.
    dynamics : GbmDynamicsSimulator
        The underlying dynamics simulator (reused from models/).
    
    Example
    -------
    >>> env = GBMHedgingEnv(config, cost_model)
    >>> state, info = env.reset(seed=42)
    >>> 
    >>> # Delta hedge for entire episode
    >>> while True:
    ...     action = state.delta_bs  # Use BSM delta
    ...     state, reward, done, _, info = env.step(action)
    ...     if done:
    ...         break
    >>> 
    >>> print(f"Terminal P&L: {info['terminal_pnl']:.4f}")
    """
    
    def __init__(
        self,
        config: HedgingConfig,
        cost_model: Optional[TransactionCostModel] = None,
        use_risk_neutral_drift: bool = True,
        market_price_of_risk: float = 0.0,
    ):
        """
        Initialize the GBM hedging environment.
        
        Parameters
        ----------
        config : HedgingConfig
            Option and market configuration.
        cost_model : TransactionCostModel, optional
            Transaction cost model. Default: ZeroCost().
        use_risk_neutral_drift : bool
            Use μ = r (risk-neutral) or μ = r + λσ (physical).
        market_price_of_risk : float
            λ in μ = r + λσ (only if use_risk_neutral_drift=False).
        """
        if cost_model is None:
            cost_model = ZeroCost()
        
        super().__init__(config, cost_model)
        
        self._use_risk_neutral_drift = use_risk_neutral_drift
        self._market_price_of_risk = market_price_of_risk
        
        # Compute drift
        if use_risk_neutral_drift:
            self._drift = config.risk_free_rate
        else:
            self._drift = config.risk_free_rate + market_price_of_risk * config.volatility
        
        # Create the dynamics simulator (reused from src/models/dynamics/)
        self._dynamics = GbmDynamicsSimulator(
            drift=self._drift,
            vol=config.volatility,
        )
    
    @property
    def drift(self) -> float:
        """Drift parameter μ used in simulation."""
        return self._drift
    
    @property
    def dynamics(self) -> GbmDynamicsSimulator:
        """The underlying GBM dynamics simulator."""
        return self._dynamics
    
    def _simulate_spot_path(self, rng: np.random.Generator) -> np.ndarray:
        """
        Simulate GBM spot price path using the existing GbmDynamicsSimulator.
        
        This delegates to `src/models/dynamics/gbm_dynamics.GbmDynamicsSimulator`,
        which uses the exact discretisation scheme by default.
        
        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.
        
        Returns
        -------
        ndarray, shape (n_steps + 1,)
            Spot prices S_0, S_1, ..., S_T.
        """
        n_steps = self._config.n_steps
        
        # Generate standard normals for the dynamics simulator
        normals = rng.standard_normal((1, n_steps))  # Shape (1, n_steps) for 1 path
        
        # Use the existing dynamics simulator
        paths = self._dynamics.simulate_paths(
            spot0=self._config.spot_initial,
            maturity=self._config.maturity,
            n_steps=n_steps,
            n_paths=1,
            normals=normals,
            scheme="exact",
        )
        
        # Return the single path (shape n_steps + 1)
        return paths[0, :]
    
    def _compute_bsm_price(self) -> float:
        """
        Compute Black-Scholes price for the option at t=0.
        
        Returns
        -------
        float
            BSM price per unit notional.
        """
        return vanilla_price(
            option_type=self._config.option_type,
            spot=self._config.spot_initial,
            strike=self._config.strike,
            expiry=self._config.maturity,
            discount_rate=self._config.risk_free_rate,
            carry=self._config.cost_of_carry,
            vol=self._config.volatility,
        )
    
    def simulate_paths(
        self,
        n_paths: int,
        seed: Optional[int] = None,
        antithetic: bool = True,
    ) -> np.ndarray:
        """
        Simulate multiple GBM paths for batch training.
        
        This uses the existing GbmDynamicsSimulator for efficient batch simulation.
        
        Parameters
        ----------
        n_paths : int
            Number of paths to simulate.
        seed : int, optional
            Random seed.
        antithetic : bool
            If True, use antithetic variates (generate n_paths/2 paths
            and reflect them). Reduces variance.
        
        Returns
        -------
        ndarray, shape (n_paths, n_steps + 1)
            Simulated spot paths.
        """
        rng = np.random.default_rng(seed)
        n_steps = self._config.n_steps
        
        if antithetic:
            # Generate half the paths and use antithetic variates
            n_half = n_paths // 2
            normals_half = rng.standard_normal((n_half, n_steps))
            normals = np.vstack([normals_half, -normals_half])
            
            if n_paths % 2 == 1:
                # Add one more if odd
                normals = np.vstack([normals, rng.standard_normal((1, n_steps))])
        else:
            normals = rng.standard_normal((n_paths, n_steps))
        
        # Use the existing dynamics simulator for batch simulation
        return self._dynamics.simulate_paths(
            spot0=self._config.spot_initial,
            maturity=self._config.maturity,
            n_steps=n_steps,
            n_paths=n_paths,
            normals=normals,
            scheme="exact",
        )
    
    def __repr__(self) -> str:
        return (
            f"GBMHedgingEnv("
            f"option={self._config.option_type}, "
            f"K={self._config.strike}, "
            f"T={self._config.maturity}, "
            f"σ={self._config.volatility}, "
            f"n_steps={self._config.n_steps}, "
            f"cost={self._cost_model})"
        )


def create_gbm_env(
    option_type: str = "call",
    strike: float = 100.0,
    maturity: float = 0.25,
    spot: float = 100.0,
    volatility: float = 0.2,
    rate: float = 0.05,
    n_steps: int = 63,
    spread_bps: float = 10.0,
    **kwargs,
) -> GBMHedgingEnv:
    """
    Convenience function to create a GBM hedging environment.
    
    Parameters
    ----------
    option_type : str
        "call" or "put".
    strike : float
        Option strike.
    maturity : float
        Time to maturity in years.
    spot : float
        Initial spot price.
    volatility : float
        Volatility σ.
    rate : float
        Risk-free rate r.
    n_steps : int
        Number of hedging steps.
    spread_bps : float
        Bid-ask spread in basis points.
    **kwargs : dict
        Additional arguments for GBMHedgingEnv.
    
    Returns
    -------
    GBMHedgingEnv
        Configured environment.
    
    Example
    -------
    >>> env = create_gbm_env(
    ...     option_type="call",
    ...     strike=100,
    ...     maturity=0.25,
    ...     spread_bps=5.0,
    ... )
    """
    from src.deep_hedging.core.costs import ProportionalCost
    
    config = HedgingConfig(
        option_type=option_type,
        strike=strike,
        maturity=maturity,
        spot_initial=spot,
        volatility=volatility,
        risk_free_rate=rate,
        n_steps=n_steps,
    )
    
    cost_model = ProportionalCost(spread_bps=spread_bps) if spread_bps > 0 else ZeroCost()
    
    return GBMHedgingEnv(config, cost_model, **kwargs)


__all__ = [
    "GBMHedgingEnv",
    "create_gbm_env",
]
