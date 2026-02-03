"""
Data Types for Deep Hedging

This module defines the core data structures used throughout the deep hedging
framework: configurations, states, results, and episode recordings.

Design Philosophy
-----------------
All types are:
- Immutable where possible (dataclasses with frozen=True for configs)
- Serialisable (to_dict/from_dict methods)
- Well-documented with clear semantics
- Compatible with both NumPy and JAX arrays
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Literal
import json

import numpy as np


# =============================================================================
# Configuration Types
# =============================================================================

@dataclass
class HedgingConfig:
    """
    Configuration for a hedging simulation/training run.
    
    This specifies the option being hedged, market parameters, and
    simulation settings. It does NOT specify the hedging strategy
    (that's determined by the agent).
    
    Parameters
    ----------
    option_type : str
        Type of option: "call" or "put".
    strike : float
        Option strike price K.
    maturity : float
        Time to maturity T in years.
    spot_initial : float
        Initial spot price S₀.
    volatility : float
        Volatility σ (for GBM) or initial vol (for stochastic vol).
    risk_free_rate : float
        Risk-free rate r (continuous compounding).
    dividend_yield : float
        Continuous dividend yield q (default 0).
    n_steps : int
        Number of hedging steps (rebalancing times).
    notional : float
        Option notional (number of units). Default 1.0.
    initial_position : float
        Initial hedge position δ₀. Default 0.0.
    initial_cash : float
        Initial cash (premium received). If None, use BSM price.
    
    Example
    -------
    >>> config = HedgingConfig(
    ...     option_type="call",
    ...     strike=100.0,
    ...     maturity=0.25,  # 3 months
    ...     spot_initial=100.0,
    ...     volatility=0.2,
    ...     risk_free_rate=0.05,
    ...     n_steps=63,  # Daily rebalancing for 3 months
    ... )
    """
    
    # Option specification
    option_type: Literal["call", "put"] = "call"
    strike: float = 100.0
    maturity: float = 0.25  # Years
    notional: float = 1.0
    
    # Market parameters
    spot_initial: float = 100.0
    volatility: float = 0.2
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    
    # Simulation settings
    n_steps: int = 63  # ~daily for 3 months
    
    # Initial conditions
    initial_position: float = 0.0
    initial_cash: Optional[float] = None  # None = use BSM price
    
    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type}")
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.spot_initial <= 0:
            raise ValueError(f"spot_initial must be positive, got {self.spot_initial}")
        if self.volatility < 0:
            raise ValueError(f"volatility must be non-negative, got {self.volatility}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
    
    @property
    def dt(self) -> float:
        """Time step size Δt = T/n_steps."""
        return self.maturity / self.n_steps
    
    @property
    def cost_of_carry(self) -> float:
        """Cost of carry b = r - q."""
        return self.risk_free_rate - self.dividend_yield
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HedgingConfig":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class HedgingState:
    """
    State of a hedging simulation at a single time step.
    
    This represents all information available to the hedging agent at time t.
    The state is used as input to the policy network.
    
    Parameters
    ----------
    spot : float
        Current spot price S_t.
    time : float
        Current time t.
    time_to_maturity : float
        Time remaining τ = T - t.
    position : float
        Current hedge position δ_{t-1} (held from previous step).
    pnl : float
        Cumulative P&L up to time t.
    step : int
        Current step index (0 to n_steps-1).
    
    Greeks (Optional)
    -----------------
    delta_bs : float, optional
        Black-Scholes delta at current state.
    gamma_bs : float, optional
        Black-Scholes gamma at current state.
    vega_bs : float, optional
        Black-Scholes vega at current state.
    
    Additional Features (Optional)
    ------------------------------
    implied_vol : float, optional
        Current implied volatility (if available).
    recent_returns : ndarray, optional
        Recent spot returns (for LSTM features).
    
    Example
    -------
    >>> state = HedgingState(
    ...     spot=102.5,
    ...     time=0.1,
    ...     time_to_maturity=0.15,
    ...     position=0.55,
    ...     pnl=1.2,
    ...     step=25,
    ...     delta_bs=0.58,
    ... )
    >>> features = state.to_array()  # For neural network input
    """
    
    # Core state
    spot: float
    time: float
    time_to_maturity: float
    position: float
    pnl: float
    step: int
    
    # Greeks (optional)
    delta_bs: Optional[float] = None
    gamma_bs: Optional[float] = None
    vega_bs: Optional[float] = None
    theta_bs: Optional[float] = None
    
    # Additional features (optional)
    implied_vol: Optional[float] = None
    recent_returns: Optional[np.ndarray] = None
    
    # Reference values (for normalisation)
    strike: Optional[float] = None
    initial_spot: Optional[float] = None
    
    def to_array(
        self,
        include_greeks: bool = True,
        normalise: bool = True,
    ) -> np.ndarray:
        """
        Convert state to feature array for neural network input.
        
        Parameters
        ----------
        include_greeks : bool
            Whether to include Greeks in features.
        normalise : bool
            Whether to normalise features (recommended for training).
        
        Returns
        -------
        ndarray
            Feature vector.
        
        Feature Order
        -------------
        If include_greeks=True and normalise=True:
        [log_moneyness, tau, position, pnl_normalised, delta, gamma_scaled, vega_scaled]
        
        If include_greeks=False:
        [log_moneyness, tau, position, pnl_normalised]
        """
        features = []
        
        # Log-moneyness: log(S/K) — more stable than S directly
        if normalise and self.strike is not None and self.strike > 0:
            log_moneyness = np.log(self.spot / self.strike)
        else:
            log_moneyness = self.spot
        features.append(log_moneyness)
        
        # Time to maturity (already in [0, T])
        features.append(self.time_to_maturity)
        
        # Current position
        features.append(self.position)
        
        # P&L (normalised by strike if available)
        if normalise and self.strike is not None and self.strike > 0:
            pnl_norm = self.pnl / self.strike
        else:
            pnl_norm = self.pnl
        features.append(pnl_norm)
        
        # Greeks
        if include_greeks:
            if self.delta_bs is not None:
                features.append(self.delta_bs)
            if self.gamma_bs is not None:
                # Gamma can be large; scale by S²
                gamma_scaled = self.gamma_bs * (self.spot ** 2) if normalise else self.gamma_bs
                features.append(gamma_scaled)
            if self.vega_bs is not None:
                # Vega scaled by S
                vega_scaled = self.vega_bs / self.spot if normalise else self.vega_bs
                features.append(vega_scaled)
        
        return np.array(features, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding arrays)."""
        d = {
            "spot": self.spot,
            "time": self.time,
            "time_to_maturity": self.time_to_maturity,
            "position": self.position,
            "pnl": self.pnl,
            "step": self.step,
        }
        if self.delta_bs is not None:
            d["delta_bs"] = self.delta_bs
        if self.gamma_bs is not None:
            d["gamma_bs"] = self.gamma_bs
        if self.vega_bs is not None:
            d["vega_bs"] = self.vega_bs
        if self.strike is not None:
            d["strike"] = self.strike
        return d


@dataclass
class HedgingEpisode:
    """
    Complete record of a single hedging episode (one option lifetime).
    
    This stores the full trajectory of states, actions, and outcomes
    for analysis and visualisation.
    
    Parameters
    ----------
    config : HedgingConfig
        Configuration used for this episode.
    spot_path : ndarray, shape (n_steps + 1,)
        Spot price path S_0, S_1, ..., S_T.
    positions : ndarray, shape (n_steps,)
        Hedge positions δ_0, δ_1, ..., δ_{n-1}.
    pnl_path : ndarray, shape (n_steps + 1,)
        Cumulative P&L at each step.
    costs : ndarray, shape (n_steps,)
        Transaction costs at each rebalancing.
    terminal_pnl : float
        Final P&L after option settlement.
    payoff : float
        Option payoff at maturity.
    
    Derived Quantities
    ------------------
    total_cost : float
        Total transaction costs paid.
    hedging_error : float
        Terminal P&L - initial premium (should be ~0 for perfect hedge).
    """
    
    config: HedgingConfig
    spot_path: np.ndarray
    positions: np.ndarray
    pnl_path: np.ndarray
    costs: np.ndarray
    terminal_pnl: float
    payoff: float
    
    # Optional: deltas for comparison
    delta_bs_path: Optional[np.ndarray] = None
    
    # Optional: additional info
    info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_steps(self) -> int:
        """Number of hedging steps."""
        return len(self.positions)
    
    @property
    def total_cost(self) -> float:
        """Total transaction costs paid."""
        return float(np.sum(self.costs))
    
    @property
    def initial_premium(self) -> float:
        """Initial premium (from initial P&L)."""
        return float(self.pnl_path[0])
    
    @property
    def hedging_pnl_without_costs(self) -> float:
        """P&L from hedging trades only (excluding option premium and costs)."""
        # Sum of position * (S_{t+1} - S_t)
        spot_changes = np.diff(self.spot_path)
        hedge_pnl = np.sum(self.positions * spot_changes)
        return float(hedge_pnl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation."""
        return {
            "config": self.config.to_dict(),
            "spot_path": self.spot_path.tolist(),
            "positions": self.positions.tolist(),
            "pnl_path": self.pnl_path.tolist(),
            "costs": self.costs.tolist(),
            "terminal_pnl": self.terminal_pnl,
            "payoff": self.payoff,
            "total_cost": self.total_cost,
            "n_steps": self.n_steps,
        }


@dataclass
class HedgingResult:
    """
    Aggregated results from multiple hedging episodes.
    
    This collects statistics over many episodes for evaluation and comparison.
    
    Parameters
    ----------
    pnl_samples : ndarray, shape (n_episodes,)
        Terminal P&L for each episode.
    cost_samples : ndarray, shape (n_episodes,)
        Total costs for each episode.
    
    Statistics
    ----------
    mean_pnl : float
        Mean terminal P&L.
    std_pnl : float
        Standard deviation of P&L.
    sharpe : float
        Sharpe ratio of P&L (mean / std).
    mean_cost : float
        Mean transaction cost.
    cvar_95 : float
        95% CVaR of P&L (worst 5% average).
    """
    
    pnl_samples: np.ndarray
    cost_samples: np.ndarray
    
    # Optional: store all episodes
    episodes: Optional[List[HedgingEpisode]] = None
    
    # Metadata
    agent_name: str = "unknown"
    risk_measure_name: str = "unknown"
    n_episodes: int = 0
    
    def __post_init__(self):
        self.n_episodes = len(self.pnl_samples)
    
    @property
    def mean_pnl(self) -> float:
        """Mean terminal P&L."""
        return float(np.mean(self.pnl_samples))
    
    @property
    def std_pnl(self) -> float:
        """Standard deviation of P&L."""
        return float(np.std(self.pnl_samples))
    
    @property
    def sharpe(self) -> float:
        """Sharpe ratio (mean / std)."""
        if self.std_pnl > 0:
            return self.mean_pnl / self.std_pnl
        return 0.0
    
    @property
    def mean_cost(self) -> float:
        """Mean transaction cost."""
        return float(np.mean(self.cost_samples))
    
    @property
    def median_pnl(self) -> float:
        """Median P&L."""
        return float(np.median(self.pnl_samples))
    
    @property
    def min_pnl(self) -> float:
        """Minimum (worst) P&L."""
        return float(np.min(self.pnl_samples))
    
    @property
    def max_pnl(self) -> float:
        """Maximum (best) P&L."""
        return float(np.max(self.pnl_samples))
    
    @property
    def cvar_95(self) -> float:
        """95% CVaR (expected shortfall): average of worst 5%."""
        threshold = np.percentile(self.pnl_samples, 5)
        worst = self.pnl_samples[self.pnl_samples <= threshold]
        if len(worst) > 0:
            return float(np.mean(worst))
        return float(threshold)
    
    @property
    def var_95(self) -> float:
        """95% VaR: 5th percentile of P&L."""
        return float(np.percentile(self.pnl_samples, 5))
    
    def summary(self) -> Dict[str, float]:
        """Return summary statistics as dictionary."""
        return {
            "n_episodes": self.n_episodes,
            "mean_pnl": self.mean_pnl,
            "std_pnl": self.std_pnl,
            "sharpe": self.sharpe,
            "median_pnl": self.median_pnl,
            "min_pnl": self.min_pnl,
            "max_pnl": self.max_pnl,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "mean_cost": self.mean_cost,
        }
    
    def __repr__(self) -> str:
        return (
            f"HedgingResult(n={self.n_episodes}, "
            f"mean_pnl={self.mean_pnl:.4f}, "
            f"std_pnl={self.std_pnl:.4f}, "
            f"sharpe={self.sharpe:.3f})"
        )


@dataclass
class DeepHedgingTrainingConfig:
    """
    Configuration for deep hedging training.
    
    Parameters
    ----------
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Number of episodes per batch.
    learning_rate : float
        Optimizer learning rate.
    risk_measure : str
        Risk measure to use: "variance", "mean_variance", "cvar", "entropic".
    risk_params : dict
        Parameters for the risk measure (e.g., {"risk_aversion": 0.5}).
    
    Network Architecture
    --------------------
    hidden_layers : list of int
        Sizes of hidden layers (e.g., [64, 64]).
    activation : str
        Activation function: "relu", "tanh", "elu".
    
    Training Options
    ----------------
    early_stopping_patience : int
        Stop if no improvement for this many epochs (0 = disabled).
    checkpoint_dir : str, optional
        Directory to save checkpoints.
    log_every : int
        Log metrics every N epochs.
    """
    
    # Training
    n_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 0.001
    
    # Risk measure
    risk_measure: str = "mean_variance"
    risk_params: Dict[str, Any] = field(default_factory=lambda: {"risk_aversion": 0.5})
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    
    # Training options
    early_stopping_patience: int = 10
    checkpoint_dir: Optional[str] = None
    log_every: int = 10
    verbose: int = 1
    
    # Variance reduction
    use_antithetic: bool = True
    use_control_variate: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "HedgingConfig",
    "HedgingState",
    "HedgingEpisode",
    "HedgingResult",
    "DeepHedgingTrainingConfig",
]
