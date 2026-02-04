"""
Multi-Asset Hedging Environment.

Environment for hedging portfolios of options across multiple underlyings,
including cross-gamma and correlation effects.

Example:
    from src.deep_hedging.environments import MultiAssetHedgingEnv
    
    env = MultiAssetHedgingEnv(
        config=MultiAssetHedgingConfig(
            n_assets=3,
            correlation=np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]]),
        ),
    )
    
    state, info = env.reset()
    action = np.array([0.9, 1.1, 0.95])  # Hedge ratios per asset
    state, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MultiAssetHedgingConfig:
    """Configuration for multi-asset hedging environment."""
    
    # Portfolio
    n_assets: int = 2
    strikes: Optional[List[float]] = None  # ATM if None
    maturities: Optional[List[float]] = None  # All same if None
    notionals: Optional[List[float]] = None  # All 1.0 if None
    option_types: Optional[List[str]] = None  # All "call" if None
    
    # Market parameters
    initial_spots: Optional[List[float]] = None  # All 100 if None
    volatilities: Optional[List[float]] = None  # All 0.2 if None
    correlation: Optional[np.ndarray] = None  # Identity if None
    risk_free_rate: float = 0.05
    dividend_yields: Optional[List[float]] = None  # All 0 if None
    
    # Simulation
    n_steps: int = 50
    maturity: float = 0.25  # Default maturity in years
    
    # Transaction costs
    proportional_cost: float = 0.001
    fixed_cost: float = 0.0
    
    # State features
    include_deltas: bool = True
    include_gammas: bool = True
    include_cross_gamma: bool = True
    include_positions: bool = True
    
    # Reward
    risk_aversion: float = 0.1


# =============================================================================
# Multi-Asset Hedging Environment
# =============================================================================


class MultiAssetHedgingEnv:
    """
    Multi-asset hedging environment.
    
    Simulates correlated asset dynamics and allows hedging
    a portfolio of options with multiple underlyings.
    
    State includes:
    - Normalized spot prices
    - Time to expiry
    - Deltas per asset
    - Cross-gammas (optional)
    - Current positions
    
    Action is vector of hedge ratios per asset.
    
    Example:
        config = MultiAssetHedgingConfig(
            n_assets=3,
            correlation=np.array([
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]),
        )
        env = MultiAssetHedgingEnv(config)
        
        state, info = env.reset()
        while True:
            action = agent.select_action(state)  # Vector of hedge ratios
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    """
    
    def __init__(
        self,
        config: Optional[MultiAssetHedgingConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize environment.
        
        Parameters
        ----------
        config : MultiAssetHedgingConfig, optional
            Environment configuration.
        seed : int, optional
            Random seed.
        """
        self.config = config or MultiAssetHedgingConfig()
        self._rng = np.random.default_rng(seed)
        
        # Initialize parameters
        n = self.config.n_assets
        
        self._initial_spots = np.array(
            self.config.initial_spots or [100.0] * n
        )
        self._volatilities = np.array(
            self.config.volatilities or [0.2] * n
        )
        self._strikes = np.array(
            self.config.strikes or self._initial_spots.copy()
        )
        self._maturities = np.array(
            self.config.maturities or [self.config.maturity] * n
        )
        self._notionals = np.array(
            self.config.notionals or [1.0] * n
        )
        self._option_types = self.config.option_types or ["call"] * n
        self._dividend_yields = np.array(
            self.config.dividend_yields or [0.0] * n
        )
        
        # Correlation matrix
        if self.config.correlation is not None:
            self._correlation = self.config.correlation
        else:
            self._correlation = np.eye(n)
        
        # Compute Cholesky decomposition for correlated simulation
        self._cholesky = np.linalg.cholesky(self._correlation)
        
        # Time step
        self._dt = self.config.maturity / self.config.n_steps
        
        # State tracking
        self._step_count: int = 0
        self._spots: np.ndarray = np.zeros(n)
        self._time_to_expiry: float = 0.0
        self._positions: np.ndarray = np.zeros(n)
        self._cash: float = 0.0
        self._pnl: float = 0.0
    
    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.config.n_assets
    
    @property
    def observation_space_dim(self) -> int:
        """Observation space dimension."""
        n = self.n_assets
        dim = n + 1  # Spots + time
        
        if self.config.include_deltas:
            dim += n
        if self.config.include_gammas:
            dim += n
        if self.config.include_cross_gamma:
            dim += n * (n - 1) // 2  # Upper triangular cross-gammas
        if self.config.include_positions:
            dim += n
        
        return dim
    
    @property
    def action_space_dim(self) -> int:
        """Action space dimension (hedge ratio per asset)."""
        return self.n_assets
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        self._step_count = 0
        self._spots = self._initial_spots.copy()
        self._time_to_expiry = self.config.maturity
        self._positions = np.zeros(self.n_assets)
        
        # Compute initial portfolio value
        self._cash = sum(
            self._price_option(i)
            for i in range(self.n_assets)
        )
        self._pnl = 0.0
        
        state = self._get_state()
        info = self._get_info()
        
        return state, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute hedging step.
        
        Parameters
        ----------
        action : ndarray
            Hedge ratios per asset (length n_assets).
        
        Returns
        -------
        state, reward, terminated, truncated, info
        """
        action = np.asarray(action).flatten()
        
        if len(action) != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} actions, got {len(action)}")
        
        # Compute deltas
        deltas = np.array([self._compute_delta(i) for i in range(self.n_assets)])
        
        # Target positions
        target_positions = action * np.abs(deltas) * self._notionals
        
        # Execute trades
        trades = target_positions - self._positions
        trade_costs = sum(
            abs(trades[i] * self._spots[i]) * self.config.proportional_cost +
            self.config.fixed_cost * (1 if abs(trades[i]) > 1e-8 else 0)
            for i in range(self.n_assets)
        )
        
        self._cash -= sum(trades * self._spots) + trade_costs
        self._positions = target_positions
        
        # Simulate correlated price moves
        Z = self._rng.standard_normal(self.n_assets)
        correlated_Z = self._cholesky @ Z
        
        for i in range(self.n_assets):
            drift = (
                self.config.risk_free_rate - self._dividend_yields[i] -
                0.5 * self._volatilities[i]**2
            )
            diffusion = self._volatilities[i] * correlated_Z[i]
            self._spots[i] *= np.exp(drift * self._dt + diffusion * np.sqrt(self._dt))
        
        # Update time
        self._time_to_expiry -= self._dt
        self._step_count += 1
        
        # Compute new option values
        new_option_values = sum(
            self._price_option(i)
            for i in range(self.n_assets)
        )
        
        # Compute P&L
        position_value = sum(self._positions * self._spots)
        old_pnl = self._pnl
        self._pnl = self._cash + position_value - new_option_values
        
        # Reward
        delta_pnl = self._pnl - old_pnl
        pnl_var = delta_pnl ** 2
        reward = delta_pnl - self.config.risk_aversion * pnl_var
        
        # Termination
        terminated = self._time_to_expiry <= 0
        truncated = self._step_count >= self.config.n_steps
        
        state = self._get_state()
        info = self._get_info()
        info["trades"] = trades.tolist()
        info["trade_costs"] = trade_costs
        
        return state, float(reward), terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """Build state observation."""
        features = []
        
        # Normalized spots
        normalized_spots = self._spots / self._initial_spots - 1.0
        features.extend(normalized_spots)
        
        # Time
        features.append(self._time_to_expiry / self.config.maturity)
        
        # Deltas
        if self.config.include_deltas:
            deltas = [self._compute_delta(i) for i in range(self.n_assets)]
            features.extend(deltas)
        
        # Gammas
        if self.config.include_gammas:
            gammas = [self._compute_gamma(i) for i in range(self.n_assets)]
            features.extend([g * s for g, s in zip(gammas, self._spots)])
        
        # Cross-gammas (correlation sensitivity)
        if self.config.include_cross_gamma:
            for i in range(self.n_assets):
                for j in range(i + 1, self.n_assets):
                    cross_gamma = self._correlation[i, j] * np.sqrt(
                        self._compute_gamma(i) * self._compute_gamma(j)
                    )
                    features.append(cross_gamma)
        
        # Positions
        if self.config.include_positions:
            deltas = [abs(self._compute_delta(i)) + 1e-8 for i in range(self.n_assets)]
            rel_positions = self._positions / np.array(deltas)
            features.extend(rel_positions)
        
        return np.array(features, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary."""
        deltas = [self._compute_delta(i) for i in range(self.n_assets)]
        
        return {
            "step": self._step_count,
            "spots": self._spots.tolist(),
            "time_to_expiry": self._time_to_expiry,
            "deltas": deltas,
            "positions": self._positions.tolist(),
            "cash": self._cash,
            "pnl": self._pnl,
        }
    
    def _price_option(self, asset_idx: int) -> float:
        """Price option for asset i using Black-Scholes."""
        S = self._spots[asset_idx]
        K = self._strikes[asset_idx]
        T = max(self._time_to_expiry, 1e-8)
        vol = self._volatilities[asset_idx]
        r = self.config.risk_free_rate
        q = self._dividend_yields[asset_idx]
        
        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        
        from scipy.stats import norm
        
        if self._option_types[asset_idx] == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return float(price * self._notionals[asset_idx])
    
    def _compute_delta(self, asset_idx: int) -> float:
        """Compute delta for asset i."""
        S = self._spots[asset_idx]
        K = self._strikes[asset_idx]
        T = max(self._time_to_expiry, 1e-8)
        vol = self._volatilities[asset_idx]
        r = self.config.risk_free_rate
        q = self._dividend_yields[asset_idx]
        
        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        
        from scipy.stats import norm
        
        if self._option_types[asset_idx] == "call":
            return float(np.exp(-q * T) * norm.cdf(d1))
        else:
            return float(np.exp(-q * T) * (norm.cdf(d1) - 1))
    
    def _compute_gamma(self, asset_idx: int) -> float:
        """Compute gamma for asset i."""
        S = self._spots[asset_idx]
        K = self._strikes[asset_idx]
        T = max(self._time_to_expiry, 1e-8)
        vol = self._volatilities[asset_idx]
        r = self.config.risk_free_rate
        q = self._dividend_yields[asset_idx]
        
        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        
        from scipy.stats import norm
        
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * vol * np.sqrt(T))
        
        return float(gamma)


__all__ = [
    "MultiAssetHedgingEnv",
    "MultiAssetHedgingConfig",
]
