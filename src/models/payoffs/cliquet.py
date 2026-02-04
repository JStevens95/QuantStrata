"""
Cliquet Option Payoffs.

Implements path-dependent payoffs for cliquet (ratchet) options,
which pay based on capped/floored periodic returns.

Example:
    from src.models.payoffs.cliquet import CliquetPayoff
    
    payoff = CliquetPayoff(
        local_cap=0.03,
        local_floor=-0.01,
        global_cap=0.20,
        global_floor=0.0,
        participation=1.0,
    )
    
    # paths shape: (n_paths, n_steps) where n_steps = n_periods + 1
    # paths[:, 0] = initial spot, paths[:, 1:] = spots at reset dates
    values = payoff.terminal_from_paths(paths)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.models.payoffs.base import BasePathPayoff1D


@dataclass(frozen=True, slots=True)
class CliquetPayoff(BasePathPayoff1D):
    """
    Cliquet (ratchet) option payoff.
    
    Calculates payoff based on sum of capped/floored periodic returns.
    
    At each reset date i, the local return is:
        r_i = clamp(S_i / S_{i-1} - 1, local_floor, local_cap)
    
    The global return is:
        R = clamp(sum(r_i), global_floor, global_cap)
    
    The payoff is:
        payoff = participation * max(R, 0)
    
    Attributes
    ----------
    local_cap : float
        Maximum return per period.
    local_floor : float
        Minimum return per period.
    global_cap : float, optional
        Maximum total accumulated return. None means no cap.
    global_floor : float
        Minimum total accumulated return.
    participation : float
        Participation rate applied to final return.
    
    Example
    -------
    >>> payoff = CliquetPayoff(
    ...     local_cap=0.03,
    ...     local_floor=-0.01,
    ...     global_floor=0.0,
    ... )
    >>> paths = np.array([[100, 103, 106, 104]])  # shape (1, 4)
    >>> value = payoff.terminal_from_paths(paths)
    >>> # Returns: [0.03 + 0.0291 + (-0.0189)] capped/floored and participation applied
    """
    
    local_cap: float
    local_floor: float
    global_cap: Optional[float] = None
    global_floor: float = 0.0
    participation: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate payoff parameters."""
        if self.local_floor > self.local_cap:
            raise ValueError(
                f"Local floor ({self.local_floor}) cannot exceed "
                f"local cap ({self.local_cap})"
            )
        
        if self.global_cap is not None and self.global_floor > self.global_cap:
            raise ValueError(
                f"Global floor ({self.global_floor}) cannot exceed "
                f"global cap ({self.global_cap})"
            )
    
    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute cliquet payoff from price paths.
        
        Parameters
        ----------
        paths : ndarray
            Price paths of shape (n_paths, n_steps).
            paths[:, 0] is the initial spot.
            paths[:, i] is the spot at reset date i.
        
        Returns
        -------
        ndarray
            Payoff values of shape (n_paths,).
        """
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_steps = paths.shape
        
        if n_steps < 2:
            # No periods, no return
            return np.zeros(n_paths)
        
        # Compute period returns: S[i] / S[i-1] - 1
        period_returns = paths[:, 1:] / paths[:, :-1] - 1
        
        # Apply local caps and floors
        capped_returns = np.clip(period_returns, self.local_floor, self.local_cap)
        
        # Sum across periods to get global return
        global_returns = np.sum(capped_returns, axis=1)
        
        # Apply global floor and cap
        global_returns = np.maximum(global_returns, self.global_floor)
        if self.global_cap is not None:
            global_returns = np.minimum(global_returns, self.global_cap)
        
        # Apply participation and ensure non-negative (typical for cliquets)
        payoff = self.participation * np.maximum(global_returns, 0.0)
        
        return payoff


@dataclass(frozen=True, slots=True)
class ReverseCliquetPayoff(BasePathPayoff1D):
    """
    Reverse cliquet payoff (put-like).
    
    Pays when the accumulated return is negative (capped/floored).
    Used for downside participation structures.
    
    Attributes
    ----------
    local_cap : float
        Maximum upside per period (positive return cap).
    local_floor : float
        Maximum downside per period (negative return floor).
    global_cap : float, optional
        Cap on total loss that can be captured.
    global_floor : float
        Floor on total loss.
    participation : float
        Participation in downside.
    """
    
    local_cap: float
    local_floor: float
    global_cap: Optional[float] = None
    global_floor: float = 0.0
    participation: float = 1.0
    
    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute reverse cliquet payoff.
        
        Parameters
        ----------
        paths : ndarray
            Price paths of shape (n_paths, n_steps).
        
        Returns
        -------
        ndarray
            Payoff values of shape (n_paths,).
        """
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_steps = paths.shape
        
        if n_steps < 2:
            return np.zeros(n_paths)
        
        # Period returns
        period_returns = paths[:, 1:] / paths[:, :-1] - 1
        
        # Local caps/floors
        capped_returns = np.clip(period_returns, self.local_floor, self.local_cap)
        
        # Global return
        global_returns = np.sum(capped_returns, axis=1)
        
        # Apply global bounds
        global_returns = np.maximum(global_returns, self.global_floor)
        if self.global_cap is not None:
            global_returns = np.minimum(global_returns, self.global_cap)
        
        # Reverse: pay on negative returns
        payoff = self.participation * np.maximum(-global_returns, 0.0)
        
        return payoff


@dataclass(frozen=True, slots=True)
class LocalCliquetPayoff(BasePathPayoff1D):
    """
    Local cliquet payoff (sum of local period payoffs).
    
    Instead of capping/flooring the sum, this structure pays
    the sum of individually computed period payoffs.
    
    payoff = sum(max(clamp(r_i, floor, cap), 0))
    
    Attributes
    ----------
    local_cap : float
        Maximum return per period.
    local_floor : float
        Minimum return per period.
    participation : float
        Participation rate.
    """
    
    local_cap: float
    local_floor: float
    participation: float = 1.0
    
    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute local cliquet payoff.
        
        Parameters
        ----------
        paths : ndarray
            Price paths of shape (n_paths, n_steps).
        
        Returns
        -------
        ndarray
            Payoff values of shape (n_paths,).
        """
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_steps = paths.shape
        
        if n_steps < 2:
            return np.zeros(n_paths)
        
        # Period returns
        period_returns = paths[:, 1:] / paths[:, :-1] - 1
        
        # Local caps/floors
        capped_returns = np.clip(period_returns, self.local_floor, self.local_cap)
        
        # Sum of positive local returns
        local_payoffs = np.maximum(capped_returns, 0.0)
        payoff = self.participation * np.sum(local_payoffs, axis=1)
        
        return payoff


__all__ = [
    "CliquetPayoff",
    "ReverseCliquetPayoff",
    "LocalCliquetPayoff",
]
