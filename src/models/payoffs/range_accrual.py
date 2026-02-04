"""
Range Accrual Payoffs.

Implements payoffs that accrue based on the proportion of time
an underlying stays within a specified range.

Example:
    from src.models.payoffs.range_accrual import RangeAccrualPayoff
    
    payoff = RangeAccrualPayoff(
        range_lower=0.03,
        range_upper=0.05,
        accrual_rate=0.06,
    )
    
    # paths shape: (n_paths, n_observations)
    # Each column is the underlying value at an observation
    values = payoff.terminal_from_paths(paths)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.models.payoffs.base import BasePathPayoff1D


@dataclass(frozen=True, slots=True)
class RangeAccrualPayoff(BasePathPayoff1D):
    """
    Range accrual payoff.
    
    Payoff = accrual_rate * (fraction of observations in range)
    
    Attributes
    ----------
    range_lower : float
        Lower bound of the range.
    range_upper : float
        Upper bound of the range.
    accrual_rate : float
        Rate paid when in range (annualized).
    time_to_maturity : float
        Time to maturity for rate scaling.
    inclusive : bool
        If True, range boundaries are inclusive.
    """
    
    range_lower: float
    range_upper: float
    accrual_rate: float = 0.06
    time_to_maturity: float = 1.0
    inclusive: bool = True
    
    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute range accrual payoff.
        
        Parameters
        ----------
        paths : ndarray
            Paths of shape (n_paths, n_observations).
            Each column is underlying value at observation time.
        
        Returns
        -------
        ndarray
            Payoff values (accrued rate * time) of shape (n_paths,).
        """
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_obs = paths.shape
        
        if n_obs == 0:
            return np.zeros(n_paths)
        
        # Check if each observation is in range
        if self.inclusive:
            in_range = (paths >= self.range_lower) & (paths <= self.range_upper)
        else:
            in_range = (paths > self.range_lower) & (paths < self.range_upper)
        
        # Fraction of observations in range
        fraction_in_range = np.mean(in_range, axis=1)
        
        # Payoff
        payoff = self.accrual_rate * self.time_to_maturity * fraction_in_range
        
        return payoff
    
    def terminal_from_paths_with_info(
        self,
        paths: np.ndarray,
    ) -> tuple:
        """
        Compute payoff with detailed information.
        
        Returns
        -------
        payoffs : ndarray
            Payoff values.
        info : dict
            Range accrual statistics.
        """
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_obs = paths.shape
        
        if self.inclusive:
            in_range = (paths >= self.range_lower) & (paths <= self.range_upper)
        else:
            in_range = (paths > self.range_lower) & (paths < self.range_upper)
        
        fraction_in_range = np.mean(in_range, axis=1)
        payoffs = self.accrual_rate * self.time_to_maturity * fraction_in_range
        
        # Statistics
        days_in_range = np.sum(in_range, axis=1)
        
        info = {
            "mean_fraction_in_range": float(np.mean(fraction_in_range)),
            "mean_days_in_range": float(np.mean(days_in_range)),
            "prob_full_accrual": float(np.mean(fraction_in_range == 1.0)),
            "prob_zero_accrual": float(np.mean(fraction_in_range == 0.0)),
            "min_fraction": float(np.min(fraction_in_range)),
            "max_fraction": float(np.max(fraction_in_range)),
        }
        
        return payoffs, info


@dataclass(frozen=True, slots=True)
class DoubleRangeAccrualPayoff(BasePathPayoff1D):
    """
    Double range accrual with different rates.
    
    Pays rate_inner when in inner range, rate_outer when in outer
    (but not inner) range, zero outside.
    
    Attributes
    ----------
    inner_lower : float
        Inner range lower bound.
    inner_upper : float
        Inner range upper bound.
    outer_lower : float
        Outer range lower bound.
    outer_upper : float
        Outer range upper bound.
    rate_inner : float
        Rate when in inner range.
    rate_outer : float
        Rate when in outer (not inner) range.
    time_to_maturity : float
        Time to maturity.
    """
    
    inner_lower: float
    inner_upper: float
    outer_lower: float
    outer_upper: float
    rate_inner: float
    rate_outer: float
    time_to_maturity: float = 1.0
    
    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """Compute double range accrual payoff."""
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_obs = paths.shape
        
        if n_obs == 0:
            return np.zeros(n_paths)
        
        # Inner range
        in_inner = (paths >= self.inner_lower) & (paths <= self.inner_upper)
        
        # Outer range (but not inner)
        in_outer = (
            ((paths >= self.outer_lower) & (paths <= self.outer_upper))
            & ~in_inner
        )
        
        # Fractions
        frac_inner = np.mean(in_inner, axis=1)
        frac_outer = np.mean(in_outer, axis=1)
        
        # Payoff
        payoff = self.time_to_maturity * (
            self.rate_inner * frac_inner + self.rate_outer * frac_outer
        )
        
        return payoff


@dataclass(frozen=True, slots=True)
class DigitalRangePayoff(BasePathPayoff1D):
    """
    Digital range payoff.
    
    Pays fixed amount if underlying stays within range for
    more than threshold fraction of observations.
    
    Attributes
    ----------
    range_lower : float
        Range lower bound.
    range_upper : float
        Range upper bound.
    payout : float
        Fixed payout if condition met.
    threshold : float
        Minimum fraction of time in range required.
    """
    
    range_lower: float
    range_upper: float
    payout: float = 1.0
    threshold: float = 0.5
    
    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """Compute digital range payoff."""
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_obs = paths.shape
        
        if n_obs == 0:
            return np.zeros(n_paths)
        
        in_range = (paths >= self.range_lower) & (paths <= self.range_upper)
        fraction = np.mean(in_range, axis=1)
        
        payoff = np.where(fraction >= self.threshold, self.payout, 0.0)
        
        return payoff


__all__ = [
    "RangeAccrualPayoff",
    "DoubleRangeAccrualPayoff",
    "DigitalRangePayoff",
]
