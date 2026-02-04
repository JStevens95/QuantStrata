"""
Autocallable Option Payoffs.

Implements path-dependent payoffs for autocallable structured products,
which may terminate early based on barrier observations.

Example:
    from src.models.payoffs.autocallable import AutocallablePayoff
    
    payoff = AutocallablePayoff(
        autocall_barrier=1.0,
        coupon_barrier=0.8,
        put_barrier=0.6,
        coupon_rate=0.10,
        observation_times=[0.25, 0.5, 0.75, 1.0],
    )
    
    # paths shape: (n_paths, n_steps) where columns are observation dates
    values, autocall_info = payoff.terminal_from_paths_with_info(paths)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.models.payoffs.base import BasePathPayoff1D


@dataclass
class AutocallablePayoff(BasePathPayoff1D):
    """
    Autocallable payoff with early termination.
    
    At each observation date:
    - If spot/initial >= autocall_barrier: early redemption at 100% + accrued coupon
    - If spot/initial >= coupon_barrier: pay coupon
    - Otherwise: no coupon (or accumulate if memory)
    
    At maturity (if not autocalled):
    - If spot/initial >= put_barrier: return 100%
    - Otherwise: return spot/initial (loss)
    
    Attributes
    ----------
    autocall_barrier : float
        Autocall trigger (fraction of initial).
    coupon_barrier : float
        Coupon payment trigger.
    put_barrier : float
        Put protection level.
    coupon_rate : float
        Annual coupon rate.
    observation_times : list of float
        Time in years to each observation.
    memory_coupon : bool
        If True, accumulate missed coupons.
    initial_spot : float
        Initial spot level (default 1.0 for normalized paths).
    """
    
    autocall_barrier: float = 1.0
    coupon_barrier: float = 0.8
    put_barrier: float = 0.6
    coupon_rate: float = 0.10
    observation_times: List[float] = field(default_factory=list)
    memory_coupon: bool = True
    initial_spot: float = 1.0
    
    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute autocallable payoff from price paths.
        
        Parameters
        ----------
        paths : ndarray
            Price paths of shape (n_paths, n_observations + 1).
            paths[:, 0] is the initial spot.
            paths[:, i] is the spot at observation i.
        
        Returns
        -------
        ndarray
            Payoff values of shape (n_paths,).
        """
        payoffs, _ = self.terminal_from_paths_with_info(paths)
        return payoffs
    
    def terminal_from_paths_with_info(
        self,
        paths: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute payoff with detailed autocall information.
        
        Parameters
        ----------
        paths : ndarray
            Price paths of shape (n_paths, n_observations + 1).
        
        Returns
        -------
        payoffs : ndarray
            Payoff values of shape (n_paths,).
        info : dict
            Information about autocall events.
        """
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_steps = paths.shape
        
        # Initial spot is first column
        initial = paths[:, 0:1]  # Keep dimension
        
        # Normalized performance at each observation
        performance = paths[:, 1:] / initial
        n_obs = performance.shape[1]
        
        # Initialize tracking
        payoffs = np.zeros(n_paths)
        autocalled = np.zeros(n_paths, dtype=bool)
        autocall_period = np.full(n_paths, -1, dtype=int)
        accrued_coupons = np.zeros(n_paths)
        total_coupons_paid = np.zeros(n_paths)
        
        # Get time intervals for coupon calculation
        if self.observation_times:
            obs_times = self.observation_times
        else:
            obs_times = list(np.linspace(0.25, 1.0, n_obs))
        
        # Process each observation
        for i in range(n_obs):
            # Skip already autocalled paths
            active = ~autocalled
            
            if not np.any(active):
                break
            
            perf_i = performance[active, i]
            
            # Period fraction for coupon
            if i == 0:
                dt = obs_times[0]
            else:
                dt = obs_times[i] - obs_times[i - 1]
            
            period_coupon = self.coupon_rate * dt
            
            # Check autocall
            autocall_trigger = perf_i >= self.autocall_barrier
            
            if np.any(autocall_trigger):
                # Identify paths that autocall
                active_indices = np.where(active)[0]
                autocall_indices = active_indices[autocall_trigger]
                
                # Payoff: 100% + coupon for this period + any accrued
                if self.memory_coupon:
                    payoffs[autocall_indices] = 1.0 + period_coupon + accrued_coupons[autocall_indices]
                else:
                    payoffs[autocall_indices] = 1.0 + period_coupon
                
                autocalled[autocall_indices] = True
                autocall_period[autocall_indices] = i
            
            # Check coupon barrier for non-autocalled active paths
            still_active = active & ~autocalled
            if np.any(still_active):
                perf_still_active = performance[still_active, i]
                coupon_trigger = perf_still_active >= self.coupon_barrier
                
                still_active_indices = np.where(still_active)[0]
                coupon_indices = still_active_indices[coupon_trigger]
                no_coupon_indices = still_active_indices[~coupon_trigger]
                
                if self.memory_coupon:
                    # Pay current + any accrued
                    coupon_payment = period_coupon + accrued_coupons[coupon_indices]
                    total_coupons_paid[coupon_indices] += coupon_payment
                    accrued_coupons[coupon_indices] = 0  # Reset accrued
                    
                    # Accrue for paths that miss coupon
                    accrued_coupons[no_coupon_indices] += period_coupon
                else:
                    # Pay current period only
                    total_coupons_paid[coupon_indices] += period_coupon
        
        # Handle paths that reach maturity (not autocalled)
        at_maturity = ~autocalled
        if np.any(at_maturity):
            final_perf = performance[at_maturity, -1]
            
            maturity_indices = np.where(at_maturity)[0]
            
            # Above put barrier: return 100% + any coupons
            above_put = final_perf >= self.put_barrier
            above_put_indices = maturity_indices[above_put]
            payoffs[above_put_indices] = 1.0 + total_coupons_paid[above_put_indices]
            
            # Below put barrier: loss on put
            below_put = ~above_put
            below_put_indices = maturity_indices[below_put]
            payoffs[below_put_indices] = final_perf[below_put] + total_coupons_paid[below_put_indices]
        
        info = {
            "autocalled": autocalled,
            "autocall_period": autocall_period,
            "prob_autocall": float(np.mean(autocalled)),
            "total_coupons_paid": total_coupons_paid,
            "mean_autocall_period": float(np.mean(autocall_period[autocalled])) if np.any(autocalled) else -1,
        }
        
        return payoffs, info


@dataclass
class SimpleAutocallablePayoff(BasePathPayoff1D):
    """
    Simplified autocallable payoff (no coupons, just early redemption).
    
    At each observation:
    - If spot >= autocall_barrier * initial: early redemption at redemption_amount
    
    At maturity:
    - If spot >= put_barrier * initial: return 1.0
    - Otherwise: return spot / initial
    
    Attributes
    ----------
    autocall_barrier : float
        Autocall trigger (fraction of initial).
    put_barrier : float
        Put protection level.
    redemption_amount : float
        Amount returned on autocall (e.g., 1.05 for 105%).
    """
    
    autocall_barrier: float = 1.0
    put_barrier: float = 0.6
    redemption_amount: float = 1.0
    
    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """Compute simple autocallable payoff."""
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        n_paths, n_steps = paths.shape
        
        initial = paths[:, 0:1]
        performance = paths[:, 1:] / initial
        
        payoffs = np.zeros(n_paths)
        autocalled = np.zeros(n_paths, dtype=bool)
        
        # Check autocall at each observation
        for i in range(performance.shape[1]):
            active = ~autocalled
            if not np.any(active):
                break
            
            trigger = performance[active, i] >= self.autocall_barrier
            active_indices = np.where(active)[0]
            autocall_indices = active_indices[trigger]
            
            payoffs[autocall_indices] = self.redemption_amount
            autocalled[autocall_indices] = True
        
        # Maturity payoff for non-autocalled
        at_maturity = ~autocalled
        if np.any(at_maturity):
            final_perf = performance[at_maturity, -1]
            maturity_indices = np.where(at_maturity)[0]
            
            above_put = final_perf >= self.put_barrier
            payoffs[maturity_indices[above_put]] = 1.0
            payoffs[maturity_indices[~above_put]] = final_perf[~above_put]
        
        return payoffs


__all__ = [
    "AutocallablePayoff",
    "SimpleAutocallablePayoff",
]
