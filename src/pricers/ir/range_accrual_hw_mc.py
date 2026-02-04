"""
Monte Carlo pricer for Range Accrual Notes under Hull-White dynamics.

Prices range accrual notes by simulating short rate paths and computing
the fraction of time the rate stays within the specified range.

Example:
    from src.pricers.ir.range_accrual_hw_mc import IrRangeAccrualHwMcPricer
    from src.instruments.ir.options.range_accrual import IrRangeAccrualNote
    
    pricer = IrRangeAccrualHwMcPricer(n_paths=100_000)
    
    note = IrRangeAccrualNote(
        notional=1_000_000,
        start_date=date(2024, 1, 1),
        maturity_date=date(2025, 1, 1),
        range_lower=0.03,
        range_upper=0.05,
        accrual_rate=0.06,
        reference_rate_id="USD3M",
    )
    
    result = pricer.price(note, market_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np

from src.instruments.ir.options.range_accrual import IrRangeAccrualNote, ObservationFrequency
from src.models.payoffs.range_accrual import RangeAccrualPayoff


# =============================================================================
# Pricing Result
# =============================================================================


@dataclass
class RangeAccrualPricingResult:
    """Result from range accrual pricing."""
    
    price: float
    standard_error: float
    
    # Greeks
    delta: Optional[float] = None  # Sensitivity to rate level
    vega: Optional[float] = None   # Sensitivity to vol
    
    # Range-specific metrics
    expected_accrual_fraction: float = 0.0
    prob_full_accrual: float = 0.0
    prob_zero_accrual: float = 0.0
    expected_coupon: float = 0.0
    
    # Diagnostics
    n_paths: int = 0
    n_observations: int = 0
    elapsed_seconds: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "price": self.price,
            "standard_error": self.standard_error,
            "delta": self.delta,
            "vega": self.vega,
            "expected_accrual_fraction": self.expected_accrual_fraction,
            "prob_full_accrual": self.prob_full_accrual,
            "prob_zero_accrual": self.prob_zero_accrual,
            "expected_coupon": self.expected_coupon,
        }


# =============================================================================
# Market Data
# =============================================================================


@dataclass
class RangeAccrualMarketData:
    """Market data for range accrual pricing."""
    
    # Current rate level
    initial_rate: float
    
    # Hull-White parameters
    mean_reversion: float = 0.03
    volatility: float = 0.01
    long_term_rate: float = 0.04
    
    # Discounting
    discount_rate: float = 0.05
    
    valuation_date: Optional[date] = None


# =============================================================================
# Range Accrual Pricer
# =============================================================================


class IrRangeAccrualHwMcPricer:
    """
    Monte Carlo pricer for IR Range Accrual under Hull-White.
    
    Simulates Hull-White short rate paths and computes the
    range accrual payoff based on the fraction of observations
    within the specified range.
    
    Hull-White dynamics:
        dr = a(θ - r)dt + σdW
    
    Where:
        a = mean reversion speed
        θ = long-term rate level
        σ = volatility
    
    Example:
        pricer = IrRangeAccrualHwMcPricer(n_paths=100_000)
        
        market_data = RangeAccrualMarketData(
            initial_rate=0.04,
            mean_reversion=0.03,
            volatility=0.01,
            long_term_rate=0.04,
        )
        
        result = pricer.price(note, market_data)
        print(f"Price: {result.price:.2f}")
        print(f"Expected accrual: {result.expected_accrual_fraction:.1%}")
    """
    
    def __init__(
        self,
        n_paths: int = 100_000,
        seed: Optional[int] = None,
        antithetic: bool = True,
    ) -> None:
        """
        Initialize range accrual pricer.
        
        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            Random seed.
        antithetic : bool
            Use antithetic variance reduction.
        """
        self.n_paths = n_paths
        self.seed = seed
        self.antithetic = antithetic
        self._rng = np.random.default_rng(seed)
    
    def price(
        self,
        instrument: IrRangeAccrualNote,
        market_data: RangeAccrualMarketData,
    ) -> RangeAccrualPricingResult:
        """
        Price a range accrual note.
        
        Parameters
        ----------
        instrument : IrRangeAccrualNote
            Range accrual note to price.
        market_data : RangeAccrualMarketData
            Market data with Hull-White parameters.
        
        Returns
        -------
        RangeAccrualPricingResult
            Pricing result.
        """
        import time
        start_time = time.time()
        
        valuation_date = market_data.valuation_date or instrument.start_date
        T = (instrument.maturity_date - valuation_date).days / 365.0
        
        if T <= 0:
            return RangeAccrualPricingResult(
                price=0.0,
                standard_error=0.0,
                n_paths=0,
            )
        
        # Determine number of observations
        n_obs = self._get_n_observations(instrument, valuation_date)
        
        # Create payoff
        payoff = RangeAccrualPayoff(
            range_lower=instrument.range_lower,
            range_upper=instrument.range_upper,
            accrual_rate=instrument.accrual_rate,
            time_to_maturity=T,
        )
        
        # Simulate rate paths
        rate_paths = self._simulate_hw_paths(
            initial_rate=market_data.initial_rate,
            mean_reversion=market_data.mean_reversion,
            volatility=market_data.volatility,
            long_term_rate=market_data.long_term_rate,
            T=T,
            n_obs=n_obs,
        )
        
        # Compute payoffs
        payoffs, info = payoff.terminal_from_paths_with_info(rate_paths)
        payoffs = payoffs * instrument.notional
        
        # Discount
        discount = np.exp(-market_data.discount_rate * T)
        discounted_payoffs = payoffs * discount
        
        # Statistics
        price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs)))
        
        expected_coupon = float(np.mean(payoffs))
        
        elapsed = time.time() - start_time
        
        return RangeAccrualPricingResult(
            price=price,
            standard_error=std_error,
            expected_accrual_fraction=info["mean_fraction_in_range"],
            prob_full_accrual=info["prob_full_accrual"],
            prob_zero_accrual=info["prob_zero_accrual"],
            expected_coupon=expected_coupon,
            n_paths=self.n_paths,
            n_observations=n_obs,
            elapsed_seconds=elapsed,
        )
    
    def _get_n_observations(
        self,
        instrument: IrRangeAccrualNote,
        valuation_date: date,
    ) -> int:
        """Determine number of observations based on frequency."""
        total_days = (instrument.maturity_date - valuation_date).days
        
        if instrument.observation_frequency == ObservationFrequency.DAILY:
            return max(1, total_days)
        elif instrument.observation_frequency == ObservationFrequency.WEEKLY:
            return max(1, total_days // 7)
        elif instrument.observation_frequency == ObservationFrequency.MONTHLY:
            months = (instrument.maturity_date.year - valuation_date.year) * 12 + \
                     (instrument.maturity_date.month - valuation_date.month)
            return max(1, months)
        
        return max(1, total_days)
    
    def _simulate_hw_paths(
        self,
        initial_rate: float,
        mean_reversion: float,
        volatility: float,
        long_term_rate: float,
        T: float,
        n_obs: int,
    ) -> np.ndarray:
        """
        Simulate Hull-White short rate paths.
        
        Uses Euler-Maruyama discretization:
            r_{t+dt} = r_t + a(θ - r_t)dt + σ√dt * Z
        
        Parameters
        ----------
        initial_rate : float
            Initial short rate.
        mean_reversion : float
            Mean reversion speed (a).
        volatility : float
            Short rate volatility (σ).
        long_term_rate : float
            Long-term rate level (θ).
        T : float
            Time horizon.
        n_obs : int
            Number of observations.
        
        Returns
        -------
        ndarray
            Rate paths of shape (n_paths, n_obs).
        """
        dt = T / n_obs
        
        n_paths = self.n_paths
        if self.antithetic:
            n_paths = n_paths // 2
        
        # Random increments
        Z = self._rng.standard_normal((n_paths, n_obs))
        
        if self.antithetic:
            Z = np.vstack([Z, -Z])
        
        # Initialize paths
        paths = np.zeros((len(Z), n_obs))
        r = np.full(len(Z), initial_rate)
        
        # Simulate
        sqrt_dt = np.sqrt(dt)
        a = mean_reversion
        theta = long_term_rate
        sigma = volatility
        
        for i in range(n_obs):
            # Euler step
            dr = a * (theta - r) * dt + sigma * sqrt_dt * Z[:, i]
            r = r + dr
            
            # Store (ensure non-negative rates)
            paths[:, i] = np.maximum(r, 0.0)
        
        return paths


__all__ = [
    "IrRangeAccrualHwMcPricer",
    "RangeAccrualPricingResult",
    "RangeAccrualMarketData",
]
