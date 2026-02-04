"""
Monte Carlo pricer for Autocallable options under GBM dynamics.

Prices autocallable products by simulating GBM paths and evaluating
the path-dependent payoff with early termination.

Example:
    from src.pricers.equity.autocallable_gbm_mc import EquityAutocallableGbmMcPricer
    from src.instruments.equity.options.autocallable import EquityAutocallableOption
    
    pricer = EquityAutocallableGbmMcPricer(n_paths=100_000)
    
    autocall = EquityAutocallableOption(
        underlying_id="SPY",
        notional=100_000,
        start_date=date(2024, 1, 1),
        maturity_date=date(2026, 1, 1),
        observation_dates=[date(2024, 7, 1), date(2025, 1, 1), ...],
        autocall_barrier=1.0,
        coupon_barrier=0.8,
        put_barrier=0.6,
        coupon_rate=0.12,
    )
    
    result = pricer.price(autocall, market_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np

from src.instruments.equity.options.autocallable import EquityAutocallableOption
from src.models.payoffs.autocallable import AutocallablePayoff


# =============================================================================
# Pricing Result
# =============================================================================


@dataclass
class AutocallablePricingResult:
    """Result from autocallable pricing."""
    
    price: float
    standard_error: float
    
    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    
    # Autocall-specific metrics
    prob_autocall: float = 0.0
    expected_autocall_period: float = -1.0
    prob_put_loss: float = 0.0
    expected_coupon: float = 0.0
    
    # Diagnostics
    n_paths: int = 0
    elapsed_seconds: float = 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "price": self.price,
            "standard_error": self.standard_error,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "prob_autocall": self.prob_autocall,
            "expected_autocall_period": self.expected_autocall_period,
            "prob_put_loss": self.prob_put_loss,
            "expected_coupon": self.expected_coupon,
        }


# =============================================================================
# Market Data
# =============================================================================


@dataclass
class AutocallableMarketData:
    """Market data for autocallable pricing."""
    
    spot: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    valuation_date: Optional[date] = None


# =============================================================================
# Autocallable Pricer
# =============================================================================


class EquityAutocallableGbmMcPricer:
    """
    Monte Carlo pricer for Equity Autocallable options under GBM.
    
    Simulates GBM paths at observation dates and evaluates the
    autocallable payoff with early termination, coupons, and put protection.
    
    Features:
    - Early termination modeling
    - Memory coupon handling
    - Put barrier protection
    - Greek computation
    - Autocall probability estimation
    
    Example:
        pricer = EquityAutocallableGbmMcPricer(n_paths=100_000)
        
        result = pricer.price(autocallable, market_data)
        print(f"Price: {result.price:.2f}")
        print(f"Autocall probability: {result.prob_autocall:.1%}")
    """
    
    def __init__(
        self,
        n_paths: int = 100_000,
        seed: Optional[int] = None,
        antithetic: bool = True,
        compute_greeks: bool = True,
    ) -> None:
        """
        Initialize autocallable pricer.
        
        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            Random seed.
        antithetic : bool
            Use antithetic variance reduction.
        compute_greeks : bool
            Compute Greeks.
        """
        self.n_paths = n_paths
        self.seed = seed
        self.antithetic = antithetic
        self.compute_greeks = compute_greeks
        self._rng = np.random.default_rng(seed)
    
    def price(
        self,
        instrument: EquityAutocallableOption,
        market_data: AutocallableMarketData,
    ) -> AutocallablePricingResult:
        """
        Price an autocallable option.
        
        Parameters
        ----------
        instrument : EquityAutocallableOption
            Autocallable to price.
        market_data : AutocallableMarketData
            Market data.
        
        Returns
        -------
        AutocallablePricingResult
            Pricing result.
        """
        import time
        start_time = time.time()
        
        valuation_date = market_data.valuation_date or instrument.start_date
        
        # Get observation times
        obs_times = self._get_observation_times(
            valuation_date,
            instrument.start_date,
            instrument.observation_dates,
            instrument.maturity_date,
        )
        
        if len(obs_times) == 0:
            return AutocallablePricingResult(
                price=0.0,
                standard_error=0.0,
                n_paths=0,
            )
        
        # Create payoff
        payoff = AutocallablePayoff(
            autocall_barrier=instrument.autocall_barrier,
            coupon_barrier=instrument.coupon_barrier,
            put_barrier=instrument.put_barrier,
            coupon_rate=instrument.coupon_rate,
            observation_times=obs_times,
            memory_coupon=instrument.memory_coupon,
            initial_spot=market_data.spot,
        )
        
        # Simulate paths
        paths = self._simulate_paths(
            spot=market_data.spot,
            volatility=market_data.volatility,
            rate=market_data.risk_free_rate,
            dividend=market_data.dividend_yield,
            obs_times=obs_times,
        )
        
        # Compute payoffs
        payoffs, info = payoff.terminal_from_paths_with_info(paths)
        payoffs = payoffs * instrument.notional
        
        # Discount to valuation date
        # Need to discount each path based on when it terminates
        T_max = obs_times[-1]
        
        # For simplicity, discount at max time
        # (more accurate: discount autocalled paths at their termination time)
        discount = np.exp(-market_data.risk_free_rate * T_max)
        discounted_payoffs = payoffs * discount
        
        # Statistics
        price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs)))
        
        # Autocall-specific metrics
        prob_autocall = info["prob_autocall"]
        expected_period = info["mean_autocall_period"]
        
        # Probability of put loss
        final_perf = paths[:, -1] / paths[:, 0]
        prob_put_loss = float(np.mean(
            (final_perf < instrument.put_barrier) & ~info["autocalled"]
        ))
        
        # Expected coupon
        expected_coupon = float(np.mean(info["total_coupons_paid"]) * instrument.notional)
        
        # Greeks
        delta, gamma, vega = None, None, None
        if self.compute_greeks:
            delta, gamma = self._compute_delta_gamma(
                instrument, market_data, payoff, obs_times
            )
            vega = self._compute_vega(
                instrument, market_data, payoff, obs_times
            )
        
        elapsed = time.time() - start_time
        
        return AutocallablePricingResult(
            price=price,
            standard_error=std_error,
            delta=delta,
            gamma=gamma,
            vega=vega,
            prob_autocall=prob_autocall,
            expected_autocall_period=expected_period,
            prob_put_loss=prob_put_loss,
            expected_coupon=expected_coupon,
            n_paths=self.n_paths,
            elapsed_seconds=elapsed,
        )
    
    def _get_observation_times(
        self,
        valuation_date: date,
        start_date: date,
        observation_dates: List[date],
        maturity_date: date,
    ) -> List[float]:
        """Get observation times in years from valuation date."""
        # Filter to future dates
        future_obs = [d for d in observation_dates if d > valuation_date]
        
        # Convert to year fractions
        times = [(d - valuation_date).days / 365.0 for d in future_obs]
        
        return times
    
    def _simulate_paths(
        self,
        spot: float,
        volatility: float,
        rate: float,
        dividend: float,
        obs_times: List[float],
    ) -> np.ndarray:
        """Simulate GBM paths at observation times."""
        n_obs = len(obs_times)
        
        if n_obs == 0:
            return np.full((self.n_paths, 1), spot)
        
        # Time deltas
        times_with_zero = [0.0] + obs_times
        dt = np.diff(times_with_zero)
        
        # Random increments
        n_paths = self.n_paths
        if self.antithetic:
            n_paths = n_paths // 2
        
        Z = self._rng.standard_normal((n_paths, n_obs))
        
        if self.antithetic:
            Z = np.vstack([Z, -Z])
        
        # Build paths
        drift = (rate - dividend - 0.5 * volatility**2)
        
        paths = np.zeros((len(Z), n_obs + 1))
        paths[:, 0] = spot
        
        for i, (dti, Zi) in enumerate(zip(dt, Z.T)):
            if dti > 0:
                log_return = drift * dti + volatility * np.sqrt(dti) * Zi
                paths[:, i + 1] = paths[:, i] * np.exp(log_return)
            else:
                paths[:, i + 1] = paths[:, i]
        
        return paths
    
    def _compute_delta_gamma(
        self,
        instrument: EquityAutocallableOption,
        market_data: AutocallableMarketData,
        payoff: AutocallablePayoff,
        obs_times: List[float],
    ) -> tuple:
        """Compute delta and gamma."""
        bump = market_data.spot * 0.01
        T = obs_times[-1] if obs_times else 1.0
        
        # Base
        paths_base = self._simulate_paths(
            market_data.spot, market_data.volatility,
            market_data.risk_free_rate, market_data.dividend_yield,
            obs_times,
        )
        payoffs_base, _ = payoff.terminal_from_paths_with_info(paths_base)
        price_base = np.mean(payoffs_base) * np.exp(-market_data.risk_free_rate * T)
        
        # Up
        paths_up = self._simulate_paths(
            market_data.spot + bump, market_data.volatility,
            market_data.risk_free_rate, market_data.dividend_yield,
            obs_times,
        )
        payoffs_up, _ = payoff.terminal_from_paths_with_info(paths_up)
        price_up = np.mean(payoffs_up) * np.exp(-market_data.risk_free_rate * T)
        
        # Down
        paths_down = self._simulate_paths(
            market_data.spot - bump, market_data.volatility,
            market_data.risk_free_rate, market_data.dividend_yield,
            obs_times,
        )
        payoffs_down, _ = payoff.terminal_from_paths_with_info(paths_down)
        price_down = np.mean(payoffs_down) * np.exp(-market_data.risk_free_rate * T)
        
        delta = (price_up - price_down) / (2 * bump) * instrument.notional
        gamma = (price_up - 2 * price_base + price_down) / (bump ** 2) * instrument.notional
        
        return float(delta), float(gamma)
    
    def _compute_vega(
        self,
        instrument: EquityAutocallableOption,
        market_data: AutocallableMarketData,
        payoff: AutocallablePayoff,
        obs_times: List[float],
    ) -> float:
        """Compute vega."""
        vol_bump = 0.01
        T = obs_times[-1] if obs_times else 1.0
        
        # Up vol
        paths_up = self._simulate_paths(
            market_data.spot, market_data.volatility + vol_bump,
            market_data.risk_free_rate, market_data.dividend_yield,
            obs_times,
        )
        payoffs_up, _ = payoff.terminal_from_paths_with_info(paths_up)
        price_up = np.mean(payoffs_up) * np.exp(-market_data.risk_free_rate * T)
        
        # Down vol
        paths_down = self._simulate_paths(
            market_data.spot, market_data.volatility - vol_bump,
            market_data.risk_free_rate, market_data.dividend_yield,
            obs_times,
        )
        payoffs_down, _ = payoff.terminal_from_paths_with_info(paths_down)
        price_down = np.mean(payoffs_down) * np.exp(-market_data.risk_free_rate * T)
        
        vega = (price_up - price_down) / (2 * vol_bump) * instrument.notional
        
        return float(vega)


__all__ = [
    "EquityAutocallableGbmMcPricer",
    "AutocallablePricingResult",
    "AutocallableMarketData",
]
