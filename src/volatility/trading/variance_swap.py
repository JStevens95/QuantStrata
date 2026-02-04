"""
Variance Swap Pricing and Hedging.

Implements variance swap valuation using:
- Log-strip replication (Carr-Madan)
- Discrete monitoring adjustments
- Greeks and hedging ratios

Reference:
- Demeterfi, Derman, Kamal, Zou (1999) "A Guide to Volatility and Variance Swaps"
- Carr & Madan (1998) "Towards a theory of volatility trading"

Example:
    from src.volatility.trading import VarianceSwap, VarianceSwapPricer
    
    # Create variance swap
    swap = VarianceSwap(
        strike_var=0.04,     # 20% vol strike
        maturity=0.5,        # 6 months
        notional=100_000,    # Variance notional
    )
    
    # Price using option strip
    pricer = VarianceSwapPricer()
    result = pricer.price(swap, spot=100, forward=101, option_chain=chain)
    
    print(f"Fair variance: {result.fair_variance:.4f}")
    print(f"Fair vol: {result.fair_vol:.2%}")
    print(f"Swap MTM: ${result.mtm:,.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Variance Swap Instrument
# =============================================================================


@dataclass
class VarianceSwap:
    """
    Variance Swap instrument.
    
    Payoff at maturity:
        notional * (realized_variance - strike_variance)
    
    Or for volatility notional:
        vega_notional * (realized_vol - strike_vol)
    
    Attributes
    ----------
    strike_var : float
        Strike variance (e.g., 0.04 for 20% vol strike).
    maturity : float
        Time to maturity in years.
    notional : float
        Variance notional.
    start_date : date, optional
        Trade start date.
    end_date : date, optional
        Maturity date.
    observation_frequency : str
        Frequency of variance observations.
    """
    
    strike_var: float
    maturity: float
    notional: float = 100_000
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    observation_frequency: str = "daily"
    
    @property
    def strike_vol(self) -> float:
        """Strike volatility."""
        return np.sqrt(self.strike_var)
    
    @property
    def vega_notional(self) -> float:
        """Vega notional (variance notional / 2 * strike_vol)."""
        return self.notional * 2 * self.strike_vol
    
    @classmethod
    def from_vol(
        cls,
        strike_vol: float,
        maturity: float,
        vega_notional: float,
    ) -> "VarianceSwap":
        """
        Create variance swap from volatility terms.
        
        Parameters
        ----------
        strike_vol : float
            Strike volatility (e.g., 0.20 for 20%).
        maturity : float
            Time to maturity.
        vega_notional : float
            Vega notional.
        
        Returns
        -------
        VarianceSwap
        """
        strike_var = strike_vol ** 2
        notional = vega_notional / (2 * strike_vol)
        return cls(strike_var=strike_var, maturity=maturity, notional=notional)
    
    def payoff(self, realized_variance: float) -> float:
        """
        Compute payoff given realized variance.
        
        Parameters
        ----------
        realized_variance : float
            Realized variance over the swap period.
        
        Returns
        -------
        float
            Payoff amount.
        """
        return self.notional * (realized_variance - self.strike_var)


# =============================================================================
# Pricing Result
# =============================================================================


@dataclass
class VarianceSwapResult:
    """Result from variance swap pricing."""
    
    # Fair value
    fair_variance: float
    fair_vol: float
    swap_value: float  # Mark-to-market
    
    # Decomposition
    replication_cost: float
    discrete_adjustment: float
    
    # Greeks
    vega: float = 0.0  # Sensitivity to fair variance change
    theta: float = 0.0  # Time decay
    
    # Replication details
    put_contribution: float = 0.0
    call_contribution: float = 0.0
    forward_adjustment: float = 0.0
    
    def summary(self) -> Dict[str, float]:
        """Get summary dictionary."""
        return {
            "fair_variance": self.fair_variance,
            "fair_vol": self.fair_vol,
            "swap_value": self.swap_value,
            "vega": self.vega,
            "theta": self.theta,
        }


# =============================================================================
# Variance Swap Pricer
# =============================================================================


class VarianceSwapPricer:
    """
    Price variance swaps using log-strip replication.
    
    The fair variance is replicated by a static portfolio of
    out-of-the-money puts and calls:
    
    σ²_fair = (2/T) * [∫₀^F (K⁻²)P(K)dK + ∫_F^∞ (K⁻²)C(K)dK]
    
    Where P(K), C(K) are put/call prices at strike K, F is forward.
    
    Example:
        pricer = VarianceSwapPricer()
        
        # With option chain
        result = pricer.price(
            swap=swap,
            spot=100,
            forward=101,
            option_strikes=strikes,
            option_prices_call=call_prices,
            option_prices_put=put_prices,
            rate=0.05,
        )
        
        # With implied volatility surface
        result = pricer.price_from_vol_surface(
            swap=swap,
            spot=100,
            forward=101,
            vol_surface=surface,
            rate=0.05,
        )
    """
    
    def __init__(
        self,
        n_integration_points: int = 100,
    ) -> None:
        """
        Initialize pricer.
        
        Parameters
        ----------
        n_integration_points : int
            Points for numerical integration.
        """
        self.n_integration_points = n_integration_points
    
    def price(
        self,
        swap: VarianceSwap,
        spot: float,
        forward: float,
        option_strikes: np.ndarray,
        option_prices_call: np.ndarray,
        option_prices_put: np.ndarray,
        rate: float = 0.05,
    ) -> VarianceSwapResult:
        """
        Price variance swap from option strip.
        
        Parameters
        ----------
        swap : VarianceSwap
            Variance swap to price.
        spot : float
            Current spot price.
        forward : float
            Forward price.
        option_strikes : ndarray
            Option strikes.
        option_prices_call : ndarray
            Call option prices.
        option_prices_put : ndarray
            Put option prices.
        rate : float
            Risk-free rate.
        
        Returns
        -------
        VarianceSwapResult
            Pricing result.
        """
        T = swap.maturity
        discount = np.exp(-rate * T)
        
        # Sort by strike
        sort_idx = np.argsort(option_strikes)
        K = option_strikes[sort_idx]
        C = option_prices_call[sort_idx]
        P = option_prices_put[sort_idx]
        
        # Find ATM index
        atm_idx = np.argmin(np.abs(K - forward))
        
        # Integrate using trapezoidal rule
        # Puts for K < F, Calls for K >= F
        
        # Put integral
        put_integral = 0.0
        for i in range(atm_idx):
            if i > 0:
                dK = K[i] - K[i - 1]
                weight = 1.0 / K[i] ** 2
                put_integral += 0.5 * (P[i] + P[i - 1]) * weight * dK
        
        # Call integral
        call_integral = 0.0
        for i in range(atm_idx, len(K) - 1):
            dK = K[i + 1] - K[i]
            weight = 1.0 / K[i] ** 2
            call_integral += 0.5 * (C[i] + C[i + 1]) * weight * dK
        
        # Forward adjustment (log term)
        forward_adj = -np.log(forward / K[atm_idx]) - (forward / K[atm_idx] - 1)
        
        # Fair variance
        fair_var = (2 / T) / discount * (put_integral + call_integral)
        fair_var += (2 / T) * forward_adj
        
        # Discrete monitoring adjustment (approx)
        n_obs = int(T * 252)  # Daily observations
        discrete_adj = fair_var / (3 * n_obs)
        
        fair_var_discrete = fair_var + discrete_adj
        fair_vol = np.sqrt(max(fair_var_discrete, 0))
        
        # Swap value
        swap_value = discount * swap.notional * (fair_var_discrete - swap.strike_var)
        
        # Simple Greeks
        vega = discount * swap.notional  # Per unit variance
        theta = rate * swap_value - swap.notional * fair_var / T
        
        return VarianceSwapResult(
            fair_variance=fair_var_discrete,
            fair_vol=fair_vol,
            swap_value=swap_value,
            replication_cost=(put_integral + call_integral) / discount,
            discrete_adjustment=discrete_adj,
            vega=vega,
            theta=theta,
            put_contribution=put_integral,
            call_contribution=call_integral,
            forward_adjustment=forward_adj,
        )
    
    def price_from_vol_surface(
        self,
        swap: VarianceSwap,
        spot: float,
        forward: float,
        vol_surface: Any,  # VolSurface object
        rate: float = 0.05,
        strike_range: Tuple[float, float] = (0.5, 2.0),
    ) -> VarianceSwapResult:
        """
        Price from volatility surface by generating synthetic option prices.
        
        Parameters
        ----------
        swap : VarianceSwap
            Variance swap to price.
        spot : float
            Current spot.
        forward : float
            Forward price.
        vol_surface : VolSurface
            Implied volatility surface.
        rate : float
            Risk-free rate.
        strike_range : tuple
            Strike range as fraction of forward (min, max).
        
        Returns
        -------
        VarianceSwapResult
            Pricing result.
        """
        T = swap.maturity
        
        # Generate strikes
        K_min = forward * strike_range[0]
        K_max = forward * strike_range[1]
        K = np.linspace(K_min, K_max, self.n_integration_points)
        
        # Get implied vols and compute option prices
        from scipy.stats import norm
        
        call_prices = np.zeros(len(K))
        put_prices = np.zeros(len(K))
        
        discount = np.exp(-rate * T)
        
        for i, strike in enumerate(K):
            # Get vol from surface (assume surface has get_vol method)
            if hasattr(vol_surface, "get_vol"):
                vol = vol_surface.get_vol(T, strike)
            else:
                # Use constant vol if surface not provided
                vol = 0.20
            
            # Black-Scholes prices
            d1 = (np.log(forward / strike) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            
            call_prices[i] = discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
            put_prices[i] = discount * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
        
        return self.price(
            swap=swap,
            spot=spot,
            forward=forward,
            option_strikes=K,
            option_prices_call=call_prices,
            option_prices_put=put_prices,
            rate=rate,
        )
    
    def price_simple(
        self,
        swap: VarianceSwap,
        implied_vol: float,
        rate: float = 0.05,
    ) -> VarianceSwapResult:
        """
        Simple pricing assuming flat volatility.
        
        For flat vol, fair variance = implied_vol^2.
        
        Parameters
        ----------
        swap : VarianceSwap
            Variance swap to price.
        implied_vol : float
            Flat implied volatility.
        rate : float
            Risk-free rate.
        
        Returns
        -------
        VarianceSwapResult
            Pricing result.
        """
        T = swap.maturity
        discount = np.exp(-rate * T)
        
        fair_var = implied_vol ** 2
        
        # Discrete adjustment
        n_obs = int(T * 252)
        discrete_adj = fair_var / (3 * n_obs)
        fair_var_discrete = fair_var + discrete_adj
        
        swap_value = discount * swap.notional * (fair_var_discrete - swap.strike_var)
        
        return VarianceSwapResult(
            fair_variance=fair_var_discrete,
            fair_vol=np.sqrt(fair_var_discrete),
            swap_value=swap_value,
            replication_cost=0.0,
            discrete_adjustment=discrete_adj,
            vega=discount * swap.notional,
            theta=-swap_value * rate,
        )


# =============================================================================
# Realized Variance Calculator
# =============================================================================


def calculate_realized_variance(
    prices: np.ndarray,
    annualization: int = 252,
    log_returns: bool = True,
) -> float:
    """
    Calculate realized variance from price series.
    
    Parameters
    ----------
    prices : ndarray
        Price series.
    annualization : int
        Number of observations per year.
    log_returns : bool
        Use log returns (True) or simple returns (False).
    
    Returns
    -------
    float
        Annualized realized variance.
    """
    if len(prices) < 2:
        return 0.0
    
    if log_returns:
        returns = np.log(prices[1:] / prices[:-1])
    else:
        returns = prices[1:] / prices[:-1] - 1
    
    # Sum of squared returns (variance swap definition)
    realized_var = np.sum(returns ** 2)
    
    # Annualize
    n_obs = len(returns)
    realized_var = realized_var * (annualization / n_obs)
    
    return float(realized_var)


__all__ = [
    "VarianceSwap",
    "VarianceSwapPricer",
    "VarianceSwapResult",
    "calculate_realized_variance",
]
