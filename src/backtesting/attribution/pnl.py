"""
P&L Attribution.

This module provides P&L decomposition into factor contributions:
- Delta P&L: Spot/forward moves
- Gamma P&L: Second-order spot effects
- Theta P&L: Time decay
- Vega P&L: Volatility moves
- Rho P&L: Rate moves
- Residual: Unexplained P&L

Example
-------
>>> from src.backtesting.attribution import attribute_pnl_to_greeks
>>>
>>> attribution = attribute_pnl_to_greeks(
...     pnl=1000.0,
...     delta=0.5,
...     gamma=0.01,
...     theta=-50.0,
...     vega=200.0,
...     rho=100.0,
...     spot_move=2.0,
...     vol_move=0.01,
...     rate_move=0.001,
...     dt=1/252,
... )
>>> print(attribution)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# =============================================================================
# P&L Breakdown
# =============================================================================

@dataclass(frozen=True, slots=True)
class PnLBreakdown:
    """
    Breakdown of P&L into component factors.
    
    Attributes
    ----------
    total_pnl : float
        Total realized P&L.
    delta_pnl : float
        P&L from delta (spot/forward moves).
    gamma_pnl : float
        P&L from gamma (second-order spot effects).
    theta_pnl : float
        P&L from theta (time decay).
    vega_pnl : float
        P&L from vega (vol moves).
    rho_pnl : float
        P&L from rho (rate moves).
    residual : float
        Unexplained P&L (higher-order effects, model error).
    explained_pnl : float
        Sum of explained components.
    explanation_ratio : float
        Fraction of P&L explained by first-order Greeks.
    """
    
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float
    residual: float
    
    @property
    def explained_pnl(self) -> float:
        """Sum of explained P&L components."""
        return self.delta_pnl + self.gamma_pnl + self.theta_pnl + self.vega_pnl + self.rho_pnl
    
    @property
    def explanation_ratio(self) -> float:
        """Fraction of P&L explained (0 to 1)."""
        if abs(self.total_pnl) < 1e-12:
            return 1.0 if abs(self.residual) < 1e-12 else 0.0
        return 1.0 - abs(self.residual) / max(abs(self.total_pnl), 1e-12)
    
    def __str__(self) -> str:
        return (
            f"PnLBreakdown\n"
            f"  Total P&L:    {self.total_pnl:+,.2f}\n"
            f"  Delta P&L:    {self.delta_pnl:+,.2f}\n"
            f"  Gamma P&L:    {self.gamma_pnl:+,.2f}\n"
            f"  Theta P&L:    {self.theta_pnl:+,.2f}\n"
            f"  Vega P&L:     {self.vega_pnl:+,.2f}\n"
            f"  Rho P&L:      {self.rho_pnl:+,.2f}\n"
            f"  Residual:     {self.residual:+,.2f}\n"
            f"  Explained:    {self.explanation_ratio:.1%}"
        )


@dataclass
class PnLAttribution:
    """
    Time series of P&L attributions.
    
    Attributes
    ----------
    dates : list
        Attribution dates.
    breakdowns : list
        PnLBreakdown for each date.
    cumulative : PnLBreakdown
        Cumulative breakdown over period.
    """
    
    dates: List = field(default_factory=list)
    breakdowns: List[PnLBreakdown] = field(default_factory=list)
    
    @property
    def cumulative(self) -> PnLBreakdown:
        """Cumulative P&L breakdown."""
        if not self.breakdowns:
            return PnLBreakdown(0, 0, 0, 0, 0, 0, 0)
        
        return PnLBreakdown(
            total_pnl=sum(b.total_pnl for b in self.breakdowns),
            delta_pnl=sum(b.delta_pnl for b in self.breakdowns),
            gamma_pnl=sum(b.gamma_pnl for b in self.breakdowns),
            theta_pnl=sum(b.theta_pnl for b in self.breakdowns),
            vega_pnl=sum(b.vega_pnl for b in self.breakdowns),
            rho_pnl=sum(b.rho_pnl for b in self.breakdowns),
            residual=sum(b.residual for b in self.breakdowns),
        )
    
    def add(self, date, breakdown: PnLBreakdown) -> None:
        """Add a daily breakdown."""
        self.dates.append(date)
        self.breakdowns.append(breakdown)
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert to arrays for analysis."""
        return {
            "total_pnl": np.array([b.total_pnl for b in self.breakdowns]),
            "delta_pnl": np.array([b.delta_pnl for b in self.breakdowns]),
            "gamma_pnl": np.array([b.gamma_pnl for b in self.breakdowns]),
            "theta_pnl": np.array([b.theta_pnl for b in self.breakdowns]),
            "vega_pnl": np.array([b.vega_pnl for b in self.breakdowns]),
            "rho_pnl": np.array([b.rho_pnl for b in self.breakdowns]),
            "residual": np.array([b.residual for b in self.breakdowns]),
        }


# =============================================================================
# Attribution Functions
# =============================================================================

def attribute_pnl_to_greeks(
    pnl: float,
    delta: float = 0.0,
    gamma: float = 0.0,
    theta: float = 0.0,
    vega: float = 0.0,
    rho: float = 0.0,
    spot_move: float = 0.0,
    vol_move: float = 0.0,
    rate_move: float = 0.0,
    dt: float = 1 / 252,
) -> PnLBreakdown:
    """
    Attribute P&L to Greek factors using Taylor expansion.
    
    P&L ≈ Δ × dS + ½Γ × dS² + Θ × dt + ν × dσ + ρ × dr + residual
    
    Parameters
    ----------
    pnl : float
        Total realized P&L.
    delta : float
        Portfolio delta.
    gamma : float
        Portfolio gamma.
    theta : float
        Portfolio theta (per day, typically negative for long options).
    vega : float
        Portfolio vega.
    rho : float
        Portfolio rho.
    spot_move : float
        Change in spot price (dS).
    vol_move : float
        Change in implied volatility (dσ).
    rate_move : float
        Change in interest rate (dr).
    dt : float
        Time elapsed (in years, e.g., 1/252 for one day).
    
    Returns
    -------
    PnLBreakdown
        P&L decomposition.
    
    Examples
    --------
    >>> breakdown = attribute_pnl_to_greeks(
    ...     pnl=1500.0,
    ...     delta=500.0,
    ...     gamma=10.0,
    ...     theta=-100.0,
    ...     vega=5000.0,
    ...     spot_move=2.0,
    ...     vol_move=0.01,
    ...     dt=1/252,
    ... )
    >>> print(breakdown.delta_pnl)  # 500 * 2 = 1000
    """
    # First-order effects
    delta_pnl = delta * spot_move
    theta_pnl = theta * dt * 252  # Convert to daily if theta is annualized
    vega_pnl = vega * vol_move
    rho_pnl = rho * rate_move
    
    # Second-order effect (gamma)
    gamma_pnl = 0.5 * gamma * spot_move ** 2
    
    # Residual
    explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
    residual = pnl - explained
    
    return PnLBreakdown(
        total_pnl=pnl,
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        theta_pnl=theta_pnl,
        vega_pnl=vega_pnl,
        rho_pnl=rho_pnl,
        residual=residual,
    )


def aggregate_attribution(
    attribution: PnLAttribution,
    frequency: str = "weekly",
) -> PnLAttribution:
    """
    Aggregate daily attribution to weekly/monthly.
    
    Parameters
    ----------
    attribution : PnLAttribution
        Daily P&L attribution.
    frequency : str
        "weekly" or "monthly".
    
    Returns
    -------
    PnLAttribution
        Aggregated attribution.
    """
    if not attribution.dates:
        return PnLAttribution()
    
    result = PnLAttribution()
    
    current_period = None
    period_breakdowns: List[PnLBreakdown] = []
    period_start_date = None
    
    for dt, breakdown in zip(attribution.dates, attribution.breakdowns):
        if frequency == "weekly":
            period = (dt.year, dt.isocalendar()[1])
        elif frequency == "monthly":
            period = (dt.year, dt.month)
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
        
        if current_period is None:
            current_period = period
            period_start_date = dt
        
        if period != current_period:
            # Aggregate previous period
            if period_breakdowns:
                agg = PnLBreakdown(
                    total_pnl=sum(b.total_pnl for b in period_breakdowns),
                    delta_pnl=sum(b.delta_pnl for b in period_breakdowns),
                    gamma_pnl=sum(b.gamma_pnl for b in period_breakdowns),
                    theta_pnl=sum(b.theta_pnl for b in period_breakdowns),
                    vega_pnl=sum(b.vega_pnl for b in period_breakdowns),
                    rho_pnl=sum(b.rho_pnl for b in period_breakdowns),
                    residual=sum(b.residual for b in period_breakdowns),
                )
                result.add(period_start_date, agg)
            
            current_period = period
            period_start_date = dt
            period_breakdowns = []
        
        period_breakdowns.append(breakdown)
    
    # Don't forget the last period
    if period_breakdowns:
        agg = PnLBreakdown(
            total_pnl=sum(b.total_pnl for b in period_breakdowns),
            delta_pnl=sum(b.delta_pnl for b in period_breakdowns),
            gamma_pnl=sum(b.gamma_pnl for b in period_breakdowns),
            theta_pnl=sum(b.theta_pnl for b in period_breakdowns),
            vega_pnl=sum(b.vega_pnl for b in period_breakdowns),
            rho_pnl=sum(b.rho_pnl for b in period_breakdowns),
            residual=sum(b.residual for b in period_breakdowns),
        )
        result.add(period_start_date, agg)
    
    return result
