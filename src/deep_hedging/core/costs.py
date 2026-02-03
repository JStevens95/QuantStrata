"""
Transaction Cost Models for Deep Hedging

This module provides generic, composable transaction cost models used in hedging
simulations. Costs are computed as a function of trade size and market state.

Theory
------
Transaction costs create a fundamental trade-off in hedging:
- Frequent rehedging → better tracking but higher costs
- Infrequent rehedging → lower costs but more tracking error

Cost models implemented:
1. ProportionalCost: C = κ * |Δδ| * S (bid-ask spread)
2. FixedCost: C = c * 1_{Δδ ≠ 0} (per-trade cost)
3. MarketImpactCost: C = λ * |Δδ|^α * S (temporary impact)
4. CombinedCost: Sum of multiple cost components

All costs are designed to be:
- Stateless (pure functions of trade and market state)
- Composable (can combine multiple cost types)
- Differentiable (for gradient-based training)

References
----------
- Almgren & Chriss (2001) "Optimal execution of portfolio transactions"
- Gatheral (2010) "No-dynamic-arbitrage and market impact"
- docs/reference/deep_hedging/theory.md Section 7
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


class TransactionCostModel(ABC):
    """
    Abstract base class for transaction cost models.
    
    A transaction cost model computes the cost of executing a trade given:
    - The trade size (change in position)
    - The current market state (spot price, volatility, etc.)
    
    All implementations must be:
    - Deterministic given inputs
    - Non-negative (costs are always >= 0)
    - Zero for zero trade size (except fixed costs)
    
    Mathematical Framework
    ----------------------
    Cost function: C(Δδ, S, σ, ...) → R⁺
    
    where:
    - Δδ: trade size (change in hedge position)
    - S: spot price
    - σ: volatility (optional)
    - ...: other market state variables
    
    Example
    -------
    >>> cost_model = ProportionalCost(spread_bps=5.0)
    >>> cost = cost_model.compute(trade_size=100.0, spot=150.0)
    >>> print(f"Cost: ${cost:.2f}")
    Cost: $7.50
    """
    
    @abstractmethod
    def compute(
        self,
        trade_size: Union[float, np.ndarray],
        spot: Union[float, np.ndarray],
        volatility: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Compute the transaction cost for a given trade.
        
        Parameters
        ----------
        trade_size : float or ndarray
            The size of the trade (Δδ = δ_new - δ_old).
            Positive = buying, negative = selling.
            Can be scalar or array for vectorised computation.
        spot : float or ndarray
            Current spot price S.
        volatility : float or ndarray, optional
            Current volatility σ (for volatility-dependent costs).
        **kwargs : dict
            Additional market state variables (e.g., volume, time of day).
        
        Returns
        -------
        float or ndarray
            Transaction cost(s). Same shape as input arrays.
            Always non-negative.
        """
        pass
    
    def __add__(self, other: "TransactionCostModel") -> "CombinedCost":
        """Combine two cost models: (C1 + C2)(Δδ, S) = C1(Δδ, S) + C2(Δδ, S)."""
        if isinstance(other, CombinedCost):
            return CombinedCost(components=[self] + other.components)
        return CombinedCost(components=[self, other])
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class ProportionalCost(TransactionCostModel):
    """
    Proportional (bid-ask spread) transaction cost model.
    
    Mathematical Form
    -----------------
    C(Δδ, S) = κ * S * |Δδ|
    
    where:
    - κ: half-spread as a fraction (e.g., 0.0001 = 1 bp)
    - S: spot price
    - |Δδ|: absolute trade size
    
    This models the bid-ask spread: buying at ask, selling at bid.
    Cost is proportional to trade notional value.
    
    Parameters
    ----------
    spread_bps : float
        Bid-ask spread in basis points.
        The half-spread κ = spread_bps / 10000 / 2.
        E.g., spread_bps=10 means 10bp round-trip = 5bp half-spread.
    
    Example
    -------
    >>> cost = ProportionalCost(spread_bps=10.0)  # 10bp spread
    >>> cost.compute(trade_size=1000, spot=100.0)
    50.0  # = 0.0005 * 100 * 1000
    
    Notes
    -----
    - This is the most common cost model in practice
    - Linear in trade size (no market impact)
    - Appropriate for liquid markets with tight spreads
    """
    
    spread_bps: float = 10.0  # Basis points (10bp = 0.1%)
    
    def __post_init__(self):
        if self.spread_bps < 0:
            raise ValueError(f"spread_bps must be non-negative, got {self.spread_bps}")
        # Half-spread as a fraction
        self._kappa = self.spread_bps / 10000.0 / 2.0
    
    def compute(
        self,
        trade_size: Union[float, np.ndarray],
        spot: Union[float, np.ndarray],
        volatility: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Compute proportional cost: κ * S * |Δδ|.
        
        The cost is symmetric for buys and sells (both pay spread).
        """
        trade_size = np.asarray(trade_size)
        spot = np.asarray(spot)
        
        cost = self._kappa * spot * np.abs(trade_size)
        
        # Return scalar if inputs were scalar
        if cost.ndim == 0:
            return float(cost)
        return cost
    
    @property
    def half_spread(self) -> float:
        """Half-spread κ as a fraction."""
        return self._kappa
    
    def __repr__(self) -> str:
        return f"ProportionalCost(spread_bps={self.spread_bps})"


@dataclass
class FixedCost(TransactionCostModel):
    """
    Fixed per-trade transaction cost model.
    
    Mathematical Form
    -----------------
    C(Δδ) = c * 1_{Δδ ≠ 0}
    
    where:
    - c: fixed cost per trade
    - 1_{Δδ ≠ 0}: indicator that a trade occurred
    
    This models fixed costs like:
    - Commission per trade
    - Clearing fees
    - Exchange fees
    
    Parameters
    ----------
    cost_per_trade : float
        Fixed cost charged for each trade (regardless of size).
    threshold : float
        Minimum trade size to incur cost. Trades smaller than this
        are considered "no trade" and incur zero cost.
        Default 1e-10 (essentially zero).
    
    Example
    -------
    >>> cost = FixedCost(cost_per_trade=5.0)
    >>> cost.compute(trade_size=1000, spot=100.0)
    5.0
    >>> cost.compute(trade_size=0.0, spot=100.0)
    0.0
    
    Notes
    -----
    - Creates discontinuity at zero trade size
    - Encourages lumpy trading (either trade a lot or don't trade)
    - Not differentiable at zero; use soft threshold for training
    """
    
    cost_per_trade: float = 1.0
    threshold: float = 1e-10
    
    def __post_init__(self):
        if self.cost_per_trade < 0:
            raise ValueError(f"cost_per_trade must be non-negative, got {self.cost_per_trade}")
    
    def compute(
        self,
        trade_size: Union[float, np.ndarray],
        spot: Union[float, np.ndarray],
        volatility: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Compute fixed cost: c * 1_{|Δδ| > threshold}.
        """
        trade_size = np.asarray(trade_size)
        
        # Indicator for non-trivial trade
        traded = np.abs(trade_size) > self.threshold
        cost = self.cost_per_trade * traded.astype(float)
        
        if cost.ndim == 0:
            return float(cost)
        return cost
    
    def __repr__(self) -> str:
        return f"FixedCost(cost_per_trade={self.cost_per_trade})"


@dataclass
class MarketImpactCost(TransactionCostModel):
    """
    Market impact (temporary) transaction cost model.
    
    Mathematical Form
    -----------------
    C(Δδ, S) = λ * S * |Δδ|^α
    
    where:
    - λ: impact coefficient
    - S: spot price
    - |Δδ|: absolute trade size
    - α: impact exponent (typically 1.0 to 2.0)
    
    Common parameterisations:
    - α = 1.0: Linear impact (same as proportional with different scaling)
    - α = 1.5: Square-root impact (empirically observed in equities)
    - α = 2.0: Quadratic impact
    
    This models the price movement caused by your own trading:
    - Large trades move the market against you
    - Cost is convex in trade size (superlinear for α > 1)
    
    Parameters
    ----------
    impact_coef : float
        Impact coefficient λ. Units depend on α.
    impact_exp : float
        Impact exponent α. Default 1.5 (square-root law).
    
    Example
    -------
    >>> cost = MarketImpactCost(impact_coef=0.001, impact_exp=1.5)
    >>> cost.compute(trade_size=100, spot=100.0)
    100.0  # = 0.001 * 100 * 100^1.5
    
    Notes
    -----
    - Square-root impact (α=1.5) is well-documented empirically
    - Makes large trades proportionally more expensive
    - Encourages splitting orders over time
    
    References
    ----------
    - Almgren & Chriss (2001): Optimal execution with linear impact
    - Gatheral (2010): Square-root law from market microstructure
    """
    
    impact_coef: float = 0.0001  # λ
    impact_exp: float = 1.5  # α (square-root law)
    
    def __post_init__(self):
        if self.impact_coef < 0:
            raise ValueError(f"impact_coef must be non-negative, got {self.impact_coef}")
        if self.impact_exp <= 0:
            raise ValueError(f"impact_exp must be positive, got {self.impact_exp}")
    
    def compute(
        self,
        trade_size: Union[float, np.ndarray],
        spot: Union[float, np.ndarray],
        volatility: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Compute market impact cost: λ * S * |Δδ|^α.
        """
        trade_size = np.asarray(trade_size)
        spot = np.asarray(spot)
        
        abs_trade = np.abs(trade_size)
        
        # Handle zero trade (0^α can be problematic for α < 1)
        cost = self.impact_coef * spot * np.power(abs_trade + 1e-12, self.impact_exp)
        
        # Zero out cost for zero trade
        cost = np.where(abs_trade < 1e-12, 0.0, cost)
        
        if cost.ndim == 0:
            return float(cost)
        return cost
    
    def __repr__(self) -> str:
        return f"MarketImpactCost(impact_coef={self.impact_coef}, impact_exp={self.impact_exp})"


@dataclass
class CombinedCost(TransactionCostModel):
    """
    Combined transaction cost model (sum of components).
    
    Mathematical Form
    -----------------
    C(Δδ, S, ...) = Σᵢ Cᵢ(Δδ, S, ...)
    
    Allows composing multiple cost types into a realistic cost model.
    
    Example
    -------
    >>> # Realistic cost: spread + impact + fixed
    >>> cost = CombinedCost([
    ...     ProportionalCost(spread_bps=5.0),
    ...     MarketImpactCost(impact_coef=0.0001),
    ...     FixedCost(cost_per_trade=1.0),
    ... ])
    >>> total = cost.compute(trade_size=1000, spot=100.0)
    
    Or use addition syntax:
    
    >>> cost = ProportionalCost(spread_bps=5.0) + MarketImpactCost(impact_coef=0.0001)
    """
    
    components: List[TransactionCostModel]
    
    def __post_init__(self):
        if not self.components:
            raise ValueError("CombinedCost requires at least one component")
    
    def compute(
        self,
        trade_size: Union[float, np.ndarray],
        spot: Union[float, np.ndarray],
        volatility: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Compute total cost as sum of all components.
        """
        total = 0.0
        for component in self.components:
            total = total + component.compute(
                trade_size=trade_size,
                spot=spot,
                volatility=volatility,
                **kwargs,
            )
        return total
    
    def __add__(self, other: TransactionCostModel) -> "CombinedCost":
        """Add another cost model to the combination."""
        if isinstance(other, CombinedCost):
            return CombinedCost(components=self.components + other.components)
        return CombinedCost(components=self.components + [other])
    
    def __repr__(self) -> str:
        return f"CombinedCost({self.components})"


@dataclass
class ZeroCost(TransactionCostModel):
    """
    Zero transaction cost (frictionless market).
    
    Useful for:
    - Benchmarking without costs
    - Testing hedging logic
    - Comparing to BSM assumptions
    """
    
    def compute(
        self,
        trade_size: Union[float, np.ndarray],
        spot: Union[float, np.ndarray],
        volatility: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """Return zero cost."""
        trade_size = np.asarray(trade_size)
        if trade_size.ndim == 0:
            return 0.0
        return np.zeros_like(trade_size, dtype=float)
    
    def __repr__(self) -> str:
        return "ZeroCost()"


def create_realistic_cost(
    spread_bps: float = 5.0,
    impact_coef: float = 0.0001,
    fixed_cost: float = 0.0,
) -> TransactionCostModel:
    """
    Create a realistic combined cost model.
    
    Parameters
    ----------
    spread_bps : float
        Bid-ask spread in basis points (default 5bp = 0.05%).
    impact_coef : float
        Market impact coefficient (default 0.0001).
    fixed_cost : float
        Fixed cost per trade (default 0 = no fixed cost).
    
    Returns
    -------
    TransactionCostModel
        Combined cost model with all specified components.
    
    Example
    -------
    >>> cost = create_realistic_cost(spread_bps=10.0, impact_coef=0.0002)
    >>> cost.compute(trade_size=500, spot=100.0)
    """
    components = []
    
    if spread_bps > 0:
        components.append(ProportionalCost(spread_bps=spread_bps))
    
    if impact_coef > 0:
        components.append(MarketImpactCost(impact_coef=impact_coef, impact_exp=1.5))
    
    if fixed_cost > 0:
        components.append(FixedCost(cost_per_trade=fixed_cost))
    
    if not components:
        return ZeroCost()
    
    if len(components) == 1:
        return components[0]
    
    return CombinedCost(components=components)


__all__ = [
    "TransactionCostModel",
    "ProportionalCost",
    "FixedCost",
    "MarketImpactCost",
    "CombinedCost",
    "ZeroCost",
    "create_realistic_cost",
]
