# src/instruments/ir/linear/bond.py
"""
Bond Instruments (Zero Coupon and Fixed Rate).

Mathematical Framework
----------------------

Zero Coupon Bond:
    A zero coupon bond pays a single face value at maturity T.
    
    PV = Face × DF(T)
    
    Where:
    - Face = face/par value (typically 100 or 1000)
    - DF(T) = discount factor to maturity
    - T = time to maturity in years

Fixed Rate (Coupon) Bond:
    A coupon bond pays periodic coupons plus face value at maturity.
    
    PV = Σ(C_i × DF(T_i)) + Face × DF(T_n)
    
    Where:
    - C_i = coupon payment at time T_i
    - C_i = Face × coupon_rate × accrual_factor_i
    - Face = face/par value
    - T_n = maturity date
    - DF(T_i) = discount factor to each payment date

Bond Pricing Conventions
------------------------
- Clean price: Price excluding accrued interest
- Dirty price: Price including accrued interest (full price)
- Accrued interest = Face × coupon_rate × (days since last coupon / days in period)

Yield Measures
--------------
- Yield to Maturity (YTM): Internal rate of return
- Current Yield: Annual coupon / Clean price
- Duration: Sensitivity to yield changes
- Convexity: Second-order sensitivity

Greeks (Risk Measures)
----------------------
- DV01: Change in PV for 1bp parallel shift in yields
- Duration: -1/PV × dPV/dy (modified duration)
- Convexity: 1/PV × d²PV/dy² (curvature)

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from src.marketdata.core.ids import MarketId
from src.instruments.core.types import DayCountConvention


# =============================================================================
# ZERO COUPON BOND - SIMPLE INSTRUMENT
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondZeroCouponSimple:
    """
    Zero coupon bond with direct parameter input.
    
    A zero coupon bond pays a single face value at maturity. There are no
    intermediate coupon payments.
    
    Parameters
    ----------
    face_value : float
        Face/par value paid at maturity (e.g., 100, 1000).
    maturity : float
        Time to maturity in years.
    discount_factor : float
        Discount factor DF(T) to maturity.
    
    Examples
    --------
    5-year zero coupon bond:
        >>> bond = IrBondZeroCouponSimple(
        ...     face_value=100.0,
        ...     maturity=5.0,
        ...     discount_factor=0.85,
        ... )
    
    Notes
    -----
    Zero coupon bonds are the building blocks of yield curve construction.
    The discount factor can be derived from the zero rate:
        DF(T) = exp(-r × T)  [continuous compounding]
        DF(T) = 1 / (1 + r)^T  [annual compounding]
    """
    face_value: float
    maturity: float                 # T = time to maturity in years
    discount_factor: float          # DF(T) = discount factor to maturity
    
    def __post_init__(self) -> None:
        """Validate zero coupon bond inputs."""
        # Face value must be positive.
        if float(self.face_value) <= 0.0:
            raise ValueError("face_value must be > 0.")
        
        # Maturity must be non-negative (0 = immediate settlement).
        if float(self.maturity) < 0.0:
            raise ValueError("maturity must be >= 0.")
        
        # Discount factor must be positive and <= 1 for positive rates.
        if float(self.discount_factor) <= 0.0:
            raise ValueError("discount_factor must be > 0.")
    
    @property
    def implied_zero_rate(self) -> float:
        """
        Implied continuously compounded zero rate.
        
        r = -ln(DF) / T
        
        Returns
        -------
        float
            Zero rate (e.g., 0.05 for 5%).
        """
        import math
        
        if float(self.maturity) <= 0.0:
            return 0.0
        
        return -math.log(float(self.discount_factor)) / float(self.maturity)


# =============================================================================
# ZERO COUPON BOND - MARKET DATA INSTRUMENT
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondZeroCoupon:
    """
    Zero coupon bond with market data lookup.
    
    Parameters
    ----------
    face_value : float
        Face/par value paid at maturity.
    maturity : float
        Time to maturity in years.
    curve_id : MarketId
        Market identifier for the discount curve.
    
    Examples
    --------
    5-year US Treasury zero coupon:
        >>> bond = IrBondZeroCoupon(
        ...     face_value=100.0,
        ...     maturity=5.0,
        ...     curve_id=MarketId("IR", "CURVE", "USD.GOVT"),
        ... )
    """
    face_value: float
    maturity: float
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate zero coupon bond inputs."""
        if float(self.face_value) <= 0.0:
            raise ValueError("face_value must be > 0.")
        if float(self.maturity) < 0.0:
            raise ValueError("maturity must be >= 0.")


# =============================================================================
# FIXED RATE BOND - SIMPLE INSTRUMENT
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondFixedRateSimple:
    """
    Fixed rate coupon bond with direct parameter input.
    
    A fixed rate bond pays periodic coupons at a fixed rate plus the
    face value at maturity.
    
    Parameters
    ----------
    face_value : float
        Face/par value paid at maturity.
    coupon_rate : float
        Annual coupon rate (e.g., 0.05 for 5%).
    coupon_times : Tuple[float, ...]
        Times (in years) of each coupon payment.
    coupon_dfs : Tuple[float, ...]
        Discount factors to each coupon payment date.
    accrued_interest : float
        Accrued interest since last coupon (for dirty price).
    
    Examples
    --------
    5-year annual coupon bond:
        >>> bond = IrBondFixedRateSimple(
        ...     face_value=100.0,
        ...     coupon_rate=0.05,           # 5% annual coupon
        ...     coupon_times=(1.0, 2.0, 3.0, 4.0, 5.0),
        ...     coupon_dfs=(0.97, 0.94, 0.91, 0.88, 0.85),
        ...     accrued_interest=2.5,       # Half-year accrued
        ... )
    
    Notes
    -----
    - Coupon payment at time T_i = Face × coupon_rate × accrual_factor
    - For annual bonds, accrual_factor = 1.0
    - For semi-annual bonds, accrual_factor = 0.5
    - The last coupon_time should equal maturity
    """
    face_value: float
    coupon_rate: float              # Annual coupon rate
    coupon_times: Tuple[float, ...]  # Payment times in years
    coupon_dfs: Tuple[float, ...]    # Discount factors to each payment
    accrued_interest: float = 0.0   # Accrued since last coupon
    
    def __post_init__(self) -> None:
        """Validate fixed rate bond inputs."""
        # Face value must be positive.
        if float(self.face_value) <= 0.0:
            raise ValueError("face_value must be > 0.")
        
        # Coupon rate must be non-negative (0 = zero coupon).
        if float(self.coupon_rate) < 0.0:
            raise ValueError("coupon_rate must be >= 0.")
        
        # Must have at least one coupon time (maturity).
        if len(self.coupon_times) == 0:
            raise ValueError("coupon_times must have at least one element.")
        
        # Coupon times and discount factors must match.
        if len(self.coupon_times) != len(self.coupon_dfs):
            raise ValueError("coupon_times and coupon_dfs must have same length.")
        
        # Coupon times must be sorted and positive.
        for i, t in enumerate(self.coupon_times):
            if t <= 0.0:
                raise ValueError(f"coupon_times[{i}] must be > 0.")
            if i > 0 and t <= self.coupon_times[i - 1]:
                raise ValueError("coupon_times must be strictly increasing.")
        
        # Discount factors must be positive.
        for i, df in enumerate(self.coupon_dfs):
            if df <= 0.0:
                raise ValueError(f"coupon_dfs[{i}] must be > 0.")
        
        # Accrued interest must be non-negative.
        if float(self.accrued_interest) < 0.0:
            raise ValueError("accrued_interest must be >= 0.")
    
    @property
    def maturity(self) -> float:
        """Time to maturity (last coupon payment date)."""
        return float(self.coupon_times[-1])
    
    @property
    def n_coupons(self) -> int:
        """Number of remaining coupon payments."""
        return len(self.coupon_times)
    
    @property
    def coupon_frequency(self) -> int:
        """
        Estimated coupon frequency per year.
        
        Based on the spacing between coupon payments.
        """
        if len(self.coupon_times) < 2:
            return 1  # Assume annual if only one payment
        
        # Average time between coupons
        avg_period = (float(self.coupon_times[-1]) - float(self.coupon_times[0])) / (len(self.coupon_times) - 1)
        
        # Round to common frequencies
        if avg_period <= 0.3:
            return 4  # Quarterly
        elif avg_period <= 0.6:
            return 2  # Semi-annual
        else:
            return 1  # Annual
    
    @property
    def coupon_amount(self) -> float:
        """
        Coupon payment amount per period.
        
        = Face × coupon_rate / frequency
        """
        return float(self.face_value) * float(self.coupon_rate) / self.coupon_frequency


# =============================================================================
# FIXED RATE BOND - MARKET DATA INSTRUMENT
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondFixedRate:
    """
    Fixed rate coupon bond with market data lookup.
    
    Parameters
    ----------
    face_value : float
        Face/par value paid at maturity.
    coupon_rate : float
        Annual coupon rate (e.g., 0.05 for 5%).
    maturity : float
        Time to maturity in years.
    frequency : int
        Coupon frequency per year (1=annual, 2=semi-annual, 4=quarterly).
    day_count : DayCountConvention
        Day count convention for accrual calculation.
    settlement_days : int
        Days to settlement (T+n convention).
    curve_id : MarketId
        Market identifier for the discount curve.
    
    Examples
    --------
    10-year semi-annual US Treasury:
        >>> bond = IrBondFixedRate(
        ...     face_value=100.0,
        ...     coupon_rate=0.04,           # 4% annual coupon
        ...     maturity=10.0,
        ...     frequency=2,                # Semi-annual
        ...     day_count="ACT/365",
        ...     curve_id=MarketId("IR", "CURVE", "USD.GOVT"),
        ... )
    
    Notes
    -----
    The pricer will generate coupon schedule and compute accrued interest.
    """
    face_value: float
    coupon_rate: float
    maturity: float
    frequency: int = 2              # Semi-annual is most common
    day_count: DayCountConvention = "ACT/365"
    settlement_days: int = 1        # T+1 settlement
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate fixed rate bond inputs."""
        if float(self.face_value) <= 0.0:
            raise ValueError("face_value must be > 0.")
        if float(self.coupon_rate) < 0.0:
            raise ValueError("coupon_rate must be >= 0.")
        if float(self.maturity) <= 0.0:
            raise ValueError("maturity must be > 0.")
        if self.frequency not in (1, 2, 4, 12):
            raise ValueError("frequency must be 1, 2, 4, or 12.")
        if self.day_count not in ("ACT/360", "ACT/365", "30/360"):
            raise ValueError(f"Invalid day_count: {self.day_count}")
        if self.settlement_days < 0:
            raise ValueError("settlement_days must be >= 0.")
    
    @property
    def coupon_period(self) -> float:
        """Time between coupon payments in years."""
        return 1.0 / self.frequency
    
    @property
    def coupon_amount(self) -> float:
        """Coupon payment amount per period."""
        return float(self.face_value) * float(self.coupon_rate) / self.frequency
    
    @property
    def n_remaining_coupons(self) -> int:
        """Estimated number of remaining coupon payments."""
        import math
        return max(1, math.ceil(float(self.maturity) * self.frequency))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Zero coupon bonds
    "IrBondZeroCoupon",
    "IrBondZeroCouponSimple",
    # Fixed rate bonds
    "IrBondFixedRate",
    "IrBondFixedRateSimple",
]
