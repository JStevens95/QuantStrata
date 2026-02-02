# src/pricers/ir/bond.py
"""
Linear Bond Pricers (Zero Coupon and Fixed Rate).

Mathematical Framework
----------------------

Zero Coupon Bond Pricing:
    PV = Face × DF(T)
    
    Where:
    - Face = face value (par)
    - DF(T) = discount factor to maturity
    - T = time to maturity

Fixed Rate Bond Pricing:
    PV = Σ(C_i × DF(T_i)) + Face × DF(T_n)
    
    Where:
    - C_i = coupon payment = Face × coupon_rate / frequency
    - DF(T_i) = discount factor to coupon date i
    - T_n = maturity date

Clean vs Dirty Price
--------------------
- Dirty Price = PV (full price)
- Clean Price = PV - Accrued Interest
- Accrued Interest = Face × coupon_rate × (days since last coupon / days in period)

Risk Measures (Greeks)
----------------------
- DV01: Change in PV for 1bp parallel shift in yields
- Modified Duration: -(1/PV) × dPV/dy
- Macaulay Duration: Weighted average time to cash flows
- Convexity: (1/PV) × d²PV/dy²

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

from src.marketdata.core.market import Market
from src.instruments.ir.linear.bond import (
    IrBondZeroCoupon, IrBondZeroCouponSimple,
    IrBondFixedRate, IrBondFixedRateSimple,
)

# Greek name type for linear bond instruments.
BondGreekName = Literal["dv01", "modified_duration", "macaulay_duration", "convexity"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _generate_coupon_schedule(
        maturity: float,
        frequency: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Generate coupon payment schedule.
    
    Parameters
    ----------
    maturity : float
        Time to maturity in years.
    frequency : int
        Number of coupons per year (1, 2, 4, 12).
    
    Returns
    -------
    Tuple[Tuple[float, ...], Tuple[float, ...]]
        (coupon_times, accrual_factors) for each payment.
    """
    period = 1.0 / frequency
    times = []
    accruals = []
    
    # Generate payment times from maturity backwards
    t = maturity
    while t > 1e-10:  # Small tolerance for floating point
        times.insert(0, t)
        accruals.insert(0, period)
        t -= period
    
    # Handle stub period if first coupon is not a full period
    if times and times[0] < period - 1e-10:
        accruals[0] = times[0]  # Stub accrual
    
    return tuple(times), tuple(accruals)


def _compute_accrued_interest(
        face_value: float,
        coupon_rate: float,
        frequency: int,
        time_since_last_coupon: float,
) -> float:
    """
    Compute accrued interest since last coupon.
    
    Parameters
    ----------
    face_value : float
        Bond face value.
    coupon_rate : float
        Annual coupon rate.
    frequency : int
        Coupons per year.
    time_since_last_coupon : float
        Time in years since last coupon.
    
    Returns
    -------
    float
        Accrued interest amount.
    """
    period = 1.0 / frequency
    accrual_fraction = time_since_last_coupon / period
    annual_coupon = face_value * coupon_rate
    return annual_coupon * accrual_fraction / frequency


# =============================================================================
# ZERO COUPON BOND PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondZeroCouponPricerSimple:
    """
    Pricer for zero coupon bond with direct parameters.
    """
    
    def price(self, trade: IrBondZeroCouponSimple) -> float:
        """
        Price a zero coupon bond.
        
        Parameters
        ----------
        trade : IrBondZeroCouponSimple
            Zero coupon bond with direct parameters.
        
        Returns
        -------
        float
            Present value of the bond.
        
        Notes
        -----
        PV = Face × DF(T)
        """
        return float(trade.face_value) * float(trade.discount_factor)
    
    def greeks(self, trade: IrBondZeroCouponSimple) -> Dict[BondGreekName, float]:
        """
        Compute risk measures for a zero coupon bond.
        
        Parameters
        ----------
        trade : IrBondZeroCouponSimple
            Zero coupon bond with direct parameters.
        
        Returns
        -------
        dict
            Risk measures: dv01, modified_duration, macaulay_duration, convexity.
        
        Notes
        -----
        For a zero coupon bond:
        - Macaulay Duration = T (maturity)
        - Modified Duration = T / (1 + y) ≈ T (for small yields)
        - DV01 = PV × Modified Duration × 0.0001
        - Convexity = T² (approximately)
        """
        face = float(trade.face_value)
        df = float(trade.discount_factor)
        T = float(trade.maturity)
        
        pv = face * df
        
        # For zero coupon: Macaulay duration = maturity
        macaulay_duration = T
        
        # Implied yield (continuously compounded)
        if T > 0:
            y = -math.log(df) / T
            # Modified duration = Macaulay / (1 + y/freq)
            # For continuous compounding, modified_duration ≈ macaulay_duration
            modified_duration = macaulay_duration
        else:
            y = 0.0
            modified_duration = 0.0
        
        # DV01 = PV × Modified Duration × 0.0001
        dv01 = pv * modified_duration * 0.0001
        
        # Convexity = T² for zero coupon (approximately)
        convexity = T * T if T > 0 else 0.0
        
        return {
            "dv01": dv01,
            "modified_duration": modified_duration,
            "macaulay_duration": macaulay_duration,
            "convexity": convexity,
        }


@dataclass(frozen=True, slots=True)
class IrBondZeroCouponPricer:
    """
    Pricer for zero coupon bond with market data lookup.
    """
    
    def price(self, trade: IrBondZeroCoupon, market: Market) -> float:
        """
        Price a zero coupon bond using market data.
        
        Parameters
        ----------
        trade : IrBondZeroCoupon
            Zero coupon bond instrument.
        market : Market
            Market snapshot.
        
        Returns
        -------
        float
            Present value of the bond.
        """
        simple = self._to_simple(trade, market)
        return IrBondZeroCouponPricerSimple().price(simple)
    
    def greeks(self, trade: IrBondZeroCoupon, market: Market) -> Dict[BondGreekName, float]:
        """Compute risk measures for a zero coupon bond."""
        simple = self._to_simple(trade, market)
        return IrBondZeroCouponPricerSimple().greeks(simple)
    
    def _to_simple(
            self,
            trade: IrBondZeroCoupon,
            market: Market,
    ) -> IrBondZeroCouponSimple:
        """Convert market-based bond to simple bond."""
        curve = market.curve(trade.curve_id)
        df = float(curve.df(trade.maturity))
        
        return IrBondZeroCouponSimple(
            face_value=trade.face_value,
            maturity=trade.maturity,
            discount_factor=df,
        )


# =============================================================================
# FIXED RATE BOND PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondFixedRatePricerSimple:
    """
    Pricer for fixed rate coupon bond with direct parameters.
    """
    
    def price(self, trade: IrBondFixedRateSimple) -> float:
        """
        Price a fixed rate coupon bond (dirty price).
        
        Parameters
        ----------
        trade : IrBondFixedRateSimple
            Fixed rate bond with direct parameters.
        
        Returns
        -------
        float
            Present value of the bond (dirty price).
        
        Notes
        -----
        PV = Σ(C_i × DF_i) + Face × DF_n
        
        This returns the dirty (full) price. For clean price:
        Clean = Dirty - Accrued Interest
        """
        face = float(trade.face_value)
        coupon_rate = float(trade.coupon_rate)
        coupon_per_period = trade.coupon_amount
        
        pv = 0.0
        
        # Sum discounted coupon payments
        for i, (t, df) in enumerate(zip(trade.coupon_times, trade.coupon_dfs)):
            pv += coupon_per_period * float(df)
        
        # Add discounted face value at maturity (last coupon date)
        pv += face * float(trade.coupon_dfs[-1])
        
        return pv
    
    def clean_price(self, trade: IrBondFixedRateSimple) -> float:
        """
        Clean price (excluding accrued interest).
        
        Returns
        -------
        float
            Clean price = Dirty price - Accrued interest.
        """
        return self.price(trade) - float(trade.accrued_interest)
    
    def greeks(self, trade: IrBondFixedRateSimple) -> Dict[BondGreekName, float]:
        """
        Compute risk measures for a fixed rate bond.
        
        Parameters
        ----------
        trade : IrBondFixedRateSimple
            Fixed rate bond with direct parameters.
        
        Returns
        -------
        dict
            Risk measures: dv01, modified_duration, macaulay_duration, convexity.
        """
        face = float(trade.face_value)
        coupon_per_period = trade.coupon_amount
        
        pv = self.price(trade)
        
        # Calculate Macaulay Duration
        # D_mac = Σ(t_i × CF_i × DF_i) / PV
        weighted_time = 0.0
        weighted_time_squared = 0.0
        
        for i, (t, df) in enumerate(zip(trade.coupon_times, trade.coupon_dfs)):
            t_float = float(t)
            df_float = float(df)
            cf = coupon_per_period
            
            # Last payment includes face value
            if i == len(trade.coupon_times) - 1:
                cf += face
            
            pv_cf = cf * df_float
            weighted_time += t_float * pv_cf
            weighted_time_squared += t_float * t_float * pv_cf
        
        # Macaulay Duration
        macaulay_duration = weighted_time / pv if pv > 0 else 0.0
        
        # Modified Duration ≈ Macaulay Duration (for continuous compounding)
        # For discrete compounding: D_mod = D_mac / (1 + y/freq)
        # We approximate with D_mod ≈ D_mac
        modified_duration = macaulay_duration
        
        # DV01 = PV × Modified Duration × 0.0001
        dv01 = pv * modified_duration * 0.0001
        
        # Convexity = Σ(t_i² × CF_i × DF_i) / PV
        convexity = weighted_time_squared / pv if pv > 0 else 0.0
        
        return {
            "dv01": dv01,
            "modified_duration": modified_duration,
            "macaulay_duration": macaulay_duration,
            "convexity": convexity,
        }
    
    def yield_to_maturity(
            self,
            trade: IrBondFixedRateSimple,
            market_price: float,
            *,
            tol: float = 1e-8,
            max_iter: int = 100,
    ) -> float:
        """
        Compute yield to maturity (YTM) given market price.
        
        Uses Newton-Raphson iteration to find the yield that
        makes PV equal to market price.
        
        Parameters
        ----------
        trade : IrBondFixedRateSimple
            Fixed rate bond.
        market_price : float
            Observed market price (dirty price).
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.
        
        Returns
        -------
        float
            Yield to maturity (continuously compounded).
        """
        face = float(trade.face_value)
        coupon_per_period = trade.coupon_amount
        times = [float(t) for t in trade.coupon_times]
        
        # Initial guess from simple yield approximation
        maturity = times[-1]
        annual_coupon = coupon_per_period * trade.coupon_frequency
        y = (annual_coupon + (face - market_price) / maturity) / ((face + market_price) / 2)
        
        for _ in range(max_iter):
            # Calculate PV at current yield
            pv = 0.0
            dpv_dy = 0.0
            
            for i, t in enumerate(times):
                cf = coupon_per_period
                if i == len(times) - 1:
                    cf += face
                
                df = math.exp(-y * t)
                pv += cf * df
                dpv_dy -= t * cf * df
            
            # Newton-Raphson update
            error = pv - market_price
            if abs(error) < tol:
                return y
            
            if abs(dpv_dy) < 1e-12:
                break
            
            y -= error / dpv_dy
        
        return y


@dataclass(frozen=True, slots=True)
class IrBondFixedRatePricer:
    """
    Pricer for fixed rate bond with market data lookup.
    """
    
    def price(self, trade: IrBondFixedRate, market: Market) -> float:
        """
        Price a fixed rate bond using market data.
        
        Parameters
        ----------
        trade : IrBondFixedRate
            Fixed rate bond instrument.
        market : Market
            Market snapshot.
        
        Returns
        -------
        float
            Present value of the bond (dirty price).
        """
        simple = self._to_simple(trade, market)
        return IrBondFixedRatePricerSimple().price(simple)
    
    def clean_price(self, trade: IrBondFixedRate, market: Market) -> float:
        """Clean price (excluding accrued interest)."""
        simple = self._to_simple(trade, market)
        return IrBondFixedRatePricerSimple().clean_price(simple)
    
    def greeks(self, trade: IrBondFixedRate, market: Market) -> Dict[BondGreekName, float]:
        """Compute risk measures for a fixed rate bond."""
        simple = self._to_simple(trade, market)
        return IrBondFixedRatePricerSimple().greeks(simple)
    
    def _to_simple(
            self,
            trade: IrBondFixedRate,
            market: Market,
    ) -> IrBondFixedRateSimple:
        """Convert market-based bond to simple bond."""
        curve = market.curve(trade.curve_id)
        
        # Generate coupon schedule
        coupon_times, accrual_factors = _generate_coupon_schedule(
            trade.maturity,
            trade.frequency,
        )
        
        # Get discount factors for each coupon date
        coupon_dfs = tuple(float(curve.df(t)) for t in coupon_times)
        
        # Compute accrued interest
        # Time since last coupon = period - time to next coupon
        period = 1.0 / trade.frequency
        if coupon_times:
            time_to_next = coupon_times[0]
            time_since_last = max(0.0, period - time_to_next)
        else:
            time_since_last = 0.0
        
        accrued = _compute_accrued_interest(
            trade.face_value,
            trade.coupon_rate,
            trade.frequency,
            time_since_last,
        )
        
        return IrBondFixedRateSimple(
            face_value=trade.face_value,
            coupon_rate=trade.coupon_rate,
            coupon_times=coupon_times,
            coupon_dfs=coupon_dfs,
            accrued_interest=accrued,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Zero coupon pricers
    "IrBondZeroCouponPricer",
    "IrBondZeroCouponPricerSimple",
    # Fixed rate pricers
    "IrBondFixedRatePricer",
    "IrBondFixedRatePricerSimple",
]
