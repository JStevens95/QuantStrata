# src/pricers/ir/european_hw.py
"""
Interest Rate Hull-White Analytic Pricers.

Pricers for zero coupon bonds, bond options, caps/floors, and swaptions
using the Hull-White one-factor short rate model.

Mathematical Framework
----------------------
The Hull-White model specifies short rate dynamics:

    dr(t) = [θ(t) - a·r(t)] dt + σ dW(t)

This leads to affine bond prices:

    P(t,T) = A(t,T) · exp(-B(t,T) · r(t))

Key Analytic Results
--------------------
**Zero Coupon Bond Price:**
    P(t,T) = A(t,T) × exp(-B(t,T) × r(t))
    
    B(t,T) = (1 - exp(-a(T-t))) / a
    A(t,T) = P(0,T)/P(0,t) × exp(B(t,T)×f(0,t) - σ²/(4a)×B(t,T)²×(1-exp(-2at)))

**European Bond Option:**
    Call = P(0,T_bond)×N(h) - K×P(0,T_opt)×N(h - σ_p)
    Put  = K×P(0,T_opt)×N(-h + σ_p) - P(0,T_bond)×N(-h)
    
    σ_p = σ × √((1-exp(-2a×T_opt))/(2a)) × B(T_opt, T_bond)
    h   = (1/σ_p) × ln(P(0,T_bond)/(K×P(0,T_opt))) + σ_p/2

**Caplet/Floorlet:**
    Caplet = (1 + τK) × Put on ZC bond with strike 1/(1+τK)
    Floorlet = (1 + τK) × Call on ZC bond with strike 1/(1+τK)

Greeks
------
- delta: Sensitivity to short rate changes
- gamma: Convexity to short rate
- vega: Sensitivity to HW vol σ
- theta: Time decay
- rho: Sensitivity to initial curve level

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, List, Callable
from scipy.stats import norm

from src.marketdata.core.market import Market
from src.instruments.ir.linear.bond import (
    IrBondZeroCoupon,
    IrBondZeroCouponSimple,
    IrBondFixedRate,
    IrBondFixedRateSimple,
)
from src.instruments.ir.options.bond import (
    IrBondEuropeanOption,
    IrBondEuropeanOptionSimple,
)
from src.instruments.ir.options.capfloor import (
    IrCapletEuropeanOption,
    IrCapletEuropeanOptionSimple,
    IrFloorletEuropeanOption,
    IrFloorletEuropeanOptionSimple,
    IrCapEuropeanOption,
    IrCapEuropeanOptionSimple,
    IrFloorEuropeanOption,
    IrFloorEuropeanOptionSimple,
    compute_accrual_factor,
)
from src.instruments.ir.options.swaption import (
    IrSwaptionEuropeanOption,
    IrSwaptionEuropeanOptionSimple,
)

from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    hw_b_factor,
    hw_zc_bond_price,
    hw_zc_bond_option_price,
    hw_caplet_price,
    hw_floorlet_price,
    hw_swaption_price_jamshidian,
)


# Greek name type.
GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _rate_from_df(*, df: float, t: float) -> float:
    """Convert discount factor to continuously-compounded rate."""
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


def _df_from_rate(*, r: float, t: float) -> float:
    """Convert continuously-compounded rate to discount factor."""
    return math.exp(-r * t)


def _forward_rate_from_dfs(
    *,
    df_start: float,
    df_end: float,
    accrual_factor: float,
) -> float:
    """Compute simple forward rate from discount factors."""
    return (df_start / df_end - 1.0) / accrual_factor


def _instantaneous_forward_rate(df_t: float, df_t_dt: float, dt: float = 0.001) -> float:
    """
    Approximate instantaneous forward rate f(0,t) from discount factors.
    
    f(0,t) ≈ -d(ln P(0,t))/dt ≈ (ln(P(0,t)) - ln(P(0,t+dt))) / dt
    """
    if df_t <= 0 or df_t_dt <= 0:
        return 0.0
    return -(math.log(df_t_dt) - math.log(df_t)) / dt


# =============================================================================
# HULL-WHITE ZERO COUPON BOND PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondZeroCouponHWPricerSimple:
    """
    Hull-White pricer for zero coupon bonds with direct parameters.
    
    Uses the closed-form affine formula:
        P(t,T) = A(t,T) × exp(-B(t,T) × r(t))
    
    Parameters
    ----------
    params : HullWhiteParameters
        Hull-White model parameters (a, sigma, r0, theta).
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrBondZeroCouponSimple) -> float:
        """
        Price a zero coupon bond using Hull-White.
        
        For t=0 (today), P(0,T) = exp(-r0 × T) under flat curve assumption.
        
        Parameters
        ----------
        trade : IrBondZeroCouponSimple
            Zero coupon bond instrument.
        
        Returns
        -------
        float
            Present value (dirty price) of the bond.
        """
        T = float(trade.maturity)
        face = float(trade.face_value)
        
        if T <= 0.0:
            return face  # Matured
        
        # Under flat curve assumption at t=0:
        # P(0,T) = exp(-r0 × T)
        r0 = self.params.r0
        df = math.exp(-r0 * T)
        
        return face * df
    
    def greeks(self, trade: IrBondZeroCouponSimple) -> Dict[str, float]:
        """
        Compute Greeks for zero coupon bond.
        
        Returns
        -------
        dict
            - delta: dP/dr (sensitivity to parallel rate shift)
            - dv01: dollar value of 1bp shift
            - modified_duration: -dP/dr / P
            - macaulay_duration: T (time to maturity)
            - convexity: d²P/dr² / P
        """
        T = float(trade.maturity)
        face = float(trade.face_value)
        
        if T <= 0.0:
            return {
                "delta": 0.0,
                "dv01": 0.0,
                "modified_duration": 0.0,
                "macaulay_duration": 0.0,
                "convexity": 0.0,
            }
        
        r0 = self.params.r0
        df = math.exp(-r0 * T)
        price = face * df
        
        # dP/dr = -T × P
        delta = -T * price
        
        # DV01 = -dP/dr × 0.0001 = T × P × 0.0001
        dv01 = T * price * 0.0001
        
        # Modified duration = T (for continuously compounded)
        mod_dur = T
        
        # Macaulay duration = T for zero coupon
        mac_dur = T
        
        # Convexity = d²P/dr² / P = T²
        convexity = T ** 2
        
        return {
            "delta": delta,
            "dv01": dv01,
            "modified_duration": mod_dur,
            "macaulay_duration": mac_dur,
            "convexity": convexity,
        }


@dataclass(frozen=True, slots=True)
class IrBondZeroCouponHWPricer:
    """
    Hull-White pricer for zero coupon bonds with market data.
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrBondZeroCoupon, market: Market) -> float:
        """Price a zero coupon bond using Hull-White with market data."""
        curve = market.curve(trade.curve_id)
        df = float(curve.df(trade.maturity))
        return float(trade.face_value) * df
    
    def greeks(self, trade: IrBondZeroCoupon, market: Market) -> Dict[str, float]:
        """Compute Greeks for zero coupon bond with market data."""
        curve = market.curve(trade.curve_id)
        T = float(trade.maturity)
        face = float(trade.face_value)
        df = float(curve.df(T))
        price = face * df
        
        if T <= 0.0:
            return {
                "delta": 0.0,
                "dv01": 0.0,
                "modified_duration": 0.0,
                "macaulay_duration": 0.0,
                "convexity": 0.0,
            }
        
        return {
            "delta": -T * price,
            "dv01": T * price * 0.0001,
            "modified_duration": T,
            "macaulay_duration": T,
            "convexity": T ** 2,
        }


# =============================================================================
# HULL-WHITE BOND OPTION PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondEuropeanOptionHWPricerSimple:
    """
    Hull-White pricer for European bond options with direct parameters.
    
    Uses the closed-form Hull-White formula for options on zero-coupon bonds.
    
    Parameters
    ----------
    params : HullWhiteParameters
        Hull-White model parameters.
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrBondEuropeanOptionSimple) -> float:
        """
        Price a bond option using Hull-White.
        
        Parameters
        ----------
        trade : IrBondEuropeanOptionSimple
            Bond option with direct parameters.
        
        Returns
        -------
        float
            Present value of the bond option.
        
        Notes
        -----
        For Hull-White pricing, we need to extract implied P(0,S) and P(0,T)
        from the forward bond price and discount factor.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_opt = float(trade.expiry)
        F = float(trade.forward_bond_price)
        df_opt = float(trade.discount_factor)
        opt_type = trade.option_type
        
        if T_opt <= 0.0:
            # Expired - intrinsic value
            if opt_type == "call":
                return N * df_opt * max(F - K, 0.0)
            return N * df_opt * max(K - F, 0.0)
        
        # P(0, T_option) = df_opt
        P_0_S = df_opt
        
        # Forward = P(0,T) / P(0,S) => P(0,T) = F × P(0,S)
        # This assumes the forward is already adjusted for any coupons
        P_0_T = F * P_0_S
        
        a = self.params.a
        sigma = self.params.sigma
        
        # Use Hull-White bond option formula
        # We need T_bond > T_option for this to work
        # Infer T_bond from P(0,T): P(0,T) = exp(-r0 × T)
        # T = -ln(P_0_T) / r0
        r0 = self.params.r0
        if r0 != 0:
            T_bond = -math.log(P_0_T) / r0
        else:
            T_bond = T_opt + 1.0  # Default assumption
        
        # Ensure T_bond > T_opt
        if T_bond <= T_opt:
            T_bond = T_opt + 0.5
        
        price = hw_zc_bond_option_price(
            K=K,
            T_option=T_opt,
            T_bond=T_bond,
            a=a,
            sigma=sigma,
            P_0_S=P_0_S,
            P_0_T=P_0_T,
            option_type=opt_type,
        )
        
        return N * price
    
    def greeks(self, trade: IrBondEuropeanOptionSimple) -> Dict[GreekName, float]:
        """
        Compute Greeks for a bond option under Hull-White.
        
        Returns
        -------
        dict
            Greeks computed via finite difference approximation.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_opt = float(trade.expiry)
        F = float(trade.forward_bond_price)
        df_opt = float(trade.discount_factor)
        opt_type = trade.option_type
        
        if T_opt <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        a = self.params.a
        sigma = self.params.sigma
        r0 = self.params.r0
        
        P_0_S = df_opt
        P_0_T = F * P_0_S
        
        if r0 != 0:
            T_bond = -math.log(P_0_T) / r0
        else:
            T_bond = T_opt + 1.0
        
        if T_bond <= T_opt:
            T_bond = T_opt + 0.5
        
        # Base price
        base_price = hw_zc_bond_option_price(
            K=K, T_option=T_opt, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=P_0_T, option_type=opt_type,
        )
        
        # Delta: dP/dF (sensitivity to forward bond price)
        dF = F * 0.001  # 0.1% bump
        price_up = hw_zc_bond_option_price(
            K=K, T_option=T_opt, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=(F + dF) * P_0_S, option_type=opt_type,
        )
        price_dn = hw_zc_bond_option_price(
            K=K, T_option=T_opt, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=(F - dF) * P_0_S, option_type=opt_type,
        )
        delta = N * (price_up - price_dn) / (2 * dF)
        
        # Gamma: d²P/dF²
        gamma = N * (price_up - 2 * base_price + price_dn) / (dF ** 2)
        
        # Vega: dP/dσ (sensitivity to HW sigma)
        d_sigma = 0.0001  # 1bp bump
        price_sigma_up = hw_zc_bond_option_price(
            K=K, T_option=T_opt, T_bond=T_bond, a=a, sigma=sigma + d_sigma,
            P_0_S=P_0_S, P_0_T=P_0_T, option_type=opt_type,
        )
        vega = N * (price_sigma_up - base_price) / d_sigma
        
        # Theta: dP/dt (time decay per year)
        dt = 1.0 / 365.0
        if T_opt > dt:
            price_theta = hw_zc_bond_option_price(
                K=K, T_option=T_opt - dt, T_bond=T_bond, a=a, sigma=sigma,
                P_0_S=P_0_S * math.exp(r0 * dt), P_0_T=P_0_T, option_type=opt_type,
            )
            theta = N * (price_theta - base_price) / dt
        else:
            theta = 0.0
        
        # Rho: dP/dr (sensitivity to rate level)
        dr = 0.0001  # 1bp
        r_up = r0 + dr
        P_0_S_up = math.exp(-r_up * T_opt)
        P_0_T_up = math.exp(-r_up * T_bond) if T_bond > 0 else P_0_T
        price_r_up = hw_zc_bond_option_price(
            K=K, T_option=T_opt, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S_up, P_0_T=P_0_T_up, option_type=opt_type,
        )
        rho = N * (price_r_up - base_price) / dr
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }


@dataclass(frozen=True, slots=True)
class IrBondEuropeanOptionHWPricer:
    """
    Hull-White pricer for European bond options with market data lookup.
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrBondEuropeanOption, market: Market) -> float:
        """Price a bond option using Hull-White with market data."""
        simple = self._to_simple(trade, market)
        return IrBondEuropeanOptionHWPricerSimple(params=self.params).price(simple)
    
    def greeks(self, trade: IrBondEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """Compute Greeks for a bond option with market data."""
        simple = self._to_simple(trade, market)
        return IrBondEuropeanOptionHWPricerSimple(params=self.params).greeks(simple)
    
    def _to_simple(self, trade: IrBondEuropeanOption, market: Market) -> IrBondEuropeanOptionSimple:
        """Convert market-based bond option to simple bond option."""
        curve = market.curve(trade.curve_id)
        
        df_expiry = float(curve.df(trade.expiry))
        spot_price = self._compute_spot_bond_price(trade, curve)
        coupon_pv = self._compute_coupon_pv_during_option(trade, curve)
        forward_price = (spot_price - coupon_pv) / df_expiry
        
        # For HW, we don't need vol surface - sigma is in params
        # But for consistency, we might want to use vol for vega scaling
        # Here we use a default or ignore vol_id since HW uses its own sigma
        
        return IrBondEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            expiry=trade.expiry,
            forward_bond_price=forward_price,
            vol=self.params.sigma,  # Use HW sigma
            discount_factor=df_expiry,
            option_type=trade.option_type,
        )
    
    def _compute_spot_bond_price(self, trade: IrBondEuropeanOption, curve) -> float:
        """Compute spot price of underlying bond."""
        face = 100.0
        
        if trade.is_zero_coupon:
            return face * float(curve.df(trade.underlying_maturity))
        
        period = 1.0 / trade.underlying_coupon_frequency
        coupon = face * trade.underlying_coupon_rate / trade.underlying_coupon_frequency
        
        pv = 0.0
        t = period
        while t <= trade.underlying_maturity + 1e-9:
            pv += coupon * float(curve.df(t))
            t += period
        
        pv += face * float(curve.df(trade.underlying_maturity))
        return pv
    
    def _compute_coupon_pv_during_option(self, trade: IrBondEuropeanOption, curve) -> float:
        """Compute PV of coupons during option life."""
        if trade.is_zero_coupon:
            return 0.0
        
        face = 100.0
        period = 1.0 / trade.underlying_coupon_frequency
        coupon = face * trade.underlying_coupon_rate / trade.underlying_coupon_frequency
        
        pv = 0.0
        t = period
        while t <= trade.expiry + 1e-9:
            if t > 0:
                pv += coupon * float(curve.df(t))
            t += period
        
        return pv


# =============================================================================
# HULL-WHITE CAPLET/FLOORLET PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrCapletEuropeanOptionHWPricerSimple:
    """
    Hull-White pricer for caplets with direct parameters.
    
    A caplet is equivalent to (1 + τK) puts on a ZC bond with strike 1/(1+τK).
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrCapletEuropeanOptionSimple) -> float:
        """
        Price a caplet using Hull-White.
        
        Parameters
        ----------
        trade : IrCapletEuropeanOptionSimple
            Caplet with direct parameters.
        
        Returns
        -------
        float
            Present value of the caplet.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        T_pay = float(trade.payment_time)
        tau = float(trade.accrual_factor)
        
        if T_fix <= 0.0:
            # Expired - use intrinsic.
            F = float(trade.forward_rate)
            df_pay = float(trade.discount_factor)
            return N * tau * df_pay * max(F - K, 0.0)
        
        # Infer P(0, T_fix) and P(0, T_pay) from HW params using flat rate approximation.
        # This is consistent with Hull-White analytic pricing under constant θ.
        r0 = self.params.r0
        P_0_fix = math.exp(-r0 * T_fix)
        P_0_pay = math.exp(-r0 * T_pay)
        
        a = self.params.a
        sigma = self.params.sigma
        
        price = hw_caplet_price(
            K=K,
            T_reset=T_fix,
            T_pay=T_pay,
            tau=tau,
            a=a,
            sigma=sigma,
            P_0_reset=P_0_fix,
            P_0_pay=P_0_pay,
            notional=N,
        )
        
        return price
    
    def greeks(self, trade: IrCapletEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks for a caplet using finite differences."""
        T_fix = float(trade.fixing_time)
        
        if T_fix <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        # Base price
        base_price = self.price(trade)
        N = float(trade.notional)
        K = float(trade.strike)
        tau = float(trade.accrual_factor)
        T_pay = float(trade.payment_time)
        
        a = self.params.a
        sigma = self.params.sigma
        r0 = self.params.r0
        
        P_0_fix = math.exp(-r0 * T_fix)
        P_0_pay = math.exp(-r0 * T_pay)
        
        # Delta: dP/dF (via rate shift)
        dr = 0.0001
        r_up = r0 + dr
        price_up = hw_caplet_price(
            K=K, T_reset=T_fix, T_pay=T_pay, tau=tau, a=a, sigma=sigma,
            P_0_reset=math.exp(-r_up * T_fix), P_0_pay=math.exp(-r_up * T_pay), notional=N,
        )
        delta = (price_up - base_price) / dr
        
        # Gamma
        r_dn = r0 - dr
        price_dn = hw_caplet_price(
            K=K, T_reset=T_fix, T_pay=T_pay, tau=tau, a=a, sigma=sigma,
            P_0_reset=math.exp(-r_dn * T_fix), P_0_pay=math.exp(-r_dn * T_pay), notional=N,
        )
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        # Vega: dP/dσ
        d_sigma = 0.0001
        price_sigma_up = hw_caplet_price(
            K=K, T_reset=T_fix, T_pay=T_pay, tau=tau, a=a, sigma=sigma + d_sigma,
            P_0_reset=P_0_fix, P_0_pay=P_0_pay, notional=N,
        )
        vega = (price_sigma_up - base_price) / d_sigma
        
        # Theta
        dt = 1.0 / 365.0
        if T_fix > dt:
            price_theta = hw_caplet_price(
                K=K, T_reset=T_fix - dt, T_pay=T_pay - dt, tau=tau, a=a, sigma=sigma,
                P_0_reset=P_0_fix * math.exp(r0 * dt), P_0_pay=P_0_pay * math.exp(r0 * dt), notional=N,
            )
            theta = (price_theta - base_price) / dt
        else:
            theta = 0.0
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": delta,  # rho ≈ delta for caplets
        }


@dataclass(frozen=True, slots=True)
class IrFloorletEuropeanOptionHWPricerSimple:
    """
    Hull-White pricer for floorlets with direct parameters.
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrFloorletEuropeanOptionSimple) -> float:
        """Price a floorlet using Hull-White."""
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        T_pay = float(trade.payment_time)
        tau = float(trade.accrual_factor)
        
        if T_fix <= 0.0:
            F = float(trade.forward_rate)
            df = float(trade.discount_factor)
            return N * tau * df * max(K - F, 0.0)
        
        r0 = self.params.r0
        P_0_fix = math.exp(-r0 * T_fix)
        P_0_pay = math.exp(-r0 * T_pay)
        
        a = self.params.a
        sigma = self.params.sigma
        
        return hw_floorlet_price(
            K=K, T_reset=T_fix, T_pay=T_pay, tau=tau, a=a, sigma=sigma,
            P_0_reset=P_0_fix, P_0_pay=P_0_pay, notional=N,
        )
    
    def greeks(self, trade: IrFloorletEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks for a floorlet."""
        T_fix = float(trade.fixing_time)
        
        if T_fix <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        base_price = self.price(trade)
        N = float(trade.notional)
        K = float(trade.strike)
        tau = float(trade.accrual_factor)
        T_pay = float(trade.payment_time)
        
        a = self.params.a
        sigma = self.params.sigma
        r0 = self.params.r0
        
        dr = 0.0001
        r_up = r0 + dr
        price_up = hw_floorlet_price(
            K=K, T_reset=T_fix, T_pay=T_pay, tau=tau, a=a, sigma=sigma,
            P_0_reset=math.exp(-r_up * T_fix), P_0_pay=math.exp(-r_up * T_pay), notional=N,
        )
        r_dn = r0 - dr
        price_dn = hw_floorlet_price(
            K=K, T_reset=T_fix, T_pay=T_pay, tau=tau, a=a, sigma=sigma,
            P_0_reset=math.exp(-r_dn * T_fix), P_0_pay=math.exp(-r_dn * T_pay), notional=N,
        )
        delta = (price_up - base_price) / dr
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        d_sigma = 0.0001
        P_0_fix = math.exp(-r0 * T_fix)
        P_0_pay = math.exp(-r0 * T_pay)
        price_sigma_up = hw_floorlet_price(
            K=K, T_reset=T_fix, T_pay=T_pay, tau=tau, a=a, sigma=sigma + d_sigma,
            P_0_reset=P_0_fix, P_0_pay=P_0_pay, notional=N,
        )
        vega = (price_sigma_up - base_price) / d_sigma
        
        dt = 1.0 / 365.0
        if T_fix > dt:
            price_theta = hw_floorlet_price(
                K=K, T_reset=T_fix - dt, T_pay=T_pay - dt, tau=tau, a=a, sigma=sigma,
                P_0_reset=P_0_fix * math.exp(r0 * dt), P_0_pay=P_0_pay * math.exp(r0 * dt), notional=N,
            )
            theta = (price_theta - base_price) / dt
        else:
            theta = 0.0
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": delta,
        }


@dataclass(frozen=True, slots=True)
class IrCapEuropeanOptionHWPricerSimple:
    """
    Hull-White pricer for caps with direct parameters.
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrCapEuropeanOptionSimple) -> float:
        """Price a cap as sum of caplets."""
        caplet_pricer = IrCapletEuropeanOptionHWPricerSimple(params=self.params)
        total_pv = 0.0
        
        for caplet in trade.caplets:
            total_pv += caplet_pricer.price(caplet)
        
        return total_pv
    
    def greeks(self, trade: IrCapEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute aggregate Greeks for a cap."""
        caplet_pricer = IrCapletEuropeanOptionHWPricerSimple(params=self.params)
        total_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        for caplet in trade.caplets:
            caplet_greeks = caplet_pricer.greeks(caplet)
            for greek in total_greeks:
                total_greeks[greek] += caplet_greeks[greek]
        
        return total_greeks


@dataclass(frozen=True, slots=True)
class IrFloorEuropeanOptionHWPricerSimple:
    """
    Hull-White pricer for floors with direct parameters.
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrFloorEuropeanOptionSimple) -> float:
        """Price a floor as sum of floorlets."""
        floorlet_pricer = IrFloorletEuropeanOptionHWPricerSimple(params=self.params)
        total_pv = 0.0
        
        for floorlet in trade.floorlets:
            total_pv += floorlet_pricer.price(floorlet)
        
        return total_pv
    
    def greeks(self, trade: IrFloorEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute aggregate Greeks for a floor."""
        floorlet_pricer = IrFloorletEuropeanOptionHWPricerSimple(params=self.params)
        total_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        for floorlet in trade.floorlets:
            floorlet_greeks = floorlet_pricer.greeks(floorlet)
            for greek in total_greeks:
                total_greeks[greek] += floorlet_greeks[greek]
        
        return total_greeks


# =============================================================================
# HULL-WHITE SWAPTION PRICER
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrSwaptionEuropeanOptionHWPricerSimple:
    """
    Hull-White pricer for European swaptions using Jamshidian decomposition.
    
    Jamshidian's trick decomposes a swaption into a portfolio of bond options.
    """
    
    params: HullWhiteParameters
    
    def price(self, trade: IrSwaptionEuropeanOptionSimple) -> float:
        """
        Price a swaption using Hull-White with Jamshidian decomposition.
        
        Parameters
        ----------
        trade : IrSwaptionEuropeanOptionSimple
            Swaption with direct parameters.
        
        Returns
        -------
        float
            Present value of the swaption.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_opt = float(trade.option_expiry)
        swap_tenor = float(trade.swap_tenor)
        is_payer = (trade.swaption_type == "payer")
        
        if T_opt <= 0.0:
            # Expired - intrinsic value.
            swap_rate = float(trade.forward_swap_rate)
            annuity = float(trade.annuity)
            if is_payer:
                return N * max(swap_rate - K, 0.0) * annuity
            return N * max(K - swap_rate, 0.0) * annuity
        
        # Generate swap schedule (semi-annual payments assumed).
        # Payment times: from T_opt + 0.5 to T_opt + swap_tenor.
        n_payments = int(swap_tenor * 2)  # Semi-annual.
        tenors = np.array([T_opt + 0.5 * (i + 1) for i in range(n_payments)])
        dcfs = np.full(n_payments, 0.5)  # Semi-annual day count fractions.
        
        a = self.params.a
        sigma = self.params.sigma
        r0 = self.params.r0
        
        # P_0 function (flat curve approximation).
        def P_0(t: float) -> float:
            return math.exp(-r0 * t)
        
        return hw_swaption_price_jamshidian(
            K=K,
            T_option=T_opt,
            swap_tenors=tenors,
            swap_dcfs=dcfs,
            a=a,
            sigma=sigma,
            P_0=P_0,
            notional=N,
            is_payer=is_payer,
        )
    
    def greeks(self, trade: IrSwaptionEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks for a swaption using finite differences."""
        T_opt = float(trade.option_expiry)
        
        if T_opt <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        base_price = self.price(trade)
        
        # Finite difference for Greeks
        params = self.params
        N = float(trade.notional)
        K = float(trade.fixed_rate)
        tenors = np.array(trade.payment_times)
        dcfs = np.array(trade.day_count_fractions)
        is_payer = (trade.option_type == "payer")
        
        a = params.a
        sigma = params.sigma
        r0 = params.r0
        
        # Delta (rate sensitivity)
        dr = 0.0001
        def P_0_up(t: float) -> float:
            return math.exp(-(r0 + dr) * t)
        def P_0_dn(t: float) -> float:
            return math.exp(-(r0 - dr) * t)
        
        price_up = hw_swaption_price_jamshidian(
            K=K, T_option=T_opt, swap_tenors=tenors, swap_dcfs=dcfs,
            a=a, sigma=sigma, P_0=P_0_up, notional=N, is_payer=is_payer,
        )
        price_dn = hw_swaption_price_jamshidian(
            K=K, T_option=T_opt, swap_tenors=tenors, swap_dcfs=dcfs,
            a=a, sigma=sigma, P_0=P_0_dn, notional=N, is_payer=is_payer,
        )
        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        # Vega
        d_sigma = 0.0001
        def P_0_base(t: float) -> float:
            return math.exp(-r0 * t)
        
        price_sigma_up = hw_swaption_price_jamshidian(
            K=K, T_option=T_opt, swap_tenors=tenors, swap_dcfs=dcfs,
            a=a, sigma=sigma + d_sigma, P_0=P_0_base, notional=N, is_payer=is_payer,
        )
        vega = (price_sigma_up - base_price) / d_sigma
        
        # Theta (simplified)
        theta = -base_price / T_opt if T_opt > 0 else 0.0
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": delta,  # rho ≈ delta
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Zero coupon bond pricers
    "IrBondZeroCouponHWPricerSimple",
    "IrBondZeroCouponHWPricer",
    # Bond option pricers
    "IrBondEuropeanOptionHWPricerSimple",
    "IrBondEuropeanOptionHWPricer",
    # Caplet/floorlet pricers
    "IrCapletEuropeanOptionHWPricerSimple",
    "IrFloorletEuropeanOptionHWPricerSimple",
    "IrCapEuropeanOptionHWPricerSimple",
    "IrFloorEuropeanOptionHWPricerSimple",
    # Swaption pricer
    "IrSwaptionEuropeanOptionHWPricerSimple",
]
