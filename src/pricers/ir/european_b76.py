# src/pricers/ir/european_b76.py
"""
Interest Rate Black76 Pricers.

Pricers for caps, floors, caplets, and floorlets using the Black76 model.

Mathematical Framework
----------------------
Black76 for a caplet:
    Caplet PV = N × τ × DF(T_pay) × [F × N(d₁) - K × N(d₂)]

Where:
- N = notional
- τ = accrual factor (day count fraction)
- DF(T_pay) = discount factor to payment date
- F = forward rate for the period
- K = strike rate
- d₁ = [ln(F/K) + σ²T_fix/2] / (σ√T_fix)
- d₂ = d₁ - σ√T_fix
- T_fix = time to fixing date

A floorlet is priced using put formula:
    Floorlet PV = N × τ × DF(T_pay) × [K × N(-d₂) - F × N(-d₁)]

A cap/floor is the sum of its constituent caplets/floorlets.

Greeks
------
- delta: dPV/dF (sensitivity to forward rate)
- gamma: d²PV/dF² (convexity)
- vega: dPV/dσ (per 1.0 absolute vol)
- theta: dPV/dt (time decay per year)
- rho: dPV/dr (discount rate sensitivity)

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, List

from src.marketdata.core.market import Market
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

# Import pure functions from the Black76 model.
from src.models.analytic.black76.base import (
    vanilla_price,
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
    vanilla_rho,
)

# Greek name type.
GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.
    
    df = exp(-rT) => r = -ln(df)/T
    """
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


def _forward_rate_from_dfs(
    *,
    df_start: float,
    df_end: float,
    accrual_factor: float,
) -> float:
    """
    Compute simple forward rate from discount factors.
    
    F = (DF(T_start) / DF(T_end) - 1) / τ
    
    Parameters
    ----------
    df_start : float
        Discount factor DF(T_start).
    df_end : float
        Discount factor DF(T_end).
    accrual_factor : float
        Day count fraction τ.
    
    Returns
    -------
    float
        Simple forward rate F.
    """
    return (df_start / df_end - 1.0) / accrual_factor


# =============================================================================
# SIMPLE PRICERS (Direct Parameters)
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrCapletEuropeanOptionB76PricerSimple:
    """
    Black76 pricer for a single caplet with direct parameters.
    
    The caplet is priced as a call option on the forward rate.
    """
    
    def price(self, trade: IrCapletEuropeanOptionSimple) -> float:
        """
        Price a caplet using Black76.
        
        Parameters
        ----------
        trade : CapletSimple
            Caplet with direct parameters.
        
        Returns
        -------
        float
            Present value of the caplet.
        """
        # Extract parameters.
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        tau = float(trade.accrual_factor)
        F = float(trade.forward_rate)
        sigma = float(trade.vol)
        df = float(trade.discount_factor)
        
        # Handle expired caplet.
        if T_fix <= 0.0:
            # Payoff is already known.
            return N * tau * df * max(F - K, 0.0)
        
        # Black76 price for call on forward.
        unit_pv = vanilla_price(
            option_type="call",
            forward=F,
            strike=K,
            expiry=T_fix,
            discount_factor=df,
            vol=sigma,
        )
        
        # Scale by notional and accrual factor.
        return N * tau * unit_pv
    
    def greeks(self, trade: IrCapletEuropeanOptionSimple) -> Dict[GreekName, float]:
        """
        Compute Greeks for a caplet.
        
        Parameters
        ----------
        trade : CapletSimple
            Caplet with direct parameters.
        
        Returns
        -------
        dict
            Greeks: delta, gamma, vega, theta, rho.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        tau = float(trade.accrual_factor)
        F = float(trade.forward_rate)
        sigma = float(trade.vol)
        df = float(trade.discount_factor)
        
        if T_fix <= 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }
        
        # Compute discount rate for theta.
        r = _rate_from_df(df=df, t=T_fix)
        
        # Scale factor.
        scale = N * tau
        
        return {
            "delta": scale * vanilla_delta(
                option_type="call", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, vol=sigma,
            ),
            "gamma": scale * vanilla_gamma(
                option_type="call", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, vol=sigma,
            ),
            "vega": scale * vanilla_vega(
                option_type="call", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, vol=sigma,
            ),
            "theta": scale * vanilla_theta(
                option_type="call", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, discount_rate=r, vol=sigma,
            ),
            "rho": scale * vanilla_rho(
                option_type="call", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, vol=sigma,
            ),
        }


@dataclass(frozen=True, slots=True)
class IrFloorletEuropeanOptionB76PricerSimple:
    """
    Black76 pricer for a single floorlet with direct parameters.
    
    The floorlet is priced as a put option on the forward rate.
    """
    
    def price(self, trade: IrFloorletEuropeanOptionSimple) -> float:
        """
        Price a floorlet using Black76.
        
        Parameters
        ----------
        trade : FloorletSimple
            Floorlet with direct parameters.
        
        Returns
        -------
        float
            Present value of the floorlet.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        tau = float(trade.accrual_factor)
        F = float(trade.forward_rate)
        sigma = float(trade.vol)
        df = float(trade.discount_factor)
        
        if T_fix <= 0.0:
            return N * tau * df * max(K - F, 0.0)
        
        unit_pv = vanilla_price(
            option_type="put",
            forward=F,
            strike=K,
            expiry=T_fix,
            discount_factor=df,
            vol=sigma,
        )
        
        return N * tau * unit_pv
    
    def greeks(self, trade: IrFloorletEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks for a floorlet."""
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        tau = float(trade.accrual_factor)
        F = float(trade.forward_rate)
        sigma = float(trade.vol)
        df = float(trade.discount_factor)
        
        if T_fix <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        r = _rate_from_df(df=df, t=T_fix)
        scale = N * tau
        
        return {
            "delta": scale * vanilla_delta(
                option_type="put", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, vol=sigma,
            ),
            "gamma": scale * vanilla_gamma(
                option_type="put", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, vol=sigma,
            ),
            "vega": scale * vanilla_vega(
                option_type="put", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, vol=sigma,
            ),
            "theta": scale * vanilla_theta(
                option_type="put", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, discount_rate=r, vol=sigma,
            ),
            "rho": scale * vanilla_rho(
                option_type="put", forward=F, strike=K, expiry=T_fix,
                discount_factor=df, vol=sigma,
            ),
        }


@dataclass(frozen=True, slots=True)
class IrCapEuropeanOptionB76PricerSimple:
    """
    Black76 pricer for a cap with direct parameters.
    
    A cap is a portfolio of caplets. The PV is the sum of caplet PVs.
    """
    
    def price(self, trade: IrCapEuropeanOptionSimple) -> float:
        """
        Price a cap as sum of caplets.
        
        Parameters
        ----------
        trade : CapSimple
            Cap with pre-computed caplets.
        
        Returns
        -------
        float
            Present value of the cap.
        """
        caplet_pricer = IrCapletEuropeanOptionB76PricerSimple()
        total_pv = 0.0
        
        for caplet in trade.caplets:
            total_pv += caplet_pricer.price(caplet)
        
        return total_pv
    
    def greeks(self, trade: IrCapEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute aggregate Greeks for a cap."""
        caplet_pricer = IrCapletEuropeanOptionB76PricerSimple()
        
        total_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        for caplet in trade.caplets:
            caplet_greeks = caplet_pricer.greeks(caplet)
            for greek in total_greeks:
                total_greeks[greek] += caplet_greeks[greek]
        
        return total_greeks


@dataclass(frozen=True, slots=True)
class IrFloorEuropeanOptionB76PricerSimple:
    """
    Black76 pricer for a floor with direct parameters.
    
    A floor is a portfolio of floorlets. The PV is the sum of floorlet PVs.
    """
    
    def price(self, trade: IrFloorEuropeanOptionSimple) -> float:
        """Price a floor as sum of floorlets."""
        floorlet_pricer = IrFloorletEuropeanOptionB76PricerSimple()
        total_pv = 0.0
        
        for floorlet in trade.floorlets:
            total_pv += floorlet_pricer.price(floorlet)
        
        return total_pv
    
    def greeks(self, trade: IrFloorEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute aggregate Greeks for a floor."""
        floorlet_pricer = IrFloorletEuropeanOptionB76PricerSimple()
        
        total_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        for floorlet in trade.floorlets:
            floorlet_greeks = floorlet_pricer.greeks(floorlet)
            for greek in total_greeks:
                total_greeks[greek] += floorlet_greeks[greek]
        
        return total_greeks


# =============================================================================
# MARKET DATA PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrCapletEuropeanOptionB76Pricer:
    """
    Black76 pricer for a single caplet with market data lookup.
    """
    
    def price(self, trade: IrCapletEuropeanOption, market: Market) -> float:
        """
        Price a caplet using Black76 with market data.
        
        Parameters
        ----------
        trade : Caplet
            Caplet instrument.
        market : Market
            Market snapshot with curves and vol surface.
        
        Returns
        -------
        float
            Present value of the caplet.
        """
        # Get curve.
        curve = market.curve(trade.curve_id)
        
        # Get discount factors.
        df_start = float(curve.df(trade.fixing_time))
        df_end = float(curve.df(trade.payment_time))
        
        # Compute accrual factor.
        tau = compute_accrual_factor(
            trade.fixing_time,
            trade.payment_time,
            trade.day_count,
        )
        
        # Compute forward rate.
        F = _forward_rate_from_dfs(df_start=df_start, df_end=df_end, accrual_factor=tau)
        
        # Get volatility.
        vol_surface = market.vol_surface(trade.vol_id)
        sigma = float(vol_surface.vol(expiry=trade.fixing_time, strike=trade.strike))
        
        # Build simple caplet and price.
        simple = IrCapletEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            fixing_time=trade.fixing_time,
            payment_time=trade.payment_time,
            accrual_factor=tau,
            forward_rate=F,
            vol=sigma,
            discount_factor=df_end,
        )
        
        return IrCapletEuropeanOptionB76PricerSimple().price(simple)
    
    def greeks(self, trade: IrCapletEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """Compute Greeks for a caplet with market data."""
        curve = market.curve(trade.curve_id)
        
        df_start = float(curve.df(trade.fixing_time))
        df_end = float(curve.df(trade.payment_time))
        
        tau = compute_accrual_factor(
            trade.fixing_time,
            trade.payment_time,
            trade.day_count,
        )
        
        F = _forward_rate_from_dfs(df_start=df_start, df_end=df_end, accrual_factor=tau)
        
        vol_surface = market.vol_surface(trade.vol_id)
        sigma = float(vol_surface.vol(expiry=trade.fixing_time, strike=trade.strike))
        
        simple = IrCapletEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            fixing_time=trade.fixing_time,
            payment_time=trade.payment_time,
            accrual_factor=tau,
            forward_rate=F,
            vol=sigma,
            discount_factor=df_end,
        )
        
        return IrCapletEuropeanOptionB76PricerSimple().greeks(simple)


@dataclass(frozen=True, slots=True)
class IrFloorletEuropeanOptionB76Pricer:
    """
    Black76 pricer for a single floorlet with market data lookup.
    """
    
    def price(self, trade: IrFloorletEuropeanOption, market: Market) -> float:
        """Price a floorlet using Black76 with market data."""
        curve = market.curve(trade.curve_id)
        
        df_start = float(curve.df(trade.fixing_time))
        df_end = float(curve.df(trade.payment_time))
        
        tau = compute_accrual_factor(
            trade.fixing_time,
            trade.payment_time,
            trade.day_count,
        )
        
        F = _forward_rate_from_dfs(df_start=df_start, df_end=df_end, accrual_factor=tau)
        
        vol_surface = market.vol_surface(trade.vol_id)
        sigma = float(vol_surface.vol(expiry=trade.fixing_time, strike=trade.strike))
        
        simple = IrFloorletEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            fixing_time=trade.fixing_time,
            payment_time=trade.payment_time,
            accrual_factor=tau,
            forward_rate=F,
            vol=sigma,
            discount_factor=df_end,
        )
        
        return IrFloorletEuropeanOptionB76PricerSimple().price(simple)
    
    def greeks(self, trade: IrFloorletEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """Compute Greeks for a floorlet with market data."""
        curve = market.curve(trade.curve_id)
        
        df_start = float(curve.df(trade.fixing_time))
        df_end = float(curve.df(trade.payment_time))
        
        tau = compute_accrual_factor(
            trade.fixing_time,
            trade.payment_time,
            trade.day_count,
        )
        
        F = _forward_rate_from_dfs(df_start=df_start, df_end=df_end, accrual_factor=tau)
        
        vol_surface = market.vol_surface(trade.vol_id)
        sigma = float(vol_surface.vol(expiry=trade.fixing_time, strike=trade.strike))
        
        simple = IrFloorletEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            fixing_time=trade.fixing_time,
            payment_time=trade.payment_time,
            accrual_factor=tau,
            forward_rate=F,
            vol=sigma,
            discount_factor=df_end,
        )
        
        return IrFloorletEuropeanOptionB76PricerSimple().greeks(simple)


@dataclass(frozen=True, slots=True)
class IrCapEuropeanOptionB76Pricer:
    """
    Black76 pricer for a cap with market data lookup.
    
    The pricer automatically generates caplets for each reset period.
    """
    
    def price(self, trade: IrCapEuropeanOption, market: Market) -> float:
        """
        Price a cap using Black76 with market data.
        
        Parameters
        ----------
        trade : Cap
            Cap instrument (defines schedule).
        market : Market
            Market snapshot.
        
        Returns
        -------
        float
            Present value of the cap.
        """
        # Generate caplet schedule.
        caplets = self._generate_caplets(trade, market)
        
        # Build simple cap and price.
        simple_cap = IrCapEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            caplets=tuple(caplets),
        )
        
        return IrCapEuropeanOptionB76PricerSimple().price(simple_cap)
    
    def greeks(self, trade: IrCapEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """Compute aggregate Greeks for a cap."""
        caplets = self._generate_caplets(trade, market)
        
        simple_cap = IrCapEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            caplets=tuple(caplets),
        )
        
        return IrCapEuropeanOptionB76PricerSimple().greeks(simple_cap)
    
    def _generate_caplets(self, trade: IrCapEuropeanOption, market: Market) -> List[IrCapletEuropeanOptionSimple]:
        """Generate caplets for each reset period."""
        curve = market.curve(trade.curve_id)
        vol_surface = market.vol_surface(trade.vol_id)
        
        caplets = []
        
        # Generate fixing/payment dates.
        t = float(trade.start_time)
        freq = float(trade.frequency)
        end = float(trade.end_time)
        
        while t < end - 1e-9:  # Small tolerance for float comparison
            t_start = t
            t_end = min(t + freq, end)
            
            # Skip if already expired.
            if t_start <= 0.0:
                t = t_end
                continue
            
            # Get discount factors.
            df_start = float(curve.df(t_start))
            df_end = float(curve.df(t_end))
            
            # Compute accrual factor.
            tau = compute_accrual_factor(t_start, t_end, trade.day_count)
            
            # Compute forward rate.
            F = _forward_rate_from_dfs(df_start=df_start, df_end=df_end, accrual_factor=tau)
            
            # Get volatility.
            sigma = float(vol_surface.vol(expiry=t_start, strike=trade.strike))
            
            # Create caplet.
            caplet = IrCapletEuropeanOptionSimple(
                notional=trade.notional,
                strike=trade.strike,
                fixing_time=t_start,
                payment_time=t_end,
                accrual_factor=tau,
                forward_rate=F,
                vol=sigma,
                discount_factor=df_end,
            )
            caplets.append(caplet)
            
            t = t_end
        
        return caplets


@dataclass(frozen=True, slots=True)
class IrFloorEuropeanOptionB76Pricer:
    """
    Black76 pricer for a floor with market data lookup.
    """
    
    def price(self, trade: IrFloorEuropeanOption, market: Market) -> float:
        """Price a floor using Black76 with market data."""
        floorlets = self._generate_floorlets(trade, market)
        
        simple_floor = IrFloorEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            floorlets=tuple(floorlets),
        )
        
        return IrFloorEuropeanOptionB76PricerSimple().price(simple_floor)
    
    def greeks(self, trade: IrFloorEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """Compute aggregate Greeks for a floor."""
        floorlets = self._generate_floorlets(trade, market)
        
        simple_floor = IrFloorEuropeanOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            floorlets=tuple(floorlets),
        )
        
        return IrFloorEuropeanOptionB76PricerSimple().greeks(simple_floor)
    
    def _generate_floorlets(self, trade: IrFloorEuropeanOption, market: Market) -> List[IrFloorletEuropeanOptionSimple]:
        """Generate floorlets for each reset period."""
        curve = market.curve(trade.curve_id)
        vol_surface = market.vol_surface(trade.vol_id)
        
        floorlets = []
        
        t = float(trade.start_time)
        freq = float(trade.frequency)
        end = float(trade.end_time)
        
        while t < end - 1e-9:
            t_start = t
            t_end = min(t + freq, end)
            
            if t_start <= 0.0:
                t = t_end
                continue
            
            df_start = float(curve.df(t_start))
            df_end = float(curve.df(t_end))
            
            tau = compute_accrual_factor(t_start, t_end, trade.day_count)
            F = _forward_rate_from_dfs(df_start=df_start, df_end=df_end, accrual_factor=tau)
            sigma = float(vol_surface.vol(expiry=t_start, strike=trade.strike))
            
            floorlet = IrFloorletEuropeanOptionSimple(
                notional=trade.notional,
                strike=trade.strike,
                fixing_time=t_start,
                payment_time=t_end,
                accrual_factor=tau,
                forward_rate=F,
                vol=sigma,
                discount_factor=df_end,
            )
            floorlets.append(floorlet)
            
            t = t_end
        
        return floorlets


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Simple pricers
    "IrCapletEuropeanOptionB76PricerSimple",
    "IrFloorletEuropeanOptionB76PricerSimple",
    "IrCapEuropeanOptionB76PricerSimple",
    "IrFloorEuropeanOptionB76PricerSimple",
    # Market data pricers
    "IrCapletEuropeanOptionB76Pricer",
    "IrFloorletEuropeanOptionB76Pricer",
    "IrCapEuropeanOptionB76Pricer",
    "IrFloorEuropeanOptionB76Pricer",
]
