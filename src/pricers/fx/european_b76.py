# src/pricers/fx/european_b76.py
"""
FX Forward Option Black76 Pricers.

Adapter pricers that map FX forward options to Black76 model pure functions.

Mathematical Framework
----------------------
The FX forward rate F for delivery at time T is:

    F = S × exp((r_d - r_f) × T)

Where:
    S = spot rate (domestic per foreign)
    r_d = domestic risk-free rate
    r_f = foreign risk-free rate
    T = time to forward delivery

The Black76 model prices the option as:
    Call: C = DF × [F×N(d₁) - K×N(d₂)]
    Put:  P = DF × [K×N(-d₂) - F×N(-d₁)]

Where DF = exp(-r_d × T_opt) is the domestic discount factor.

Greeks
------
- delta_forward: dPV/dF (sensitivity to forward rate)
- delta_spot: dPV/dS = delta_forward × exp((r_d - r_f) × T_fwd)
- gamma: d²PV/dF²
- vega: dPV/dσ (per 1.0 vol)
- theta: -dPV/dt (time decay)
- rho_domestic: dPV/dr_d
- rho_foreign: dPV/dr_f

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.core.market import Market
from src.instruments.fx.options.forward import (
    EuropeanFxForwardOption,
    EuropeanFxForwardOptionSimple,
)

# Import pure functions from Black76 model.
from src.models.analytic.black76.base import (
    vanilla_price,
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
    vanilla_rho,
)


# Greek names for FX forward options.
GreekName = Literal[
    "delta_forward",
    "delta_spot",
    "gamma",
    "vega",
    "theta",
    "rho_domestic",
    "rho_foreign",
]


def _rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.

    df = exp(-rT) => r = -ln(df)/T

    Parameters
    ----------
    df : float
        Discount factor.
    t : float
        Time to maturity.

    Returns
    -------
    float
        Continuously compounded rate.
    """
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


@dataclass(frozen=True, slots=True)
class FxForwardOptionBlack76Pricer:
    """
    Black76 pricer for FX forward options.

    This pricer:
    1. Extracts spot, curves, and vol from the market
    2. Computes the forward rate F = S × exp((r_d - r_f) × T_fwd)
    3. Applies Black76 formulas for pricing and Greeks

    Examples
    --------
    >>> pricer = FxForwardOptionBlack76Pricer()
    >>> pv = pricer.price(option, market)
    >>> greeks = pricer.greeks(option, market)
    """

    def price(self, option: EuropeanFxForwardOption, market: Market) -> float:
        """
        Compute present value of FX forward option.

        Parameters
        ----------
        option : EuropeanFxForwardOption
            The option to price.
        market : Market
            Market snapshot containing spot, curves, and vol surface.

        Returns
        -------
        float
            Present value in domestic currency.
        """
        # Extract market data.
        spot = market.quote(option.spot_id)
        dom_curve = market.curve(option.domestic_curve_id)
        for_curve = market.curve(option.foreign_curve_id)
        vol_surface = market.vol_surface(option.vol_id)

        # Get discount factors.
        t_opt = float(option.expiry)
        t_fwd = float(option.forward_expiry)

        df_dom_opt = dom_curve.df(t_opt)  # For discounting option payoff.
        df_dom_fwd = dom_curve.df(t_fwd)  # For forward calculation.
        df_for_fwd = for_curve.df(t_fwd)  # For forward calculation.

        # Compute forward rate: F = S × (DF_for / DF_dom) for forward delivery.
        # Equivalently: F = S × exp((r_d - r_f) × T_fwd).
        forward = spot * (df_for_fwd / df_dom_fwd) if df_dom_fwd > 0 else spot

        # Get volatility at strike for option expiry.
        vol = vol_surface.vol(t_opt, float(option.strike))

        # Apply Black76.
        unit_price = vanilla_price(
            option_type=option.option_type,
            forward=forward,
            strike=float(option.strike),
            expiry=t_opt,
            discount_factor=df_dom_opt,
            vol=vol,
        )

        return unit_price * float(option.notional)

    def greeks(self, option: EuropeanFxForwardOption, market: Market) -> Dict[GreekName, float]:
        """
        Compute Greeks for FX forward option.

        Parameters
        ----------
        option : EuropeanFxForwardOption
            The option to price.
        market : Market
            Market snapshot.

        Returns
        -------
        dict
            Dictionary with keys: delta_forward, delta_spot, gamma, vega,
            theta, rho_domestic, rho_foreign.
        """
        # Extract market data.
        spot = market.quote(option.spot_id)
        dom_curve = market.curve(option.domestic_curve_id)
        for_curve = market.curve(option.foreign_curve_id)
        vol_surface = market.vol_surface(option.vol_id)

        t_opt = float(option.expiry)
        t_fwd = float(option.forward_expiry)
        notional = float(option.notional)
        strike = float(option.strike)

        # Discount factors.
        df_dom_opt = dom_curve.df(t_opt)
        df_dom_fwd = dom_curve.df(t_fwd)
        df_for_fwd = for_curve.df(t_fwd)

        # Forward rate.
        forward = spot * (df_for_fwd / df_dom_fwd) if df_dom_fwd > 0 else spot

        # Volatility.
        vol = vol_surface.vol(t_opt, strike)

        # Rates (for theta and rho calculations).
        r_dom = _rate_from_df(df=df_dom_opt, t=t_opt)

        # Black76 Greeks (per unit notional).
        delta_fwd = vanilla_delta(
            option_type=option.option_type,
            forward=forward,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_dom_opt,
            vol=vol,
        )

        gamma_fwd = vanilla_gamma(
            option_type=option.option_type,
            forward=forward,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_dom_opt,
            vol=vol,
        )

        vega_fwd = vanilla_vega(
            option_type=option.option_type,
            forward=forward,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_dom_opt,
            vol=vol,
        )

        theta_fwd = vanilla_theta(
            option_type=option.option_type,
            forward=forward,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_dom_opt,
            discount_rate=r_dom,
            vol=vol,
        )

        rho_fwd = vanilla_rho(
            option_type=option.option_type,
            forward=forward,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_dom_opt,
            vol=vol,
        )

        # Convert forward delta to spot delta.
        # delta_spot = delta_forward × (dF/dS) = delta_forward × (DF_for / DF_dom).
        fwd_ratio = df_for_fwd / df_dom_fwd if df_dom_fwd > 0 else 1.0
        delta_spot = delta_fwd * fwd_ratio

        # Rho decomposition for FX:
        # dPV/dr_d comes from:
        #   1. Change in discount factor: -T_opt × PV
        #   2. Change in forward: dF/dr_d = -T_fwd × F × (DF_for/DF_dom)
        # dPV/dr_f comes from:
        #   1. Change in forward: dF/dr_f = +T_fwd × F × (DF_for/DF_dom)
        #
        # For simplicity, we return the discount rho and adjust.
        # Full decomposition would require chain rule through forward.
        pv_unit = vanilla_price(
            option_type=option.option_type,
            forward=forward,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_dom_opt,
            vol=vol,
        )

        # Approximate rho_domestic and rho_foreign.
        # rho_domestic ≈ rho (discount effect) + delta_forward × (dF/dr_d)
        # rho_foreign ≈ delta_forward × (dF/dr_f)
        rho_domestic = rho_fwd + delta_fwd * (-t_fwd * forward)
        rho_foreign = delta_fwd * (t_fwd * forward)

        return {
            "delta_forward": delta_fwd * notional,
            "delta_spot": delta_spot * notional,
            "gamma": gamma_fwd * notional,
            "vega": vega_fwd * notional,
            "theta": theta_fwd * notional,
            "rho_domestic": rho_domestic * notional,
            "rho_foreign": rho_foreign * notional,
        }


@dataclass(frozen=True, slots=True)
class FxForwardOptionBlack76PricerSimple:
    """
    Simplified Black76 pricer for FX forward options.

    Use when forward rate, vol, and discount factor are provided directly.

    Examples
    --------
    >>> pricer = FxForwardOptionBlack76PricerSimple()
    >>> pv = pricer.price(option)
    >>> greeks = pricer.greeks(option)
    """

    def price(self, option: EuropeanFxForwardOptionSimple) -> float:
        """
        Compute present value using Black76.

        Parameters
        ----------
        option : EuropeanFxForwardOptionSimple
            Option with embedded market parameters.

        Returns
        -------
        float
            Present value in domestic currency.
        """
        unit_price = vanilla_price(
            option_type=option.option_type,
            forward=float(option.forward_rate),
            strike=float(option.strike),
            expiry=float(option.expiry),
            discount_factor=float(option.discount_factor),
            vol=float(option.vol),
        )

        return unit_price * float(option.notional)

    def greeks(self, option: EuropeanFxForwardOptionSimple) -> Dict[str, float]:
        """
        Compute Greeks using Black76.

        Parameters
        ----------
        option : EuropeanFxForwardOptionSimple
            Option with embedded market parameters.

        Returns
        -------
        dict
            Dictionary with keys: delta, gamma, vega, theta, rho.
        """
        t = float(option.expiry)
        df = float(option.discount_factor)
        notional = float(option.notional)

        # Compute rate from discount factor.
        r = _rate_from_df(df=df, t=t)

        delta = vanilla_delta(
            option_type=option.option_type,
            forward=float(option.forward_rate),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            vol=float(option.vol),
        )

        gamma = vanilla_gamma(
            option_type=option.option_type,
            forward=float(option.forward_rate),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            vol=float(option.vol),
        )

        vega = vanilla_vega(
            option_type=option.option_type,
            forward=float(option.forward_rate),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            vol=float(option.vol),
        )

        theta = vanilla_theta(
            option_type=option.option_type,
            forward=float(option.forward_rate),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            discount_rate=r,
            vol=float(option.vol),
        )

        rho = vanilla_rho(
            option_type=option.option_type,
            forward=float(option.forward_rate),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            vol=float(option.vol),
        )

        return {
            "delta": delta * notional,
            "gamma": gamma * notional,
            "vega": vega * notional,
            "theta": theta * notional,
            "rho": rho * notional,
        }


__all__ = [
    "FxForwardOptionBlack76Pricer",
    "FxForwardOptionBlack76PricerSimple",
]
