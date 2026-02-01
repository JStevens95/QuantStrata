# src/pricers/equity/european_b76.py
"""
Equity European Black76 Pricers.

Adapter pricers that map equity futures options to Black76 model pure functions.

Mathematical Framework
----------------------
The equity futures price F for delivery at time T is:

    F = S × exp((r - q) × T)

Where:
    S = spot index level
    r = risk-free rate
    q = continuous dividend yield
    T = time to futures delivery

The Black76 model prices the option as:
    Call: C = DF × [F×N(d₁) - K×N(d₂)]
    Put:  P = DF × [K×N(-d₂) - F×N(-d₁)]

Where DF = exp(-r × T_opt) is the discount factor.

Greeks
------
- delta_futures: dPV/dF (sensitivity to futures price)
- delta_spot: dPV/dS = delta_futures × exp((r - q) × T_fut)
- gamma: d²PV/dF²
- vega: dPV/dσ (per 1.0 vol)
- theta: -dPV/dt (time decay)
- rho: dPV/dr (rate sensitivity)

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.core.market import Market
from src.instruments.equity.options.futures import (
    EquityFuturesEuropeanOption,
    EquityFuturesEuropeanOptionSimple,
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


# Greek names for equity futures options.
GreekName = Literal[
    "delta_futures",
    "delta_spot",
    "gamma",
    "vega",
    "theta",
    "rho",
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
class EquityFuturesEuropeanOptionB76Pricer:
    """
    Black76 pricer for equity futures options.

    This pricer:
    1. Extracts spot, curve, and vol from the market
    2. Computes the futures price F = S × exp((r - q) × T_fut)
    3. Applies Black76 formulas for pricing and Greeks

    Examples
    --------
    >>> pricer = EquityFuturesEuropeanOptionB76Pricer()
    >>> pv = pricer.price(option, market)
    >>> greeks = pricer.greeks(option, market)
    """

    def price(self, option: EquityFuturesEuropeanOption, market: Market) -> float:
        """
        Compute present value of equity futures option.

        Parameters
        ----------
        option : EquityFuturesEuropeanOption
            The option to price.
        market : Market
            Market snapshot containing spot, curve, and vol surface.

        Returns
        -------
        float
            Present value in currency units.
        """
        # Extract market data.
        spot = market.quote(option.spot_id)
        curve = market.curve(option.curve_id)
        vol_surface = market.vol_surface(option.vol_id)

        # Get discount factors and rate.
        t_opt = float(option.expiry)
        t_fut = float(option.futures_expiry)
        q = float(option.dividend_yield)

        df_opt = curve.df(t_opt)  # For discounting option payoff.
        df_fut = curve.df(t_fut)  # For futures calculation.
        r = _rate_from_df(df=df_fut, t=t_fut)

        # Compute futures price: F = S × exp((r - q) × T_fut).
        futures_price = spot * math.exp((r - q) * t_fut)

        # Get volatility at strike for option expiry.
        vol = vol_surface.vol(t_opt, float(option.strike))

        # Apply Black76.
        unit_price = vanilla_price(
            option_type=option.option_type,
            forward=futures_price,
            strike=float(option.strike),
            expiry=t_opt,
            discount_factor=df_opt,
            vol=vol,
        )

        return unit_price * float(option.notional)

    def greeks(self, option: EquityFuturesEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """
        Compute Greeks for equity futures option.

        Parameters
        ----------
        option : EquityFuturesEuropeanOption
            The option to price.
        market : Market
            Market snapshot.

        Returns
        -------
        dict
            Dictionary with keys: delta_futures, delta_spot, gamma, vega,
            theta, rho.
        """
        # Extract market data.
        spot = market.quote(option.spot_id)
        curve = market.curve(option.curve_id)
        vol_surface = market.vol_surface(option.vol_id)

        t_opt = float(option.expiry)
        t_fut = float(option.futures_expiry)
        notional = float(option.notional)
        strike = float(option.strike)
        q = float(option.dividend_yield)

        # Discount factor and rate.
        df_opt = curve.df(t_opt)
        df_fut = curve.df(t_fut)
        r = _rate_from_df(df=df_fut, t=t_fut)

        # Futures price.
        futures_price = spot * math.exp((r - q) * t_fut)

        # Volatility.
        vol = vol_surface.vol(t_opt, strike)

        # Black76 Greeks (per unit notional).
        delta_fut = vanilla_delta(
            option_type=option.option_type,
            forward=futures_price,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_opt,
            vol=vol,
        )

        gamma_fut = vanilla_gamma(
            option_type=option.option_type,
            forward=futures_price,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_opt,
            vol=vol,
        )

        vega_fut = vanilla_vega(
            option_type=option.option_type,
            forward=futures_price,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_opt,
            vol=vol,
        )

        theta_fut = vanilla_theta(
            option_type=option.option_type,
            forward=futures_price,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_opt,
            discount_rate=r,
            vol=vol,
        )

        rho_fut = vanilla_rho(
            option_type=option.option_type,
            forward=futures_price,
            strike=strike,
            expiry=t_opt,
            discount_factor=df_opt,
            vol=vol,
        )

        # Convert futures delta to spot delta.
        # delta_spot = delta_futures × (dF/dS) = delta_futures × exp((r - q) × T_fut).
        fwd_ratio = math.exp((r - q) * t_fut)
        delta_spot = delta_fut * fwd_ratio

        return {
            "delta_futures": delta_fut * notional,
            "delta_spot": delta_spot * notional,
            "gamma": gamma_fut * notional,
            "vega": vega_fut * notional,
            "theta": theta_fut * notional,
            "rho": rho_fut * notional,
        }


@dataclass(frozen=True, slots=True)
class EquityFuturesEuropeanOptionB76PricerSimple:
    """
    Simplified Black76 pricer for equity futures options.

    Use when futures price, vol, and discount factor are provided directly.

    Examples
    --------
    >>> pricer = EquityFuturesEuropeanOptionB76PricerSimple()
    >>> pv = pricer.price(option)
    >>> greeks = pricer.greeks(option)
    """

    def price(self, option: EquityFuturesEuropeanOptionSimple) -> float:
        """
        Compute present value using Black76.

        Parameters
        ----------
        option : EquityFuturesEuropeanOptionSimple
            Option with embedded market parameters.

        Returns
        -------
        float
            Present value in currency units.
        """
        unit_price = vanilla_price(
            option_type=option.option_type,
            forward=float(option.futures_price),
            strike=float(option.strike),
            expiry=float(option.expiry),
            discount_factor=float(option.discount_factor),
            vol=float(option.vol),
        )

        return unit_price * float(option.notional)

    def greeks(self, option: EquityFuturesEuropeanOptionSimple) -> Dict[str, float]:
        """
        Compute Greeks using Black76.

        Parameters
        ----------
        option : EquityFuturesEuropeanOptionSimple
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
            forward=float(option.futures_price),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            vol=float(option.vol),
        )

        gamma = vanilla_gamma(
            option_type=option.option_type,
            forward=float(option.futures_price),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            vol=float(option.vol),
        )

        vega = vanilla_vega(
            option_type=option.option_type,
            forward=float(option.futures_price),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            vol=float(option.vol),
        )

        theta = vanilla_theta(
            option_type=option.option_type,
            forward=float(option.futures_price),
            strike=float(option.strike),
            expiry=t,
            discount_factor=df,
            discount_rate=r,
            vol=float(option.vol),
        )

        rho = vanilla_rho(
            option_type=option.option_type,
            forward=float(option.futures_price),
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
    "EquityFuturesEuropeanOptionB76Pricer",
    "EquityFuturesEuropeanOptionB76PricerSimple",
]
