"""
Black-Karasinski Monte Carlo Pricers for Interest Rate Derivatives.

This module provides Monte Carlo pricers for IR instruments under the
Black-Karasinski short rate model where the LOG of the short rate
follows a mean-reverting Gaussian process.

Model: d(ln r) = [θ - a·ln r] dt + σ dW

Key Features
------------
- Log-normal rate distribution (rates always positive)
- Simulation-based pricing (no closed-form bond prices)
- Suitable for positive-rate environments

Supported Instruments
---------------------
- Zero-coupon bonds (via MC simulation)
- European bond options (via MC simulation)
- Caplets/Floorlets (via MC simulation)

Note: Black-Karasinski does not have closed-form bond prices, so all
pricing is done via Monte Carlo simulation.

Author: QuantStrata Team
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Optional

# Import instruments (only Simple variants used for MC pricing).
from src.instruments.ir.linear.bond import IrBondZeroCouponSimple
from src.instruments.ir.options.bond import IrBondEuropeanOptionSimple
from src.instruments.ir.options.capfloor import (
    IrCapletEuropeanOptionSimple,
    IrFloorletEuropeanOptionSimple,
)

# Import Black-Karasinski model.
from src.models.short_rate.black_karasinski import (
    BlackKarasinskiParameters,
    BlackKarasinskiDynamics,
)

# Import base MC infrastructure.
from src.models.numeric.monte_carlo.base import MonteCarloEstimate
from src.models.numeric.monte_carlo.estimators import estimate_from_samples


# Greek name type.
GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]


# =============================================================================
# MC CONFIGURATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class BKMCConfig:
    """
    Configuration for Black-Karasinski Monte Carlo pricing.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths. Default: 100,000.
    n_steps : int
        Number of time steps. Default: 252 (daily for 1 year).
    seed : int, optional
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates. Default: True.
    """

    n_paths: int = 100_000
    n_steps: int = 252
    seed: Optional[int] = None
    antithetic: bool = True


# Default MC configuration.
DEFAULT_BK_MC_CONFIG = BKMCConfig()


# =============================================================================
# ZERO COUPON BOND MC PRICER
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondZeroCouponBKMCPricerSimple:
    """
    Monte Carlo pricer for zero coupon bonds under Black-Karasinski.

    The bond price is computed as:
        P(0, T) = E[exp(-∫₀ᵀ r(s) ds)]

    Parameters
    ----------
    params : BlackKarasinskiParameters
        Black-Karasinski model parameters.
    config : BKMCConfig
        Monte Carlo configuration.
    """

    params: BlackKarasinskiParameters
    config: BKMCConfig = DEFAULT_BK_MC_CONFIG

    def price(self, trade: IrBondZeroCouponSimple) -> float:
        """
        Price a zero coupon bond using Monte Carlo.

        Parameters
        ----------
        trade : IrBondZeroCouponSimple
            Zero coupon bond instrument.

        Returns
        -------
        float
            MC estimate of bond price.
        """
        T = float(trade.maturity)
        face = float(trade.face_value)

        if T <= 0.0:
            return face

        dynamics = BlackKarasinskiDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T,
            n_paths=self.config.n_paths,
            n_steps=max(int(T * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )

        # Price = face × E[DF]
        mean_df = np.mean(sim.discount_factors)

        return face * mean_df

    def price_with_estimate(self, trade: IrBondZeroCouponSimple) -> MonteCarloEstimate:
        """
        Price with full Monte Carlo estimate including standard error.

        Returns
        -------
        MonteCarloEstimate
            Contains mean, stderr, n_paths, and 95% confidence interval.
        """
        T = float(trade.maturity)
        face = float(trade.face_value)

        if T <= 0.0:
            return MonteCarloEstimate(mean=face, stderr=0.0, n_paths=1)

        dynamics = BlackKarasinskiDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T,
            n_paths=self.config.n_paths,
            n_steps=max(int(T * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )

        payoffs = face * sim.discount_factors

        return estimate_from_samples(
            payoffs,
            meta={
                "instrument": "IrBondZeroCoupon",
                "model": "BlackKarasinski",
                "maturity": T,
                "seed": self.config.seed,
            },
        )

    def greeks(self, trade: IrBondZeroCouponSimple) -> Dict[str, float]:
        """Compute Greeks via finite difference bumping."""
        base_price = self.price(trade)
        T = float(trade.maturity)

        if T <= 0.0:
            return {"delta": 0.0, "dv01": 0.0, "vega": 0.0}

        # Delta: bump r0
        dr = 0.0001
        params_up = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 + dr,
            theta=self.params.theta,
        )
        pricer_up = IrBondZeroCouponBKMCPricerSimple(params=params_up, config=self.config)
        price_up = pricer_up.price(trade)
        delta = (price_up - base_price) / dr

        # Vega: bump sigma
        d_sigma = 0.001
        params_sigma_up = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma + d_sigma,
            r0=self.params.r0,
            theta=self.params.theta,
        )
        pricer_sigma_up = IrBondZeroCouponBKMCPricerSimple(
            params=params_sigma_up, config=self.config
        )
        price_sigma_up = pricer_sigma_up.price(trade)
        vega = (price_sigma_up - base_price) / d_sigma

        return {
            "delta": delta,
            "dv01": -delta * 0.0001,  # DV01 = -dP/dr × 1bp
            "vega": vega,
        }


# =============================================================================
# BOND OPTION MC PRICER
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondEuropeanOptionBKMCPricerSimple:
    """
    Monte Carlo pricer for European bond options under Black-Karasinski.

    Parameters
    ----------
    params : BlackKarasinskiParameters
        Black-Karasinski model parameters.
    config : BKMCConfig
        Monte Carlo configuration.
    """

    params: BlackKarasinskiParameters
    config: BKMCConfig = DEFAULT_BK_MC_CONFIG

    def price(self, trade: IrBondEuropeanOptionSimple) -> float:
        """
        Price a European bond option using Monte Carlo.

        Parameters
        ----------
        trade : IrBondEuropeanOptionSimple
            Bond option instrument.

        Returns
        -------
        float
            MC estimate of option price.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_opt = float(trade.expiry)
        F = float(trade.forward_bond_price)
        opt_type = trade.option_type

        if T_opt <= 0.0:
            # Expired - intrinsic value.
            if opt_type == "call":
                return N * max(F - K, 0.0)
            return N * max(K - F, 0.0)

        # Infer T_bond from forward price and r0.
        df = float(trade.discount_factor)
        r0 = self.params.r0
        T_bond = T_opt + 1.0 if r0 == 0 else T_opt - math.log(F * df) / r0
        if T_bond <= T_opt:
            T_bond = T_opt + 0.5

        # Simulate to option expiry.
        dynamics = BlackKarasinskiDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_opt,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_opt * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )

        # At option expiry, compute bond price P(T_opt, T_bond).
        # Under BK, use first-order approximation: P ≈ exp(-r(T_opt) × τ)
        r_T = sim.terminal_rates
        tau = T_bond - T_opt
        P_bond = np.exp(-r_T * tau) * 100.0  # Face = 100

        # Compute option payoffs.
        if opt_type == "call":
            payoffs = np.maximum(P_bond - K, 0.0)
        else:
            payoffs = np.maximum(K - P_bond, 0.0)

        # Discount payoffs to t=0.
        discounted_payoffs = sim.discount_factors * payoffs * N / 100.0

        return float(np.mean(discounted_payoffs))

    def price_with_estimate(self, trade: IrBondEuropeanOptionSimple) -> MonteCarloEstimate:
        """Price with full Monte Carlo estimate including standard error."""
        N = float(trade.notional)
        K = float(trade.strike)
        T_opt = float(trade.expiry)
        F = float(trade.forward_bond_price)
        opt_type = trade.option_type

        if T_opt <= 0.0:
            intrinsic = N * max(F - K, 0.0) if opt_type == "call" else N * max(K - F, 0.0)
            return MonteCarloEstimate(mean=intrinsic, stderr=0.0, n_paths=1)

        df = float(trade.discount_factor)
        r0 = self.params.r0
        T_bond = T_opt + 1.0 if r0 == 0 else T_opt - math.log(F * df) / r0
        if T_bond <= T_opt:
            T_bond = T_opt + 0.5

        dynamics = BlackKarasinskiDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_opt,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_opt * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )

        r_T = sim.terminal_rates
        tau = T_bond - T_opt
        P_bond = np.exp(-r_T * tau) * 100.0

        if opt_type == "call":
            payoffs = np.maximum(P_bond - K, 0.0)
        else:
            payoffs = np.maximum(K - P_bond, 0.0)

        discounted_payoffs = sim.discount_factors * payoffs * N / 100.0

        return estimate_from_samples(
            discounted_payoffs,
            meta={
                "instrument": "IrBondEuropeanOption",
                "model": "BlackKarasinski",
                "option_type": opt_type,
                "expiry": T_opt,
                "strike": K,
            },
        )

    def greeks(self, trade: IrBondEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks via finite difference bumping."""
        T_opt = float(trade.expiry)

        if T_opt <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        base_price = self.price(trade)

        # Delta/Gamma: bump r0
        dr = 0.0001
        params_up = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 + dr,
            theta=self.params.theta,
        )
        params_dn = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 - dr,
            theta=self.params.theta,
        )
        price_up = IrBondEuropeanOptionBKMCPricerSimple(
            params=params_up, config=self.config
        ).price(trade)
        price_dn = IrBondEuropeanOptionBKMCPricerSimple(
            params=params_dn, config=self.config
        ).price(trade)

        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)

        # Vega: bump sigma
        d_sigma = 0.001
        params_sigma_up = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma + d_sigma,
            r0=self.params.r0,
            theta=self.params.theta,
        )
        price_sigma_up = IrBondEuropeanOptionBKMCPricerSimple(
            params=params_sigma_up, config=self.config
        ).price(trade)
        vega = (price_sigma_up - base_price) / d_sigma

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": 0.0,  # Would need time-decay simulation
            "rho": delta,  # Approximation for short rate models
        }


# =============================================================================
# CAPLET MC PRICER
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrCapletEuropeanOptionBKMCPricerSimple:
    """
    Monte Carlo pricer for caplets under Black-Karasinski.

    A caplet pays: N × τ × max(L(T_fix, T_pay) - K, 0) at T_pay

    Parameters
    ----------
    params : BlackKarasinskiParameters
        Black-Karasinski model parameters.
    config : BKMCConfig
        Monte Carlo configuration.
    """

    params: BlackKarasinskiParameters
    config: BKMCConfig = DEFAULT_BK_MC_CONFIG

    def price(self, trade: IrCapletEuropeanOptionSimple) -> float:
        """
        Price a caplet using Monte Carlo.

        Parameters
        ----------
        trade : IrCapletEuropeanOptionSimple
            Caplet instrument.

        Returns
        -------
        float
            MC estimate of caplet price.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        T_pay = float(trade.payment_time)
        tau = float(trade.accrual_factor)

        if T_fix <= 0.0:
            # Expired - use intrinsic.
            F = float(trade.forward_rate)
            df = float(trade.discount_factor)
            return N * tau * df * max(F - K, 0.0)

        # Simulate to fixing time.
        dynamics = BlackKarasinskiDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_fix,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_fix * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )

        # At fixing time, the forward rate L(T_fix, T_pay) ≈ r(T_fix) for short periods.
        # For simplicity, use L ≈ r (this is an approximation).
        L_fix = sim.terminal_rates

        # Caplet payoff: τ × max(L - K, 0)
        payoffs = tau * np.maximum(L_fix - K, 0.0)

        # Discount to t=0 (we need DF from T_pay, approximate with DF from T_fix × extra).
        tau_extra = T_pay - T_fix
        df_extra = np.exp(-sim.terminal_rates * tau_extra)
        df_total = sim.discount_factors * df_extra

        discounted_payoffs = N * df_total * payoffs

        return float(np.mean(discounted_payoffs))

    def greeks(self, trade: IrCapletEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks via finite difference bumping."""
        T_fix = float(trade.fixing_time)

        if T_fix <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        base_price = self.price(trade)

        # Delta: bump r0
        dr = 0.0001
        params_up = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 + dr,
            theta=self.params.theta,
        )
        params_dn = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 - dr,
            theta=self.params.theta,
        )
        price_up = IrCapletEuropeanOptionBKMCPricerSimple(
            params=params_up, config=self.config
        ).price(trade)
        price_dn = IrCapletEuropeanOptionBKMCPricerSimple(
            params=params_dn, config=self.config
        ).price(trade)

        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)

        # Vega: bump sigma
        d_sigma = 0.001
        params_sigma_up = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma + d_sigma,
            r0=self.params.r0,
            theta=self.params.theta,
        )
        price_sigma_up = IrCapletEuropeanOptionBKMCPricerSimple(
            params=params_sigma_up, config=self.config
        ).price(trade)
        vega = (price_sigma_up - base_price) / d_sigma

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": 0.0,
            "rho": delta,
        }


# =============================================================================
# FLOORLET MC PRICER
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrFloorletEuropeanOptionBKMCPricerSimple:
    """
    Monte Carlo pricer for floorlets under Black-Karasinski.

    A floorlet pays: N × τ × max(K - L(T_fix, T_pay), 0) at T_pay

    Parameters
    ----------
    params : BlackKarasinskiParameters
        Black-Karasinski model parameters.
    config : BKMCConfig
        Monte Carlo configuration.
    """

    params: BlackKarasinskiParameters
    config: BKMCConfig = DEFAULT_BK_MC_CONFIG

    def price(self, trade: IrFloorletEuropeanOptionSimple) -> float:
        """
        Price a floorlet using Monte Carlo.

        Parameters
        ----------
        trade : IrFloorletEuropeanOptionSimple
            Floorlet instrument.

        Returns
        -------
        float
            MC estimate of floorlet price.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        T_pay = float(trade.payment_time)
        tau = float(trade.accrual_factor)

        if T_fix <= 0.0:
            # Expired - use intrinsic.
            F = float(trade.forward_rate)
            df = float(trade.discount_factor)
            return N * tau * df * max(K - F, 0.0)

        # Simulate to fixing time.
        dynamics = BlackKarasinskiDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_fix,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_fix * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )

        # At fixing time, forward rate L ≈ r (approximation).
        L_fix = sim.terminal_rates

        # Floorlet payoff: τ × max(K - L, 0)
        payoffs = tau * np.maximum(K - L_fix, 0.0)

        # Discount to t=0.
        tau_extra = T_pay - T_fix
        df_extra = np.exp(-sim.terminal_rates * tau_extra)
        df_total = sim.discount_factors * df_extra

        discounted_payoffs = N * df_total * payoffs

        return float(np.mean(discounted_payoffs))

    def greeks(self, trade: IrFloorletEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks via finite difference bumping."""
        T_fix = float(trade.fixing_time)

        if T_fix <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        base_price = self.price(trade)

        # Delta: bump r0
        dr = 0.0001
        params_up = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 + dr,
            theta=self.params.theta,
        )
        params_dn = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 - dr,
            theta=self.params.theta,
        )
        price_up = IrFloorletEuropeanOptionBKMCPricerSimple(
            params=params_up, config=self.config
        ).price(trade)
        price_dn = IrFloorletEuropeanOptionBKMCPricerSimple(
            params=params_dn, config=self.config
        ).price(trade)

        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)

        # Vega: bump sigma
        d_sigma = 0.001
        params_sigma_up = BlackKarasinskiParameters(
            a=self.params.a,
            sigma=self.params.sigma + d_sigma,
            r0=self.params.r0,
            theta=self.params.theta,
        )
        price_sigma_up = IrFloorletEuropeanOptionBKMCPricerSimple(
            params=params_sigma_up, config=self.config
        ).price(trade)
        vega = (price_sigma_up - base_price) / d_sigma

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": 0.0,
            "rho": delta,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "BKMCConfig",
    "DEFAULT_BK_MC_CONFIG",
    # Base MC result type (re-export for convenience)
    "MonteCarloEstimate",
    # Zero coupon bond MC pricer
    "IrBondZeroCouponBKMCPricerSimple",
    # Bond option MC pricer
    "IrBondEuropeanOptionBKMCPricerSimple",
    # Caplet/Floorlet MC pricers
    "IrCapletEuropeanOptionBKMCPricerSimple",
    "IrFloorletEuropeanOptionBKMCPricerSimple",
]
