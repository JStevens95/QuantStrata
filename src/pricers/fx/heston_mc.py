"""
Heston Monte Carlo Pricer for FX Options.

This module provides Monte Carlo pricing of European vanilla options under
the Heston stochastic volatility model.

Pricing Framework
-----------------
Under the Heston model, the risk-neutral dynamics are:

    dS_t = (r_d - r_f) S_t dt + √V_t S_t dW_t^S
    dV_t = κ(θ - V_t) dt + ξ √V_t dW_t^V
    Corr(dW_t^S, dW_t^V) = ρ

European option price is computed as:
    C = e^(-r_d T) E[(S_T - K)^+]
    P = e^(-r_d T) E[(K - S_T)^+]

Implementation Notes
--------------------
1. Paths are simulated using HestonDynamics with user-specified scheme.
2. Variance reduction via antithetic variates.
3. Standard error computed for confidence intervals.
4. Control variate using delta (optional, advanced).

References
----------
- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic
  Volatility."
- Andersen, L. (2008). "Efficient Simulation of the Heston Model."
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Protocol

from src.models.stochastic_volatility.heston import (
    HestonDynamics,
    HestonParameters,
    HestonSimulation,
    HestonScheme,
)
from src.models.payoffs.types import OptionType


# =============================================================================
# Market data protocol
# =============================================================================

class HestonMarketData(Protocol):
    """Protocol for market data needed by Heston pricer."""

    @property
    def spot(self) -> float:
        """Current spot price."""
        ...

    @property
    def domestic_rate(self) -> float:
        """Domestic risk-free rate (continuous)."""
        ...

    @property
    def foreign_rate(self) -> float:
        """Foreign/dividend rate (continuous)."""
        ...

    @property
    def heston_params(self) -> HestonParameters:
        """Heston model parameters."""
        ...


# =============================================================================
# Simulation result
# =============================================================================

@dataclass(frozen=True, slots=True)
class HestonMcResult:
    """
    Result container for Heston Monte Carlo pricing.

    Attributes
    ----------
    price : float
        Option price.
    std_error : float
        Standard error of the price estimate.
    n_paths : int
        Number of paths used.
    n_steps : int
        Number of time steps used.
    mean_terminal_spot : float
        Average terminal spot (for debugging).
    mean_terminal_vol : float
        Average terminal volatility (for debugging).
    simulation : HestonSimulation
        Full simulation object (for Greeks, diagnostics).
    """

    price: float
    std_error: float
    n_paths: int
    n_steps: int
    mean_terminal_spot: float
    mean_terminal_vol: float
    simulation: HestonSimulation

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """95% confidence interval for the price."""
        z = 1.96
        return (self.price - z * self.std_error, self.price + z * self.std_error)


# =============================================================================
# Heston MC Pricer
# =============================================================================

@dataclass(frozen=True, slots=True)
class FxHestonMcPricer:
    """
    Monte Carlo pricer for FX European options under Heston model.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths. Default: 100_000.
    n_steps : int
        Number of time steps. Default: 252 (daily).
    scheme : HestonScheme
        Variance discretization scheme. Default: "full_truncation".
    seed : int, optional
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates. Default: True.

    Examples
    --------
    >>> from src.pricers.fx.heston_mc import FxHestonMcPricer
    >>> from src.models.stochastic_vol.heston import HestonParameters
    >>> pricer = FxHestonMcPricer(n_paths=50000, n_steps=100)
    >>> params = HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)
    >>> result = pricer.price_european(
    ...     spot=100.0, strike=100.0, maturity=1.0,
    ...     domestic_rate=0.05, foreign_rate=0.02,
    ...     heston_params=params, option_type="call"
    ... )
    >>> result.price  # Option price.
    """

    n_paths: int = 100_000
    n_steps: int = 252
    scheme: HestonScheme = "full_truncation"
    seed: Optional[int] = None
    antithetic: bool = True

    def __post_init__(self) -> None:
        """Validate pricer configuration."""
        if self.n_paths <= 0:
            raise ValueError("n_paths must be > 0.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be > 0.")

    def price_european(
        self,
        spot: float,
        strike: float,
        maturity: float,
        domestic_rate: float,
        foreign_rate: float,
        heston_params: HestonParameters,
        option_type: OptionType,
    ) -> HestonMcResult:
        """
        Price a European vanilla option under Heston model.

        Parameters
        ----------
        spot : float
            Current spot price S_0.
        strike : float
            Option strike K.
        maturity : float
            Time to maturity T (years).
        domestic_rate : float
            Domestic risk-free rate r_d.
        foreign_rate : float
            Foreign rate r_f (or dividend yield).
        heston_params : HestonParameters
            Heston model parameters.
        option_type : OptionType
            "call" or "put".

        Returns
        -------
        HestonMcResult
            Pricing result with price, std error, and diagnostics.
        """
        # Validate inputs.
        if spot <= 0.0:
            raise ValueError("spot must be > 0.")
        if strike <= 0.0:
            raise ValueError("strike must be > 0.")
        if maturity <= 0.0:
            raise ValueError("maturity must be > 0.")

        # Compute drift μ = r_d - r_f.
        drift = domestic_rate - foreign_rate

        # Create dynamics simulator.
        dynamics = HestonDynamics(params=heston_params, drift=drift)

        # Simulate paths.
        simulation = dynamics.simulate(
            spot0=spot,
            maturity=maturity,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            scheme=self.scheme,
            seed=self.seed,
            antithetic=self.antithetic,
        )

        # Extract terminal spots.
        S_T = simulation.terminal_spots

        # Compute payoffs.
        if option_type == "call":
            payoffs = np.maximum(S_T - strike, 0.0)
        elif option_type == "put":
            payoffs = np.maximum(strike - S_T, 0.0)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")

        # Discount factor.
        discount = np.exp(-domestic_rate * maturity)

        # Discounted payoffs.
        discounted_payoffs = discount * payoffs

        # Price estimate (mean).
        price = float(np.mean(discounted_payoffs))

        # Standard error.
        std_error = float(np.std(discounted_payoffs) / np.sqrt(simulation.n_paths))

        # Diagnostics.
        mean_terminal_spot = float(np.mean(S_T))
        mean_terminal_vol = float(np.mean(np.sqrt(np.maximum(simulation.terminal_variances, 0.0))))

        return HestonMcResult(
            price=price,
            std_error=std_error,
            n_paths=simulation.n_paths,
            n_steps=self.n_steps,
            mean_terminal_spot=mean_terminal_spot,
            mean_terminal_vol=mean_terminal_vol,
            simulation=simulation,
        )

    def price_with_greeks(
        self,
        spot: float,
        strike: float,
        maturity: float,
        domestic_rate: float,
        foreign_rate: float,
        heston_params: HestonParameters,
        option_type: OptionType,
        bump_spot: float = 0.01,
        bump_vol: float = 0.01,
    ) -> dict:
        """
        Price option and compute Greeks via finite difference.

        Parameters
        ----------
        spot, strike, maturity, domestic_rate, foreign_rate, heston_params, option_type :
            See price_european().
        bump_spot : float
            Relative spot bump for delta/gamma (1% default).
        bump_vol : float
            Absolute vol bump for vega (1% default).

        Returns
        -------
        dict
            Dictionary with 'price', 'delta', 'gamma', 'vega', 'theta', 'rho'.
        """
        # Base price.
        base = self.price_european(
            spot=spot, strike=strike, maturity=maturity,
            domestic_rate=domestic_rate, foreign_rate=foreign_rate,
            heston_params=heston_params, option_type=option_type,
        )

        # Delta: ∂V/∂S.
        ds = spot * bump_spot
        price_up = self.price_european(
            spot=spot + ds, strike=strike, maturity=maturity,
            domestic_rate=domestic_rate, foreign_rate=foreign_rate,
            heston_params=heston_params, option_type=option_type,
        ).price
        price_down = self.price_european(
            spot=spot - ds, strike=strike, maturity=maturity,
            domestic_rate=domestic_rate, foreign_rate=foreign_rate,
            heston_params=heston_params, option_type=option_type,
        ).price
        delta = (price_up - price_down) / (2 * ds)

        # Gamma: ∂²V/∂S².
        gamma = (price_up - 2 * base.price + price_down) / (ds ** 2)

        # Vega: ∂V/∂σ (bump v0 and theta together).
        v0_up = heston_params.v0 * (1 + bump_vol) ** 2
        theta_up = heston_params.theta * (1 + bump_vol) ** 2
        params_up = HestonParameters(
            kappa=heston_params.kappa,
            theta=theta_up,
            xi=heston_params.xi,
            v0=v0_up,
            rho=heston_params.rho,
        )
        price_vol_up = self.price_european(
            spot=spot, strike=strike, maturity=maturity,
            domestic_rate=domestic_rate, foreign_rate=foreign_rate,
            heston_params=params_up, option_type=option_type,
        ).price
        vega = (price_vol_up - base.price) / bump_vol

        # Theta: -∂V/∂T (approximate with small time bump).
        dt = 1.0 / 365.0  # 1 day.
        if maturity > dt:
            price_t_minus = self.price_european(
                spot=spot, strike=strike, maturity=maturity - dt,
                domestic_rate=domestic_rate, foreign_rate=foreign_rate,
                heston_params=heston_params, option_type=option_type,
            ).price
            theta = -(base.price - price_t_minus) / dt
        else:
            theta = 0.0

        # Rho: ∂V/∂r.
        dr = 0.01  # 1bp.
        price_r_up = self.price_european(
            spot=spot, strike=strike, maturity=maturity,
            domestic_rate=domestic_rate + dr, foreign_rate=foreign_rate,
            heston_params=heston_params, option_type=option_type,
        ).price
        rho = (price_r_up - base.price) / dr

        return {
            "price": base.price,
            "std_error": base.std_error,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }


# =============================================================================
# Convenience function
# =============================================================================

def price_heston_european(
    spot: float,
    strike: float,
    maturity: float,
    domestic_rate: float,
    foreign_rate: float,
    kappa: float,
    theta: float,
    xi: float,
    v0: float,
    rho: float,
    option_type: OptionType,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> float:
    """
    Convenience function to price European option under Heston.

    Parameters
    ----------
    spot, strike, maturity, domestic_rate, foreign_rate :
        Market and trade parameters.
    kappa, theta, xi, v0, rho :
        Heston model parameters.
    option_type :
        "call" or "put".
    n_paths, n_steps, seed :
        Simulation parameters.

    Returns
    -------
    float
        Option price.

    Examples
    --------
    >>> from src.pricers.fx.heston_mc import price_heston_european
    >>> price = price_heston_european(
    ...     spot=100.0, strike=100.0, maturity=1.0,
    ...     domestic_rate=0.05, foreign_rate=0.02,
    ...     kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7,
    ...     option_type="call", n_paths=50000, seed=42
    ... )
    """
    params = HestonParameters(kappa=kappa, theta=theta, xi=xi, v0=v0, rho=rho)
    pricer = FxHestonMcPricer(n_paths=n_paths, n_steps=n_steps, seed=seed)
    result = pricer.price_european(
        spot=spot, strike=strike, maturity=maturity,
        domestic_rate=domestic_rate, foreign_rate=foreign_rate,
        heston_params=params, option_type=option_type,
    )
    return result.price
