"""
Spread Option Pricers.

This module provides pricing for spread options:
- Monte Carlo pricing (MultiAssetSpreadEuropeanOptionMcPricer)
- Kirk's approximation (closed-form)
- Margrabe's formula (exact for exchange options)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm

from src.instruments.multi_asset.spread import (
    MultiAssetSpreadEuropeanOption,
    MultiAssetExchangeEuropeanOption,
)
from src.models.numeric.monte_carlo.multi_asset import (
    CorrelationMatrix,
    MultiAssetGBM,
)


# =============================================================================
# Simulation Artifact
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetSpreadEuropeanOptionMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for spread options.

    Notes
    -----
    - discounted_payoffs are already scaled by notional.
    """

    # --- Resolved inputs ---
    spot1: float  # Initial spot asset 1.
    spot2: float  # Initial spot asset 2.
    strike: float  # Strike K.
    maturity: float  # Expiry T.
    r: float  # Risk-free rate.
    option_type: str  # "call" or "put".
    notional: float  # Notional scaling.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths.
    antithetic: bool  # Antithetic flag.
    seed: Optional[int]  # RNG seed.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots, shape (n_paths_effective, 2).
    spread_values: np.ndarray  # Spread values at expiry, shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples.


# =============================================================================
# Monte Carlo Pricer Class
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetSpreadEuropeanOptionMcPricer:
    """
    Monte Carlo pricer for European multi-asset spread options.

    Consistent with FX pricer class conventions.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths. Default 200,000.
    seed : int or None
        RNG seed for reproducibility. Default 7.
    antithetic : bool
        Whether to use antithetic variates. Default True.
    """

    n_paths: int = 200_000
    seed: Optional[int] = 7
    antithetic: bool = True

    def price(
        self,
        trade: MultiAssetSpreadEuropeanOption,
        spot1: float,
        spot2: float,
        r: float,
        q1: float,
        q2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
    ) -> float:
        """
        Compute the present value of the spread option.

        Parameters
        ----------
        trade : MultiAssetSpreadEuropeanOption
            The spread option instrument.
        spot1, spot2 : float
            Current spot prices.
        r : float
            Risk-free rate.
        q1, q2 : float
            Dividend yields.
        sigma1, sigma2 : float
            Volatilities.
        rho : float
            Correlation.

        Returns
        -------
        float
            Present value scaled by notional.
        """
        sim = self.run(trade, spot1, spot2, r, q1, q2, sigma1, sigma2, rho)
        return float(sim.discounted_payoffs.mean())

    def price_with_std_error(
        self,
        trade: MultiAssetSpreadEuropeanOption,
        spot1: float,
        spot2: float,
        r: float,
        q1: float,
        q2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
    ) -> tuple[float, float]:
        """
        Compute price and standard error.

        Returns
        -------
        tuple[float, float]
            (price, standard_error) scaled by notional.
        """
        sim = self.run(trade, spot1, spot2, r, q1, q2, sigma1, sigma2, rho)
        price = float(sim.discounted_payoffs.mean())
        std_error = float(sim.discounted_payoffs.std() / np.sqrt(sim.n_paths_effective))
        return price, std_error

    def run(
        self,
        trade: MultiAssetSpreadEuropeanOption,
        spot1: float,
        spot2: float,
        r: float,
        q1: float,
        q2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
    ) -> MultiAssetSpreadEuropeanOptionMcSimulation:
        """
        Run the full simulation and return artifact with all details.
        """
        corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
        correlation = CorrelationMatrix(corr_matrix)

        gbm = MultiAssetGBM(
            spots=np.array([spot1, spot2]),
            r=r,
            dividends=np.array([q1, q2]),
            volatilities=np.array([sigma1, sigma2]),
            correlation=correlation,
        )

        terminals = gbm.simulate_terminal(
            maturity=trade.expiry,
            n_paths=self.n_paths,
            seed=self.seed,
            antithetic=self.antithetic,
        )

        n_paths_eff = terminals.shape[0]
        spread_values = terminals[:, 0] - terminals[:, 1]

        if trade.option_type == "call":
            payoffs = np.maximum(spread_values - trade.strike, 0)
        else:  # put
            payoffs = np.maximum(trade.strike - spread_values, 0)

        discount = np.exp(-r * trade.expiry)
        discounted_payoffs = (discount * payoffs * trade.notional).astype(np.float64)

        return MultiAssetSpreadEuropeanOptionMcSimulation(
            spot1=spot1,
            spot2=spot2,
            strike=trade.strike,
            maturity=trade.expiry,
            r=r,
            option_type=trade.option_type,
            notional=trade.notional,
            n_paths_requested=self.n_paths,
            n_paths_effective=n_paths_eff,
            antithetic=self.antithetic,
            seed=self.seed,
            terminal_spots=terminals,
            spread_values=spread_values,
            discounted_payoffs=discounted_payoffs,
        )


# =============================================================================
# Kirk's Approximation (Analytic)
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetSpreadEuropeanOptionKirkPricer:
    """
    Kirk's approximation pricer for European spread options.

    An efficient closed-form approximation for spread options.
    """

    def price(
        self,
        trade: MultiAssetSpreadEuropeanOption,
        spot1: float,
        spot2: float,
        r: float,
        q1: float,
        q2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
    ) -> float:
        """
        Compute Kirk's approximation for the spread option.

        Parameters
        ----------
        trade : MultiAssetSpreadEuropeanOption
            The spread option instrument.
        spot1, spot2 : float
            Current spot prices.
        r : float
            Risk-free rate.
        q1, q2 : float
            Dividend yields.
        sigma1, sigma2 : float
            Volatilities.
        rho : float
            Correlation.

        Returns
        -------
        float
            Approximate price scaled by notional.
        """
        T = trade.expiry
        K = trade.strike

        F1 = spot1 * np.exp((r - q1) * T)
        F2 = spot2 * np.exp((r - q2) * T)

        F2_K = F2 + K

        if F2_K <= 0:
            if trade.option_type == "call":
                return np.exp(-r * T) * max(F1 - F2 - K, 0) * trade.notional
            else:
                return np.exp(-r * T) * max(K - (F1 - F2), 0) * trade.notional

        w = F2 / F2_K
        sigma_eff = np.sqrt(sigma1 ** 2 - 2 * rho * sigma1 * sigma2 * w + (sigma2 * w) ** 2)

        d1 = (np.log(F1 / F2_K) + 0.5 * sigma_eff ** 2 * T) / (sigma_eff * np.sqrt(T))
        d2 = d1 - sigma_eff * np.sqrt(T)

        call_price = np.exp(-r * T) * (F1 * norm.cdf(d1) - F2_K * norm.cdf(d2))

        if trade.option_type == "call":
            return float(call_price) * trade.notional
        else:
            # Put-call parity for spreads
            forward_spread = F1 - F2 - K
            put_price = call_price - np.exp(-r * T) * forward_spread
            return float(max(put_price, 0)) * trade.notional


# =============================================================================
# Margrabe's Formula (Exact for Exchange Options)
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetExchangeEuropeanOptionMargrabePricer:
    """
    Margrabe's formula pricer for exchange options (spread with K=0).

    Exact closed-form solution for the option to exchange one asset for another.
    """

    def price(
        self,
        trade: MultiAssetExchangeEuropeanOption,
        spot1: float,
        spot2: float,
        r: float,
        q1: float,
        q2: float,
        sigma1: float,
        sigma2: float,
        rho: float,
    ) -> float:
        """
        Compute Margrabe's formula for the exchange option.

        Parameters
        ----------
        trade : MultiAssetExchangeEuropeanOption
            The exchange option instrument.
        spot1, spot2 : float
            Current spot prices.
        r : float
            Risk-free rate (not used directly, but for consistency).
        q1, q2 : float
            Dividend yields.
        sigma1, sigma2 : float
            Volatilities.
        rho : float
            Correlation.

        Returns
        -------
        float
            Exact exchange option price scaled by notional.
        """
        T = trade.expiry

        S1_adj = spot1 * np.exp(-q1 * T)
        S2_adj = spot2 * np.exp(-q2 * T)

        sigma = np.sqrt(sigma1 ** 2 - 2 * rho * sigma1 * sigma2 + sigma2 ** 2)

        if sigma < 1e-10:
            return float(max(S1_adj - S2_adj, 0)) * trade.notional

        d1 = (np.log(S1_adj / S2_adj) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = S1_adj * norm.cdf(d1) - S2_adj * norm.cdf(d2)

        return float(price) * trade.notional
