"""
Rainbow Option Monte Carlo Pricers.

This module provides Monte Carlo pricing for rainbow options (best-of, worst-of)
using pricer classes consistent with FX pricer conventions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.instruments.multi_asset.rainbow import (
    MultiAssetBestOfEuropeanOption,
    MultiAssetWorstOfEuropeanOption,
)
from src.models.numeric.monte_carlo.multi_asset import (
    CorrelationMatrix,
    MultiAssetGBM,
)


# =============================================================================
# Simulation Artifacts
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetBestOfEuropeanOptionMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for best-of options.

    Notes
    -----
    - discounted_payoffs are already scaled by notional.
    """

    # --- Resolved inputs ---
    spots: np.ndarray  # Initial spots, shape (n_assets,).
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
    terminal_spots: np.ndarray  # Terminal spots, shape (n_paths_effective, n_assets).
    best_values: np.ndarray  # Best of values at expiry, shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples.


@dataclass(frozen=True, slots=True)
class MultiAssetWorstOfEuropeanOptionMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for worst-of options.

    Notes
    -----
    - discounted_payoffs are already scaled by notional.
    """

    # --- Resolved inputs ---
    spots: np.ndarray  # Initial spots, shape (n_assets,).
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
    terminal_spots: np.ndarray  # Terminal spots, shape (n_paths_effective, n_assets).
    worst_values: np.ndarray  # Worst of values at expiry, shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples.


# =============================================================================
# Best-Of Pricer Class
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetBestOfEuropeanOptionMcPricer:
    """
    Monte Carlo pricer for European multi-asset best-of options.

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
        trade: MultiAssetBestOfEuropeanOption,
        spots: np.ndarray,
        r: float,
        dividends: np.ndarray,
        volatilities: np.ndarray,
        correlation: CorrelationMatrix,
    ) -> float:
        """
        Compute the present value of the best-of option.

        Parameters
        ----------
        trade : MultiAssetBestOfEuropeanOption
            The best-of option instrument.
        spots : np.ndarray
            Current spot prices, shape (n_assets,).
        r : float
            Risk-free rate.
        dividends : np.ndarray
            Dividend yields, shape (n_assets,).
        volatilities : np.ndarray
            Volatilities, shape (n_assets,).
        correlation : CorrelationMatrix
            Correlation structure.

        Returns
        -------
        float
            Present value scaled by notional.
        """
        sim = self.run(trade, spots, r, dividends, volatilities, correlation)
        return float(sim.discounted_payoffs.mean())

    def price_with_std_error(
        self,
        trade: MultiAssetBestOfEuropeanOption,
        spots: np.ndarray,
        r: float,
        dividends: np.ndarray,
        volatilities: np.ndarray,
        correlation: CorrelationMatrix,
    ) -> tuple[float, float]:
        """
        Compute price and standard error.

        Returns
        -------
        tuple[float, float]
            (price, standard_error) scaled by notional.
        """
        sim = self.run(trade, spots, r, dividends, volatilities, correlation)
        price = float(sim.discounted_payoffs.mean())
        std_error = float(sim.discounted_payoffs.std() / np.sqrt(sim.n_paths_effective))
        return price, std_error

    def run(
        self,
        trade: MultiAssetBestOfEuropeanOption,
        spots: np.ndarray,
        r: float,
        dividends: np.ndarray,
        volatilities: np.ndarray,
        correlation: CorrelationMatrix,
    ) -> MultiAssetBestOfEuropeanOptionMcSimulation:
        """
        Run the full simulation and return artifact with all details.
        """
        gbm = MultiAssetGBM(
            spots=spots,
            r=r,
            dividends=dividends,
            volatilities=volatilities,
            correlation=correlation,
        )

        terminals = gbm.simulate_terminal(
            maturity=trade.expiry,
            n_paths=self.n_paths,
            seed=self.seed,
            antithetic=self.antithetic,
        )

        n_paths_eff = terminals.shape[0]
        best_values = np.max(terminals, axis=1)

        if trade.option_type == "call":
            payoffs = np.maximum(best_values - trade.strike, 0)
        else:  # put
            payoffs = np.maximum(trade.strike - best_values, 0)

        discount = np.exp(-r * trade.expiry)
        discounted_payoffs = (discount * payoffs * trade.notional).astype(np.float64)

        return MultiAssetBestOfEuropeanOptionMcSimulation(
            spots=spots.copy(),
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
            best_values=best_values,
            discounted_payoffs=discounted_payoffs,
        )


# =============================================================================
# Worst-Of Pricer Class
# =============================================================================

@dataclass(frozen=True, slots=True)
class MultiAssetWorstOfEuropeanOptionMcPricer:
    """
    Monte Carlo pricer for European multi-asset worst-of options.

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
        trade: MultiAssetWorstOfEuropeanOption,
        spots: np.ndarray,
        r: float,
        dividends: np.ndarray,
        volatilities: np.ndarray,
        correlation: CorrelationMatrix,
    ) -> float:
        """
        Compute the present value of the worst-of option.

        Parameters
        ----------
        trade : MultiAssetWorstOfEuropeanOption
            The worst-of option instrument.
        spots : np.ndarray
            Current spot prices, shape (n_assets,).
        r : float
            Risk-free rate.
        dividends : np.ndarray
            Dividend yields, shape (n_assets,).
        volatilities : np.ndarray
            Volatilities, shape (n_assets,).
        correlation : CorrelationMatrix
            Correlation structure.

        Returns
        -------
        float
            Present value scaled by notional.
        """
        sim = self.run(trade, spots, r, dividends, volatilities, correlation)
        return float(sim.discounted_payoffs.mean())

    def price_with_std_error(
        self,
        trade: MultiAssetWorstOfEuropeanOption,
        spots: np.ndarray,
        r: float,
        dividends: np.ndarray,
        volatilities: np.ndarray,
        correlation: CorrelationMatrix,
    ) -> tuple[float, float]:
        """
        Compute price and standard error.

        Returns
        -------
        tuple[float, float]
            (price, standard_error) scaled by notional.
        """
        sim = self.run(trade, spots, r, dividends, volatilities, correlation)
        price = float(sim.discounted_payoffs.mean())
        std_error = float(sim.discounted_payoffs.std() / np.sqrt(sim.n_paths_effective))
        return price, std_error

    def run(
        self,
        trade: MultiAssetWorstOfEuropeanOption,
        spots: np.ndarray,
        r: float,
        dividends: np.ndarray,
        volatilities: np.ndarray,
        correlation: CorrelationMatrix,
    ) -> MultiAssetWorstOfEuropeanOptionMcSimulation:
        """
        Run the full simulation and return artifact with all details.
        """
        gbm = MultiAssetGBM(
            spots=spots,
            r=r,
            dividends=dividends,
            volatilities=volatilities,
            correlation=correlation,
        )

        terminals = gbm.simulate_terminal(
            maturity=trade.expiry,
            n_paths=self.n_paths,
            seed=self.seed,
            antithetic=self.antithetic,
        )

        n_paths_eff = terminals.shape[0]
        worst_values = np.min(terminals, axis=1)

        if trade.option_type == "call":
            payoffs = np.maximum(worst_values - trade.strike, 0)
        else:  # put
            payoffs = np.maximum(trade.strike - worst_values, 0)

        discount = np.exp(-r * trade.expiry)
        discounted_payoffs = (discount * payoffs * trade.notional).astype(np.float64)

        return MultiAssetWorstOfEuropeanOptionMcSimulation(
            spots=spots.copy(),
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
            worst_values=worst_values,
            discounted_payoffs=discounted_payoffs,
        )
