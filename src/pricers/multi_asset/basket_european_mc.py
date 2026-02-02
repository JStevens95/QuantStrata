"""
Basket Option Monte Carlo Pricers.

This module provides Monte Carlo pricing for basket options using pricer classes
consistent with FX pricer conventions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.instruments.multi_asset.basket import MultiAssetBasketEuropeanOption
from src.models.numeric.monte_carlo.multi_asset import (
    CorrelationMatrix,
    MultiAssetGBM,
)


@dataclass(frozen=True, slots=True)
class MultiAssetBasketEuropeanOptionMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for basket options.

    Notes
    -----
    - discounted_payoffs are already scaled by notional.
    - terminal_spots shape is (n_paths_effective, n_assets).
    """

    # --- Resolved inputs ---
    spots: np.ndarray  # Initial spots, shape (n_assets,).
    strike: float  # Strike K.
    maturity: float  # Expiry T.
    r: float  # Risk-free rate.
    weights: np.ndarray  # Portfolio weights.
    option_type: str  # "call" or "put".
    notional: float  # Notional scaling.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths (may be doubled for antithetic).
    antithetic: bool  # Antithetic flag.
    seed: Optional[int]  # RNG seed.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots, shape (n_paths_effective, n_assets).
    basket_values: np.ndarray  # Basket values at expiry, shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples, shape (n_paths_effective,).


@dataclass(frozen=True, slots=True)
class MultiAssetBasketEuropeanOptionMcPricer:
    """
    Monte Carlo pricer for European multi-asset basket options.

    Consistent with FX pricer class conventions (e.g., FxVanillaEuropeanOptionMcPricer).

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
        trade: MultiAssetBasketEuropeanOption,
        spots: np.ndarray,
        r: float,
        dividends: np.ndarray,
        volatilities: np.ndarray,
        correlation: CorrelationMatrix,
    ) -> float:
        """
        Compute the present value of the basket option.

        Parameters
        ----------
        trade : MultiAssetBasketEuropeanOption
            The basket option instrument.
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
        trade: MultiAssetBasketEuropeanOption,
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
        trade: MultiAssetBasketEuropeanOption,
        spots: np.ndarray,
        r: float,
        dividends: np.ndarray,
        volatilities: np.ndarray,
        correlation: CorrelationMatrix,
    ) -> MultiAssetBasketEuropeanOptionMcSimulation:
        """
        Run the full simulation and return artifact with all details.

        Parameters
        ----------
        trade : MultiAssetBasketEuropeanOption
            The basket option instrument.
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
        MultiAssetBasketEuropeanOptionMcSimulation
            Simulation artifact with all inputs, settings, and outputs.
        """
        weights = np.array(trade.weights)

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
        basket_values = terminals @ weights

        if trade.option_type == "call":
            payoffs = np.maximum(basket_values - trade.strike, 0)
        else:  # put
            payoffs = np.maximum(trade.strike - basket_values, 0)

        discount = np.exp(-r * trade.expiry)
        discounted_payoffs = (discount * payoffs * trade.notional).astype(np.float64)

        return MultiAssetBasketEuropeanOptionMcSimulation(
            spots=spots.copy(),
            strike=trade.strike,
            maturity=trade.expiry,
            r=r,
            weights=weights.copy(),
            option_type=trade.option_type,
            notional=trade.notional,
            n_paths_requested=self.n_paths,
            n_paths_effective=n_paths_eff,
            antithetic=self.antithetic,
            seed=self.seed,
            terminal_spots=terminals,
            basket_values=basket_values,
            discounted_payoffs=discounted_payoffs,
        )
