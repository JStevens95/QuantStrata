"""
Equity European Vanilla Monte Carlo Pricer

Monte Carlo simulation for European equity options.

Author: QuantStrata Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption
from src.marketdata.core.market import Market
from src.models.numeric.monte_carlo.rng import NormalRng
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff
from src.models.payoffs.types import OptionType


def _rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.

    Parameters
    ----------
    df : float
        Discount factor
    t : float
        Time to maturity

    Returns
    -------
    float
        Continuously compounded rate
    """
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


@dataclass(frozen=True, slots=True)
class EquityMcSimulation:
    """
    Monte Carlo simulation artifact for equity vanilla options.

    Contains all inputs and outputs for analysis and diagnostics.

    Attributes
    ----------
    spot0 : float
        Initial spot price
    strike : float
        Strike price
    maturity : float
        Time to expiry
    df : float
        Discount factor
    drift : float
        Risk-neutral drift (r - q)
    sigma : float
        Volatility
    option_type : OptionType
        "call" or "put"
    notional : float
        Contract notional
    n_paths_requested : int
        Requested number of paths
    n_paths_effective : int
        Actual paths (may differ with antithetic)
    n_steps : int
        Number of time steps
    scheme : GbmScheme
        GBM discretization scheme
    antithetic : bool
        Whether antithetic variates used
    seed : Optional[int]
        RNG seed
    terminal_spots : np.ndarray
        Terminal spot values, shape (n_paths,)
    discounted_payoffs : np.ndarray
        Discounted payoffs, shape (n_paths,)
    paths : Optional[np.ndarray]
        Full paths if stored, shape (n_paths, n_steps+1)
    """

    # Resolved inputs
    spot0: float
    strike: float
    maturity: float
    df: float
    drift: float
    sigma: float
    option_type: OptionType
    notional: float

    # Simulation settings
    n_paths_requested: int
    n_paths_effective: int
    n_steps: int
    scheme: GbmScheme
    antithetic: bool
    seed: Optional[int]

    # Outputs
    terminal_spots: np.ndarray
    discounted_payoffs: np.ndarray
    paths: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class EquityEuropeanVanillaMcPricer:
    """
    Monte Carlo pricer for European equity vanilla options.

    Model
    -----
    Simulates GBM under risk-neutral measure:

        dS = (r - q) S dt + σ S dW

    Where:
    - r = risk-free rate (from discount curve)
    - q = continuous dividend yield
    - σ = implied volatility

    Pricing
    -------
    PV = notional × df(T) × E[payoff(S_T)]

    The expectation is estimated by Monte Carlo simulation.

    Variance Reduction
    ------------------
    - Antithetic variates: For each path with increments Z, also simulate -Z
    - This reduces variance by exploiting symmetry

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths (default: 200,000)
    seed : Optional[int]
        RNG seed for reproducibility
    antithetic : bool
        Use antithetic variates (default: True)
    n_steps : int
        Time steps per path (default: 1 for European)
    scheme : GbmScheme
        GBM discretization: "exact", "euler", or "milstein"
    """

    n_paths: int = 200_000
    seed: Optional[int] = 42
    antithetic: bool = True
    n_steps: int = 1  # For European vanilla, 1 step is sufficient
    scheme: GbmScheme = "exact"

    def price(self, trade: EuropeanEquityVanillaOption, market: Market) -> float:
        """
        Price European equity vanilla option via Monte Carlo.

        Parameters
        ----------
        trade : EuropeanEquityVanillaOption
            Option to price
        market : Market
            Market snapshot

        Returns
        -------
        float
            Present value
        """
        sim = self.run(trade, market, store_paths=False)
        return float(sim.discounted_payoffs.mean())

    def run(
            self,
            trade: EuropeanEquityVanillaOption,
            market: Market,
            *,
            store_paths: bool = False,
            paths_keep: int = 0,
    ) -> EquityMcSimulation:
        """
        Run Monte Carlo simulation and return full results.

        Parameters
        ----------
        trade : EuropeanEquityVanillaOption
            Option to simulate
        market : Market
            Market snapshot
        store_paths : bool
            Whether to store full paths (memory intensive)
        paths_keep : int
            If >0, only keep first N paths (0 = keep all if storing)

        Returns
        -------
        EquityMcSimulation
            Simulation artifact with all inputs and outputs
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)

    def sample_terminal_spots(self, trade: EuropeanEquityVanillaOption, market: Market) -> np.ndarray:
        """Return terminal spot distribution."""
        return self.run(trade, market, store_paths=False).terminal_spots

    def sample_discounted_payoffs(self, trade: EuropeanEquityVanillaOption, market: Market) -> np.ndarray:
        """Return discounted payoff samples."""
        return self.run(trade, market, store_paths=False).discounted_payoffs

    def _run_simulation(
            self,
            trade: EuropeanEquityVanillaOption,
            market: Market,
            *,
            store_paths: bool,
            paths_keep: int,
    ) -> EquityMcSimulation:
        """Core simulation logic."""

        # Validate pricer settings
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        # Read trade inputs
        option_type: OptionType = trade.option_type
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)

        # Build payoff from payoff library
        payoff_fn = require_terminal_payoff(build_payoff_1d(trade))

        # Validate
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")

        # Get discount factor and rate
        if T == 0.0:
            df = 1.0
            r = 0.0
        else:
            df = float(market.curve(trade.curve_id).df(T))
            r = _rate_from_df(df=df, t=T)

        # Risk-neutral drift for equity: r - q
        drift = float(r - q)

        # Get implied volatility
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Handle degenerate case: T = 0
        if T == 0.0:
            terminal_spots = np.array([S0], dtype=np.float64)
            payoff = payoff_fn.terminal(spot=terminal_spots)
            discounted_payoffs = (df * payoff * notional).astype(np.float64, copy=False)

            return EquityMcSimulation(
                spot0=S0,
                strike=K,
                maturity=T,
                df=df,
                drift=drift,
                sigma=sigma,
                option_type=option_type,
                notional=notional,
                n_paths_requested=self.n_paths,
                n_paths_effective=1,
                n_steps=0,
                scheme=self.scheme,
                antithetic=self.antithetic,
                seed=self.seed,
                terminal_spots=terminal_spots,
                discounted_payoffs=discounted_payoffs,
                paths=np.array([[S0]], dtype=np.float64) if store_paths else None,
            )

        # Generate random numbers
        rng = NormalRng(seed=self.seed)
        normals = rng.standard_normals(
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])

        # Simulate GBM paths
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)
        all_paths = simulator.simulate_paths(
            spot0=S0,
            maturity=T,
            n_steps=self.n_steps,
            n_paths=n_paths_eff,
            normals=normals,
            scheme=self.scheme,
            dtype=np.float64,
        )

        # Extract terminal spots
        terminal_spots = all_paths[:, -1].copy()

        # Compute payoffs
        payoff = payoff_fn.terminal(spot=terminal_spots)
        discounted_payoffs = (df * payoff * notional).astype(np.float64, copy=False)

        # Handle path storage
        if not store_paths:
            kept_paths = None
        else:
            if paths_keep < 0:
                raise ValueError("paths_keep must be >= 0.")
            if paths_keep == 0:
                kept_paths = all_paths.copy()
            else:
                kept_paths = all_paths[:min(paths_keep, n_paths_eff), :].copy()

        return EquityMcSimulation(
            spot0=S0,
            strike=K,
            maturity=T,
            df=df,
            drift=drift,
            sigma=sigma,
            option_type=option_type,
            notional=notional,
            n_paths_requested=self.n_paths,
            n_paths_effective=n_paths_eff,
            n_steps=self.n_steps,
            scheme=self.scheme,
            antithetic=self.antithetic,
            seed=self.seed,
            terminal_spots=terminal_spots,
            discounted_payoffs=discounted_payoffs,
            paths=kept_paths,
        )
