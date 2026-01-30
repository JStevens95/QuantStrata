# src/pricers/equity/european_mc.py
"""
Equity European Monte Carlo Pricers.

Monte Carlo simulation for European equity options under Black-Scholes with dividends:
- Vanilla (call/put)
- Barrier (knock-in/out)
- Asian (arithmetic/geometric averaging)
- Lookback (fixed/floating strike)

Author: QuantStrata Team
"""
from __future__ import annotations  # Enable forward references.

import math  # Standard math functions (log, exp, sqrt).
import numpy as np  # NumPy for fast vectorized Monte Carlo operations.
from dataclasses import dataclass  # Dataclasses for small immutable pricer/config objects.
from typing import Optional  # Optional for seed/paths.

from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption  # Equity vanilla instrument.
from src.instruments.equity.options.barrier import EuropeanEquityBarrierOption  # Equity barrier instrument.
from src.instruments.equity.options.asian import EuropeanEquityAsianOption  # Equity Asian instrument.
from src.instruments.equity.options.lookback import EuropeanEquityLookbackOption  # Equity lookback instrument.

from src.marketdata.core.market import Market  # Market snapshot interface.
from src.models.numeric.monte_carlo.rng import NormalRng  # Reproducible normal RNG.
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme  # GBM simulator and scheme.

from src.models.payoffs.types import OptionType, BarrierDirection, BarrierStyle  # Canonical type literals.
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff, require_path_payoff  # Payoff factory.


def _rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.

    Parameters
    ----------
    df : float
        Discount factor df(T).
    t : float
        Time to maturity T.

    Returns
    -------
    float
        Continuously compounded rate r = -ln(df)/T.
    """
    if t <= 0.0:  # At or before expiry, rate is zero.
        return 0.0
    if df <= 0.0:  # Discount factor must be positive.
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)  # Standard DF-to-rate conversion.


# ======================================================================================
# Simulation Artifacts (returned by run(...), so callers can plot/analyse without rerunning)
# ======================================================================================


@dataclass(frozen=True, slots=True)
class EquityMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single *vanilla* trade on a single Market snapshot.

    Notes
    -----
    - discounted_payoffs are in currency units and already scaled by trade.notional.
    - paths are optional and can be very large; prefer storing only a small subset.
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike K.
    maturity: float  # Expiry T.
    df: float  # Discount factor df(T).
    drift: float  # Risk-neutral drift (r - q).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    notional: float  # Notional scaling.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths (may be rounded up for antithetic).
    n_steps: int  # Number of time steps.
    scheme: GbmScheme  # GBM scheme.
    antithetic: bool  # Antithetic flag.
    seed: Optional[int]  # RNG seed.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots S(T), shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples, shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).


@dataclass(frozen=True, slots=True)
class EquityBarrierMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single *barrier* trade on a single Market snapshot.

    Notes
    -----
    - Barrier payoffs are path-dependent, so the simulation always *computes from paths*.
    - discounted_payoffs are in currency units and already scaled by trade.notional.
    - paths are optional in the returned artifact; prefer storing only a subset for plotting.
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike K.
    barrier_level: float  # Barrier level B.
    barrier_direction: BarrierDirection  # "up" or "down".
    barrier_style: BarrierStyle  # "knock_out" or "knock_in".
    rebate_amount: float  # Rebate paid at expiry (per unit notional).
    maturity: float  # Expiry T.
    df: float  # Discount factor df(T).
    drift: float  # Risk-neutral drift (r - q).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    notional: float  # Notional scaling.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths (may be rounded up for antithetic).
    n_steps: int  # Number of time steps (monitoring points = n_steps+1 including S0).
    scheme: GbmScheme  # GBM scheme.
    antithetic: bool  # Antithetic flag.
    seed: Optional[int]  # RNG seed.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots S(T), shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples, shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).


@dataclass(frozen=True, slots=True)
class EquityAsianMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single Asian option trade.

    This dataclass stores all inputs and outputs from a Monte Carlo simulation,
    allowing callers to analyze results without re-running the simulation.

    Notes
    -----
    - discounted_payoffs are in currency units and already scaled by trade.notional.
    - paths are optional and can be very large; prefer storing only a small subset.
    - average_spots stores the computed average for each path (useful for diagnostics).
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike price K.
    maturity: float  # Expiry T (year fraction).
    df: float  # Discount factor df(T).
    drift: float  # Risk-neutral drift (r - q).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    averaging_type: str  # "arithmetic" or "geometric".
    notional: float  # Notional scaling.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths (may be rounded up for antithetic).
    n_steps: int  # Number of time steps (monitoring points = n_steps + 1 including S0).
    scheme: GbmScheme  # GBM scheme ("exact" or "euler").
    antithetic: bool  # Antithetic variates flag.
    seed: Optional[int]  # RNG seed for reproducibility.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots S(T), shape (n_paths_effective,).
    average_spots: np.ndarray  # Average spots over each path, shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples, shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).


@dataclass(frozen=True, slots=True)
class EquityLookbackMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single lookback option trade.

    This dataclass stores all inputs and outputs from a Monte Carlo simulation,
    allowing callers to analyze results without re-running the simulation.

    Notes
    -----
    - discounted_payoffs are in currency units and already scaled by trade.notional.
    - paths are optional and can be very large; prefer storing only a small subset.
    - max_spots and min_spots store the computed extrema for each path (useful for diagnostics).
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike price K (used for fixed_strike only).
    maturity: float  # Expiry T (year fraction).
    df: float  # Discount factor df(T).
    drift: float  # Risk-neutral drift (r - q).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    lookback_type: str  # "floating_strike" or "fixed_strike".
    notional: float  # Notional scaling.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths (may be rounded up for antithetic).
    n_steps: int  # Number of time steps (monitoring points = n_steps + 1 including S0).
    scheme: GbmScheme  # GBM scheme ("exact" or "euler").
    antithetic: bool  # Antithetic variates flag.
    seed: Optional[int]  # RNG seed for reproducibility.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots S(T), shape (n_paths_effective,).
    max_spots: np.ndarray  # Maximum spots over each path, shape (n_paths_effective,).
    min_spots: np.ndarray  # Minimum spots over each path, shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples, shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).


# ======================================================================================
# Vanilla MC Pricer
# ======================================================================================


@dataclass(frozen=True, slots=True)
class EquityEuropeanVanillaMcPricer:
    """
    Monte Carlo pricer for European equity vanilla options under Black-Scholes with dividends.

    Model
    -----
    Simulates GBM under risk-neutral measure:

        dS = (r - q) S dt + σ S dW

    Where:
    - r = risk-free rate (from discount curve)
    - q = continuous dividend yield
    - σ = implied volatility

    Mapping
    -------
    - r = -ln(df)/T  (from discount curve)
    - drift = r - q  (cost-of-carry for equity)
    - PV = notional × df(T) × E[payoff(S_T)]

    Notes
    -----
    - Vanilla options require only terminal spot S(T), so n_steps=1 is sufficient.
    - Antithetic variates reduce variance by pairing paths with opposite Brownian increments.
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths.
    seed: Optional[int] = 42  # RNG seed for reproducibility.
    antithetic: bool = True  # Use antithetic variates to reduce variance.

    n_steps: int = 1  # Number of time steps (exact GBM allows 1 for vanilla).
    scheme: GbmScheme = "exact"  # Default to exact terminal sampling for GBM.

    def price(self, trade: EuropeanEquityVanillaOption, market: Market) -> float:
        """
        Price European equity vanilla option via Monte Carlo.

        Parameters
        ----------
        trade : EuropeanEquityVanillaOption
            The vanilla option instrument to price.
        market : Market
            Market snapshot containing spot, vol, and curves.

        Returns
        -------
        float
            Present value in currency units.
        """
        sim = self.run(trade, market, store_paths=False)  # Run once without storing paths.
        return float(sim.discounted_payoffs.mean())  # PV is the mean of discounted payoff samples.

    def run(
        self,
        trade: EuropeanEquityVanillaOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> EquityMcSimulation:
        """
        Run Monte Carlo simulation and return full simulation artifact.

        Parameters
        ----------
        trade : EuropeanEquityVanillaOption
            The vanilla option instrument to simulate.
        market : Market
            Market snapshot containing spot, vol, and curves.
        store_paths : bool, optional
            If True, store simulated paths in the returned artifact (default: False).
        paths_keep : int, optional
            Number of paths to keep if store_paths=True (0 = keep all, default: 0).

        Returns
        -------
        EquityMcSimulation
            Complete simulation artifact with inputs and outputs.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)  # Single source of truth.

    def sample_terminal_spots(self, trade: EuropeanEquityVanillaOption, market: Market) -> np.ndarray:
        """Return terminal spot distribution."""
        return self.run(trade, market, store_paths=False).terminal_spots  # Delegate to run().

    def sample_discounted_payoffs(self, trade: EuropeanEquityVanillaOption, market: Market) -> np.ndarray:
        """Return discounted payoff samples."""
        return self.run(trade, market, store_paths=False).discounted_payoffs  # Delegate to run().

    def _run_simulation(
        self,
        trade: EuropeanEquityVanillaOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> EquityMcSimulation:
        """
        Internal method that performs the actual Monte Carlo simulation.

        This is the single source of truth for simulation logic. All public methods
        delegate to this method to ensure consistency.
        """
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:  # Validate path count.
            raise ValueError("n_paths must be positive.")  # Fail fast.
        if self.n_steps <= 0:  # Validate step count.
            raise ValueError("n_steps must be positive.")  # Fail fast.

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type  # Read option type ("call" or "put").
        spot0 = float(market.quote(trade.spot_id))  # Read spot S0 from market.
        strike = float(trade.strike)  # Read strike K.
        maturity = float(trade.expiry)  # Read maturity T.
        notional = float(trade.notional)  # Read notional.
        q = float(trade.dividend_yield)  # Read continuous dividend yield.

        # Build payoff from payoff library (single responsibility: payoff definition lives there).
        payoff_fn = require_terminal_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Validate trade inputs
        # -----------------------------
        if maturity < 0.0:  # Validate maturity.
            raise ValueError("expiry must be >= 0.")  # Fail fast.
        if notional < 0.0:  # Validate notional.
            raise ValueError("notional must be >= 0.")  # Fail fast.
        if spot0 <= 0.0:  # Validate spot.
            raise ValueError("spot must be > 0.")  # Fail fast.

        # -----------------------------
        # Resolve discount factor and rate
        # -----------------------------
        df = float(market.curve(trade.curve_id).df(maturity))  # Discount factor from curve.
        r = _rate_from_df(df=df, t=maturity)  # Convert DF to continuously-compounded rate.

        # Risk-neutral drift for equity: r - q (cost-of-carry with dividend yield).
        drift = float(r - q)

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))  # Implied vol.
        if sigma < 0.0:  # Validate vol.
            raise ValueError("Implied vol must be non-negative.")  # Fail fast.

        # -----------------------------
        # Deterministic expiry handling
        # -----------------------------
        if maturity == 0.0:  # Handle T=0 case (intrinsic value).
            terminal_spots = np.array([spot0], dtype=np.float64)  # Terminal spot is spot0.
            payoff = payoff_fn.terminal(spot=terminal_spots)  # Compute terminal payoff vector.
            discounted_payoffs = (float(df) * payoff * notional).astype(np.float64, copy=False)  # Discount + scale.
            return EquityMcSimulation(
                spot0=spot0,
                strike=strike,
                maturity=maturity,
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
                paths=np.array([[spot0]], dtype=np.float64) if store_paths else None,
            )

        # -----------------------------
        # Generate standard normals
        # -----------------------------
        rng = NormalRng(seed=self.seed)  # Construct RNG with seed for reproducibility.
        normals = rng.standard_normals(  # Draw standard normals for the simulator.
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])  # Effective path count after antithetic adjustment.

        # -----------------------------
        # Simulate GBM paths
        # -----------------------------
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)  # Construct GBM simulator.
        all_paths = simulator.simulate_paths(  # Simulate paths.
            spot0=spot0,
            maturity=maturity,
            n_steps=self.n_steps,
            n_paths=n_paths_eff,
            normals=normals,
            scheme=self.scheme,
            dtype=np.float64,
        )

        # -----------------------------
        # Compute payoff
        # -----------------------------
        terminal_spots = all_paths[:, -1].copy()  # Extract terminal spots S(T).
        payoff = payoff_fn.terminal(spot=terminal_spots)  # Compute terminal payoff per path.
        discounted_payoffs = (float(df) * payoff * notional).astype(np.float64, copy=False)  # Discount + scale.

        # -----------------------------
        # Optionally retain paths for plotting
        # -----------------------------
        if not store_paths:  # If we are not storing paths...
            kept_paths = None  # ...store nothing.
        else:
            if paths_keep < 0:  # Validate keep count.
                raise ValueError("paths_keep must be >= 0.")  # Fail fast.
            if paths_keep == 0:  # Keep all paths if requested.
                kept_paths = all_paths.copy()  # Copy for safety.
            else:  # Otherwise keep a subset.
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()  # Copy only kept rows.

        return EquityMcSimulation(
            spot0=spot0,
            strike=strike,
            maturity=maturity,
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


# ======================================================================================
# Barrier MC Pricer
# ======================================================================================


@dataclass(frozen=True, slots=True)
class EquityEuropeanBarrierMcPricer:
    """
    Monte Carlo pricer for European equity single-barrier options under Black-Scholes with dividends.

    Monitoring (V1)
    ---------------
    - Discrete monitoring on simulated path points (including S0 and intermediate steps).
    - Continuous-monitoring adjustments (e.g., Brownian-bridge) can be added in future versions.

    Model
    -----
    Simulates GBM under risk-neutral measure:

        dS = (r - q) S dt + σ S dW

    Mapping
    -------
    - r = -ln(df)/T  (from discount curve)
    - drift = r - q  (cost-of-carry for equity)
    - PV = notional × df(T) × E[barrier_payoff(paths)]

    Barrier Types
    -------------
    - Up-and-Out: Knocked out if max(path) > barrier
    - Up-and-In: Knocked in if max(path) > barrier
    - Down-and-Out: Knocked out if min(path) < barrier
    - Down-and-In: Knocked in if min(path) < barrier

    Notes
    -----
    - Path-dependent: requires full simulated paths (not just terminal spots).
    - More monitoring points (n_steps) give better approximation to continuous barrier.
    - Discrete monitoring underestimates barrier hit probability vs continuous.
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths.
    seed: Optional[int] = 42  # RNG seed for reproducibility.
    antithetic: bool = True  # Antithetic variates.

    n_steps: int = 64  # Barrier needs monitoring points; choose a denser default than vanilla.
    scheme: GbmScheme = "exact"  # Exact GBM steps (exact per step) is typically preferred.

    def price(self, trade: EuropeanEquityBarrierOption, market: Market) -> float:
        """
        Price equity barrier option via Monte Carlo.

        Parameters
        ----------
        trade : EuropeanEquityBarrierOption
            The barrier option instrument to price.
        market : Market
            Market snapshot containing spot, vol, and curves.

        Returns
        -------
        float
            Present value in currency units.
        """
        sim = self.run(trade, market, store_paths=False)  # Run once without storing paths.
        return float(sim.discounted_payoffs.mean())  # PV is the mean of discounted payoff samples.

    def run(
        self,
        trade: EuropeanEquityBarrierOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> EquityBarrierMcSimulation:
        """
        Run Monte Carlo simulation and return full simulation artifact.

        Parameters
        ----------
        trade : EuropeanEquityBarrierOption
            The barrier option instrument to simulate.
        market : Market
            Market snapshot containing spot, vol, and curves.
        store_paths : bool, optional
            If True, store simulated paths in the returned artifact (default: False).
        paths_keep : int, optional
            Number of paths to keep if store_paths=True (0 = keep all, default: 0).

        Returns
        -------
        EquityBarrierMcSimulation
            Complete simulation artifact with inputs and outputs.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)  # Single source of truth.

    def sample_terminal_spots(self, trade: EuropeanEquityBarrierOption, market: Market) -> np.ndarray:
        """Return terminal spot distribution."""
        return self.run(trade, market, store_paths=False).terminal_spots  # Delegate to run().

    def sample_discounted_payoffs(self, trade: EuropeanEquityBarrierOption, market: Market) -> np.ndarray:
        """Return discounted payoff samples."""
        return self.run(trade, market, store_paths=False).discounted_payoffs  # Delegate to run().

    def _run_simulation(
        self,
        trade: EuropeanEquityBarrierOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> EquityBarrierMcSimulation:
        """
        Internal method that performs the actual Monte Carlo simulation.

        This is the single source of truth for simulation logic. All public methods
        delegate to this method to ensure consistency.
        """
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:  # Validate path count.
            raise ValueError("n_paths must be positive.")  # Fail fast.
        if self.n_steps <= 0:  # Validate step count.
            raise ValueError("n_steps must be positive (barriers need monitoring points).")  # Fail fast.

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type  # Read option type ("call" or "put").
        spot0 = float(market.quote(trade.spot_id))  # Read spot S0 from market.
        strike = float(trade.strike)  # Read strike K.
        maturity = float(trade.expiry)  # Read maturity T.
        notional = float(trade.notional)  # Read notional.
        q = float(trade.dividend_yield)  # Read continuous dividend yield.

        barrier_level = float(trade.barrier_level)  # Read barrier level B.
        barrier_direction: BarrierDirection = trade.barrier_direction  # Read barrier direction.
        barrier_style: BarrierStyle = trade.barrier_style  # Read barrier style.
        rebate_amount = float(trade.rebate_amount)  # Read rebate amount.

        # -----------------------------
        # Validate trade inputs
        # -----------------------------
        if maturity < 0.0:  # Validate maturity.
            raise ValueError("expiry must be >= 0.")  # Fail fast.
        if notional < 0.0:  # Validate notional.
            raise ValueError("notional must be >= 0.")  # Fail fast.
        if spot0 <= 0.0:  # Validate spot.
            raise ValueError("spot must be > 0.")  # Fail fast.
        if barrier_level <= 0.0:  # Validate barrier level.
            raise ValueError("barrier_level must be > 0.")  # Fail fast.
        if rebate_amount < 0.0:  # Validate rebate.
            raise ValueError("rebate_amount must be >= 0.")  # Fail fast.

        # -----------------------------
        # Resolve discount factor and rate
        # -----------------------------
        df = float(market.curve(trade.curve_id).df(maturity))  # Discount factor from curve.
        r = _rate_from_df(df=df, t=maturity)  # Convert DF to continuously-compounded rate.

        # Risk-neutral drift for equity: r - q (cost-of-carry with dividend yield).
        drift = float(r - q)

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))  # Implied vol.
        if sigma < 0.0:  # Validate vol.
            raise ValueError("Implied vol must be non-negative.")  # Fail fast.

        # -----------------------------
        # Build payoff from payoff library
        # -----------------------------
        # Build payoff via factory and enforce path-dependent contract.
        payoff_fn = require_path_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Deterministic expiry handling
        # -----------------------------
        if maturity == 0.0:  # Handle T=0 case.
            # With T=0, we treat the "path" as containing only S0.
            # Barrier hit is evaluated on that single observation (consistent with discrete monitoring).
            paths = np.array([[spot0]], dtype=np.float64)  # Single path with one point.
            terminal_spots = np.array([spot0], dtype=np.float64)  # Terminal equals initial.

            payoff = payoff_fn.terminal_from_paths(paths)  # Per-unit-notional payoff.
            discounted_payoffs = (float(df) * payoff * notional).astype(np.float64, copy=False)  # Discount + scale.

            return EquityBarrierMcSimulation(
                spot0=spot0,
                strike=strike,
                barrier_level=barrier_level,
                barrier_direction=barrier_direction,
                barrier_style=barrier_style,
                rebate_amount=rebate_amount,
                maturity=maturity,
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
                paths=paths.copy() if store_paths else None,
            )

        # -----------------------------
        # Generate standard normals
        # -----------------------------
        rng = NormalRng(seed=self.seed)  # Construct RNG with seed for reproducibility.
        normals = rng.standard_normals(  # Draw standard normals for the simulator.
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])  # Effective path count after antithetic adjustment.

        # -----------------------------
        # Simulate GBM paths
        # -----------------------------
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)  # Construct GBM simulator.
        all_paths = simulator.simulate_paths(  # Simulate paths.
            spot0=spot0,
            maturity=maturity,
            n_steps=self.n_steps,
            n_paths=n_paths_eff,
            normals=normals,
            scheme=self.scheme,
            dtype=np.float64,
        )

        # Terminal spot for each path (useful for plotting / diagnostics).
        terminal_spots = all_paths[:, -1].copy()

        # -----------------------------
        # Compute payoff from full path (path-dependent)
        # -----------------------------
        payoff = payoff_fn.terminal_from_paths(all_paths)  # Per-unit-notional payoff.
        discounted_payoffs = (float(df) * payoff * notional).astype(np.float64, copy=False)  # Discount + scale.

        # -----------------------------
        # Optionally retain paths for plotting
        # -----------------------------
        if not store_paths:  # If we are not storing paths...
            kept_paths = None  # ...store nothing.
        else:
            if paths_keep < 0:  # Validate keep count.
                raise ValueError("paths_keep must be >= 0.")  # Fail fast.
            if paths_keep == 0:  # Keep all paths if requested.
                kept_paths = all_paths.copy()  # Copy for safety.
            else:  # Otherwise keep a subset.
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()  # Copy only kept rows.

        return EquityBarrierMcSimulation(
            spot0=spot0,
            strike=strike,
            barrier_level=barrier_level,
            barrier_direction=barrier_direction,
            barrier_style=barrier_style,
            rebate_amount=rebate_amount,
            maturity=maturity,
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


# ======================================================================================
# Asian MC Pricer
# ======================================================================================


@dataclass(frozen=True, slots=True)
class EquityEuropeanAsianMcPricer:
    """
    Monte Carlo pricer for European equity Asian options under Black-Scholes with dividends.

    Asian options pay based on the average price over the option's life, rather than
    the terminal price. This reduces volatility and makes them cheaper than standard
    European options.

    Pricing Method
    --------------
    - Simulate GBM paths with discrete monitoring points
    - Compute average spot price over each path (arithmetic or geometric)
    - Apply vanilla-style payoff to the average
    - Discount and take expectation

    Mathematical Formula
    --------------------
    PV = notional × df(T) × E[ max(Avg(S_t) - K, 0) ]  (call)
    PV = notional × df(T) × E[ max(K - Avg(S_t), 0) ]  (put)

    Where:
    - Avg(S_t) = arithmetic or geometric mean over monitoring points
    - E[...] is the risk-neutral expectation
    - df(T) is the discount factor

    Mapping
    -------
    - r = -ln(df)/T  (from discount curve)
    - drift = r - q  (cost-of-carry for equity)
    - PV = notional × df(T) × E[payoff(paths)]

    Average Types
    -------------
    - Arithmetic: A = (1/n) × Σ S_i  (more common in practice)
    - Geometric: A = (∏ S_i)^(1/n)  (has closed-form solution)

    Notes
    -----
    - Path-dependent: requires full simulated paths (not just terminal spots).
    - Arithmetic averaging is more common but has no closed-form solution.
    - Geometric averaging has closed-form solutions but is less common in practice.
    - More monitoring points (n_steps) give better approximation to continuous averaging.
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths (higher = more accurate but slower).
    seed: Optional[int] = 42  # RNG seed for reproducibility (set to None for random).
    antithetic: bool = True  # Use antithetic variates to reduce variance (recommended).

    n_steps: int = 64  # Number of time steps (more steps = more monitoring points for averaging).
    scheme: GbmScheme = "exact"  # GBM scheme ("exact" recommended for GBM).

    def price(self, trade: EuropeanEquityAsianOption, market: Market) -> float:
        """
        Price an Asian option using Monte Carlo simulation.

        This is the main pricing method that returns the present value (PV) of the option.

        Parameters
        ----------
        trade : EuropeanEquityAsianOption
            The Asian option instrument to price.
        market : Market
            Market snapshot containing spot, vol, and curves.

        Returns
        -------
        float
            Present value in currency units.
        """
        # Run simulation without storing paths (faster for pricing only).
        sim = self.run(trade, market, store_paths=False)
        # PV is the sample mean of discounted payoffs.
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: EuropeanEquityAsianOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> EquityAsianMcSimulation:
        """
        Run Monte Carlo simulation and return full simulation artifact.

        This method performs the full simulation and returns a dataclass containing
        all inputs and outputs, useful for diagnostics, plotting, and analysis.

        Parameters
        ----------
        trade : EuropeanEquityAsianOption
            The Asian option instrument to simulate.
        market : Market
            Market snapshot containing spot, vol, and curves.
        store_paths : bool, optional
            If True, store simulated paths in the returned artifact (default: False).
        paths_keep : int, optional
            Number of paths to keep if store_paths=True (0 = keep all, default: 0).

        Returns
        -------
        EquityAsianMcSimulation
            Complete simulation artifact with inputs and outputs.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)  # Single source of truth.

    def sample_terminal_spots(self, trade: EuropeanEquityAsianOption, market: Market) -> np.ndarray:
        """
        Sample terminal spot prices from the simulation.

        Useful for diagnostics and plotting the distribution of terminal spots.

        Parameters
        ----------
        trade : EuropeanEquityAsianOption
            The Asian option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Terminal spot prices, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).terminal_spots  # Delegate to run().

    def sample_average_spots(self, trade: EuropeanEquityAsianOption, market: Market) -> np.ndarray:
        """
        Sample average spot prices from the simulation.

        Useful for diagnostics and understanding the distribution of averages.

        Parameters
        ----------
        trade : EuropeanEquityAsianOption
            The Asian option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Average spot prices, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).average_spots  # Delegate to run().

    def sample_discounted_payoffs(self, trade: EuropeanEquityAsianOption, market: Market) -> np.ndarray:
        """
        Sample discounted payoffs from the simulation.

        Useful for computing confidence intervals and understanding payoff distribution.

        Parameters
        ----------
        trade : EuropeanEquityAsianOption
            The Asian option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Discounted payoffs, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).discounted_payoffs  # Delegate to run().

    def _run_simulation(
        self,
        trade: EuropeanEquityAsianOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> EquityAsianMcSimulation:
        """
        Internal method that performs the actual Monte Carlo simulation.

        This is the single source of truth for simulation logic. All public methods
        delegate to this method to ensure consistency.

        Parameters
        ----------
        trade : EuropeanEquityAsianOption
            The Asian option instrument to simulate.
        market : Market
            Market snapshot.
        store_paths : bool
            Whether to store paths in the returned artifact.
        paths_keep : int
            Number of paths to keep if storing (0 = all).

        Returns
        -------
        EquityAsianMcSimulation
            Complete simulation artifact.
        """
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:  # Ensure positive path count.
            raise ValueError("n_paths must be positive.")  # Fail fast.
        if self.n_steps <= 0:  # Ensure positive step count (needed for averaging).
            raise ValueError("n_steps must be positive (Asian options need monitoring points).")  # Fail fast.

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type  # "call" or "put".
        spot0 = float(market.quote(trade.spot_id))  # Initial spot S0.
        strike = float(trade.strike)  # Strike price K.
        maturity = float(trade.expiry)  # Time to expiry T.
        notional = float(trade.notional)  # Notional amount.
        q = float(trade.dividend_yield)  # Continuous dividend yield.
        averaging_type = str(trade.averaging_type)  # "arithmetic" or "geometric".

        # -----------------------------
        # Validate trade inputs
        # -----------------------------
        if maturity < 0.0:  # Expiry cannot be negative.
            raise ValueError("expiry must be >= 0.")  # Fail fast.
        if notional < 0.0:  # Notional cannot be negative.
            raise ValueError("notional must be >= 0.")  # Fail fast.
        if spot0 <= 0.0:  # Spot must be positive.
            raise ValueError("spot must be > 0.")  # Fail fast.

        # -----------------------------
        # Resolve discount factor and rate
        # -----------------------------
        df = float(market.curve(trade.curve_id).df(maturity))  # Discount factor from curve.
        r = _rate_from_df(df=df, t=maturity)  # Convert DF to continuously-compounded rate.

        # Risk-neutral drift for equity: r - q (cost-of-carry with dividend yield).
        drift = float(r - q)

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))  # Implied vol.
        if sigma < 0.0:  # Volatility cannot be negative.
            raise ValueError("Implied vol must be non-negative.")  # Fail fast.

        # -----------------------------
        # Build payoff from payoff library
        # -----------------------------
        # Build payoff via factory and enforce path-dependent contract.
        # The payoff library is the single source of truth for payoff logic.
        payoff_fn = require_path_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Deterministic expiry handling
        # -----------------------------
        if maturity == 0.0:  # Handle T=0 case.
            # At expiry, the "path" contains only S0 (which equals S_T).
            # The average is just S0, and payoff is computed on that.
            paths = np.array([[spot0]], dtype=np.float64)  # Single path with one point.
            terminal_spots = np.array([spot0], dtype=np.float64)  # Terminal equals initial.

            # Compute payoff from paths (average will be S0).
            payoff = payoff_fn.terminal_from_paths(paths)  # Per-unit-notional payoff.
            discounted_payoffs = (float(df) * payoff * notional).astype(np.float64, copy=False)  # Discount + scale.

            # Compute average for diagnostics (will be S0).
            average_spots = np.array([spot0], dtype=np.float64)

            return EquityAsianMcSimulation(
                spot0=spot0,
                strike=strike,
                maturity=maturity,
                df=df,
                drift=drift,
                sigma=sigma,
                option_type=option_type,
                averaging_type=averaging_type,
                notional=notional,
                n_paths_requested=self.n_paths,
                n_paths_effective=1,
                n_steps=0,
                scheme=self.scheme,
                antithetic=self.antithetic,
                seed=self.seed,
                terminal_spots=terminal_spots,
                average_spots=average_spots,
                discounted_payoffs=discounted_payoffs,
                paths=paths.copy() if store_paths else None,
            )

        # -----------------------------
        # Generate standard normals
        # -----------------------------
        rng = NormalRng(seed=self.seed)  # Construct RNG with seed for reproducibility.
        normals = rng.standard_normals(  # Draw standard normals for the simulator.
            self.n_paths,  # Number of paths requested.
            self.n_steps,  # Number of steps per path.
            antithetic=self.antithetic,  # Use antithetic variates if enabled.
            dtype=np.float64,  # Use float64 for numerical precision.
        )
        n_paths_eff = int(normals.shape[0])  # Effective path count (may be doubled if antithetic).

        # -----------------------------
        # Simulate GBM paths
        # -----------------------------
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)  # Construct GBM simulator.
        all_paths = simulator.simulate_paths(  # Simulate paths.
            spot0=spot0,  # Initial spot.
            maturity=maturity,  # Time to expiry.
            n_steps=self.n_steps,  # Number of time steps.
            n_paths=n_paths_eff,  # Number of paths.
            normals=normals,  # Pre-generated random normals.
            scheme=self.scheme,  # GBM scheme ("exact" or "euler").
            dtype=np.float64,  # Use float64 for precision.
        )

        # Extract terminal spots for diagnostics (last column of paths).
        terminal_spots = all_paths[:, -1].copy()

        # -----------------------------
        # Compute payoff from full path (path-dependent)
        # -----------------------------
        # The payoff function computes the average over each path and applies vanilla payoff.
        payoff = payoff_fn.terminal_from_paths(all_paths)  # Per-unit-notional payoff.

        # Discount and scale by notional.
        # PV = notional × df(T) × E[payoff].
        discounted_payoffs = (float(df) * payoff * notional).astype(np.float64, copy=False)

        # Compute average spots for diagnostics.
        # For diagnostics, we compute the average directly.
        if averaging_type == "arithmetic":  # Arithmetic average: A = (1/n) × Σ S_i.
            average_spots = np.mean(all_paths, axis=1, dtype=np.float64)
        else:  # Geometric average: A = exp((1/n) × Σ ln(S_i)).
            log_paths = np.log(all_paths)
            log_mean = np.mean(log_paths, axis=1, dtype=np.float64)
            average_spots = np.exp(log_mean)

        # -----------------------------
        # Optionally retain paths for plotting
        # -----------------------------
        if not store_paths:  # If not storing paths...
            kept_paths = None  # ...store nothing.
        else:
            if paths_keep < 0:  # Validate keep count.
                raise ValueError("paths_keep must be >= 0.")  # Fail fast.
            if paths_keep == 0:  # Store all if requested.
                kept_paths = all_paths.copy()  # Copy for safety.
            else:  # Store subset otherwise.
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()  # Copy subset.

        return EquityAsianMcSimulation(
            spot0=spot0,
            strike=strike,
            maturity=maturity,
            df=df,
            drift=drift,
            sigma=sigma,
            option_type=option_type,
            averaging_type=averaging_type,
            notional=notional,
            n_paths_requested=self.n_paths,
            n_paths_effective=n_paths_eff,
            n_steps=self.n_steps,
            scheme=self.scheme,
            antithetic=self.antithetic,
            seed=self.seed,
            terminal_spots=terminal_spots,
            average_spots=average_spots,
            discounted_payoffs=discounted_payoffs,
            paths=kept_paths,
        )


# ======================================================================================
# Lookback MC Pricer
# ======================================================================================


@dataclass(frozen=True, slots=True)
class EquityEuropeanLookbackMcPricer:
    """
    Monte Carlo pricer for European equity lookback options under Black-Scholes with dividends.

    Lookback options give the holder the benefit of hindsight—the payoff depends on the
    maximum or minimum spot price achieved over the option's life.

    Pricing Method
    --------------
    - Simulate GBM paths with discrete monitoring points
    - Track maximum and minimum spot prices over each path
    - Apply lookback payoff using path extrema
    - Discount and take expectation

    Mathematical Formula
    --------------------
    Fixed Strike:
    - Call: PV = notional × df(T) × E[ max(S_max - K, 0) ]
    - Put:  PV = notional × df(T) × E[ max(K - S_min, 0) ]

    Floating Strike:
    - Call: PV = notional × df(T) × E[ S_T - S_min ]  (always ITM)
    - Put:  PV = notional × df(T) × E[ S_max - S_T ]  (always ITM)

    Mapping
    -------
    - r = -ln(df)/T  (from discount curve)
    - drift = r - q  (cost-of-carry for equity)
    - PV = notional × df(T) × E[payoff(paths)]

    Lookback Types
    --------------
    - fixed_strike: Uses strike K, payoff on path extremum
    - floating_strike: Strike is set to path extremum, always ITM

    Notes
    -----
    - Path-dependent: requires full simulated paths (not just terminal spots).
    - Closed-form solutions exist for continuous monitoring.
    - MC with discrete monitoring underestimates extremes (discretization bias).
    - Higher n_steps improves accuracy but increases computation.
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths.
    seed: Optional[int] = 42  # RNG seed for reproducibility.
    antithetic: bool = True  # Antithetic variates.

    n_steps: int = 252  # Daily monitoring for 1 year (dense monitoring for extrema).
    scheme: GbmScheme = "exact"  # GBM scheme ("exact" recommended).

    def price(self, trade: EuropeanEquityLookbackOption, market: Market) -> float:
        """
        Price a lookback option using Monte Carlo simulation.

        This is the main pricing method that returns the present value (PV) of the option.

        Parameters
        ----------
        trade : EuropeanEquityLookbackOption
            The lookback option instrument to price.
        market : Market
            Market snapshot containing spot, vol, and curves.

        Returns
        -------
        float
            Present value in currency units.
        """
        # Run simulation without storing paths (faster for pricing only).
        sim = self.run(trade, market, store_paths=False)
        # PV is the sample mean of discounted payoffs.
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: EuropeanEquityLookbackOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> EquityLookbackMcSimulation:
        """
        Run Monte Carlo simulation and return full simulation artifact.

        This method performs the full simulation and returns a dataclass containing
        all inputs and outputs, useful for diagnostics, plotting, and analysis.

        Parameters
        ----------
        trade : EuropeanEquityLookbackOption
            The lookback option instrument to simulate.
        market : Market
            Market snapshot containing spot, vol, and curves.
        store_paths : bool, optional
            If True, store simulated paths in the returned artifact (default: False).
        paths_keep : int, optional
            Number of paths to keep if store_paths=True (0 = keep all, default: 0).

        Returns
        -------
        EquityLookbackMcSimulation
            Complete simulation artifact with inputs and outputs.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)  # Single source of truth.

    def sample_terminal_spots(self, trade: EuropeanEquityLookbackOption, market: Market) -> np.ndarray:
        """
        Sample terminal spot prices from the simulation.

        Useful for diagnostics and plotting the distribution of terminal spots.

        Parameters
        ----------
        trade : EuropeanEquityLookbackOption
            The lookback option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Terminal spot prices, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).terminal_spots  # Delegate to run().

    def sample_max_spots(self, trade: EuropeanEquityLookbackOption, market: Market) -> np.ndarray:
        """
        Sample maximum spot prices from the simulation.

        Useful for diagnostics and understanding the distribution of path maxima.

        Parameters
        ----------
        trade : EuropeanEquityLookbackOption
            The lookback option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Maximum spot prices, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).max_spots  # Delegate to run().

    def sample_min_spots(self, trade: EuropeanEquityLookbackOption, market: Market) -> np.ndarray:
        """
        Sample minimum spot prices from the simulation.

        Useful for diagnostics and understanding the distribution of path minima.

        Parameters
        ----------
        trade : EuropeanEquityLookbackOption
            The lookback option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Minimum spot prices, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).min_spots  # Delegate to run().

    def sample_discounted_payoffs(self, trade: EuropeanEquityLookbackOption, market: Market) -> np.ndarray:
        """
        Sample discounted payoffs from the simulation.

        Useful for computing confidence intervals and understanding payoff distribution.

        Parameters
        ----------
        trade : EuropeanEquityLookbackOption
            The lookback option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Discounted payoffs, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).discounted_payoffs  # Delegate to run().

    def _run_simulation(
        self,
        trade: EuropeanEquityLookbackOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> EquityLookbackMcSimulation:
        """
        Internal method that performs the actual Monte Carlo simulation.

        This is the single source of truth for simulation logic. All public methods
        delegate to this method to ensure consistency.

        Parameters
        ----------
        trade : EuropeanEquityLookbackOption
            The lookback option instrument to simulate.
        market : Market
            Market snapshot.
        store_paths : bool
            Whether to store paths in the returned artifact.
        paths_keep : int
            Number of paths to keep if storing (0 = all).

        Returns
        -------
        EquityLookbackMcSimulation
            Complete simulation artifact.
        """
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:  # Ensure positive path count.
            raise ValueError("n_paths must be positive.")  # Fail fast.
        if self.n_steps <= 0:  # Ensure positive step count (needed for extrema).
            raise ValueError("n_steps must be positive (lookbacks need monitoring points).")  # Fail fast.

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type  # "call" or "put".
        lookback_type = str(trade.lookback_type)  # "floating_strike" or "fixed_strike".
        spot0 = float(market.quote(trade.spot_id))  # Initial spot S0.
        strike = float(trade.strike)  # Strike price K (used for fixed_strike only).
        maturity = float(trade.expiry)  # Time to expiry T.
        notional = float(trade.notional)  # Notional amount.
        q = float(trade.dividend_yield)  # Continuous dividend yield.

        # -----------------------------
        # Validate trade inputs
        # -----------------------------
        if maturity < 0.0:  # Expiry cannot be negative.
            raise ValueError("expiry must be >= 0.")  # Fail fast.
        if notional < 0.0:  # Notional cannot be negative.
            raise ValueError("notional must be >= 0.")  # Fail fast.
        if spot0 <= 0.0:  # Spot must be positive.
            raise ValueError("spot must be > 0.")  # Fail fast.

        # -----------------------------
        # Resolve discount factor and rate
        # -----------------------------
        df = float(market.curve(trade.curve_id).df(maturity))  # Discount factor from curve.
        r = _rate_from_df(df=df, t=maturity)  # Convert DF to continuously-compounded rate.

        # Risk-neutral drift for equity: r - q (cost-of-carry with dividend yield).
        drift = float(r - q)

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))  # Implied vol.
        if sigma < 0.0:  # Volatility cannot be negative.
            raise ValueError("Implied vol must be non-negative.")  # Fail fast.

        # -----------------------------
        # Build payoff from payoff library
        # -----------------------------
        # Build payoff via factory and enforce path-dependent contract.
        payoff_fn = require_path_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Deterministic expiry handling
        # -----------------------------
        if maturity == 0.0:  # Handle T=0 case.
            # At expiry, the "path" contains only S0.
            # max = min = S0, payoff is computed on that.
            paths = np.array([[spot0]], dtype=np.float64)  # Single path with one point.
            terminal_spots = np.array([spot0], dtype=np.float64)  # Terminal equals initial.
            max_spots = np.array([spot0], dtype=np.float64)  # Max is S0.
            min_spots = np.array([spot0], dtype=np.float64)  # Min is S0.

            # Compute payoff from paths.
            payoff = payoff_fn.terminal_from_paths(paths)  # Per-unit-notional payoff.
            discounted_payoffs = (float(df) * payoff * notional).astype(np.float64, copy=False)  # Discount + scale.

            return EquityLookbackMcSimulation(
                spot0=spot0,
                strike=strike,
                maturity=maturity,
                df=df,
                drift=drift,
                sigma=sigma,
                option_type=option_type,
                lookback_type=lookback_type,
                notional=notional,
                n_paths_requested=self.n_paths,
                n_paths_effective=1,
                n_steps=0,
                scheme=self.scheme,
                antithetic=self.antithetic,
                seed=self.seed,
                terminal_spots=terminal_spots,
                max_spots=max_spots,
                min_spots=min_spots,
                discounted_payoffs=discounted_payoffs,
                paths=paths.copy() if store_paths else None,
            )

        # -----------------------------
        # Generate standard normals
        # -----------------------------
        rng = NormalRng(seed=self.seed)  # Construct RNG with seed for reproducibility.
        normals = rng.standard_normals(  # Draw standard normals for the simulator.
            self.n_paths,  # Number of paths requested.
            self.n_steps,  # Number of steps per path.
            antithetic=self.antithetic,  # Use antithetic variates if enabled.
            dtype=np.float64,  # Use float64 for numerical precision.
        )
        n_paths_eff = int(normals.shape[0])  # Effective path count (may be doubled if antithetic).

        # -----------------------------
        # Simulate GBM paths
        # -----------------------------
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)  # Construct GBM simulator.
        all_paths = simulator.simulate_paths(  # Simulate paths.
            spot0=spot0,  # Initial spot.
            maturity=maturity,  # Time to expiry.
            n_steps=self.n_steps,  # Number of time steps.
            n_paths=n_paths_eff,  # Number of paths.
            normals=normals,  # Pre-generated random normals.
            scheme=self.scheme,  # GBM scheme ("exact" or "euler").
            dtype=np.float64,  # Use float64 for precision.
        )

        # Extract terminal spots for diagnostics (last column of paths).
        terminal_spots = all_paths[:, -1].copy()

        # Compute path extrema for diagnostics.
        max_spots = np.max(all_paths, axis=1)  # Maximum spot over each path.
        min_spots = np.min(all_paths, axis=1)  # Minimum spot over each path.

        # -----------------------------
        # Compute payoff from full path (path-dependent)
        # -----------------------------
        # The payoff function computes the lookback payoff using path extrema.
        payoff = payoff_fn.terminal_from_paths(all_paths)  # Per-unit-notional payoff.

        # Discount and scale by notional.
        # PV = notional × df(T) × E[payoff].
        discounted_payoffs = (float(df) * payoff * notional).astype(np.float64, copy=False)

        # -----------------------------
        # Optionally retain paths for plotting
        # -----------------------------
        if not store_paths:  # If not storing paths...
            kept_paths = None  # ...store nothing.
        else:
            if paths_keep < 0:  # Validate keep count.
                raise ValueError("paths_keep must be >= 0.")  # Fail fast.
            if paths_keep == 0:  # Store all if requested.
                kept_paths = all_paths.copy()  # Copy for safety.
            else:  # Store subset otherwise.
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()  # Copy subset.

        return EquityLookbackMcSimulation(
            spot0=spot0,
            strike=strike,
            maturity=maturity,
            df=df,
            drift=drift,
            sigma=sigma,
            option_type=option_type,
            lookback_type=lookback_type,
            notional=notional,
            n_paths_requested=self.n_paths,
            n_paths_effective=n_paths_eff,
            n_steps=self.n_steps,
            scheme=self.scheme,
            antithetic=self.antithetic,
            seed=self.seed,
            terminal_spots=terminal_spots,
            max_spots=max_spots,
            min_spots=min_spots,
            discounted_payoffs=discounted_payoffs,
            paths=kept_paths,
        )
