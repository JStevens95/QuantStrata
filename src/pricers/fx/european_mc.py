# src/pricers/fx/european_mc.py
from __future__ import annotations  # Enable forward references.

import numpy as np  # NumPy for fast vectorized Monte Carlo operations.
from dataclasses import dataclass  # Dataclasses for small immutable pricer/config objects.
from typing import Literal, Optional  # Literal for payoff-type tags; Optional for seed/paths.

from src.instruments.core.types import TouchStyle
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption  # FX vanilla instrument.
from src.instruments.fx.options.digital import EuropeanFxDigitalOption  # FX digital instrument.
from src.instruments.fx.options.barrier import EuropeanFxBarrierOption  # FX barrier instrument.
from src.instruments.fx.options.asian import EuropeanFxAsianOption  # FX Asian instrument.
from src.instruments.fx.options.lookback import EuropeanFxLookbackOption  # FX lookback instrument.
from src.instruments.fx.options.double_barrier import EuropeanFxDoubleBarrierOption  # FX double barrier instrument.
from src.instruments.fx.options.touch import EuropeanFxTouchOption  # FX touch instrument.

from src.marketdata.core.market import Market  # Market snapshot interface.
from src.models.numeric.monte_carlo.rng import NormalRng  # Reproducible normal RNG.
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme  # GBM simulator and scheme.
from src.pricers.fx.european_bsm import _rate_from_df  # DF -> continuous rate helper (shared with BSM adapters).

from src.models.payoffs.types import OptionType, BarrierDirection, BarrierStyle, DigitalPayoff   # Canonical option type ("call"/"put").
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff, require_path_payoff



# ======================================================================================
# Simulation artifacts (returned by run(...), so callers can plot/analyse without rerunning)
# ======================================================================================

@dataclass(frozen=True, slots=True)
class FxMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single *vanilla* trade on a single Market snapshot.

    Notes
    -----
    - discounted_payoffs are in *domestic currency* and already scaled by trade.notional.
    - paths are optional and can be very large; prefer storing only a small subset.
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike K.
    maturity: float  # Expiry T.
    df_domestic: float  # Domestic discount factor df_d(T).
    drift: float  # Domestic-measure drift (r_d - r_f).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    notional: float  # Foreign notional scaling.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths (may be rounded up for antithetic).
    n_steps: int  # Number of time steps.
    scheme: GbmScheme  # GBM scheme.
    antithetic: bool  # Antithetic flag.
    seed: Optional[int]  # RNG seed.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots S(T), shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples, domestic, shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).


@dataclass(frozen=True, slots=True)
class FxDigitalMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single *digital* trade on a single Market snapshot.

    Notes
    -----
    - discounted_payoffs are in *domestic currency*.
    - EuropeanFxDigitalOption currently has no "notional"; payout_amount is the contract payout size.
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike K.
    maturity: float  # Expiry T.
    df_domestic: float  # Domestic discount factor df_d(T).
    drift: float  # Domestic-measure drift (r_d - r_f).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    payoff: DigitalPayoff  # "cash" or "asset".
    payout_amount: float  # Cash amount (domestic) or asset units (foreign) depending on payoff.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths (may be rounded up for antithetic).
    n_steps: int  # Number of time steps.
    scheme: GbmScheme  # GBM scheme.
    antithetic: bool  # Antithetic flag.
    seed: Optional[int]  # RNG seed.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots S(T), shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples, domestic, shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).



@dataclass(frozen=True, slots=True)
class FxBarrierMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single *barrier* trade on a single Market snapshot.

    Notes
    -----
    - Barrier payoffs are path-dependent, so the simulation always *computes from paths*.
    - discounted_payoffs are in *domestic currency* and already scaled by trade.notional.
    - paths are optional in the returned artifact; prefer storing only a subset for plotting.
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike K.
    barrier_level: float  # Barrier level B.
    barrier_direction: BarrierDirection  # "up" or "down".
    barrier_style: BarrierStyle  # "knock_out" or "knock_in".
    rebate_amount: float  # Rebate paid at expiry (domestic per 1 foreign notional).
    maturity: float  # Expiry T.
    df_domestic: float  # Domestic discount factor df_d(T).
    drift: float  # Domestic-measure drift (r_d - r_f).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    notional: float  # Foreign notional scaling.

    # --- Simulation settings ---
    n_paths_requested: int  # Requested number of paths.
    n_paths_effective: int  # Effective paths (may be rounded up for antithetic).
    n_steps: int  # Number of time steps (monitoring points = n_steps+1 including S0).
    scheme: GbmScheme  # GBM scheme.
    antithetic: bool  # Antithetic flag.
    seed: Optional[int]  # RNG seed.

    # --- Outputs ---
    terminal_spots: np.ndarray  # Terminal spots S(T), shape (n_paths_effective,).
    discounted_payoffs: np.ndarray  # Discounted payoff samples (domestic), shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).


@dataclass(frozen=True, slots=True)
class FxAsianMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single Asian option trade.

    This dataclass stores all inputs and outputs from a Monte Carlo simulation,
    allowing callers to analyze results without re-running the simulation.

    Notes
    -----
    - discounted_payoffs are in *domestic currency* and already scaled by trade.notional.
    - paths are optional and can be very large; prefer storing only a small subset.
    - average_spots stores the computed average for each path (useful for diagnostics).
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike price K.
    maturity: float  # Expiry T (year fraction).
    df_domestic: float  # Domestic discount factor df_d(T).
    drift: float  # Domestic-measure drift (r_d - r_f).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    averaging_type: str  # "arithmetic" or "geometric".
    notional: float  # Foreign notional scaling.

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
    discounted_payoffs: np.ndarray  # Discounted payoff samples, domestic, shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).


@dataclass(frozen=True, slots=True)
class FxLookbackMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single lookback option trade.

    This dataclass stores all inputs and outputs from a Monte Carlo simulation,
    allowing callers to analyze results without re-running the simulation.

    Notes
    -----
    - discounted_payoffs are in *domestic currency* and already scaled by trade.notional.
    - paths are optional and can be very large; prefer storing only a small subset.
    - max_spots and min_spots store the computed extrema for each path (useful for diagnostics).
    """

    # --- Resolved inputs (useful for reporting) ---
    spot0: float  # Initial spot S0.
    strike: float  # Strike price K (0 for floating strike).
    maturity: float  # Expiry T (year fraction).
    df_domestic: float  # Domestic discount factor df_d(T).
    drift: float  # Domestic-measure drift (r_d - r_f).
    sigma: float  # Volatility.
    option_type: OptionType  # "call" or "put".
    lookback_type: str  # "floating_strike" or "fixed_strike".
    notional: float  # Foreign notional scaling.

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
    discounted_payoffs: np.ndarray  # Discounted payoff samples, domestic, shape (n_paths_effective,).
    paths: Optional[np.ndarray] = None  # Optional stored paths, shape (n_kept, n_steps+1).


@dataclass(frozen=True, slots=True)
class FxDoubleBarrierMcSimulation:
    """
    Monte Carlo simulation artifact for a double barrier option.

    This artifact captures all inputs, settings, and outputs from a simulation,
    enabling post-hoc analysis without re-running the simulation.

    Attributes
    ----------
    spot0 : float
        Initial spot price S0.
    strike : float
        Strike price K.
    lower_barrier : float
        Lower barrier level L.
    upper_barrier : float
        Upper barrier level U.
    barrier_style : BarrierStyle
        "knock_out" or "knock_in".
    rebate_amount : float
        Rebate paid at expiry (domestic per unit notional).
    maturity : float
        Time to expiry T.
    df_domestic : float
        Domestic discount factor df_d(T).
    drift : float
        Domestic-measure drift (r_d - r_f).
    sigma : float
        Volatility.
    option_type : OptionType
        "call" or "put".
    notional : float
        Foreign notional scaling.
    n_paths_requested : int
        Requested number of paths.
    n_paths_effective : int
        Effective paths (may be doubled for antithetic).
    n_steps : int
        Number of time steps (monitoring points = n_steps + 1).
    scheme : GbmScheme
        GBM simulation scheme.
    antithetic : bool
        Whether antithetic variates were used.
    seed : int or None
        RNG seed.
    terminal_spots : np.ndarray
        Terminal spot prices S(T), shape (n_paths_effective,).
    discounted_payoffs : np.ndarray
        Discounted payoff samples (domestic), shape (n_paths_effective,).
    min_spots : np.ndarray
        Minimum spot per path, shape (n_paths_effective,).
    max_spots : np.ndarray
        Maximum spot per path, shape (n_paths_effective,).
    exited_corridor : np.ndarray
        Boolean array indicating if path exited corridor.
    paths : np.ndarray or None
        Optional stored paths, shape (n_kept, n_steps+1).
    """

    # --- Resolved inputs ---
    spot0: float
    strike: float
    lower_barrier: float
    upper_barrier: float
    barrier_style: BarrierStyle
    rebate_amount: float
    maturity: float
    df_domestic: float
    drift: float
    sigma: float
    option_type: OptionType
    notional: float

    # --- Simulation settings ---
    n_paths_requested: int
    n_paths_effective: int
    n_steps: int
    scheme: GbmScheme
    antithetic: bool
    seed: Optional[int]

    # --- Outputs ---
    terminal_spots: np.ndarray      # S_T per path.
    discounted_payoffs: np.ndarray  # Discounted payoffs (domestic).
    min_spots: np.ndarray           # min(S_t) per path.
    max_spots: np.ndarray           # max(S_t) per path.
    exited_corridor: np.ndarray     # Boolean: did path exit corridor?
    paths: Optional[np.ndarray] = None  # Optional stored paths.


@dataclass(frozen=True, slots=True)
class FxTouchMcSimulation:
    """
    Monte Carlo simulation artifact for a touch option.

    Attributes
    ----------
    spot0 : float
        Initial spot price S0.
    barrier_level : float
        Barrier level H.
    barrier_direction : BarrierDirection
        "up" or "down".
    touch_style : TouchStyle
        "one_touch" or "no_touch".
    payout_amount : float
        Fixed payout Q (per unit notional).
    maturity : float
        Time to expiry T.
    df_domestic : float
        Domestic discount factor df_d(T).
    drift : float
        Domestic-measure drift (r_d - r_f).
    sigma : float
        Volatility.
    notional : float
        Scaling factor.
    n_paths_requested : int
        Requested number of paths.
    n_paths_effective : int
        Effective paths (may be doubled for antithetic).
    n_steps : int
        Number of time steps.
    scheme : GbmScheme
        GBM simulation scheme.
    antithetic : bool
        Whether antithetic variates were used.
    seed : int or None
        RNG seed.
    terminal_spots : np.ndarray
        Terminal spot prices S(T), shape (n_paths_effective,).
    discounted_payoffs : np.ndarray
        Discounted payoff samples, shape (n_paths_effective,).
    touched : np.ndarray
        Boolean array indicating if barrier was touched.
    paths : np.ndarray or None
        Optional stored paths.
    """

    # Resolved inputs.
    spot0: float
    barrier_level: float
    barrier_direction: BarrierDirection
    touch_style: TouchStyle
    payout_amount: float
    maturity: float
    df_domestic: float
    drift: float
    sigma: float
    notional: float

    # Simulation settings.
    n_paths_requested: int
    n_paths_effective: int
    n_steps: int
    scheme: GbmScheme
    antithetic: bool
    seed: Optional[int]

    # Outputs.
    terminal_spots: np.ndarray
    discounted_payoffs: np.ndarray
    touched: np.ndarray  # Boolean: did path touch barrier?
    paths: Optional[np.ndarray] = None


# ======================================================================================
# Vanilla MC pricer (already integrated with payoff library via VanillaPayoff)
# ======================================================================================

@dataclass(frozen=True, slots=True)
class FxEuropeanVanillaMcPricer:
    """
    Monte Carlo pricer for European FX vanilla options under Garman–Kohlhagen.

    Mapping
    -------
    - r_d = -ln(df_d)/T
    - r_f = -ln(df_f)/T
    - drift = r_d - r_f  (domestic measure)
    - PV = notional * df_d(T) * E[ payoff(S_T) ]
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths.
    seed: Optional[int] = 7  # RNG seed for reproducibility.
    antithetic: bool = True  # Use antithetic variates to reduce variance.

    n_steps: int = 1  # Number of time steps (exact GBM allows 1).
    scheme: GbmScheme = "exact"  # Default to exact terminal sampling for GBM.

    def price(self, trade: EuropeanFxVanillaOption, market: Market) -> float:
        sim = self.run(trade, market, store_paths=False)  # Run once without storing paths.
        return float(sim.discounted_payoffs.mean())  # PV is the mean of discounted payoff samples.

    def run(
        self,
        trade: EuropeanFxVanillaOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxMcSimulation:
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)  # Single source of truth.

    def sample_terminal_spots(self, trade: EuropeanFxVanillaOption, market: Market) -> np.ndarray:
        return self.run(trade, market, store_paths=False).terminal_spots  # Delegate to run().

    def sample_discounted_payoffs(self, trade: EuropeanFxVanillaOption, market: Market) -> np.ndarray:
        return self.run(trade, market, store_paths=False).discounted_payoffs  # Delegate to run().

    def _run_simulation(
        self,
        trade: EuropeanFxVanillaOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxMcSimulation:
        if self.n_paths <= 0:  # Validate path count.
            raise ValueError("n_paths must be positive.")  # Fail fast.
        if self.n_steps <= 0:  # Validate step count.
            raise ValueError("n_steps must be positive.")  # Fail fast.

        option_type: OptionType = trade.option_type  # Read option type.
        spot0 = float(market.quote(trade.spot_id))  # Read spot S0.
        strike = float(trade.strike)  # Read strike K.
        maturity = float(trade.expiry)  # Read maturity T.
        notional = float(trade.notional)  # Read notional (foreign units).

        payoff_fn = require_terminal_payoff(build_payoff_1d(trade))  # Build payoff function from payoff library.

        if maturity < 0.0:  # Validate maturity.
            raise ValueError("expiry must be >= 0.")  # Fail fast.
        if notional < 0.0:  # Validate notional.
            raise ValueError("notional must be >= 0.")  # Fail fast.

        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))  # Domestic discount factor.
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))  # Foreign discount factor.

        r_d = _rate_from_df(df=df_d, t=maturity)  # Domestic rate.
        r_f = _rate_from_df(df=df_f, t=maturity)  # Foreign rate.

        drift = float(r_d - r_f)  # Domestic-measure drift for FX spot.

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))  # Implied vol.
        if sigma < 0.0:  # Validate vol.
            raise ValueError("Implied vol must be non-negative.")  # Fail fast.

        if maturity == 0.0:  # Handle deterministic expiry.
            terminal_spots = np.array([spot0], dtype=np.float64)  # Terminal spot is spot0.
            payoff = payoff_fn.terminal(spot=terminal_spots)  # Compute terminal payoff vector.
            discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)  # Discount + scale.
            return FxMcSimulation(
                spot0=spot0,
                strike=strike,
                maturity=maturity,
                df_domestic=df_d,
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

        rng = NormalRng(seed=self.seed)  # Construct RNG.
        normals = rng.standard_normals(  # Draw standard normals for the simulator.
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])  # Effective path count after antithetic adjustment.

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

        terminal_spots = all_paths[:, -1].copy()  # Extract terminal spots S(T).
        payoff = payoff_fn.terminal(spot=terminal_spots)  # Compute terminal payoff per path.

        discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)  # Discount + scale.

        if not store_paths:  # If we are not storing paths...
            kept_paths = None  # ...store nothing.
        else:
            if paths_keep < 0:  # Validate keep count.
                raise ValueError("paths_keep must be >= 0.")  # Fail fast.
            if paths_keep == 0:  # Keep all paths if requested.
                kept_paths = all_paths.copy()  # Copy for safety.
            else:  # Otherwise keep a subset.
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()  # Copy only kept rows.

        return FxMcSimulation(
            spot0=spot0,
            strike=strike,
            maturity=maturity,
            df_domestic=df_d,
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
# Digital MC pricer (cash + asset digitals in one adapter, using payoff library)
# ======================================================================================

@dataclass(frozen=True, slots=True)
class FxEuropeanDigitalMcPricer:
    """
    Monte Carlo pricer for European FX digital options under Garman–Kohlhagen.

    Supports
    --------
    - trade.payoff == "cash": cash-or-nothing
    - trade.payoff == "asset": asset-or-nothing (pays foreign units, valued in domestic via S_T)

    Mapping
    -------
    - r_d = -ln(df_d)/T
    - r_f = -ln(df_f)/T
    - drift = r_d - r_f  (domestic measure)
    - PV = df_d(T) * E[ payoff(S_T) ]   (payoff returns domestic value at expiry)

    Notes
    -----
    - EuropeanFxDigitalOption currently has no notional; payout_amount is taken as the contract payout size.
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths.
    seed: Optional[int] = 7  # RNG seed for reproducibility.
    antithetic: bool = True  # Antithetic variates.

    n_steps: int = 1  # Number of time steps.
    scheme: GbmScheme = "exact"  # GBM scheme ("exact" recommended here).

    def price(self, trade: EuropeanFxDigitalOption, market: Market) -> float:
        sim = self.run(trade, market, store_paths=False)  # Run once without storing paths.
        return float(sim.discounted_payoffs.mean())  # PV is the mean of discounted payoff samples.

    def run(
        self,
        trade: EuropeanFxDigitalOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxDigitalMcSimulation:
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)  # Single source of truth.

    def sample_terminal_spots(self, trade: EuropeanFxDigitalOption, market: Market) -> np.ndarray:
        return self.run(trade, market, store_paths=False).terminal_spots  # Delegate to run().

    def sample_discounted_payoffs(self, trade: EuropeanFxDigitalOption, market: Market) -> np.ndarray:
        return self.run(trade, market, store_paths=False).discounted_payoffs  # Delegate to run().

    def _run_simulation(
        self,
        trade: EuropeanFxDigitalOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxDigitalMcSimulation:
        if self.n_paths <= 0:  # Validate path count.
            raise ValueError("n_paths must be positive.")  # Fail fast.
        if self.n_steps <= 0:  # Validate step count.
            raise ValueError("n_steps must be positive.")  # Fail fast.

        option_type: OptionType = trade.option_type  # Read option type.
        payoff_style: DigitalPayoff = trade.payoff  # Read payoff style ("cash" or "asset").
        payout_amount = float(trade.payout_amount)  # Read payout amount (cash domestic or asset units foreign).

        spot0 = float(market.quote(trade.spot_id))  # Read spot S0.
        strike = float(trade.strike)  # Read strike K.
        maturity = float(trade.expiry)  # Read maturity T.

        if maturity < 0.0:  # Validate maturity.
            raise ValueError("expiry must be >= 0.")  # Fail fast.

        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))  # Domestic discount factor.
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))  # Foreign discount factor.

        r_d = _rate_from_df(df=df_d, t=maturity)  # Domestic rate.
        r_f = _rate_from_df(df=df_f, t=maturity)  # Foreign rate.

        drift = float(r_d - r_f)  # Domestic-measure drift.

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))  # Implied vol.
        if sigma < 0.0:  # Validate vol.
            raise ValueError("Implied vol must be non-negative.")  # Fail fast.

        # Build the payoff function from the payoff library (single responsibility: payoff definition lives there).
        payoff_fn = require_terminal_payoff(build_payoff_1d(trade))

        if maturity == 0.0:  # Deterministic expiry handling.
            terminal_spots = np.array([spot0], dtype=np.float64)  # Terminal spot is spot0.
            payoff = payoff_fn.terminal(spot=terminal_spots)  # Compute terminal payoff in domestic units.
            discounted_payoffs = (float(df_d) * payoff).astype(np.float64, copy=False)  # Discount in domestic.
            return FxDigitalMcSimulation(
                spot0=spot0,
                strike=strike,
                maturity=maturity,
                df_domestic=df_d,
                drift=drift,
                sigma=sigma,
                option_type=option_type,
                payoff=payoff_style,
                payout_amount=payout_amount,
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

        rng = NormalRng(seed=self.seed)  # Construct RNG.
        normals = rng.standard_normals(  # Draw standard normals.
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])  # Effective path count.

        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)  # Construct simulator.
        all_paths = simulator.simulate_paths(  # Simulate GBM paths.
            spot0=spot0,
            maturity=maturity,
            n_steps=self.n_steps,
            n_paths=n_paths_eff,
            normals=normals,
            scheme=self.scheme,
            dtype=np.float64,
        )

        terminal_spots = all_paths[:, -1].copy()  # Extract terminal spots S(T).
        payoff = payoff_fn.terminal(spot=terminal_spots)  # Compute terminal payoff in domestic units.
        discounted_payoffs = (float(df_d) * payoff).astype(np.float64, copy=False)  # Discount domestically.

        if not store_paths:  # If we do not store paths...
            kept_paths = None  # ...store none.
        else:
            if paths_keep < 0:  # Validate keep count.
                raise ValueError("paths_keep must be >= 0.")  # Fail fast.
            if paths_keep == 0:  # Store all if requested.
                kept_paths = all_paths.copy()  # Copy for safety.
            else:  # Store subset otherwise.
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()  # Copy subset.

        return FxDigitalMcSimulation(
            spot0=spot0,
            strike=strike,
            maturity=maturity,
            df_domestic=df_d,
            drift=drift,
            sigma=sigma,
            option_type=option_type,
            payoff=payoff_style,
            payout_amount=payout_amount,
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


@dataclass(frozen=True, slots=True)
class FxEuropeanBarrierMcPricer:
    """
    Monte Carlo pricer for European FX single-barrier options under Garman–Kohlhagen.

    Monitoring (V1)
    ---------------
    - Discrete monitoring on simulated path points (including S0 and intermediate steps).
    - Continuous-monitoring adjustments (e.g. Brownian-bridge) can be added later.

    Mapping
    -------
    - r_d = -ln(df_d)/T
    - r_f = -ln(df_f)/T
    - drift = r_d - r_f  (domestic measure)
    - PV = notional * df_d(T) * E[ barrier_payoff(paths) ]
      where barrier_payoff(paths) returns the *expiry payoff in domestic currency per 1 foreign notional*.
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths.
    seed: Optional[int] = 7  # RNG seed for reproducibility.
    antithetic: bool = True  # Antithetic variates.

    n_steps: int = 64  # Barrier needs monitoring points; choose a denser default than vanilla/digital.
    scheme: GbmScheme = "exact"  # Exact GBM steps (exact per step) is typically preferred.

    def price(self, trade: EuropeanFxBarrierOption, market: Market) -> float:
        sim = self.run(trade, market, store_paths=False)
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: EuropeanFxBarrierOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxBarrierMcSimulation:
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)

    def sample_terminal_spots(self, trade: EuropeanFxBarrierOption, market: Market) -> np.ndarray:
        return self.run(trade, market, store_paths=False).terminal_spots

    def sample_discounted_payoffs(self, trade: EuropeanFxBarrierOption, market: Market) -> np.ndarray:
        return self.run(trade, market, store_paths=False).discounted_payoffs

    def _run_simulation(
        self,
        trade: EuropeanFxBarrierOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxBarrierMcSimulation:
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive (barriers need monitoring points).")

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type
        spot0 = float(market.quote(trade.spot_id))
        strike = float(trade.strike)
        maturity = float(trade.expiry)
        notional = float(trade.notional)

        barrier_level = float(trade.barrier_level)
        barrier_direction: BarrierDirection = trade.barrier_direction  # type: ignore[assignment]
        barrier_style: BarrierStyle = trade.barrier_style  # type: ignore[assignment]
        rebate_amount = float(trade.rebate_amount)

        if maturity < 0.0:
            raise ValueError("expiry must be >= 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")
        if spot0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if barrier_level <= 0.0:
            raise ValueError("barrier_level must be > 0.")
        if rebate_amount < 0.0:
            raise ValueError("rebate_amount must be >= 0.")

        # -----------------------------
        # Resolve discount factors and rates
        # -----------------------------
        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))

        r_d = _rate_from_df(df=df_d, t=maturity)
        r_f = _rate_from_df(df=df_f, t=maturity)

        drift = float(r_d - r_f)

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # -----------------------------
        # Build payoff from payoff library
        # -----------------------------
        # Build payoff via factory and enforce path-dependent contract
        payoff_fn = require_path_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Deterministic expiry handling
        # -----------------------------
        if maturity == 0.0:
            # With T=0, we treat the “path” as containing only S0.
            # Barrier hit is evaluated on that single observation (consistent with discrete monitoring).
            paths = np.array([[spot0]], dtype=np.float64)
            terminal_spots = np.array([spot0], dtype=np.float64)

            payoff = payoff_fn.terminal_from_paths(paths)  # per-unit-notional, domestic
            discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

            return FxBarrierMcSimulation(
                spot0=spot0,
                strike=strike,
                barrier_level=barrier_level,
                barrier_direction=barrier_direction,
                barrier_style=barrier_style,
                rebate_amount=rebate_amount,
                maturity=maturity,
                df_domestic=df_d,
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
        rng = NormalRng(seed=self.seed)
        normals = rng.standard_normals(
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])

        # -----------------------------
        # Simulate GBM paths
        # -----------------------------
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)
        all_paths = simulator.simulate_paths(
            spot0=spot0,
            maturity=maturity,
            n_steps=self.n_steps,
            n_paths=n_paths_eff,
            normals=normals,
            scheme=self.scheme,
            dtype=np.float64,
        )

        # Terminal spot for each path (useful for plotting / diagnostics)
        terminal_spots = all_paths[:, -1].copy()

        # -----------------------------
        # Compute payoff from full path (path-dependent)
        # -----------------------------
        payoff = payoff_fn.terminal_from_paths(all_paths)  # per-unit-notional, domestic

        # Discount domestically and scale by notional
        discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

        # -----------------------------
        # Optionally retain paths for plotting
        # -----------------------------
        if not store_paths:
            kept_paths = None
        else:
            if paths_keep < 0:
                raise ValueError("paths_keep must be >= 0.")
            if paths_keep == 0:
                kept_paths = all_paths.copy()
            else:
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()

        return FxBarrierMcSimulation(
            spot0=spot0,
            strike=strike,
            barrier_level=barrier_level,
            barrier_direction=barrier_direction,
            barrier_style=barrier_style,
            rebate_amount=rebate_amount,
            maturity=maturity,
            df_domestic=df_d,
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


@dataclass(frozen=True, slots=True)
class FxEuropeanAsianMcPricer:
    """
    Monte Carlo pricer for European FX Asian options under Garman–Kohlhagen.

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
    PV = notional * df_d(T) * E[ max(Avg(S_t) - K, 0) ]  (call)
    PV = notional * df_d(T) * E[ max(K - Avg(S_t), 0) ]  (put)

    Where:
    - Avg(S_t) = arithmetic or geometric mean over monitoring points
    - E[...] is the risk-neutral expectation
    - df_d(T) is the domestic discount factor

    Mapping
    -------
    - r_d = -ln(df_d)/T  (domestic rate)
    - r_f = -ln(df_f)/T  (foreign rate)
    - drift = r_d - r_f  (domestic measure)
    - PV = notional * df_d(T) * E[ payoff(paths) ]

    Notes
    -----
    - Path-dependent: requires full simulated paths (not just terminal spots).
    - Arithmetic averaging is more common but has no closed-form solution.
    - Geometric averaging has closed-form solutions but is less common in practice.
    - More monitoring points (n_steps) give better approximation to continuous averaging.
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths (higher = more accurate but slower).
    seed: Optional[int] = 7  # RNG seed for reproducibility (set to None for random).
    antithetic: bool = True  # Use antithetic variates to reduce variance (recommended).

    n_steps: int = 64  # Number of time steps (more steps = more monitoring points for averaging).
    scheme: GbmScheme = "exact"  # GBM scheme ("exact" recommended for GBM).

    def price(self, trade: EuropeanFxAsianOption, market: Market) -> float:
        """
        Price an Asian option using Monte Carlo simulation.

        This is the main pricing method that returns the present value (PV) of the option.

        Parameters
        ----------
        trade : EuropeanFxAsianOption
            The Asian option instrument to price.
        market : Market
            Market snapshot containing spot, vol, and curves.

        Returns
        -------
        float
            Present value in domestic currency.
        """
        # Run simulation without storing paths (faster for pricing only)
        sim = self.run(trade, market, store_paths=False)
        # PV is the sample mean of discounted payoffs
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: EuropeanFxAsianOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxAsianMcSimulation:
        """
        Run Monte Carlo simulation and return full simulation artifact.

        This method performs the full simulation and returns a dataclass containing
        all inputs and outputs, useful for diagnostics, plotting, and analysis.

        Parameters
        ----------
        trade : EuropeanFxAsianOption
            The Asian option instrument to simulate.
        market : Market
            Market snapshot containing spot, vol, and curves.
        store_paths : bool, optional
            If True, store simulated paths in the returned artifact (default: False).
        paths_keep : int, optional
            Number of paths to keep if store_paths=True (0 = keep all, default: 0).

        Returns
        -------
        FxAsianMcSimulation
            Complete simulation artifact with inputs and outputs.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)

    def sample_terminal_spots(self, trade: EuropeanFxAsianOption, market: Market) -> np.ndarray:
        """
        Sample terminal spot prices from the simulation.

        Useful for diagnostics and plotting the distribution of terminal spots.

        Parameters
        ----------
        trade : EuropeanFxAsianOption
            The Asian option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Terminal spot prices, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).terminal_spots

    def sample_average_spots(self, trade: EuropeanFxAsianOption, market: Market) -> np.ndarray:
        """
        Sample average spot prices from the simulation.

        Useful for diagnostics and understanding the distribution of averages.

        Parameters
        ----------
        trade : EuropeanFxAsianOption
            The Asian option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Average spot prices, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).average_spots

    def sample_discounted_payoffs(self, trade: EuropeanFxAsianOption, market: Market) -> np.ndarray:
        """
        Sample discounted payoffs from the simulation.

        Useful for computing confidence intervals and understanding payoff distribution.

        Parameters
        ----------
        trade : EuropeanFxAsianOption
            The Asian option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Discounted payoffs, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).discounted_payoffs

    def _run_simulation(
        self,
        trade: EuropeanFxAsianOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxAsianMcSimulation:
        """
        Internal method that performs the actual Monte Carlo simulation.

        This is the single source of truth for simulation logic. All public methods
        delegate to this method to ensure consistency.

        Parameters
        ----------
        trade : EuropeanFxAsianOption
            The Asian option instrument to simulate.
        market : Market
            Market snapshot.
        store_paths : bool
            Whether to store paths in the returned artifact.
        paths_keep : int
            Number of paths to keep if storing (0 = all).

        Returns
        -------
        FxAsianMcSimulation
            Complete simulation artifact.
        """
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:  # Ensure positive path count
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:  # Ensure positive step count (needed for averaging)
            raise ValueError("n_steps must be positive (Asian options need monitoring points).")

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type  # "call" or "put"
        spot0 = float(market.quote(trade.spot_id))  # Initial spot S0
        strike = float(trade.strike)  # Strike price K
        maturity = float(trade.expiry)  # Time to expiry T
        notional = float(trade.notional)  # Foreign notional
        averaging_type = str(trade.averaging_type)  # "arithmetic" or "geometric"

        # Validate inputs
        if maturity < 0.0:  # Expiry cannot be negative
            raise ValueError("expiry must be >= 0.")
        if notional < 0.0:  # Notional cannot be negative
            raise ValueError("notional must be >= 0.")
        if spot0 <= 0.0:  # Spot must be positive
            raise ValueError("spot must be > 0.")

        # -----------------------------
        # Resolve discount factors and rates
        # -----------------------------
        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))  # Domestic discount factor
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))  # Foreign discount factor

        # Convert discount factors to continuously-compounded rates
        r_d = _rate_from_df(df=df_d, t=maturity)  # Domestic rate: r_d = -ln(df_d)/T
        r_f = _rate_from_df(df=df_f, t=maturity)  # Foreign rate: r_f = -ln(df_f)/T

        # Domestic-measure drift for FX spot (Garman-Kohlhagen)
        drift = float(r_d - r_f)

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))
        if sigma < 0.0:  # Volatility cannot be negative
            raise ValueError("Implied vol must be non-negative.")

        # -----------------------------
        # Build payoff from payoff library
        # -----------------------------
        # Build payoff via factory and enforce path-dependent contract
        # The payoff library is the single source of truth for payoff logic
        payoff_fn = require_path_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Deterministic expiry handling
        # -----------------------------
        if maturity == 0.0:
            # At expiry, the "path" contains only S0 (which equals S_T)
            # The average is just S0, and payoff is computed on that
            paths = np.array([[spot0]], dtype=np.float64)  # Single path with one point
            terminal_spots = np.array([spot0], dtype=np.float64)  # Terminal equals initial

            # Compute payoff from paths (average will be S0)
            payoff = payoff_fn.terminal_from_paths(paths)  # Per-unit-notional, domestic
            discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

            # Compute average for diagnostics (will be S0)
            average_spots = np.array([spot0], dtype=np.float64)

            return FxAsianMcSimulation(
                spot0=spot0,
                strike=strike,
                maturity=maturity,
                df_domestic=df_d,
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
        rng = NormalRng(seed=self.seed)  # Construct RNG with seed for reproducibility
        normals = rng.standard_normals(
            self.n_paths,  # Number of paths requested
            self.n_steps,  # Number of steps per path
            antithetic=self.antithetic,  # Use antithetic variates if enabled
            dtype=np.float64,  # Use float64 for numerical precision
        )
        n_paths_eff = int(normals.shape[0])  # Effective path count (may be doubled if antithetic)

        # -----------------------------
        # Simulate GBM paths
        # -----------------------------
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)  # Construct GBM simulator
        all_paths = simulator.simulate_paths(
            spot0=spot0,  # Initial spot
            maturity=maturity,  # Time to expiry
            n_steps=self.n_steps,  # Number of time steps
            n_paths=n_paths_eff,  # Number of paths
            normals=normals,  # Pre-generated random normals
            scheme=self.scheme,  # GBM scheme ("exact" or "euler")
            dtype=np.float64,  # Use float64 for precision
        )

        # Extract terminal spots for diagnostics (last column of paths)
        terminal_spots = all_paths[:, -1].copy()

        # -----------------------------
        # Compute payoff from full path (path-dependent)
        # -----------------------------
        # The payoff function computes the average over each path and applies vanilla payoff
        payoff = payoff_fn.terminal_from_paths(all_paths)  # Per-unit-notional, domestic

        # Discount domestically and scale by notional
        # PV = notional * df_d(T) * E[payoff]
        discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

        # Compute average spots for diagnostics (extract from payoff if possible, or recompute)
        # For diagnostics, we compute the average directly
        if averaging_type == "arithmetic":
            average_spots = np.mean(all_paths, axis=1, dtype=np.float64)
        else:  # geometric
            log_paths = np.log(all_paths)
            log_mean = np.mean(log_paths, axis=1, dtype=np.float64)
            average_spots = np.exp(log_mean)

        # -----------------------------
        # Optionally retain paths for plotting
        # -----------------------------
        if not store_paths:  # If not storing paths...
            kept_paths = None  # ...store nothing
        else:
            if paths_keep < 0:  # Validate keep count
                raise ValueError("paths_keep must be >= 0.")
            if paths_keep == 0:  # Keep all if requested
                kept_paths = all_paths.copy()  # Copy for safety
            else:  # Keep subset otherwise
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()  # Copy subset

        return FxAsianMcSimulation(
            spot0=spot0,
            strike=strike,
            maturity=maturity,
            df_domestic=df_d,
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


@dataclass(frozen=True, slots=True)
class FxEuropeanLookbackMcPricer:
    """
    Monte Carlo pricer for European FX Lookback options under Garman–Kohlhagen.

    Lookback options pay based on the maximum or minimum price over the option's
    life, providing "perfect hindsight" for optimal entry/exit timing.

    Pricing Method
    --------------
    - Simulate GBM paths with discrete monitoring points
    - Compute path extrema (max and min) for each path
    - Apply lookback payoff formula (floating or fixed strike)
    - Discount and take expectation

    Mathematical Formula
    --------------------
    Floating strike:
        PV = notional * df_d(T) * E[ S_T - min(S_t) ]  (call)
        PV = notional * df_d(T) * E[ max(S_t) - S_T ]  (put)

    Fixed strike:
        PV = notional * df_d(T) * E[ max(max(S_t) - K, 0) ]  (call)
        PV = notional * df_d(T) * E[ max(K - min(S_t), 0) ]  (put)

    Where:
    - E[...] is the risk-neutral expectation
    - df_d(T) is the domestic discount factor
    - M_T = max(S_t) and m_T = min(S_t) over monitoring points

    Mapping
    -------
    - r_d = -ln(df_d)/T  (domestic rate)
    - r_f = -ln(df_f)/T  (foreign rate)
    - drift = r_d - r_f  (domestic measure)
    - PV = notional * df_d(T) * E[ payoff(paths) ]

    Notes
    -----
    - Path-dependent: requires full simulated paths (not just terminal spots).
    - Floating strike lookbacks are always ITM (payoff >= 0).
    - More monitoring points (n_steps) give better approximation to continuous.
    - Lookback options are always >= vanilla options (captures optimal timing).
    """

    n_paths: int = 200_000  # Number of Monte Carlo paths (higher = more accurate but slower).
    seed: Optional[int] = 7  # RNG seed for reproducibility (set to None for random).
    antithetic: bool = True  # Use antithetic variates to reduce variance (recommended).

    n_steps: int = 64  # Number of time steps (more steps = better extremum approximation).
    scheme: GbmScheme = "exact"  # GBM scheme ("exact" recommended for GBM).

    def price(self, trade: EuropeanFxLookbackOption, market: Market) -> float:
        """
        Price a lookback option using Monte Carlo simulation.

        This is the main pricing method that returns the present value (PV) of the option.

        Parameters
        ----------
        trade : EuropeanFxLookbackOption
            The lookback option instrument to price.
        market : Market
            Market snapshot containing spot, vol, and curves.

        Returns
        -------
        float
            Present value in domestic currency.
        """
        # Run simulation without storing paths (faster for pricing only)
        sim = self.run(trade, market, store_paths=False)
        # PV is the sample mean of discounted payoffs
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: EuropeanFxLookbackOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxLookbackMcSimulation:
        """
        Run Monte Carlo simulation and return full simulation artifact.

        This method performs the full simulation and returns a dataclass containing
        all inputs and outputs, useful for diagnostics, plotting, and analysis.

        Parameters
        ----------
        trade : EuropeanFxLookbackOption
            The lookback option instrument to simulate.
        market : Market
            Market snapshot containing spot, vol, and curves.
        store_paths : bool, optional
            If True, store simulated paths in the returned artifact (default: False).
        paths_keep : int, optional
            Number of paths to keep if store_paths=True (0 = keep all, default: 0).

        Returns
        -------
        FxLookbackMcSimulation
            Complete simulation artifact with inputs and outputs.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)

    def sample_terminal_spots(self, trade: EuropeanFxLookbackOption, market: Market) -> np.ndarray:
        """
        Sample terminal spot prices from the simulation.

        Parameters
        ----------
        trade : EuropeanFxLookbackOption
            The lookback option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Terminal spot prices, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).terminal_spots

    def sample_max_spots(self, trade: EuropeanFxLookbackOption, market: Market) -> np.ndarray:
        """
        Sample maximum spot prices from the simulation.

        Parameters
        ----------
        trade : EuropeanFxLookbackOption
            The lookback option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Maximum spot prices over each path, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).max_spots

    def sample_min_spots(self, trade: EuropeanFxLookbackOption, market: Market) -> np.ndarray:
        """
        Sample minimum spot prices from the simulation.

        Parameters
        ----------
        trade : EuropeanFxLookbackOption
            The lookback option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Minimum spot prices over each path, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).min_spots

    def sample_discounted_payoffs(self, trade: EuropeanFxLookbackOption, market: Market) -> np.ndarray:
        """
        Sample discounted payoffs from the simulation.

        Parameters
        ----------
        trade : EuropeanFxLookbackOption
            The lookback option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        np.ndarray
            Discounted payoffs, shape (n_paths_effective,).
        """
        return self.run(trade, market, store_paths=False).discounted_payoffs

    def _run_simulation(
        self,
        trade: EuropeanFxLookbackOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxLookbackMcSimulation:
        """
        Internal method that performs the actual Monte Carlo simulation.

        This is the single source of truth for simulation logic. All public methods
        delegate to this method to ensure consistency.

        Parameters
        ----------
        trade : EuropeanFxLookbackOption
            The lookback option instrument to simulate.
        market : Market
            Market snapshot.
        store_paths : bool
            Whether to store paths in the returned artifact.
        paths_keep : int
            Number of paths to keep if storing (0 = all).

        Returns
        -------
        FxLookbackMcSimulation
            Complete simulation artifact.
        """
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive (lookback options need monitoring points).")

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type
        spot0 = float(market.quote(trade.spot_id))
        strike = float(trade.strike)  # 0 for floating strike
        maturity = float(trade.expiry)
        notional = float(trade.notional)
        lookback_type = str(trade.lookback_type)

        # Validate inputs
        if maturity < 0.0:
            raise ValueError("expiry must be >= 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")
        if spot0 <= 0.0:
            raise ValueError("spot must be > 0.")

        # -----------------------------
        # Resolve discount factors and rates
        # -----------------------------
        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))

        r_d = _rate_from_df(df=df_d, t=maturity)
        r_f = _rate_from_df(df=df_f, t=maturity)

        drift = float(r_d - r_f)

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        # For lookbacks, use ATM vol or strike-based vol for fixed strike
        vol_strike = strike if lookback_type == "fixed_strike" and strike > 0 else spot0
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=vol_strike))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # -----------------------------
        # Build payoff from payoff library
        # -----------------------------
        payoff_fn = require_path_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Deterministic expiry handling
        # -----------------------------
        if maturity == 0.0:
            # At expiry, the "path" contains only S0 (which equals S_T)
            # For lookback: max = min = S0 = S_T
            paths = np.array([[spot0]], dtype=np.float64)
            terminal_spots = np.array([spot0], dtype=np.float64)
            max_spots = np.array([spot0], dtype=np.float64)
            min_spots = np.array([spot0], dtype=np.float64)

            # Compute payoff from paths
            payoff = payoff_fn.terminal_from_paths(paths)
            discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

            return FxLookbackMcSimulation(
                spot0=spot0,
                strike=strike,
                maturity=maturity,
                df_domestic=df_d,
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
        rng = NormalRng(seed=self.seed)
        normals = rng.standard_normals(
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])

        # -----------------------------
        # Simulate GBM paths
        # -----------------------------
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)
        all_paths = simulator.simulate_paths(
            spot0=spot0,
            maturity=maturity,
            n_steps=self.n_steps,
            n_paths=n_paths_eff,
            normals=normals,
            scheme=self.scheme,
            dtype=np.float64,
        )

        # Extract terminal spots (last column)
        terminal_spots = all_paths[:, -1].copy()

        # Compute path extrema for diagnostics
        max_spots = np.max(all_paths, axis=1)
        min_spots = np.min(all_paths, axis=1)

        # -----------------------------
        # Compute payoff from full path
        # -----------------------------
        payoff = payoff_fn.terminal_from_paths(all_paths)

        # Discount domestically and scale by notional
        discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

        # -----------------------------
        # Optionally retain paths
        # -----------------------------
        if not store_paths:
            kept_paths = None
        else:
            if paths_keep < 0:
                raise ValueError("paths_keep must be >= 0.")
            if paths_keep == 0:
                kept_paths = all_paths.copy()
            else:
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()

        return FxLookbackMcSimulation(
            spot0=spot0,
            strike=strike,
            maturity=maturity,
            df_domestic=df_d,
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


@dataclass(frozen=True, slots=True)
class FxEuropeanDoubleBarrierMcPricer:
    """
    Monte Carlo pricer for European FX double barrier options.

    This pricer simulates GBM paths and evaluates whether the path stays within
    the corridor [lower_barrier, upper_barrier]. The payoff is:
        - Knock-out: vanilla if stayed in corridor, else rebate
        - Knock-in:  vanilla if exited corridor, else rebate

    Pricing Formula
    ---------------
    PV = notional * df_d(T) * E[Payoff(paths)]

    where Payoff is computed using the payoff library's DoubleBarrierPayoff.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths. Default 200,000.
    seed : int or None
        RNG seed for reproducibility. Default 7.
    antithetic : bool
        Whether to use antithetic variates. Default True.
    n_steps : int
        Number of time steps (monitoring points). Default 64.
    scheme : GbmScheme
        GBM simulation scheme. Default "exact".

    Examples
    --------
    >>> from src.pricers.fx.double_barrier_mc import FxEuropeanDoubleBarrierMcPricer
    >>> pricer = FxEuropeanDoubleBarrierMcPricer(n_paths=100_000)
    >>> pv = pricer.price(trade, market)
    """

    # Monte Carlo configuration.
    n_paths: int = 200_000                # Number of simulation paths.
    seed: Optional[int] = 7               # RNG seed for reproducibility.
    antithetic: bool = True               # Antithetic variates for variance reduction.

    # Path simulation configuration.
    n_steps: int = 64                     # Monitoring points for barrier checking.
    scheme: GbmScheme = "exact"           # GBM scheme (exact per step).

    def price(self, trade: EuropeanFxDoubleBarrierOption, market: Market) -> float:
        """
        Compute the present value of the double barrier option.

        Parameters
        ----------
        trade : EuropeanFxDoubleBarrierOption
            The double barrier option instrument.
        market : Market
            Market snapshot with spot, curves, and vol surface.

        Returns
        -------
        float
            Present value in domestic currency.
        """
        # Run simulation and return mean discounted payoff.
        sim = self.run(trade, market, store_paths=False)
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: EuropeanFxDoubleBarrierOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxDoubleBarrierMcSimulation:
        """
        Run the full simulation and return artifact with all details.

        Parameters
        ----------
        trade : EuropeanFxDoubleBarrierOption
            The double barrier option instrument.
        market : Market
            Market snapshot.
        store_paths : bool
            Whether to store paths in the artifact. Default False.
        paths_keep : int
            Number of paths to keep if storing. 0 means keep all.

        Returns
        -------
        FxDoubleBarrierMcSimulation
            Simulation artifact with all inputs, settings, and outputs.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)

    def sample_terminal_spots(self, trade: EuropeanFxDoubleBarrierOption, market: Market) -> np.ndarray:
        """Return terminal spots S(T) from simulation."""
        return self.run(trade, market, store_paths=False).terminal_spots

    def sample_discounted_payoffs(self, trade: EuropeanFxDoubleBarrierOption, market: Market) -> np.ndarray:
        """Return discounted payoff samples from simulation."""
        return self.run(trade, market, store_paths=False).discounted_payoffs

    def _run_simulation(
        self,
        trade: EuropeanFxDoubleBarrierOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxDoubleBarrierMcSimulation:
        """
        Internal simulation implementation.

        Steps
        -----
        1. Validate pricer configuration.
        2. Read trade inputs from instrument.
        3. Resolve market data (spot, curves, vol).
        4. Build payoff function from payoff library.
        5. Generate random normals.
        6. Simulate GBM paths.
        7. Compute payoffs using path-dependent payoff.
        8. Package results into simulation artifact.
        """
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive (barriers need monitoring points).")

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type
        spot0 = float(market.quote(trade.spot_id))      # Initial spot S0.
        strike = float(trade.strike)                    # Strike K.
        maturity = float(trade.expiry)                  # Expiry T.
        notional = float(trade.notional)                # Foreign notional.

        lower_barrier = float(trade.lower_barrier)      # Lower barrier L.
        upper_barrier = float(trade.upper_barrier)      # Upper barrier U.
        barrier_style: BarrierStyle = trade.barrier_style  # type: ignore[assignment]
        rebate_amount = float(trade.rebate_amount)      # Rebate at expiry.

        # -----------------------------
        # Validate inputs
        # -----------------------------
        if maturity < 0.0:
            raise ValueError("expiry must be >= 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")
        if spot0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if lower_barrier <= 0.0:
            raise ValueError("lower_barrier must be > 0.")
        if upper_barrier <= 0.0:
            raise ValueError("upper_barrier must be > 0.")
        if lower_barrier >= upper_barrier:
            raise ValueError("lower_barrier must be < upper_barrier.")
        if rebate_amount < 0.0:
            raise ValueError("rebate_amount must be >= 0.")

        # Validate spot is inside corridor (required for double barrier semantics).
        if spot0 <= lower_barrier:
            raise ValueError(f"spot ({spot0}) must be > lower_barrier ({lower_barrier}).")
        if spot0 >= upper_barrier:
            raise ValueError(f"spot ({spot0}) must be < upper_barrier ({upper_barrier}).")

        # -----------------------------
        # Resolve discount factors and rates
        # -----------------------------
        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))  # Domestic DF.
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))   # Foreign DF.

        r_d = _rate_from_df(df=df_d, t=maturity)  # Domestic rate.
        r_f = _rate_from_df(df=df_f, t=maturity)  # Foreign rate.

        drift = float(r_d - r_f)  # Domestic-measure drift.

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # -----------------------------
        # Build payoff from payoff library
        # -----------------------------
        payoff_fn = require_path_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Handle deterministic expiry (T=0)
        # -----------------------------
        if maturity == 0.0:
            # Single observation at S0.
            paths = np.array([[spot0]], dtype=np.float64)
            terminal_spots = np.array([spot0], dtype=np.float64)
            min_spots = np.array([spot0], dtype=np.float64)
            max_spots = np.array([spot0], dtype=np.float64)

            # Check corridor exit (should be False if spot inside corridor).
            exited = (spot0 <= lower_barrier) | (spot0 >= upper_barrier)
            exited_corridor = np.array([exited], dtype=bool)

            payoff = payoff_fn.terminal_from_paths(paths)
            discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

            return FxDoubleBarrierMcSimulation(
                spot0=spot0,
                strike=strike,
                lower_barrier=lower_barrier,
                upper_barrier=upper_barrier,
                barrier_style=barrier_style,
                rebate_amount=rebate_amount,
                maturity=maturity,
                df_domestic=df_d,
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
                min_spots=min_spots,
                max_spots=max_spots,
                exited_corridor=exited_corridor,
                paths=paths.copy() if store_paths else None,
            )

        # -----------------------------
        # Generate standard normals
        # -----------------------------
        rng = NormalRng(seed=self.seed)
        normals = rng.standard_normals(
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])

        # -----------------------------
        # Simulate GBM paths
        # -----------------------------
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)
        all_paths = simulator.simulate_paths(
            spot0=spot0,
            maturity=maturity,
            n_steps=self.n_steps,
            n_paths=n_paths_eff,
            normals=normals,
            scheme=self.scheme,
            dtype=np.float64,
        )

        # -----------------------------
        # Extract path statistics
        # -----------------------------
        terminal_spots = all_paths[:, -1].copy()        # S_T per path.
        min_spots = np.min(all_paths, axis=1)           # min(S_t) per path.
        max_spots = np.max(all_paths, axis=1)           # max(S_t) per path.

        # Determine if corridor was exited.
        hit_lower = (min_spots <= lower_barrier)
        hit_upper = (max_spots >= upper_barrier)
        exited_corridor = hit_lower | hit_upper

        # -----------------------------
        # Compute payoffs from full paths
        # -----------------------------
        payoff = payoff_fn.terminal_from_paths(all_paths)  # Per unit notional, domestic.

        # Discount and scale by notional.
        discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

        # -----------------------------
        # Optionally retain paths for plotting
        # -----------------------------
        if not store_paths:
            kept_paths = None
        else:
            if paths_keep < 0:
                raise ValueError("paths_keep must be >= 0.")
            if paths_keep == 0:
                kept_paths = all_paths.copy()
            else:
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()

        # -----------------------------
        # Return simulation artifact
        # -----------------------------
        return FxDoubleBarrierMcSimulation(
            spot0=spot0,
            strike=strike,
            lower_barrier=lower_barrier,
            upper_barrier=upper_barrier,
            barrier_style=barrier_style,
            rebate_amount=rebate_amount,
            maturity=maturity,
            df_domestic=df_d,
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
            min_spots=min_spots,
            max_spots=max_spots,
            exited_corridor=exited_corridor,
            paths=kept_paths,
        )


@dataclass(frozen=True, slots=True)
class FxEuropeanTouchMcPricer:
    """
    Monte Carlo pricer for European FX touch options.

    This pricer simulates GBM paths and evaluates whether the barrier is
    touched, paying a fixed amount based on touch style.

    Pricing Formula
    ---------------
    PV = notional * df_d(T) * E[Payoff]

    where Payoff is computed using the TouchPayoff from the payoff library.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths. Default 200,000.
    seed : int or None
        RNG seed for reproducibility. Default 7.
    antithetic : bool
        Whether to use antithetic variates. Default True.
    n_steps : int
        Number of time steps. Default 64.
    scheme : GbmScheme
        GBM simulation scheme. Default "exact".
    """

    n_paths: int = 200_000
    seed: Optional[int] = 7
    antithetic: bool = True
    n_steps: int = 64
    scheme: GbmScheme = "exact"

    def price(self, trade: EuropeanFxTouchOption, market: Market) -> float:
        """
        Compute the present value of the touch option.

        Parameters
        ----------
        trade : EuropeanFxTouchOption
            The touch option instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        float
            Present value in domestic currency.
        """
        sim = self.run(trade, market, store_paths=False)
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: EuropeanFxTouchOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxTouchMcSimulation:
        """
        Run the full simulation and return artifact with all details.

        Parameters
        ----------
        trade : EuropeanFxTouchOption
            The touch option instrument.
        market : Market
            Market snapshot.
        store_paths : bool
            Whether to store paths. Default False.
        paths_keep : int
            Number of paths to keep if storing. 0 = all.

        Returns
        -------
        FxTouchMcSimulation
            Simulation artifact.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)

    def sample_terminal_spots(self, trade: EuropeanFxTouchOption, market: Market) -> np.ndarray:
        """Return terminal spots S(T)."""
        return self.run(trade, market, store_paths=False).terminal_spots

    def sample_discounted_payoffs(self, trade: EuropeanFxTouchOption, market: Market) -> np.ndarray:
        """Return discounted payoff samples."""
        return self.run(trade, market, store_paths=False).discounted_payoffs

    def _run_simulation(
        self,
        trade: EuropeanFxTouchOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxTouchMcSimulation:
        """Internal simulation implementation."""
        # Validate pricer configuration.
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        # Read trade inputs.
        spot0 = float(market.quote(trade.spot_id))
        barrier_level = float(trade.barrier_level)
        barrier_direction: BarrierDirection = trade.barrier_direction  # type: ignore[assignment]
        touch_style: TouchStyle = trade.touch_style  # type: ignore[assignment]
        payout_amount = float(trade.payout_amount)
        maturity = float(trade.expiry)
        notional = float(trade.notional)

        # Validate inputs.
        if maturity < 0.0:
            raise ValueError("expiry must be >= 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")
        if spot0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if barrier_level <= 0.0:
            raise ValueError("barrier_level must be > 0.")
        if payout_amount < 0.0:
            raise ValueError("payout_amount must be >= 0.")

        # Resolve discount factors.
        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))

        r_d = _rate_from_df(df=df_d, t=maturity)
        r_f = _rate_from_df(df=df_f, t=maturity)
        drift = float(r_d - r_f)

        # Resolve implied volatility (use barrier as strike for vol lookup).
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=barrier_level))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Build payoff from payoff library.
        payoff_fn = require_path_payoff(build_payoff_1d(trade))

        # Handle deterministic expiry (T=0).
        if maturity == 0.0:
            paths = np.array([[spot0]], dtype=np.float64)
            terminal_spots = np.array([spot0], dtype=np.float64)

            # Check if barrier touched at T=0.
            if barrier_direction == "up":
                touched_val = (spot0 >= barrier_level)
            else:
                touched_val = (spot0 <= barrier_level)
            touched = np.array([touched_val], dtype=bool)

            payoff = payoff_fn.terminal_from_paths(paths)
            discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

            return FxTouchMcSimulation(
                spot0=spot0,
                barrier_level=barrier_level,
                barrier_direction=barrier_direction,
                touch_style=touch_style,
                payout_amount=payout_amount,
                maturity=maturity,
                df_domestic=df_d,
                drift=drift,
                sigma=sigma,
                notional=notional,
                n_paths_requested=self.n_paths,
                n_paths_effective=1,
                n_steps=0,
                scheme=self.scheme,
                antithetic=self.antithetic,
                seed=self.seed,
                terminal_spots=terminal_spots,
                discounted_payoffs=discounted_payoffs,
                touched=touched,
                paths=paths.copy() if store_paths else None,
            )

        # Generate standard normals.
        rng = NormalRng(seed=self.seed)
        normals = rng.standard_normals(
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])

        # Simulate GBM paths.
        simulator = GbmDynamicsSimulator(drift=drift, vol=sigma)
        all_paths = simulator.simulate_paths(
            spot0=spot0,
            maturity=maturity,
            n_steps=self.n_steps,
            n_paths=n_paths_eff,
            normals=normals,
            scheme=self.scheme,
            dtype=np.float64,
        )

        # Extract terminal spots and touch status.
        terminal_spots = all_paths[:, -1].copy()

        if barrier_direction == "up":
            max_s = np.max(all_paths, axis=1)
            touched = (max_s >= barrier_level)
        else:
            min_s = np.min(all_paths, axis=1)
            touched = (min_s <= barrier_level)

        # Compute payoffs.
        payoff = payoff_fn.terminal_from_paths(all_paths)
        discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

        # Store paths if requested.
        if not store_paths:
            kept_paths = None
        else:
            if paths_keep < 0:
                raise ValueError("paths_keep must be >= 0.")
            if paths_keep == 0:
                kept_paths = all_paths.copy()
            else:
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()

        return FxTouchMcSimulation(
            spot0=spot0,
            barrier_level=barrier_level,
            barrier_direction=barrier_direction,
            touch_style=touch_style,
            payout_amount=payout_amount,
            maturity=maturity,
            df_domestic=df_d,
            drift=drift,
            sigma=sigma,
            notional=notional,
            n_paths_requested=self.n_paths,
            n_paths_effective=n_paths_eff,
            n_steps=self.n_steps,
            scheme=self.scheme,
            antithetic=self.antithetic,
            seed=self.seed,
            terminal_spots=terminal_spots,
            discounted_payoffs=discounted_payoffs,
            touched=touched,
            paths=kept_paths,
        )