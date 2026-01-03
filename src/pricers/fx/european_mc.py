# src/pricers/fx/european_mc.py
from __future__ import annotations  # Enable forward references.

import numpy as np  # NumPy for fast vectorized Monte Carlo operations.
from dataclasses import dataclass  # Dataclasses for small immutable pricer/config objects.
from typing import Literal, Optional  # Literal for payoff-type tags; Optional for seed/paths.

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption  # FX vanilla instrument.
from src.instruments.fx.options.digital import EuropeanFxDigitalOption  # FX digital instrument.
from src.marketdata.market import Market  # Market snapshot interface.
from src.models.numeric.monte_carlo.rng import NormalRng  # Reproducible normal RNG.
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme  # GBM simulator and scheme.
from src.pricers.fx.european_bsm import _rate_from_df  # DF -> continuous rate helper (shared with BSM adapters).

from src.models.payoffs.types import OptionType  # Canonical option type ("call"/"put").
from src.models.payoffs.vanilla import VanillaPayoff  # Vanilla payoff from payoff library.
from src.models.payoffs.digital import DigitalCashPayoff, DigitalAssetPayoff  # Digital payoffs from payoff library.


DigitalPayoff = Literal["cash", "asset"]  # Digital payoff styles on the instrument.


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

        payoff_fn = VanillaPayoff(option_type=option_type, strike=strike)  # Build payoff function from payoff library.

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
        if payoff_style == "cash":  # Cash-or-nothing digital...
            payoff_fn = DigitalCashPayoff(option_type=option_type, strike=strike, cash=payout_amount)  # cash*1{ITM}
        elif payoff_style == "asset":  # Asset-or-nothing digital...
            payoff_fn = DigitalAssetPayoff(option_type=option_type, strike=strike, asset_units=payout_amount)  # units*S*1{ITM}
        else:  # Defensive branch (should be prevented by instrument validation).
            raise ValueError(f"Unsupported digital payoff style: {payoff_style!r}.")  # Fail fast.

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