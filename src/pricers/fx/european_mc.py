# src/pricers/fx/european_mc.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.marketdata.market import Market
from src.models.numeric.monte_carlo.rng import NormalRng
from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator, GbmScheme
from src.pricers.fx.european_bsm import _rate_from_df


OptionType = Literal["call", "put"]


@dataclass(frozen=True, slots=True)
class FxMcSimulation:
    """
    Reusable Monte Carlo simulation artifact for a single trade priced on a single Market snapshot.

    Purpose
    -------
    This object exists so you can:
      - compute PV (mean of discounted payoffs)
      - plot terminal spot distribution S(T)
      - plot discounted payoff distribution
      - plot a subset (or all) simulated paths
    without re-running the Monte Carlo simulation.

    Notes
    -----
    - `discounted_payoffs` are in *domestic currency* and already scaled by `trade.notional`.
    - `paths` is optional and can be very large; prefer storing a small subset for plotting.
    """

    # Inputs / resolved market parameters (useful for reporting)
    spot0: float
    strike: float
    maturity: float
    df_domestic: float
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
    terminal_spots: np.ndarray            # shape (n_paths_effective,)
    discounted_payoffs: np.ndarray        # shape (n_paths_effective,)
    paths: Optional[np.ndarray] = None    # shape (n_paths_kept, n_steps+1) if stored


@dataclass(frozen=True, slots=True)
class FxEuropeanVanillaMcPricer:
    """
    Monte Carlo pricer for European FX vanilla options under Garman–Kohlhagen.

    Mapping (consistent with FxEuropeanVanillaBsmPricer)
    ---------------------------------------------------
    - Domestic and foreign curves provide discount factors df_d(T), df_f(T)
    - Rates:
        r_d = -ln(df_d)/T
        r_f = -ln(df_f)/T
    - Under the domestic measure:
        dS/S = (r_d - r_f) dt + sigma dW
    - PV (domestic):
        PV = notional * df_d(T) * E[ payoff(S_T) ]

    Implementation
    --------------
    - Uses your GBM simulator: `GbmDynamicsSimulator`
    - Uses your RNG: `NormalRng` (supports antithetic variates)
    - Default is scheme="exact" and n_steps=1 (exact terminal distribution for GBM)

    Reporting / analysis support
    ----------------------------
    Call `run(...)` to get an `FxMcSimulation` object containing terminal spots and
    discounted payoffs (and optionally paths), so you can create plots without
    re-simulating.
    """

    # Core MC controls
    n_paths: int = 200_000
    seed: Optional[int] = 7
    antithetic: bool = True

    # Dynamics controls (kept configurable for scheme comparisons / future extensions)
    n_steps: int = 1
    scheme: GbmScheme = "exact"

    def price(self, trade: EuropeanFxVanillaOption, market: Market) -> float:
        """
        Return PV in domestic currency.

        Notes
        -----
        This method intentionally does *not* store paths. It is the fast/default
        pricing path and is safe to use inside PortfolioPricer / risk runs.
        """
        sim = self.run(trade, market, store_paths=False)
        return float(sim.discounted_payoffs.mean())

    def run(
        self,
        trade: EuropeanFxVanillaOption,
        market: Market,
        *,
        store_paths: bool = False,
        paths_keep: int = 0,
    ) -> FxMcSimulation:
        """
        Run Monte Carlo once and return a reusable simulation artifact.

        Parameters
        ----------
        trade:
            European FX vanilla option (call/put).
        market:
            Market snapshot providing spot, curves, and vol surface.
        store_paths:
            If True, also store simulated paths for plotting.
        paths_keep:
            If `store_paths=True`, controls how many paths are retained:
              - paths_keep > 0 : store only the first `paths_keep` paths (recommended)
              - paths_keep == 0: store *all* paths (can be very large)

        Returns
        -------
        FxMcSimulation
            Terminal spots and discounted payoff samples (and optionally paths).

        Production guidance
        -------------------
        - For reporting, prefer `store_paths=True, paths_keep=200..2000` (plotting).
        - For analytics like terminal distributions / CI, paths are not needed.
        """
        return self._run_simulation(trade, market, store_paths=store_paths, paths_keep=paths_keep)

    def sample_terminal_spots(self, trade: EuropeanFxVanillaOption, market: Market) -> np.ndarray:
        """
        Convenience helper: return simulated terminal spots S(T).

        This delegates to `run(...)` and does not duplicate simulation logic.
        """
        return self.run(trade, market, store_paths=False).terminal_spots

    def sample_discounted_payoffs(self, trade: EuropeanFxVanillaOption, market: Market) -> np.ndarray:
        """
        Convenience helper: return discounted payoff samples (domestic), scaled by notional.

        This delegates to `run(...)` and does not duplicate simulation logic.
        """
        return self.run(trade, market, store_paths=False).discounted_payoffs

    # ---------------------------------------------------------------------
    # Internal implementation (single source of truth for simulation)
    # ---------------------------------------------------------------------

    def _run_simulation(
        self,
        trade: EuropeanFxVanillaOption,
        market: Market,
        *,
        store_paths: bool,
        paths_keep: int,
    ) -> FxMcSimulation:
        # -----------------------------
        # Validate pricer configuration
        # -----------------------------
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")

        # -----------------------------
        # Read trade inputs
        # -----------------------------
        option_type: OptionType = trade.option_type  # type: ignore[assignment]
        spot0 = float(market.quote(trade.spot_id))
        strike = float(trade.strike)
        maturity = float(trade.expiry)
        notional = float(trade.notional)

        if maturity < 0.0:
            raise ValueError("expiry must be >= 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")

        # -----------------------------
        # Resolve discount factors and rates
        # -----------------------------
        df_d = float(market.curve(trade.domestic_curve_id).df(maturity))
        df_f = float(market.curve(trade.foreign_curve_id).df(maturity))

        r_d = _rate_from_df(df=df_d, t=maturity)
        r_f = _rate_from_df(df=df_f, t=maturity)

        # Domestic-measure drift for FX spot under GK:
        drift = float(r_d - r_f)

        # -----------------------------
        # Resolve implied volatility
        # -----------------------------
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=maturity, strike=strike))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # -----------------------------
        # Deterministic expiry handling
        # -----------------------------
        if maturity == 0.0:
            # At expiry, S(T) is just spot0 and discount factor is 1.0 in theory.
            if option_type == "call":
                payoff = max(spot0 - strike, 0.0)
            else:
                payoff = max(strike - spot0, 0.0)

            terminal_spots = np.array([spot0], dtype=np.float64)
            discounted_payoffs = np.array([notional * payoff], dtype=np.float64)

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

        # -----------------------------
        # Generate standard normals
        # -----------------------------
        # Shape required by simulator: (n_paths, n_steps)
        rng = NormalRng(seed=self.seed)
        normals = rng.standard_normals(
            self.n_paths,
            self.n_steps,
            antithetic=self.antithetic,
            dtype=np.float64,
        )
        n_paths_eff = int(normals.shape[0])  # antithetic may round up to an even number

        # -----------------------------
        # Simulate GBM paths (vectorized)
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

        # Terminal spot for each path
        terminal_spots = all_paths[:, -1].copy()

        # -----------------------------
        # Compute payoff per path (undiscounted, per 1 unit foreign)
        # -----------------------------
        if option_type == "call":
            payoff = np.maximum(terminal_spots - strike, 0.0)
        else:
            payoff = np.maximum(strike - terminal_spots, 0.0)

        # Discount domestically and scale by notional (domestic PV samples)
        discounted_payoffs = (float(df_d) * payoff * notional).astype(np.float64, copy=False)

        # -----------------------------
        # Optionally retain paths for plotting
        # -----------------------------
        kept_paths: Optional[np.ndarray]
        if not store_paths:
            kept_paths = None
        else:
            # Store a subset by default (recommended). If paths_keep==0, store all.
            if paths_keep < 0:
                raise ValueError("paths_keep must be >= 0.")
            if paths_keep == 0:
                kept_paths = all_paths.copy()
            else:
                kept_paths = all_paths[: min(paths_keep, n_paths_eff), :].copy()

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