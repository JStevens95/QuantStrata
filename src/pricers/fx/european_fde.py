from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

# Project imports: instruments + market snapshot
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.instruments.fx.options.digital import EuropeanFxDigitalOption
from src.marketdata.market import Market

# Project imports: finite-difference engine building blocks
from src.models.numeric.finite_difference.boundaries import BoundaryPair, DirichletBC
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.schemes import ThetaScheme, solve_pde_theta
from src.models.numeric.finite_difference.diagnostics import FdDiagnostics

# Project imports: payoff library (terminal conditions)
from src.models.payoffs.types import OptionType
from src.models.payoffs.base import BasePayoff1D
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff

# Keep rate/DF conversion consistent across BSM/MC/FD pricers
from src.pricers.fx.european_bsm import _rate_from_df


# ======================================================================================
# Shared type aliases
# ======================================================================================

# Output keys for greeks (kept aligned with your BSM / MC adaptors)
GreekName = Literal["delta", "gamma", "vega", "rho_domestic", "rho_foreign"]


# ======================================================================================
# Vanilla FD pricer (European FX vanilla) using payoff library
# ======================================================================================

@dataclass(frozen=True, slots=True)
class FxEuropeanVanillaFdPricer:
    """
    Finite-difference (PDE) pricer for European FX vanilla options under Garman–Kohlhagen.

    What is being solved?
    ---------------------
    Under GK with constant parameters (r_d, r_f, sigma), the option value V(t,S)
    solves a Black–Scholes type PDE. We solve it backward from terminal payoff.

    Where the payoff library fits
    -----------------------------
    The payoff library defines the terminal condition:
        V(T, S) = payoff(S)
    so this pricer should *not* duplicate payoff logic inline.
    """

    # -----------------------------
    # Grid controls (accuracy knobs)
    # -----------------------------
    n_space: int = 401
    n_time_steps: int = 200
    n_std: float = 6.0

    # -----------------------------
    # Time-stepping scheme
    # -----------------------------
    theta: float = 0.5  # 0.5 = Crank–Nicolson, 1.0 = fully implicit

    # -----------------------------
    # Work in log-space (recommended)
    # -----------------------------
    use_log_space: bool = True

    # -----------------------------
    # Greek bump sizes (V1)
    # -----------------------------
    spot_rel_bump: float = 1e-4
    vol_abs_bump: float = 1e-4
    rate_abs_bump: float = 1e-4

    # =====================================================================
    # Public API
    # =====================================================================

    def price(self, trade: EuropeanFxVanillaOption, market: Market) -> float:
        """
        Price in domestic currency, scaled by trade.notional.

        Implementation detail
        ---------------------
        We solve PDE for *per-unit* value, then multiply by notional.
        """
        # Solve once (per unit notional) and retrieve PV at spot via interpolation.
        pv_per_unit, _ctx = self._price_per_unit_and_context(trade, market)

        # Scale to trade notional (foreign units notionally, consistent with your BSM/MC conventions).
        return float(trade.notional) * float(pv_per_unit)

    def greeks(self, trade: EuropeanFxVanillaOption, market: Market) -> Dict[GreekName, float]:
        """
        Compute FD greeks (scaled by notional).

        Policy (V1 stability):
        ----------------------
        - Delta/Gamma: extracted from the same solved surface (no re-solve)
        - Vega/Rhos: bump parameter and re-solve on *same grids*
        """
        # -----------------------------
        # Read trade inputs and validate
        # -----------------------------
        option_type: OptionType = trade.option_type  # type: ignore[assignment]
        notional = float(trade.notional)

        S0 = float(market.quote(trade.spot_id))    # spot at valuation
        K = float(trade.strike)                    # strike
        T = float(trade.expiry)                    # time-to-expiry

        # Basic sanity checks to avoid undefined behaviour in PDE / log.
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")

        # At expiry, greeks are not stable for kinked payoff (discontinuity in derivative).
        # Returning zeros keeps portfolio/risk tooling stable.
        if T == 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho_domestic": 0.0, "rho_foreign": 0.0}

        # -----------------------------
        # Base solve + context (grids + parameters)
        # -----------------------------
        pv0_per_unit, ctx = self._price_per_unit_and_context(trade, market)

        # Extract the pieces we need so bumped re-solves reuse the same grids.
        x_grid: SpatialGrid1D = ctx["x_grid"]  # type: ignore[assignment]
        t_grid: TimeGrid = ctx["t_grid"]       # type: ignore[assignment]
        x0 = float(ctx["x0"])                  # coordinate corresponding to S0 on the chosen grid
        r_d0 = float(ctx["r_d"])               # domestic rate (continuous)
        r_f0 = float(ctx["r_f"])               # foreign rate (continuous)
        sigma0 = float(ctx["sigma"])           # implied vol used for PDE
        S_min = float(ctx["S_min"])            # left boundary spot
        S_max = float(ctx["S_max"])            # right boundary spot

        # Build terminal payoff via payoff library (single source of truth).
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Solve PDE once at base parameters (reused for delta/gamma extraction).
        sol0 = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        )

        # -----------------------------
        # Delta/Gamma (no re-solve)
        # -----------------------------
        # Extract derivatives from the solved t=0 slice around the query point.
        delta_per_unit, gamma_per_unit = self._delta_gamma_from_solution(sol=sol0, x_grid=x_grid, xq=x0)

        # -----------------------------
        # Vega (bump sigma + re-solve)
        # -----------------------------
        eps_v = float(self.vol_abs_bump)           # absolute vol bump
        sigma_up = max(0.0, sigma0 + eps_v)        # enforce non-negative vol
        sigma_dn = max(0.0, sigma0 - eps_v)

        # Re-solve PDE with sigma bumped up, evaluate PV at x0
        pv_sig_up = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0,
            sigma=sigma_up,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        # Re-solve PDE with sigma bumped down, evaluate PV at x0
        pv_sig_dn = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0,
            sigma=sigma_dn,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        # Central-difference derivative w.r.t. sigma
        vega_per_unit = (float(pv_sig_up) - float(pv_sig_dn)) / (2.0 * eps_v)

        # -----------------------------
        # Rhos (bump r_d and r_f + re-solve)
        # -----------------------------
        eps_r = float(self.rate_abs_bump)  # absolute rate bump

        # Domestic rho: bump r_d (keep r_f fixed)
        pv_rd_up = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d0 + eps_r,
            r_f=r_f0,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        pv_rd_dn = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d0 - eps_r,
            r_f=r_f0,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        rho_domestic_per_unit = (float(pv_rd_up) - float(pv_rd_dn)) / (2.0 * eps_r)

        # Foreign rho: bump r_f (keep r_d fixed)
        pv_rf_up = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0 + eps_r,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        pv_rf_dn = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0 - eps_r,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        rho_foreign_per_unit = (float(pv_rf_up) - float(pv_rf_dn)) / (2.0 * eps_r)

        # -----------------------------
        # Scale per-unit greeks by notional and return
        # -----------------------------
        return {
            "delta": float(notional * delta_per_unit),
            "gamma": float(notional * gamma_per_unit),
            "vega": float(notional * vega_per_unit),
            "rho_domestic": float(notional * rho_domestic_per_unit),
            "rho_foreign": float(notional * rho_foreign_per_unit),
        }

    def diagnostics(
        self,
        trade: EuropeanFxVanillaOption,
        market: Market,
        *,
        store_surface: bool = True,
    ) -> FdDiagnostics:
        """
        Provide grids and solved t=0 curve for plotting and debugging.

        This avoids duplicating PDE setup logic in example notebooks/scripts.
        """
        option_type: OptionType = trade.option_type  # type: ignore[assignment]

        # Extract trade parameters
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # Pull discount factors at expiry from curves
        df_d_T = float(market.curve(trade.domestic_curve_id).df(T))
        df_f_T = float(market.curve(trade.foreign_curve_id).df(T))

        # Convert df -> continuously compounded rates
        r_d = float(_rate_from_df(df=df_d_T, t=T))
        r_f = float(_rate_from_df(df=df_f_T, t=T))

        # Use constant vol at (T, K) consistent with BSM PDE
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma <= 0.0:
            # A zero-vol “surface” is trivial and not meaningful to visualize as PDE surface.
            raise ValueError("diagnostics() expects sigma > 0 so the surface is meaningful.")

        # Choose domain bounds based on lognormal width around S0
        width = float(self.n_std) * float(sigma) * math.sqrt(float(T))
        S_min = max(1e-12, S0 * math.exp(-width))
        S_max = S0 * math.exp(+width)

        # Ensure K is inside domain for good accuracy around payoff kink
        if K > 0.0:
            S_min = min(S_min, 0.5 * K)
            S_max = max(S_max, 2.0 * K)

        # Build spatial grid in log-space or spot-space
        if self.use_log_space:
            x_grid = SpatialGrid1D.log_uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="logS")
            x0 = float(math.log(S0))
        else:
            x_grid = SpatialGrid1D.uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="S")
            x0 = float(S0)

        # Build time grid from 0..T with n_time_steps+1 nodes
        t_grid = TimeGrid.uniform(t0=0.0, t1=float(T), n=int(self.n_time_steps) + 1, name="t")

        # Create payoff from payoff library (terminal condition)
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Solve PDE and keep surface if requested (surface is big -> optional)
        sol = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d,
            r_f=r_f,
            sigma=sigma,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=float(S_min),
            S_right=float(S_max),
            store_surface=bool(store_surface),
        )

        # For plotting convenience, expose "spot grid" as linear-space spot values
        spot_grid = np.asarray(x_grid.linear_space_values(), dtype=np.float64)
        time_grid = np.asarray(t_grid.t, dtype=np.float64)

        return FdDiagnostics(
            x_grid=x_grid,
            t_grid=t_grid,
            spot_grid=spot_grid,
            time_grid=time_grid,
            values_t0_per_unit=np.asarray(sol.values_t0, dtype=np.float64),
            surface_per_unit=None if sol.surface is None else np.asarray(sol.surface, dtype=np.float64),
            spot0=float(S0),
            strike=float(K),
            expiry=float(T),
            r_d=float(r_d),
            r_f=float(r_f),
            sigma=float(sigma),
            x0=float(x0),
            meta={
                "n_space": float(self.n_space),
                "n_time_steps": float(self.n_time_steps),
                "theta": float(self.theta),
                "use_log_space": bool(self.use_log_space),
            },
        )

    # =====================================================================
    # Internal helpers (single-source-of-truth PDE setup)
    # =====================================================================

    def _price_per_unit_and_context(
        self,
        trade: EuropeanFxVanillaOption,
        market: Market,
    ) -> Tuple[float, Dict[str, object]]:
        """
        Solve PDE (or use degenerate shortcuts) and return:
          - PV per unit notional
          - context (grids + params) so greeks can reuse grids/bounds
        """
        option_type: OptionType = trade.option_type  # type: ignore[assignment]

        # Read spot/strike/expiry
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # Validate
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0 for log-space PDE.")

        # Build payoff object
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # If T == 0, PV is simply payoff(S0) (no discounting)
        if T == 0.0:
            pv0 = float(payoff.terminal(np.asarray([S0], dtype=np.float64))[0])
            return pv0, {"x_grid": None, "t_grid": None}

        # Pull discount factors from curves
        df_d_T = float(market.curve(trade.domestic_curve_id).df(T))
        df_f_T = float(market.curve(trade.foreign_curve_id).df(T))

        # Convert DFs to continuous rates
        r_d = float(_rate_from_df(df=df_d_T, t=T))
        r_f = float(_rate_from_df(df=df_f_T, t=T))

        # Constant volatility consistent with PDE assumptions
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Zero-vol shortcut: terminal spot is deterministic at forward F0
        if sigma == 0.0:
            F0 = S0 * math.exp((r_d - r_f) * T)
            disc = math.exp(-r_d * T)
            pv0 = disc * float(payoff.terminal(np.asarray([F0], dtype=np.float64))[0])
            return float(pv0), {"x_grid": None, "t_grid": None}

        # Choose lognormal-width domain around S0
        width = float(self.n_std) * float(sigma) * math.sqrt(float(T))
        S_min = max(1e-12, S0 * math.exp(-width))
        S_max = S0 * math.exp(+width)

        # Ensure strike is comfortably inside domain
        if K > 0.0:
            S_min = min(S_min, 0.5 * K)
            S_max = max(S_max, 2.0 * K)

        # Build spatial grid
        if self.n_space < 3:
            raise ValueError("n_space must be >= 3.")
        if self.use_log_space:
            x_grid = SpatialGrid1D.log_uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="logS")
            x0 = float(math.log(S0))
        else:
            x_grid = SpatialGrid1D.uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="S")
            x0 = float(S0)

        # Build time grid
        n_nodes = int(self.n_time_steps) + 1
        if n_nodes < 2:
            raise ValueError("n_time_steps must be >= 1.")
        t_grid = TimeGrid.uniform(t0=0.0, t1=float(T), n=n_nodes, name="t")

        # Solve PDE once (base solve)
        sol = self._solve_on_grids(
            payoff=payoff,
            option_type=option_type,
            K=K,
            T=T,
            r_d=r_d,
            r_f=r_f,
            sigma=sigma,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=float(S_min),
            S_right=float(S_max),
        )

        # Interpolate PV at the point corresponding to spot S0
        pv_per_unit = float(sol.value_at(x0))

        # Defensive numerical check
        if not math.isfinite(pv_per_unit):
            raise ValueError(f"Non-finite FD PV per unit computed: {pv_per_unit}")

        # Context allows re-using same grids for bumped re-solves
        ctx: Dict[str, object] = {
            "x_grid": x_grid,
            "t_grid": t_grid,
            "x0": x0,
            "r_d": r_d,
            "r_f": r_f,
            "sigma": sigma,
            "S_min": float(S_min),
            "S_max": float(S_max),
        }
        return pv_per_unit, ctx

    def _solve_on_grids(
        self,
        *,
        payoff: BasePayoff1D,
        option_type: OptionType,
        K: float,
        T: float,
        r_d: float,
        r_f: float,
        sigma: float,
        x_grid: SpatialGrid1D,
        t_grid: TimeGrid,
        S_left: float,
        S_right: float,
        store_surface: bool = False,
    ):
        """
        Solve the GK PDE on fixed grids.

        This method is the *single source of truth* for:
          - terminal payoff mapping
          - boundary conditions (vanilla asymptotics)
          - PDE coefficient functions (log-space vs spot-space)
          - theta-scheme stepping call
        """

        # Terminal condition function V(T, x) evaluated on the spatial grid nodes.
        def terminal_payoff(x: np.ndarray) -> np.ndarray:
            # Convert grid coordinate x to spot S depending on grid type.
            spot = np.exp(x) if x_grid.is_log_space else x
            # Delegate to payoff library.
            return payoff.terminal(spot).astype(np.float64, copy=False)

        # Discount factors to go from terminal time T back to intermediate time t.
        def _df_dom_tau(t: float) -> float:
            tau = max(0.0, float(T) - float(t))
            return float(math.exp(-float(r_d) * tau))

        def _df_for_tau(t: float) -> float:
            tau = max(0.0, float(T) - float(t))
            return float(math.exp(-float(r_f) * tau))

        # Vanilla boundary conditions:
        # - For call: V -> 0 as S -> 0 ; V -> S*df_f - K*df_d as S -> inf
        # - For put : V -> K*df_d - S*df_f as S -> 0 ; V -> 0 as S -> inf
        if option_type == "call":
            left_bc = DirichletBC(side="left", value=lambda _t: 0.0)
            right_bc = DirichletBC(
                side="right",
                value=lambda t: (float(S_right) * _df_for_tau(t)) - (float(K) * _df_dom_tau(t)),
            )
        else:
            left_bc = DirichletBC(
                side="left",
                value=lambda t: max(0.0, (float(K) * _df_dom_tau(t)) - (float(S_left) * _df_for_tau(t))),
            )
            right_bc = DirichletBC(side="right", value=lambda _t: 0.0)

        # Bundle boundary conditions for solver
        boundaries = BoundaryPair(left=left_bc, right=right_bc)

        # PDE coefficients:
        # In log-space (x=ln S) the PDE coefficients are constants:
        #   a = (r_d - r_f) - 0.5*sigma^2
        #   b = 0.5*sigma^2
        #   c = r_d
        if x_grid.is_log_space:
            a = float((r_d - r_f) - 0.5 * sigma * sigma)
            b = float(0.5 * sigma * sigma)
            c = float(r_d)
        else:
            # In spot-space (x=S) coefficients depend on x:
            #   a(x) = (r_d - r_f) * x
            #   b(x) = 0.5 * sigma^2 * x^2
            def a(x: np.ndarray, _t: float) -> np.ndarray:
                return (float(r_d - r_f) * x).astype(np.float64, copy=False)

            def b(x: np.ndarray, _t: float) -> np.ndarray:
                return (0.5 * float(sigma) * float(sigma) * x * x).astype(np.float64, copy=False)

            c = float(r_d)

        # Choose theta scheme (CN or implicit)
        scheme = ThetaScheme(theta=float(self.theta))

        # Solve PDE via your engine
        return solve_pde_theta(
            x_grid=x_grid,
            t_grid=t_grid,
            terminal_payoff=terminal_payoff,
            a=a,
            b=b,
            c=c,
            boundaries=boundaries,
            scheme=scheme,
            store_surface=bool(store_surface),
        )

    # -----------------------------
    # Derivatives helpers for delta/gamma extraction
    # -----------------------------

    @staticmethod
    def _quad_derivatives_at(
        x0: float,
        x1: float,
        x2: float,
        y0: float,
        y1: float,
        y2: float,
        xq: float,
    ) -> Tuple[float, float]:
        """
        Differentiate the quadratic interpolant through three points and evaluate at xq.

        Returns:
          (first_derivative, second_derivative)
        """
        den0 = (x0 - x1) * (x0 - x2)
        den1 = (x1 - x0) * (x1 - x2)
        den2 = (x2 - x0) * (x2 - x1)

        if den0 == 0.0 or den1 == 0.0 or den2 == 0.0:
            raise ValueError("Degenerate quadratic stencil (duplicate x nodes).")

        # First derivatives of Lagrange basis polynomials at xq
        dL0 = (2.0 * xq - x1 - x2) / den0
        dL1 = (2.0 * xq - x0 - x2) / den1
        dL2 = (2.0 * xq - x0 - x1) / den2

        # Second derivatives of Lagrange basis polynomials are constant = 2/den
        d2L0 = 2.0 / den0
        d2L1 = 2.0 / den1
        d2L2 = 2.0 / den2

        # Combine basis derivatives to get derivatives of interpolant
        dy = y0 * dL0 + y1 * dL1 + y2 * dL2
        d2y = y0 * d2L0 + y1 * d2L1 + y2 * d2L2
        return float(dy), float(d2y)

    def _delta_gamma_from_solution(
        self,
        *,
        sol,
        x_grid: SpatialGrid1D,
        xq: float,
    ) -> Tuple[float, float]:
        """
        Compute delta and gamma from the solved t=0 curve.

        Approach:
        - Choose local stencil of three grid nodes around xq
        - Differentiate quadratic interpolant to get dv/dx and d2v/dx2
        - If log-space, convert to S-derivatives via chain rule:
            delta = dV/dS = (dV/dx) * (dx/dS) = (dV/dx) / S
            gamma = d2V/dS2 = (d2V/dx2 - dV/dx) / S^2
        """
        x = x_grid.x                 # grid nodes in x-space
        v = sol.values_t0            # option values at t=0 on those nodes

        # Find index j such that x[j] <= xq < x[j+1]
        j = int(np.searchsorted(x, float(xq), side="right") - 1)

        # Clamp to ensure we can form (j-1, j, j+1) stencil
        j = max(1, min(j, int(x.size) - 2))

        # Stencil nodes and values
        x0, x1, x2 = float(x[j - 1]), float(x[j]), float(x[j + 1])
        y0, y1, y2 = float(v[j - 1]), float(v[j]), float(v[j + 1])

        # Derivatives in x-space
        dv_dx, d2v_dx2 = self._quad_derivatives_at(x0, x1, x2, y0, y1, y2, float(xq))

        # If grid uses x = ln(S), convert to derivatives w.r.t. S
        if x_grid.is_log_space:
            S = float(math.exp(float(xq)))        # recover spot S from x
            delta = dv_dx / S
            gamma = (d2v_dx2 - dv_dx) / (S * S)
            return float(delta), float(gamma)

        # Otherwise x is S already
        return float(dv_dx), float(d2v_dx2)


# ======================================================================================
# Digital FD pricer (European FX digitals) using payoff library
# ======================================================================================

@dataclass(frozen=True, slots=True)
class FxEuropeanDigitalFdPricer:
    """
    Finite-difference (PDE) pricer for European FX digitals under GK.

    Payoff styles supported
    -----------------------
    - cash : DigitalCashPayoff  (pays cash in domestic currency)
    - asset: DigitalAssetPayoff (pays units * S in domestic currency)

    Practical note on convergence
    -----------------------------
    Digital payoffs are discontinuous at strike, so FD convergence is slower than
    vanilla. Expect to need higher n_space / n_time_steps (or payoff smoothing).
    """

    # Use denser defaults than vanilla to reduce oscillations / diffusion error.
    n_space: int = 801
    n_time_steps: int = 400
    n_std: float = 7.0

    theta: float = 0.5
    use_log_space: bool = True

    vol_abs_bump: float = 1e-4
    rate_abs_bump: float = 1e-4

    # =====================================================================
    # Public API
    # =====================================================================

    def price(self, trade: EuropeanFxDigitalOption, market: Market) -> float:
        """Return domestic PV (already in domestic units for both cash and asset)."""
        pv, _ctx = self._price_and_context(trade, market)
        return float(pv)

    def greeks(self, trade: EuropeanFxDigitalOption, market: Market) -> Dict[GreekName, float]:
        """
        V1 greeks for digitals (robust policy):

        - Delta/Gamma: set to 0 (unstable for discontinuous payoff unless smoothed)
        - Vega/Rhos: bump-and-reprice via FD re-solves
        """
        # Read core parameters
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # If at/after expiry, greeks are not meaningful in this stable V1 policy.
        if T <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho_domestic": 0.0, "rho_foreign": 0.0}

        # Base solve + context provides grids and parameters for bumped re-solves.
        pv0, ctx = self._price_and_context(trade, market)

        x_grid: SpatialGrid1D = ctx["x_grid"]  # type: ignore[assignment]
        t_grid: TimeGrid = ctx["t_grid"]       # type: ignore[assignment]
        x0 = float(ctx["x0"])
        r_d0 = float(ctx["r_d"])
        r_f0 = float(ctx["r_f"])
        sigma0 = float(ctx["sigma"])
        S_min = float(ctx["S_min"])
        S_max = float(ctx["S_max"])

        # Construct payoff via payoff library
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # -----------------------------
        # Vega (bump sigma)
        # -----------------------------
        eps_v = float(self.vol_abs_bump)
        sigma_up = max(0.0, sigma0 + eps_v)
        sigma_dn = max(0.0, sigma0 - eps_v)

        pv_sig_up = self._solve_on_grids_digital(
            payoff=payoff,
            trade=trade,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0,
            sigma=sigma_up,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        pv_sig_dn = self._solve_on_grids_digital(
            payoff=payoff,
            trade=trade,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0,
            sigma=sigma_dn,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        vega = (float(pv_sig_up) - float(pv_sig_dn)) / (2.0 * eps_v)

        # -----------------------------
        # Rhos (bump r_d and r_f)
        # -----------------------------
        eps_r = float(self.rate_abs_bump)

        pv_rd_up = self._solve_on_grids_digital(
            payoff=payoff,
            trade=trade,
            K=K,
            T=T,
            r_d=r_d0 + eps_r,
            r_f=r_f0,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        pv_rd_dn = self._solve_on_grids_digital(
            payoff=payoff,
            trade=trade,
            K=K,
            T=T,
            r_d=r_d0 - eps_r,
            r_f=r_f0,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        rho_domestic = (float(pv_rd_up) - float(pv_rd_dn)) / (2.0 * eps_r)

        pv_rf_up = self._solve_on_grids_digital(
            payoff=payoff,
            trade=trade,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0 + eps_r,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        pv_rf_dn = self._solve_on_grids_digital(
            payoff=payoff,
            trade=trade,
            K=K,
            T=T,
            r_d=r_d0,
            r_f=r_f0 - eps_r,
            sigma=sigma0,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
        ).value_at(x0)

        rho_foreign = (float(pv_rf_up) - float(pv_rf_dn)) / (2.0 * eps_r)

        # Return robust “V1” greek set
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": float(vega),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }

    # =====================================================================
    # Internals
    # =====================================================================

    def _price_and_context(self, trade: EuropeanFxDigitalOption, market: Market) -> Tuple[float, Dict[str, object]]:
        """
        Price digital via PDE solve (or degenerate shortcuts) and return PV + context.
        """

        # Read trade/market values
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # Sanity checks
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0 for log-space PDE.")

        # Build payoff from payoff library
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # If expiry now, PV is immediate payoff at S0 (no discounting)
        if T == 0.0:
            pv0 = float(payoff.terminal(np.asarray([S0], dtype=np.float64))[0])
            return pv0, {"x_grid": None, "t_grid": None}

        # Pull discount factors at expiry
        df_d_T = float(market.curve(trade.domestic_curve_id).df(T))
        df_f_T = float(market.curve(trade.foreign_curve_id).df(T))

        # Convert DFs to continuous rates
        r_d = float(_rate_from_df(df=df_d_T, t=T))
        r_f = float(_rate_from_df(df=df_f_T, t=T))

        # Constant vol for PDE
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Zero-vol shortcut: deterministic terminal spot is forward
        if sigma == 0.0:
            F0 = S0 * math.exp((r_d - r_f) * T)
            disc = math.exp(-r_d * T)
            pv0 = disc * float(payoff.terminal(np.asarray([F0], dtype=np.float64))[0])
            return float(pv0), {"x_grid": None, "t_grid": None}

        # Choose domain width around S0 (lognormal scale)
        width = float(self.n_std) * float(sigma) * math.sqrt(float(T))
        S_min = max(1e-12, S0 * math.exp(-width))
        S_max = S0 * math.exp(+width)

        # Ensure strike is within domain
        if K > 0.0:
            S_min = min(S_min, 0.5 * K)
            S_max = max(S_max, 2.0 * K)

        # Build spatial grid in log-space or spot-space
        if self.use_log_space:
            x_grid = SpatialGrid1D.log_uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="logS")
            x0 = float(math.log(S0))
        else:
            x_grid = SpatialGrid1D.uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="S")
            x0 = float(S0)

        # Build time grid
        n_nodes = int(self.n_time_steps) + 1
        t_grid = TimeGrid.uniform(t0=0.0, t1=float(T), n=n_nodes, name="t")

        # Solve PDE with digital-specific boundary conditions
        sol = self._solve_on_grids_digital(
            payoff=payoff,
            trade=trade,
            K=K,
            T=T,
            r_d=r_d,
            r_f=r_f,
            sigma=sigma,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=float(S_min),
            S_right=float(S_max),
        )

        # Interpolate PV at x0
        pv = float(sol.value_at(x0))
        if not math.isfinite(pv):
            raise ValueError(f"Non-finite FD PV computed: {pv}")

        # Package context for bumped re-solves
        return pv, {
            "x_grid": x_grid,
            "t_grid": t_grid,
            "x0": x0,
            "r_d": r_d,
            "r_f": r_f,
            "sigma": sigma,
            "S_min": float(S_min),
            "S_max": float(S_max),
        }

    def _solve_on_grids_digital(
        self,
        *,
        payoff: BasePayoff1D,
        trade: EuropeanFxDigitalOption,
        K: float,
        T: float,
        r_d: float,
        r_f: float,
        sigma: float,
        x_grid: SpatialGrid1D,
        t_grid: TimeGrid,
        S_left: float,
        S_right: float,
        store_surface: bool = False,
    ):
        """
        Solve GK PDE for digitals with payout-specific boundary conditions.

        Why boundaries matter more for digitals:
        ---------------------------------------
        Vanilla boundaries use asymptotic linear payoff growth (call) or bounded payoff (put).
        Digitals instead tend to constants (cash) or linear-in-S (asset) depending on payout type.
        """

        # Terminal condition uses payoff library
        def terminal_payoff(x: np.ndarray) -> np.ndarray:
            spot = np.exp(x) if x_grid.is_log_space else x
            return payoff.terminal(spot).astype(np.float64, copy=False)

        # Discount factors from time t to expiry T
        def _df_dom_tau(t: float) -> float:
            tau = max(0.0, float(T) - float(t))
            return float(math.exp(-float(r_d) * tau))

        def _df_for_tau(t: float) -> float:
            tau = max(0.0, float(T) - float(t))
            return float(math.exp(-float(r_f) * tau))

        # Determine option type for boundary direction
        option_type: OptionType = trade.option_type  # type: ignore[assignment]

        # Construct boundary conditions based on payoff type:
        #
        # - cash digital:
        #     call: as S->inf, probability ITM -> 1, so V -> cash * df_d(tau)
        #     put : as S->0  , probability ITM -> 1, so V -> cash * df_d(tau)
        #
        # - asset digital:
        #     call: as S->inf, ITM -> 1, so V -> units * S * df_f(tau) (foreign discounting)
        #     put : as S->0  , ITM -> 1, so V -> units * S * df_f(tau)
        #
        # Other side goes to ~0 (OTM prob -> 0).
        if trade.payoff == "cash":
            cash = float(trade.payout_amount)

            if option_type == "call":
                left_bc = DirichletBC(side="left", value=lambda _t: 0.0)
                right_bc = DirichletBC(side="right", value=lambda t: cash * _df_dom_tau(t))
            else:
                left_bc = DirichletBC(side="left", value=lambda t: cash * _df_dom_tau(t))
                right_bc = DirichletBC(side="right", value=lambda _t: 0.0)

        else:
            units = float(trade.payout_amount)

            if option_type == "call":
                left_bc = DirichletBC(side="left", value=lambda _t: 0.0)
                right_bc = DirichletBC(side="right", value=lambda t: units * float(S_right) * _df_for_tau(t))
            else:
                left_bc = DirichletBC(side="left", value=lambda t: units * float(S_left) * _df_for_tau(t))
                right_bc = DirichletBC(side="right", value=lambda _t: 0.0)

        # Bundle BCs
        boundaries = BoundaryPair(left=left_bc, right=right_bc)

        # PDE coefficients (same as vanilla)
        if x_grid.is_log_space:
            a = float((r_d - r_f) - 0.5 * sigma * sigma)
            b = float(0.5 * sigma * sigma)
            c = float(r_d)
        else:
            def a(x: np.ndarray, _t: float) -> np.ndarray:
                return (float(r_d - r_f) * x).astype(np.float64, copy=False)

            def b(x: np.ndarray, _t: float) -> np.ndarray:
                return (0.5 * float(sigma) * float(sigma) * x * x).astype(np.float64, copy=False)

            c = float(r_d)

        # Choose theta scheme
        scheme = ThetaScheme(theta=float(self.theta))

        # Solve PDE
        return solve_pde_theta(
            x_grid=x_grid,
            t_grid=t_grid,
            terminal_payoff=terminal_payoff,
            a=a,
            b=b,
            c=c,
            boundaries=boundaries,
            scheme=scheme,
            store_surface=bool(store_surface),
        )
