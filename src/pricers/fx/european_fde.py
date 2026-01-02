from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.marketdata.market import Market

from src.models.numeric.finite_difference.boundaries import BoundaryPair, DirichletBC
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.schemes import ThetaScheme, solve_pde_theta
from src.models.numeric.finite_difference.diagnostics import FdDiagnostics

# Keep conventions aligned across BSM/MC/FD
from src.pricers.fx.european_bsm import _rate_from_df


OptionType = Literal["call", "put"]
GreekName = Literal["delta", "gamma", "vega", "rho_domestic", "rho_foreign"]


@dataclass(frozen=True, slots=True)
class FxEuropeanVanillaFdPricer:
    """
    Finite-difference (PDE) pricer for European FX vanilla options under Garman–Kohlhagen.

    Adds V1 greeks via:
      - Delta/Gamma: interpolate on the *same* solved surface (no re-solve)
      - Vega/Rhos: bump parameter and re-solve the PDE on the *same grids*
    """

    # -----------------------------
    # Grid controls
    # -----------------------------
    n_space: int = 401           # spatial nodes including boundaries
    n_time_steps: int = 200      # number of time steps (nodes = n_time_steps + 1)
    n_std: float = 6.0           # domain width in sigma*sqrt(T) around S0

    # -----------------------------
    # Time stepping
    # -----------------------------
    theta: float = 0.5           # 0.5=Crank–Nicolson, 1.0=implicit

    # -----------------------------
    # Space selection
    # -----------------------------
    use_log_space: bool = True   # recommended

    # -----------------------------
    # Greek bump sizes (V1 defaults)
    # -----------------------------
    spot_rel_bump: float = 1e-4      # 1bp spot bump for delta/gamma
    vol_abs_bump: float = 1e-4       # 1bp vol bump for vega
    rate_abs_bump: float = 1e-4      # 1bp rate bump for rhos

    # =====================================================================
    # Public API
    # =====================================================================

    def price(self, trade: EuropeanFxVanillaOption, market: Market) -> float:
        """
        Return domestic PV (scaled by trade.notional).
        """
        # Solve once and read PV at spot (or log-spot) via interpolation.
        pv_per_unit, _ = self._price_per_unit_and_context(trade, market)
        return float(trade.notional) * float(pv_per_unit)

    def greeks(self, trade: EuropeanFxVanillaOption, market: Market) -> Dict[GreekName, float]:
        """
        Return FD greeks (scaled by trade.notional) using stable V1 bump-and-reprice.

        Delta/Gamma:
          - computed from the solved V(t=0, x) curve using interpolation at bumped spot

        Vega/Rhos:
          - require re-solving PDE with bumped sigma / r_d / r_f
        """
        # -----------------------------
        # Read trade + market inputs
        # -----------------------------
        option_type: OptionType = trade.option_type  # type: ignore[assignment]
        notional = float(trade.notional)

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")

        # Deterministic expiry: greeks are not meaningful at T=0 (discontinuous payoffs).
        # We return 0s so PortfolioPricer remains stable.
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho_domestic": 0.0,
                "rho_foreign": 0.0,
            }

        # -----------------------------
        # Base solve + cached context
        # -----------------------------
        pv0_per_unit, ctx = self._price_per_unit_and_context(trade, market)

        # Unpack context needed to do parameter-bumped re-solves on the same grids.
        x_grid, t_grid = ctx["x_grid"], ctx["t_grid"]
        x0 = float(ctx["x0"])
        r_d0 = float(ctx["r_d"])
        r_f0 = float(ctx["r_f"])
        sigma0 = float(ctx["sigma"])
        S_min = float(ctx["S_min"])
        S_max = float(ctx["S_max"])

        # -----------------------------
        # Delta/Gamma from the same surface (no re-solve)
        # -----------------------------
        # We recompute a solution once and then evaluate at bumped x0.
        # This avoids re-solving and yields much smoother delta/gamma.
        sol0 = self._solve_on_grids(
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

        # Spot bumps (relative). Keep bumps small so interpolation is accurate.
        eps_s = float(self.spot_rel_bump)
        S_up = S0 * (1.0 + eps_s)
        S_dn = S0 * (1.0 - eps_s)

        # Convert bumped spot to the grid coordinate used by the PDE engine.
        if x_grid.is_log_space:
            x_up = float(math.log(S_up))
            x_dn = float(math.log(S_dn))
            h = float(S0 * eps_s)  # derivative w.r.t. S, so use S-space step size
        else:
            x_up = float(S_up)
            x_dn = float(S_dn)
            h = float(S0 * eps_s)

        v_up = float(sol0.value_at(x_up))
        v_dn = float(sol0.value_at(x_dn))
        v_0 = float(sol0.value_at(x0))

        # -----------------------------
        # Delta/Gamma from the same solved surface (NO re-solve)
        # Use grid derivatives (stable) + log-space chain rule if needed.
        # -----------------------------
        delta_per_unit, gamma_per_unit = self._delta_gamma_from_solution(
            sol=sol0, x_grid=x_grid, xq=x0,
        )

        # -----------------------------
        # Vega: bump sigma and re-solve
        # -----------------------------
        eps_v = float(self.vol_abs_bump)
        sigma_up = max(0.0, sigma0 + eps_v)
        sigma_dn = max(0.0, sigma0 - eps_v)

        pv_sig_up = self._solve_on_grids(
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

        pv_sig_dn = self._solve_on_grids(
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

        vega_per_unit = (float(pv_sig_up) - float(pv_sig_dn)) / (2.0 * eps_v)

        # -----------------------------
        # Rhos: bump r_d and r_f and re-solve
        # -----------------------------
        eps_r = float(self.rate_abs_bump)

        pv_rd_up = self._solve_on_grids(
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

        pv_rf_up = self._solve_on_grids(
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
        # Scale all greeks by notional
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
        Run a diagnostic FD solve and return grids + t=0 curve + (optional) surface.

        This is what examples / reporting should use (so you don’t duplicate PDE setup).
        """
        option_type: OptionType = trade.option_type  # type: ignore[assignment]

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        df_d_T = float(market.curve(trade.domestic_curve_id).df(T))
        df_f_T = float(market.curve(trade.foreign_curve_id).df(T))
        r_d = float(_rate_from_df(df=df_d_T, t=T))
        r_f = float(_rate_from_df(df=df_f_T, t=T))

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma <= 0.0:
            raise ValueError("diagnostics() expects sigma > 0 so the surface is meaningful.")

        # Domain choice matches pricer
        width = float(self.n_std) * float(sigma) * math.sqrt(float(T))
        S_min = max(1e-12, S0 * math.exp(-width))
        S_max = S0 * math.exp(+width)
        if K > 0.0:
            S_min = min(S_min, 0.5 * K)
            S_max = max(S_max, 2.0 * K)

        # Grids
        if self.use_log_space:
            x_grid = SpatialGrid1D.log_uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="logS")
            x0 = float(math.log(S0))
        else:
            x_grid = SpatialGrid1D.uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="S")
            x0 = float(S0)

        t_grid = TimeGrid.uniform(t0=0.0, t1=float(T), n=int(self.n_time_steps) + 1, name="t")

        sol = self._solve_on_grids(
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
            meta={"n_space": float(self.n_space), "n_time_steps": float(self.n_time_steps), "theta": float(self.theta)},
        )

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _price_per_unit_and_context(
        self,
        trade: EuropeanFxVanillaOption,
        market: Market,
    ) -> Tuple[float, Dict[str, object]]:
        """
        Compute PV per unit notional and return a context dict containing:
          - grids, parameters, interpolation coordinate, domain bounds
        so greeks can reuse grids and avoid re-deriving domain choices.
        """
        option_type: OptionType = trade.option_type  # type: ignore[assignment]

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0 for log-space PDE.")

        # At expiry: PV per unit is the payoff evaluated at spot.
        if T == 0.0:
            if option_type == "call":
                return float(max(S0 - K, 0.0)), {"x_grid": None, "t_grid": None}

        # Domestic/foreign curves provide discount factors at T.
        df_d_T = float(market.curve(trade.domestic_curve_id).df(T))
        df_f_T = float(market.curve(trade.foreign_curve_id).df(T))

        # Convert discount factors to continuous rates.
        r_d = float(_rate_from_df(df=df_d_T, t=T))
        r_f = float(_rate_from_df(df=df_f_T, t=T))

        # Constant vol at (T, K) matches BSM PDE.
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Zero-vol deterministic case: forward payoff discounted.
        if sigma == 0.0:
            F0 = S0 * math.exp((r_d - r_f) * T)
            disc = math.exp(-r_d * T)
            if option_type == "call":
                return float(disc * max(F0 - K, 0.0)), {"x_grid": None, "t_grid": None}
            return float(disc * max(K - F0, 0.0)), {"x_grid": None, "t_grid": None}

        # Choose S-domain using +/- n_std * sigma * sqrt(T) around S0 (lognormal width).
        width = float(self.n_std) * float(sigma) * math.sqrt(float(T))
        S_min = max(1e-12, S0 * math.exp(-width))
        S_max = S0 * math.exp(+width)

        # Ensure K is inside domain for interpolation accuracy near payoff kink.
        if K > 0.0:
            S_min = min(S_min, 0.5 * K)
            S_max = max(S_max, 2.0 * K)

        # Build spatial grid.
        if self.n_space < 3:
            raise ValueError("n_space must be >= 3.")

        if self.use_log_space:
            x_grid = SpatialGrid1D.log_uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="logS")
            x0 = float(math.log(S0))
        else:
            x_grid = SpatialGrid1D.uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="S")
            x0 = float(S0)

        # Build time grid.
        n_nodes = int(self.n_time_steps) + 1
        if n_nodes < 2:
            raise ValueError("n_time_steps must be >= 1.")
        t_grid = TimeGrid.uniform(t0=0.0, t1=float(T), n=n_nodes, name="t")

        # Solve PDE and interpolate PV per unit at x0.
        sol = self._solve_on_grids(
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

        pv_per_unit = float(sol.value_at(x0))
        if not math.isfinite(pv_per_unit):
            raise ValueError(f"Non-finite FD PV per unit computed: {pv_per_unit}")

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
            store_surface: bool = False,  # <-- ADD (default keeps old callers working)
    ):
        """
        Solve the constant-parameter GK PDE on pre-built grids.

        Keeping the grids fixed across parameter bumps makes vega/rhos
        more stable and comparable.
        """

        def terminal_payoff(x: np.ndarray) -> np.ndarray:
            S = np.exp(x) if x_grid.is_log_space else x
            if option_type == "call":
                return np.maximum(S - float(K), 0.0).astype(np.float64, copy=False)
            return np.maximum(float(K) - S, 0.0).astype(np.float64, copy=False)

        def _df_dom_tau(t: float) -> float:
            tau = max(0.0, float(T) - float(t))
            return float(math.exp(-float(r_d) * tau))

        def _df_for_tau(t: float) -> float:
            tau = max(0.0, float(T) - float(t))
            return float(math.exp(-float(r_f) * tau))

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

        boundaries = BoundaryPair(left=left_bc, right=right_bc)

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

        scheme = ThetaScheme(theta=float(self.theta))

        return solve_pde_theta(
            x_grid=x_grid,
            t_grid=t_grid,
            terminal_payoff=terminal_payoff,
            a=a,
            b=b,
            c=c,
            boundaries=boundaries,
            scheme=scheme,
            store_surface=bool(store_surface),  # <-- PASS THROUGH
        )

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
        First/second derivative of the quadratic interpolant through (x0,y0),(x1,y1),(x2,y2)
        evaluated at xq.

        Returns
        -------
        (dy_dx, d2y_dx2)
        """
        den0 = (x0 - x1) * (x0 - x2)
        den1 = (x1 - x0) * (x1 - x2)
        den2 = (x2 - x0) * (x2 - x1)

        if den0 == 0.0 or den1 == 0.0 or den2 == 0.0:
            raise ValueError("Degenerate quadratic stencil (duplicate x nodes).")

        # Lagrange basis derivatives:
        # d/dx [(x-xj)(x-xk)] = (x-xk) + (x-xj) = 2x - xj - xk
        dL0 = (2.0 * xq - x1 - x2) / den0
        dL1 = (2.0 * xq - x0 - x2) / den1
        dL2 = (2.0 * xq - x0 - x1) / den2

        # d2/dx2 [(x-xj)(x-xk)] = 2
        d2L0 = 2.0 / den0
        d2L1 = 2.0 / den1
        d2L2 = 2.0 / den2

        dy = y0 * dL0 + y1 * dL1 + y2 * dL2
        d2y = y0 * d2L0 + y1 * d2L1 + y2 * d2L2
        return float(dy), float(d2y)

    def _delta_gamma_from_solution(
        self,
        *,
        sol,               # FdSolution
        x_grid: SpatialGrid1D,
        xq: float,
    ) -> Tuple[float, float]:
        """
        Compute (delta_per_unit, gamma_per_unit) at xq using a local quadratic stencil
        on the solved t=0 slice, then apply log-space chain rule if needed.
        """
        x = x_grid.x
        v = sol.values_t0

        # choose stencil centered around xq: (j-1, j, j+1)
        j = int(np.searchsorted(x, float(xq), side="right") - 1)
        # clamp so j-1 >= 0 and j+1 <= n-1
        j = max(1, min(j, int(x.size) - 2))

        x0, x1, x2 = float(x[j - 1]), float(x[j]), float(x[j + 1])
        y0, y1, y2 = float(v[j - 1]), float(v[j]), float(v[j + 1])

        dv_dx, d2v_dx2 = self._quad_derivatives_at(x0, x1, x2, y0, y1, y2, float(xq))

        if x_grid.is_log_space:
            S = float(math.exp(float(xq)))
            delta = dv_dx / S
            gamma = (d2v_dx2 - dv_dx) / (S * S)
            return float(delta), float(gamma)

        # x == S space
        return float(dv_dx), float(d2v_dx2)