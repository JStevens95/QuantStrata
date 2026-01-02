from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

from src.instruments.fx.options.vanilla import AmericanFxVanillaOption
from src.marketdata.market import Market
from src.models.numeric.finite_difference.boundaries import BoundaryPair, DirichletBC
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.schemes import ThetaScheme, solve_pde_theta_american
from src.models.numeric.finite_difference.diagnostics import FdDiagnostics
from src.pricers.fx.european_bsm import _rate_from_df


OptionType = Literal["call", "put"]
GreekName = Literal["delta", "gamma", "vega", "rho_domestic", "rho_foreign"]


@dataclass(frozen=True, slots=True)
class FxAmericanVanillaFdPricer:
    """
    Finite-difference (PDE) pricer for American FX vanilla options under Garman–Kohlhagen.

    PDE (domestic measure)
    ----------------------
    dS/S = (r_d - r_f) dt + sigma dW

    V solves (in S-space):
        V_t + (r_d - r_f) S V_S + 0.5 sigma^2 S^2 V_SS - r_d V = 0

    American constraint (early exercise)
    -----------------------------------
        V(S,t) >= intrinsic(S)
        where intrinsic(S) = max(S-K,0) (call) or max(K-S,0) (put)

    Numerical method
    ----------------
    - Log-space grid by default (x = ln S)
    - Theta-scheme in time (CN/implicit)
    - PSOR projected iterations each step to enforce the inequality constraint

    Output convention
    -----------------
    PV is per 1 unit of foreign notional (same convention as your BSM/European FD),
    then scaled by trade.notional.
    """

    # Grid controls
    n_space: int = 401
    n_time_steps: int = 240
    n_std: float = 6.0

    # Time stepping
    theta: float = 0.5
    use_log_space: bool = True

    # PSOR controls
    psor_omega: float = 1.2
    psor_tol: float = 1e-10
    psor_max_iter: int = 50_000

    # Greek bump sizes
    spot_rel_bump: float = 1e-4
    vol_abs_bump: float = 1e-4
    rate_abs_bump: float = 1e-4

    def price(self, trade: AmericanFxVanillaOption, market: Market) -> float:
        pv_per_unit, _ = self._price_per_unit_and_context(trade, market)
        return float(trade.notional) * float(pv_per_unit)

    def greeks(self, trade: AmericanFxVanillaOption, market: Market) -> Dict[GreekName, float]:
        option_type: OptionType = trade.option_type  # type: ignore[assignment]
        notional = float(trade.notional)

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        if T <= 0.0:
            # At expiry, greeks are not stable; return 0s for portfolio safety.
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho_domestic": 0.0, "rho_foreign": 0.0}

        pv0_per_unit, ctx = self._price_per_unit_and_context(trade, market)

        x_grid = ctx["x_grid"]
        t_grid = ctx["t_grid"]
        x0 = float(ctx["x0"])
        r_d0 = float(ctx["r_d"])
        r_f0 = float(ctx["r_f"])
        sigma0 = float(ctx["sigma"])
        S_min = float(ctx["S_min"])
        S_max = float(ctx["S_max"])

        # Base surface (used for delta/gamma only).
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
            store_surface=False,
        )

        # -------------------------
        # Delta / Gamma (wrt spot)
        # -------------------------
        eps_s = float(self.spot_rel_bump)
        S_up = S0 * (1.0 + eps_s)
        S_dn = S0 * (1.0 - eps_s)
        h = float(S0 * eps_s)

        if x_grid.is_log_space:
            x_up = float(math.log(S_up))
            x_dn = float(math.log(S_dn))
        else:
            x_up = float(S_up)
            x_dn = float(S_dn)

        v_up = float(sol0.value_at(x_up))
        v_dn = float(sol0.value_at(x_dn))
        v_0 = float(sol0.value_at(x0))

        delta_per_unit = (v_up - v_dn) / (2.0 * h)
        gamma_per_unit = (v_up - 2.0 * v_0 + v_dn) / (h * h)

        # -------------------------
        # Vega (bump sigma)
        # -------------------------
        eps_v = float(self.vol_abs_bump)
        sigma_up = max(0.0, sigma0 + eps_v)
        sigma_dn = max(0.0, sigma0 - eps_v)

        pv_sig_up = float(
            self._solve_on_grids(
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
                store_surface=False,
            ).value_at(x0)
        )
        pv_sig_dn = float(
            self._solve_on_grids(
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
                store_surface=False,
            ).value_at(x0)
        )
        vega_per_unit = (pv_sig_up - pv_sig_dn) / (2.0 * eps_v)

        # -------------------------
        # Rho domestic (bump r_d)
        # -------------------------
        eps_r = float(self.rate_abs_bump)

        pv_rd_up = float(
            self._solve_on_grids(
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
                store_surface=False,
            ).value_at(x0)
        )
        pv_rd_dn = float(
            self._solve_on_grids(
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
                store_surface=False,
            ).value_at(x0)
        )
        rho_domestic_per_unit = (pv_rd_up - pv_rd_dn) / (2.0 * eps_r)

        # -------------------------
        # Rho foreign (bump r_f)
        # -------------------------
        pv_rf_up = float(
            self._solve_on_grids(
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
                store_surface=False,
            ).value_at(x0)
        )
        pv_rf_dn = float(
            self._solve_on_grids(
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
                store_surface=False,
            ).value_at(x0)
        )
        rho_foreign_per_unit = (pv_rf_up - pv_rf_dn) / (2.0 * eps_r)

        return {
            "delta": float(notional * delta_per_unit),
            "gamma": float(notional * gamma_per_unit),
            "vega": float(notional * vega_per_unit),
            "rho_domestic": float(notional * rho_domestic_per_unit),
            "rho_foreign": float(notional * rho_foreign_per_unit),
        }

    def diagnostics(
        self,
        trade: AmericanFxVanillaOption,
        market: Market,
        *,
        store_surface: bool = False,
    ) -> FdDiagnostics:
        """
        Solve the American PDE and return diagnostics for plotting / debugging.

        Notes
        -----
        - Returned arrays are per 1 unit notional (unscaled).
        - spot_grid is always in linear space S (even if log-space grid is used).
        """
        pv_per_unit, ctx = self._price_per_unit_and_context(trade, market)

        x_grid = ctx.get("x_grid", None)
        t_grid = ctx.get("t_grid", None)

        if x_grid is None or t_grid is None:
            raise ValueError("diagnostics() requires T>0 and sigma>0 (FD grids must exist).")

        option_type: OptionType = trade.option_type  # type: ignore[assignment]

        sol = self._solve_on_grids(
            option_type=option_type,
            K=float(trade.strike),
            T=float(trade.expiry),
            r_d=float(ctx["r_d"]),
            r_f=float(ctx["r_f"]),
            sigma=float(ctx["sigma"]),
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=float(ctx["S_min"]),
            S_right=float(ctx["S_max"]),
            store_surface=bool(store_surface),
        )

        spot_grid = np.asarray(x_grid.linear_space_values(), dtype=np.float64)
        time_grid = np.asarray(t_grid.t, dtype=np.float64)

        surface = None
        if store_surface and sol.surface is not None:
            surface = np.asarray(sol.surface, dtype=np.float64)

        return FdDiagnostics(
            x_grid=x_grid,
            t_grid=t_grid,
            spot_grid=spot_grid,
            time_grid=time_grid,
            values_t0_per_unit=np.asarray(sol.values_t0, dtype=np.float64),
            surface_per_unit=surface,
            spot0=float(market.quote(trade.spot_id)),
            strike=float(trade.strike),
            expiry=float(trade.expiry),
            r_d=float(ctx["r_d"]),
            r_f=float(ctx["r_f"]),
            sigma=float(ctx["sigma"]),
            x0=float(ctx["x0"]),
            meta={
                "pv_per_unit": float(pv_per_unit),
                "S_min": float(ctx["S_min"]),
                "S_max": float(ctx["S_max"]),
                "n_space": float(self.n_space),
                "n_time_steps": float(self.n_time_steps),
                "theta": float(self.theta),
            },
        )

    # =====================================================================
    # Internals
    # =====================================================================

    def _price_per_unit_and_context(
        self,
        trade: AmericanFxVanillaOption,
        market: Market,
    ) -> Tuple[float, Dict[str, object]]:
        option_type: OptionType = trade.option_type  # type: ignore[assignment]

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0 for FD.")

        if T == 0.0:
            if option_type == "call":
                return float(max(S0 - K, 0.0)), {"x_grid": None, "t_grid": None}
            return float(max(K - S0, 0.0)), {"x_grid": None, "t_grid": None}

        df_d_T = float(market.curve(trade.domestic_curve_id).df(T))
        df_f_T = float(market.curve(trade.foreign_curve_id).df(T))
        r_d = float(_rate_from_df(df=df_d_T, t=T))
        r_f = float(_rate_from_df(df=df_f_T, t=T))

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")
        if sigma == 0.0:
            # With zero vol, American collapses to immediate exercise or discounted forward intrinsic,
            # but the safe choice is: value = intrinsic now (can exercise immediately).
            if option_type == "call":
                return float(max(S0 - K, 0.0)), {"x_grid": None, "t_grid": None}
            return float(max(K - S0, 0.0)), {"x_grid": None, "t_grid": None}

        width = float(self.n_std) * float(sigma) * math.sqrt(float(T))
        S_min = max(1e-12, S0 * math.exp(-width))
        S_max = S0 * math.exp(+width)

        if K > 0.0:
            S_min = min(S_min, 0.5 * K)
            S_max = max(S_max, 2.0 * K)

        if self.n_space < 3:
            raise ValueError("n_space must be >= 3.")

        if self.use_log_space:
            x_grid = SpatialGrid1D.log_uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="logS")
            x0 = float(math.log(S0))
        else:
            x_grid = SpatialGrid1D.uniform(x_min=S_min, x_max=S_max, n=int(self.n_space), name="S")
            x0 = float(S0)

        n_nodes = int(self.n_time_steps) + 1
        if n_nodes < 2:
            raise ValueError("n_time_steps must be >= 1.")
        t_grid = TimeGrid.uniform(t0=0.0, t1=float(T), n=n_nodes, name="t")

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
            store_surface=False,
        )

        pv_per_unit = float(sol.value_at(x0))
        if not math.isfinite(pv_per_unit):
            raise ValueError(f"Non-finite American FD PV per unit computed: {pv_per_unit}")

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
        store_surface: bool,
    ):
        # Payoff functions (time-independent for vanilla)
        def terminal_payoff(x: np.ndarray) -> np.ndarray:
            S = np.exp(x) if x_grid.is_log_space else x
            if option_type == "call":
                return np.maximum(S - float(K), 0.0).astype(np.float64, copy=False)
            return np.maximum(float(K) - S, 0.0).astype(np.float64, copy=False)

        def intrinsic_payoff(x: np.ndarray) -> np.ndarray:
            # For vanilla, intrinsic equals terminal payoff form (but applies at all times).
            return terminal_payoff(x)

        # American boundary conditions: use intrinsic asymptotes (exercise-available).
        if option_type == "call":
            left_bc = DirichletBC(side="left", value=lambda _t: 0.0)
            right_bc = DirichletBC(side="right", value=lambda _t: max(0.0, float(S_right) - float(K)))
        else:
            left_bc = DirichletBC(side="left", value=lambda _t: max(0.0, float(K) - float(S_left)))
            right_bc = DirichletBC(side="right", value=lambda _t: 0.0)

        boundaries = BoundaryPair(left=left_bc, right=right_bc)

        # PDE coefficients in canonical form:
        #   V_t + a V_x + b V_xx - c V = 0
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

        return solve_pde_theta_american(
            x_grid=x_grid,
            t_grid=t_grid,
            terminal_payoff=terminal_payoff,
            intrinsic_payoff=intrinsic_payoff,
            a=a,
            b=b,
            c=c,
            boundaries=boundaries,
            scheme=scheme,
            store_surface=bool(store_surface),
            psor_omega=float(self.psor_omega),
            psor_tol=float(self.psor_tol),
            psor_max_iter=int(self.psor_max_iter),
        )