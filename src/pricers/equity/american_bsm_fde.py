"""
Equity American Vanilla Finite Difference Pricer

PDE-based pricer with early exercise using PSOR.

Author: QuantStrata Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

from src.instruments.equity.options.vanilla import EquityVanillaAmericanOption
from src.marketdata.core.market import Market
from src.models.numeric.finite_difference.boundaries import BoundaryPair, DirichletBC
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.schemes import ThetaScheme, solve_pde_theta_american
from src.models.numeric.finite_difference.diagnostics import FdDiagnostics

OptionType = Literal["call", "put"]
GreekName = Literal["delta", "gamma", "vega", "rho", "theta"]


def _rate_from_df(*, df: float, t: float) -> float:
    """Convert discount factor to continuously-compounded rate."""
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


@dataclass(frozen=True, slots=True)
class EquityVanillaAmericanOptionFdPricer:
    """
    Finite Difference pricer for American equity vanilla options.

    PDE with Early Exercise
    -----------------------
    Solves the Black-Scholes PDE subject to:

        V(S, t) ≥ intrinsic(S) = max(±(S - K), 0)

    This is a free boundary problem (Linear Complementarity Problem).

    Numerical Method
    ----------------
    - Log-space transformation
    - Theta-scheme time stepping (Crank-Nicolson by default)
    - PSOR (Projected Successive Over-Relaxation) for inequality constraint

    Early Exercise Premium
    ----------------------
    American options have value ≥ European due to early exercise:

    - PUT: Deep ITM puts should be exercised to receive K immediately
           Exercise when S* < K × r / (r + q) approximately

    - CALL on dividend-paying stock: May exercise just before dividend
           to capture the dividend

    - CALL on non-dividend stock: Never exercise early (American = European)

    Parameters
    ----------
    n_space : int
        Number of spatial grid points
    n_time_steps : int
        Number of time steps
    n_std : float
        Domain width in standard deviations
    theta : float
        Time-stepping parameter (0.5 = CN)
    use_log_space : bool
        Use log-space transformation
    psor_omega : float
        Over-relaxation parameter (1 < ω < 2 typical)
    psor_tol : float
        Convergence tolerance for PSOR
    psor_max_iter : int
        Maximum PSOR iterations per time step
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

    def price(self, trade: EquityVanillaAmericanOption, market: Market) -> float:
        """
        Price American equity option via finite difference with PSOR.

        Parameters
        ----------
        trade : EquityVanillaAmericanOption
            Option to price
        market : Market
            Market snapshot

        Returns
        -------
        float
            Present value including early exercise premium
        """
        pv_per_unit, _ = self._price_per_unit_and_context(trade, market)
        return float(trade.notional) * float(pv_per_unit)

    def greeks(self, trade: EquityVanillaAmericanOption, market: Market) -> Dict[GreekName, float]:
        """
        Compute Greeks via bump-and-reprice.

        Parameters
        ----------
        trade : EquityVanillaAmericanOption
            Option to analyze
        market : Market
            Market snapshot

        Returns
        -------
        Dict[GreekName, float]
            Greeks dictionary
        """
        option_type: OptionType = trade.option_type
        notional = float(trade.notional)

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)

        # At expiry, greeks are unstable
        if T <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0, "theta": 0.0}

        _, ctx = self._price_per_unit_and_context(trade, market)
        if ctx.get("x_grid") is None:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0, "theta": 0.0}

        x_grid = ctx["x_grid"]
        t_grid = ctx["t_grid"]
        x0 = float(ctx["x0"])
        r = float(ctx["r"])
        sigma = float(ctx["sigma"])
        S_min = float(ctx["S_min"])
        S_max = float(ctx["S_max"])

        # Base solve
        sol0 = self._solve_on_grids(
            option_type=option_type,
            K=K,
            T=T,
            r=r,
            q=q,
            sigma=sigma,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
            store_surface=False,
        )

        # Delta/Gamma via spot bump
        eps_s = float(self.spot_rel_bump)
        S_up = S0 * (1.0 + eps_s)
        S_dn = S0 * (1.0 - eps_s)
        h = S0 * eps_s

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

        # Vega
        eps_v = float(self.vol_abs_bump)
        sigma_up = max(0.0, sigma + eps_v)
        sigma_dn = max(0.0, sigma - eps_v)

        pv_sig_up = self._solve_on_grids(
            option_type=option_type, K=K, T=T, r=r, q=q, sigma=sigma_up,
            x_grid=x_grid, t_grid=t_grid, S_left=S_min, S_right=S_max, store_surface=False,
        ).value_at(x0)

        pv_sig_dn = self._solve_on_grids(
            option_type=option_type, K=K, T=T, r=r, q=q, sigma=sigma_dn,
            x_grid=x_grid, t_grid=t_grid, S_left=S_min, S_right=S_max, store_surface=False,
        ).value_at(x0)

        vega_per_unit = (float(pv_sig_up) - float(pv_sig_dn)) / (2.0 * eps_v)

        # Rho
        eps_r = float(self.rate_abs_bump)

        pv_r_up = self._solve_on_grids(
            option_type=option_type, K=K, T=T, r=r + eps_r, q=q, sigma=sigma,
            x_grid=x_grid, t_grid=t_grid, S_left=S_min, S_right=S_max, store_surface=False,
        ).value_at(x0)

        pv_r_dn = self._solve_on_grids(
            option_type=option_type, K=K, T=T, r=r - eps_r, q=q, sigma=sigma,
            x_grid=x_grid, t_grid=t_grid, S_left=S_min, S_right=S_max, store_surface=False,
        ).value_at(x0)

        rho_per_unit = (float(pv_r_up) - float(pv_r_dn)) / (2.0 * eps_r)

        return {
            "delta": float(notional * delta_per_unit),
            "gamma": float(notional * gamma_per_unit),
            "vega": float(notional * vega_per_unit),
            "rho": float(notional * rho_per_unit),
            "theta": 0.0,  # TODO: Implement proper theta
        }

    def diagnostics(
            self,
            trade: EquityVanillaAmericanOption,
            market: Market,
            *,
            store_surface: bool = False,
    ) -> FdDiagnostics:
        """
        Get diagnostic information for plotting early exercise boundary.

        Parameters
        ----------
        trade : EquityVanillaAmericanOption
            Option to analyze
        market : Market
            Market snapshot
        store_surface : bool
            Whether to store full time-space surface

        Returns
        -------
        FdDiagnostics
            Diagnostic container
        """
        _, ctx = self._price_per_unit_and_context(trade, market)

        if ctx.get("x_grid") is None:
            raise ValueError("diagnostics() requires T > 0 and sigma > 0.")

        option_type: OptionType = trade.option_type
        x_grid = ctx["x_grid"]
        t_grid = ctx["t_grid"]

        sol = self._solve_on_grids(
            option_type=option_type,
            K=float(trade.strike),
            T=float(trade.expiry),
            r=float(ctx["r"]),
            q=float(trade.dividend_yield),
            sigma=float(ctx["sigma"]),
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=float(ctx["S_min"]),
            S_right=float(ctx["S_max"]),
            store_surface=store_surface,
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
            r_d=float(ctx["r"]),
            r_f=float(trade.dividend_yield),
            sigma=float(ctx["sigma"]),
            x0=float(ctx["x0"]),
            meta={
                "S_min": float(ctx["S_min"]),
                "S_max": float(ctx["S_max"]),
                "n_space": self.n_space,
                "n_time_steps": self.n_time_steps,
                "theta": self.theta,
                "psor_omega": self.psor_omega,
            },
        )

    def _price_per_unit_and_context(
            self,
            trade: EquityVanillaAmericanOption,
            market: Market,
    ) -> Tuple[float, Dict[str, object]]:
        """Solve American PDE and return price + context."""

        option_type: OptionType = trade.option_type

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")

        # At expiry, return intrinsic value
        if T == 0.0:
            if option_type == "call":
                return float(max(S0 - K, 0.0)), {"x_grid": None, "t_grid": None}
            return float(max(K - S0, 0.0)), {"x_grid": None, "t_grid": None}

        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Zero vol: immediate exercise may be optimal
        if sigma == 0.0:
            if option_type == "call":
                return float(max(S0 - K, 0.0)), {"x_grid": None, "t_grid": None}
            return float(max(K - S0, 0.0)), {"x_grid": None, "t_grid": None}

        # Build grids
        width = float(self.n_std) * sigma * math.sqrt(T)
        S_min = max(1e-12, S0 * math.exp(-width))
        S_max = S0 * math.exp(+width)

        if K > 0.0:
            S_min = min(S_min, 0.5 * K)
            S_max = max(S_max, 2.0 * K)

        if self.n_space < 3:
            raise ValueError("n_space must be >= 3.")

        if self.use_log_space:
            x_grid = SpatialGrid1D.log_uniform(x_min=S_min, x_max=S_max, n=self.n_space, name="logS")
            x0 = float(math.log(S0))
        else:
            x_grid = SpatialGrid1D.uniform(x_min=S_min, x_max=S_max, n=self.n_space, name="S")
            x0 = float(S0)

        n_nodes = self.n_time_steps + 1
        if n_nodes < 2:
            raise ValueError("n_time_steps must be >= 1.")
        t_grid = TimeGrid.uniform(t0=0.0, t1=T, n=n_nodes, name="t")

        sol = self._solve_on_grids(
            option_type=option_type,
            K=K,
            T=T,
            r=r,
            q=q,
            sigma=sigma,
            x_grid=x_grid,
            t_grid=t_grid,
            S_left=S_min,
            S_right=S_max,
            store_surface=False,
        )

        pv_per_unit = float(sol.value_at(x0))
        if not math.isfinite(pv_per_unit):
            raise ValueError(f"Non-finite American FD PV: {pv_per_unit}")

        ctx: Dict[str, object] = {
            "x_grid": x_grid,
            "t_grid": t_grid,
            "x0": x0,
            "r": r,
            "sigma": sigma,
            "S_min": S_min,
            "S_max": S_max,
        }
        return pv_per_unit, ctx

    def _solve_on_grids(
            self,
            *,
            option_type: OptionType,
            K: float,
            T: float,
            r: float,
            q: float,
            sigma: float,
            x_grid: SpatialGrid1D,
            t_grid: TimeGrid,
            S_left: float,
            S_right: float,
            store_surface: bool,
    ):
        """Solve American PDE with PSOR."""

        # Terminal payoff
        def terminal_payoff(x: np.ndarray) -> np.ndarray:
            S = np.exp(x) if x_grid.is_log_space else x
            if option_type == "call":
                return np.maximum(S - K, 0.0).astype(np.float64, copy=False)
            return np.maximum(K - S, 0.0).astype(np.float64, copy=False)

        # Intrinsic value (same as terminal for vanilla)
        def intrinsic_payoff(x: np.ndarray) -> np.ndarray:
            return terminal_payoff(x)

        # Boundary conditions (use intrinsic at boundaries)
        if option_type == "call":
            left_bc = DirichletBC(side="left", value=lambda _t: 0.0)
            right_bc = DirichletBC(side="right", value=lambda _t: max(0.0, S_right - K))
        else:
            left_bc = DirichletBC(side="left", value=lambda _t: max(0.0, K - S_left))
            right_bc = DirichletBC(side="right", value=lambda _t: 0.0)

        boundaries = BoundaryPair(left=left_bc, right=right_bc)

        # PDE coefficients
        if x_grid.is_log_space:
            a = float((r - q) - 0.5 * sigma * sigma)
            b = float(0.5 * sigma * sigma)
            c = float(r)
        else:
            def a(x: np.ndarray, _t: float) -> np.ndarray:
                return ((r - q) * x).astype(np.float64, copy=False)

            def b(x: np.ndarray, _t: float) -> np.ndarray:
                return (0.5 * sigma * sigma * x * x).astype(np.float64, copy=False)

            c = float(r)

        scheme = ThetaScheme(theta=self.theta)

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
            store_surface=store_surface,
            psor_omega=self.psor_omega,
            psor_tol=self.psor_tol,
            psor_max_iter=self.psor_max_iter,
        )
