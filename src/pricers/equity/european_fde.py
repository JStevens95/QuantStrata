"""
Equity European Vanilla Finite Difference Pricer

PDE-based pricer using Crank-Nicolson scheme.

Author: QuantStrata Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

from src.instruments.equity.options.vanilla import EquityVanillaEuropeanOption
from src.marketdata.core.market import Market
from src.models.numeric.finite_difference.boundaries import BoundaryPair, DirichletBC
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.schemes import ThetaScheme, solve_pde_theta
from src.models.numeric.finite_difference.diagnostics import FdDiagnostics
from src.models.payoffs.types import OptionType
from src.models.payoffs.base import BasePayoff1D
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff

GreekName = Literal["delta", "gamma", "vega", "rho", "theta"]


def _rate_from_df(*, df: float, t: float) -> float:
    """Convert discount factor to continuously-compounded rate."""
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


@dataclass(frozen=True, slots=True)
class EquityVanillaEuropeanOptionFdPricer:
    """
    Finite Difference pricer for European equity vanilla options.

    PDE
    ---
    Under risk-neutral measure with dividend yield q:

        ∂V/∂t + (r-q)S ∂V/∂S + ½σ²S² ∂²V/∂S² = rV

    In log-space (x = ln S), coefficients become constants:

        ∂V/∂t + a ∂V/∂x + b ∂²V/∂x² = cV

    Where:
    - a = (r - q) - ½σ²
    - b = ½σ²
    - c = r

    Numerical Method
    ----------------
    - Log-space transformation for stability
    - Theta-scheme (default: Crank-Nicolson, θ = 0.5)
    - Thomas algorithm for tridiagonal systems

    Greeks
    ------
    - Delta/Gamma: Extracted from solved surface via quadratic interpolation
    - Vega/Rho: Bump-and-reprice with re-solve

    Parameters
    ----------
    n_space : int
        Number of spatial grid points
    n_time_steps : int
        Number of time steps
    n_std : float
        Domain width in standard deviations
    theta : float
        Time-stepping parameter (0.5 = CN, 1.0 = implicit)
    use_log_space : bool
        Use log-space transformation (recommended)
    """

    # Grid controls
    n_space: int = 401
    n_time_steps: int = 200
    n_std: float = 6.0

    # Time stepping
    theta: float = 0.5  # Crank-Nicolson
    use_log_space: bool = True

    # Greek bump sizes
    spot_rel_bump: float = 1e-4
    vol_abs_bump: float = 1e-4
    rate_abs_bump: float = 1e-4

    def price(self, trade: EquityVanillaEuropeanOption, market: Market) -> float:
        """
        Price European equity option via finite difference.

        Parameters
        ----------
        trade : EquityVanillaEuropeanOption
            Option to price
        market : Market
            Market snapshot

        Returns
        -------
        float
            Present value
        """
        pv_per_unit, _ = self._price_per_unit_and_context(trade, market)
        return float(trade.notional) * float(pv_per_unit)

    def greeks(self, trade: EquityVanillaEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """
        Compute Greeks via finite difference methods.

        Parameters
        ----------
        trade : EquityVanillaEuropeanOption
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

        # Validate
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")

        # At expiry, greeks are unstable
        if T == 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0, "theta": 0.0}

        # Get base context
        _, ctx = self._price_per_unit_and_context(trade, market)
        if ctx.get("x_grid") is None:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0, "theta": 0.0}

        x_grid: SpatialGrid1D = ctx["x_grid"]
        t_grid: TimeGrid = ctx["t_grid"]
        x0 = float(ctx["x0"])
        r = float(ctx["r"])
        sigma = float(ctx["sigma"])
        S_min = float(ctx["S_min"])
        S_max = float(ctx["S_max"])

        # Build payoff
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Solve base case
        sol0 = self._solve_on_grids(
            payoff=payoff,
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
        )

        # Delta/Gamma from surface
        delta_per_unit, gamma_per_unit = self._delta_gamma_from_solution(sol=sol0, x_grid=x_grid, xq=x0)

        # Vega (bump sigma)
        eps_v = float(self.vol_abs_bump)
        sigma_up = max(0.0, sigma + eps_v)
        sigma_dn = max(0.0, sigma - eps_v)

        pv_sig_up = self._solve_on_grids(
            payoff=payoff, option_type=option_type, K=K, T=T, r=r, q=q, sigma=sigma_up,
            x_grid=x_grid, t_grid=t_grid, S_left=S_min, S_right=S_max,
        ).value_at(x0)

        pv_sig_dn = self._solve_on_grids(
            payoff=payoff, option_type=option_type, K=K, T=T, r=r, q=q, sigma=sigma_dn,
            x_grid=x_grid, t_grid=t_grid, S_left=S_min, S_right=S_max,
        ).value_at(x0)

        vega_per_unit = (float(pv_sig_up) - float(pv_sig_dn)) / (2.0 * eps_v)

        # Rho (bump r)
        eps_r = float(self.rate_abs_bump)

        pv_r_up = self._solve_on_grids(
            payoff=payoff, option_type=option_type, K=K, T=T, r=r + eps_r, q=q, sigma=sigma,
            x_grid=x_grid, t_grid=t_grid, S_left=S_min, S_right=S_max,
        ).value_at(x0)

        pv_r_dn = self._solve_on_grids(
            payoff=payoff, option_type=option_type, K=K, T=T, r=r - eps_r, q=q, sigma=sigma,
            x_grid=x_grid, t_grid=t_grid, S_left=S_min, S_right=S_max,
        ).value_at(x0)

        rho_per_unit = (float(pv_r_up) - float(pv_r_dn)) / (2.0 * eps_r)

        # Theta (approximate via time bump)
        # Note: FD theta is more complex; using simplified approximation
        theta_per_unit = 0.0  # TODO: Implement proper theta

        return {
            "delta": float(notional * delta_per_unit),
            "gamma": float(notional * gamma_per_unit),
            "vega": float(notional * vega_per_unit),
            "rho": float(notional * rho_per_unit),
            "theta": float(notional * theta_per_unit),
        }

    def diagnostics(
            self,
            trade: EquityVanillaEuropeanOption,
            market: Market,
            *,
            store_surface: bool = True,
    ) -> FdDiagnostics:
        """
        Get diagnostic information for plotting.

        Parameters
        ----------
        trade : EquityVanillaEuropeanOption
            Option to analyze
        market : Market
            Market snapshot
        store_surface : bool
            Whether to store full time-space surface

        Returns
        -------
        FdDiagnostics
            Diagnostic container with grids and values
        """
        option_type: OptionType = trade.option_type

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)

        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma <= 0.0:
            raise ValueError("diagnostics() requires sigma > 0.")

        # Build grids
        width = float(self.n_std) * sigma * math.sqrt(T)
        S_min = max(1e-12, S0 * math.exp(-width))
        S_max = S0 * math.exp(+width)

        if K > 0.0:
            S_min = min(S_min, 0.5 * K)
            S_max = max(S_max, 2.0 * K)

        if self.use_log_space:
            x_grid = SpatialGrid1D.log_uniform(x_min=S_min, x_max=S_max, n=self.n_space, name="logS")
            x0 = float(math.log(S0))
        else:
            x_grid = SpatialGrid1D.uniform(x_min=S_min, x_max=S_max, n=self.n_space, name="S")
            x0 = float(S0)

        t_grid = TimeGrid.uniform(t0=0.0, t1=T, n=self.n_time_steps + 1, name="t")

        payoff = require_terminal_payoff(build_payoff_1d(trade))

        sol = self._solve_on_grids(
            payoff=payoff,
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
            store_surface=store_surface,
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
            spot0=S0,
            strike=K,
            expiry=T,
            r_d=r,
            r_f=q,  # Using r_f slot for dividend yield
            sigma=sigma,
            x0=x0,
            meta={
                "n_space": self.n_space,
                "n_time_steps": self.n_time_steps,
                "theta": self.theta,
                "use_log_space": self.use_log_space,
            },
        )

    def _price_per_unit_and_context(
            self,
            trade: EquityVanillaEuropeanOption,
            market: Market,
    ) -> Tuple[float, Dict[str, object]]:
        """Solve PDE and return price + context for Greeks."""

        option_type: OptionType = trade.option_type

        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")

        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # At expiry
        if T == 0.0:
            pv0 = float(payoff.terminal(np.asarray([S0], dtype=np.float64))[0])
            return pv0, {"x_grid": None, "t_grid": None}

        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Zero vol shortcut
        if sigma == 0.0:
            F = S0 * math.exp((r - q) * T)
            disc = math.exp(-r * T)
            pv0 = disc * float(payoff.terminal(np.asarray([F], dtype=np.float64))[0])
            return float(pv0), {"x_grid": None, "t_grid": None}

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
            payoff=payoff,
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
        )

        pv_per_unit = float(sol.value_at(x0))

        if not math.isfinite(pv_per_unit):
            raise ValueError(f"Non-finite FD PV computed: {pv_per_unit}")

        ctx: Dict[str, object] = {
            "x_grid": x_grid,
            "t_grid": t_grid,
            "x0": x0,
            "r": r,
            "q": q,
            "sigma": sigma,
            "S_min": S_min,
            "S_max": S_max,
        }
        return pv_per_unit, ctx

    def _solve_on_grids(
            self,
            *,
            payoff: BasePayoff1D,
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
            store_surface: bool = False,
    ):
        """Solve the BS PDE on given grids."""

        # Terminal condition
        def terminal_payoff(x: np.ndarray) -> np.ndarray:
            spot = np.exp(x) if x_grid.is_log_space else x
            return payoff.terminal(spot).astype(np.float64, copy=False)

        # Discount factors for boundary conditions
        def _df_tau(t: float) -> float:
            tau = max(0.0, T - t)
            return float(math.exp(-r * tau))

        def _df_div_tau(t: float) -> float:
            tau = max(0.0, T - t)
            return float(math.exp(-q * tau))

        # Boundary conditions
        if option_type == "call":
            left_bc = DirichletBC(side="left", value=lambda _t: 0.0)
            right_bc = DirichletBC(
                side="right",
                value=lambda t: (float(S_right) * _df_div_tau(t)) - (float(K) * _df_tau(t)),
            )
        else:
            left_bc = DirichletBC(
                side="left",
                value=lambda t: max(0.0, (float(K) * _df_tau(t)) - (float(S_left) * _df_div_tau(t))),
            )
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

        return solve_pde_theta(
            x_grid=x_grid,
            t_grid=t_grid,
            terminal_payoff=terminal_payoff,
            a=a,
            b=b,
            c=c,
            boundaries=boundaries,
            scheme=scheme,
            store_surface=store_surface,
        )

    @staticmethod
    def _quad_derivatives_at(
            x0: float, x1: float, x2: float,
            y0: float, y1: float, y2: float,
            xq: float,
    ) -> Tuple[float, float]:
        """Differentiate quadratic interpolant."""
        den0 = (x0 - x1) * (x0 - x2)
        den1 = (x1 - x0) * (x1 - x2)
        den2 = (x2 - x0) * (x2 - x1)

        if den0 == 0.0 or den1 == 0.0 or den2 == 0.0:
            raise ValueError("Degenerate quadratic stencil.")

        dL0 = (2.0 * xq - x1 - x2) / den0
        dL1 = (2.0 * xq - x0 - x2) / den1
        dL2 = (2.0 * xq - x0 - x1) / den2

        d2L0 = 2.0 / den0
        d2L1 = 2.0 / den1
        d2L2 = 2.0 / den2

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
        """Extract delta and gamma from solved surface."""

        x = x_grid.x
        v = sol.values_t0

        j = int(np.searchsorted(x, float(xq), side="right") - 1)
        j = max(1, min(j, int(x.size) - 2))

        x0, x1, x2 = float(x[j - 1]), float(x[j]), float(x[j + 1])
        y0, y1, y2 = float(v[j - 1]), float(v[j]), float(v[j + 1])

        dv_dx, d2v_dx2 = self._quad_derivatives_at(x0, x1, x2, y0, y1, y2, float(xq))

        if x_grid.is_log_space:
            S = float(math.exp(xq))
            delta = dv_dx / S
            gamma = (d2v_dx2 - dv_dx) / (S * S)
            return float(delta), float(gamma)

        return float(dv_dx), float(d2v_dx2)
