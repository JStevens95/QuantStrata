# src/pricers/ir/european_fde.py
"""
Interest Rate Finite Difference Pricers.

Finite Difference Engine (FDE) pricers for IR derivatives using short rate models.

Mathematical Framework
----------------------
For short rate models like Hull-White, the PDE for a zero-coupon bond P(r,t) is:

    ∂P/∂t + (θ(t) - a·r)·∂P/∂r + ½σ²·∂²P/∂r² - r·P = 0

with terminal condition P(r,T) = 1.

For a derivative V(r,t) with payoff H(P(r,T)) at time T:

    ∂V/∂t + (θ(t) - a·r)·∂V/∂r + ½σ²·∂²V/∂r² - r·V = 0

with terminal condition V(r,T) = H(P(r,T)).

Grid Construction
-----------------
- r-axis: Centered around r0 with range covering ±4-5 standard deviations
- t-axis: From 0 to maturity with uniform or non-uniform spacing
- Boundary conditions: Typically P → 0 as r → +∞, P → face as r → -∞

Numerical Schemes
-----------------
- Explicit: Simple but conditionally stable
- Implicit: Unconditionally stable, requires tridiagonal solve
- Crank-Nicolson: Second-order accurate, unconditionally stable (default)
- Theta scheme: Generalization with stability parameter θ

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Optional

# Import instruments (only Simple variants used for FD pricing).
from src.instruments.ir.linear.bond import IrBondZeroCouponSimple
from src.instruments.ir.options.bond import IrBondEuropeanOptionSimple
from src.instruments.ir.options.capfloor import (
    IrCapletEuropeanOptionSimple,
    IrFloorletEuropeanOptionSimple,
)

# Import Hull-White model.
from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    hw_b_factor,
)

# Import base FD infrastructure.
from src.models.numeric.finite_difference.grids import SpatialGrid1D, TimeGrid
from src.models.numeric.finite_difference.tridiagonal import solve_tridiagonal


# Greek name type.
GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]

# FD scheme type.
FDScheme = Literal["explicit", "implicit", "crank_nicolson"]


# =============================================================================
# FD CONFIGURATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class FDConfig:
    """
    Configuration for Finite Difference pricing.
    
    Parameters
    ----------
    n_r : int
        Number of grid points in r-direction. Default: 200.
    n_t : int
        Number of time steps. Default: 200.
    r_std_mult : float
        Number of standard deviations for r-grid range. Default: 5.0.
    scheme : FDScheme
        Numerical scheme. Default: "crank_nicolson".
    theta : float
        Theta parameter for theta-scheme (0=explicit, 1=implicit, 0.5=CN).
        Default: 0.5 (Crank-Nicolson).
    """
    n_r: int = 200
    n_t: int = 200
    r_std_mult: float = 5.0
    scheme: FDScheme = "crank_nicolson"
    theta: float = 0.5


# Default FD configuration.
DEFAULT_FD_CONFIG = FDConfig()


# =============================================================================
# HULL-WHITE FD GRID (using base infrastructure)
# =============================================================================


@dataclass(frozen=True, slots=True)
class HWGrid:
    """
    Grid for Hull-White finite difference.
    
    Wraps base FD grid infrastructure (SpatialGrid1D, TimeGrid) with
    Hull-White specific grid construction logic.
    
    Attributes
    ----------
    spatial_grid : SpatialGrid1D
        Spatial grid for short rate r (from base FD infrastructure).
    time_grid : TimeGrid
        Time grid (from base FD infrastructure).
    
    Properties
    ----------
    r_grid, t_grid, dr, dt, n_r, n_t : Convenience accessors.
    """
    spatial_grid: SpatialGrid1D
    time_grid: TimeGrid
    
    @property
    def r_grid(self) -> np.ndarray:
        """Short rate grid values."""
        return self.spatial_grid.x
    
    @property
    def t_grid(self) -> np.ndarray:
        """Time grid values."""
        return self.time_grid.t
    
    @property
    def dr(self) -> float:
        """Grid spacing in r-direction."""
        return self.spatial_grid.dx
    
    @property
    def dt(self) -> float:
        """Grid spacing in t-direction (uniform assumption)."""
        return float(self.time_grid.dt[0])
    
    @property
    def n_r(self) -> int:
        """Number of r grid points."""
        return self.spatial_grid.n
    
    @property
    def n_t(self) -> int:
        """Number of time steps."""
        return self.time_grid.n - 1
    
    @classmethod
    def build(
        cls,
        params: HullWhiteParameters,
        maturity: float,
        config: FDConfig,
    ) -> "HWGrid":
        """
        Build FD grid for Hull-White model using base FD infrastructure.
        
        Parameters
        ----------
        params : HullWhiteParameters
            Hull-White model parameters.
        maturity : float
            Time to maturity.
        config : FDConfig
            Grid configuration.
        
        Returns
        -------
        HWGrid
            Constructed grid using SpatialGrid1D and TimeGrid.
        """
        r0 = params.r0
        sigma = params.sigma
        a = params.a
        
        # Compute long-term standard deviation of r.
        # Var[r(∞)] = σ²/(2a)
        long_term_std = sigma / np.sqrt(2.0 * a) if a > 1e-10 else sigma * np.sqrt(maturity)
        
        # r-grid range centered around r0.
        r_min = r0 - config.r_std_mult * long_term_std
        r_max = r0 + config.r_std_mult * long_term_std
        
        # Build grids using base infrastructure.
        spatial_grid = SpatialGrid1D.uniform(
            x_min=r_min,
            x_max=r_max,
            n=config.n_r,
            name="r",
        )
        
        time_grid = TimeGrid.uniform(
            t0=0.0,
            t1=maturity,
            n=config.n_t + 1,
            name="t",
        )
        
        return cls(
            spatial_grid=spatial_grid,
            time_grid=time_grid,
        )


# =============================================================================
# HULL-WHITE FD ZERO COUPON BOND PRICER
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondZeroCouponFDPricerSimple:
    """
    Finite Difference pricer for zero coupon bonds under Hull-White.
    
    Solves the PDE:
        ∂P/∂t + (θ - a·r)·∂P/∂r + ½σ²·∂²P/∂r² - r·P = 0
    
    with terminal condition P(r,T) = 1.
    
    Parameters
    ----------
    params : HullWhiteParameters
        Hull-White model parameters.
    config : FDConfig
        Finite difference configuration.
    """
    
    params: HullWhiteParameters
    config: FDConfig = DEFAULT_FD_CONFIG
    
    def price(self, trade: IrBondZeroCouponSimple) -> float:
        """
        Price a zero coupon bond using finite differences.
        
        Parameters
        ----------
        trade : IrBondZeroCouponSimple
            Zero coupon bond instrument.
        
        Returns
        -------
        float
            Present value of the bond.
        """
        T = float(trade.maturity)
        face = float(trade.face_value)
        
        if T <= 0.0:
            return face
        
        # Build grid.
        grid = HWGrid.build(self.params, T, self.config)
        
        # Solve PDE.
        P_0 = self._solve_bond_pde(grid, T)
        
        # Interpolate at r0.
        r0 = self.params.r0
        price_at_r0 = np.interp(r0, grid.r_grid, P_0)
        
        return face * price_at_r0
    
    def _solve_bond_pde(self, grid: HWGrid, T: float) -> np.ndarray:
        """
        Solve the bond pricing PDE backward in time.
        
        Returns
        -------
        np.ndarray
            Bond prices at t=0 for each r on the grid.
        """
        a = self.params.a
        sigma = self.params.sigma
        theta = self.params.theta
        
        dr = grid.dr
        dt = grid.dt
        n_r = grid.n_r
        r = grid.r_grid
        
        # Terminal condition: P(r, T) = 1.
        P = np.ones(n_r)
        
        # FD coefficients (for interior points).
        # Using theta-scheme: θ·implicit + (1-θ)·explicit
        fd_theta = self.config.theta
        
        # Coefficients for -∂²P/∂r² term.
        alpha = 0.5 * sigma ** 2 / dr ** 2
        
        # Coefficients for -(θ - a·r)·∂P/∂r term (central difference).
        # drift = θ - a·r (will be computed per point)
        
        # Time stepping (backward from T to 0).
        for n in range(grid.n_t, 0, -1):
            t_n = grid.t_grid[n]
            
            # Build tridiagonal system.
            # For interior points i = 1, ..., n_r-2.
            
            # Drift at each grid point.
            drift = theta - a * r
            
            # Convection coefficient (central difference for first derivative).
            beta = drift / (2.0 * dr)
            
            # Coefficients for explicit part [I + (1-θ)·dt·L].
            # L·P_i = (α+β)·P_{i+1} + (-2α-r_i)·P_i + (α-β)·P_{i-1}
            lower_exp = dt * (1 - fd_theta) * (alpha - beta[1:n_r - 1])
            diag_exp = 1.0 - dt * (1 - fd_theta) * (2 * alpha + r[1:n_r - 1])
            upper_exp = dt * (1 - fd_theta) * (alpha + beta[1:n_r - 1])
            
            # Coefficients for implicit part [I - θ·dt·L].
            lower_imp = -dt * fd_theta * (alpha - beta[1:n_r - 1])
            diag_imp = 1.0 + dt * fd_theta * (2 * alpha + r[1:n_r - 1])
            upper_imp = -dt * fd_theta * (alpha + beta[1:n_r - 1])
            
            # RHS from explicit step.
            rhs = np.zeros(n_r - 2)
            for i in range(1, n_r - 1):
                idx = i - 1
                rhs[idx] = (
                    lower_exp[idx] * P[i - 1]
                    + diag_exp[idx] * P[i]
                    + upper_exp[idx] * P[i + 1]
                )
            
            # Boundary conditions.
            # At r_min (low rates): P → exp(-r_min × τ) where τ = remaining time.
            # At r_max (high rates): P → exp(-r_max × τ) (far from money, still discounted).
            tau_remaining = grid.t_grid[-1] - t_n  # Time REMAINING to maturity (T - t).
            P_low = np.exp(-r[0] * tau_remaining) if tau_remaining > 0 else 1.0
            P_high = np.exp(-r[-1] * tau_remaining) if tau_remaining > 0 else 1.0
            
            # Adjust RHS for boundary.
            rhs[0] -= lower_imp[0] * P_low
            rhs[-1] -= upper_imp[-1] * P_high
            
            # Solve tridiagonal system for interior.
            P_interior = solve_tridiagonal(
                lower_imp[1:],
                diag_imp,
                upper_imp[:-1],
                rhs,
            )
            
            # Update P.
            P[0] = P_low
            P[1:n_r - 1] = P_interior
            P[-1] = P_high
        
        return P
    
    def greeks(self, trade: IrBondZeroCouponSimple) -> Dict[str, float]:
        """Compute Greeks via finite difference bumping."""
        base_price = self.price(trade)
        T = float(trade.maturity)
        
        if T <= 0.0:
            return {"delta": 0.0, "dv01": 0.0, "vega": 0.0}
        
        # Delta: bump r0.
        dr = 0.0001
        params_up = HullWhiteParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 + dr,
            theta=self.params.theta,
        )
        price_up = IrBondZeroCouponFDPricerSimple(params=params_up, config=self.config).price(trade)
        delta = (price_up - base_price) / dr
        
        # DV01.
        dv01 = abs(delta) * 0.0001
        
        # Vega: bump sigma.
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a,
            sigma=self.params.sigma + d_sigma,
            r0=self.params.r0,
            theta=self.params.theta,
        )
        price_sigma_up = IrBondZeroCouponFDPricerSimple(
            params=params_sigma_up, config=self.config
        ).price(trade)
        vega = (price_sigma_up - base_price) / d_sigma
        
        return {
            "delta": delta,
            "dv01": dv01,
            "vega": vega,
        }


# =============================================================================
# HULL-WHITE FD BOND OPTION PRICER
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondEuropeanOptionFDPricerSimple:
    """
    Finite Difference pricer for European bond options under Hull-White.
    
    Solves the PDE for option value V(r,t) with terminal condition:
        V(r, T_opt) = max(P(r, T_opt; T_bond) - K, 0)  for call
        V(r, T_opt) = max(K - P(r, T_opt; T_bond), 0)  for put
    
    Parameters
    ----------
    params : HullWhiteParameters
        Hull-White model parameters.
    config : FDConfig
        Finite difference configuration.
    """
    
    params: HullWhiteParameters
    config: FDConfig = DEFAULT_FD_CONFIG
    
    def price(self, trade: IrBondEuropeanOptionSimple) -> float:
        """
        Price a bond option using finite differences.
        
        Parameters
        ----------
        trade : IrBondEuropeanOptionSimple
            Bond option with direct parameters.
        
        Returns
        -------
        float
            Present value of the option.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_opt = float(trade.expiry)
        F = float(trade.forward_bond_price)
        df = float(trade.discount_factor)
        opt_type = trade.option_type
        
        if T_opt <= 0.0:
            if opt_type == "call":
                return N * df * max(F - K, 0.0)
            return N * df * max(K - F, 0.0)
        
        # Infer bond maturity.
        r0 = self.params.r0
        if r0 != 0:
            T_bond = -math.log(F * df) / r0 if F * df > 0 else T_opt + 1.0
        else:
            T_bond = T_opt + 1.0
        
        if T_bond <= T_opt:
            T_bond = T_opt + 0.5
        
        # Build grid to option expiry.
        grid = HWGrid.build(self.params, T_opt, self.config)
        
        # First, compute bond prices P(r, T_opt; T_bond) for all r on grid.
        # Using HW analytic formula for P at option expiry.
        tau = T_bond - T_opt
        a = self.params.a
        sigma = self.params.sigma
        theta = self.params.theta
        r_grid = grid.r_grid
        
        # B(T_opt, T_bond).
        B = hw_b_factor(a, tau)
        
        # Approximate A(T_opt, T_bond) - for flat curve assumption.
        log_A = -theta * tau + (sigma ** 2 / (4 * a)) * B ** 2 * (1 - np.exp(-2 * a * T_opt))
        P_bond = np.exp(log_A - B * r_grid)
        
        # Scale to match forward (P_bond represents % of face).
        P_bond_scaled = P_bond * 100.0  # Face = 100
        
        # Terminal condition at T_opt.
        if opt_type == "call":
            V = np.maximum(P_bond_scaled - K, 0.0)
        else:
            V = np.maximum(K - P_bond_scaled, 0.0)
        
        # Solve option PDE backward from T_opt to 0.
        V = self._solve_option_pde(grid, V)
        
        # Interpolate at r0.
        price_at_r0 = np.interp(r0, r_grid, V)
        
        return N * price_at_r0 / 100.0  # Normalize
    
    def _solve_option_pde(self, grid: HWGrid, V_terminal: np.ndarray) -> np.ndarray:
        """
        Solve option pricing PDE backward in time.
        
        Parameters
        ----------
        grid : HWGrid
            FD grid.
        V_terminal : np.ndarray
            Terminal condition (payoff at expiry).
        
        Returns
        -------
        np.ndarray
            Option values at t=0 for each r on grid.
        """
        a = self.params.a
        sigma = self.params.sigma
        theta = self.params.theta
        
        dr = grid.dr
        dt = grid.dt
        n_r = grid.n_r
        r = grid.r_grid
        
        V = V_terminal.copy()
        fd_theta = self.config.theta
        
        # Coefficients.
        alpha = 0.5 * sigma ** 2 / dr ** 2
        drift = theta - a * r
        beta = drift / (2.0 * dr)
        
        # Time stepping.
        for n in range(grid.n_t, 0, -1):
            # Explicit coefficients [I + (1-θ)·dt·L].
            lower_exp = dt * (1 - fd_theta) * (alpha - beta[1:n_r - 1])
            diag_exp = 1.0 - dt * (1 - fd_theta) * (2 * alpha + r[1:n_r - 1])
            upper_exp = dt * (1 - fd_theta) * (alpha + beta[1:n_r - 1])
            
            # Implicit coefficients [I - θ·dt·L].
            lower_imp = -dt * fd_theta * (alpha - beta[1:n_r - 1])
            diag_imp = 1.0 + dt * fd_theta * (2 * alpha + r[1:n_r - 1])
            upper_imp = -dt * fd_theta * (alpha + beta[1:n_r - 1])
            
            # RHS from explicit step.
            rhs = np.zeros(n_r - 2)
            for i in range(1, n_r - 1):
                idx = i - 1
                rhs[idx] = (
                    lower_exp[idx] * V[i - 1]
                    + diag_exp[idx] * V[i]
                    + upper_exp[idx] * V[i + 1]
                )
            
            # Boundary conditions (option value → 0 at extremes).
            V_low = 0.0
            V_high = 0.0
            
            rhs[0] -= lower_imp[0] * V_low
            rhs[-1] -= upper_imp[-1] * V_high
            
            # Solve.
            V_interior = solve_tridiagonal(
                lower_imp[1:],
                diag_imp,
                upper_imp[:-1],
                rhs,
            )
            
            V[0] = V_low
            V[1:n_r - 1] = V_interior
            V[-1] = V_high
        
        return V
    
    def greeks(self, trade: IrBondEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks via finite difference bumping."""
        T_opt = float(trade.expiry)
        
        if T_opt <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        base_price = self.price(trade)
        
        # Delta.
        dr = 0.0001
        params_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma,
            r0=self.params.r0 + dr, theta=self.params.theta,
        )
        params_dn = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma,
            r0=self.params.r0 - dr, theta=self.params.theta,
        )
        price_up = IrBondEuropeanOptionFDPricerSimple(params=params_up, config=self.config).price(trade)
        price_dn = IrBondEuropeanOptionFDPricerSimple(params=params_dn, config=self.config).price(trade)
        
        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        # Vega.
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma + d_sigma,
            r0=self.params.r0, theta=self.params.theta,
        )
        price_sigma_up = IrBondEuropeanOptionFDPricerSimple(
            params=params_sigma_up, config=self.config
        ).price(trade)
        vega = (price_sigma_up - base_price) / d_sigma
        
        # Theta (simplified).
        theta_greek = -base_price / T_opt if T_opt > 0 else 0.0
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta_greek,
            "rho": delta,
        }


# =============================================================================
# HULL-WHITE FD CAPLET/FLOORLET PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrCapletEuropeanOptionFDPricerSimple:
    """
    Finite Difference pricer for caplets under Hull-White.
    """
    
    params: HullWhiteParameters
    config: FDConfig = DEFAULT_FD_CONFIG
    
    def price(self, trade: IrCapletEuropeanOptionSimple) -> float:
        """Price a caplet using finite differences."""
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        T_pay = float(trade.payment_time)
        tau = float(trade.accrual_factor)
        
        if T_fix <= 0.0:
            F = float(trade.forward_rate)
            df = float(trade.discount_factor)
            return N * tau * df * max(F - K, 0.0)
        
        # Caplet = (1 + τK) × put on ZC bond
        # Build equivalent bond option.
        K_bond = 1.0 / (1.0 + tau * K)
        
        r0 = self.params.r0
        P_0_fix = math.exp(-r0 * T_fix)
        P_0_pay = math.exp(-r0 * T_pay)
        F_bond = P_0_pay / P_0_fix  # Forward bond price
        
        # Create bond option.
        bond_option = IrBondEuropeanOptionSimple(
            notional=N * (1.0 + tau * K),
            strike=K_bond * 100.0,  # Scale to face=100
            expiry=T_fix,
            forward_bond_price=F_bond * 100.0,
            vol=self.params.sigma,  # Not used directly in HW FD
            discount_factor=P_0_fix,
            option_type="put",
        )
        
        # Price using FD bond option pricer.
        fd_pricer = IrBondEuropeanOptionFDPricerSimple(params=self.params, config=self.config)
        return fd_pricer.price(bond_option)
    
    def greeks(self, trade: IrCapletEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks via finite difference."""
        T_fix = float(trade.fixing_time)
        
        if T_fix <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        base_price = self.price(trade)
        
        dr = 0.0001
        params_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma,
            r0=self.params.r0 + dr, theta=self.params.theta,
        )
        params_dn = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma,
            r0=self.params.r0 - dr, theta=self.params.theta,
        )
        price_up = IrCapletEuropeanOptionFDPricerSimple(params=params_up, config=self.config).price(trade)
        price_dn = IrCapletEuropeanOptionFDPricerSimple(params=params_dn, config=self.config).price(trade)
        
        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma + d_sigma,
            r0=self.params.r0, theta=self.params.theta,
        )
        price_sigma_up = IrCapletEuropeanOptionFDPricerSimple(
            params=params_sigma_up, config=self.config
        ).price(trade)
        vega = (price_sigma_up - base_price) / d_sigma
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": 0.0,
            "rho": delta,
        }


@dataclass(frozen=True, slots=True)
class IrFloorletEuropeanOptionFDPricerSimple:
    """
    Finite Difference pricer for floorlets under Hull-White.
    """
    
    params: HullWhiteParameters
    config: FDConfig = DEFAULT_FD_CONFIG
    
    def price(self, trade: IrFloorletEuropeanOptionSimple) -> float:
        """Price a floorlet using finite differences."""
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        T_pay = float(trade.payment_time)
        tau = float(trade.accrual_factor)
        
        if T_fix <= 0.0:
            F = float(trade.forward_rate)
            df = float(trade.discount_factor)
            return N * tau * df * max(K - F, 0.0)
        
        # Floorlet = (1 + τK) × call on ZC bond.
        K_bond = 1.0 / (1.0 + tau * K)
        
        r0 = self.params.r0
        P_0_fix = math.exp(-r0 * T_fix)
        P_0_pay = math.exp(-r0 * T_pay)
        F_bond = P_0_pay / P_0_fix
        
        bond_option = IrBondEuropeanOptionSimple(
            notional=N * (1.0 + tau * K),
            strike=K_bond * 100.0,
            expiry=T_fix,
            forward_bond_price=F_bond * 100.0,
            vol=self.params.sigma,
            discount_factor=P_0_fix,
            option_type="call",
        )
        
        fd_pricer = IrBondEuropeanOptionFDPricerSimple(params=self.params, config=self.config)
        return fd_pricer.price(bond_option)
    
    def greeks(self, trade: IrFloorletEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks via finite difference."""
        T_fix = float(trade.fixing_time)
        
        if T_fix <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        base_price = self.price(trade)
        
        dr = 0.0001
        params_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma,
            r0=self.params.r0 + dr, theta=self.params.theta,
        )
        params_dn = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma,
            r0=self.params.r0 - dr, theta=self.params.theta,
        )
        price_up = IrFloorletEuropeanOptionFDPricerSimple(params=params_up, config=self.config).price(trade)
        price_dn = IrFloorletEuropeanOptionFDPricerSimple(params=params_dn, config=self.config).price(trade)
        
        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma + d_sigma,
            r0=self.params.r0, theta=self.params.theta,
        )
        price_sigma_up = IrFloorletEuropeanOptionFDPricerSimple(
            params=params_sigma_up, config=self.config
        ).price(trade)
        vega = (price_sigma_up - base_price) / d_sigma
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": 0.0,
            "rho": delta,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "FDConfig",
    "DEFAULT_FD_CONFIG",
    "FDScheme",
    # Grid
    "HWGrid",
    # Zero coupon bond FD pricer
    "IrBondZeroCouponFDPricerSimple",
    # Bond option FD pricer
    "IrBondEuropeanOptionFDPricerSimple",
    # Caplet/Floorlet FD pricers
    "IrCapletEuropeanOptionFDPricerSimple",
    "IrFloorletEuropeanOptionFDPricerSimple",
]
