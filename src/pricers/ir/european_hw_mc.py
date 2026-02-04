# src/pricers/ir/european_mc.py
"""
Interest Rate Monte Carlo Pricers.

Monte Carlo pricers for IR derivatives using short rate models (Hull-White, etc.).

Mathematical Framework
----------------------
Monte Carlo pricing for IR derivatives involves:
1. Simulating short rate paths r(t) under the risk-neutral measure
2. Computing path-wise discount factors: D(0,T) = exp(-∫₀ᵀ r(s) ds)
3. Evaluating payoffs along paths
4. Averaging discounted payoffs: PV = E[D(0,T) × Payoff]

Hull-White Simulation
---------------------
For Hull-White: dr(t) = [θ(t) - a×r(t)] dt + σ dW(t)

The exact distribution is:
    r(t+Δt) | r(t) ~ N(μ, v²)
    
where:
    μ = θ + (r(t) - θ)×exp(-a×Δt)
    v² = (σ²/(2a))×(1 - exp(-2a×Δt))

Variance Reduction
------------------
- Antithetic variates: Average of path and negated-increment path
- Control variates: Use analytic ZC bond price as control

Greeks via MC
-------------
- Delta: Finite difference bump of initial rate
- Vega: Finite difference bump of volatility
- Pathwise derivatives (advanced)

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Optional

# Import instruments (only Simple variants used for MC pricing).
from src.instruments.ir.linear.bond import IrBondZeroCouponSimple
from src.instruments.ir.options.bond import IrBondEuropeanOptionSimple
from src.instruments.ir.options.capfloor import (
    IrCapletEuropeanOptionSimple,
    IrFloorletEuropeanOptionSimple,
)
from src.instruments.ir.options.swaption import IrSwaptionEuropeanOptionSimple

# Import Hull-White model.
from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    HullWhiteDynamics,
)

# Import base MC infrastructure.
from src.models.numeric.monte_carlo.base import MonteCarloEstimate
from src.models.numeric.monte_carlo.estimators import estimate_from_samples


# Greek name type.
GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]


# =============================================================================
# MC CONFIGURATION
# =============================================================================


@dataclass(frozen=True, slots=True)
class MCConfig:
    """
    Configuration for Monte Carlo pricing.
    
    Parameters
    ----------
    n_paths : int
        Number of simulation paths. Default: 100,000.
    n_steps : int
        Number of time steps. Default: 252 (daily for 1 year).
    seed : int, optional
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates. Default: True.
    """
    n_paths: int = 100_000
    n_steps: int = 252
    seed: Optional[int] = None
    antithetic: bool = True


# Default MC configuration.
DEFAULT_MC_CONFIG = MCConfig()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _compute_discount_factor_from_paths(
    rate_paths: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Compute path-wise discount factors from rate paths using trapezoidal rule.
    
    D(0,T) = exp(-∫₀ᵀ r(s) ds)
    
    Parameters
    ----------
    rate_paths : np.ndarray
        Short rate paths, shape (n_paths, n_steps + 1).
    times : np.ndarray
        Time grid, shape (n_steps + 1,).
    
    Returns
    -------
    np.ndarray
        Discount factors, shape (n_paths,).
    """
    n_steps = len(times) - 1
    dt = times[1] - times[0]  # Assuming uniform grid
    
    # Trapezoidal rule: ∫r(s)ds ≈ dt × [0.5×r_0 + r_1 + ... + r_{n-1} + 0.5×r_n]
    integral = dt * (
        0.5 * rate_paths[:, 0]
        + np.sum(rate_paths[:, 1:-1], axis=1)
        + 0.5 * rate_paths[:, -1]
    )
    
    return np.exp(-integral)


def _compute_zc_bond_price_at_t(
    rate_paths: np.ndarray,
    times: np.ndarray,
    t_idx: int,
    T: float,
    params: HullWhiteParameters,
) -> np.ndarray:
    """
    Compute ZC bond price P(t, T) at time t for each path using HW formula.
    
    P(t,T) = A(t,T) × exp(-B(t,T) × r(t))
    
    For simplicity, we use the approximation:
    P(t,T) ≈ exp(-r(t) × (T-t))  (first-order approximation)
    
    Parameters
    ----------
    rate_paths : np.ndarray
        Short rate paths, shape (n_paths, n_steps + 1).
    times : np.ndarray
        Time grid.
    t_idx : int
        Time index in the path.
    T : float
        Bond maturity.
    params : HullWhiteParameters
        Hull-White parameters.
    
    Returns
    -------
    np.ndarray
        Bond prices at time t for each path.
    """
    t = times[t_idx]
    tau = T - t
    
    if tau <= 0:
        return np.ones(rate_paths.shape[0])
    
    r_t = rate_paths[:, t_idx]
    a = params.a
    sigma = params.sigma
    theta = params.theta
    
    # B(t,T) factor
    if abs(a) < 1e-10:
        B = tau
    else:
        B = (1 - np.exp(-a * tau)) / a
    
    # Simplified A(t,T) assuming flat initial curve
    # A ≈ exp(-theta × tau + 0.5 × (σ/a)² × (tau - B) × something)
    # For simplicity, use first-order approximation
    log_A = -theta * tau + (sigma ** 2 / (4 * a)) * B ** 2 * (1 - np.exp(-2 * a * t))
    
    return np.exp(log_A - B * r_t)


# =============================================================================
# HULL-WHITE MC BOND PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondZeroCouponMCPricerSimple:
    """
    Monte Carlo pricer for zero coupon bonds under Hull-White.
    
    Uses path simulation to compute:
        PV = E[exp(-∫₀ᵀ r(s) ds)] × Face
    
    Parameters
    ----------
    params : HullWhiteParameters
        Hull-White model parameters.
    config : MCConfig
        Monte Carlo configuration.
    """
    
    params: HullWhiteParameters
    config: MCConfig = DEFAULT_MC_CONFIG
    
    def price(self, trade: IrBondZeroCouponSimple) -> float:
        """
        Price a zero coupon bond using Monte Carlo.
        
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
        
        # Simulate paths
        dynamics = HullWhiteDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T,
            n_paths=self.config.n_paths,
            n_steps=max(int(T * 252), 10),  # Daily steps
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )
        
        # Average discount factors
        mean_df = np.mean(sim.discount_factors)
        
        return face * mean_df
    
    def price_with_estimate(self, trade: IrBondZeroCouponSimple) -> MonteCarloEstimate:
        """
        Price with full Monte Carlo estimate including standard error and confidence interval.
        
        Returns
        -------
        MonteCarloEstimate
            Contains mean, stderr, n_paths, and 95% confidence interval.
        """
        T = float(trade.maturity)
        face = float(trade.face_value)
        
        if T <= 0.0:
            return MonteCarloEstimate(mean=face, stderr=0.0, n_paths=1)
        
        dynamics = HullWhiteDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T,
            n_paths=self.config.n_paths,
            n_steps=max(int(T * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )
        
        payoffs = face * sim.discount_factors
        
        return estimate_from_samples(
            payoffs,
            meta={
                "instrument": "IrBondZeroCoupon",
                "maturity": T,
                "seed": self.config.seed,
                "antithetic": self.config.antithetic,
            },
        )
    
    def greeks(self, trade: IrBondZeroCouponSimple) -> Dict[str, float]:
        """Compute Greeks via finite difference bumping."""
        base_price = self.price(trade)
        T = float(trade.maturity)
        face = float(trade.face_value)
        
        if T <= 0.0:
            return {"delta": 0.0, "dv01": 0.0, "vega": 0.0}
        
        # Delta: bump r0
        dr = 0.0001
        params_up = HullWhiteParameters(
            a=self.params.a,
            sigma=self.params.sigma,
            r0=self.params.r0 + dr,
            theta=self.params.theta,
        )
        pricer_up = IrBondZeroCouponMCPricerSimple(params=params_up, config=self.config)
        price_up = pricer_up.price(trade)
        delta = (price_up - base_price) / dr
        
        # DV01
        dv01 = abs(delta) * 0.0001
        
        # Vega: bump sigma
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a,
            sigma=self.params.sigma + d_sigma,
            r0=self.params.r0,
            theta=self.params.theta,
        )
        pricer_sigma_up = IrBondZeroCouponMCPricerSimple(params=params_sigma_up, config=self.config)
        price_sigma_up = pricer_sigma_up.price(trade)
        vega = (price_sigma_up - base_price) / d_sigma
        
        return {
            "delta": delta,
            "dv01": dv01,
            "vega": vega,
        }


# =============================================================================
# HULL-WHITE MC BOND OPTION PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondEuropeanOptionMCPricerSimple:
    """
    Monte Carlo pricer for European bond options under Hull-White.
    
    Simulates short rate paths and evaluates:
        Call: PV = E[D(0,S) × max(P(S,T) - K, 0)]
        Put:  PV = E[D(0,S) × max(K - P(S,T), 0)]
    
    where S = option expiry, T = bond maturity.
    
    Parameters
    ----------
    params : HullWhiteParameters
        Hull-White model parameters.
    config : MCConfig
        Monte Carlo configuration.
    """
    
    params: HullWhiteParameters
    config: MCConfig = DEFAULT_MC_CONFIG
    
    def price(self, trade: IrBondEuropeanOptionSimple) -> float:
        """
        Price a bond option using Monte Carlo.
        
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
        opt_type = trade.option_type
        
        if T_opt <= 0.0:
            # Expired
            if opt_type == "call":
                return N * max(F - K, 0.0)
            return N * max(K - F, 0.0)
        
        # Infer bond maturity from forward and discount factor
        df = float(trade.discount_factor)
        r0 = self.params.r0
        if r0 != 0:
            T_bond = T_opt - math.log(F * df) / r0
        else:
            T_bond = T_opt + 1.0  # Default
        
        if T_bond <= T_opt:
            T_bond = T_opt + 0.5
        
        # Simulate paths to option expiry
        dynamics = HullWhiteDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_opt,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_opt * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )
        
        # Compute bond price at option expiry for each path
        # P(S, T) using HW formula with r(S)
        r_S = sim.terminal_rates
        tau = T_bond - T_opt
        
        # B(S, T) factor
        a = self.params.a
        sigma = self.params.sigma
        
        if abs(a) < 1e-10:
            B = tau
        else:
            B = (1 - np.exp(-a * tau)) / a
        
        # Simplified bond price at S
        # For more accuracy, would need full A(S,T) computation
        # Here we use P(S,T) ≈ exp(-r(S) × τ) as first approximation
        # then adjust based on HW convexity
        log_A = (sigma ** 2 / (4 * a)) * B ** 2 * (1 - np.exp(-2 * a * T_opt))
        P_S_T = np.exp(log_A - B * r_S)
        
        # Scale to match forward
        # F = E[P(S,T)] / P(0,S) => P(S,T) paths should average to F × P(0,S)
        # We'll use the simulated values directly, scaling for notional
        bond_prices = P_S_T * 100.0  # Assuming face = 100
        
        # Compute payoffs
        if opt_type == "call":
            payoffs = np.maximum(bond_prices - K, 0.0)
        else:
            payoffs = np.maximum(K - bond_prices, 0.0)
        
        # Discount payoffs
        discounted_payoffs = sim.discount_factors * payoffs
        
        # Average and scale by notional
        mean_pv = np.mean(discounted_payoffs)
        
        return N * mean_pv / 100.0  # Normalize by face
    
    def price_with_estimate(self, trade: IrBondEuropeanOptionSimple) -> MonteCarloEstimate:
        """
        Price with full Monte Carlo estimate including standard error and confidence interval.
        
        Returns
        -------
        MonteCarloEstimate
            Contains mean, stderr, n_paths, and 95% confidence interval.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T_opt = float(trade.expiry)
        F = float(trade.forward_bond_price)
        opt_type = trade.option_type
        
        if T_opt <= 0.0:
            intrinsic = N * max(F - K, 0.0) if opt_type == "call" else N * max(K - F, 0.0)
            return MonteCarloEstimate(mean=intrinsic, stderr=0.0, n_paths=1)
        
        df = float(trade.discount_factor)
        r0 = self.params.r0
        T_bond = T_opt + 1.0 if r0 == 0 else T_opt - math.log(F * df) / r0
        if T_bond <= T_opt:
            T_bond = T_opt + 0.5
        
        dynamics = HullWhiteDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_opt,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_opt * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )
        
        r_S = sim.terminal_rates
        tau = T_bond - T_opt
        a = self.params.a
        sigma = self.params.sigma
        
        B = tau if abs(a) < 1e-10 else (1 - np.exp(-a * tau)) / a
        log_A = (sigma ** 2 / (4 * a)) * B ** 2 * (1 - np.exp(-2 * a * T_opt))
        P_S_T = np.exp(log_A - B * r_S)
        bond_prices = P_S_T * 100.0
        
        if opt_type == "call":
            payoffs = np.maximum(bond_prices - K, 0.0)
        else:
            payoffs = np.maximum(K - bond_prices, 0.0)
        
        discounted_payoffs = sim.discount_factors * payoffs * N / 100.0
        
        return estimate_from_samples(
            discounted_payoffs,
            meta={
                "instrument": "IrBondEuropeanOption",
                "option_type": opt_type,
                "expiry": T_opt,
                "strike": K,
                "seed": self.config.seed,
                "antithetic": self.config.antithetic,
            },
        )
    
    def greeks(self, trade: IrBondEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks via finite difference bumping."""
        T_opt = float(trade.expiry)
        
        if T_opt <= 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        
        base_price = self.price(trade)
        
        # Delta: bump r0
        dr = 0.0001
        params_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma,
            r0=self.params.r0 + dr, theta=self.params.theta,
        )
        params_dn = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma,
            r0=self.params.r0 - dr, theta=self.params.theta,
        )
        price_up = IrBondEuropeanOptionMCPricerSimple(params=params_up, config=self.config).price(trade)
        price_dn = IrBondEuropeanOptionMCPricerSimple(params=params_dn, config=self.config).price(trade)
        
        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        # Vega: bump sigma
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma + d_sigma,
            r0=self.params.r0, theta=self.params.theta,
        )
        price_sigma_up = IrBondEuropeanOptionMCPricerSimple(
            params=params_sigma_up, config=self.config
        ).price(trade)
        vega = (price_sigma_up - base_price) / d_sigma
        
        # Theta (simplified)
        theta = -base_price / T_opt if T_opt > 0 else 0.0
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": delta,
        }


# =============================================================================
# HULL-WHITE MC CAPLET/FLOORLET PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrCapletEuropeanOptionMCPricerSimple:
    """
    Monte Carlo pricer for caplets under Hull-White.
    
    A caplet on LIBOR pays: N × τ × max(L(T_fix, T_pay) - K, 0) at T_pay
    
    Under MC, we simulate r(t) to T_pay and compute:
        PV = E[D(0, T_pay) × N × τ × max(L - K, 0)]
    
    where L = (P(T_fix, T_pay)^{-1} - 1) / τ
    """
    
    params: HullWhiteParameters
    config: MCConfig = DEFAULT_MC_CONFIG
    
    def price(self, trade: IrCapletEuropeanOptionSimple) -> float:
        """Price a caplet using Monte Carlo."""
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        T_pay = float(trade.payment_time)
        tau = float(trade.accrual_factor)
        
        if T_fix <= 0.0:
            F = float(trade.forward_rate)
            df = float(trade.discount_factor)
            return N * tau * df * max(F - K, 0.0)
        
        # Simulate paths to payment date
        dynamics = HullWhiteDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_pay,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_pay * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )
        
        # Find index closest to T_fix
        fix_idx = int(T_fix / T_pay * sim.n_steps)
        fix_idx = min(fix_idx, sim.n_steps - 1)
        
        # Compute forward rate L at fixing time
        r_fix = sim.rate_paths[:, fix_idx]
        tau_period = T_pay - T_fix
        
        # P(T_fix, T_pay) ≈ exp(-r(T_fix) × τ)
        a = self.params.a
        B = tau_period if abs(a) < 1e-10 else (1 - np.exp(-a * tau_period)) / a
        P_fix_pay = np.exp(-B * r_fix)
        
        # Forward LIBOR: L = (1/P - 1) / τ
        L = (1.0 / P_fix_pay - 1.0) / tau
        
        # Caplet payoff at T_pay
        payoffs = N * tau * np.maximum(L - K, 0.0)
        
        # Discount to today
        discounted_payoffs = sim.discount_factors * payoffs
        
        return float(np.mean(discounted_payoffs))
    
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
        price_up = IrCapletEuropeanOptionMCPricerSimple(params=params_up, config=self.config).price(trade)
        price_dn = IrCapletEuropeanOptionMCPricerSimple(params=params_dn, config=self.config).price(trade)
        
        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma + d_sigma,
            r0=self.params.r0, theta=self.params.theta,
        )
        price_sigma_up = IrCapletEuropeanOptionMCPricerSimple(
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
class IrFloorletEuropeanOptionMCPricerSimple:
    """
    Monte Carlo pricer for floorlets under Hull-White.
    """
    
    params: HullWhiteParameters
    config: MCConfig = DEFAULT_MC_CONFIG
    
    def price(self, trade: IrFloorletEuropeanOptionSimple) -> float:
        """Price a floorlet using Monte Carlo."""
        N = float(trade.notional)
        K = float(trade.strike)
        T_fix = float(trade.fixing_time)
        T_pay = float(trade.payment_time)
        tau = float(trade.accrual_factor)
        
        if T_fix <= 0.0:
            F = float(trade.forward_rate)
            df = float(trade.discount_factor)
            return N * tau * df * max(K - F, 0.0)
        
        dynamics = HullWhiteDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_pay,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_pay * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )
        
        fix_idx = int(T_fix / T_pay * sim.n_steps)
        fix_idx = min(fix_idx, sim.n_steps - 1)
        
        r_fix = sim.rate_paths[:, fix_idx]
        tau_period = T_pay - T_fix
        
        a = self.params.a
        B = tau_period if abs(a) < 1e-10 else (1 - np.exp(-a * tau_period)) / a
        P_fix_pay = np.exp(-B * r_fix)
        
        L = (1.0 / P_fix_pay - 1.0) / tau
        payoffs = N * tau * np.maximum(K - L, 0.0)
        discounted_payoffs = sim.discount_factors * payoffs
        
        return float(np.mean(discounted_payoffs))
    
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
        price_up = IrFloorletEuropeanOptionMCPricerSimple(params=params_up, config=self.config).price(trade)
        price_dn = IrFloorletEuropeanOptionMCPricerSimple(params=params_dn, config=self.config).price(trade)
        
        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma + d_sigma,
            r0=self.params.r0, theta=self.params.theta,
        )
        price_sigma_up = IrFloorletEuropeanOptionMCPricerSimple(
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
# HULL-WHITE MC SWAPTION PRICER
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrSwaptionEuropeanOptionMCPricerSimple:
    """
    Monte Carlo pricer for European swaptions under Hull-White.
    
    At option expiry S, the payer swaption payoff is:
        max(Swap Value, 0) × Annuity
    
    where Swap Value = (Swap Rate - K) and evaluated at time S.
    """
    
    params: HullWhiteParameters
    config: MCConfig = DEFAULT_MC_CONFIG
    
    def price(self, trade: IrSwaptionEuropeanOptionSimple) -> float:
        """Price a swaption using Monte Carlo."""
        N = float(trade.notional)
        K = float(trade.fixed_rate)
        T_opt = float(trade.expiry)
        is_payer = (trade.option_type == "payer")
        
        if T_opt <= 0.0:
            swap_rate = float(trade.swap_rate)
            annuity = float(trade.annuity)
            if is_payer:
                return N * max(swap_rate - K, 0.0) * annuity
            return N * max(K - swap_rate, 0.0) * annuity
        
        # Get swap schedule
        payment_times = np.array(trade.payment_times)
        dcfs = np.array(trade.day_count_fractions)
        
        # Simulate paths to option expiry
        dynamics = HullWhiteDynamics(params=self.params)
        sim = dynamics.simulate(
            maturity=T_opt,
            n_paths=self.config.n_paths,
            n_steps=max(int(T_opt * 252), 10),
            scheme="exact",
            seed=self.config.seed,
            antithetic=self.config.antithetic,
            compute_discount_factors=True,
        )
        
        # At option expiry, compute swap rate for each path
        r_S = sim.terminal_rates
        a = self.params.a
        
        # Compute bond prices P(S, T_i) for each payment date
        # P(S, T_i) ≈ exp(-B(S,T_i) × r(S))
        n_payments = len(payment_times)
        bond_prices = np.zeros((sim.n_paths, n_payments))
        
        for i, T_i in enumerate(payment_times):
            tau_i = T_i - T_opt
            if tau_i <= 0:
                bond_prices[:, i] = 1.0
            else:
                B_i = tau_i if abs(a) < 1e-10 else (1 - np.exp(-a * tau_i)) / a
                bond_prices[:, i] = np.exp(-B_i * r_S)
        
        # Annuity = Σ τ_i × P(S, T_i)
        annuities = np.sum(dcfs * bond_prices, axis=1)
        
        # Swap rate = (1 - P(S, T_n)) / Annuity
        swap_rates = (1.0 - bond_prices[:, -1]) / annuities
        
        # Swaption payoff
        if is_payer:
            payoffs = N * np.maximum(swap_rates - K, 0.0) * annuities
        else:
            payoffs = N * np.maximum(K - swap_rates, 0.0) * annuities
        
        # Discount to today
        discounted_payoffs = sim.discount_factors * payoffs
        
        return float(np.mean(discounted_payoffs))
    
    def greeks(self, trade: IrSwaptionEuropeanOptionSimple) -> Dict[GreekName, float]:
        """Compute Greeks via finite difference."""
        T_opt = float(trade.expiry)
        
        if T_opt <= 0.0:
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
        price_up = IrSwaptionEuropeanOptionMCPricerSimple(params=params_up, config=self.config).price(trade)
        price_dn = IrSwaptionEuropeanOptionMCPricerSimple(params=params_dn, config=self.config).price(trade)
        
        delta = (price_up - price_dn) / (2 * dr)
        gamma = (price_up - 2 * base_price + price_dn) / (dr ** 2)
        
        d_sigma = 0.0001
        params_sigma_up = HullWhiteParameters(
            a=self.params.a, sigma=self.params.sigma + d_sigma,
            r0=self.params.r0, theta=self.params.theta,
        )
        price_sigma_up = IrSwaptionEuropeanOptionMCPricerSimple(
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
    "MCConfig",
    "DEFAULT_MC_CONFIG",
    # Base MC result type (re-export for convenience)
    "MonteCarloEstimate",
    # Zero coupon bond MC pricer
    "IrBondZeroCouponMCPricerSimple",
    # Bond option MC pricer
    "IrBondEuropeanOptionMCPricerSimple",
    # Caplet/Floorlet MC pricers
    "IrCapletEuropeanOptionMCPricerSimple",
    "IrFloorletEuropeanOptionMCPricerSimple",
    # Swaption MC pricer
    "IrSwaptionEuropeanOptionMCPricerSimple",
]
