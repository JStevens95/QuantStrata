"""
Hull-White One-Factor Short Rate Model Implementation.

This module provides the Hull-White model for pricing interest rate derivatives
where the short rate follows a mean-reverting Gaussian (Ornstein-Uhlenbeck) process.

Mathematical Framework
----------------------
The Hull-White model specifies the short rate dynamics under the risk-neutral measure:

    dr(t) = [θ(t) - a·r(t)] dt + σ dW(t)

where:
    - r(t): instantaneous short rate
    - θ(t): time-dependent drift (fitted to initial term structure)
    - a: mean reversion speed (a > 0)
    - σ: short rate volatility (σ > 0)

Key Properties
--------------
1. **Affine Model**: Bond prices have exponential-affine form P(t,T) = A(t,T)·exp(-B(t,T)·r(t))
2. **Gaussian Distribution**: r(t) is normally distributed (can go negative)
3. **Mean Reversion**: Long-term rate converges to θ/a
4. **Analytic Tractability**: Closed-form solutions for ZC bonds and European options
5. **Term Structure Fitting**: θ(t) calibrated to match initial yield curve exactly

Closed-Form Solutions
---------------------
**Zero-Coupon Bond Price:**
    P(t,T) = A(t,T) · exp(-B(t,T) · r(t))
    
    B(t,T) = (1 - exp(-a(T-t))) / a
    
    A(t,T) = P(0,T)/P(0,t) · exp(B(t,T)·f(0,t) - σ²/(4a)·B(t,T)²·(1-exp(-2at)))

**European Bond Option:**
    Call: P(0,T_bond)·N(h) - K·P(0,T_option)·N(h - σ_p)
    Put:  K·P(0,T_option)·N(-h + σ_p) - P(0,T_bond)·N(-h)
    
    σ_p = σ · √((1-exp(-2a·T_option))/(2a)) · B(T_option, T_bond)
    h = (1/σ_p) · ln(P(0,T_bond)/(K·P(0,T_option))) + σ_p/2

Simulation Schemes
------------------
- **Exact**: Uses exact OU transition distribution
- **Euler**: Simple Euler-Maruyama discretization

References
----------
- Hull, J. & White, A. (1990). "Pricing Interest-Rate-Derivative Securities."
  Review of Financial Studies.
- Hull, J. & White, A. (1994). "Numerical Procedures for Implementing Term
  Structure Models I: Single-Factor Models."
- Brigo, D. & Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Callable
from scipy.stats import norm

from src.models.numeric.monte_carlo.rng import NormalRng


# =============================================================================
# Type definitions
# =============================================================================

# Discretization schemes for Hull-White short rate process.
HullWhiteScheme = Literal["exact", "euler"]


# =============================================================================
# Hull-White Parameters
# =============================================================================

@dataclass(frozen=True)
class HullWhiteParameters:
    """
    Parameters for the Hull-White one-factor short rate model.

    The Hull-White model specifies:
        dr(t) = [θ(t) - a·r(t)] dt + σ dW(t)

    For practical use, we typically work with a constant mean reversion level,
    which simplifies to:
        dr(t) = a·(θ - r(t)) dt + σ dW(t)

    where θ = long-term rate level.

    Parameters
    ----------
    a : float
        Mean reversion speed (a > 0). Higher values = faster reversion.
        Typical values: 0.01 to 0.5.
    sigma : float
        Short rate volatility (σ > 0).
        Typical values: 0.005 to 0.02 (50-200 basis points annualized).
    r0 : float
        Initial short rate r(0).
        Can be negative (Hull-White allows negative rates).
    theta : float, optional
        Long-term mean reversion level. If None, will be inferred from
        initial term structure during pricing.
        Default: r0 (flat term structure assumption).

    Attributes
    ----------
    half_life : float
        Time for rate to revert halfway to mean: ln(2)/a.
    long_term_vol : float
        Asymptotic volatility of r(t): σ/√(2a).

    Examples
    --------
    >>> from src.models.short_rate.hull_white import HullWhiteParameters
    >>> params = HullWhiteParameters(
    ...     a=0.1,       # Mean reversion speed (10% per year)
    ...     sigma=0.01,  # 1% (100 bp) short rate vol
    ...     r0=0.03,     # 3% initial short rate
    ...     theta=0.04,  # 4% long-term rate
    ... )
    >>> params.half_life
    6.931471805599453
    >>> params.long_term_vol
    0.02236...
    """

    a: float       # Mean reversion speed
    sigma: float   # Short rate volatility
    r0: float      # Initial short rate
    theta: float = None  # Long-term mean level (optional)

    def __post_init__(self) -> None:
        """Validate Hull-White parameters."""
        # Validate a (mean reversion speed).
        if not np.isfinite(self.a):
            raise ValueError("a must be finite.")
        if self.a <= 0.0:
            raise ValueError("a (mean reversion speed) must be > 0.")

        # Validate sigma (volatility).
        if not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite.")
        if self.sigma <= 0.0:
            raise ValueError("sigma (volatility) must be > 0.")

        # Validate r0 (initial rate - can be negative for Hull-White).
        if not np.isfinite(self.r0):
            raise ValueError("r0 must be finite.")

        # Set theta to r0 if not provided (flat term structure).
        if self.theta is None:
            object.__setattr__(self, 'theta', self.r0)

        # Validate theta.
        if not np.isfinite(self.theta):
            raise ValueError("theta must be finite.")

    @property
    def half_life(self) -> float:
        """
        Half-life of mean reversion: time for r(t) to revert halfway to mean.
        
        Half-life = ln(2) / a
        """
        return np.log(2.0) / self.a

    @property
    def long_term_vol(self) -> float:
        """
        Asymptotic (long-term) volatility of r(t).
        
        As t → ∞, Var[r(t)] → σ²/(2a), so long-term vol = σ/√(2a).
        """
        return self.sigma / np.sqrt(2.0 * self.a)

    def expected_rate(self, t: float) -> float:
        """
        Expected short rate E[r(t)] under constant theta assumption.
        
        E[r(t)] = θ + (r₀ - θ)·exp(-a·t)
        """
        return self.theta + (self.r0 - self.theta) * np.exp(-self.a * t)

    def variance_rate(self, t: float) -> float:
        """
        Variance of short rate Var[r(t)].
        
        Var[r(t)] = (σ²/(2a))·(1 - exp(-2a·t))
        """
        return (self.sigma ** 2 / (2.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * t))

    def std_rate(self, t: float) -> float:
        """Standard deviation of r(t)."""
        return np.sqrt(self.variance_rate(t))


# =============================================================================
# Hull-White Simulation Output
# =============================================================================

@dataclass(frozen=True)
class HullWhiteSimulation:
    """
    Output container for Hull-White path simulation.

    Attributes
    ----------
    rate_paths : np.ndarray
        Simulated short rate paths, shape (n_paths, n_steps + 1).
    times : np.ndarray
        Time grid, shape (n_steps + 1,).
    params : HullWhiteParameters
        Hull-White parameters used in simulation.
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps.
    scheme : HullWhiteScheme
        Discretization scheme used.
    seed : int or None
        Random seed used.
    discount_factors : np.ndarray, optional
        Simulated discount factors exp(-∫r(s)ds), shape (n_paths,).
    """

    rate_paths: np.ndarray
    times: np.ndarray
    params: HullWhiteParameters
    n_paths: int
    n_steps: int
    scheme: HullWhiteScheme
    seed: Optional[int]
    discount_factors: Optional[np.ndarray] = None

    @property
    def terminal_rates(self) -> np.ndarray:
        """Terminal short rate values r(T)."""
        return self.rate_paths[:, -1]

    @property
    def maturity(self) -> float:
        """Time to maturity T."""
        return float(self.times[-1])

    @property
    def mean_terminal_rate(self) -> float:
        """Mean of terminal rates across paths."""
        return float(np.mean(self.terminal_rates))

    @property
    def std_terminal_rate(self) -> float:
        """Standard deviation of terminal rates across paths."""
        return float(np.std(self.terminal_rates))


# =============================================================================
# Hull-White Dynamics Simulator
# =============================================================================

@dataclass(frozen=True)
class HullWhiteDynamics:
    """
    Simulator for Hull-White one-factor short rate dynamics.

    Simulates the short rate process under the risk-neutral measure:
        dr(t) = [θ(t) - a·r(t)] dt + σ dW(t)

    For constant θ, this simplifies to an Ornstein-Uhlenbeck process:
        dr(t) = a·(θ - r(t)) dt + σ dW(t)

    Parameters
    ----------
    params : HullWhiteParameters
        Hull-White model parameters (a, σ, r₀, θ).

    Examples
    --------
    >>> from src.models.short_rate.hull_white import HullWhiteDynamics, HullWhiteParameters
    >>> params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03, theta=0.04)
    >>> dynamics = HullWhiteDynamics(params=params)
    >>> sim = dynamics.simulate(maturity=1.0, n_paths=10000, n_steps=252)
    >>> sim.mean_terminal_rate  # Should be close to expected_rate(1.0)
    """

    params: HullWhiteParameters

    def simulate(
        self,
        maturity: float,
        n_paths: int,
        n_steps: int,
        scheme: HullWhiteScheme = "exact",
        seed: Optional[int] = None,
        antithetic: bool = True,
        compute_discount_factors: bool = True,
    ) -> HullWhiteSimulation:
        """
        Simulate Hull-White short rate paths.

        Parameters
        ----------
        maturity : float
            Time to maturity T > 0.
        n_paths : int
            Number of paths to simulate.
        n_steps : int
            Number of time steps.
        scheme : HullWhiteScheme
            Discretization scheme:
            - "exact": Exact OU transition (recommended, preserves distribution)
            - "euler": Euler-Maruyama (simpler, may have discretization error)
        seed : int, optional
            Random seed for reproducibility.
        antithetic : bool
            Use antithetic variates for variance reduction.
        compute_discount_factors : bool
            If True, compute path-wise discount factors using trapezoidal rule.

        Returns
        -------
        HullWhiteSimulation
            Container with simulated paths and metadata.
        """
        # Validate inputs.
        if maturity <= 0.0:
            raise ValueError("maturity must be > 0.")
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0.")
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0.")

        # Use reproducible RNG from base models.
        rng = NormalRng(seed=seed)

        # Time discretization.
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity, n_steps + 1)

        # Generate standard normal increments using NormalRng.
        # NormalRng handles antithetic variates internally and rounds up to even count.
        Z = rng.standard_normals(n_paths, n_steps, antithetic=antithetic)
        n_actual = Z.shape[0]

        # Initialize rate paths.
        r = np.zeros((n_actual, n_steps + 1))
        r[:, 0] = self.params.r0

        # Extract parameters.
        a = self.params.a
        sigma = self.params.sigma
        theta = self.params.theta

        # Simulate paths.
        if scheme == "exact":
            # Exact OU transition: r(t+dt) | r(t) ~ N(μ, σ²)
            # μ = θ + (r(t) - θ)·exp(-a·dt)
            # σ² = (σ²/(2a))·(1 - exp(-2a·dt))
            exp_adt = np.exp(-a * dt)
            var_dt = (sigma ** 2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * dt))
            std_dt = np.sqrt(var_dt)

            for i in range(n_steps):
                r[:, i + 1] = theta + (r[:, i] - theta) * exp_adt + std_dt * Z[:, i]

        elif scheme == "euler":
            # Euler-Maruyama: r(t+dt) = r(t) + a(θ - r(t))dt + σ·√dt·Z
            for i in range(n_steps):
                r[:, i + 1] = r[:, i] + a * (theta - r[:, i]) * dt + sigma * sqrt_dt * Z[:, i]

        else:
            raise ValueError(f"Unknown scheme: {scheme}. Use 'exact' or 'euler'.")

        # Compute discount factors if requested.
        discount_factors = None
        if compute_discount_factors:
            # Trapezoidal rule for ∫r(s)ds.
            # integral ≈ dt/2 · (r_0 + 2·r_1 + 2·r_2 + ... + 2·r_{n-1} + r_n)
            integral = dt * (0.5 * r[:, 0] + np.sum(r[:, 1:-1], axis=1) + 0.5 * r[:, -1])
            discount_factors = np.exp(-integral)

        return HullWhiteSimulation(
            rate_paths=r,
            times=times,
            params=self.params,
            n_paths=n_actual,
            n_steps=n_steps,
            scheme=scheme,
            seed=seed,
            discount_factors=discount_factors,
        )


# =============================================================================
# Hull-White Analytic Functions (Pure Functions)
# =============================================================================

def hw_b_factor(a: float, tau: float) -> float:
    """
    Compute the B(t,T) factor in Hull-White bond pricing.
    
    B(t,T) = (1 - exp(-a·τ)) / a
    
    where τ = T - t.
    
    Parameters
    ----------
    a : float
        Mean reversion speed.
    tau : float
        Time to maturity (T - t).
        
    Returns
    -------
    float
        B factor value.
    """
    if abs(a) < 1e-10:
        # Limit as a → 0: B → τ.
        return tau
    return (1.0 - np.exp(-a * tau)) / a


def hw_zc_bond_price(
    r: float,
    t: float,
    T: float,
    a: float,
    sigma: float,
    P_0_t: float,
    P_0_T: float,
    f_0_t: float,
) -> float:
    """
    Compute zero-coupon bond price P(t,T) under Hull-White model.
    
    P(t,T) = A(t,T) · exp(-B(t,T) · r(t))
    
    where:
        B(t,T) = (1 - exp(-a(T-t))) / a
        A(t,T) = P(0,T)/P(0,t) · exp(B(t,T)·f(0,t) - σ²/(4a)·B(t,T)²·(1-exp(-2at)))
    
    Parameters
    ----------
    r : float
        Current short rate r(t).
    t : float
        Current time.
    T : float
        Bond maturity (T > t).
    a : float
        Mean reversion speed.
    sigma : float
        Short rate volatility.
    P_0_t : float
        Initial ZC bond price P(0,t).
    P_0_T : float
        Initial ZC bond price P(0,T).
    f_0_t : float
        Initial instantaneous forward rate f(0,t).
        
    Returns
    -------
    float
        Zero-coupon bond price P(t,T).
    """
    if T <= t:
        return 1.0 if T == t else P_0_T / P_0_t  # Edge case.

    tau = T - t
    B = hw_b_factor(a, tau)

    # A(t,T) computation.
    exp_2at = np.exp(-2.0 * a * t)
    log_A = np.log(P_0_T / P_0_t) + B * f_0_t - (sigma ** 2 / (4.0 * a)) * B ** 2 * (1.0 - exp_2at)

    return np.exp(log_A - B * r)


def hw_zc_bond_option_price(
    K: float,
    T_option: float,
    T_bond: float,
    a: float,
    sigma: float,
    P_0_S: float,
    P_0_T: float,
    option_type: str = "call",
) -> float:
    """
    Compute European option on zero-coupon bond under Hull-White.
    
    This is a closed-form solution for a European call/put on a ZC bond.
    
    Call: max(P(S,T) - K, 0)
    Put:  max(K - P(S,T), 0)
    
    where S = option expiry, T = bond maturity (T > S).
    
    Formulas
    --------
    Call = P(0,T)·N(h) - K·P(0,S)·N(h - σ_p)
    Put  = K·P(0,S)·N(-h + σ_p) - P(0,T)·N(-h)
    
    σ_p = σ · √((1-exp(-2aS))/(2a)) · B(S,T)
    h   = (1/σ_p) · ln(P(0,T)/(K·P(0,S))) + σ_p/2
    
    Parameters
    ----------
    K : float
        Strike price (bond price strike).
    T_option : float
        Option expiry time S.
    T_bond : float
        Underlying bond maturity T (must be > T_option).
    a : float
        Mean reversion speed.
    sigma : float
        Short rate volatility.
    P_0_S : float
        Initial ZC bond price P(0,S) at option expiry.
    P_0_T : float
        Initial ZC bond price P(0,T) at bond maturity.
    option_type : str
        "call" or "put".
        
    Returns
    -------
    float
        Option price.
        
    Raises
    ------
    ValueError
        If T_bond <= T_option or invalid option_type.
    """
    if T_bond <= T_option:
        raise ValueError("Bond maturity must be greater than option expiry.")

    if K <= 0:
        raise ValueError("Strike K must be > 0.")

    # B(S, T) factor.
    B_S_T = hw_b_factor(a, T_bond - T_option)

    # σ_p: volatility of forward bond price.
    if abs(a) < 1e-10:
        # Limit as a → 0.
        sigma_p = sigma * np.sqrt(T_option) * B_S_T
    else:
        sigma_p = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * T_option)) / (2.0 * a)) * B_S_T

    if sigma_p < 1e-12:
        # Zero vol case - intrinsic value.
        forward_bond = P_0_T / P_0_S
        if option_type.lower() == "call":
            return P_0_S * max(forward_bond - K, 0.0)
        elif option_type.lower() == "put":
            return P_0_S * max(K - forward_bond, 0.0)
        else:
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    # h parameter.
    h = (1.0 / sigma_p) * np.log(P_0_T / (K * P_0_S)) + sigma_p / 2.0

    # Option prices.
    if option_type.lower() == "call":
        return P_0_T * norm.cdf(h) - K * P_0_S * norm.cdf(h - sigma_p)
    elif option_type.lower() == "put":
        return K * P_0_S * norm.cdf(-h + sigma_p) - P_0_T * norm.cdf(-h)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")


def hw_caplet_price(
    K: float,
    T_reset: float,
    T_pay: float,
    tau: float,
    a: float,
    sigma: float,
    P_0_reset: float,
    P_0_pay: float,
    notional: float = 1.0,
) -> float:
    """
    Price a caplet under Hull-White using the ZC bond option formula.
    
    A caplet pays: N · τ · max(L(T_reset, T_pay) - K, 0) at T_pay
    
    This is equivalent to: (1 + τK) puts on ZC bond with strike 1/(1+τK).
    
    Parameters
    ----------
    K : float
        Cap strike rate.
    T_reset : float
        Reset (fixing) date.
    T_pay : float
        Payment date (T_pay > T_reset).
    tau : float
        Day count fraction for the period.
    a : float
        Mean reversion speed.
    sigma : float
        Short rate volatility.
    P_0_reset : float
        Initial ZC bond price P(0, T_reset).
    P_0_pay : float
        Initial ZC bond price P(0, T_pay).
    notional : float
        Notional amount.
        
    Returns
    -------
    float
        Caplet price.
    """
    # Strike for the equivalent bond put.
    K_bond = 1.0 / (1.0 + tau * K)

    # Caplet = (1 + τK) × put on ZC bond.
    put_price = hw_zc_bond_option_price(
        K=K_bond,
        T_option=T_reset,
        T_bond=T_pay,
        a=a,
        sigma=sigma,
        P_0_S=P_0_reset,
        P_0_T=P_0_pay,
        option_type="put",
    )

    return notional * (1.0 + tau * K) * put_price


def hw_floorlet_price(
    K: float,
    T_reset: float,
    T_pay: float,
    tau: float,
    a: float,
    sigma: float,
    P_0_reset: float,
    P_0_pay: float,
    notional: float = 1.0,
) -> float:
    """
    Price a floorlet under Hull-White using the ZC bond option formula.
    
    A floorlet pays: N · τ · max(K - L(T_reset, T_pay), 0) at T_pay
    
    This is equivalent to: (1 + τK) calls on ZC bond with strike 1/(1+τK).
    
    Parameters
    ----------
    K : float
        Floor strike rate.
    T_reset : float
        Reset (fixing) date.
    T_pay : float
        Payment date.
    tau : float
        Day count fraction.
    a : float
        Mean reversion speed.
    sigma : float
        Short rate volatility.
    P_0_reset : float
        Initial ZC bond price P(0, T_reset).
    P_0_pay : float
        Initial ZC bond price P(0, T_pay).
    notional : float
        Notional amount.
        
    Returns
    -------
    float
        Floorlet price.
    """
    K_bond = 1.0 / (1.0 + tau * K)

    call_price = hw_zc_bond_option_price(
        K=K_bond,
        T_option=T_reset,
        T_bond=T_pay,
        a=a,
        sigma=sigma,
        P_0_S=P_0_reset,
        P_0_T=P_0_pay,
        option_type="call",
    )

    return notional * (1.0 + tau * K) * call_price


def hw_swaption_price_jamshidian(
    K: float,
    T_option: float,
    swap_tenors: np.ndarray,
    swap_dcfs: np.ndarray,
    a: float,
    sigma: float,
    P_0: Callable[[float], float],
    notional: float = 1.0,
    is_payer: bool = True,
) -> float:
    """
    Price a European swaption under Hull-White using Jamshidian decomposition.
    
    Jamshidian's trick decomposes a swaption into a portfolio of bond options.
    
    For a payer swaption (option to pay fixed, receive floating):
        Value = Σᵢ cᵢ · Put(Kᵢ, T_option, Tᵢ)
    
    For a receiver swaption (option to receive fixed, pay floating):
        Value = Σᵢ cᵢ · Call(Kᵢ, T_option, Tᵢ)
    
    where cᵢ are the fixed leg cash flows and Kᵢ are the bond price strikes.
    
    Parameters
    ----------
    K : float
        Swap fixed rate (strike).
    T_option : float
        Swaption expiry.
    swap_tenors : np.ndarray
        Payment dates of the underlying swap [T_1, T_2, ..., T_n].
    swap_dcfs : np.ndarray
        Day count fractions for each period [τ_1, τ_2, ..., τ_n].
    a : float
        Mean reversion speed.
    sigma : float
        Short rate volatility.
    P_0 : Callable[[float], float]
        Function returning initial ZC bond price P(0, T) for any T.
    notional : float
        Swap notional.
    is_payer : bool
        True for payer swaption, False for receiver swaption.
        
    Returns
    -------
    float
        Swaption price.
    """
    n = len(swap_tenors)
    if len(swap_dcfs) != n:
        raise ValueError("swap_tenors and swap_dcfs must have same length.")

    # Cash flows: c_i = K * τ_i for i < n, c_n = 1 + K * τ_n.
    cash_flows = K * swap_dcfs.copy()
    cash_flows[-1] += 1.0

    # Find r* such that swap has zero value (sum of discounted cash flows = 1).
    # This requires numerical root finding in general.
    # For simplicity, we use the fact that at r*, P(T_option, T_i; r*) are known.
    
    P_0_S = P_0(T_option)
    
    # Compute forward bond prices F_i = P(0,T_i) / P(0,S).
    forward_bonds = np.array([P_0(T_i) / P_0_S for T_i in swap_tenors])

    # Strike prices for individual bond options.
    # At r*, the swap value is zero, so Σ c_i P(S,T_i;r*) = 1.
    # We need to solve for r* and then K_i = P(S,T_i;r*).
    
    # Simplified approach: use forward swap rate and approximate.
    # For exact Jamshidian, we'd solve for r* numerically.
    
    # Here we use a simpler direct decomposition approach.
    # The payer swaption can be written as:
    # max(Σ c_i P(S,T_i) - 1, 0) = max(Σ c_i P - 1, 0)
    
    # Using Jamshidian: find r* such that Σ c_i P(S,T_i;r*) = 1.
    from scipy.optimize import brentq

    def swap_value_at_r(r_star: float) -> float:
        """Swap value as function of short rate at option expiry."""
        total = 0.0
        for i, T_i in enumerate(swap_tenors):
            B_i = hw_b_factor(a, T_i - T_option)
            # Approximate P(S, T_i; r*) using simplified A factor.
            # A(S,T) ≈ P(0,T)/P(0,S) for flat initial curve.
            A_i = P_0(T_i) / P_0_S
            total += cash_flows[i] * A_i * np.exp(-B_i * r_star)
        return total - 1.0

    # Find r* in reasonable range.
    try:
        r_star = brentq(swap_value_at_r, -0.5, 0.5)
    except ValueError:
        # If root not found, use current rate as approximation.
        r_star = -np.log(P_0_S) / T_option if T_option > 0 else 0.0

    # Compute strike prices K_i = P(S, T_i; r*).
    K_strikes = np.zeros(n)
    for i, T_i in enumerate(swap_tenors):
        B_i = hw_b_factor(a, T_i - T_option)
        A_i = P_0(T_i) / P_0_S
        K_strikes[i] = A_i * np.exp(-B_i * r_star)

    # Sum of bond options.
    option_type = "put" if is_payer else "call"
    total_price = 0.0

    for i, T_i in enumerate(swap_tenors):
        bond_option_price = hw_zc_bond_option_price(
            K=K_strikes[i],
            T_option=T_option,
            T_bond=T_i,
            a=a,
            sigma=sigma,
            P_0_S=P_0_S,
            P_0_T=P_0(T_i),
            option_type=option_type,
        )
        total_price += cash_flows[i] * bond_option_price

    return notional * total_price
