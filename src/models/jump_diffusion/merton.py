"""
Merton Jump-Diffusion Model Implementation.

This module provides the Merton (1976) jump-diffusion model for pricing
derivatives where the underlying experiences both continuous diffusion
and discrete jumps.

Mathematical Framework
----------------------
Under the risk-neutral measure, the asset price follows:

    dS_t / S_t = (r - q - λκ) dt + σ dW_t + (J - 1) dN_t

where:
    - r: Risk-free rate
    - q: Dividend/foreign rate
    - σ: Diffusion volatility
    - W_t: Standard Brownian motion
    - N_t: Poisson process with intensity λ
    - J: Jump multiplier, J = exp(Y) where Y ~ N(μ_J, σ_J²)
    - κ = E[J - 1] = exp(μ_J + σ_J²/2) - 1

Key Properties
--------------
1. **Fat tails**: Jumps generate heavier tails than pure GBM
2. **Implied vol smile**: Creates steep short-term smiles
3. **Semi-closed form**: European options via Fourier methods
4. **Jump clustering**: Can model market crashes

Parameters
----------
- σ (sigma): Diffusion volatility (continuous component)
- λ (lambda_): Jump intensity (expected jumps per year)
- μ_J (mu_j): Mean of log-jump size
- σ_J (sigma_j): Std dev of log-jump size

References
----------
- Merton, R.C. (1976). "Option pricing when underlying stock returns are
  discontinuous." Journal of Financial Economics.
- Cont, R. & Tankov, P. (2004). Financial Modelling with Jump Processes.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.models.numeric.monte_carlo.rng import NormalRng


# =============================================================================
# Merton Parameters
# =============================================================================

@dataclass(frozen=True, slots=True)
class MertonParameters:
    """
    Parameters for the Merton jump-diffusion model.

    The Merton model specifies:
        dS_t / S_t = (μ - λκ) dt + σ dW_t + (J - 1) dN_t

    where J = exp(Y), Y ~ N(μ_J, σ_J²), and κ = E[J - 1].

    Parameters
    ----------
    sigma : float
        Diffusion volatility σ > 0.
    lambda_ : float
        Jump intensity λ ≥ 0 (expected number of jumps per year).
        Use lambda_ to avoid conflict with Python keyword.
    mu_j : float
        Mean of log-jump size μ_J (can be negative for crash-like jumps).
    sigma_j : float
        Standard deviation of log-jump size σ_J ≥ 0.

    Attributes
    ----------
    expected_jump : float
        Expected jump multiplier E[J] = exp(μ_J + σ_J²/2).
    kappa : float
        Expected relative jump κ = E[J - 1] = E[J] - 1.
    total_variance : float
        Total instantaneous variance (diffusion + jump contribution).

    Examples
    --------
    >>> params = MertonParameters(
    ...     sigma=0.2,      # 20% diffusion vol
    ...     lambda_=0.5,    # 0.5 jumps/year expected
    ...     mu_j=-0.1,      # Negative mean (crash-like jumps)
    ...     sigma_j=0.2,    # 20% jump size uncertainty
    ... )
    >>> params.kappa  # Expected relative jump
    -0.08...
    >>> params.expected_jump
    0.91...
    """

    sigma: float       # Diffusion volatility σ
    lambda_: float     # Jump intensity λ (jumps per year)
    mu_j: float        # Mean of log-jump
    sigma_j: float     # Std dev of log-jump

    def __post_init__(self) -> None:
        """Validate Merton parameters."""
        # Validate sigma (diffusion volatility)
        if not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite.")
        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0.")

        # Validate lambda_ (jump intensity)
        if not np.isfinite(self.lambda_):
            raise ValueError("lambda_ must be finite.")
        if self.lambda_ < 0.0:
            raise ValueError("lambda_ must be >= 0.")

        # Validate mu_j (mean of log-jump)
        if not np.isfinite(self.mu_j):
            raise ValueError("mu_j must be finite.")

        # Validate sigma_j (std dev of log-jump)
        if not np.isfinite(self.sigma_j):
            raise ValueError("sigma_j must be finite.")
        if self.sigma_j < 0.0:
            raise ValueError("sigma_j must be >= 0.")

    @property
    def expected_jump(self) -> float:
        """
        Expected jump multiplier E[J].

        Since J = exp(Y) with Y ~ N(μ_J, σ_J²):
            E[J] = exp(μ_J + σ_J²/2)
        """
        return np.exp(self.mu_j + 0.5 * self.sigma_j**2)

    @property
    def kappa(self) -> float:
        """
        Expected relative jump κ = E[J - 1].

        This is the drift adjustment for the risk-neutral measure.
        """
        return self.expected_jump - 1.0

    @property
    def jump_variance(self) -> float:
        """
        Variance of single jump multiplier Var[J].

        Var[J] = E[J²] - E[J]² = exp(2μ_J + 2σ_J²) - exp(2μ_J + σ_J²)
        """
        ej2 = np.exp(2 * self.mu_j + 2 * self.sigma_j**2)
        ej = self.expected_jump
        return ej2 - ej**2

    @property
    def total_variance_rate(self) -> float:
        """
        Total variance rate σ² + λ E[J²] - λ(E[J])².

        This represents the instantaneous variance contribution from
        both diffusion and jump components.
        """
        # Diffusion variance
        diff_var = self.sigma**2
        # Jump variance contribution: λ × Var[J] + λ × (E[J-1])²
        # = λ × (E[J²] - 2E[J] + 1)
        ej2 = np.exp(2 * self.mu_j + 2 * self.sigma_j**2)
        jump_var = self.lambda_ * (ej2 - 2 * self.expected_jump + 1)
        return diff_var + jump_var

    @property
    def equivalent_bs_vol(self) -> float:
        """
        Approximate equivalent Black-Scholes volatility.

        For short maturities, this gives a rough approximation.
        """
        return np.sqrt(self.total_variance_rate)

    def expected_num_jumps(self, T: float) -> float:
        """Expected number of jumps in time interval [0, T]."""
        return self.lambda_ * T


# =============================================================================
# Merton Simulation Output
# =============================================================================

@dataclass(frozen=True, slots=True)
class MertonSimulation:
    """
    Output container for Merton jump-diffusion path simulation.

    Attributes
    ----------
    spot_paths : np.ndarray
        Simulated spot paths, shape (n_paths, n_steps + 1).
    jump_counts : np.ndarray
        Cumulative jump counts per path, shape (n_paths, n_steps + 1).
    times : np.ndarray
        Time grid, shape (n_steps + 1,).
    params : MertonParameters
        Merton parameters used in simulation.
    drift : float
        Drift rate used.
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps.
    seed : int or None
        Random seed used.
    """

    spot_paths: np.ndarray
    jump_counts: np.ndarray
    times: np.ndarray
    params: MertonParameters
    drift: float
    n_paths: int
    n_steps: int
    seed: Optional[int]

    @property
    def terminal_spots(self) -> np.ndarray:
        """Terminal spot values S_T."""
        return self.spot_paths[:, -1]

    @property
    def total_jumps(self) -> np.ndarray:
        """Total number of jumps per path."""
        return self.jump_counts[:, -1]

    @property
    def maturity(self) -> float:
        """Time to maturity T."""
        return float(self.times[-1])

    @property
    def average_jumps_per_path(self) -> float:
        """Average number of jumps across all paths."""
        return float(np.mean(self.total_jumps))

    @property
    def paths_with_jumps(self) -> int:
        """Number of paths that experienced at least one jump."""
        return int(np.sum(self.total_jumps > 0))

    @property
    def jump_fraction(self) -> float:
        """Fraction of paths that experienced at least one jump."""
        return self.paths_with_jumps / self.n_paths


# =============================================================================
# Merton Dynamics Simulator
# =============================================================================

@dataclass(frozen=True, slots=True)
class MertonDynamics:
    """
    Simulator for Merton jump-diffusion dynamics.

    Simulates the process under the risk-neutral measure:
        dS_t / S_t = (μ - λκ) dt + σ dW_t + (J - 1) dN_t

    where:
        - μ = r - q (risk-neutral drift)
        - κ = E[J - 1] (expected relative jump)
        - J = exp(Y), Y ~ N(μ_J, σ_J²)

    Parameters
    ----------
    params : MertonParameters
        Merton model parameters (σ, λ, μ_J, σ_J).
    drift : float
        Drift coefficient μ = r - q (risk-neutral drift).

    Examples
    --------
    >>> from src.models.jump_diffusion import MertonDynamics, MertonParameters
    >>> params = MertonParameters(sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2)
    >>> dynamics = MertonDynamics(params=params, drift=0.03)
    >>> sim = dynamics.simulate(
    ...     spot0=100.0, maturity=1.0, n_paths=10000, n_steps=252
    ... )
    >>> sim.terminal_spots.mean()  # Around 100 * exp((0.03 - λκ)*T)
    """

    params: MertonParameters
    drift: float  # μ = r - q

    def simulate(
        self,
        spot0: float,
        maturity: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
        antithetic: bool = True,
    ) -> MertonSimulation:
        """
        Simulate Merton jump-diffusion paths.

        Uses exact simulation for the diffusion component combined with
        Poisson-distributed jump times and log-normal jump sizes.

        Parameters
        ----------
        spot0 : float
            Initial spot price S_0 > 0.
        maturity : float
            Time to maturity T > 0.
        n_paths : int
            Number of paths to simulate.
        n_steps : int
            Number of time steps.
        seed : int, optional
            Random seed for reproducibility.
        antithetic : bool
            Use antithetic variates for variance reduction.
            Only applies to diffusion component.

        Returns
        -------
        MertonSimulation
            Container with simulated paths and metadata.
        """
        # Validate inputs
        if spot0 <= 0.0:
            raise ValueError("spot0 must be > 0.")
        if maturity <= 0.0:
            raise ValueError("maturity must be > 0.")
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0.")
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0.")

        # Initialize RNG
        rng = NormalRng(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Time discretization
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        times = np.linspace(0.0, maturity, n_steps + 1)

        # Handle antithetic variates for diffusion
        if antithetic:
            n_base = (n_paths + 1) // 2
            n_actual = 2 * n_base
        else:
            n_base = n_paths
            n_actual = n_paths

        # Extract parameters
        sigma = self.params.sigma
        lambda_ = self.params.lambda_
        mu_j = self.params.mu_j
        sigma_j = self.params.sigma_j
        kappa = self.params.kappa

        # Adjusted drift: μ - λκ (compensates for expected jump contribution)
        adjusted_drift = self.drift - lambda_ * kappa

        # Generate diffusion increments
        Z_diff = rng.standard_normals(n_base * n_steps, 1).reshape(n_base, n_steps)
        dW_base = sqrt_dt * Z_diff

        if antithetic:
            dW = np.vstack([dW_base, -dW_base])
        else:
            dW = dW_base

        # Initialize path arrays
        S = np.zeros((n_actual, n_steps + 1))
        jump_counts = np.zeros((n_actual, n_steps + 1), dtype=np.int32)
        S[:, 0] = spot0

        # Simulate paths step by step
        for i in range(n_steps):
            S_curr = S[:, i]

            # 1. Diffusion step (log-Euler for positivity)
            log_diffusion = (adjusted_drift - 0.5 * sigma**2) * dt + sigma * dW[:, i]

            # 2. Jump component
            # Number of jumps in this time step ~ Poisson(λ dt)
            n_jumps = np.random.poisson(lambda_ * dt, n_actual)

            # Log-jump sizes: sum of n_jumps log-normal variates
            # For efficiency, compute total log-jump for each path
            log_jump = np.zeros(n_actual)
            for p in range(n_actual):
                if n_jumps[p] > 0:
                    # Each jump has size J = exp(Y), Y ~ N(μ_J, σ_J²)
                    # Product of n_jumps such jumps has log = sum of n_jumps Y's
                    # Sum of n_jumps N(μ_J, σ_J²) is N(n * μ_J, n * σ_J²)
                    jump_log_mean = n_jumps[p] * mu_j
                    jump_log_std = np.sqrt(n_jumps[p]) * sigma_j
                    log_jump[p] = np.random.normal(jump_log_mean, jump_log_std)

            # 3. Combined step
            S_next = S_curr * np.exp(log_diffusion + log_jump)

            S[:, i + 1] = S_next
            jump_counts[:, i + 1] = jump_counts[:, i] + n_jumps

        return MertonSimulation(
            spot_paths=S,
            jump_counts=jump_counts,
            times=times,
            params=self.params,
            drift=self.drift,
            n_paths=n_actual,
            n_steps=n_steps,
            seed=seed,
        )

    def simulate_exact(
        self,
        spot0: float,
        maturity: float,
        n_paths: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Exact simulation of terminal spot S_T (single time step).

        For European option pricing, we can simulate S_T directly without
        intermediate time steps, which is more efficient.

        The exact distribution of S_T is:
            S_T = S_0 × exp((μ - λκ - σ²/2)T + σ√T Z) × ∏_{i=1}^{N_T} J_i

        where N_T ~ Poisson(λT) and J_i = exp(Y_i), Y_i ~ N(μ_J, σ_J²).

        Parameters
        ----------
        spot0 : float
            Initial spot price.
        maturity : float
            Time to maturity T.
        n_paths : int
            Number of samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Terminal spot values S_T, shape (n_paths,).
        """
        if spot0 <= 0.0:
            raise ValueError("spot0 must be > 0.")
        if maturity <= 0.0:
            raise ValueError("maturity must be > 0.")

        if seed is not None:
            np.random.seed(seed)

        # Extract parameters
        sigma = self.params.sigma
        lambda_ = self.params.lambda_
        mu_j = self.params.mu_j
        sigma_j = self.params.sigma_j
        kappa = self.params.kappa

        # Adjusted drift
        adjusted_drift = self.drift - lambda_ * kappa

        # Diffusion component
        Z = np.random.standard_normal(n_paths)
        log_diffusion = (adjusted_drift - 0.5 * sigma**2) * maturity + sigma * np.sqrt(maturity) * Z

        # Jump component
        n_jumps = np.random.poisson(lambda_ * maturity, n_paths)
        log_jump = np.zeros(n_paths)

        # For paths with jumps, compute total log-jump
        mask = n_jumps > 0
        if np.any(mask):
            jump_log_mean = n_jumps[mask] * mu_j
            jump_log_std = np.sqrt(n_jumps[mask]) * sigma_j
            log_jump[mask] = np.random.normal(jump_log_mean, jump_log_std)

        # Terminal spot
        S_T = spot0 * np.exp(log_diffusion + log_jump)

        return S_T


# =============================================================================
# Merton European Option Pricing (Semi-Closed Form)
# =============================================================================

def merton_european_call(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lambda_: float,
    mu_j: float,
    sigma_j: float,
    n_terms: int = 50,
) -> float:
    """
    Merton (1976) European call option price via infinite series.

    The price is a weighted sum of Black-Scholes prices:
        C = Σ_{n=0}^{∞} P(N_T = n) × C_BS(S, K, T, r_n, q, σ_n)

    where:
        - P(N_T = n) = exp(-λ'T) (λ'T)^n / n!  (modified Poisson)
        - λ' = λ × E[J] = λ × exp(μ_J + σ_J²/2)
        - r_n = r - λκ + n × log(E[J]) / T
        - σ_n² = σ² + n × σ_J² / T

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend/foreign rate.
    sigma : float
        Diffusion volatility.
    lambda_ : float
        Jump intensity.
    mu_j : float
        Mean of log-jump.
    sigma_j : float
        Std dev of log-jump.
    n_terms : int
        Number of terms in series (default 50).

    Returns
    -------
    float
        European call option price.
    """
    from scipy.stats import norm

    # Expected jump multiplier
    E_J = np.exp(mu_j + 0.5 * sigma_j**2)
    kappa = E_J - 1.0

    # Modified intensity λ' = λ × E[J]
    lambda_prime = lambda_ * E_J

    # Initialize price
    price = 0.0

    for n in range(n_terms):
        # Poisson probability P(N_T = n) with intensity λ'T
        poisson_prob = np.exp(-lambda_prime * T) * (lambda_prime * T)**n / math.factorial(n)

        if poisson_prob < 1e-15:
            break  # Remaining terms negligible

        # Adjusted parameters for n jumps
        r_n = r - lambda_ * kappa + n * np.log(E_J) / T
        sigma_n_sq = sigma**2 + n * sigma_j**2 / T
        sigma_n = np.sqrt(sigma_n_sq)

        # Black-Scholes call with adjusted parameters
        d1 = (np.log(S / K) + (r_n - q + 0.5 * sigma_n_sq) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)

        bs_call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2)

        price += poisson_prob * bs_call

    return price


def merton_european_put(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lambda_: float,
    mu_j: float,
    sigma_j: float,
    n_terms: int = 50,
) -> float:
    """
    Merton (1976) European put option price via put-call parity.

    P = C - S × exp(-qT) + K × exp(-rT)

    See `merton_european_call` for parameter descriptions.
    """
    call = merton_european_call(S, K, T, r, q, sigma, lambda_, mu_j, sigma_j, n_terms)
    put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
    return put


def merton_implied_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lambda_: float,
    mu_j: float,
    sigma_j: float,
) -> float:
    """
    Compute Black-Scholes implied volatility from Merton price.

    Parameters
    ----------
    S, K, T, r, q : float
        Standard option parameters.
    sigma, lambda_, mu_j, sigma_j : float
        Merton parameters.

    Returns
    -------
    float
        Black-Scholes implied volatility.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm

    # Compute Merton price
    merton_price = merton_european_call(S, K, T, r, q, sigma, lambda_, mu_j, sigma_j)

    # BS call formula
    def bs_call(vol):
        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Find implied vol
    def objective(vol):
        return bs_call(vol) - merton_price

    try:
        impl_vol = brentq(objective, 0.001, 5.0)
    except ValueError:
        # Price outside bounds
        impl_vol = np.nan

    return impl_vol
