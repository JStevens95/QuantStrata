"""
LIBOR Market Model (LMM) Implementation

The LIBOR Market Model (also known as BGM model after Brace-Gatarek-Musiela)
models the evolution of multiple forward LIBOR rates under the spot measure.

Model Dynamics (under terminal measure Q^N):
    dF_i(t) / F_i(t) = μ_i(t) dt + σ_i(t) · dW_i(t)

Where:
- F_i(t): Forward rate for period [T_i, T_{i+1}]
- σ_i(t): Instantaneous volatility of F_i
- W_i(t): Brownian motion with correlation ρ_ij between W_i and W_j
- μ_i(t): Drift term (measure-dependent for no-arbitrage)

Under the spot measure (rolling bank account numeraire):
    μ_i(t) = Σ_{j=β(t)}^{i} [ρ_ij · σ_i(t) · σ_j(t) · τ_j · F_j(t)] / [1 + τ_j · F_j(t)]

Where β(t) is the index of the first forward rate not yet fixed.

Author: QuantStrata
Phase: 3.8 - LIBOR Market Model
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from src.models.numeric.monte_carlo.rng import NormalRng


# =============================================================================
# Correlation Structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class LMMCorrelation:
    """
    Correlation structure for forward rates in the LMM.

    The correlation matrix ρ_ij determines the co-movement of forward rates.
    Several parametric forms are commonly used in practice.

    Parameters
    ----------
    n_forwards : int
        Number of forward rates.
    correlation_type : str
        Type of correlation structure:
        - "flat": Constant correlation between all pairs
        - "exponential": ρ_ij = exp(-β|i-j|)
        - "custom": User-provided correlation matrix
    flat_corr : float
        Constant correlation (for "flat" type). Default 0.5.
    beta : float
        Decay parameter for exponential correlation. Default 0.1.
    correlation_matrix : Optional[np.ndarray]
        Custom correlation matrix (n_forwards x n_forwards).

    Notes
    -----
    The exponential form ρ_ij = exp(-β|T_i - T_j|) is most common in practice,
    capturing the intuition that nearby forwards are more correlated.
    """

    n_forwards: int
    correlation_type: Literal["flat", "exponential", "custom"] = "exponential"
    flat_corr: float = 0.5
    beta: float = 0.1
    correlation_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.n_forwards < 1:
            raise ValueError("n_forwards must be >= 1")

        if self.correlation_type == "flat":
            if not -1 <= self.flat_corr <= 1:
                raise ValueError("flat_corr must be in [-1, 1]")

        if self.correlation_type == "exponential":
            if self.beta < 0:
                raise ValueError("beta must be non-negative")

        if self.correlation_type == "custom":
            if self.correlation_matrix is None:
                raise ValueError("correlation_matrix required for custom type")
            if self.correlation_matrix.shape != (self.n_forwards, self.n_forwards):
                raise ValueError(
                    f"correlation_matrix shape {self.correlation_matrix.shape} "
                    f"doesn't match ({self.n_forwards}, {self.n_forwards})"
                )

    def get_correlation_matrix(self, tenors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the correlation matrix.

        Parameters
        ----------
        tenors : Optional[np.ndarray]
            Forward rate fixing times (for exponential correlation).
            If None, uses indices [0, 1, 2, ...] as proxies.

        Returns
        -------
        np.ndarray
            Correlation matrix of shape (n_forwards, n_forwards).
        """
        n = self.n_forwards

        if self.correlation_type == "flat":
            corr = np.full((n, n), self.flat_corr)
            np.fill_diagonal(corr, 1.0)
            return corr

        elif self.correlation_type == "exponential":
            if tenors is None:
                tenors = np.arange(n, dtype=float)
            corr = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    corr[i, j] = np.exp(-self.beta * abs(tenors[i] - tenors[j]))
            return corr

        else:  # custom
            return self.correlation_matrix.copy()

    def get_cholesky(self, tenors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the Cholesky decomposition of the correlation matrix.

        Returns L such that L @ L.T = correlation_matrix.

        Parameters
        ----------
        tenors : Optional[np.ndarray]
            Forward rate fixing times.

        Returns
        -------
        np.ndarray
            Lower triangular Cholesky factor.
        """
        corr = self.get_correlation_matrix(tenors)
        return np.linalg.cholesky(corr)


# =============================================================================
# LMM Parameters
# =============================================================================


@dataclass(frozen=True, slots=True)
class LMMParameters:
    """
    Parameters for the LIBOR Market Model.

    Parameters
    ----------
    tenors : np.ndarray
        Forward rate fixing times T_0, T_1, ..., T_{n-1} (years).
        These define the periods [T_i, T_{i+1}] for each forward.
    accrual_factors : np.ndarray
        Day count fractions τ_i for each forward period.
        Usually τ_i ≈ T_{i+1} - T_i.
    initial_forwards : np.ndarray
        Initial forward rates F_i(0) for each period.
    volatilities : np.ndarray
        Instantaneous volatilities σ_i for each forward rate.
        Can be scalar (flat) or time-dependent.
    correlation : LMMCorrelation
        Correlation structure between forward rates.

    Notes
    -----
    For a standard LMM with n forwards:
    - tenors has n+1 elements: T_0, T_1, ..., T_n
    - accrual_factors has n elements: τ_0, ..., τ_{n-1}
    - initial_forwards has n elements: F_0(0), ..., F_{n-1}(0)
    - volatilities has n elements: σ_0, ..., σ_{n-1}
    """

    tenors: np.ndarray
    accrual_factors: np.ndarray
    initial_forwards: np.ndarray
    volatilities: np.ndarray
    correlation: LMMCorrelation

    def __post_init__(self) -> None:
        """Validate parameters."""
        # Convert to numpy arrays
        tenors = np.asarray(self.tenors, dtype=float)
        tau = np.asarray(self.accrual_factors, dtype=float)
        f0 = np.asarray(self.initial_forwards, dtype=float)
        vol = np.asarray(self.volatilities, dtype=float)

        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "accrual_factors", tau)
        object.__setattr__(self, "initial_forwards", f0)
        object.__setattr__(self, "volatilities", vol)

        n = len(f0)

        # Validate dimensions
        if len(tenors) != n + 1:
            raise ValueError(f"tenors should have {n + 1} elements, got {len(tenors)}")
        if len(tau) != n:
            raise ValueError(f"accrual_factors should have {n} elements, got {len(tau)}")
        if len(vol) != n:
            raise ValueError(f"volatilities should have {n} elements, got {len(vol)}")
        if self.correlation.n_forwards != n:
            raise ValueError(
                f"correlation.n_forwards={self.correlation.n_forwards} != {n} forwards"
            )

        # Validate values
        if not np.all(np.diff(tenors) > 0):
            raise ValueError("tenors must be strictly increasing")
        if np.any(tau <= 0):
            raise ValueError("accrual_factors must be positive")
        if np.any(f0 <= 0):
            raise ValueError("initial_forwards must be positive")
        if np.any(vol <= 0):
            raise ValueError("volatilities must be positive")

    @property
    def n_forwards(self) -> int:
        """Number of forward rates."""
        return len(self.initial_forwards)

    @property
    def terminal_time(self) -> float:
        """Terminal time T_n."""
        return float(self.tenors[-1])


# =============================================================================
# LMM Simulation Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class LMMSimulation:
    """
    Result of LMM Monte Carlo simulation.

    Attributes
    ----------
    forwards : np.ndarray
        Simulated forward rates of shape (n_paths, n_forwards, n_steps+1).
        forwards[p, i, k] = F_i(t_k) for path p.
    time_grid : np.ndarray
        Time points t_0, t_1, ..., t_K.
    numeraire : np.ndarray
        Numeraire (rolling bank account) values of shape (n_paths, n_steps+1).
    discount_factors : np.ndarray
        Discount factors P(0, T_i) computed from initial forwards.
    """

    forwards: np.ndarray
    time_grid: np.ndarray
    numeraire: np.ndarray
    discount_factors: np.ndarray


# =============================================================================
# LMM Dynamics
# =============================================================================


class LMMDynamics:
    """
    LIBOR Market Model dynamics and Monte Carlo simulation.

    This class implements the log-Euler discretization of the LMM under
    the spot measure (rolling bank account numeraire).

    The drift correction ensures no-arbitrage:
        μ_i(t) = Σ_{j=β(t)}^{i} [ρ_ij · σ_i · σ_j · τ_j · F_j(t)] / [1 + τ_j · F_j(t)]

    The discretization uses:
        F_i(t+Δt) = F_i(t) · exp[(μ_i - σ_i²/2)Δt + σ_i·√Δt·Z_i]

    Where Z = L·ξ with L being the Cholesky factor of the correlation matrix
    and ξ being independent standard normals.

    Parameters
    ----------
    params : LMMParameters
        Model parameters.
    """

    def __init__(self, params: LMMParameters) -> None:
        self.params = params
        self._cholesky = params.correlation.get_cholesky(params.tenors[:-1])

    def simulate(
        self,
        n_paths: int,
        n_steps_per_period: int = 10,
        seed: Optional[int] = None,
        antithetic: bool = False,
    ) -> LMMSimulation:
        """
        Simulate forward rate paths using log-Euler discretization.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths.
        n_steps_per_period : int
            Number of time steps per forward period. Default 10.
        seed : Optional[int]
            Random seed for reproducibility.
        antithetic : bool
            Use antithetic variates. If True, generates n_paths/2 paths
            and their antithetic pairs.

        Returns
        -------
        LMMSimulation
            Simulation results including forward rates and numeraire.
        """
        params = self.params
        n = params.n_forwards
        tenors = params.tenors
        tau = params.accrual_factors
        f0 = params.initial_forwards
        vol = params.volatilities
        L = self._cholesky

        # Build time grid
        time_points = [0.0]
        for i in range(n):
            t_start = tenors[i]
            t_end = tenors[i + 1]
            steps = np.linspace(t_start, t_end, n_steps_per_period + 1)[1:]
            time_points.extend(steps)
        time_grid = np.array(time_points)
        n_steps = len(time_grid) - 1

        # Adjust paths for antithetic
        if antithetic:
            actual_paths = n_paths // 2
        else:
            actual_paths = n_paths

        # Initialize RNG
        rng = NormalRng(seed=seed)

        # Generate random numbers: (n_paths, n_steps, n_forwards)
        # standard_normals takes (n_samples, dimension), so generate flat and reshape
        total_samples = actual_paths * n_steps
        Z_flat = rng.standard_normals(total_samples, d=n, antithetic=False)
        Z_raw = Z_flat.reshape(actual_paths, n_steps, n)

        if antithetic:
            Z_raw = np.concatenate([Z_raw, -Z_raw], axis=0)
            actual_paths = n_paths

        # Correlate the random numbers using Cholesky
        # Z_corr[p, k, :] = L @ Z_raw[p, k, :]
        Z_corr = np.einsum("ij,pkj->pki", L, Z_raw)

        # Initialize forwards: (n_paths, n_forwards, n_steps+1)
        forwards = np.zeros((actual_paths, n, n_steps + 1))
        forwards[:, :, 0] = f0

        # Initialize numeraire (rolling bank account)
        numeraire = np.ones((actual_paths, n_steps + 1))

        # Track which forwards are still alive (not fixed)
        # beta[k] = index of first forward not yet fixed at time t_k
        beta = np.zeros(n_steps + 1, dtype=int)
        for k, t in enumerate(time_grid):
            # Find first i such that T_i > t
            idx = np.searchsorted(tenors[:-1], t, side="right")
            beta[k] = min(idx, n - 1)

        # Simulate
        for k in range(n_steps):
            t_k = time_grid[k]
            dt = time_grid[k + 1] - t_k
            sqrt_dt = np.sqrt(dt)
            b = beta[k]  # First alive forward

            F_k = forwards[:, :, k]  # (n_paths, n_forwards)

            # Compute drift for each alive forward
            # μ_i = Σ_{j=b}^{i} [ρ_ij · σ_i · σ_j · τ_j · F_j] / [1 + τ_j · F_j]
            drift = np.zeros((actual_paths, n))

            for i in range(b, n):
                for j in range(b, i + 1):
                    rho_ij = L[i, :] @ L[j, :]  # Correlation from Cholesky
                    term = (rho_ij * vol[i] * vol[j] * tau[j] * F_k[:, j]) / (
                        1.0 + tau[j] * F_k[:, j]
                    )
                    drift[:, i] += term

            # Log-Euler step for alive forwards
            for i in range(b, n):
                log_F = np.log(F_k[:, i])
                log_F_new = (
                    log_F
                    + (drift[:, i] - 0.5 * vol[i] ** 2) * dt
                    + vol[i] * sqrt_dt * Z_corr[:, k, i]
                )
                forwards[:, i, k + 1] = np.exp(log_F_new)

            # Copy fixed forwards
            forwards[:, :b, k + 1] = forwards[:, :b, k]

            # Update numeraire
            # B(t+dt) = B(t) * (1 + τ_{β(t)} * F_{β(t)}) if crossing period boundary
            if k + 1 < n_steps and beta[k + 1] > b:
                # Crossed a fixing date
                numeraire[:, k + 1] = numeraire[:, k] * (1.0 + tau[b] * F_k[:, b])
            else:
                # Simple compounding approximation within period
                numeraire[:, k + 1] = numeraire[:, k] * np.exp(
                    F_k[:, b] * dt / (1.0 + tau[b] * F_k[:, b])
                )

        # Compute initial discount factors from forward rates
        discount_factors = np.ones(n + 1)
        for i in range(n):
            discount_factors[i + 1] = discount_factors[i] / (1.0 + tau[i] * f0[i])

        return LMMSimulation(
            forwards=forwards,
            time_grid=time_grid,
            numeraire=numeraire,
            discount_factors=discount_factors,
        )

    def price_caplet(
        self,
        fixing_index: int,
        strike: float,
        n_paths: int = 100_000,
        seed: Optional[int] = None,
    ) -> float:
        """
        Price a caplet using Monte Carlo.

        The caplet pays max(F_i(T_i) - K, 0) * τ_i at time T_{i+1}.

        Parameters
        ----------
        fixing_index : int
            Index i of the forward rate.
        strike : float
            Caplet strike rate K.
        n_paths : int
            Number of MC paths.
        seed : Optional[int]
            Random seed.

        Returns
        -------
        float
            Caplet price (as a fraction of notional).
        """
        sim = self.simulate(n_paths=n_paths, seed=seed, antithetic=True)

        i = fixing_index
        tau_i = self.params.accrual_factors[i]

        # Find time index for T_i (fixing time)
        T_i = self.params.tenors[i]
        t_idx = np.searchsorted(sim.time_grid, T_i)
        t_idx = min(t_idx, len(sim.time_grid) - 1)

        # Get F_i at fixing time
        F_i_fixing = sim.forwards[:, i, t_idx]

        # Caplet payoff at T_{i+1}
        payoff = np.maximum(F_i_fixing - strike, 0.0) * tau_i

        # Discount to t=0
        df = sim.discount_factors[i + 1]
        price = df * np.mean(payoff)

        return float(price)

    def price_floorlet(
        self,
        fixing_index: int,
        strike: float,
        n_paths: int = 100_000,
        seed: Optional[int] = None,
    ) -> float:
        """
        Price a floorlet using Monte Carlo.

        The floorlet pays max(K - F_i(T_i), 0) * τ_i at time T_{i+1}.

        Parameters
        ----------
        fixing_index : int
            Index i of the forward rate.
        strike : float
            Floorlet strike rate K.
        n_paths : int
            Number of MC paths.
        seed : Optional[int]
            Random seed.

        Returns
        -------
        float
            Floorlet price (as a fraction of notional).
        """
        sim = self.simulate(n_paths=n_paths, seed=seed, antithetic=True)

        i = fixing_index
        tau_i = self.params.accrual_factors[i]

        T_i = self.params.tenors[i]
        t_idx = np.searchsorted(sim.time_grid, T_i)
        t_idx = min(t_idx, len(sim.time_grid) - 1)

        F_i_fixing = sim.forwards[:, i, t_idx]
        payoff = np.maximum(strike - F_i_fixing, 0.0) * tau_i

        df = sim.discount_factors[i + 1]
        price = df * np.mean(payoff)

        return float(price)

    def price_swaption(
        self,
        start_index: int,
        end_index: int,
        strike: float,
        is_payer: bool = True,
        n_paths: int = 100_000,
        seed: Optional[int] = None,
    ) -> float:
        """
        Price a European swaption using Monte Carlo.

        A payer swaption gives the right to enter a swap paying fixed.
        The exercise payoff at T_{start} is:
            max(S - K, 0) * A  for payer
            max(K - S, 0) * A  for receiver

        Where S is the forward swap rate and A is the annuity.

        Parameters
        ----------
        start_index : int
            Index of the first forward (swap start).
        end_index : int
            Index of the last forward + 1 (swap end).
        strike : float
            Swaption strike rate K.
        is_payer : bool
            True for payer swaption, False for receiver.
        n_paths : int
            Number of MC paths.
        seed : Optional[int]
            Random seed.

        Returns
        -------
        float
            Swaption price (as a fraction of notional).
        """
        sim = self.simulate(n_paths=n_paths, seed=seed, antithetic=True)

        # Find time index for exercise (T_{start_index})
        T_exercise = self.params.tenors[start_index]
        t_idx = np.searchsorted(sim.time_grid, T_exercise)
        t_idx = min(t_idx, len(sim.time_grid) - 1)

        # Get forward rates at exercise
        F_exercise = sim.forwards[:, :, t_idx]  # (n_paths, n_forwards)

        # Compute forward swap rate and annuity at exercise
        tau = self.params.accrual_factors

        # Discount factors relative to T_{start_index}
        # P(T_start, T_j) = Π_{k=start}^{j-1} 1/(1 + τ_k * F_k)
        n_paths_actual = F_exercise.shape[0]
        df_rel = np.ones((n_paths_actual, end_index - start_index + 1))
        for j in range(start_index, end_index):
            df_rel[:, j - start_index + 1] = (
                df_rel[:, j - start_index] / (1.0 + tau[j] * F_exercise[:, j])
            )

        # Annuity: A = Σ_{j=start}^{end-1} τ_j * P(T_start, T_{j+1})
        annuity = np.zeros(n_paths_actual)
        for j in range(start_index, end_index):
            annuity += tau[j] * df_rel[:, j - start_index + 1]

        # Forward swap rate: S = (1 - P(T_start, T_end)) / A
        swap_rate = (1.0 - df_rel[:, -1]) / annuity

        # Swaption payoff
        if is_payer:
            payoff = np.maximum(swap_rate - strike, 0.0) * annuity
        else:
            payoff = np.maximum(strike - swap_rate, 0.0) * annuity

        # Discount to t=0
        df_to_exercise = sim.discount_factors[start_index]
        price = df_to_exercise * np.mean(payoff)

        return float(price)
