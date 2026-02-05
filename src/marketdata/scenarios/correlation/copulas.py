"""
Copula Models for Dependence Structures.

Provides various copula families for modeling multivariate dependence:

1. **Gaussian Copula**
   - Implied by multivariate normal distribution
   - No tail dependence
   - Easy to parameterize with correlation matrix

2. **Student-t Copula**
   - Symmetric tail dependence
   - Extra parameter: degrees of freedom
   - Lower df = stronger tail dependence

3. **Archimedean Copulas**
   - Clayton: Lower tail dependence
   - Gumbel: Upper tail dependence

Mathematical Background
-----------------------
Gaussian Copula:
    C(u) = Φₙ(Φ⁻¹(u₁), ..., Φ⁻¹(uₙ); Σ)
    where Φₙ is multivariate normal CDF, Φ⁻¹ is normal quantile

Student-t Copula:
    C(u) = tₙ,ν(t⁻¹_ν(u₁), ..., t⁻¹_ν(uₙ); Σ, ν)
    Tail dependence: λ = 2tᵥ₊₁(-√((ν+1)(1-ρ)/(1+ρ)))

Clayton Copula (bivariate):
    C(u₁, u₂) = (u₁^{-θ} + u₂^{-θ} - 1)^{-1/θ}
    Lower tail dependence: λ_L = 2^{-1/θ}

Gumbel Copula (bivariate):
    C(u₁, u₂) = exp(-((-log u₁)^θ + (-log u₂)^θ)^{1/θ})
    Upper tail dependence: λ_U = 2 - 2^{1/θ}

References
----------
- Nelsen, R.B. (2006). "An Introduction to Copulas."
- Embrechts, P., Lindskog, F. & McNeil, A. (2003). "Modelling Dependence
  with Copulas and Applications to Risk Management."
- McNeil, A.J., Frey, R. & Embrechts, P. (2015). "Quantitative Risk
  Management: Concepts, Techniques and Tools."
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Protocol
from scipy import stats


class Copula(Protocol):
    """Protocol for copula implementations."""

    def sample(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate samples from the copula.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Uniform samples, shape (n_samples, n_dimensions).
        """
        ...

    @property
    def n_dim(self) -> int:
        """Number of dimensions."""
        ...


@dataclass(slots=True)
class GaussianCopula:
    """
    Gaussian Copula.

    The Gaussian copula is the dependence structure implied by the
    multivariate normal distribution. It has no tail dependence,
    which means extreme events tend to occur independently.

    Parameters
    ----------
    correlation : np.ndarray
        Correlation matrix, shape (n, n).

    Examples
    --------
    >>> corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    >>> copula = GaussianCopula(correlation=corr)
    >>> samples = copula.sample(10000, seed=42)
    >>> print(samples.shape)  # (10000, 2)
    >>> print(np.corrcoef(samples.T))  # Approximately corr

    Notes
    -----
    The Gaussian copula is simple but may underestimate joint tail risk.
    For financial applications with tail dependence, consider Student-t
    or Clayton copulas.
    """

    correlation: np.ndarray
    _cholesky: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.correlation = np.asarray(self.correlation, dtype=np.float64)
        self._validate()
        self._cholesky = np.linalg.cholesky(self.correlation)

    def _validate(self) -> None:
        """Validate correlation matrix."""
        n = self.correlation.shape[0]
        if self.correlation.shape != (n, n):
            raise ValueError("correlation must be square")
        if not np.allclose(self.correlation, self.correlation.T):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(self.correlation), 1.0):
            raise ValueError("correlation diagonal must be 1")
        eigenvalues = np.linalg.eigvalsh(self.correlation)
        if np.min(eigenvalues) <= 0:
            raise ValueError("correlation must be positive definite")

    @property
    def n_dim(self) -> int:
        return self.correlation.shape[0]

    def sample(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate samples from Gaussian copula.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Uniform samples, shape (n_samples, n_dim).
        """
        rng = np.random.default_rng(seed)

        # Generate correlated normals
        z = rng.standard_normal((n_samples, self.n_dim))
        z_correlated = z @ self._cholesky.T

        # Transform to uniform via normal CDF
        u = stats.norm.cdf(z_correlated)

        return u

    def tail_dependence(self) -> float:
        """
        Return tail dependence coefficient.

        For Gaussian copula, tail dependence is 0 for all correlations < 1.
        """
        return 0.0


@dataclass(slots=True)
class StudentTCopula:
    """
    Student-t Copula.

    The t-copula exhibits symmetric tail dependence, making it suitable
    for modeling joint extreme events. Lower degrees of freedom implies
    stronger tail dependence.

    Parameters
    ----------
    correlation : np.ndarray
        Correlation matrix, shape (n, n).
    df : float
        Degrees of freedom (must be > 2 for finite variance).
        Typical values: 3-8 for heavy tails, >30 approaches Gaussian.

    Examples
    --------
    >>> corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    >>> copula = StudentTCopula(correlation=corr, df=4)
    >>> samples = copula.sample(10000, seed=42)
    >>>
    >>> # Check tail dependence
    >>> print(f"Tail dependence: {copula.tail_dependence():.4f}")

    Notes
    -----
    Tail dependence formula (bivariate case):
    λ = 2 × t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))

    For ρ=0.6, ν=4: λ ≈ 0.25 (significant tail dependence)
    For ρ=0.6, ν=30: λ ≈ 0.03 (near Gaussian)
    """

    correlation: np.ndarray
    df: float = 4.0
    _cholesky: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.correlation = np.asarray(self.correlation, dtype=np.float64)
        if self.df <= 2:
            raise ValueError("df must be > 2 for finite variance")
        self._validate()
        self._cholesky = np.linalg.cholesky(self.correlation)

    def _validate(self) -> None:
        """Validate correlation matrix."""
        n = self.correlation.shape[0]
        if self.correlation.shape != (n, n):
            raise ValueError("correlation must be square")
        if not np.allclose(self.correlation, self.correlation.T):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(self.correlation), 1.0):
            raise ValueError("correlation diagonal must be 1")
        eigenvalues = np.linalg.eigvalsh(self.correlation)
        if np.min(eigenvalues) <= 0:
            raise ValueError("correlation must be positive definite")

    @property
    def n_dim(self) -> int:
        return self.correlation.shape[0]

    def sample(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate samples from t-copula.

        Uses the factorization: T = Z × √(ν/S) where Z ~ N(0, Σ)
        and S ~ χ²(ν).

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Uniform samples, shape (n_samples, n_dim).
        """
        rng = np.random.default_rng(seed)

        # Generate correlated normals
        z = rng.standard_normal((n_samples, self.n_dim))
        z_correlated = z @ self._cholesky.T

        # Generate chi-squared for scaling
        chi2 = rng.chisquare(df=self.df, size=n_samples)
        scaling = np.sqrt(self.df / chi2)[:, np.newaxis]

        # t-distributed samples
        t_samples = z_correlated * scaling

        # Transform to uniform via t CDF
        u = stats.t.cdf(t_samples, df=self.df)

        return u

    def tail_dependence(self) -> float:
        """
        Compute average pairwise tail dependence coefficient.

        For t-copula with correlation ρ and df ν:
        λ = 2 × t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
        """
        n = self.n_dim
        nu = self.df

        # Compute average over off-diagonal correlations
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                rho = self.correlation[i, j]
                arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
                lam = 2 * stats.t.cdf(arg, df=nu + 1)
                total += lam
                count += 1

        return total / count if count > 0 else 0.0


@dataclass(slots=True)
class ClaytonCopula:
    """
    Clayton Copula (Archimedean).

    Exhibits lower tail dependence (joint crashes) but no upper tail
    dependence. Suitable for modeling downside risk.

    Parameters
    ----------
    theta : float
        Dependence parameter, must be > 0.
        Higher theta = stronger lower tail dependence.
        θ → 0: independence
        θ → ∞: comonotonicity
    n_dim : int
        Number of dimensions.

    Examples
    --------
    >>> copula = ClaytonCopula(theta=2.0, n_dim=2)
    >>> samples = copula.sample(10000, seed=42)
    >>>
    >>> # Lower tail dependence
    >>> print(f"Lower tail λ_L: {copula.lower_tail_dependence():.4f}")
    >>> print(f"Upper tail λ_U: {copula.upper_tail_dependence():.4f}")

    Notes
    -----
    For bivariate Clayton:
    C(u₁, u₂) = (u₁^{-θ} + u₂^{-θ} - 1)^{-1/θ}
    Lower tail dependence: λ_L = 2^{-1/θ}
    Upper tail dependence: λ_U = 0
    """

    theta: float
    _n_dim: int = 2

    def __post_init__(self) -> None:
        if self.theta <= 0:
            raise ValueError("theta must be > 0 for Clayton copula")

    @property
    def n_dim(self) -> int:
        return self._n_dim

    def sample(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate samples from Clayton copula.

        Uses conditional distribution method for d dimensions.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Uniform samples, shape (n_samples, n_dim).
        """
        rng = np.random.default_rng(seed)

        d = self.n_dim
        theta = self.theta

        # Marshall-Olkin algorithm for Archimedean copulas
        # Generate gamma random variable
        v = rng.gamma(shape=1.0 / theta, scale=1.0, size=n_samples)
        
        # Generate exponential random variables
        e = rng.exponential(scale=1.0, size=(n_samples, d))
        
        # Apply Laplace transform inverse
        # For Clayton: φ(t) = (1 + t)^{-1/θ}
        # φ⁻¹(t) = (t^{-θ} - 1) / θ (not used directly)
        
        # Using conditional simulation for better numerical stability
        u = np.empty((n_samples, d), dtype=np.float64)
        
        for i in range(n_samples):
            # Generate using the gamma-exponential mixture
            for j in range(d):
                # Transform: u = (1 + e/v)^{-1/θ}
                u[i, j] = np.power(1 + e[i, j] / v[i], -1.0 / theta)

        return u

    def lower_tail_dependence(self) -> float:
        """Lower tail dependence coefficient λ_L = 2^{-1/θ}."""
        return np.power(2.0, -1.0 / self.theta)

    def upper_tail_dependence(self) -> float:
        """Upper tail dependence (always 0 for Clayton)."""
        return 0.0

    def tail_dependence(self) -> float:
        """Return lower tail dependence (the relevant one for crashes)."""
        return self.lower_tail_dependence()


@dataclass(slots=True)
class GumbelCopula:
    """
    Gumbel Copula (Archimedean).

    Exhibits upper tail dependence (joint booms) but no lower tail
    dependence. The opposite of Clayton.

    Parameters
    ----------
    theta : float
        Dependence parameter, must be >= 1.
        θ = 1: independence
        θ → ∞: comonotonicity
    n_dim : int
        Number of dimensions.

    Examples
    --------
    >>> copula = GumbelCopula(theta=3.0, n_dim=2)
    >>> samples = copula.sample(10000, seed=42)
    >>>
    >>> print(f"Lower tail λ_L: {copula.lower_tail_dependence():.4f}")
    >>> print(f"Upper tail λ_U: {copula.upper_tail_dependence():.4f}")

    Notes
    -----
    For bivariate Gumbel:
    C(u₁, u₂) = exp(-((-log u₁)^θ + (-log u₂)^θ)^{1/θ})
    Upper tail dependence: λ_U = 2 - 2^{1/θ}
    Lower tail dependence: λ_L = 0
    """

    theta: float
    _n_dim: int = 2

    def __post_init__(self) -> None:
        if self.theta < 1:
            raise ValueError("theta must be >= 1 for Gumbel copula")

    @property
    def n_dim(self) -> int:
        return self._n_dim

    def sample(
        self,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate samples from Gumbel copula.

        Uses the positive stable distribution method.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Uniform samples, shape (n_samples, n_dim).
        """
        rng = np.random.default_rng(seed)

        d = self.n_dim
        theta = self.theta

        if theta == 1.0:
            # Independence case
            return rng.uniform(size=(n_samples, d))

        alpha = 1.0 / theta

        # Generate positive stable random variable S with Laplace transform
        # E[exp(-tS)] = exp(-t^alpha)
        # Using Chambers-Mallows-Stuck method
        v = self._sample_stable(alpha, n_samples, rng)

        # Generate exponentials
        e = rng.exponential(scale=1.0, size=(n_samples, d))

        # Apply transformation
        u = np.empty((n_samples, d), dtype=np.float64)
        for i in range(n_samples):
            for j in range(d):
                # u = exp(-(e/S)^alpha)
                u[i, j] = np.exp(-np.power(e[i, j] / v[i], alpha))

        return u

    def _sample_stable(
        self,
        alpha: float,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Sample from positive stable distribution.

        Using Chambers-Mallows-Stuck algorithm for α ∈ (0, 1).
        """
        if alpha >= 1.0:
            return np.ones(n_samples)

        # Uniform on (-π/2, π/2)
        w = rng.uniform(-np.pi / 2, np.pi / 2, size=n_samples)
        # Exponential(1)
        e = rng.exponential(scale=1.0, size=n_samples)

        # Chambers-Mallows-Stuck formula
        t1 = np.sin(alpha * (w + np.pi / 2)) / np.power(np.cos(w), 1 / alpha)
        t2 = np.power(
            np.cos(w - alpha * (w + np.pi / 2)) / e,
            (1 - alpha) / alpha
        )

        return t1 * t2

    def lower_tail_dependence(self) -> float:
        """Lower tail dependence (always 0 for Gumbel)."""
        return 0.0

    def upper_tail_dependence(self) -> float:
        """Upper tail dependence coefficient λ_U = 2 - 2^{1/θ}."""
        return 2.0 - np.power(2.0, 1.0 / self.theta)

    def tail_dependence(self) -> float:
        """Return upper tail dependence."""
        return self.upper_tail_dependence()
