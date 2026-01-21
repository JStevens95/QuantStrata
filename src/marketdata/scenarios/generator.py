from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """
    Declarative scenario specification.

    Factors
    -------
    You define named factors like:
      - "FX:EURUSD"
      - "IR:USD:LEVEL"
      - "IR:USD:SLOPE"
      - "EQ:SPX"
      - "CR:CDX_IG"

    The driver outputs a correlated normal shock cube Z[t,s,f].
    """
    factor_names: tuple[str, ...]
    correlation: np.ndarray  # shape (F,F)
    dt: float = 1.0 / 252.0  # default daily step in year fractions


@dataclass(slots=True)
class ScenarioDriver:
    """
    Correlated scenario driver producing Gaussian shocks.

    This is intentionally simple in Phase 0:
      - generates Z ~ N(0, Corr) independently per time step
      - does not impose mean reversion or term structure models yet

    In later phases, generators will transform Z into:
      - GBM increments for spot
      - yield curve factor shocks
      - spread shocks, etc.
    """

    spec: ScenarioSpec

    def __post_init__(self) -> None:
        corr = np.asarray(self.spec.correlation, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError("ScenarioSpec.correlation must be a square 2D matrix.")
        if corr.shape[0] != len(self.spec.factor_names):
            raise ValueError("ScenarioSpec.factor_names length must match correlation dimension.")
        if np.any(~np.isfinite(corr)):
            raise ValueError("ScenarioSpec.correlation must be finite.")
        if float(self.spec.dt) <= 0.0 or not np.isfinite(float(self.spec.dt)):
            raise ValueError("ScenarioSpec.dt must be finite and > 0.")

        # Ensure symmetry (small numerical tolerance) and unit diagonal.
        if np.max(np.abs(corr - corr.T)) > 1e-10:
            raise ValueError("ScenarioSpec.correlation must be symmetric.")
        if np.max(np.abs(np.diag(corr) - 1.0)) > 1e-10:
            raise ValueError("ScenarioSpec.correlation must have ones on the diagonal.")

        # Cholesky can fail if matrix is not SPD; keep a clear error.
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as e:
            raise ValueError("ScenarioSpec.correlation must be positive definite (Cholesky failed).") from e

    def sample_shocks(self, *, rng: np.random.Generator, n_time: int, n_scenarios: int) -> np.ndarray:
        """
        Generate correlated standard normal shocks Z[t,s,f].

        Returns
        -------
        np.ndarray
            Shape (n_time, n_scenarios, n_factors)
        """
        n_factors = len(self.spec.factor_names)

        # Generate iid standard normals first.
        z = rng.normal(loc=0.0, scale=1.0, size=(n_time, n_scenarios, n_factors))

        # Apply correlation via Cholesky.
        chol = np.linalg.cholesky(np.asarray(self.spec.correlation, dtype=float))
        # For each (t,s), apply chol @ z_vec
        z_corr = z @ chol.T
        return z_corr