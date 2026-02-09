# src/models/numeric/dynamics/gbm.py
from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass
from typing import Callable, Literal

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


# --------------------------------------------------------------------------------------
# Typing
# --------------------------------------------------------------------------------------

# Supported discretization schemes for geometric Brownian motion (GBM).
GbmScheme = Literal["exact", "euler", "milstein"]

# A single time-step update function:
#   s_next = step(s, mu, sigma, dt, sqrt_dt, z)
# where:
#   s      : current spot values for all paths, shape (n_paths,)
#   mu     : drift
#   sigma  : volatility
#   dt     : timestep size
#   sqrt_dt: sqrt(dt) (precomputed once for efficiency)
#   z      : standard normals for this step, shape (n_paths,)
_GbmStepFn = Callable[[np.ndarray, float, float, float, float, np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------
# Scheme implementations (pure step math)
# --------------------------------------------------------------------------------------

def _gbm_step_exact(
    spot: np.ndarray,
    drift: float,
    vol: float,
    dt: float,
    sqrt_dt: float,
    z: np.ndarray,
) -> np.ndarray:
    """
    Apply the *exact* GBM transition over one time-step.

    Model
    -----
    dS_t = mu S_t dt + sigma S_t dW_t

    Exact discretization
    --------------------
    S_{t+dt} = S_t * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )

    Parameters
    ----------
    spot:
        Current spot for all paths, shape (n_paths,).
    drift:
        Drift parameter mu (continuous-time).
    vol:
        Volatility parameter sigma (>= 0).
    dt:
        Time-step length.
    sqrt_dt:
        sqrt(dt).
    z:
        Standard normal variates for this step, shape (n_paths,).

    Returns
    -------
    np.ndarray
        Next-step spot values, shape (n_paths,).
    """
    # Compute the log-return increment for each path.
    log_increment = (drift - 0.5 * vol * vol) * dt + vol * sqrt_dt * z
    # Apply multiplicative lognormal evolution.
    return spot * np.exp(log_increment)


def _gbm_step_euler(
    spot: np.ndarray,
    drift: float,
    vol: float,
    dt: float,
    sqrt_dt: float,
    z: np.ndarray,
) -> np.ndarray:
    """
    Apply the Euler–Maruyama discretization for GBM over one time-step.

    Euler step
    ----------
    S_{t+dt} = S_t + mu*S_t*dt + sigma*S_t*dW
    with dW = sqrt(dt)*Z

    Notes
    -----
    - Euler is *not* positivity-preserving for GBM (S can become negative).
    - Useful for comparisons / debugging, but 'exact' is preferred for GBM.
    """
    dW = sqrt_dt * z
    return spot + drift * spot * dt + vol * spot * dW


def _gbm_step_milstein(
    spot: np.ndarray,
    drift: float,
    vol: float,
    dt: float,
    sqrt_dt: float,
    z: np.ndarray,
) -> np.ndarray:
    """
    Apply the Milstein discretization for GBM over one time-step.

    Milstein step
    -------------
    S_{t+dt} = S_t + mu*S_t*dt + sigma*S_t*dW + 0.5*sigma^2*S_t*(dW^2 - dt)

    Notes
    -----
    - For GBM, Milstein often improves strong convergence over Euler.
    - Still not as clean as the 'exact' GBM step (which you should default to).
    """
    dW = sqrt_dt * z
    return spot + drift * spot * dt + vol * spot * dW + 0.5 * vol * vol * spot * (dW * dW - dt)


# Dispatch table: scheme name -> step function.
_GBM_STEP_FUNCTIONS: dict[GbmScheme, _GbmStepFn] = {
    "exact": _gbm_step_exact,
    "euler": _gbm_step_euler,
    "milstein": _gbm_step_milstein,
}


# --------------------------------------------------------------------------------------
# Public dynamics object
# --------------------------------------------------------------------------------------

@dataclass(**_DATACLASS_KW)
class GbmDynamicsSimulator:
    """
    Geometric Brownian Motion (GBM) dynamics simulator.

    This class is intentionally small: it is responsible for *path generation only*.
    Pricing logic belongs in pricers; random-number generation belongs in your RNG module.

    Model
    -----
    dS_t = mu S_t dt + sigma S_t dW_t

    Parameters
    ----------
    drift:
        The continuous-time drift (mu).
    vol:
        The continuous-time volatility (sigma). Must be non-negative.
    """

    drift: float
    vol: float

    def simulate_paths(
        self,
        *,
        spot0: float,
        maturity: float,
        n_steps: int,
        n_paths: int,
        normals: np.ndarray,
        scheme: GbmScheme = "exact",
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """
        Simulate GBM spot paths on an equally-spaced time grid.

        Parameters
        ----------
        spot0:
            Initial spot S(0).
        maturity:
            Time horizon in years (or consistent time units).
        n_steps:
            Number of time steps over [0, maturity]. Output has n_steps+1 columns.
        n_paths:
            Number of simulated paths (rows).
        normals:
            Standard normal draws with shape (n_paths, n_steps).
            This is passed in so RNG concerns are cleanly separated.
        scheme:
            Discretization scheme: "exact" (recommended), "euler", "milstein".
        dtype:
            Output dtype for returned array. float64 recommended for stability.

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_steps + 1) with simulated spots.
            Column 0 is spot0, column i is S(t_i).

        Raises
        ------
        ValueError
            If inputs have invalid shapes or values.
        """
        # -----------------------------
        # Validate scalar inputs
        # -----------------------------
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive.")
        if spot0 <= 0.0 and scheme == "exact":
            # Exact GBM assumes S0 > 0 (log-normal).
            raise ValueError("spot0 must be positive for exact GBM.")
        if self.vol < 0.0:
            raise ValueError("vol must be non-negative.")
        if scheme not in _GBM_STEP_FUNCTIONS:
            raise ValueError(f"Unknown scheme='{scheme}'. Allowed: {list(_GBM_STEP_FUNCTIONS)}")

        # -----------------------------
        # Validate and normalize normals
        # -----------------------------
        normals = np.asarray(normals, dtype=np.float64)
        expected_shape = (int(n_paths), int(n_steps))
        if normals.shape != expected_shape:
            raise ValueError(f"normals must have shape {expected_shape}, got {normals.shape}.")

        # -----------------------------
        # Precompute time-step constants
        # -----------------------------
        dt = float(maturity) / float(n_steps)      # length of each time step
        sqrt_dt = float(np.sqrt(dt))               # sqrt(dt) used in dW = sqrt(dt)*Z

        # Resolve the scheme implementation once (avoids branching inside the loop).
        step_fn = _GBM_STEP_FUNCTIONS[scheme]

        # -----------------------------
        # Allocate output paths array
        # -----------------------------
        # paths[p, i] = spot for path p at time index i
        paths = np.empty((n_paths, n_steps + 1), dtype=dtype)

        # Current spot values for all paths at the current step (work array in float64).
        spot = np.full((n_paths,), float(spot0), dtype=np.float64)

        # Store initial spot at t=0.
        paths[:, 0] = spot.astype(dtype, copy=False)

        # -----------------------------
        # Time stepping
        # -----------------------------
        drift = float(self.drift)
        vol = float(self.vol)

        for step_idx in range(n_steps):
            # Grab the standard normals for this time step (one per path).
            z = normals[:, step_idx]

            # Advance all paths by one step using the chosen scheme.
            spot = step_fn(spot, drift, vol, dt, sqrt_dt, z)

            # Store results for this time point into output matrix.
            paths[:, step_idx + 1] = spot.astype(dtype, copy=False)

        return paths