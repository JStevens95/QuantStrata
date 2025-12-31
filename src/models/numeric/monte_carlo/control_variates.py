from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ControlVariateResult:
    """
    Result from a control variate adjustment.

    adjusted:
        Adjusted samples: y - beta*(c - E[c])
    beta:
        Estimated control variate coefficient.
    """

    adjusted: np.ndarray
    beta: float


def apply_control_variate(
    *,
    y: np.ndarray,
    c: np.ndarray,
    c_expectation: float,
) -> ControlVariateResult:
    """
    Apply a single control variate.

    Parameters
    ----------
    y:
        Payoff samples (shape (n,)).
    c:
        Control samples (shape (n,)).
    c_expectation:
        Known expectation E[c].

    Returns
    -------
    ControlVariateResult(adjusted, beta)

    Notes
    -----
    beta is estimated via least squares:
        beta = Cov(y, c) / Var(c)
    with numerical guards for Var(c) ~ 0.
    """
    yv = np.asarray(y, dtype=np.float64)
    cv = np.asarray(c, dtype=np.float64)

    if yv.ndim != 1 or cv.ndim != 1:
        raise ValueError("y and c must be 1D arrays.")
    if yv.size != cv.size:
        raise ValueError("y and c must have the same length.")
    if yv.size == 0:
        raise ValueError("y and c must be non-empty.")

    c_centered = cv - float(cv.mean())
    var_c = float(np.dot(c_centered, c_centered)) / max(1, (cv.size - 1))
    if var_c <= 0.0:
        # Degenerate control variate; no adjustment.
        return ControlVariateResult(adjusted=yv.copy(), beta=0.0)

    y_centered = yv - float(yv.mean())
    cov_yc = float(np.dot(y_centered, c_centered)) / max(1, (cv.size - 1))

    beta = cov_yc / var_c
    adjusted = yv - beta * (cv - float(c_expectation))
    return ControlVariateResult(adjusted=adjusted, beta=float(beta))