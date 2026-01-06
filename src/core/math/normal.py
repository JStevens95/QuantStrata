from __future__ import annotations

import math
import numpy as np
from typing import Final


# -----------------------------------------------------------------------------
# Standard normal distribution helpers (pure math, dependency-free)
# -----------------------------------------------------------------------------

_SQRT_2: Final[float] = math.sqrt(2.0)
_SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)


def std_norm_pdf(x: float) -> float:
    """
    Standard normal probability density function φ(x).

    Parameters
    ----------
    x:
        Point at which to evaluate the PDF.

    Returns
    -------
    float
        φ(x) = exp(-x²/2) / sqrt(2π)
    """
    x = float(x)
    return float(math.exp(-0.5 * x * x) / _SQRT_2PI)


def std_norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function Φ(x).

    Implementation
    --------------
    Uses the error function:
        Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))

    Parameters
    ----------
    x:
        Point at which to evaluate the CDF.

    Returns
    -------
    float
        Φ(x)
    """
    x = float(x)
    return float(0.5 * (1.0 + math.erf(x / _SQRT_2)))


def std_normal_ppf(p: np.ndarray) -> np.ndarray:
    """
    Approximate inverse CDF (quantile) for standard normal using Acklam's approximation.

    Notes
    -----
    - This avoids SciPy dependency.
    - Accuracy is excellent for plotting / diagnostics.
    """
    p = np.asarray(p, dtype=np.float64)

    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("Probabilities must be in (0, 1).")

    # Coefficients in rational approximations
    a = np.array(
        [-3.969683028665376e01,  2.209460984245205e02, -2.759285104469687e02,
          1.383577518672690e02, -3.066479806614716e01,  2.506628277459239e00],
        dtype=np.float64,
    )
    b = np.array(
        [-5.447609879822406e01,  1.615858368580409e02, -1.556989798598866e02,
          6.680131188771972e01, -1.328068155288572e01],
        dtype=np.float64,
    )
    c = np.array(
        [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00,  4.374664141464968e00,  2.938163982698783e00],
        dtype=np.float64,
    )
    d = np.array(
        [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e00,
          3.754408661907416e00],
        dtype=np.float64,
    )

    # Break-points
    plow = 0.02425
    phigh = 1.0 - plow

    x = np.empty_like(p)

    # Lower region
    mask = p < plow
    if np.any(mask):
        q = np.sqrt(-2.0 * np.log(p[mask]))
        x[mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                  ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    # Central region
    mask = (p >= plow) & (p <= phigh)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q * q
        x[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                  (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

    # Upper region
    mask = p > phigh
    if np.any(mask):
        q = np.sqrt(-2.0 * np.log(1.0 - p[mask]))
        x[mask] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    return x