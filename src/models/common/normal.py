from __future__ import annotations

import math
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