from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FlatVolSurface:
    """
    Flat implied volatility surface.

    This is the minimal and most stable surface for V1 because:
      - it requires only one parameter (a single implied vol),
      - it is easy to generate synthetically and to test,
      - it supports immediate option pricing (Black–Scholes / Garman–Kohlhagen style).

    Parameters
    ----------
    implied_vol:
        Constant implied volatility (e.g., 0.12 for 12% annualized).

    Notes
    -----
    - `expiry` is expressed in year fractions from as-of.
    - `strike` is passed for interface compatibility but is not used for a flat surface.
    - This object is intentionally pure and side-effect free.
    """

    implied_vol: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.implied_vol):
            raise ValueError("FlatVolSurface.implied_vol must be finite.")
        if self.implied_vol <= 0.0:
            raise ValueError("FlatVolSurface.implied_vol must be strictly positive.")

    def vol(self, expiry: float, strike: float) -> float:
        """
        Return the implied volatility for a given expiry and strike.

        Parameters
        ----------
        expiry:
            Time to expiry in year fractions (>= 0).
        strike:
            Strike (unused for flat surface; kept for interface consistency).

        Returns
        -------
        float
            Constant implied volatility.
        """
        # Expiry validation is helpful to catch upstream bugs early.
        if not np.isfinite(expiry):
            raise ValueError("expiry must be finite.")
        if expiry < 0.0:
            raise ValueError("expiry must be >= 0.")
        # strike is not used; we do not validate it here to keep the surface generic.
        return float(self.implied_vol)