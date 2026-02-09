"""
Lookback option payoff implementations (path-dependent).

Lookback options pay based on the maximum or minimum price of the underlying
over the option's life. Two main variants exist:
- Floating strike: Strike determined by path extremum
- Fixed strike: Payoff based on path extremum vs fixed strike

Mathematical Foundations
------------------------
Lookback options exploit the maximum/minimum of Brownian motion:

Under GBM, let M_T = max_{0<=t<=T} S_t and m_T = min_{0<=t<=T} S_t

For continuous monitoring, there are closed-form solutions:
- The distribution of M_T (and m_T) under GBM is known (reflection principle)
- Goldman-Sosin-Gatto (1979) derived the pricing formulas

For discrete monitoring:
- No simple closed-form exists
- MC simulation is the standard approach
- More monitoring points → continuous approximation

Key Properties
--------------
1. Floating strike lookback call: Payoff = S_T - m_T >= 0 (always ITM)
2. Floating strike lookback put:  Payoff = M_T - S_T >= 0 (always ITM)
3. Fixed strike lookback call:    Payoff = max(M_T - K, 0)
4. Fixed strike lookback put:     Payoff = max(K - m_T, 0)

5. Lookback >= Vanilla (captures optimal timing)
6. Floating strike lookbacks are ALWAYS in-the-money

Interview Points
----------------
- Delta of floating strike lookback call at inception ≈ 2 (very sensitive)
- Lookback premium represents value of "perfect market timing"
- Discrete monitoring creates "continuation value" vs continuous
- Reflection principle: key mathematical tool for continuous case
"""
from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass
from typing import Literal

from src.models.payoffs.base import BasePathPayoff1D, _as_paths_array, _validate_option_type
from src.models.payoffs.types import OptionType
from src.instruments.core.types import LookbackType

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class LookbackPayoff(BasePathPayoff1D):
    """
    Lookback option payoff (path-dependent): pays based on path extremum.

    This payoff computes the maximum or minimum spot price over the simulated path
    and uses it to determine the payoff.

    Mathematical Definition
    -----------------------
    Let M_T = max(S_t) and m_T = min(S_t) over the path.

    **Floating Strike:**
    - Call: Payoff = S_T - m_T (always >= 0, guaranteed profit)
    - Put:  Payoff = M_T - S_T (always >= 0, guaranteed profit)

    **Fixed Strike (K):**
    - Call: Payoff = max(M_T - K, 0) (option on maximum spot)
    - Put:  Payoff = max(K - m_T, 0) (option on minimum spot)

    Path Structure
    --------------
    - `paths` has shape (n_paths, n_steps + 1)
    - Column 0 is S0 (initial spot)
    - Columns 1..n_steps are intermediate monitoring points
    - Last column is S_T (terminal spot)
    - All points contribute to the max/min calculation

    Returns
    -------
    np.ndarray
        Payoff per unit notional, shape (n_paths,), in domestic currency.

    Notes
    -----
    - This is path-dependent: we need the full path to compute extrema.
    - Floating strike lookbacks are ALWAYS in-the-money (payoff >= 0).
    - More monitoring points give better approximation to continuous extremum.
    - Discrete monitoring typically gives lower value than continuous.
    """

    option_type: OptionType  # "call" or "put"
    lookback_type: LookbackType  # "floating_strike" or "fixed_strike"
    strike: float = 0.0  # Strike price K (only used for fixed_strike)

    def __post_init__(self) -> None:
        """
        Validate payoff parameters.

        Ensures that the payoff is well-formed before computation.
        """
        # Validate option type: must be "call" or "put"
        _validate_option_type(self.option_type)

        # Validate lookback type: must be "floating_strike" or "fixed_strike"
        if self.lookback_type not in ("floating_strike", "fixed_strike"):
            raise ValueError("lookback_type must be 'floating_strike' or 'fixed_strike'.")

        # Validate strike for fixed_strike type
        if self.lookback_type == "fixed_strike":
            if float(self.strike) <= 0.0:
                raise ValueError("strike must be > 0 for fixed_strike lookback.")

    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute lookback option payoff from full simulated paths.

        This method:
        1. Computes the max/min spot price over each path
        2. Applies the appropriate lookback payoff formula
        3. Returns payoffs per unit notional

        Parameters
        ----------
        paths : np.ndarray
            Spot paths, shape (n_paths, n_steps + 1).
            Column 0 is S0, last column is S_T, intermediate columns are monitoring points.

        Returns
        -------
        np.ndarray
            Payoffs per unit notional, shape (n_paths,), in domestic currency.
        """
        # Validate and normalize paths array to ensure correct shape and dtype
        p = _as_paths_array(paths)

        # Extract terminal spots S_T (last column of paths)
        # This is needed for floating strike lookbacks
        terminal_spots = p[:, -1]

        # Compute path extrema
        # Maximum spot over each path (for put floating strike, call fixed strike)
        max_spots = np.max(p, axis=1)
        # Minimum spot over each path (for call floating strike, put fixed strike)
        min_spots = np.min(p, axis=1)

        # Apply appropriate payoff formula based on lookback type and option type
        if self.lookback_type == "floating_strike":
            # Floating strike: strike is determined by path extremum
            payoffs = self._floating_strike_payoff(terminal_spots, max_spots, min_spots)
        else:
            # Fixed strike: payoff based on path extremum vs fixed strike K
            payoffs = self._fixed_strike_payoff(max_spots, min_spots)

        # Return as float64 array (ensures consistent dtype across payoffs)
        return payoffs.astype(np.float64, copy=False)

    def _floating_strike_payoff(
        self,
        terminal_spots: np.ndarray,
        max_spots: np.ndarray,
        min_spots: np.ndarray,
    ) -> np.ndarray:
        """
        Compute floating strike lookback payoff.

        Floating strike lookbacks have their strike determined by the path extremum:
        - Call: Strike = min(S_t), Payoff = S_T - min(S_t) >= 0
        - Put:  Strike = max(S_t), Payoff = max(S_t) - S_T >= 0

        These are ALWAYS in-the-money: the payoff is guaranteed to be >= 0.

        Parameters
        ----------
        terminal_spots : np.ndarray
            Terminal spots S_T, shape (n_paths,).
        max_spots : np.ndarray
            Maximum spots over each path, shape (n_paths,).
        min_spots : np.ndarray
            Minimum spots over each path, shape (n_paths,).

        Returns
        -------
        np.ndarray
            Payoffs, shape (n_paths,).
        """
        if self.option_type == "call":
            # Floating strike call: Payoff = S_T - min(S_t)
            # The "strike" is the minimum spot, so you always buy at the lowest point
            # This is always >= 0 since S_T >= min(S_t) by definition
            return terminal_spots - min_spots
        else:
            # Floating strike put: Payoff = max(S_t) - S_T
            # The "strike" is the maximum spot, so you always sell at the highest point
            # This is always >= 0 since max(S_t) >= S_T by definition
            return max_spots - terminal_spots

    def _fixed_strike_payoff(
        self,
        max_spots: np.ndarray,
        min_spots: np.ndarray,
    ) -> np.ndarray:
        """
        Compute fixed strike lookback payoff.

        Fixed strike lookbacks have a predetermined strike K, and the payoff
        depends on the path extremum:
        - Call: Payoff = max(max(S_t) - K, 0)  (option on maximum spot)
        - Put:  Payoff = max(K - min(S_t), 0)  (option on minimum spot)

        Parameters
        ----------
        max_spots : np.ndarray
            Maximum spots over each path, shape (n_paths,).
        min_spots : np.ndarray
            Minimum spots over each path, shape (n_paths,).

        Returns
        -------
        np.ndarray
            Payoffs, shape (n_paths,).
        """
        k = float(self.strike)

        if self.option_type == "call":
            # Fixed strike call: Payoff = max(max(S_t) - K, 0)
            # This is an option on the maximum spot over the path
            # You profit if the maximum exceeds the strike
            return np.maximum(max_spots - k, 0.0)
        else:
            # Fixed strike put: Payoff = max(K - min(S_t), 0)
            # This is an option on the minimum spot over the path
            # You profit if the minimum is below the strike
            return np.maximum(k - min_spots, 0.0)
