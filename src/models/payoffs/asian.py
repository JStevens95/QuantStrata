from __future__ import annotations

import sys
import numpy as np
from dataclasses import dataclass
from typing import Literal

from src.models.payoffs.base import BasePathPayoff1D, _as_paths_array, _validate_option_type
from src.models.payoffs.types import OptionType

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}

# Define averaging type for Asian payoffs
AsianAveragingType = Literal["arithmetic", "geometric"]


@dataclass(**_DATACLASS_KW)
class AsianPayoff(BasePathPayoff1D):
    """
    Asian option payoff (path-dependent): pays based on average price over path.

    This payoff computes the average of spot prices over the simulated path and applies
    a vanilla-style payoff to that average. The averaging can be arithmetic or geometric.

    Mathematical Definition
    -----------------------
    For a call option:
        Payoff = max(Avg(S_t) - K, 0)
    
    For a put option:
        Payoff = max(K - Avg(S_t), 0)

    Where:
    - Arithmetic average: Avg(S_t) = (S_1 + S_2 + ... + S_n) / n
    - Geometric average: Avg(S_t) = (S_1 * S_2 * ... * S_n)^(1/n)

    Path Structure
    --------------
    - `paths` has shape (n_paths, n_steps + 1)
    - Column 0 is S0 (initial spot)
    - Columns 1..n_steps are intermediate monitoring points
    - Last column is S_T (terminal spot)
    - All points are used in the average calculation

    Returns
    -------
    np.ndarray
        Payoff per unit notional, shape (n_paths,), in domestic currency.

    Notes
    -----
    - This is path-dependent: we need the full path to compute the average.
    - Geometric averaging has closed-form solutions under GBM, but arithmetic is
      more common in practice and requires numerical methods (MC).
    - The average includes S0 and all intermediate points, providing a realistic
      discrete monitoring approximation.
    """

    option_type: OptionType  # "call" or "put"
    strike: float  # Strike price K
    averaging_type: AsianAveragingType = "arithmetic"  # "arithmetic" or "geometric"

    def __post_init__(self) -> None:
        """
        Validate payoff parameters.

        Ensures that the payoff is well-formed before computation.
        """
        # Validate option type: must be "call" or "put"
        _validate_option_type(self.option_type)

        # Validate strike: must be positive
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Validate averaging type: must be "arithmetic" or "geometric"
        if self.averaging_type not in ("arithmetic", "geometric"):
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'.")

    def terminal_from_paths(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute Asian option payoff from full simulated paths.

        This method:
        1. Computes the average spot price over each path
        2. Applies vanilla-style payoff to the average
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

        # Compute average spot price for each path
        # The average is computed over all columns (S0, intermediate points, S_T)
        average_spots = self._compute_average(p)

        # Extract strike as float for vectorized operations
        k = float(self.strike)

        # Apply vanilla-style payoff to the average spot
        if self.option_type == "call":
            # Call payoff: max(Avg(S) - K, 0)
            # np.maximum ensures we get 0 when average is below strike
            payoffs = np.maximum(average_spots - k, 0.0)
        else:
            # Put payoff: max(K - Avg(S), 0)
            # np.maximum ensures we get 0 when average is above strike
            payoffs = np.maximum(k - average_spots, 0.0)

        # Return as float64 array (ensures consistent dtype across payoffs)
        return payoffs.astype(np.float64, copy=False)

    def _compute_average(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute average spot price over each path.

        This is a helper method that handles both arithmetic and geometric averaging.
        The average is computed over all columns of the path array (including S0 and S_T).

        Parameters
        ----------
        paths : np.ndarray
            Spot paths, shape (n_paths, n_steps + 1).

        Returns
        -------
        np.ndarray
            Average spot prices, shape (n_paths,).
        """
        if self.averaging_type == "arithmetic":
            # Arithmetic mean: (S_1 + S_2 + ... + S_n) / n
            # np.mean along axis=1 computes mean across columns (time) for each path
            # This gives us the arithmetic average for each path
            return np.mean(paths, axis=1, dtype=np.float64)

        # Geometric mean: (S_1 * S_2 * ... * S_n)^(1/n)
        # For numerical stability, we use log-space:
        #   log(geometric_mean) = mean(log(S_i))
        #   geometric_mean = exp(mean(log(S_i)))
        # This avoids overflow when multiplying many large numbers
        log_paths = np.log(paths)  # Take log of each spot value
        log_mean = np.mean(log_paths, axis=1, dtype=np.float64)  # Mean of logs
        geometric_mean = np.exp(log_mean)  # Exponentiate to get geometric mean
        return geometric_mean
