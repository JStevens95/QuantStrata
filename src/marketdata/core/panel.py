from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, slots=True)
class Panel:
    """
    A lightweight numeric container for MarketDataset storage.

    Attributes
    ----------
    data:
        Numpy array containing the panel values.
    axis_names:
        Tuple of axis names matching data.ndim (e.g. ("time", "scenario", "tenor")).

    Notes
    -----
    This object stays intentionally minimal: (data, axis_names).
    The factories convert blocks into rich Curve/VolSurface objects at snapshot-time.
    """
    data: np.ndarray
    axis_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        # Ensure numpy array backing.
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Panel.data must be a numpy.ndarray.")

        # Ensure axis_names matches dimensionality.
        if self.data.ndim != len(self.axis_names):
            raise ValueError(
                f"Panel axis mismatch: data.ndim={self.data.ndim} but axis_names={self.axis_names}."
            )

        # Validate axis name strings are non-empty (helps debugging and slicing logic).
        for ax in self.axis_names:
            if not str(ax).strip():
                raise ValueError("Panel.axis_names must contain non-empty strings.")

    def scalar_at(self, time_idx: int, scenario_idx: int = 0) -> float:
        """
        Extract a scalar for panels shaped [T] or [T,S].

        This is intentionally strict because it is used for Market.quote().
        If you want blocks, use `_slice_params` (in dataset.py) instead.
        """
        x = self.data

        # Panel with only time axis: [T]
        if x.ndim == 1:
            return float(x[time_idx])

        # Panel with time + scenario axes: [T,S]
        if x.ndim == 2:
            return float(x[time_idx, scenario_idx])

        # Anything else is not a scalar quote panel.
        raise ValueError(f"Panel.scalar_at supports ndim in {{1,2}} only; got ndim={x.ndim}.")