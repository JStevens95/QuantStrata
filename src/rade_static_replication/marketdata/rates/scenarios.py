"""Rates scenario set — shocked curve (+ cube) states aligned to a RatesSnapshot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.rade_static_replication.domain.errors import MarketDataError
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot
from src.rade_static_replication.marketdata.scenarios import ScenarioSet


@dataclass(frozen=True)
class RatesScenarioSet(ScenarioSet):
    """Shocked rates states (absolute levels) aligned to a :class:`RatesSnapshot`."""
    curve: np.ndarray = None  # type: ignore[assignment]  # (n, n_pillars)
    curve_tenors: np.ndarray = None
    vol: Optional[np.ndarray] = None        # (n, n_exp, n_ten, n_k)
    vol_expiries: Optional[np.ndarray] = None
    vol_swap_tenors: Optional[np.ndarray] = None
    vol_strikes: Optional[np.ndarray] = None

    def validate_against(self, snap: RatesSnapshot) -> None:
        n = self.n_scenarios
        problems: List[str] = []
        if self.curve.shape != (n, self.curve_tenors.size):
            problems.append(f"curve {self.curve.shape} != ({n}, {self.curve_tenors.size})")
        if self.vol is not None:
            expected = (n, self.vol_expiries.size, self.vol_swap_tenors.size, self.vol_strikes.size)
            if self.vol.shape != expected:
                problems.append(f"vol {self.vol.shape} != {expected}")
        if problems:
            raise MarketDataError(f"{self.factor_id} rates scenario set misaligned: " + "; ".join(problems))
