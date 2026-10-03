"""FX scenario set — shocked FX states aligned to an FXSnapshot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.rade_static_replication.domain.errors import MarketDataError
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot
from src.rade_static_replication.marketdata.scenarios import ScenarioSet


@dataclass(frozen=True)
class FXScenarioSet(ScenarioSet):
    """Shocked FX states (absolute levels) aligned to an :class:`FXSnapshot`."""
    spot: np.ndarray = None  # type: ignore[assignment]  # (n,)
    vol: np.ndarray = None                                # (n, n_exp, n_k)
    vol_expiries: np.ndarray = None
    vol_strikes: np.ndarray = None
    domestic_rate: np.ndarray = None                      # (n, n_dom)
    domestic_tenors: np.ndarray = None
    foreign_rate: np.ndarray = None                       # (n, n_for)
    foreign_tenors: np.ndarray = None

    def validate_against(self, snap: FXSnapshot) -> None:
        n = self.n_scenarios
        problems: List[str] = []
        if self.spot.shape != (n,):
            problems.append(f"spot {self.spot.shape} != ({n},)")
        if self.vol.shape != (n, self.vol_expiries.size, self.vol_strikes.size):
            problems.append(
                f"vol {self.vol.shape} != ({n}, {self.vol_expiries.size}, {self.vol_strikes.size})"
            )
        if self.domestic_rate.shape != (n, self.domestic_tenors.size):
            problems.append(f"domestic_rate {self.domestic_rate.shape} != ({n}, {self.domestic_tenors.size})")
        if self.foreign_rate.shape != (n, self.foreign_tenors.size):
            problems.append(f"foreign_rate {self.foreign_rate.shape} != ({n}, {self.foreign_tenors.size})")
        if problems:
            raise MarketDataError(f"{self.factor_id} FX scenario set misaligned: " + "; ".join(problems))
