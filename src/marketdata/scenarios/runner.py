from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.scenarios.interfaces import MarketView, ScenarioPack, ScenarioShock


@dataclass(frozen=True, slots=True)
class ScenarioRunner:
    """
    Helper that bridges:
      MarketDataset.snapshot(...) -> Market (which is a MarketView)
    and applies ScenarioShocks (wrappers) to produce shocked MarketViews.
    """
    dataset: MarketDataset

    def base_snapshot(self, *, time_idx: int, scenario_idx: int = 0) -> MarketView:
        return self.dataset.snapshot(time_idx=time_idx, scenario_idx=scenario_idx)

    def shocked_snapshot(
        self,
        *,
        time_idx: int,
        scenario_idx: int = 0,
        shock: ScenarioShock,
    ) -> MarketView:
        base = self.base_snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
        return shock.apply(base)

    def apply_pack(
        self,
        *,
        time_idx: int,
        scenario_idx: int = 0,
        pack: ScenarioPack,
    ) -> Mapping[str, MarketView]:
        base = self.base_snapshot(time_idx=time_idx, scenario_idx=scenario_idx)
        return pack.apply_all(base)