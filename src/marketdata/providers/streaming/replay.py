"""
Replay stream provider: yields (timestamp, Market) from a MarketDataset.

Use for testing and for running strategies on historical data in streaming mode
without an external API.
"""

from __future__ import annotations

from typing import AsyncIterator, List, Optional, Tuple

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.market import Market


class ReplayStreamProvider:
    """
    Stream provider that replays a MarketDataset or a list of (timestamp, Market).

    Yields (timestamp, Market) in order. No external API; for simulation and tests.
    """

    def __init__(
        self,
        dataset: Optional[MarketDataset] = None,
        snapshots: Optional[List[Tuple[str, Market]]] = None,
        *,
        scenario_idx: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        dataset : MarketDataset or None
            If provided, stream is built from dataset.dates and dataset.snapshot(time_idx, scenario_idx).
        snapshots : list of (timestamp, Market) or None
            If provided, stream yields these in order. Ignored if dataset is provided.
        scenario_idx : int
            Scenario index when using dataset (default 0).
        """
        if dataset is not None and snapshots is not None:
            raise ValueError("Provide either dataset or snapshots, not both.")
        if dataset is None and snapshots is None:
            raise ValueError("Provide either dataset or snapshots.")
        self._dataset = dataset
        self._snapshots = snapshots if snapshots is not None else []
        self._scenario_idx = int(scenario_idx)

    async def stream(self) -> AsyncIterator[tuple[str, Market]]:
        if self._dataset is not None:
            for time_idx in range(len(self._dataset.dates)):
                ts = self._dataset.dates[time_idx]
                market = self._dataset.snapshot(time_idx=time_idx, scenario_idx=self._scenario_idx)
                yield (ts, market)
        else:
            for ts, market in self._snapshots:
                yield (ts, market)
