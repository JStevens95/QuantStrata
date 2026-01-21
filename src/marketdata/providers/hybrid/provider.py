from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.market import Market
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest
from src.marketdata.providers.synthetic.engine import SyntheticMarketEngine


class MarketDataProvider(Protocol):
    """
    Provider protocol used by the Hybrid provider.

    This matches the shape of your existing provider API:
      - get_market(MarketRequest) -> Market
      - get_timeseries(TimeseriesRequest) -> MarketDataset
    """

    def get_market(self, request: MarketRequest) -> Market: ...
    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset: ...


@dataclass(frozen=True, slots=True)
class HybridMarketDataProvider:
    """
    Hybrid provider that merges real providers and synthetic fill.

    Phase 0 behaviour
    -----------------
    - Calls the primary provider if supplied, else uses synthetic only.
    - Returns the primary dataset unchanged for now.

    Phase 1+ behaviour (we will implement next)
    ------------------------------------------
    - Detect missing MarketIds/panels in the primary dataset
    - Call synthetic engine to generate only missing pieces
    - Merge into a single dataset with provenance metadata
    """

    synthetic_engine: SyntheticMarketEngine
    primary: Optional[MarketDataProvider] = None

    def get_market(self, request: MarketRequest) -> Market:
        """
        Build a single Market snapshot.

        Implementation detail: call get_timeseries() for one date and slice snapshot.
        """
        ds = self.get_timeseries(
            TimeseriesRequest(
                start=request.asof,
                end=request.asof,
                freq="D",
                universe=request.universe,
                scenarios=max(1, 1 if request.scenario is None else int(request.scenario) + 1),
            )
        )
        scenario_idx = 0 if request.scenario is None else int(request.scenario)
        return ds.snapshot(time_idx=0, scenario_idx=scenario_idx)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        """
        Return a MarketDataset sourced from primary provider, then filled by synthetic engine.

        Note
        ----
        The merge logic is intentionally deferred to Phase 1 so that we land the
        core synthetic engine cleanly first.
        """
        if self.primary is None:
            return self.synthetic_engine.generate(request)

        primary_ds = self.primary.get_timeseries(request)

        # Phase 1 will merge here.
        return primary_ds