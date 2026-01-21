from __future__ import annotations

from dataclasses import dataclass, field

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.market import Market
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest
from src.marketdata.providers.interfaces import MarketDataProvider

from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.engine import SyntheticMarketEngine
from src.marketdata.providers.synthetic.generators.foreign_exchange import register_fx_generators
from src.marketdata.providers.synthetic.generators.interest_rate import register_ir_generators
from src.marketdata.providers.synthetic.registry import SyntheticRegistry


@dataclass(frozen=True, slots=True)
class SyntheticProvider(MarketDataProvider):
    """
    Synthetic market-data provider (Vn façade).

    Responsibilities
    ---------------
    - Build registry (generator coverage per asset class)
    - Build engine (dependency closure + deterministic generation)
    - Delegate get_market / get_timeseries

    Notes
    -----
    - name is exposed as a read-only @property (Protocol-safe).
    - Determinism is controlled by (seed, MarketId.key()) in the engine/generators.
    """
    seed: int = 7
    config: SyntheticProviderConfig = SyntheticProviderConfig()

    _name: str = "SyntheticProvider"
    _engine: SyntheticMarketEngine = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    @property
    def name(self) -> str:
        return str(self._name)

    def __post_init__(self) -> None:
        registry = SyntheticRegistry()

        # Register asset-class generators (one module per asset class).
        register_fx_generators(registry=registry, base_seed=int(self.seed), config=self.config)
        register_ir_generators(registry=registry, base_seed=int(self.seed), config=self.config)

        # Construct engine and store on frozen instance.
        engine = SyntheticMarketEngine(seed=int(self.seed), registry=registry)
        object.__setattr__(self, "_engine", engine)

    def get_market(self, request: MarketRequest) -> Market:
        return self._engine.get_market(request)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        return self._engine.get_timeseries(request)