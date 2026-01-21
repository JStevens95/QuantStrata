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
    Synthetic market-data provider (Vn public façade).

    Responsibilities
    ---------------
    - Build a SyntheticRegistry (register FX/IR/... generators).
    - Build a SyntheticMarketEngine (dependency closure + deterministic orchestration).
    - Delegate get_market/get_timeseries to the engine.

    Notes
    -----
    - Provider API stays stable; engine/generators can evolve behind it.
    - Determinism is enforced by per-MarketId RNG substreams.
    """

    seed: int = 7
    config: SyntheticProviderConfig = field(default_factory=SyntheticProviderConfig)
    _name: str = "SyntheticProvider"

    _engine: SyntheticMarketEngine = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    @property
    def name(self) -> str:
        return str(self._name)

    def __post_init__(self) -> None:
        registry = SyntheticRegistry()

        register_fx_generators(registry=registry, base_seed=int(self.seed), config=self.config)
        register_ir_generators(registry=registry, base_seed=int(self.seed), config=self.config)

        engine = SyntheticMarketEngine(seed=int(self.seed), registry=registry)
        object.__setattr__(self, "_engine", engine)

    def get_market(self, request: MarketRequest) -> Market:
        return self._engine.get_market(request)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        ds = self._engine.get_timeseries(request)

        # Non-breaking provenance improvement:
        # ensure dataset.meta reflects the provider façade name, not the internal engine.
        meta0 = dict(ds.meta or {})
        meta = {
            **meta0,
            "provider": self.name,
            "provider_mode": "synthetic_generation",
            "freq": str(meta0.get("freq", request.freq)).strip().upper(),
        }

        # Return a new MarketDataset with identical storage but updated meta.
        return MarketDataset(
            dates=list(ds.dates),
            n_scenarios=int(ds.n_scenarios),
            panels=ds.panels,
            curve_params=ds.curve_params,
            curve_factories=ds.curve_factories,
            vol_params=ds.vol_params,
            vol_factories=ds.vol_factories,
            meta=meta,
        )