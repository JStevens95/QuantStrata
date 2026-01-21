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
    Synthetic market-data provider (public façade).

    Responsibilities
    ---------------
    - Register asset-class generators (FX/IR/...) into a SyntheticRegistry.
    - Build a SyntheticMarketEngine that performs dependency closure + deterministic generation.
    - Expose get_market/get_timeseries via the MarketDataProvider protocol.

    Notes
    -----
    We store provider identification as a concrete *field* `name` (not a property),
    to avoid accidental resolution to Protocol-level descriptors.
    """

    seed: int = 7
    config: SyntheticProviderConfig = field(default_factory=SyntheticProviderConfig)

    # IMPORTANT: concrete field (not @property) to satisfy runtime expectations/tests.
    name: str = "SyntheticProvider"

    _engine: SyntheticMarketEngine = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """
        Build the registry and engine once (frozen dataclass; use object.__setattr__).
        """
        registry = SyntheticRegistry()

        # Register generator families. Each generator should use deterministic substreams.
        register_fx_generators(registry=registry, base_seed=int(self.seed), config=self.config)
        register_ir_generators(registry=registry, base_seed=int(self.seed), config=self.config)

        engine = SyntheticMarketEngine(seed=int(self.seed), registry=registry)
        object.__setattr__(self, "_engine", engine)

    # -------------------------------------------------------------------------
    # MarketDataProvider API
    # -------------------------------------------------------------------------

    def get_market(self, request: MarketRequest) -> Market:
        return self._engine.get_market(request)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        """
        Delegate to the engine, then ensure provider meta is stamped consistently.
        """
        ds = self._engine.get_timeseries(request)

        # Provider meta should reflect the façade name, not internal engine details.
        meta0 = dict(ds.meta or {})
        meta = {
            **meta0,
            "provider": str(self.name),
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