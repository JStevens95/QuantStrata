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

    Why keep this class?
    --------------------
    - Your examples/tests already import and use SyntheticProvider.
    - We want the public provider API stable.
    - Internally we upgrade architecture to a registry + engine model.

    Responsibilities
    ---------------
    - Construct a SyntheticRegistry (FX/IR/EQ/... generators registered here).
    - Construct a SyntheticMarketEngine (core orchestration).
    - Delegate get_market/get_timeseries to the engine.

    Non-responsibilities
    --------------------
    - This provider does not implement market generation logic.
      That lives in: src/marketdata/synthetic/generators/<asset_class>.py
    """
    name: str = "SyntheticProvider"
    seed: int = 7
    config: SyntheticProviderConfig = SyntheticProviderConfig()

    # Private engine instance (constructed in __post_init__).
    _engine: SyntheticMarketEngine = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """
        Construct the registry + engine.

        Notes
        -----
        We build the registry here so:
        - provider instances are self-contained
        - configuration is bound at construction time
        - downstream calls are clean and fast
        """
        registry = SyntheticRegistry()

        # Register asset-class generators (one module per asset class).
        register_fx_generators(registry=registry, base_seed=int(self.seed), config=self.config)
        register_ir_generators(registry=registry, base_seed=int(self.seed), config=self.config)

        # Create the orchestration engine.
        engine = SyntheticMarketEngine(seed=int(self.seed), registry=registry)

        # Store the engine on this frozen dataclass.
        object.__setattr__(self, "_engine", engine)

    # ---------------------------------------------------------------------
    # Public API (stable)
    # ---------------------------------------------------------------------

    def get_market(self, request: MarketRequest) -> Market:
        """
        Return a single Market snapshot for request.asof.

        Delegates to SyntheticMarketEngine.
        """
        return self._engine.get_market(request)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        """
        Return a MarketDataset across request.start..request.end and scenarios.

        Delegates to SyntheticMarketEngine.
        """
        return self._engine.get_timeseries(request)