from __future__ import annotations

from typing import Tuple

from src.marketdata.core.ids import MarketId
from src.marketdata.synthetic.context import SyntheticGenerationState
from src.marketdata.synthetic.registry import SyntheticRegistry


def test_registry_register_and_lookup_is_case_insensitive() -> None:
    """Registry keys normalise casing for predictable desk behaviour."""
    reg = SyntheticRegistry()

    def _gen(mid: MarketId, state: SyntheticGenerationState) -> None:  # pragma: no cover
        # Generator is never called here; this test is about routing.
        return None

    # Register with lower-case inputs.
    reg.register(asset_class="fx", mkt_type="spot", generator=_gen)

    # Lookup uses MarketId fields and should match regardless of case.
    mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    assert reg.get(market_id=mid) is _gen


def test_registry_requirements_default_empty() -> None:
    """If requirements resolver not registered, requirements() returns empty tuple."""
    reg = SyntheticRegistry()
    mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    assert reg.requirements(market_id=mid) == tuple()


def test_registry_requirements_resolver_is_called() -> None:
    """If registered, requirements() should call the resolver and return a tuple."""
    reg = SyntheticRegistry()

    def _reqs(mid: MarketId) -> Tuple[MarketId, ...]:
        return (MarketId(asset_class="FX", mkt_type="SPOT", name=mid.name),)

    def _gen(mid: MarketId, state: SyntheticGenerationState) -> None:  # pragma: no cover
        return None

    reg.register(asset_class="FX", mkt_type="VOL", generator=_gen, requirements=_reqs)

    vol_mid = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    reqs = reg.requirements(market_id=vol_mid)

    assert len(reqs) == 1
    assert reqs[0].mkt_type == "SPOT"
    assert reqs[0].name == "EURUSD"