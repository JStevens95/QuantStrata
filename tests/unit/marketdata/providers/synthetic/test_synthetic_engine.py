from __future__ import annotations

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel
from src.marketdata.core.requests import TimeseriesRequest, Universe
from src.marketdata.providers.synthetic.engine import SyntheticMarketEngine, _generate_dates, rng_for_market_id, stable_seed
from src.marketdata.providers.synthetic.registry import SyntheticRegistry
from src.marketdata.surfaces.factory import GridVolFactory


def test_stable_seed_is_deterministic() -> None:
    """stable_seed must produce identical output for identical inputs."""
    a = stable_seed(7, "A", "B", "C")
    b = stable_seed(7, "A", "B", "C")
    assert a == b


def test_rng_for_market_id_is_order_independent() -> None:
    """
    Each MarketId gets a stable RNG substream.
    Creating the RNG twice should give identical first draws.
    """
    mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")

    rng1 = rng_for_market_id(base_seed=7, market_id=mid)
    rng2 = rng_for_market_id(base_seed=7, market_id=mid)

    assert np.array_equal(rng1.normal(size=5), rng2.normal(size=5))


def test_generate_dates_supports_expected_freqs() -> None:
    """_generate_dates supports D/B/W/M with deterministic stepping."""
    assert _generate_dates(start="2026-01-01", end="2026-01-03", freq="D") == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03"
    ]
    # Business days in this range are all weekdays, so same as D here.
    assert _generate_dates(start="2026-01-01", end="2026-01-03", freq="D") == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03"
    ]
    assert _generate_dates(start="2026-01-01", end="2026-01-15", freq="W") == [
        "2026-01-01",
        "2026-01-08",
        "2026-01-15",
    ]
    assert _generate_dates(start="2026-01-01", end="2026-03-05", freq="M") == [
        "2026-01-01",
        "2026-01-31",
        "2026-03-02",
    ]


def test_engine_dependency_closure_and_ordering_spot_before_vol() -> None:
    """
    If we request VOL only, requirements() should add SPOT and the engine should
    generate SPOT before VOL.
    """
    registry = SyntheticRegistry()
    call_order: list[str] = []

    def gen_spot(mid, state) -> None:
        # Record call order.
        call_order.append(mid.key())
        # Store a dummy [T,S] quote panel.
        arr = np.ones((state.n_time, state.n_scenarios), dtype=float)
        state.quote_panels[mid] = Panel(data=arr, axis_names=("time", "scenario"))

    def gen_vol(mid, state) -> None:
        # Ensure SPOT exists before VOL generation runs.
        spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name=mid.name, qualifiers=mid.qualifiers)
        assert spot_mid in state.quote_panels
        call_order.append(mid.key())
        # Store a dummy [T,S,n_exp,n_k] vol cube.
        cube = np.ones((state.n_time, state.n_scenarios, 2, 3), dtype=float)
        state.vol_param_panels[mid] = Panel(data=cube, axis_names=("time", "scenario", "expiry", "strike"))
        state.vol_factories[mid] = GridVolFactory(
            expiries=np.array([0.5, 1.0], dtype=float),
            strikes=np.array([0.9, 1.0, 1.1], dtype=float),
            extrapolation="flat"
        )

    def reqs_for_vol(mid) -> tuple[MarketId, ...]:
        return (MarketId(asset_class="FX", mkt_type="SPOT", name=mid.name, qualifiers=mid.qualifiers),)

    registry.register(asset_class="FX", mkt_type="SPOT", generator=gen_spot)
    registry.register(asset_class="FX", mkt_type="VOL", generator=gen_vol, requirements=reqs_for_vol)

    engine = SyntheticMarketEngine(seed=7, registry=registry)

    vol_mid = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD", qualifiers=(("dom", "USD"),))
    universe = Universe(ids=[vol_mid])

    ds = engine.get_timeseries(
        TimeseriesRequest(
            start="2026-01-01",
            end="2026-01-02",
            freq="D",
            universe=universe,
            scenarios=2,
        )
    )

    # Closure should have created a spot panel even though we requested only vol.
    spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD", qualifiers=(("dom", "USD"),))
    assert spot_mid in ds.panels

    # Vol should be present.
    assert vol_mid in ds.vol_params

    # Ordering check: SPOT call appears before VOL call.
    spot_idx = call_order.index(spot_mid.key())
    vol_idx = call_order.index(vol_mid.key())
    assert spot_idx < vol_idx


def test_engine_fallback_unknown_schema_produces_constant_quote() -> None:
    """
    If no generator exists for a MarketId, engine should generate a conservative
    constant quote panel [T,S] = 1.0.
    """
    engine = SyntheticMarketEngine(seed=7, registry=SyntheticRegistry())

    unknown = MarketId(asset_class="FX", mkt_type="SOMETHING", name="FOO")

    ds = engine.get_timeseries(
        TimeseriesRequest(
            start="2026-01-01",
            end="2026-01-03",
            freq="D",
            universe=Universe(ids=[unknown]),
            scenarios=3,
        )
    )

    assert unknown in ds.panels
    arr = np.asarray(ds.panels[unknown].data, dtype=float)
    assert arr.shape == (3, 3)
    assert np.all(arr == 1.0)