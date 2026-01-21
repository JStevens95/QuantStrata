from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.artifacts import save_market_dataset
from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.panel import Panel
from src.marketdata.providers.factory import (
    ProviderBuildError,
    StaticProviderSpec,
    SyntheticProviderSpec,
    build_provider,
)
from src.marketdata.providers.static.provider import StaticProvider
from src.marketdata.providers.synthetic.provider import SyntheticProvider


def _make_tiny_dataset() -> MarketDataset:
    dates = ["2026-01-01", "2026-01-02"]
    n_scenarios = 2

    mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    arr = np.full((len(dates), n_scenarios), 1.10, dtype=float)
    panels = {mid: Panel(data=arr, axis_names=("time", "scenario"))}

    # Keep curves/vols empty for this test dataset.
    return MarketDataset(
        dates=dates,
        n_scenarios=n_scenarios,
        panels=panels,
        curve_params={},
        curve_factories={},
        vol_params={},
        vol_factories={},
        meta={"freq": "D"},
    )


def test_build_provider_synthetic_returns_synthetic_provider() -> None:
    provider = build_provider(SyntheticProviderSpec(seed=123))
    assert isinstance(provider, SyntheticProvider)
    assert provider.name == "SyntheticProvider"


def test_build_provider_static_from_dataset_returns_static_provider() -> None:
    ds = _make_tiny_dataset()
    provider = build_provider(StaticProviderSpec(dataset=ds))
    assert isinstance(provider, StaticProvider)
    assert provider.name == "StaticProvider"


def test_build_provider_static_from_path_uses_artifact_loader(tmp_path) -> None:
    ds = _make_tiny_dataset()
    artifact_dir = save_market_dataset(ds, tmp_path / "mds", overwrite=True)

    provider = build_provider(StaticProviderSpec(dataset_path=str(artifact_dir)))
    assert isinstance(provider, StaticProvider)

    # Smoke: ensure provider can return a timeseries slice without error.
    # (This is the main “wiring” guarantee.)
    from src.marketdata.core.requests import TimeseriesRequest, Universe

    req = TimeseriesRequest(
        start="2026-01-01",
        end="2026-01-02",
        freq="D",
        universe=Universe(ids=tuple(ds.panels.keys())),
        scenarios=2,
    )
    out = provider.get_timeseries(req)
    assert out.dates == ["2026-01-01", "2026-01-02"]
    assert out.n_scenarios == 2


def test_build_provider_static_missing_dataset_and_path_raises() -> None:
    with pytest.raises(ProviderBuildError, match="requires either"):
        build_provider(StaticProviderSpec(dataset=None, dataset_path=None))