from __future__ import annotations

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import TimeseriesRequest, Universe
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.provider import SyntheticProvider


def _make_test_universe() -> Universe:
    """
    Build a Universe that exercises:
    - FX spot generation
    - FX vol generation (with dom/for qualifiers so closure adds IR curves)
    - IR curve generation (via closure)
    """
    # FX spot is explicit.
    spot = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")

    # FX vol requests a surface. We encode dom/for so FX VOL requirements include IR curves.
    vol = MarketId(
        asset_class="FX",
        mkt_type="VOL",
        name="EURUSD",
        qualifiers=(("dom", "USD"), ("for", "EUR"), ("cut", "NY"), ("conv", "delta25")),
    )

    # We do NOT explicitly request curves here; dependency closure should add them.
    return Universe(ids=[spot, vol])


def test_provider_get_timeseries_builds_dataset_with_expected_panels() -> None:
    """
    Provider should return a MarketDataset with:
    - quote panels for requested FX SPOT
    - vol params for requested FX VOL
    - curve params for dependency-closed IR curves (USD and EUR)
    """
    # Provider config: use zeros curve method (Stage 2 implementation).
    config = SyntheticProviderConfig(curve_method="zeros")

    # Create provider with a fixed seed for determinism.
    provider = SyntheticProvider(seed=7, config=config)

    # Build request.
    universe = _make_test_universe()
    request = TimeseriesRequest(
        start="2026-01-01",
        end="2026-01-03",
        freq="D",
        universe=universe,
        scenarios=2,
    )

    # Generate dataset.
    ds = provider.get_timeseries(request)

    # Basic dataset invariants.
    assert len(ds.dates) == 3
    assert int(ds.n_scenarios) == 2

    # Ensure SPOT panel exists (explicit request).
    spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    assert spot_mid in ds.panels

    # Ensure VOL panel exists (explicit request).
    vol_mid = MarketId(
        asset_class="FX",
        mkt_type="VOL",
        name="EURUSD",
        qualifiers=(("dom", "USD"), ("for", "EUR"), ("cut", "NY"), ("conv", "delta25")),
    )
    assert vol_mid in ds.vol_params

    # Ensure dependency closure produced curves (implicit request).
    usd_curve = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),))
    eur_curve = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers=(("ccy", "EUR"),))

    assert usd_curve in ds.curve_params
    assert eur_curve in ds.curve_params

    # Check canonical shapes (spot: [T,S]).
    spot_arr = np.asarray(ds.panels[spot_mid].data, dtype=float)
    assert spot_arr.shape == (3, 2)

    # Check canonical shapes (curve: [T,S,K,2]).
    usd_curve_arr = np.asarray(ds.curve_params[usd_curve].data, dtype=float)
    assert usd_curve_arr.ndim == 4
    assert usd_curve_arr.shape[0] == 3
    assert usd_curve_arr.shape[1] == 2
    assert usd_curve_arr.shape[-1] == 2  # [tenor, zero_rate]

    # Check canonical shapes (vol: [T,S,n_exp,n_k]).
    vol_arr = np.asarray(ds.vol_params[vol_mid].data, dtype=float)
    assert vol_arr.ndim == 4
    assert vol_arr.shape[0] == 3
    assert vol_arr.shape[1] == 2


def test_provider_is_deterministic_for_same_seed_and_config() -> None:
    """
    Two providers with the same seed/config should generate identical datasets.

    We compare specific panels exactly (not approximately) because:
    - RNG is stable per MarketId key
    - We use pure numpy generation
    """
    config = SyntheticProviderConfig(curve_method="zeros")

    provider_a = SyntheticProvider(seed=7, config=config)
    provider_b = SyntheticProvider(seed=7, config=config)

    universe = _make_test_universe()

    request = TimeseriesRequest(
        start="2026-01-01",
        end="2026-01-03",
        freq="D",
        universe=universe,
        scenarios=2,
    )

    ds_a = provider_a.get_timeseries(request)
    ds_b = provider_b.get_timeseries(request)

    # Compare SPOT array exactly.
    spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    a_spot = np.asarray(ds_a.panels[spot_mid].data, dtype=float)
    b_spot = np.asarray(ds_b.panels[spot_mid].data, dtype=float)
    assert np.array_equal(a_spot, b_spot)

    # Compare one curve array exactly (USD OIS).
    usd_curve = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),))
    a_curve = np.asarray(ds_a.curve_params[usd_curve].data, dtype=float)
    b_curve = np.asarray(ds_b.curve_params[usd_curve].data, dtype=float)
    assert np.array_equal(a_curve, b_curve)

    # Compare vol cube exactly.
    vol_mid = MarketId(
        asset_class="FX",
        mkt_type="VOL",
        name="EURUSD",
        qualifiers=(("dom", "USD"), ("for", "EUR"), ("cut", "NY"), ("conv", "delta25")),
    )
    a_vol = np.asarray(ds_a.vol_params[vol_mid].data, dtype=float)
    b_vol = np.asarray(ds_b.vol_params[vol_mid].data, dtype=float)
    assert np.array_equal(a_vol, b_vol)