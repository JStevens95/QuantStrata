from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.synthetic.config import SyntheticProviderConfig
from src.marketdata.synthetic.specs import CurveZeroSpec, SpotGbmSpec, VolGridSmileSpec


def test_config_returns_defaults_when_no_overrides() -> None:
    """
    SyntheticProviderConfig should return default specs when no overrides exist.

    This test is intentionally simple:
    - It validates the config "routing" behaviour (not numeric generation).
    """
    # Build a config using defaults (constructor defaults).
    config = SyntheticProviderConfig()

    # Create representative MarketIds for each spec accessor.
    spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    curve_mid = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),))
    vol_mid = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")

    # Ensure returned objects are the expected spec types.
    assert isinstance(config.spot_spec(spot_mid), SpotGbmSpec)
    assert isinstance(config.curve_zero_spec(curve_mid), CurveZeroSpec)
    assert isinstance(config.vol_spec(vol_mid), VolGridSmileSpec)

    # Ensure curve method returns a normalized string.
    assert config.curve_method_for(curve_mid) in {"zeros", "bootstrap"}


def test_config_curve_method_override_is_applied_per_market_id() -> None:
    """
    curve_method_overrides should take precedence over the global curve_method.

    This is desk-grade important because per-curve method routing is a real workflow.
    """
    # Create a curve MarketId.
    curve_mid = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),))

    # Create config with default curve_method="zeros".
    config = SyntheticProviderConfig(curve_method="zeros")

    # By default, this MarketId should use "zeros".
    assert config.curve_method_for(curve_mid) == "zeros"

    # Now override that exact MarketId to "bootstrap".
    config2 = SyntheticProviderConfig(
        curve_method="zeros",
        curve_method_overrides={curve_mid: "bootstrap"},
    )

    # The override should win.
    assert config2.curve_method_for(curve_mid) == "bootstrap"


def test_config_raises_on_invalid_curve_method_override() -> None:
    """
    Invalid curve methods should raise a ValueError.
    """
    curve_mid = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),))

    # Override with an invalid value.
    config = SyntheticProviderConfig(curve_method_overrides={curve_mid: "not-a-method"})  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        _ = config.curve_method_for(curve_mid)


def test_config_spec_override_by_market_id_object_identity() -> None:
    """
    Overrides are keyed by MarketId instances.

    This test validates the current semantics:
    - If your MarketId implements hashing/equality by fields, this works as expected.
    - If not, this will highlight that and force us to standardize keying.
    """
    # MarketId for FX SPOT.
    mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")

    # Define a custom spot spec override.
    override_spec = SpotGbmSpec(initial_level=1.25, drift=0.01, vol=0.15)

    # Build config with an override for that MarketId.
    config = SyntheticProviderConfig(spot_overrides={mid: override_spec})

    # The accessor should return the override spec.
    got = config.spot_spec(mid)
    assert got is override_spec
    assert float(got.initial_level) == 1.25

    # Sanity: different MarketId should not pick up the override.
    other = MarketId(asset_class="FX", mkt_type="SPOT", name="GBPUSD")
    got_other = config.spot_spec(other)
    assert got_other is not override_spec