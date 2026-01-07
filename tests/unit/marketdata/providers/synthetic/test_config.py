from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.specs import SpotGbmSpec, CurveBootstrapSpec, CurveZeroSpec


def test_config_spot_override_resolves() -> None:
    mid = MarketId("FX", "SPOT", "EURUSD")

    cfg = SyntheticProviderConfig(
        spot=SpotGbmSpec(initial_level=1.10),
        spot_overrides={mid: SpotGbmSpec(initial_level=1.25)},
    )

    assert cfg.spot_spec(mid).initial_level == 1.25


def test_config_curve_method_default_and_override() -> None:
    mid = MarketId("IR", "CURVE", "USD.OIS")

    cfg = SyntheticProviderConfig(curve_method="zeros")
    assert cfg.curve_method_for(mid) == "zeros"

    cfg2 = SyntheticProviderConfig(
        curve_method="zeros",
        curve_method_overrides={mid: "bootstrap"},
    )
    assert cfg2.curve_method_for(mid) == "bootstrap"


def test_config_curve_method_invalid_raises() -> None:
    mid = MarketId("IR", "CURVE", "USD.OIS")
    cfg = SyntheticProviderConfig(curve_method_overrides={mid: "nonsense"})

    with pytest.raises(ValueError, match="Invalid curve_method"):
        _ = cfg.curve_method_for(mid)


def test_config_curve_specs_override_resolve() -> None:
    mid = MarketId("IR", "CURVE", "USD.OIS")

    cfg = SyntheticProviderConfig(
        curve_zero=CurveZeroSpec(tenors=np.array([0.5, 1.0], dtype=float)),
        curve_zero_overrides={mid: CurveZeroSpec(tenors=np.array([0.25, 0.5], dtype=float))},
        curve_bootstrap=CurveBootstrapSpec(),
        curve_bootstrap_overrides={mid: CurveBootstrapSpec(swap_maturities=(1.0, 2.0))},
    )

    assert cfg.curve_zero_spec(mid).tenors.shape == (2,)
    assert cfg.curve_bootstrap_spec(mid).swap_maturities == (1.0, 2.0)