from __future__ import annotations

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.synthetic.config import SyntheticProviderConfig
from src.marketdata.synthetic.specs import CurveZeroSpec
from src.marketdata.synthetic.context import SyntheticGenerationState
from src.marketdata.synthetic.generators.interest_rate import _IrGenerators


def _make_state(*, n_time: int, n_scenarios: int) -> SyntheticGenerationState:
    """Build a minimal state container for generator-level tests."""
    return SyntheticGenerationState(
        dates=["2026-01-01"] * n_time,
        n_time=n_time,
        n_scenarios=n_scenarios,
        quote_panels={},
        curve_param_panels={},
        curve_factories={},
        vol_param_panels={},
        vol_factories={},
        spot_cache={},
    )


def test_ir_curve_generator_zeros_produces_params_and_factory() -> None:
    """IR curve generator should store curve params and wire a ZeroRateCurveFactory."""
    mid = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),))

    cfg = SyntheticProviderConfig(
        curve_method="zeros",
        curve_zero=CurveZeroSpec(tenors=np.array([0.25, 0.5, 1.0], dtype=float)),
    )

    ir = _IrGenerators(base_seed=7, config=cfg)
    state = _make_state(n_time=2, n_scenarios=3)

    ir.generate_curve(mid, state)

    assert mid in state.curve_param_panels
    assert mid in state.curve_factories

    arr = np.asarray(state.curve_param_panels[mid].data, dtype=float)
    assert arr.shape == (2, 3, 3, 2)  # [T,S,K,2]
    assert state.curve_param_panels[mid].axis_names == ("time", "scenario", "tenor", "cols")