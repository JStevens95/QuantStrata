from __future__ import annotations

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.providers.synthetic.config import SyntheticProviderConfig
from src.marketdata.providers.synthetic.specs import CurveZeroSpec, VolGridSmileSpec
from src.marketdata.providers.synthetic.context import SyntheticGenerationState
from src.marketdata.providers.synthetic.generators.foreign_exchange import _FxGenerators, _default_ir_curve_id
from src.marketdata.providers.synthetic.generators.interest_rate import _IrGenerators


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


def test_fx_vol_requirements_include_spot_and_curves_when_dom_for_present() -> None:
    """FX VOL should require SPOT + dom/for curves when qualifiers specify dom/for."""
    cfg = SyntheticProviderConfig()
    fx = _FxGenerators(base_seed=7, config=cfg)

    vol_mid = MarketId(
        asset_class="FX",
        mkt_type="VOL",
        name="EURUSD",
        qualifiers=(("dom", "USD"), ("for", "EUR")),
    )

    reqs = fx.requirements_for_vol(vol_mid)
    keys = {r.key() for r in reqs}

    assert "FX.SPOT.EURUSD|dom=USD|for=EUR" in keys
    assert _default_ir_curve_id("USD").key() in keys
    assert _default_ir_curve_id("EUR").key() in keys


def test_fx_generators_spot_then_vol_produce_expected_panels() -> None:
    """
    FX spot should produce [T,S].
    FX vol should produce [T,S,n_exp,n_k] and be strictly positive (floored).
    """
    n_time = 3
    n_scenarios = 2

    cfg = SyntheticProviderConfig(
        curve_method="zeros",
        curve_zero=CurveZeroSpec(tenors=np.array([0.25, 0.5, 1.0], dtype=float)),
        vol=VolGridSmileSpec(
            expiries=np.array([0.5, 1.0], dtype=float),
            strikes=np.array([0.9, 1.0, 1.1], dtype=float),
        ),
    )

    state = _make_state(n_time=n_time, n_scenarios=n_scenarios)

    # Build curve deps via the IR generator (so FX vol can read carry).
    ir = _IrGenerators(base_seed=7, config=cfg)
    usd_curve = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=(("ccy", "USD"),))
    eur_curve = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers=(("ccy", "EUR"),))
    ir.generate_curve(usd_curve, state)
    ir.generate_curve(eur_curve, state)

    fx = _FxGenerators(base_seed=7, config=cfg)

    spot_mid = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD", qualifiers=(("dom", "USD"), ("for", "EUR")))
    vol_mid = MarketId(
        asset_class="FX",
        mkt_type="VOL",
        name="EURUSD",
        qualifiers=(("dom", "USD"), ("for", "EUR")),
    )

    # Generate SPOT first.
    fx.generate_spot(spot_mid, state)
    spot = np.asarray(state.quote_panels[spot_mid].data, dtype=float)
    assert spot.shape == (n_time, n_scenarios)

    # Generate VOL next.
    fx.generate_vol_grid_forward_moneyness(vol_mid, state)
    vol = np.asarray(state.vol_param_panels[vol_mid].data, dtype=float)

    assert vol.shape == (n_time, n_scenarios, 2, 3)  # [T,S,n_exp,n_k]
    assert np.all(np.isfinite(vol))
    assert np.min(vol) > 0.0