from __future__ import annotations

import numpy as np

from src.marketdata.providers.synthetic.context import SyntheticGenerationState


def test_generation_state_is_mutable_container() -> None:
    """
    SyntheticGenerationState is intentionally a mutable “run context”:
    generators write into dicts owned by the engine.
    """
    # Build a minimal state with empty containers.
    state = SyntheticGenerationState(
        dates=["2026-01-01", "2026-01-02"],
        n_time=2,
        n_scenarios=3,
        quote_panels={},
        curve_param_panels={},
        curve_factories={},
        vol_param_panels={},
        vol_factories={},
        spot_cache={},
    )

    # Mutate the cache to confirm it behaves as a shared container.
    state.spot_cache["dummy"] = np.array([1.0])  # type: ignore[index]
    assert "dummy" in state.spot_cache