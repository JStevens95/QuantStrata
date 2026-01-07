from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict

import pytest

from src.marketdata.core.ids import MarketId
from src.pricers.fx.spot import FxSpotPricer
from src.instruments.fx.linear.spot import FxSpot


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------

def _construct_dataclass(cls: type[Any], **kwargs: Any) -> Any:
    """
    Construct a dataclass (or any callable class) by filtering kwargs
    down to the constructor signature.

    Why:
    - Your instruments may evolve (e.g., renaming notional fields, adding optional fields).
    - This keeps unit tests stable across small Vn refactors.
    """
    sig = inspect.signature(cls)
    params = sig.parameters

    filtered = {k: v for k, v in kwargs.items() if k in params}

    missing_required: list[str] = []
    for name, p in params.items():
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            if p.default is inspect._empty and name not in filtered:
                missing_required.append(name)

    if missing_required:
        raise TypeError(f"Missing required constructor args for {cls.__name__}: {missing_required}")

    return cls(**filtered)


@dataclass(frozen=True)
class _DummyMarket:
    """
    Minimal market stub for FxSpotPricer tests.

    The pricer only needs:
      - market.quote(spot_id) -> float
    """
    spot_id: MarketId
    spot_value: float

    def quote(self, market_id: MarketId) -> float:
        if market_id != self.spot_id:
            raise KeyError(f"Unknown quote id: {market_id}")
        return float(self.spot_value)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture()
def ids() -> Dict[str, MarketId]:
    return {
        "spot": MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD", qualifiers=()),
    }


@pytest.fixture()
def pricer() -> FxSpotPricer:
    return FxSpotPricer()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_fx_spot_price_equals_spot_times_multiplier(ids: Dict[str, MarketId], pricer: FxSpotPricer) -> None:
    """
    PV should be linear in spot:
        PV = contract_multiplier * spot
    """
    spot = 1.2345
    multiplier = 2.0

    market = _DummyMarket(spot_id=ids["spot"], spot_value=spot)

    trade = _construct_dataclass(
        FxSpot,
        spot_id=ids["spot"],
        contract_multiplier=multiplier,
    )

    pv = pricer.price(trade, market)
    assert pv == pytest.approx(multiplier * spot, rel=0.0, abs=0.0)


def test_fx_spot_delta_equals_multiplier(ids: Dict[str, MarketId], pricer: FxSpotPricer) -> None:
    """
    Delta for linear spot is constant:
        delta = dPV/dS = contract_multiplier
    """
    multiplier = 3.5
    market = _DummyMarket(spot_id=ids["spot"], spot_value=1.1)

    trade = _construct_dataclass(
        FxSpot,
        spot_id=ids["spot"],
        contract_multiplier=multiplier,
    )

    greeks = pricer.greeks(trade, market)
    assert greeks["delta"] == pytest.approx(multiplier, rel=0.0, abs=0.0)


@pytest.mark.parametrize("bad_spot", [float("nan"), float("inf"), float("-inf")])
def test_fx_spot_price_raises_on_non_finite_spot(ids: Dict[str, MarketId], pricer: FxSpotPricer, bad_spot: float) -> None:
    """
    The pricer should reject non-finite spot quotes (NaN/Inf) defensively.
    """
    market = _DummyMarket(spot_id=ids["spot"], spot_value=bad_spot)

    trade = _construct_dataclass(
        FxSpot,
        spot_id=ids["spot"],
        contract_multiplier=1.0,
    )

    with pytest.raises(ValueError, match="Spot quote must be finite"):
        pricer.price(trade, market)