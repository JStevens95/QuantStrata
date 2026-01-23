# tests/orchestrator/pipelines/pricing/test_price_portfolio.py

from __future__ import annotations

import pytest

from src.orchestrator.config.schemas import IOConfig, RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.orchestrator.core.errors import StepError

from src.orchestrator.pipelines.marketdata import build_timeseries as md_pipeline
from src.orchestrator.pipelines.pricing import price_portfolio as px_pipeline

from src.marketdata.core.ids import MarketId
from src.portfolio.core import Portfolio, Position
from src.instruments.fx.linear.spot import FxSpot


def _build_market_snapshot(tmp_path) -> object:
    params = {
        "marketdata": {
            "provider": {"type": "synthetic", "seed": 123},
            "ids": ["FX.SPOT.EURUSD"],
            "start": "2026-01-01",
            "end": "2026-01-03",
            "freq": "D",
            "scenarios": 1,
            "snapshot": {"time": "last", "scenario_idx": 0},
        },
        "pricing": {"registry": {"kind": "default"}},
    }

    cfg = RunConfig(
        pipeline="marketdata.build_timeseries",
        io=IOConfig(workdir=str(tmp_path)),
        params=params,
    )
    pipeline = md_pipeline.build_pipeline(cfg)

    ctx = Context(
        run_id="test_run",
        cfg=cfg,
        logger=None,
        artifact_store=None,
        provider=None,
        state={},
    )

    out = PipelineRunner().run(pipeline, ctx)
    return out.get(Keys.MARKET)


def _run_pricing_pipeline(*, tmp_path, state: dict) -> Context:
    params = {
        "pricing": {"registry": {"kind": "default"}},
        # Keep params dict valid; pricing pipeline only reads pricing block
    }
    cfg = RunConfig(
        pipeline="pricing.price_portfolio",
        io=IOConfig(workdir=str(tmp_path)),
        params=params,
    )
    pipeline = px_pipeline.build_pipeline(cfg)

    ctx = Context(
        run_id="test_run",
        cfg=cfg,
        logger=None,
        artifact_store=None,
        provider=None,
        state=state,
    )
    return PipelineRunner().run(pipeline, ctx)


def test_price_portfolio_missing_inputs_fail_fast(tmp_path) -> None:
    with pytest.raises(StepError) as excinfo:
        _run_pricing_pipeline(tmp_path=tmp_path, state={})

    assert isinstance(excinfo.value.__cause__, KeyError)
    assert "Missing ctx.state['portfolio']" in str(excinfo.value.__cause__)


def test_price_portfolio_happy_path_spot_only(tmp_path) -> None:
    market = _build_market_snapshot(tmp_path)

    spot_id = MarketId.parse("FX.SPOT.EURUSD")
    spot = float(market.quote(spot_id))

    portfolio = Portfolio(
        positions=[
            Position(
                position_id="SPOT_1",
                instrument=FxSpot(spot_id=spot_id, contract_multiplier=1.0),
                quantity=100_000.0,
            )
        ]
    )

    state = {Keys.MARKET: market, Keys.PORTFOLIO: portfolio}

    out = _run_pricing_pipeline(tmp_path=tmp_path, state=state)

    # Registry + outputs exist
    assert Keys.PRICER_REGISTRY in out.state
    assert Keys.PORTFOLIO_PRICING_RESULT in out.state
    assert Keys.PORTFOLIO_PRICING_SUMMARY in out.state

    res = out.get(Keys.PORTFOLIO_PRICING_RESULT)
    summary = out.get(Keys.PORTFOLIO_PRICING_SUMMARY)

    assert summary["n_positions"] == 1
    assert summary["has_total_greeks"] in (True, False)

    # Basic PV sanity: should scale with spot and quantity (tolerant)
    expected = 100_000.0 * spot
    assert res.totals.pv == pytest.approx(expected, rel=1e-9, abs=1e-6)