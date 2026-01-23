# tests/orchestrator/pipelines/marketdata/test_build_timeseries.py

from __future__ import annotations

import pytest

from src.orchestrator.config.schemas import IOConfig, RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.orchestrator.core.errors import StepError

from src.orchestrator.pipelines.marketdata import build_timeseries as md_pipeline


def _run_md_pipeline(params: dict, tmp_path) -> Context:
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
    runner = PipelineRunner()
    return runner.run(pipeline, ctx)


def test_build_timeseries_happy_path_with_snapshot(tmp_path) -> None:
    params = {
        "marketdata": {
            "provider": {"type": "synthetic", "seed": 123, "name": "SyntheticProvider"},
            "ids": ["FX.SPOT.EURUSD"],
            "start": "2026-01-01",
            "end": "2026-01-03",
            "freq": "D",
            "scenarios": 1,
            "snapshot": {"time": "last", "scenario_idx": 0},
        }
    }

    ctx = _run_md_pipeline(params, tmp_path)

    # Provider exists
    assert ctx.provider is not None

    # Core state keys exist
    assert Keys.MARKET_ID_STRINGS in ctx.state
    assert Keys.MARKET_IDS in ctx.state
    assert Keys.UNIVERSE in ctx.state
    assert Keys.REQUEST in ctx.state
    assert Keys.DATASET in ctx.state

    # Snapshot enabled => market exists
    assert Keys.MARKET in ctx.state
    assert Keys.MARKET_SNAPSHOT_SUMMARY in ctx.state

    dataset = ctx.get(Keys.DATASET)
    assert len(dataset.dates) > 0


def test_build_timeseries_no_snapshot_is_noop(tmp_path) -> None:
    params = {
        "marketdata": {
            "provider": {"type": "synthetic", "seed": 123},
            "ids": ["FX.SPOT.EURUSD"],
            "start": "2026-01-01",
            "end": "2026-01-03",
            "freq": "D",
            "scenarios": 1,
            # no "snapshot"
        }
    }

    ctx = _run_md_pipeline(params, tmp_path)

    assert Keys.DATASET in ctx.state
    assert Keys.MARKET not in ctx.state
    assert Keys.MARKET_SNAPSHOT_SUMMARY not in ctx.state


def test_build_timeseries_invalid_provider_type_raises(tmp_path) -> None:
    params = {
        "marketdata": {
            "provider": {"type": "definitely_not_real"},
            "ids": ["FX.SPOT.EURUSD"],
            "start": "2026-01-01",
            "end": "2026-01-02",
            "freq": "D",
            "scenarios": 1,
        }
    }

    with pytest.raises(StepError) as excinfo:
        _run_md_pipeline(params, tmp_path)

    # Original exception is preserved as __cause__
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "Unsupported provider type" in str(excinfo.value.__cause__)


def test_build_timeseries_snapshot_time_out_of_range_raises(tmp_path) -> None:
    params = {
        "marketdata": {
            "provider": {"type": "synthetic", "seed": 123},
            "ids": ["FX.SPOT.EURUSD"],
            "start": "2026-01-01",
            "end": "2026-01-02",
            "freq": "D",
            "scenarios": 1,
            "snapshot": {"time": 999, "scenario_idx": 0},
        }
    }

    with pytest.raises(StepError) as excinfo:
        _run_md_pipeline(params, tmp_path)

    assert isinstance(excinfo.value.__cause__, IndexError)
    assert "snapshot.time out of range" in str(excinfo.value.__cause__)
