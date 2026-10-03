"""End-to-end smoke test: mock portfolio + mock market data → artifacts."""
from __future__ import annotations

import pandas as pd

from src.rade_static_replication.adapters.market_data import MockMarketDataClient
from src.rade_static_replication.dev.mock.portfolio import MockPortfolioSource
from src.rade_static_replication.pipeline import Orchestrator, RunConfig
from src.rade_static_replication.qa import check_artifacts, check_market_data_client

N_SCEN = 64


def _run(tmp_path):
    run = RunConfig(
        portfolio_source=MockPortfolioSource(n_fx_trades=12, n_ir_trades=6, n_scenarios=N_SCEN),
        market_data_client=MockMarketDataClient(n_scenarios=N_SCEN, seed=3),
        output_dir=tmp_path,
        run_name="test_run",
    )
    return Orchestrator(run).run()


def test_pipeline_produces_aligned_artifacts(tmp_path):
    result = _run(tmp_path)
    assert result.n_clusters == 2
    assert {j.asset_class for j in result.jobs} == {"fx", "rates"}

    for job in result.jobs:
        elem, tgt = job.elementary_pnl, job.target_pnl
        assert elem.shape[0] == N_SCEN
        assert tgt.shape[0] == N_SCEN
        assert list(elem.index) == list(tgt.index)        # shared scenario axis
        assert elem.shape[1] > 0 and tgt.shape[1] > 0


def test_artifacts_pass_conformance(tmp_path):
    result = _run(tmp_path)
    report = check_artifacts(str(result.run_dir))
    assert report.is_ok, str(report)


def test_market_data_client_conformance():
    client = MockMarketDataClient(n_scenarios=N_SCEN)
    report = check_market_data_client(
        client, fx_factors=["FX.SPOT.USD.GBP"], ir_factors=["IR_CURVE_SWAP.GBP"],
        expected_n_scenarios=N_SCEN,
    )
    assert report.is_ok, str(report)


def test_from_mode_builds_mock_clients(tmp_path):
    from src.rade_static_replication.adapters import MockMarketDataClient as MMD
    from src.rade_static_replication.dev.mock.portfolio import MockPortfolioSource as MPS

    run = RunConfig.from_mode(
        "mock", output_dir=tmp_path, run_name="r",
        portfolio_kwargs=dict(n_fx_trades=8, n_ir_trades=4, n_scenarios=N_SCEN),
        market_kwargs=dict(n_scenarios=N_SCEN),
    )
    assert run.mode == "mock"
    assert isinstance(run.portfolio_source, MPS)
    assert isinstance(run.market_data_client, MMD)
    result = Orchestrator(run).run()
    assert result.n_clusters == 2


def test_from_mode_api_selects_sage_and_star(tmp_path):
    from src.rade_static_replication.adapters import SagePortfolioClient, StarMarketDataClient

    run = RunConfig.from_mode(
        "api", output_dir=tmp_path,
        portfolio_kwargs=dict(cob_date="2026-05-29"),
        market_kwargs=dict(cob_date="2026-05-29", n_scenarios=N_SCEN),
    )
    assert isinstance(run.portfolio_source, SagePortfolioClient)   # Sage = trade api
    assert isinstance(run.market_data_client, StarMarketDataClient)  # STAR = market-data api
