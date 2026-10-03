"""
End-to-end test: orchestrated mock run → rade_ml_pt-shaped artifacts.

Locks the consumer contract so it cannot silently drift:
  - PnL files are [scenarios x trade-ids] parquet
  - elementary column ids carry the 'UND|TYPE|...' shape
  - attribute files are dict[str, list] with trade_id + yrs_to_maturity
  - attribute length matches PnL columns; scenario rows align
"""
from __future__ import annotations

import pickle

import pandas as pd
import pytest

from src.rade_sr.contracts import (
    check_artifacts_roundtrip,
    check_market_data_client,
    check_portfolio_source,
)
from src.rade_sr.replication.orchestrator import run_from_source
from src.rade_sr.sources import MockMarketDataClient, MockPortfolioSource

N_SCENARIOS = 48
TRADE_CONFIG = {
    "maturities": [0.25, 1.0],
    "fx": {"strike_pcts": [0.95, 1.00, 1.05], "include_digitals": True},
}


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("rade_sr_artifacts")
    result = run_from_source(
        portfolio_source=MockPortfolioSource(
            n_trades=120, n_scenarios=N_SCENARIOS, seed=7,
        ),
        market_data_client=MockMarketDataClient(n_scenarios=N_SCENARIOS, seed=11),
        trade_config=TRADE_CONFIG,
        max_workers=1,
        output_dir=str(out_dir),
    )
    return result, str(out_dir)


def test_jobs_and_manifest_written(run_result):
    result, out_dir = run_result
    assert result.jobs, "no jobs produced"
    assert result.jobs_manifest_path is not None
    with open(result.jobs_manifest_path, "rb") as f:
        manifest = pickle.load(f)
    assert len(manifest) == len(result.jobs)
    info = manifest[0]["cluster_info"]
    for key in (
        "target_pnl_path", "elementary_pnl_path",
        "target_attribs_path", "elementary_attribs_path",
    ):
        assert key in info


def test_elementary_pnl_orientation_and_ids(run_result):
    _, out_dir = run_result
    with open(f"{out_dir}/jobs.pkl", "rb") as f:
        manifest = pickle.load(f)
    info = manifest[0]["cluster_info"]
    elem = pd.read_parquet(info["elementary_pnl_path"])
    # rows = scenarios, columns = trades
    assert elem.shape[0] == N_SCENARIOS
    assert elem.shape[1] > 0
    assert "|" in str(elem.columns[0])  # UND|TYPE|... format


def test_attributes_are_dict_of_lists(run_result):
    _, out_dir = run_result
    with open(f"{out_dir}/jobs.pkl", "rb") as f:
        manifest = pickle.load(f)
    info = manifest[0]["cluster_info"]
    with open(info["elementary_attribs_path"], "rb") as f:
        elem_attr = pickle.load(f)
    elem = pd.read_parquet(info["elementary_pnl_path"])
    assert isinstance(elem_attr, dict)
    assert "trade_id" in elem_attr and "yrs_to_maturity" in elem_attr
    assert len(elem_attr["trade_id"]) == elem.shape[1]
    assert all(isinstance(v, list) for v in elem_attr.values())


def test_conformance_harness_passes(run_result):
    result, out_dir = run_result
    src = MockPortfolioSource(n_trades=60, n_scenarios=N_SCENARIOS, seed=7)
    client = MockMarketDataClient(n_scenarios=N_SCENARIOS, seed=11)
    assert check_portfolio_source(src).passed
    assert check_market_data_client(client).passed
    assert check_artifacts_roundtrip(out_dir).passed
