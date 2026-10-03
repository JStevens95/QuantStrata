"""Integration test: full pipeline against mock clients."""
import pickle

from src.rade_static_replication import run
from src.rade_static_replication.clients.mock import MockMarketDataClient, MockPortfolioClient
from src.rade_static_replication.config.schema import (
    AssetFactorRule,
    ClusteringConfig,
    ElementaryConfig,
    FactorResolutionConfig,
    OrchestratorConfig,
    OutputConfig,
)


def _config(tmp_path, mapping_csv) -> OrchestratorConfig:
    return OrchestratorConfig(
        cob_date="20240102",
        factors=FactorResolutionConfig(rules={
            "fx": AssetFactorRule(
                asset_class="fx", source="file", attrs_key="ccy", mapping_key="currency",
                factor_cols=["fx_risk_factor_1", "fx_risk_factor_2"],
                dependency_cols=["ir_risk_factor_1", "ir_risk_factor_2"],
                asset_class_of={
                    "ir_risk_factor_1": "rates", "ir_risk_factor_2": "rates",
                },
                mapping_file=str(mapping_csv),
            ),
            "rates": AssetFactorRule(
                asset_class="rates", source="attribute",
                factor_cols=["RatesFactor"],
            ),
        }),
        elementary=ElementaryConfig(grids={}),
        clustering=ClusteringConfig(keys=["AssetClass", "DeskName"]),
        output=OutputConfig(root=tmp_path, run_id="test_run"),
    )


def test_full_pipeline(tmp_path):
    mapping = (
        "currency,fx_risk_factor_1,fx_risk_factor_2,ir_risk_factor_1,ir_risk_factor_2\n"
        "EURUSD,FX.SPOT.USD.EUR,FX.SPOT.USD.USD,IR_CURVE_SWAP.EUR,IR_CURVE_SWAP.USD\n"
        "GBPUSD,FX.SPOT.USD.GBP,FX.SPOT.USD.USD,IR_CURVE_SWAP.GBP,IR_CURVE_SWAP.USD\n"
        "AUDUSD,FX.SPOT.USD.AUD,FX.SPOT.USD.USD,IR_CURVE_SWAP.AUD,IR_CURVE_SWAP.USD\n"
    )
    mapping_csv = tmp_path / "fx_mapping.csv"
    mapping_csv.write_text(mapping)

    ctx = run(_config(tmp_path, mapping_csv), MockPortfolioClient(), MockMarketDataClient())

    assert len(ctx.clusters.clusters) > 0
    assert ctx.elementary.n_trades > 0

    run_dir = tmp_path / "test_run"
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "jobs.pkl").exists()
    assert (run_dir / "config.snapshot.yaml").exists()

    with open(run_dir / "jobs.pkl", "rb") as fh:
        jobs = pickle.load(fh)
    assert len(jobs) == len(ctx.clusters.clusters)
    for entry in jobs:
        assert entry["n_scenarios"] >= 0
        assert (run_dir / "clusters" / entry["cluster_id"]).exists()
