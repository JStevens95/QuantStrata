"""Round-trip tests: mock raw portfolio files -> Step 0 ingestion.

Validates that the generator emits the exact raw schema and that the normalizer
collapses it correctly (one row per trade, risk types as columns, AssetClass
'ALL' handled, NotionalSign derived, FX risk factors + dependencies inferred).
"""
import numpy as np
import pandas as pd
import pytest

from src.rade_sr.mock.portfolio import (
    RAW_ATTRIBUTE_COLUMNS,
    RISK_TYPE_SETS,
    generate_mock_portfolio,
)
from src.rade_sr.replication.ingest import (
    IngestConfig,
    ingest_portfolio,
    normalize_attributes,
)

N_TRADES = 400
N_SCENARIOS = 60


@pytest.fixture(scope="module")
def raw():
    attributes, pnl = generate_mock_portfolio(
        n_trades=N_TRADES, n_scenarios=N_SCENARIOS, seed=123,
    )
    return attributes, pnl


def test_raw_attribute_schema(raw):
    attributes, _ = raw
    assert list(attributes.columns) == RAW_ATTRIBUTE_COLUMNS
    # raw is exploded: strictly more rows than unique trades
    assert len(attributes) > attributes["PTSDealNumber"].nunique()
    assert attributes["PTSDealNumber"].nunique() == N_TRADES


def test_notional_rows_are_all_asset_class(raw):
    attributes, _ = raw
    notional = attributes[attributes["RiskType"] == "Notional"]
    assert (notional["AssetClass"] == "ALL").all()
    non_notional = attributes[attributes["RiskType"] != "Notional"]
    assert (non_notional["AssetClass"] == "FX").all()


def test_risk_types_present(raw):
    attributes, _ = raw
    assert set(attributes["RiskType"].unique()) == set(RISK_TYPE_SETS["FX"])


def test_pnl_schema(raw):
    _, pnl = raw
    assert "AssetClass" in pnl.columns
    scenario_cols = [c for c in pnl.columns if c != "AssetClass"]
    assert len(scenario_cols) == N_SCENARIOS
    # scenario columns are YYYYMMDD date strings
    assert all(len(c) == 8 and c.isdigit() for c in scenario_cols)
    assert len(pnl) == N_TRADES


def test_normalize_one_row_per_trade(raw):
    attributes, _ = raw
    norm = normalize_attributes(attributes)
    assert len(norm) == N_TRADES
    assert norm.index.is_unique
    # AssetClass resolved to the real class, never 'ALL'
    assert (norm["AssetClass"] == "FX").all()


def test_normalize_risk_pivot_and_sign(raw):
    attributes, _ = raw
    norm = normalize_attributes(attributes)
    for rt in RISK_TYPE_SETS["FX"]:
        assert rt in norm.columns
    assert set(np.unique(norm["NotionalSign"])).issubset({-1.0, 1.0})
    # signed notional sign matches direction
    assert np.allclose(np.sign(norm["SignedNotional"]), norm["NotionalSign"])


def test_risk_value_sum_across_curves(raw):
    """Pivot must sum the value across (trade, risk_type, curve) rows."""
    attributes, _ = raw
    norm = normalize_attributes(attributes)
    tid = norm.index[0]
    expected_fxpv = attributes[
        (attributes["PTSDealNumber"] == tid) & (attributes["RiskType"] == "FXPV")
    ]["Sum_RiskValuesUSD_Net"].sum()
    assert norm.loc[tid, "FXPV"] == pytest.approx(expected_fxpv, rel=1e-6)


def test_asset_config_and_dependencies(raw):
    attributes, pnl = raw
    inputs = ingest_portfolio(attributes, pnl)

    assert len(inputs.asset_config) >= 1
    for rf, cfg in inputs.asset_config.items():
        assert cfg["asset_class"] == "fx"
        assert len(rf) == 6
        assert cfg["base_ccy"] == rf[:3]
        assert cfg["quote_ccy"] == rf[3:6]
        deps = inputs.dependency_graph[rf]
        assert deps == [f"{rf[:3]}_IR", f"{rf[3:6]}_IR"]


def test_load_pnl_shape_and_alignment(raw):
    attributes, pnl = raw
    inputs = ingest_portfolio(attributes, pnl)
    assert inputs.target_pnl.shape == (N_TRADES, N_SCENARIOS)
    assert "AssetClass" not in inputs.target_pnl.columns
    # every priced trade has attributes
    assert set(inputs.target_pnl.index) == set(inputs.attributes.index)


def test_custom_ingest_config_columns(raw):
    """Schema is a contract: column names are overridable via IngestConfig."""
    attributes, _ = raw
    cfg = IngestConfig()
    norm = normalize_attributes(attributes, cfg)
    assert "expiry_years" in norm.columns
    assert (norm["expiry_years"] > 0).all()
