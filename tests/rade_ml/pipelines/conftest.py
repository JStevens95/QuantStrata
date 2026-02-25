"""Shared fixtures for rade_ml pipeline tests."""
import numpy as np
import pandas as pd
import pytest


def _make_trade_attrs(
    trade_ids: list,
    *,
    product_type: str = "vanilla_option",
    product_subtype: str = "european",
    trade_type: str = "option",
) -> dict:
    """Create attribute dict for trades (elementary or target)."""
    n = len(trade_ids)
    return {
        "trade_id": trade_ids,
        "moneyness": [1.0] * n,
        "yrs_to_maturity": [0.5] * n,
        "delta": [0.5] * n,
        "vega": [0.1] * n,
        "product_type": [product_type] * n,
        "product_subtype": [product_subtype] * n,
        "trade_type": [trade_type] * n,
        "underlying_risk_factors": [["FX"] for _ in range(n)],
    }


@pytest.fixture
def hybrid_gnn_rnn_synthetic_job(tmp_path):
    """
    Create synthetic PnL and attributes for Hybrid GNN-RNN pipeline testing.

    Writes temp files and returns a job dict with cluster_info paths.
    Trade IDs use format: UNDERLYING|PRODUCT_TYPE|id for dimension_reduction.
    """
    np.random.seed(42)
    n_scenarios = 60
    n_elementary = 6
    n_target = 2

    elem_ids = [
        "EURUSD|vanilla_option|1",
        "EURUSD|vanilla_option|2",
        "EURUSD|forward|1",
        "GBPUSD|vanilla_option|1",
        "GBPUSD|forward|1",
        "GBPUSD|swap|1",
    ]
    tgt_ids = ["EURUSD|target|1", "GBPUSD|target|1"]

    # PnL: [scenarios x trades]
    elem_pnl = pd.DataFrame(
        np.random.randn(n_scenarios, n_elementary).astype(np.float32) * 0.01,
        columns=elem_ids,
        index=pd.RangeIndex(n_scenarios),
    )
    tgt_pnl = pd.DataFrame(
        np.random.randn(n_scenarios, n_target).astype(np.float32) * 0.01,
        columns=tgt_ids,
        index=pd.RangeIndex(n_scenarios),
    )

    elem_attrs = _make_trade_attrs(elem_ids)
    tgt_attrs = _make_trade_attrs(tgt_ids)

    # Write to temp files (pickle for simplicity)
    from src.rade_ml.data.io import CacheLoader

    base = tmp_path / "hybrid_gnn_rnn_test"
    base.mkdir(parents=True, exist_ok=True)

    paths = {
        "elementary_pnl_path": str(base / "elem_pnl.pkl"),
        "target_pnl_path": str(base / "tgt_pnl.pkl"),
        "elementary_attribs_path": str(base / "elem_attrs.pkl"),
        "target_attribs_path": str(base / "tgt_attrs.pkl"),
    }
    CacheLoader.save_data(elem_pnl, paths["elementary_pnl_path"])
    CacheLoader.save_data(tgt_pnl, paths["target_pnl_path"])
    CacheLoader.save_data(elem_attrs, paths["elementary_attribs_path"])
    CacheLoader.save_data(tgt_attrs, paths["target_attribs_path"])

    return {
        "cluster_info": paths,
    }
