"""
Tests for portfolio factor-universe resolution.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.rade_sr.core.exceptions import ConfigurationError
from src.rade_sr.portfolio.factor_universe import build_rules, factor_universe


@pytest.fixture
def fx_mapping() -> pd.DataFrame:
    return pd.DataFrame({
        "ccy": ["AED", "GBPJPY"],
        "fx_risk_factor_1": ["FX.SPOT.USD.AED", "FX.SPOT.USD.GBP"],
        "fx_risk_factor_2": ["FX.SPOT.USD.USD", "FX.SPOT.USD.JPY"],
        "ir_risk_factor_1": ["IR_CURVE_SWAP.AED", "IR_CURVE_SWAP.GBP"],
        "ir_risk_factor_2": ["IR_CURVE.RFR.USD.SOFR", "IR_CURVE.RFR.USD.JIBOR"],
    })


def _fx_rules(mapping: pd.DataFrame):
    return build_rules({
        "fx": {
            "source": "mapping",
            "mapping": mapping,
            "mapping_key_col": "ccy",
            "key_field": "ccy",
            "factor_cols": ["fx_risk_factor_1", "fx_risk_factor_2"],
            "dependency_cols": ["ir_risk_factor_1", "ir_risk_factor_2"],
        },
    })


def test_single_currency_aed(fx_mapping):
    attrs = pd.DataFrame({"AssetClass": ["fx"], "ccy": ["AED"]}, index=["t1"])
    assert factor_universe(attrs, _fx_rules(fx_mapping)) == sorted({
        "FX.SPOT.USD.AED", "FX.SPOT.USD.USD",
        "IR_CURVE_SWAP.AED", "IR_CURVE.RFR.USD.SOFR",
    })


def test_fx_spots_only(fx_mapping):
    attrs = pd.DataFrame({"AssetClass": ["fx"], "ccy": ["AED"]}, index=["t1"])
    assert factor_universe(
        attrs, _fx_rules(fx_mapping), include_dependencies=False,
    ) == ["FX.SPOT.USD.AED", "FX.SPOT.USD.USD"]


def test_attribute_source(fx_mapping):
    attrs = pd.DataFrame(
        {"AssetClass": ["rates", "fx"], "RiskFactor": ["IR_CURVE_SWAP.GBP", ""], "ccy": ["", "GBPJPY"]},
        index=["ir1", "fx1"],
    )
    rules = build_rules({
        "rates": {"source": "attribute", "factor_cols": ["RiskFactor"]},
        "fx": {
            "source": "mapping",
            "mapping": fx_mapping,
            "mapping_key_col": "ccy",
            "key_field": "ccy",
            "factor_cols": ["fx_risk_factor_1"],
        },
    })
    ids = factor_universe(attrs, rules)
    assert "IR_CURVE_SWAP.GBP" in ids
    assert "FX.SPOT.USD.GBP" in ids


def test_missing_source_raises():
    with pytest.raises(ConfigurationError):
        build_rules({"fx": {"factor_cols": ["x"]}})


def test_relative_mapping_path(tmp_path):
    csv = tmp_path / "fx.csv"
    csv.write_text(
        "ccy,fx_risk_factor_1,fx_risk_factor_2,ir_risk_factor_1,ir_risk_factor_2\n"
        "AED,FX.SPOT.USD.AED,,,\n"
    )
    rules = build_rules({
        "fx": {
            "source": "mapping",
            "mapping": "fx.csv",
            "key_field": "ccy",
            "factor_cols": ["fx_risk_factor_1"],
        },
    }, base_dir=tmp_path)
    attrs = pd.DataFrame({"AssetClass": ["fx"], "ccy": ["AED"]}, index=["t1"])
    assert factor_universe(attrs, rules) == ["FX.SPOT.USD.AED"]
