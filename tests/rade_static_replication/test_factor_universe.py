"""Tests for config-driven risk-factor resolution."""
from __future__ import annotations

import pandas as pd
import pytest

from src.rade_static_replication.core.exceptions import ConfigurationError
from src.rade_static_replication.portfolio.factor_universe import (
    build_rules,
    factor_universe,
    resolve_factor_map,
)

FX_MAPPING = pd.DataFrame({
    "ccy": ["AED", "GBPJPY"],
    "fx_risk_factor_1": ["FX.SPOT.USD.AED", "FX.SPOT.USD.GBP"],
    "fx_risk_factor_2": ["FX.SPOT.USD.USD", "FX.SPOT.USD.JPY"],
    "ir_risk_factor_1": ["IR_CURVE_SWAP.AED", "IR_CURVE_SWAP.GBP"],
    "ir_risk_factor_2": ["IR_CURVE.RFR.USD.SOFR", "IR_CURVE.RFR.USD.JIBOR"],
})


def _fx_rule(**overrides):
    spec = {
        "fx": {
            "source": "mapping",
            "mapping": FX_MAPPING,
            "mapping_key_col": "ccy",
            "key_field": "ccy",
            "factor_cols": ["fx_risk_factor_1", "fx_risk_factor_2"],
            "dependency_cols": ["ir_risk_factor_1", "ir_risk_factor_2"],
        }
    }
    spec["fx"].update(overrides)
    return build_rules(spec)


def test_single_currency_aed_resolves_two_usd_spots():
    attrs = pd.DataFrame({"AssetClass": ["fx"], "ccy": ["AED"]}, index=["T1"])
    fmap = resolve_factor_map(attrs, _fx_rule())
    assert fmap["FX.SPOT.USD.AED"] == "fx"
    assert fmap["FX.SPOT.USD.USD"] == "fx"
    assert fmap["IR_CURVE_SWAP.AED"] == "rates"  # dependency classified as rates


def test_cross_currency_gbpjpy_resolves_both_legs():
    attrs = pd.DataFrame({"AssetClass": ["fx"], "ccy": ["GBPJPY"]}, index=["T1"])
    ids = factor_universe(attrs, _fx_rule(), include_dependencies=False)
    assert ids == ["FX.SPOT.USD.GBP", "FX.SPOT.USD.JPY"]


def test_attribute_source_reads_row_directly():
    rules = build_rules({"rates": {"source": "attribute", "factor_cols": ["RiskFactor"]}})
    attrs = pd.DataFrame(
        {"AssetClass": ["rates", "rates"], "RiskFactor": ["IR_CURVE_SWAP.GBP", "IR_CURVE_SWAP.GBP"]},
        index=["T1", "T2"],
    )
    assert factor_universe(attrs, rules) == ["IR_CURVE_SWAP.GBP"]


def test_missing_source_raises():
    with pytest.raises(ConfigurationError):
        build_rules({"fx": {"factor_cols": ["x"]}})


def test_missing_column_raises():
    attrs = pd.DataFrame({"AssetClass": ["fx"]}, index=["T1"])  # no ccy
    with pytest.raises(ConfigurationError):
        resolve_factor_map(attrs, _fx_rule())
