"""Tests for portfolio risk-factor resolution."""
from __future__ import annotations

import pandas as pd
import pytest

from src.static_replication.config.factor_schema import (
    asset_class_from_block_key,
    parse_factor_rules,
)
from src.static_replication.core.exceptions import (
    ConfigurationError,
    FactorResolutionError,
)
from src.static_replication.portfolio.resolution.resolver import resolve_factors
from src.static_replication.portfolio.resolution.rules import USD_SOFR_FACTOR_ID


@pytest.fixture
def mapping_csv(tmp_path):
    path = tmp_path / "fx_mapping.csv"
    path.write_text(
        "currency,fx_risk_factor_1,fx_risk_factor_2,ir_risk_factor_1,ir_risk_factor_2\n"
        "EURUSD,FX.SPOT.USD.EUR,FX.SPOT.USD.USD,IR_CURVE_SWAP.EUR,IR_CURVE_SWAP.USD\n"
        "GBPUSD,FX.SPOT.USD.GBP,FX.SPOT.USD.USD,IR_CURVE_SWAP.GBP,IR_CURVE_SWAP.USD\n"
    )
    return path


def test_block_key_to_registry_class():
    assert asset_class_from_block_key("fx_config") == "FX"
    assert asset_class_from_block_key("ir_config") == "IR"


def test_fx_file_resolution_drops_usd_usd_leg(mapping_csv):
    factor_config = {
        "fx_config": {
            "source": "file",
            "mapping_file": str(mapping_csv),
            "mapping_key": "currency",
            "attrs_key": "UnderlyingOption",
            "factor_cols": ["fx_risk_factor_1", "fx_risk_factor_2"],
            "dependency_cols": ["ir_risk_factor_1", "ir_risk_factor_2"],
            "asset_class_of": {
                "ir_risk_factor_1": "IR",
                "ir_risk_factor_2": "IR",
            },
        },
    }
    attributes = pd.DataFrame(
        {
            "AssetClassCode": ["FX"],
            "UnderlyingOption": ["EURUSD"],
        },
        index=["trade-1"],
    )
    universe = resolve_factors(attributes, factor_config)

    assert "FX.SPOT.USD.EUR" in universe.specs
    assert "FX.SPOT.USD.USD" not in universe.specs
    assert "IR_CURVE_SWAP.EUR" in universe.specs
    assert universe.specs["FX.SPOT.USD.EUR"].asset_class == "FX"
    assert universe.specs["IR_CURVE_SWAP.EUR"].asset_class == "IR"
    assert universe.factors_by_trade["trade-1"] == ["FX.SPOT.USD.EUR"]
    eur = universe.specs["FX.SPOT.USD.EUR"]
    assert set(eur.dependencies) == {"IR_CURVE_SWAP.EUR", USD_SOFR_FACTOR_ID}


def test_cross_pair_per_leg_dependencies(tmp_path):
    mapping = tmp_path / "fx.csv"
    mapping.write_text(
        "currency,fx_risk_factor_1,fx_risk_factor_2,ir_risk_factor_1,ir_risk_factor_2\n"
        "EURCNH,FX.SPOT.USD.EUR,FX.SPOT.USD.CNH,IR_CURVE_SWAP.EUR,IR_CURVE_SWAP.CNH\n"
    )
    factor_config = {
        "fx_config": {
            "source": "file",
            "mapping_file": str(mapping),
            "mapping_key": "currency",
            "attrs_key": "UnderlyingOption",
            "factor_cols": ["fx_risk_factor_1", "fx_risk_factor_2"],
            "dependency_cols": ["ir_risk_factor_1", "ir_risk_factor_2"],
            "asset_class_of": {"ir_risk_factor_1": "IR", "ir_risk_factor_2": "IR"},
        },
    }
    attributes = pd.DataFrame(
        {"AssetClassCode": ["FX"], "UnderlyingOption": ["EURCNH"]},
        index=["t1"],
    )
    universe = resolve_factors(attributes, factor_config)

    eur = universe.specs["FX.SPOT.USD.EUR"]
    cnh = universe.specs["FX.SPOT.USD.CNH"]
    assert set(eur.dependencies) == {"IR_CURVE_SWAP.EUR", USD_SOFR_FACTOR_ID}
    assert set(cnh.dependencies) == {"IR_CURVE_SWAP.CNH", USD_SOFR_FACTOR_ID}
    assert eur.dependencies != cnh.dependencies


def test_ir_attribute_resolution():
    factor_config = {
        "ir_config": {
            "source": "attribute",
            "factor_cols": ["CurveCode"],
        },
    }
    attributes = pd.DataFrame(
        {"AssetClassCode": ["IR"], "CurveCode": ["IR_CURVE_SWAP.GBP"]},
        index=["ir-1"],
    )
    universe = resolve_factors(attributes, factor_config)

    assert universe.primary_ids == ["IR_CURVE_SWAP.GBP"]
    assert universe.specs["IR_CURVE_SWAP.GBP"].asset_class == "IR"


def test_missing_mapping_key_raises(mapping_csv):
    factor_config = {
        "fx_config": {
            "source": "file",
            "mapping_file": str(mapping_csv),
            "attrs_key": "UnderlyingOption",
            "factor_cols": ["fx_risk_factor_1"],
        },
    }
    attributes = pd.DataFrame(
        {"AssetClassCode": ["FX"], "UnderlyingOption": ["NOPE"]},
        index=["t1"],
    )
    with pytest.raises(FactorResolutionError, match="not found"):
        resolve_factors(attributes, factor_config)


def test_empty_factor_config_raises():
    with pytest.raises(ConfigurationError):
        parse_factor_rules({})
