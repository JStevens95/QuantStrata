"""
Module is responsible for any functions which are concerned with standardising input data for ml models.
"""
from __future__ import annotations

from typing import Any
from sklearn import preprocessing

from src.rade_ml.validation.exceptions import UndefinedTransformerType


def get_transformer(transform_type: str) -> Any:
    """Get standard transformer object."""
    if transform_type.lower() == "standard":
        return preprocessing.StandardScaler()
    elif transform_type.lower() == "minmax":
        return preprocessing.MinMaxScaler()
    elif transform_type.lower() == "norm":
        return preprocessing.Normalizer()
    elif transform_type.lower() == "zscore":
        return preprocessing.ZScoreNormalizer()
    elif transform_type.lower() == "power":
        return preprocessing.PowerTransformer()
    elif transform_type.lower() == "robust":
        return preprocessing.RobustScaler()
    else:
        raise UndefinedTransformerType(f"Unknown transformer type {transform_type}")