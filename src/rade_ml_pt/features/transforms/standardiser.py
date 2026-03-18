"""
Module is responsible for any functions which are concerned with standardising input data for ml models.

Available transform types:
    - ``none``             : identity (no transformation).
    - ``standard``         : sklearn StandardScaler (mean=0, std=1 per column).
    - ``minmax``           : sklearn MinMaxScaler ([0, 1] per column).
    - ``norm``             : sklearn Normalizer (unit-norm rows).
    - ``zscore``           : alias for ``standard``.
    - ``power``            : sklearn PowerTransformer (Yeo-Johnson per column).
    - ``robust``           : sklearn RobustScaler (median / IQR per column).
    - ``signed_log``       : sign-preserving log1p compression (good for fat-tailed PnL).
    - ``signed_log_standard`` : signed_log followed by StandardScaler (compression + centering).
    - ``signed_log_robust``   : signed_log followed by RobustScaler (compression + robust centering).
"""
from __future__ import annotations

import numpy as np

from typing import Any
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from src.rade_ml_pt.validation.exceptions import UndefinedTransformerType


def _passthrough(x: Any) -> Any:
    return x


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """Sign-preserving log compression: sign(x) * log1p(|x|).

    Compresses heavy-tailed PnL distributions (e.g. exotic trades with
    barrier knock-in spikes) while preserving direction. A 500k spike
    becomes ~13.1 while a 10k value becomes ~9.2, shrinking the ratio
    from 50:1 to ~1.4:1 so the model can learn both regimes.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_expm1(x: np.ndarray) -> np.ndarray:
    """Inverse of _signed_log1p: sign(x) * expm1(|x|)."""
    return np.sign(x) * np.expm1(np.abs(x))


class SignedLogTransformer(BaseEstimator, TransformerMixin):
    """Stateless sign-preserving log1p / expm1 transformer.

    Implements the sklearn transformer interface (fit/transform/inverse_transform)
    so it composes cleanly in a ``sklearn.pipeline.Pipeline`` with StandardScaler
    or RobustScaler.
    """

    def fit(self, X: np.ndarray, y: Any = None) -> "SignedLogTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _signed_log1p(np.asarray(X, dtype=np.float64))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return _signed_expm1(np.asarray(X, dtype=np.float64))


def get_transformer(transform_type: str) -> Any:
    """Get standard transformer object.

    :param transform_type: one of ``none``, ``standard``, ``minmax``, ``norm``,
        ``zscore``, ``power``, ``robust``, ``signed_log``, ``signed_log_standard``,
        ``signed_log_robust``.
    :returns: sklearn-compatible transformer with fit / transform / inverse_transform.
    """
    name = transform_type.lower()

    if name == "none":
        return FunctionTransformer(func=_passthrough, inverse_func=_passthrough)
    elif name in ("standard", "zscore"):
        return preprocessing.StandardScaler()
    elif name == "minmax":
        return preprocessing.MinMaxScaler()
    elif name == "norm":
        return preprocessing.Normalizer()
    elif name == "power":
        return preprocessing.PowerTransformer()
    elif name == "robust":
        return preprocessing.RobustScaler()
    elif name == "signed_log":
        return SignedLogTransformer()
    elif name == "signed_log_standard":
        return Pipeline([
            ("signed_log", SignedLogTransformer()),
            ("standard", preprocessing.StandardScaler()),
        ])
    elif name == "signed_log_robust":
        return Pipeline([
            ("signed_log", SignedLogTransformer()),
            ("robust", preprocessing.RobustScaler()),
        ])
    else:
        raise UndefinedTransformerType(f"Unknown transformer type {transform_type}")
