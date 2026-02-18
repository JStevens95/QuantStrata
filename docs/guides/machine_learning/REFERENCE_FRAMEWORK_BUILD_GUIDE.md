# Machine Learning Framework — Build-From-Scratch Reference Guide

This document provides implementation phases and the full content of every file needed to replicate the `machine_learning` structure from scratch. **Use this as a reference—no files are created in your workspace by this guide.** Copy and adapt for your work environment.

---

## Implementation Phases Overview

| Phase | Scope | Key Deliverables | Est. Effort |
|-------|-------|------------------|-------------|
| **1** | Core + Config | Base classes, protocols, config dataclasses, types | 1–2 days |
| **2** | Data | Dataset, normalization, types, pricing/GNN data builders | 2–3 days |
| **3** | Models | Pricing MLP, GNN-RNN hybrid + layers | 3–4 days |
| **4** | Training + Pipelines | Trainer, TrainingManager, pipelines, callbacks | 2–3 days |
| **5** | Evaluation + Inference | Evaluator, metrics, model I/O, Predictor | 1–2 days |
| **6** | Registry + Tracking | Model registry, experiment tracking (MLflow/W&B) | 1–2 days |
| **7** | Tuning | Optuna search space, pruners, run_optuna_tuning | 1 day |
| **8** | Features | Feature schemas, registry, standardiser, GNN encoders | 2 days |
| **9** | Validation | Input validation, schema checks, deployment gates | 1–2 days |
| **10** | Monitoring | Baselines, drift detection, logger, alerts | 1–2 days |
| **11** | Ensemble | Keyed registry, routers, ClusterEnsembleWrapper | 2 days |

---

## Folder Structure

```
machine_learning/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py           # BaseModel, PricingModel, CalibrationModel, PortfolioModel
│   ├── config.py         # TrainingConfig, OptimizerConfig, EarlyStoppingConfig, etc.
│   ├── protocols.py      # Trainable protocol, KerasTrainableAdapter
│   ├── types.py          # TrainingConfig, TrainingResult, CheckpointInfo
│   ├── callbacks.py      # Custom Keras callbacks
│   └── tracking.py       # ExperimentTracker protocol, MLflow/W&B/InMemory
├── data/
│   ├── __init__.py
│   ├── dataset.py        # TFDataset, NormalizationStats
│   ├── types.py          # MLDataset, PricingFeatures, CalibrationFeatures
│   ├── common/
│   │   └── __init__.py
│   ├── pricing/
│   │   ├── __init__.py
│   │   └── build.py      # build_pricing_data
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── build.py      # build_calibration_dataset
│   └── gnn_rnn_hybrid/
│       ├── __init__.py
│       ├── build.py      # build_gnn_data
│       ├── dataset_utils.py
│       ├── portfolio_builder.py
│       └── synthetic.py
├── models/
│   ├── __init__.py
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── model.py      # MLPPricer
│   │   └── config.py
│   └── gnn_rnn_hybrid/
│       ├── __init__.py
│       ├── model.py      # HybridGnnRnn
│       ├── config.py
│       └── layers/
│           ├── __init__.py
│           ├── attention_layer.py
│           ├── fusion_layer.py
│           ├── gnn_layers.py
│           ├── projection_layer.py
│           └── rnn_layers.py
├── training/
│   ├── __init__.py
│   └── trainer.py        # Trainer, TrainingResult, fit_model
├── pipelines/
│   ├── __init__.py
│   ├── training.py       # run_training, TrainingLoop
│   ├── evaluation.py
│   ├── inference.py
│   └── tuning.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py      # Evaluator, EvaluationResult
│   └── metrics.py        # compute_metrics, PricingMetrics
├── inference/
│   ├── __init__.py
│   ├── model_io.py       # save_model, load_model, ModelArtifact
│   └── predictor.py      # Predictor, BatchPredictor
├── registry/
│   ├── __init__.py
│   └── registry.py       # ModelRegistry, ModelVersion, ModelStage
├── tuning/
│   ├── __init__.py
│   └── search_space.py   # SearchSpace, OptunaSearchSpace, run_optuna_tuning
├── utilities/
│   ├── __init__.py
│   ├── trade_attribute_encoder.py
│   └── trade_graph_builder.py
├── calibration/
│   ├── __init__.py
│   └── training_manager.py  # TrainingManager, TrainingConfiguration
│
│   # Hedge-fund additions (Phases 8–11)
├── features/
│   ├── __init__.py
│   ├── schema.py         # FeatureSchema, PricingFeatureSchema, GnnFeatureSchema
│   ├── registry.py       # FeatureRegistry, get_registry
│   ├── transforms/
│   │   ├── __init__.py
│   │   └── standardiser.py  # Standardiser (zscore, minmax)
│   └── gnn.py            # TradeAttributeEncoder, TradeGraphBuilder (or re-export from utilities)
├── validation/
│   ├── __init__.py
│   ├── base.py           # Validator protocol, ValidationResult
│   ├── inputs.py         # validate_features, validate_targets
│   ├── schema.py         # validate_against_schema
│   └── deployment_gates.py  # DeploymentGate, ThresholdGate
├── monitoring/
│   ├── __init__.py
│   ├── baselines.py      # save_baseline, load_baseline
│   ├── drift.py          # check_feature_drift, check_prediction_drift
│   ├── logger.py         # PredictionLogger
│   └── alerts.py         # AlertHandler, threshold alerting
└── ensemble/
    ├── __init__.py
    ├── registry.py       # KeyedModelRegistry
    ├── ensemble.py       # KeyedModelEnsemble
    └── routers/
        ├── __init__.py
        └── cluster.py    # ClusterRouter (key_for_trade, keys_for_scenarios)
```

---

## Phase 8: Features — File Contents

### `features/__init__.py`

```python
"""
Central feature definitions and transforms.

Eliminates training-serving skew by defining feature logic once and reusing
in both training and inference.
"""

from .schema import FeatureSchema, PricingFeatureSchema, GnnFeatureSchema
from .registry import FeatureRegistry, get_registry
from .transforms.standardiser import Standardiser

__all__ = [
    "FeatureSchema",
    "PricingFeatureSchema",
    "GnnFeatureSchema",
    "FeatureRegistry",
    "get_registry",
    "Standardiser",
]
```

---

### `features/schema.py`

```python
"""
Feature schemas: define what features exist and how they are computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureSchema:
    """
    Schema for a set of features used by a model.

    Attributes
    ----------
    name : str
        Identifier (e.g. "pricing", "gnn").
    feature_names : List[str]
        Ordered names of features.
    dtypes : List[str], optional
        Expected dtypes per feature.
    transform_ids : List[str], optional
        Transform id per feature ("zscore", "none", etc.).
    metadata : Dict[str, Any], optional
        Extra metadata.
    """

    name: str
    feature_names: List[str]
    dtypes: Optional[List[str]] = None
    transform_ids: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        self.dtypes = self.dtypes or ["float32"] * n
        self.transform_ids = self.transform_ids or ["none"] * n
        if len(self.dtypes) != n or len(self.transform_ids) != n:
            raise ValueError("dtypes/transform_ids length must match feature_names")

    def n_features(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_names": self.feature_names,
            "dtypes": self.dtypes,
            "transform_ids": self.transform_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSchema":
        return cls(
            name=d["name"],
            feature_names=d["feature_names"],
            dtypes=d.get("dtypes"),
            transform_ids=d.get("transform_ids"),
            metadata=d.get("metadata", {}),
        )


def PricingFeatureSchema(include_rate_foreign: bool = False) -> FeatureSchema:
    """Standard schema for option pricing: spot, strike, vol, rate, expiry, option_type [, rate_foreign]."""
    names = ["spot", "strike", "vol", "rate", "expiry", "option_type"]
    transforms = ["none", "none", "zscore", "zscore", "none", "none"]
    if include_rate_foreign:
        names.append("rate_foreign")
        transforms.append("zscore")
    return FeatureSchema(name="pricing", feature_names=names, transform_ids=transforms)


def GnnFeatureSchema() -> FeatureSchema:
    """Schema for GNN-RNN: moneyness, time_to_maturity, delta, vega, embeddings."""
    return FeatureSchema(
        name="gnn",
        feature_names=[
            "moneyness", "time_to_maturity", "normalised_delta", "normalised_vega",
            "product_type_embedding", "product_subtype_embedding", "underlying_risk_factors_embedding",
        ],
        transform_ids=["zscore", "zscore", "zscore", "zscore", "onehot", "onehot", "multilabel"],
    )
```

---

### `features/registry.py`

```python
"""
Feature registry: maps transform ids to compute functions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

TransformFn = Callable[..., Any]


class FeatureRegistry:
    """Registry of transform functions by id. Ensures training and inference use same logic."""

    def __init__(self) -> None:
        self._transforms: Dict[str, TransformFn] = {}

    def register(self, transform_id: str, fn: TransformFn) -> None:
        self._transforms[transform_id] = fn

    def get(self, transform_id: str) -> Optional[TransformFn]:
        return self._transforms.get(transform_id)

    def get_or_raise(self, transform_id: str) -> TransformFn:
        fn = self._transforms.get(transform_id)
        if fn is None:
            raise KeyError(f"Transform '{transform_id}' not registered")
        return fn


_REGISTRY: Optional[FeatureRegistry] = None


def get_registry() -> FeatureRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = FeatureRegistry()
        # Register defaults: none, zscore, minmax via Standardiser
    return _REGISTRY
```

---

### `features/transforms/__init__.py`

```python
"""Feature transforms: standardiser and pluggable transforms."""

from .standardiser import Standardiser

__all__ = ["Standardiser"]
```

---

### `features/transforms/standardiser.py`

```python
"""
Standardiser: z-score or min-max normalization.

Fittable (fit/transform) and serialisable (to_dict/from_dict).
Use for feature and target scaling in both training and inference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class Standardiser:
    """
    Normalise data using z-score or min-max.

    Attributes
    ----------
    method : str
        'zscore' or 'minmax'.
    mean_, std_ : np.ndarray, optional
        Fitted statistics (zscore).
    min_val_, max_val_ : np.ndarray, optional
        Fitted min/max (minmax).
    """

    def __init__(self, method: str = "zscore") -> None:
        if method not in ("zscore", "minmax"):
            raise ValueError("method must be 'zscore' or 'minmax'")
        self.method = method
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_val_: Optional[np.ndarray] = None
        self.max_val_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "Standardiser":
        """Compute statistics from data."""
        data = np.asarray(data)
        if self.method == "zscore":
            self.mean_ = np.mean(data, axis=0)
            self.std_ = np.std(data, axis=0) + 1e-8
        else:
            self.min_val_ = np.min(data, axis=0)
            self.max_val_ = np.max(data, axis=0)
            self.max_val_ = np.where(
                self.max_val_ - self.min_val_ < 1e-8, 1.0, self.max_val_ - self.min_val_
            )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization using fitted stats."""
        data = np.asarray(data)
        if self.method == "zscore":
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Call fit() before transform()")
            return (data - self.mean_) / self.std_
        if self.min_val_ is None or self.max_val_ is None:
            raise RuntimeError("Call fit() before transform()")
        return (data - self.min_val_) / self.max_val_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        data = np.asarray(data)
        if self.method == "zscore":
            return data * self.std_ + self.mean_
        return data * self.max_val_ + self.min_val_

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for storage with model artifacts."""
        out: Dict[str, Any] = {"method": self.method}
        if self.method == "zscore":
            out["mean"] = self.mean_.tolist() if self.mean_ is not None else None
            out["std"] = self.std_.tolist() if self.std_ is not None else None
        else:
            out["min_val"] = self.min_val_.tolist() if self.min_val_ is not None else None
            out["max_val"] = self.max_val_.tolist() if self.max_val_ is not None else None
        return out

    def from_dict(self, d: Dict[str, Any]) -> "Standardiser":
        """Load from serialised dict."""
        self.method = d["method"]
        if self.method == "zscore":
            self.mean_ = np.array(d["mean"]) if d.get("mean") else None
            self.std_ = np.array(d["std"]) if d.get("std") else None
        else:
            self.min_val_ = np.array(d["min_val"]) if d.get("min_val") else None
            self.max_val_ = np.array(d["max_val"]) if d.get("max_val") else None
        return self
```

---

## Phase 9: Validation — File Contents

### `validation/__init__.py`

```python
"""Input validation and deployment gates."""

from .base import ValidationResult, Validator
from .inputs import validate_features, validate_targets
from .schema import validate_against_schema
from .deployment_gates import DeploymentGate, ThresholdGate

__all__ = [
    "ValidationResult",
    "Validator",
    "validate_features",
    "validate_targets",
    "validate_against_schema",
    "DeploymentGate",
    "ThresholdGate",
]
```

---

### `validation/base.py`

```python
"""
Base validation types: ValidationResult and Validator protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable


@dataclass
class ValidationResult:
    """
    Result of a validation check.

    Attributes
    ----------
    passed : bool
        True if validation passed.
    messages : List[str]
        Error or info messages.
    details : dict, optional
        Extra details (e.g. which indices failed).
    """

    passed: bool
    messages: List[str]
    details: Any = None

    def __bool__(self) -> bool:
        return self.passed


@runtime_checkable
class Validator(Protocol):
    """Protocol for validators."""

    def validate(self, data: Any, **kwargs: Any) -> ValidationResult:
        ...
```

---

### `validation/inputs.py`

```python
"""
Input validation: features and targets.
"""

from __future__ import annotations

import numpy as np

from .base import ValidationResult


def validate_features(
    features: np.ndarray,
    min_samples: int = 1,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> ValidationResult:
    """
    Validate feature array: shape, dtypes, NaN, Inf.

    Returns
    -------
    ValidationResult
    """
    messages: list = []
    if not isinstance(features, np.ndarray):
        return ValidationResult(False, ["features must be np.ndarray"])

    if features.size == 0:
        messages.append("features is empty")

    if features.shape[0] < min_samples:
        messages.append(f"n_samples {features.shape[0]} < min_samples {min_samples}")

    if not allow_nan and np.any(np.isnan(features)):
        messages.append("features contains NaN")

    if not allow_inf and np.any(np.isinf(features)):
        messages.append("features contains Inf")

    return ValidationResult(passed=len(messages) == 0, messages=messages)


def validate_targets(
    targets: np.ndarray,
    min_samples: int = 1,
    allow_nan: bool = False,
) -> ValidationResult:
    """Validate target array."""
    messages: list = []
    if not isinstance(targets, np.ndarray):
        return ValidationResult(False, ["targets must be np.ndarray"])
    if targets.size == 0:
        messages.append("targets is empty")
    if targets.shape[0] < min_samples:
        messages.append(f"n_samples {targets.shape[0]} < min_samples {min_samples}")
    if not allow_nan and np.any(np.isnan(targets)):
        messages.append("targets contains NaN")
    return ValidationResult(passed=len(messages) == 0, messages=messages)
```

---

### `validation/schema.py`

```python
"""
Schema validation: check data matches a FeatureSchema.
"""

from __future__ import annotations

import numpy as np

from .base import ValidationResult
from features.schema import FeatureSchema


def validate_against_schema(
    features: np.ndarray,
    schema: FeatureSchema,
) -> ValidationResult:
    """
    Check that features array has correct number of columns and matches schema.
    """
    messages: list = []
    n_cols = schema.n_features()
    if features.ndim != 2:
        return ValidationResult(False, ["features must be 2D (n_samples, n_features)"])
    if features.shape[1] != n_cols:
        messages.append(f"Expected {n_cols} features, got {features.shape[1]}")
    return ValidationResult(passed=len(messages) == 0, messages=messages)
```

---

### `validation/deployment_gates.py`

```python
"""
Deployment gates: block model promotion if metrics fail thresholds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GateRule:
    """Single rule: metric_name, operator, threshold."""

    metric_name: str
    operator: str  # ">=", "<=", ">", "<", "=="
    threshold: float

    def check(self, value: float) -> bool:
        if self.operator == ">=":
            return value >= self.threshold
        if self.operator == "<=":
            return value <= self.threshold
        if self.operator == ">":
            return value > self.threshold
        if self.operator == "<":
            return value < self.threshold
        if self.operator == "==":
            return value == self.threshold
        return False


class DeploymentGate(ABC):
    """Abstract gate: pass/fail based on evaluation result."""

    @abstractmethod
    def pass_(self, eval_result: Dict[str, Any]) -> bool:
        """Return True if model passes gate."""
        ...


class ThresholdGate(DeploymentGate):
    """
    Gate with list of rules. All must pass.
    Example: MAPE < 0.05, r2 >= 0.9.
    """

    def __init__(self, rules: List[GateRule]) -> None:
        self.rules = rules

    def pass_(self, eval_result: Dict[str, Any]) -> bool:
        metrics = eval_result.get("metrics", eval_result)
        for rule in self.rules:
            val = metrics.get(rule.metric_name)
            if val is None:
                return False
            if not rule.check(float(val)):
                return False
        return True
```

---

## Phase 10: Monitoring — File Contents

### `monitoring/__init__.py`

```python
"""Drift detection and performance baselines."""

from .baselines import BaselineStore, save_baseline, load_baseline
from .drift import DriftChecker, check_feature_drift
from .logger import PredictionLogger
from .alerts import AlertHandler

__all__ = [
    "BaselineStore",
    "save_baseline",
    "load_baseline",
    "DriftChecker",
    "check_feature_drift",
    "PredictionLogger",
    "AlertHandler",
]
```

---

### `monitoring/baselines.py`

```python
"""
Baseline storage: save/load feature and prediction statistics from training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BaselineStore:
    """
    Stores mean, std, percentiles for features and predictions.
    Used by drift checker to compare live data to training baseline.
    """

    feature_mean: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None
    prediction_mean: Optional[float] = None
    prediction_std: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist() if self.feature_mean is not None else None,
            "feature_std": self.feature_std.tolist() if self.feature_std is not None else None,
            "prediction_mean": self.prediction_mean,
            "prediction_std": self.prediction_std,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaselineStore":
        return cls(
            feature_mean=np.array(d["feature_mean"]) if d.get("feature_mean") else None,
            feature_std=np.array(d["feature_std"]) if d.get("feature_std") else None,
            prediction_mean=d.get("prediction_mean"),
            prediction_std=d.get("prediction_std"),
            metadata=d.get("metadata", {}),
        )


def save_baseline(baseline: BaselineStore, path: str | Path) -> None:
    """Save baseline to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline.to_dict(), f, indent=2)


def load_baseline(path: str | Path) -> BaselineStore:
    """Load baseline from JSON."""
    with open(path, "r") as f:
        return BaselineStore.from_dict(json.load(f))
```

---

### `monitoring/drift.py`

```python
"""
Drift detection: compare live features/predictions to baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .baselines import BaselineStore


@dataclass
class DriftResult:
    """Result of drift check."""

    exceeded: bool
    message: str
    details: Optional[dict] = None


def check_feature_drift(
    features: np.ndarray,
    baseline: BaselineStore,
    n_std_threshold: float = 2.0,
) -> DriftResult:
    """
    Check if live feature mean/std deviates from baseline beyond n_std.
    Simple heuristic: alert if (live_mean - baseline_mean) / baseline_std > n_std.
    """
    if baseline.feature_mean is None or baseline.feature_std is None:
        return DriftResult(False, "No feature baseline")

    live_mean = np.mean(features, axis=0)
    diff = np.abs(live_mean - baseline.feature_mean)
    z = diff / (baseline.feature_std + 1e-8)
    max_z = np.max(z)
    exceeded = max_z > n_std_threshold
    return DriftResult(
        exceeded=exceeded,
        message=f"max_z={max_z:.3f} (threshold={n_std_threshold})",
        details={"max_z": float(max_z), "threshold": n_std_threshold},
    )
```

---

### `monitoring/logger.py`

```python
"""
Prediction logger: log predictions + context for analysis and audit.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Log predictions and metadata (e.g. model_id, timestamp, batch_id).
    Enables post-hoc analysis and performance monitoring.
    """

    def __init__(self, log_dir: Optional[str | Path] = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else None

    def log(
        self,
        predictions: Any,
        model_id: str = "",
        batch_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "batch_id": batch_id,
            "n_predictions": len(predictions) if hasattr(predictions, "__len__") else None,
            "metadata": metadata or {},
        }
        logger.debug("Prediction log: %s", entry)
        if self.log_dir:
            path = self.log_dir / f"pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{batch_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(entry, f, indent=2)
```

---

### `monitoring/alerts.py`

```python
"""
Alert handling when drift or gate failures occur.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class AlertHandler:
    """
    Dispatch alerts (e.g. log, email, webhook).
    Default: log only. Register callbacks for custom handling.
    """

    def __init__(self) -> None:
        self._handlers: List[Callable[[str, dict], None]] = []

    def add_handler(self, fn: Callable[[str, dict], None]) -> None:
        self._handlers.append(fn)

    def alert(self, message: str, details: Optional[dict] = None) -> None:
        details = details or {}
        logger.warning("ALERT: %s | %s", message, details)
        for fn in self._handlers:
            try:
                fn(message, details)
            except Exception as e:
                logger.exception("Alert handler failed: %s", e)
```

---

## Phase 11: Ensemble — File Contents

### `ensemble/__init__.py`

```python
"""Multi-model orchestration for per-cluster models."""

from .registry import KeyedModelRegistry
from .ensemble import KeyedModelEnsemble
from .routers.cluster import ClusterRouter

__all__ = ["KeyedModelRegistry", "KeyedModelEnsemble", "ClusterRouter"]
```

---

### `ensemble/registry.py`

```python
"""
Keyed model registry: maps key (e.g. cluster_id) to model path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class KeyedModelRegistry:
    """
    Registry mapping key -> model_path.
    Used for per-cluster models: cluster_0 -> path/to/model_0, etc.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self._registry: Dict[str, str] = {}

    def register(self, key: str, model_path: str | Path) -> None:
        self._registry[key] = str(model_path)

    def get_path(self, key: str) -> Optional[str]:
        return self._registry.get(key)

    def get_path_or_raise(self, key: str) -> str:
        path = self._registry.get(key)
        if path is None:
            raise KeyError(f"No model registered for key '{key}'")
        return path

    def keys(self) -> list:
        return list(self._registry.keys())
```

---

### `ensemble/ensemble.py`

```python
"""
Keyed model ensemble: route inputs to correct model by key.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class KeyedModelEnsemble:
    """
    Ensemble that routes each input to a model identified by router key.

    Attributes
    ----------
    registry : KeyedModelRegistry
        Maps key -> model path.
    router : callable
        (input) -> key or (inputs) -> keys.
    loader : callable
        (path) -> model. Default: load from path.
    predictor : callable
        (model, input) -> prediction.
    """

    def __init__(
        self,
        registry: "KeyedModelRegistry",
        router: Callable[..., str | List[str]],
        loader: Optional[Callable[[str], Any]] = None,
        predictor: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self.registry = registry
        self.router = router
        self.loader = loader or (lambda p: None)  # Caller provides
        self.predictor = predictor or (lambda m, x: m.predict(x))
        self._models: Dict[str, Any] = {}

    def get_model(self, key: str) -> Any:
        """Load and cache model by key."""
        if key not in self._models:
            path = self.registry.get_path_or_raise(key)
            self._models[key] = self.loader(path)
        return self._models[key]

    def predict_single(self, inp: Any) -> Any:
        """Route single input to model and return prediction."""
        key = self.router(inp)
        model = self.get_model(key)
        return self.predictor(model, inp)

    def predict_batch(self, inputs: List[Any]) -> List[Any]:
        """Route each input and aggregate predictions."""
        keys = self.router(inputs)
        if isinstance(keys, str):
            keys = [keys] * len(inputs)
        results = []
        for k, inp in zip(keys, inputs):
            model = self.get_model(k)
            results.append(self.predictor(model, inp))
        return results
```

---

### `ensemble/routers/__init__.py`

```python
"""Routers for ensemble: map inputs to model keys."""

from .cluster import ClusterRouter

__all__ = ["ClusterRouter"]
```

---

### `ensemble/routers/cluster.py`

```python
"""
Cluster router: route trades/scenarios by cluster_id.
"""

from __future__ import annotations

from typing import Any, Callable, List, Union


class ClusterRouter:
    """
    Routes trades and scenarios to cluster ids.
    Use with KeyedModelEnsemble for per-cluster GNN-RNN models.

    Attributes
    ----------
    key_for_trade_fn : callable
        (trade) -> cluster_id.
    keys_for_scenarios_fn : callable
        (scenarios) -> list of cluster_ids (one per scenario or aggregated).
    """

    def __init__(
        self,
        key_for_trade_fn: Callable[[Any], str],
        keys_for_scenarios_fn: Callable[[Any], List[str]],
    ) -> None:
        self.key_for_trade_fn = key_for_trade_fn
        self.keys_for_scenarios_fn = keys_for_scenarios_fn

    def key_for_trade(self, trade: Any) -> str:
        return self.key_for_trade_fn(trade)

    def keys_for_scenarios(self, scenarios: Any) -> List[str]:
        return self.keys_for_scenarios_fn(scenarios)
```

---

## Summary

- **Phases 1–7** map to the existing QuantStrata `machine_learning` layout (core, data, models, training, evaluation, inference, registry, tuning, utilities, calibration).
- **Phases 8–11** add the hedge-fund–style features, validation, monitoring, and ensemble modules.
- Adjust import paths (e.g. `src.machine_learning` vs `machine_learning`) to match your package structure.
- The TradeAttributeEncoder and TradeGraphBuilder can live in `utilities/` and be re-exported or wrapped by `features/gnn.py` as needed.
