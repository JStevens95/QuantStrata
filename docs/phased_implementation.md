# Phased Implementation: `machine_learning_pt/` (PyTorch)

## Overview

This document details the creation of `src/machine_learning_pt/`, a PyTorch-native ML
framework that mirrors the structure and functionality of the existing TensorFlow
framework at `src/machine_learning/`. The two frameworks are fully independent — they
share no imports, no runtime state, and no base classes. They do share the same
*design philosophy*, directory layout, config schemas, and result types so that a
user familiar with one can immediately navigate the other.

### Design Principles

1. **Mirror structure** — every directory and public API in `machine_learning/` has a
   corresponding counterpart in `machine_learning_pt/`.
2. **Independence** — no cross-imports between the two frameworks. Each has its own
   `__init__.py`, its own base classes, its own training loop.
3. **Shared contracts** — both frameworks produce the same result dataclasses
   (`TrainingResult`, `EvaluationResult`) and consume the same config dataclasses
   (`TrainingConfig`, `ModelConfig`). These pure-Python types live in a shared
   location or are duplicated identically.
4. **Idiomatic PyTorch** — the PyTorch framework should feel native to PyTorch users
   (explicit training loops, `nn.Module`, `DataLoader`), not a Keras wrapper.

### Directory Layout

```
src/machine_learning_pt/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py              # BaseModel(nn.Module), PricingModel, etc.
│   ├── config.py            # Shared configs with PT-native build() methods
│   ├── device.py            # Device selection utility
│   ├── protocols.py         # Trainable protocol + TorchTrainableAdapter
│   ├── tracking.py          # Copied verbatim (framework-agnostic)
│   └── types.py             # Copied verbatim (framework-agnostic)
├── data/
│   ├── __init__.py
│   ├── dataset.py           # DictDataset, build_dataloader, SyntheticData, generators
│   ├── result.py            # Copied verbatim (framework-agnostic)
│   ├── manifest.py          # Copied verbatim (framework-agnostic)
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── build.py         # build_pricing_data() → DataLoader
│   │   └── config.py        # Copied verbatim
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── build.py         # build_calibration_data() → DataLoader
│   │   └── config.py        # Copied verbatim
│   └── gnn_rnn_hybrid/
│       ├── __init__.py
│       ├── build.py         # build_gnn_data() → DataLoader
│       ├── config.py        # Copied verbatim
│       ├── synthetic.py     # Copied verbatim (pure NumPy)
│       └── portfolio_builder.py  # Copied verbatim (pure NumPy)
├── models/
│   ├── __init__.py
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── model.py         # MLPPricer(PricingModel) using nn.Module
│   │   └── config.py        # Copied verbatim
│   └── gnn_rnn_hybrid/
│       ├── __init__.py
│       ├── model.py         # HybridGnnRnn(BaseModel) using nn.Module
│       ├── config.py        # Copied verbatim
│       └── layers/
│           ├── __init__.py
│           ├── gnn_layers.py
│           ├── rnn_layers.py
│           ├── attention_layer.py
│           ├── fusion_layer.py
│           └── projection_layer.py
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Explicit PyTorch training loop
│   ├── callbacks.py         # Hook-based callback system
│   └── schedules.py         # LR scheduler factories
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py         # DataLoader-based evaluation
│   └── metrics.py           # torchmetrics + sklearn wrappers
├── inference/
│   ├── __init__.py
│   ├── model_io.py          # torch.save / torch.load + joblib scalers
│   └── predictor.py         # Predictor, BatchPredictor
├── validation/
│   ├── __init__.py
│   └── gates.py             # Copied verbatim (framework-agnostic)
├── monitoring/
│   ├── __init__.py
│   └── drift.py             # Copied verbatim (framework-agnostic)
├── pipelines/
│   ├── __init__.py
│   ├── training.py
│   ├── evaluation.py
│   ├── inference.py
│   └── tuning.py
├── registry/
│   ├── __init__.py
│   └── registry.py          # Model registry with torch.save/load
├── tuning/
│   ├── __init__.py
│   └── search_space.py      # Copied verbatim (framework-agnostic)
└── utilities/
    ├── trade_graph_builder.py   # Copied verbatim (pure NumPy)
    └── trade_attribute_encoder.py  # Copied verbatim (pure NumPy)
```

### What Can Be Copied Verbatim (Zero Changes)

These files have no TensorFlow dependency — they are pure Python / NumPy / sklearn:

| File | Reason |
|------|--------|
| `core/types.py` | Pure dataclasses (TrainingResult, EvaluationResult, etc.) |
| `core/tracking.py` | MLflow / W&B / InMemory tracker — pure Python |
| `data/result.py` | DataBuildResult base dataclass |
| `data/manifest.py` | DatasetManifest — pure NumPy |
| `data/*/config.py` (all 3) | Pure dataclasses |
| `data/gnn_rnn_hybrid/synthetic.py` | Pure NumPy data generation |
| `data/gnn_rnn_hybrid/portfolio_builder.py` | Pure NumPy |
| `models/pricing/config.py` | Pure dataclass |
| `models/gnn_rnn_hybrid/config.py` | Pure dataclass |
| `validation/gates.py` | Pure Python |
| `monitoring/drift.py` | Pure NumPy / SciPy |
| `tuning/search_space.py` | Pure Python / Optuna |
| `utilities/trade_graph_builder.py` | Pure NumPy / sklearn |
| `utilities/trade_attribute_encoder.py` | Pure NumPy |

**~16 files, zero effort.** Copy them in, update the import paths from
`src.machine_learning.` to `src.machine_learning_pt.`.

---

## Phase 0: Foundation

**Goal**: Create the directory skeleton and the shared utilities that every later
phase depends on.

**Effort**: 0.5 day

### 0.1 Create directory structure

Create every `__init__.py` as an empty file. Populate later in each phase.

### 0.2 `core/device.py` (new file)

Centralised device selection. Every other module imports from here.

```python
"""Device selection utility for PyTorch training and inference."""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Resolve the best available device.

    Priority: explicit argument > CUDA > MPS (Apple Silicon) > CPU.

    Parameters
    ----------
    device : str, optional
        Force a specific device ("cuda", "mps", "cpu").

    Returns
    -------
    torch.device
    """
    if device is not None:
        resolved = torch.device(device)
        logger.info("Using device: %s (explicit)", resolved)
        return resolved

    if torch.cuda.is_available():
        resolved = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        resolved = torch.device("mps")
    else:
        resolved = torch.device("cpu")

    logger.info("Using device: %s (auto-detected)", resolved)
    return resolved
```

### 0.3 `core/config.py` — PyTorch-native build methods

Copy the TF version verbatim, then replace the two `build()` methods.

**`OptimizerConfig.build()`** — In TF this returns a Keras optimizer. In PyTorch,
optimizers require `model.parameters()` at construction time, so `build()` returns
a *factory function* (partial):

```python
def build(self, params) -> "torch.optim.Optimizer":
    """
    Build a PyTorch optimizer.

    Parameters
    ----------
    params : iterable
        Model parameters (from model.parameters()).

    Returns
    -------
    torch.optim.Optimizer
    """
    import torch.optim as optim

    name = self.name.lower()
    if name == "adam":
        return optim.Adam(
            params, lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            weight_decay=self.weight_decay,
        )
    elif name == "adamw":
        return optim.AdamW(
            params, lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
            weight_decay=self.weight_decay,
        )
    elif name == "sgd":
        return optim.SGD(
            params, lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
    elif name == "rmsprop":
        return optim.RMSprop(
            params, lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {self.name}")
```

Key difference from TF version: the signature changes from `build() -> Optimizer` to
`build(params) -> Optimizer` because PyTorch optimizers bind to parameters at
construction. The `clipnorm` / `clipvalue` fields are handled at training time
(gradient clipping is a training-loop concern in PyTorch, not an optimizer concern).

**`LRScheduleConfig.build()`** — returns a `torch.optim.lr_scheduler` instance:

```python
def build(self, optimizer: "torch.optim.Optimizer",
          total_steps: Optional[int] = None) -> Any:
    """
    Build a PyTorch learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose LR will be scheduled.
    total_steps : int, optional
        Total training steps (required for cosine / warmup_cosine).

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler or None
    """
    import math
    import torch.optim.lr_scheduler as lr_sched

    if self.schedule == "constant":
        return None

    elif self.schedule == "exponential":
        gamma = self.decay_rate ** (1.0 / max(self.decay_steps, 1))
        return lr_sched.ExponentialLR(optimizer, gamma=gamma)

    elif self.schedule == "step":
        return lr_sched.StepLR(
            optimizer, step_size=self.decay_steps, gamma=self.decay_rate,
        )

    elif self.schedule in ("cosine", "warmup_cosine"):
        if total_steps is None:
            raise ValueError("total_steps required for cosine schedule")

        warmup = self.warmup_steps if self.schedule == "warmup_cosine" else 0
        min_factor = self.min_lr / max(self.initial_lr, 1e-12)

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            progress = min(progress, 1.0)
            return max(min_factor,
                       0.5 * (1.0 + math.cos(math.pi * progress)))

        return lr_sched.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown schedule: {self.schedule}")
```

**`DataPipelineConfig.to_build_kwargs()`** — minor rename. In the TF version
this returns kwargs for `build_tf_dataset`. In the PT version it returns kwargs
for `build_dataloader`. The field names are the same except `cache` and
`shuffle_buffer` are replaced by PyTorch equivalents:

```python
def to_build_kwargs(self) -> Dict[str, Any]:
    """Return kwargs for build_dataloader()."""
    return {
        "batch_size": self.batch_size,
        "shuffle": self.shuffle,
        "drop_last": self.drop_remainder,
        "pin_memory": True,
        "num_workers": 0,
    }
```

### 0.4 Copy framework-agnostic files

Copy the ~16 files listed in the table above, updating import paths.

---

## Phase 1: Data Layer

**Goal**: Replace `build_tf_dataset()` with `build_dataloader()`. All data builders
produce `torch.utils.data.DataLoader` objects.

**Effort**: 1–1.5 days

### 1.1 `data/dataset.py` — Core Data Utilities

This is the PyTorch equivalent of the TF `build_tf_dataset` function. The key
architectural mapping:

| TF Concept | PyTorch Equivalent |
|---|---|
| `tf.data.Dataset.from_tensor_slices` | `torch.utils.data.Dataset.__getitem__` |
| `.cache()` | `pin_memory=True` + data stays in RAM natively |
| `.shuffle(buffer)` | `DataLoader(shuffle=True)` (full shuffle) |
| `.batch(size)` | `DataLoader(batch_size=size)` |
| `.map(fn)` for static inputs | Handled in `__getitem__` |
| `.prefetch(AUTOTUNE)` | `DataLoader(num_workers=N)` |

```python
"""
Dataset utilities for the PyTorch ML framework.

Provides:
    - DictDataset:  torch Dataset supporting both simple (X, y) and
      static+variable (GNN) input patterns.
    - build_dataloader:  Factory that wraps arrays into a batched,
      shuffled DataLoader — the PyTorch equivalent of build_tf_dataset.
    - SyntheticData, create_pricing_dataset, create_calibration_dataset:
      Pure NumPy generators (identical to the TF version).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DictDataset(Dataset):
    """
    PyTorch Dataset supporting simple and dict-input models.

    Handles two patterns:
        1. Simple (pricing / calibration): variable_inputs is an ndarray.
           __getitem__ returns (tensor, target).
        2. Static + variable (GNN): variable_inputs is a dict of ndarrays
           (per-sample), static_inputs is a dict of ndarrays (shared).
           __getitem__ returns ({**static, **variable[idx]}, target).

    Parameters
    ----------
    variable_inputs : ndarray or dict of ndarray
        Per-sample data. First dimension must equal len(targets).
    targets : ndarray
        Target values, shape (n_samples,) or (n_samples, n_outputs).
    static_inputs : dict of ndarray, optional
        Arrays shared across all samples (e.g. adjacency matrix).
        Converted to tensors once at init, returned in every __getitem__.
    variable_input_key : str
        Dict key for variable_inputs when it is a plain ndarray and
        static_inputs is present. Default "features".
    """

    def __init__(
        self,
        variable_inputs: Union[np.ndarray, Dict[str, np.ndarray]],
        targets: np.ndarray,
        static_inputs: Optional[Dict[str, np.ndarray]] = None,
        variable_input_key: str = "features",
    ) -> None:
        targets = np.asarray(targets, dtype=np.float32)
        self.n_samples = len(targets)
        self.targets = torch.from_numpy(targets)
        self.variable_input_key = variable_input_key

        # Variable inputs (per-sample, sliced by index)
        if isinstance(variable_inputs, dict):
            self.var_dict = {
                k: torch.from_numpy(np.asarray(v, dtype=np.float32))
                for k, v in variable_inputs.items()
            }
            self.is_dict = True
        else:
            arr = np.asarray(variable_inputs, dtype=np.float32)
            self.var_tensor = torch.from_numpy(arr)
            self.is_dict = False

        # Static inputs (shared, not sliced)
        self.static: Optional[Dict[str, torch.Tensor]] = None
        if static_inputs:
            self.static = {}
            for k, v in static_inputs.items():
                arr = np.asarray(v)
                if np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32)
                elif np.issubdtype(arr.dtype, np.integer):
                    arr = arr.astype(np.int32)
                self.static[k] = torch.from_numpy(arr)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        target = self.targets[idx]

        if self.is_dict:
            inputs = {k: v[idx] for k, v in self.var_dict.items()}
        else:
            inputs = self.var_tensor[idx]

        if self.static is not None:
            if isinstance(inputs, dict):
                inputs = {**self.static, **inputs}
            else:
                inputs = {**self.static, self.variable_input_key: inputs}

        return inputs, target


def build_dataloader(
    variable_inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    targets: np.ndarray,
    static_inputs: Optional[Dict[str, np.ndarray]] = None,
    variable_input_key: str = "features",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Build a batched DataLoader from arrays — the PyTorch equivalent of
    build_tf_dataset.

    Parameters
    ----------
    variable_inputs : ndarray or dict of ndarray
        Per-sample inputs. First dim = n_samples.
    targets : ndarray
        Targets, shape (n_samples,) or (n_samples, d).
    static_inputs : dict of ndarray, optional
        Shared arrays injected into every sample (e.g. graph structure).
    variable_input_key : str
        Key used when variable_inputs is ndarray + static_inputs is set.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Shuffle each epoch.
    num_workers : int
        Background data-loading processes. 0 = main process only.
    pin_memory : bool
        Pin memory for faster GPU transfer.
    drop_last : bool
        Drop the final incomplete batch.

    Returns
    -------
    DataLoader
    """
    dataset = DictDataset(
        variable_inputs, targets, static_inputs, variable_input_key,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
```

The `SyntheticData`, `create_pricing_dataset`, and `create_calibration_dataset`
functions are pure NumPy — copy them verbatim from the TF version.

### 1.2 `data/pricing/build.py`

Replace calls from `build_tf_dataset(...)` to `build_dataloader(...)`.
Replace return type annotation from `tf.data.Dataset` to `DataLoader`.

The data preparation logic (sklearn StandardScaler, train_test_split) is identical.

### 1.3 `data/calibration/build.py`

Same approach as pricing.

### 1.4 `data/gnn_rnn_hybrid/build.py`

The static + variable pattern maps directly:

```python
# TF version
train_ds = build_tf_dataset(
    variable_inputs=train_var,
    targets=train_tgt,
    static_inputs=static,
    **pipe_kwargs,
)

# PyTorch version — identical call signature
train_loader = build_dataloader(
    variable_inputs=train_var,
    targets=train_tgt,
    static_inputs=static,
    **pipe_kwargs,
)
```

---

## Phase 2: Base Model Hierarchy

**Goal**: Create `nn.Module`-based base classes that mirror `BaseModel`,
`PricingModel`, `CalibrationModel`, and `PortfolioModel`.

**Effort**: 1–1.5 days

### 2.1 `core/base.py`

```python
"""
Base model classes for PyTorch-native ML models.

Architecture:
    BaseModel (nn.Module)
        ├── PricingModel      — option / derivative pricing
        ├── CalibrationModel  — model parameter calibration
        └── PortfolioModel    — portfolio-level predictions (GNN-RNN)
"""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Abstract base for all ML models in the PyTorch framework.

    Provides:
        - Consistent metadata tracking
        - Parameter counting
        - Standardised forward() interface

    All models should inherit from this (or its subclasses).
    """

    def __init__(self, name: str = "base_model", **kwargs):
        super().__init__()
        self._model_name = name
        self._model_metadata: Dict[str, Any] = {
            "model_name": name,
            "model_class": self.__class__.__name__,
            "created_at": datetime.utcnow().isoformat(),
            "framework": "pytorch",
            "framework_version": torch.__version__,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._model_metadata.copy()

    def update_metadata(self, **kwargs) -> None:
        self._model_metadata.update(kwargs)

    @abstractmethod
    def forward(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError

    def summary_dict(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": self._model_name,
            "class": self.__class__.__name__,
            "trainable_params": trainable,
            "non_trainable_params": total - trainable,
            "metadata": self.metadata,
        }


class PricingModel(BaseModel):
    """
    Base class for option / derivative pricing models.

    Adds price_with_greeks() using torch.autograd.
    """

    def __init__(self, name: str = "pricing_model",
                 output_greeks: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_greeks = output_greeks
        self.update_metadata(model_type="pricing", output_greeks=output_greeks)

    @property
    def feature_names(self) -> List[str]:
        return ["spot", "strike", "volatility", "rate",
                "time_to_expiry", "option_type"]

    def price(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)

    def price_with_greeks(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute price and Greeks via torch.autograd.

        Equivalent to the TF GradientTape version.
        """
        inputs = inputs.clone().detach().requires_grad_(True)

        price = self.forward(inputs)
        price.sum().backward(create_graph=True, retain_graph=True)
        grads = inputs.grad

        result = {"price": price.detach()}
        if grads is not None:
            result["delta"] = grads[:, 0:1].detach()
            result["vega"] = grads[:, 2:3].detach()
            result["theta"] = -grads[:, 4:5].detach()
            result["rho"] = grads[:, 3:4].detach()
        else:
            zeros = torch.zeros(inputs.shape[0], 1)
            result.update({k: zeros for k in
                          ("delta", "vega", "theta", "rho")})

        return result


class CalibrationModel(BaseModel):
    """Base class for calibration networks."""

    def __init__(self, name: str = "calibration_model",
                 target_model: str = "unknown", n_parameters: int = 1,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_model = target_model
        self.n_parameters = n_parameters
        self.update_metadata(model_type="calibration",
                             target_model=target_model,
                             n_parameters=n_parameters)

    def calibrate(self, market_data: torch.Tensor) -> torch.Tensor:
        return self.forward(market_data)

    def calibrate_with_bounds(self, market_data: torch.Tensor,
                              lower: Optional[torch.Tensor] = None,
                              upper: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
        raw = self.forward(market_data)
        if lower is not None and upper is not None:
            return lower + (upper - lower) * torch.sigmoid(raw)
        return raw


class PortfolioModel(BaseModel):
    """
    Base class for portfolio-level models (GNN-RNN, graph models).

    Accepts dict inputs: trade_features, adjacency_matrix, pnl_history, etc.
    """

    def __init__(self, name: str = "portfolio_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.update_metadata(model_type="portfolio")

    @property
    def required_inputs(self) -> List[str]:
        return ["trade_features", "adjacency_matrix"]
```

### 2.2 `core/protocols.py`

The `Trainable` protocol is framework-agnostic — copy as-is. Replace
`KerasTrainableAdapter` with:

```python
class TorchTrainableAdapter:
    """Wraps any nn.Module as a Trainable."""

    def __init__(self, model: nn.Module, loss_fn=None):
        self.model = model
        self._loss_fn = loss_fn or nn.MSELoss()

    def forward(self, inputs):
        return self.model(inputs)

    def compute_loss(self, y_true, y_pred):
        return float(self._loss_fn(y_pred, y_true).item())

    def get_parameters(self):
        return {"state_dict": self.model.state_dict()}

    def set_parameters(self, params):
        self.model.load_state_dict(params["state_dict"])
```

---

## Phase 3: Training Infrastructure

**Goal**: Build the explicit PyTorch training loop, callback system, and LR schedulers.
This is the largest behavioral change from TF.

**Effort**: 2–2.5 days

### 3.1 `training/trainer.py`

The TF `Trainer` wraps `model.fit()` in one call. The PyTorch `Trainer` owns the
full train/validate loop.

```python
"""
PyTorch Trainer with explicit training loop.

Replaces the Keras model.fit() pattern with a loop that gives full
control over forward, backward, optimizer step, and metric logging.
"""
from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.machine_learning_pt.core.config import TrainingConfig
from src.machine_learning_pt.core.device import get_device
from src.machine_learning_pt.core.types import TrainingResult
from src.machine_learning_pt.training.callbacks import (
    CallbackList, get_standard_callbacks,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    High-level PyTorch trainer.

    Manages: device placement, optimizer creation, LR scheduling,
    gradient clipping, early stopping, checkpointing, and metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        custom_callbacks: Optional[List] = None,
        device: Optional[str] = None,
    ):
        self.config = config
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.custom_callbacks = custom_callbacks or []

        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)

    def _resolve_loss(self, name: str) -> nn.Module:
        """Map loss name string to PyTorch loss module."""
        mapping = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),
            "l1": nn.L1Loss(),
            "huber": nn.HuberLoss(),
            "smooth_l1": nn.SmoothL1Loss(),
        }
        if name.lower() not in mapping:
            raise ValueError(f"Unknown loss: {name}")
        return mapping[name.lower()]

    def _to_device(self, data):
        """Move batch inputs to device (handles tensors and dicts)."""
        if isinstance(data, dict):
            return {k: v.to(self.device) for k, v in data.items()}
        return data.to(self.device)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> TrainingResult:
        """
        Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data.
        val_loader : DataLoader, optional
            Validation data.

        Returns
        -------
        TrainingResult
        """
        optimizer = self.config.optimizer.build(self.model.parameters())
        loss_fn = self._resolve_loss(self.config.loss)
        scheduler = None
        if self.config.lr_schedule is not None:
            scheduler = self.config.lr_schedule.build(optimizer)

        callbacks = get_standard_callbacks(self.config)
        callbacks.extend(self.custom_callbacks)
        cb = CallbackList(callbacks)

        history: Dict[str, List[float]] = {}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()

        cb.on_train_begin(self.model, self.config)

        for epoch in range(1, self.config.epochs + 1):
            # --- Train ---
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = self._to_device(batch_x)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = loss_fn(preds.squeeze(), batch_y.squeeze())
                loss.backward()

                # Gradient clipping
                if self.config.optimizer.clipnorm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimizer.clipnorm,
                    )
                if self.config.optimizer.clipvalue is not None:
                    nn.utils.clip_grad_value_(
                        self.model.parameters(),
                        self.config.optimizer.clipvalue,
                    )

                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            history.setdefault("loss", []).append(avg_train_loss)

            # --- Validate ---
            avg_val_loss = None
            if val_loader is not None:
                avg_val_loss = self._validate(val_loader, loss_fn)
                history.setdefault("val_loss", []).append(avg_val_loss)

            # --- LR scheduler step ---
            if scheduler is not None:
                scheduler.step()
                history.setdefault("lr", []).append(
                    optimizer.param_groups[0]["lr"]
                )

            # --- Callbacks ---
            epoch_logs = {"loss": avg_train_loss}
            if avg_val_loss is not None:
                epoch_logs["val_loss"] = avg_val_loss
            cb.on_epoch_end(epoch, epoch_logs)

            # --- Early stopping ---
            monitor_val = avg_val_loss if avg_val_loss is not None else avg_train_loss
            if monitor_val < best_val_loss:
                best_val_loss = monitor_val
                best_epoch = epoch
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in
                    self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            es = self.config.early_stopping
            if es is not None and patience_counter >= es.patience:
                logger.info("Early stopping at epoch %d", epoch)
                if es.restore_best_weights:
                    self.model.load_state_dict(best_state)
                break

        total_time = time.time() - start_time
        final_epoch = len(history.get("loss", []))
        cb.on_train_end(history)

        return TrainingResult(
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss if best_val_loss < float("inf") else None,
            best_train_loss=min(history.get("loss", [float("inf")])),
            final_epoch=final_epoch,
            training_time_seconds=total_time,
            stopped_early=final_epoch < self.config.epochs,
            config=self.config.to_dict(),
            model_summary=self._get_model_summary(),
        )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, loss_fn: nn.Module) -> float:
        self.model.eval()
        losses = []
        for batch_x, batch_y in val_loader:
            batch_x = self._to_device(batch_x)
            batch_y = batch_y.to(self.device)
            preds = self.model(batch_x)
            loss = loss_fn(preds.squeeze(), batch_y.squeeze())
            losses.append(loss.item())
        return float(np.mean(losses))

    def _get_model_summary(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        return {
            "name": getattr(self.model, "_model_name", type(self.model).__name__),
            "trainable_params": trainable,
            "non_trainable_params": total - trainable,
        }
```

### 3.2 `training/callbacks.py`

Replace Keras callback inheritance with a simple hook protocol:

```python
class TrainingCallback:
    """Base class for training callbacks (hook-based)."""
    def on_train_begin(self, model, config): ...
    def on_train_end(self, history): ...
    def on_epoch_end(self, epoch, logs): ...


class CallbackList:
    """Container that dispatches to multiple callbacks."""
    def __init__(self, callbacks):
        self.callbacks = callbacks
    def on_train_begin(self, model, config):
        for cb in self.callbacks:
            cb.on_train_begin(model, config)
    def on_epoch_end(self, epoch, logs):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
    def on_train_end(self, history):
        for cb in self.callbacks:
            cb.on_train_end(history)


class MetricsLogger(TrainingCallback):
    """Logs training metrics to a JSON file (identical purpose to TF version)."""
    ...


class PricingErrorCallback(TrainingCallback):
    """Tracks pricing-specific error metrics (identical purpose to TF version)."""
    ...
```

### 3.3 `training/schedules.py`

Provide factory functions that return `torch.optim.lr_scheduler` instances:

```python
def build_warmup_cosine(optimizer, warmup_steps, total_steps, min_lr=1e-6):
    """Warmup + cosine decay LR schedule."""
    import math
    from torch.optim.lr_scheduler import LambdaLR

    min_factor = min_lr / optimizer.defaults["lr"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
```

---

## Phase 4: Model Architectures

**Goal**: Port all model architectures to `nn.Module`.

**Effort**: 3–4 days

### 4.1 `models/pricing/model.py` — MLPPricer (~0.5 day)

Straightforward translation. Key mappings:

| TF | PyTorch |
|---|---|
| `layers.Dense(units, activation=None)` | `nn.Linear(in_features, units)` |
| `layers.BatchNormalization()` | `nn.BatchNorm1d(units)` |
| `layers.Dropout(rate)` | `nn.Dropout(rate)` |
| `layers.Activation("relu")` | `nn.ReLU()` |
| `tf.keras.regularizers.l2(w)` | `weight_decay=w` in optimizer |
| `model(dummy_input)` to build | Not needed — dims specified at init |

The `MLPPricer.__init__` must accept `n_features` (input dimension) because
PyTorch requires it for `nn.Linear`. In TF, Keras inferred it from the first batch.

```python
class MLPPricer(PricingModel):
    def __init__(self, n_features: int = 6, hidden_units=[128, 64, 32],
                 activation="relu", dropout_rate=0.1,
                 use_batch_norm=True, **kwargs):
        super().__init__(name="mlp_pricer", output_greeks=True, **kwargs)
        layers = []
        in_dim = n_features
        for units in hidden_units:
            layers.append(nn.Linear(in_dim, units))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, inputs):
        return self.output_layer(self.hidden(inputs))
```

Note: `nn.Sequential` eliminates the need for manual loop in `forward()`.

### 4.2 `models/gnn_rnn_hybrid/layers/` — Custom Layers (~2–3 days)

Each TF `Layer` becomes an `nn.Module`. The translation is mechanical:

| TF Pattern | PyTorch Equivalent |
|---|---|
| `self.add_weight(shape, initializer)` | `nn.Parameter(torch.empty(shape))` + `nn.init.*` |
| `tf.matmul(a, b)` | `torch.matmul(a, b)` |
| `tf.gather(tensor, indices)` | `tensor[indices]` or `torch.index_select` |
| `tf.concat([a, b], axis=-1)` | `torch.cat([a, b], dim=-1)` |
| `tf.keras.layers.LSTM(units, return_sequences)` | `nn.LSTM(input_size, units, batch_first=True)` |
| `tf.keras.layers.GRU(units)` | `nn.GRU(input_size, units, batch_first=True)` |
| `tf.keras.layers.Dense(units)` | `nn.Linear(in_features, units)` |
| `tf.nn.relu(x)` | `torch.relu(x)` or `F.relu(x)` |
| `tf.reduce_mean(x, axis)` | `x.mean(dim=axis)` |
| `tf.expand_dims(x, axis)` | `x.unsqueeze(axis)` |
| `tf.squeeze(x, axis)` | `x.squeeze(axis)` |

Key difference for **RNN layers**: PyTorch `nn.LSTM` returns
`(output, (h_n, c_n))`, while Keras returns just `output` by default.
The hidden states must be explicitly unpacked.

Key difference for **GNN layers**: graph operations (message passing,
aggregation) are pure tensor math — the logic transfers directly. If needed,
`torch_geometric` provides optimised sparse operations, but the current manual
implementations translate cleanly.

### 4.3 `models/gnn_rnn_hybrid/model.py` — HybridGnnRnn (~0.5 day)

Compose the ported layers. Dict input handling is simpler in PyTorch — no
`tf.keras.Input` specs needed, just accept a dict in `forward()`.

---

## Phase 5: Evaluation and Inference

**Goal**: Port the evaluation pipeline, metrics, predictor, and model I/O.

**Effort**: 1–1.5 days

### 5.1 `evaluation/metrics.py`

Replace 6 custom `tf.keras.metrics.Metric` subclasses.

For stateful metrics used during training, use `torchmetrics`:

```python
import torchmetrics

# Direct replacements
mae = torchmetrics.MeanAbsoluteError()
mse = torchmetrics.MeanSquaredError()
r2 = torchmetrics.R2Score()
mape = torchmetrics.MeanAbsolutePercentageError()
```

For evaluation-time metrics (called once on full arrays), keep the existing
`compute_metrics()` function which delegates to `sklearn.metrics` — it is
already framework-agnostic.

### 5.2 `evaluation/evaluator.py`

Replace `tf.data.Dataset` iteration with `DataLoader` iteration:

```python
@torch.no_grad()
def _collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch_x, batch_y in loader:
        if isinstance(batch_x, dict):
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
        else:
            batch_x = batch_x.to(device)
        preds = model(batch_x).cpu().numpy().flatten()
        all_preds.append(preds)
        all_targets.append(batch_y.numpy().flatten())
    return np.concatenate(all_preds), np.concatenate(all_targets)
```

The rest of `Evaluator` (compute_metrics, summary, plotting) is identical.

### 5.3 `inference/model_io.py`

Replace TF SavedModel with PyTorch state_dict:

```python
# Artifact structure:
#   model_dir/
#   ├── model.pt               # state_dict
#   ├── model_class.json       # {"module": "...", "class": "...", "config": {...}}
#   ├── config.json            # TrainingConfig
#   ├── metadata.json
#   ├── feature_scaler.joblib
#   ├── target_scaler.joblib
#   └── training_history.json

def save_model(model, path, config=None, metadata=None,
               feature_scaler=None, target_scaler=None, **kwargs):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path / "model.pt")
    # Also save model class info for reconstruction
    model_info = {
        "module": type(model).__module__,
        "class": type(model).__name__,
    }
    with open(path / "model_class.json", "w") as f:
        json.dump(model_info, f, indent=2)
    # scalers, config, metadata — identical to TF version
    ...

def load_model(path, model_factory=None, custom_objects=None, **kwargs):
    path = Path(path)
    state_dict = torch.load(path / "model.pt", map_location="cpu")

    if model_factory is not None:
        model = model_factory()
    else:
        # Auto-reconstruct from model_class.json
        ...

    model.load_state_dict(state_dict)
    model.eval()
    # Load scalers, config, metadata — identical to TF version
    ...
```

Key difference: PyTorch `load_model` needs a `model_factory` to reconstruct
the architecture, or model_class.json to auto-import. This replaces the Keras
`@register_keras_serializable` pattern.

### 5.4 `inference/predictor.py`

Replace `model.predict()` with `model.eval()` + `torch.no_grad()`:

```python
class Predictor:
    def __init__(self, model, feature_scaler=None, target_scaler=None,
                 device=None):
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device).eval()
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    @torch.no_grad()
    def predict(self, features, normalize=True, denormalize=True):
        if isinstance(features, dict):
            tensor_features = {
                k: torch.tensor(v, dtype=torch.float32).to(self.device)
                for k, v in features.items()
            }
            preds = self.model(tensor_features).cpu().numpy().flatten()
        else:
            features = np.asarray(features, dtype=np.float32)
            if normalize and self.feature_scaler is not None:
                features = self.feature_scaler.transform(features)
            tensor = torch.tensor(features).to(self.device)
            preds = self.model(tensor).cpu().numpy().flatten()

        if denormalize and self.target_scaler is not None:
            preds = self.target_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).flatten()
        return preds
```

The `BatchPredictor` class (ensemble predictions) is identical except it wraps
PyTorch `Predictor` instances.

---

## Phase Summary

| Phase | Files | New | Copied | Effort |
|-------|-------|-----|--------|--------|
| 0 — Foundation | 18 | 2 | 16 | 0.5 day |
| 1 — Data Layer | 5 | 1 | 0 | 1–1.5 days |
| 2 — Base Model | 2 | 2 | 0 | 1–1.5 days |
| 3 — Training | 3 | 3 | 0 | 2–2.5 days |
| 4 — Models | 9 | 9 | 0 | 3–4 days |
| 5 — Eval + Inference | 4 | 4 | 0 | 1–1.5 days |
| **Total** | **~41** | **~21** | **~16** | **8–11 days** |

### Implementation Order

Phases 1 and 2 are independent and can be done in parallel. Phase 3 depends on
both (the Trainer uses DataLoader from Phase 1 and nn.Module from Phase 2).
Phase 4 only depends on Phase 2. Phase 5 depends on Phases 3 and 4.

```
Phase 0 ──┬── Phase 1 (data) ──┐
           │                     ├── Phase 3 (training) ── Phase 5 (eval/inference)
           └── Phase 2 (models) ┘         │
                      │                    │
                      └── Phase 4 (layers) ┘
```

### Validation Checkpoints

After each phase, run a smoke test to verify:

- **Phase 0**: All imports resolve. Config objects serialise/deserialise.
- **Phase 1**: `build_dataloader()` produces batches. Iterate and print shapes.
- **Phase 2**: Instantiate `MLPPricer`, call `model(dummy_input)`, verify output shape.
- **Phase 3**: Train `MLPPricer` on synthetic data for 5 epochs. Verify `TrainingResult`.
- **Phase 4**: Instantiate `HybridGnnRnn`, forward pass with dict input, verify output.
- **Phase 5**: Full pipeline: build data → train → evaluate → save → load → predict.
