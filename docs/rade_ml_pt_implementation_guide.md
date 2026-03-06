# PyTorch Migration: Manual Implementation Guide

## Overview

This guide walks through implementing the `rade_ml_pt` package on a work environment
where files must be copied manually. Each phase is self-contained and testable.

**Prerequisites:**
- Python 3.12+
- `pytest.ini` at project root with `addopts = -p no:metadata` (fixes Windows DLL conflict)
- Dependencies installed from `requirements_rade_ml_pt.txt`

**Run tests with:**
```bash
python -m pytest <test_path> -q
```

---

## Phase 0: Setup (2 files) -- COMPLETED

| # | File | Action |
|---|------|--------|
| 1 | `requirements_rade_ml_pt.txt` | Create at project root |
| 2 | `src/rade_ml_pt/__init__.py` | Create minimal (wire up exports at the end) |

**Checkpoint:** `pip install -r requirements_rade_ml_pt.txt`

---

## Phase 1: Validation + Core Types (foundation everything else imports) -- COMPLETED

### Source (copy from rade_ml, update imports `src.rade_ml.` -> `src.rade_ml_pt.`)

| # | File | Internal imports to update |
|---|------|--------------------------|
| 1 | `validation/base.py` | None (standalone) |
| 2 | `validation/exceptions.py` | None (standalone) |
| 3 | `core/types.py` | None (standalone) |

### `__init__.py` files

| # | File | Action |
|---|------|--------|
| 1 | `validation/__init__.py` | Copy from rade_ml, update imports |
| 2 | `core/__init__.py` | Create minimal (update later as core grows) |

### Tests (copy from rade_ml, update imports `src.rade_ml.` -> `src.rade_ml_pt.`)

| # | File | Imports to update |
|---|------|-------------------|
| 1 | `tests/rade_ml_pt/__init__.py` | Create empty |
| 2 | `tests/rade_ml_pt/validation/__init__.py` | Create empty |
| 3 | `tests/rade_ml_pt/validation/test_validation_base.py` | `from src.rade_ml_pt.validation.base import ...` |
| 4 | `tests/rade_ml_pt/validation/test_validation_exceptions.py` | `from src.rade_ml_pt.validation.exceptions import ...` |
| 5 | `tests/rade_ml_pt/core/__init__.py` | Create empty |
| 6 | `tests/rade_ml_pt/core/test_core_types.py` | `from src.rade_ml_pt.core.types import ...` |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/validation/ tests/rade_ml_pt/core/test_core_types.py -q`

---

## Phase 2: Core Framework (PyTorch-ported) -- COMPLETED

### Source (PORT -- rewritten for PyTorch)

| # | File | Key changes from TF |
|---|------|---------------------|
| 1 | `core/base.py` | `tf.keras.Model` -> `nn.Module`, `call()` -> `forward()` |
| 2 | `core/config.py` | `tf.keras.optimizers.*` -> `torch.optim.*`, `tf.keras.optimizers.schedules.*` -> `torch.optim.lr_scheduler.*` |

Update `core/__init__.py` to export `BaseModel` + config classes.

### Tests (PORT)

| # | File |
|---|------|
| 1 | `tests/rade_ml_pt/core/test_core_base.py` |
| 2 | `tests/rade_ml_pt/core/test_core_config.py` |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/core/ -q`

---

## Phase 3: Data I/O + Configs (TF-free leaf modules) -- COMPLETED

### Source (copy from rade_ml, update imports)

| # | File | Internal imports to update |
|---|------|--------------------------|
| 1 | `data/io.py` | `from src.rade_ml_pt.validation.exceptions import ...` |
| 2 | `data/hybrid_gnn_rnn/config.py` | `from src.rade_ml_pt.core.config import DataPipelineConfig` |
| 3 | `data/hybrid_gnn_rnn/plots.py` | None (matplotlib/seaborn/networkx only) |
| 4 | `pipelines/config.py` | None (standalone dataclass) |
| 5 | `registry/entry.py` | None (standalone dataclass) |
| 6 | `evaluation/metrics.py` | None (numpy only) |

### `__init__.py` files

| # | File | Action |
|---|------|--------|
| 1 | `data/__init__.py` | Create minimal (update later in Phase 5) |
| 2 | `data/hybrid_gnn_rnn/__init__.py` | Copy from rade_ml, update imports |
| 3 | `pipelines/__init__.py` | Create minimal (update later in Phase 10) |
| 4 | `registry/__init__.py` | Create minimal (update later in Phase 9) |
| 5 | `evaluation/__init__.py` | Create minimal (update later in Phase 9) |
| 6 | `evaluation/plots/__init__.py` | Create minimal (needed for Phase 4) |

### Tests (copy from rade_ml, update imports)

| # | File | Imports to update |
|---|------|-------------------|
| 1 | `tests/rade_ml_pt/data/__init__.py` | Create empty |
| 2 | `tests/rade_ml_pt/data/test_data_io.py` | `from src.rade_ml_pt.data.io import CacheLoader` and `from src.rade_ml_pt.validation.exceptions import ...` |
| 3 | `tests/rade_ml_pt/pipelines/__init__.py` | Create empty |
| 4 | `tests/rade_ml_pt/pipelines/test_pipelines_config.py` | `from src.rade_ml_pt.pipelines.config import PipelineConfig` |
| 5 | `tests/rade_ml_pt/registry/__init__.py` | Create empty |
| 6 | `tests/rade_ml_pt/registry/test_registry_entry.py` | `from src.rade_ml_pt.registry.entry import RegistryEntry` |
| 7 | `tests/rade_ml_pt/evaluation/__init__.py` | Create empty |
| 8 | `tests/rade_ml_pt/evaluation/test_evaluation_metrics.py` | `from src.rade_ml_pt.evaluation.metrics import rmse, mape, ...` |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/data/test_data_io.py tests/rade_ml_pt/pipelines/test_pipelines_config.py tests/rade_ml_pt/registry/test_registry_entry.py tests/rade_ml_pt/evaluation/test_evaluation_metrics.py -q`

---

## Phase 4: Remaining TF-Free Modules (features, tracking, tuning, plots) -- COMPLETED

### Source (copy from rade_ml, update imports)

| # | File | Internal imports to update |
|---|------|--------------------------|
| 1 | `features/transforms/standardiser.py` | None (sklearn only) |
| 2 | `features/transforms/dimensionality.py` | None (sklearn/scipy only) |
| 3 | `utilities/attribute_encoder.py` | None (sklearn only) |
| 4 | `tracking/run.py` | None (standalone) |
| 5 | `tracking/tracker.py` | `from src.rade_ml_pt.tracking.run import Run` |
| 6 | `tuning/tuner.py` | `from src.rade_ml_pt.core.types import TuningResult` |
| 7 | `tuning/plots.py` | None (optuna/matplotlib only) |
| 8 | `training/plots.py` | None (matplotlib only) |
| 9 | `evaluation/plots/residuals.py` | None (matplotlib/seaborn only) |
| 10 | `evaluation/plots/predictions.py` | None (matplotlib only) |

### `__init__.py` files

| # | File | Action |
|---|------|--------|
| 1 | `features/__init__.py` | Copy from rade_ml, update imports |
| 2 | `features/transforms/__init__.py` | Copy from rade_ml, update imports |
| 3 | `tracking/__init__.py` | Copy from rade_ml, update imports |
| 4 | `tuning/__init__.py` | Copy from rade_ml, update imports |
| 5 | `utilities/__init__.py` | Create minimal (update later in Phase 7) |
| 6 | `training/__init__.py` | Create minimal (update later in Phase 8) |

### Tests (copy from rade_ml, update imports)

| # | File | Imports to update |
|---|------|-------------------|
| 1 | `tests/rade_ml_pt/features/__init__.py` | Create empty |
| 2 | `tests/rade_ml_pt/features/test_features_standardiser.py` | `from src.rade_ml_pt.features.transforms.standardiser import ...` |
| 3 | `tests/rade_ml_pt/tracking/__init__.py` | Create empty |
| 4 | `tests/rade_ml_pt/tracking/test_tracking_run.py` | `from src.rade_ml_pt.tracking.run import Run` |
| 5 | `tests/rade_ml_pt/tracking/test_tracking_tracker.py` | `from src.rade_ml_pt.tracking.tracker import ExperimentTracker` |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/features/ tests/rade_ml_pt/tracking/ -q`

*All 20 TF-free source files and 13 TF-free test files are now done.*

---

## Phase 5: Data Pipeline (PyTorch-ported) -- COMPLETED

### Source (PORT)

| # | File | Key changes from TF |
|---|------|---------------------|
| 1 | `data/dataset.py` | `tf.data.Dataset` -> `torch.utils.data.Dataset` + `DataLoader` |
| 2 | `data/result.py` | `tf.data.Dataset` type hints -> `DataLoader` |
| 3 | `data/hybrid_gnn_rnn/build.py` | `tf.SparseTensor` -> `torch.sparse_coo_tensor`, indices `[2,nnz]` |

Update `data/__init__.py` to export `build_dataloader`, `DataBuildResult`, `CacheLoader`.

### Tests (PORT)

| # | File |
|---|------|
| 1 | `tests/rade_ml_pt/data/test_data_dataset.py` |
| 2 | `tests/rade_ml_pt/data/test_data_result.py` |
| 3 | `tests/rade_ml_pt/data/hybrid_gnn_rnn/__init__.py` (create empty) |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/data/ -q`

---

## Phase 6: Model Layers (PyTorch-ported, most complex) -- COMPLETED

### Source (PORT -- do one at a time, test after each)

| # | File | Key changes from TF |
|---|------|---------------------|
| 1 | `models/hybrid_gnn_rnn/layers/gnn_layers.py` | `torch.sparse.mm`, `scatter_reduce` |
| 2 | `models/hybrid_gnn_rnn/layers/rnn_layers.py` | `nn.LSTM/GRU`, `batch_first=True`, training state propagation in `_build()` |
| 3 | `models/hybrid_gnn_rnn/layers/fusion_layer.py` | Padded tensors + masks replacing `RaggedTensor` |
| 4 | `models/hybrid_gnn_rnn/layers/attention_layer.py` | `index_select`, sparse submatrix extraction, optional `return_attention` |
| 5 | `models/hybrid_gnn_rnn/layers/projection_layer.py` | `nn.Parameter` / `nn.UninitializedParameter` lazy init matching TF `build()`, `torch.einsum`, `torch.topk`, `F.normalize` |

### `__init__.py` files

| # | File | Action |
|---|------|--------|
| 1 | `models/__init__.py` | Create empty |
| 2 | `models/hybrid_gnn_rnn/__init__.py` | Create minimal (update in Phase 7) |
| 3 | `models/hybrid_gnn_rnn/layers/__init__.py` | Export all 7 layer classes |

### Tests (PORT -- one per layer, test after each)

| # | File | Checkpoint |
|---|------|------------|
| 1 | `tests/rade_ml_pt/models/__init__.py` | Create empty |
| 2 | `tests/rade_ml_pt/models/hybrid_gnn_rnn/__init__.py` | Create empty |
| 3 | `tests/rade_ml_pt/models/hybrid_gnn_rnn/layers/__init__.py` | Create empty |
| 4 | `test_gnn_layers.py` | `pytest .../test_gnn_layers.py -q` |
| 5 | `test_rnn_layers.py` | `pytest .../test_rnn_layers.py -q` |
| 6 | `test_fusion_layer.py` | `pytest .../test_fusion_layer.py -q` |
| 7 | `test_attention_layer.py` | `pytest .../test_attention_layer.py -q` |
| 8 | `test_projection_layer.py` | `pytest .../test_projection_layer.py -q` |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/models/hybrid_gnn_rnn/layers/ -q`

---

## Phase 7: Model Assembly + Graph Builder (PyTorch-ported)

### Source (PORT)

| # | File | Key changes from TF |
|---|------|---------------------|
| 1 | `utilities/graph_builder.py` | `torch.sparse_coo_tensor` (indices `[2, nnz]`) |
| 2 | `models/hybrid_gnn_rnn/config.py` | Already copied in Phase 3 |
| 3 | `models/hybrid_gnn_rnn/model.py` | Assembles all layers, reconstructs sparse tensors |

Update `models/hybrid_gnn_rnn/__init__.py` and `utilities/__init__.py`.

### Tests (PORT)

| # | File |
|---|------|
| 1 | `tests/rade_ml_pt/models/hybrid_gnn_rnn/conftest.py` (provides fixtures) |
| 2 | `tests/rade_ml_pt/models/hybrid_gnn_rnn/test_hybrid_gnn_rnn_model.py` |
| 3 | `tests/rade_ml_pt/models/hybrid_gnn_rnn/test_hybrid_gnn_rnn_config.py` (copy, TF-free) |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/models/ -q`

---

## Phase 8: Training Infrastructure (PyTorch-ported)

### Source (PORT)

| # | File | Key changes from TF |
|---|------|---------------------|
| 1 | `training/schedules.py` | `WarmupCosineSchedule` as `LambdaLR` |
| 2 | `training/callbacks.py` | Hook-based `Callback` base, EarlyStopping, ModelCheckpoint, etc. |
| 3 | `training/strategy.py` | `torch.device` selection (cuda/mps/cpu) |
| 4 | `training/trainer.py` | Manual train loop replacing `model.fit()` |

Update `training/__init__.py`.

### Tests (PORT)

| # | File |
|---|------|
| 1 | `tests/rade_ml_pt/training/__init__.py` (create empty) |
| 2 | `test_training_schedules.py` |
| 3 | `test_training_callbacks.py` |
| 4 | `test_training_trainer.py` |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/training/ -q`

---

## Phase 9: Evaluation, Inference, Registry (PyTorch-ported)

### Source (PORT)

| # | File | Key changes from TF |
|---|------|---------------------|
| 1 | `evaluation/evaluator.py` | `model.eval()` + `torch.no_grad()` + DataLoader loop |
| 2 | `inference/runner.py` | `torch.load()`, `torch.no_grad()` prediction |
| 3 | `registry/store.py` | `torch.save(model)` / `torch.load()`, `.pt` files |

Update `evaluation/__init__.py`, `inference/__init__.py`, `registry/__init__.py`.

### Tests (PORT)

| # | File |
|---|------|
| 1 | `tests/rade_ml_pt/inference/__init__.py` (create empty) |
| 2 | `test_evaluation_evaluator.py` |
| 3 | `test_inference_runner.py` |
| 4 | `test_registry_store.py` |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/evaluation/ tests/rade_ml_pt/inference/ tests/rade_ml_pt/registry/ -q`

---

## Phase 10: Pipelines (PyTorch-ported)

### Source (PORT)

| # | File | Key changes from TF |
|---|------|---------------------|
| 1 | `pipelines/base.py` | Abstract classes with `nn.Module` types, `model.to(device)` |
| 2 | `pipelines/hybrid_gnn_rnn/train.py` | `torch.save()` for dataset persistence |
| 3 | `pipelines/hybrid_gnn_rnn/eval.py` | `torch.load()` + DataLoader reconstruction |
| 4 | `pipelines/hybrid_gnn_rnn/infer.py` | `torch.tensor()` for model inputs |
| 5 | `pipelines/hybrid_gnn_rnn/tune.py` | Optuna + PyTorch model build |

Update `pipelines/__init__.py` and `pipelines/hybrid_gnn_rnn/__init__.py`.

### Tests

| # | File |
|---|------|
| 1 | `tests/rade_ml_pt/pipelines/test_pipelines_base.py` (copy, TF-free) |

**Checkpoint:** `python -m pytest tests/rade_ml_pt/pipelines/ -q`

---

## Phase 11: Wire Up + Final Sweep

Update `src/rade_ml_pt/__init__.py` to export all public symbols.

**Checkpoint:** `python -m pytest tests/rade_ml_pt/ -q` -- expect **304 passed**

---

## Phase 12: Example Scripts

| # | File |
|---|------|
| 1 | `examples/rade_ml_pt/hybrid_gnn_rnn/01_train_hybrid_gnn_rnn.py` |
| 2 | `examples/rade_ml_pt/hybrid_gnn_rnn/02_train_hybrid_gnn_rnn_toy_regression.py` |
| 3 | `examples/rade_ml_pt/hybrid_gnn_rnn/03_train_hybrid_gnn_rnn_graph_aligned_toy.py` |

---

## Summary

| Phase | Description | Files | Type | Status |
|-------|-------------|-------|------|--------|
| 0 | Setup | 2 | Create | DONE |
| 1 | Validation + Core types | 10 | Copy | DONE |
| 2 | Core framework | 4 | **PORT** | DONE |
| 3 | Data I/O + configs | 18 | Copy | DONE |
| 4 | Features, tracking, tuning, plots | 19 | Copy | DONE |
| 5 | Data pipeline | 6 | **PORT** | DONE |
| 6 | Model layers | 13 | **PORT** | DONE |
| 7 | Model + graph builder | 6 | **PORT** | |
| 8 | Training infrastructure | 8 | **PORT** | |
| 9 | Eval, inference, registry | 7 | **PORT** | |
| 10 | Pipelines | 7 | **PORT** | |
| 11 | Top-level init + final sweep | 1 | Wire up | |
| 12 | Example scripts | 3 | **PORT** | |

---

## Known Issues

- **Windows DLL conflict:** `pytest-metadata` plugin conflicts with PyTorch's `c10.dll`.
  Fix: add `addopts = -p no:metadata` to `pytest.ini`.
- **`__init__.py` files:** Several need to be created as minimal stubs initially, then
  updated with full exports once the modules they reference are implemented.
  Attempting to import from a module that doesn't exist yet will cause ImportError.

---

## Post-Migration Enhancements

Items to revisit once the full migration is complete:

- **Custom LSTM/GRU activations:** PyTorch's `nn.LSTM` / `nn.GRU` hardcode their internal
  activations (tanh for cell state, sigmoid for gates) inside fused cuDNN kernels. The
  `activation` and `recurrent_activation` config parameters are stored but only used by
  the dense fallback branch. To honour non-default activations for recurrent layers,
  implement custom `LSTMCell` / `GRUCell` modules that expose the full gate equations in
  Python with swappable activation functions, wrapped in a `StackedRNN` multi-layer
  container that matches the `nn.LSTM` / `nn.GRU` return API. Auto-select between cuDNN
  (default activations) and custom cells (non-default) in `RnnBlock._build()` to preserve
  performance when custom activations are not needed.
