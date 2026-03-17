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

## Phase 7: Model Assembly + Graph Builder (PyTorch-ported) -- COMPLETED

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

## Phase 8: Training Infrastructure (PyTorch-ported) -- COMPLETED

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

## Phase 9: Evaluation, Inference, Registry (PyTorch-ported) -- COMPLETED

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
so| 7 | Model + graph builder | 6 | **PORT** | DONE |
| 8 | Training infrastructure | 8 | **PORT** | DONE |
| 9 | Eval, inference, registry | 7 | **PORT** | DONE |
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

- **Trainer built-in verbose epoch progress:** The `Trainer.fit()` loop now prints a
  Keras-style one-liner per epoch (`Epoch 1/500  loss=...  val_loss=...  [2.3s]`) when
  `TrainingConfig.verbose=True`. This is independent of the `MetricsLogger` callback
  (which requires `log_dir` to be set). Port this change to the work version of
  `src/rade_ml_pt/training/trainer.py`: add `epoch_start = time.time()` at the top of
  the epoch loop, then after history accumulation print the progress line gated on
  `self.config.verbose`.

- **Hybrid GNN-RNN integration tests:** Add dedicated unit tests for the model-specific
  modules that currently only have coverage through higher-level pipeline / model tests:
  - `tests/rade_ml_pt/data/hybrid_gnn_rnn/test_deep_hedging_build.py` — test `build()`
    end-to-end (sparse tensor construction, DataLoader shapes, config-driven splits).
  - `tests/rade_ml_pt/data/hybrid_gnn_rnn/test_deep_hedging_config.py` — test
    `DeepHedgingDataConfig` serialisation roundtrips and field validation.
  - `tests/rade_ml_pt/pipelines/hybrid_gnn_rnn/test_train_pipeline.py` — test training
    pipeline orchestration (data build → model build → trainer.fit → checkpoint save).
  - `tests/rade_ml_pt/pipelines/hybrid_gnn_rnn/test_eval_pipeline.py` — test evaluation
    pipeline (checkpoint load → evaluator.run → EvaluationResult).
  - `tests/rade_ml_pt/pipelines/hybrid_gnn_rnn/test_infer_pipeline.py` — test inference
    pipeline (checkpoint load → predict → InferenceResult).
  - `tests/rade_ml_pt/pipelines/hybrid_gnn_rnn/test_tune_pipeline.py` — test tuning
    pipeline (Optuna trial → model build → score).

---

## Hybrid GNN-RNN inference pipeline (`infer.py`)

### Clean pattern: same keys as training, no targets, one assembly point

The model’s `forward()` expects a single dict with **exactly** these keys (see `_REQUIRED_KEYS` in the model):

- `trade_features`, `pnl_history`, `adjacency_indices`, `adjacency_values`, `adjacency_dense_shape`, `elementary_indices`, `target_indices`

Training adds a **targets** key for the loss; at inference you **never** add targets. So the inference “batch” is the same dict shape as training, with **different dimensions** (e.g. different number of scenarios, or more nodes for new trades), and **no** `targets` key.

**Recommended structure:**

1. **Load once (read-only)**  
   Load all inference artifacts into a single **inference context**: `graph_builder`, `encoder`, `scalers`, `portfolio`, `elementary_universe` (and any precomputed static pieces you want). Do not mutate these; use them only to **build** the model input dict.

2. **Single assembly function**  
   Implement one helper that turns “static” and “variable” pieces into the dict the model expects:

   ```text
   _build_model_input_dict(static_dict, pnl_history) -> Dict[str, Tensor]
   ```

   - **static_dict**: `trade_features`, `adjacency_indices`, `adjacency_values`, `adjacency_dense_shape`, `elementary_indices`, `target_indices` (all tensors or arrays; you convert to tensor once here).
   - **pnl_history**: `[n_scenarios, seq_len, n_elementary]` — the only per-scenario part.
   - Return a dict with exactly the 7 keys above (same names as training). No `targets`.

   This is the **only** place that knows how to assemble the model input. Both new_scenarios and new_trades end by calling this.

3. **Branch only on how you get static_dict and pnl_history**

   - **new_scenarios**  
     - Static: from loaded artifacts (graph_builder + encoder) — same graph, same trade_features, same indices. Build `static_dict` from them.  
     - Variable: build **pnl_history** from scenario data (portfolio + universe + PnL calculator), then scale with loaded scalers.  
     - Call `_build_model_input_dict(static_dict, pnl_history)` → pass to `model.forward(inputs)`.

   - **new_trades**  
     - Static: extend graph via `build_graph_projection`, get new trade_features, new adjacency, new `elementary_indices` / `target_indices`. Build `static_dict` from these.  
     - Variable: **pnl_history** — same as training scenario set or pad/zeros for new target columns, scaled with loaded scalers.  
     - Call `_build_model_input_dict(static_dict, pnl_history)` → pass to `model.forward(inputs)`.

4. **Inference “dataset”**  
   You don’t need a separate Dataset class. The pipeline produces **one** dict (with the 7 keys) and calls `model.forward(inputs)`. If `n_scenarios` is very large, batch **pnl_history** (e.g. chunk into batches of 256), keep the same static_dict for every chunk, run `forward()` per chunk and concatenate predictions. The contract is always: same key names as training, no targets, dimensions can differ (batch size, or graph size for new_trades).

5. **Summary**

   | Step | What |
   |------|------|
   | Load | One-off: artifacts → inference context (graph_builder, encoder, scalers, portfolio, universe). |
   | Branch | new_scenarios vs new_trades → only affects how you build `static_dict` and `pnl_history`. |
   | Assemble | Single `_build_model_input_dict(static_dict, pnl_history)` → dict with 7 keys, no targets. |
   | Predict | `model.forward(inputs)` (optionally batched over pnl_history chunks). |

   This keeps logic in one place, avoids duplicating the “training batch shape” logic, and makes it explicit that inference uses the same names as training with different dimensions and no targets.

### `run()` docstring (loading and steps with function names)

Use the following docstring on `HybridGnnRnnInferencePipeline.run()` so loading and each step are documented with the corresponding method names:

```python
def run(self) -> InferenceResult:
    """
    Execute the full inference pipeline.

    Input mode is taken from config (e.g. ``config.metadata["inference"]["input_mode"]``):
    - ``"new_trades"``: CSV of new trade attributes (same schema as training).
    - ``"new_scenarios"``: Folder of risk-factor scenario CSVs (same trades, new paths).

    Loading
    -------
    1. ``load_runner()`` — Load model and InferenceRunner from registry.
    2. ``_load_registry_artifacts()`` — Load graph_builder, encoder, scalers, job
       (asset portfolio, elementary trade objects), and any target/trade metadata
       from the registered version directory.

    Steps
    -----
    3. Resolve input_mode from config.
    4. If new_trades: ``_run_new_trades()`` — Load new trade CSV, extend graph via
       ``build_graph_projection()``, build model inputs, ``runner.predict()``, then ``post_infer()``.
    5. If new_scenarios: ``_run_new_scenarios()`` — Load job and scenario CSVs,
       build new elementary PnL (existing pricing logic), same graph and trade features,
       build model inputs, ``runner.predict()``, then ``post_infer()``.
    6. ``post_infer()`` — Inference-specific analytics (logging, save predictions CSV, optional plots).

    Returns
    -------
    InferenceResult
    """
```

### `_load_registry_artifacts()` docstring

```python
def _load_registry_artifacts(self, version_dir: Path) -> Dict[str, Any]:
    """
    Load everything saved at training from the registered version directory.

    Steps
    -----
    1. Load model state (e.g. ``torch.load(version_dir / "model.pt")`` or via registry).
    2. Load graph_builder (e.g. ``TradeGraphBuilder.load(path)`` or joblib/pickle).
    3. Load encoder (e.g. ``TradeAttributeEncoder.load(path)``).
    4. Load scalers: target PnL scaler, elementary PnL scaler (for inverse transform if needed).
    5. Load job: asset portfolio and elementary trade objects (for ``_run_new_scenarios()``).
    6. Optionally load target_pnl columns or target_attributes for ordering/labels.

    Returns
    -------
    Dict with keys such as graph_builder, encoder, scalers, job, target_labels, etc.
    """
```

### `_run_new_trades()` docstring

```python
def _run_new_trades(self) -> InferenceResult:
    """
    New trade attributes CSV → extend graph, build inputs, predict.

    Steps
    -----
    1. Load new trade attributes CSV (schema matches training; validation handled elsewhere).
    2. Encode new trades: merge with original attribs or encode new only; use ``encoder.transform()``.
    3. Extend graph: ``graph_builder.build_graph_projection(adjacency_matrix=..., encoded_trades=..., new_targets=n_new, k=...)`` — trained adjacency unchanged, new edges for new trades only.
    4. Build trade_features (e.g. ``graph_builder._weighted_features(encoded_trades)``), pnl_history (from registry; same scenarios, new target columns padded/zeros or your rule).
    5. Build model input dict: trade_features, pnl_history, adjacency_*, elementary_indices, target_indices.
    6. ``runner.predict(inputs)`` → InferenceResult.
    7. ``post_infer(result, config)``.

    Returns
    -------
    InferenceResult
    """
```

### `_run_new_scenarios()` docstring

```python
def _run_new_scenarios(self) -> InferenceResult:
    """
    Folder of scenario CSVs → new elementary PnLs, same graph, predict.

    Steps
    -----
    1. Load job from registry (``_load_registry_artifacts()``): asset portfolio, elementary trade objects.
    2. Load scenario CSVs from config path (e.g. ``config.metadata["inference"]["new_scenarios_dir"]``).
    3. Build new elementary PnL: existing pricing logic (e.g. price elementary trades under each scenario using job).
    4. Target PnL: same trades as training; compute if pricing available, else placeholders/zeros per product requirement.
    5. Scale PnLs with scalers from registry (transform only; fit was at training).
    6. Reuse graph from registry: same adjacency, same trade_features (no new trades).
    7. Build model input dict: trade_features (unchanged), pnl_history = new elementary PnL matrix, adjacency unchanged, elementary_indices/target_indices unchanged.
    8. ``runner.predict(inputs)`` → InferenceResult.
    9. ``post_infer(result, config)``.

    Returns
    -------
    InferenceResult
    """
```

### `prepare_inputs()` docstring

```python
def prepare_inputs(self, config: PipelineConfig) -> Dict[str, Any]:
    """
    Build model-ready inputs from the pipeline config.

    Used when the base ``run()`` flow is kept: base calls ``prepare_inputs()`` then
    ``runner.predict(prepared["inputs"])``. For the two-mode inference design, either:

    Option A — Dispatch here: resolve input_mode, call ``_prepare_inputs_new_trades()``
    or ``_prepare_inputs_new_scenarios()``, return same dict shape (inputs, sample_ids, metadata).

    Option B — Override ``run()``: do not use base ``prepare_inputs()``; ``run()`` calls
    ``_run_new_trades()`` or ``_run_new_scenarios()`` which build inputs and call
    ``runner.predict()`` directly.

    Returns
    -------
    Dict with "inputs" (model input dict), optional "sample_ids", optional "metadata".
    """
```

### `post_infer()` docstring

```python
def post_infer(self, result: InferenceResult, config: PipelineConfig) -> None:
    """
    Inference-specific analytics after prediction.

    Steps
    -----
    1. Log summary: ``result.n_samples``, mean/std/min/max of ``result.predictions``.
    2. If ``config.artifacts_dir``: save predictions (e.g. CSV with sample_ids and predictions).
    3. Optional: save plots (e.g. prediction distribution), export to downstream format.
    """
```

---

## Artifact strategy: what to save at training for eval / inference

Training consumes a **job** object (from rade_sr preprocessing) that includes asset portfolio, elementary trade universe, preprocessing config, and the ability to compute elementary PnLs (e.g. a PnL calculator). For **eval** and **inference** (standalone and ensemble) we need:

- **Asset portfolio** and **elementary trade universe** — to build or extend inputs (e.g. new scenario → new elementary PnLs).
- **PnL calculator** (or equivalent) — to turn new scenario data into elementary PnL for inference.

You want to keep **rade_sr** (preprocessing) separate from **rade_ml**: eval/inference should not depend on re-running preprocessing or on user-defined paths that point into rade_sr at runtime. So the question is what to **persist at training time** so that eval/inference are self-contained.

### Option A: Extract and save only portfolio + elementary trade universe (recommended)

**What:** At training time, extract from the job the **asset portfolio** and **elementary trade universe** and save them under the registry version (e.g. `registry/{version}/inference_artifacts/portfolio.pt`, `elementary_universe.pt` or similar). Use a format rade_ml owns (e.g. pickle, joblib, or a small serialized schema).

**Pros:**

- Only the data needed for inference is stored; no full job blob.
- rade_ml defines the artifact contract; eval/inference load from registry and do not call back into rade_sr for *data*.
- Lighter than saving the whole job; versioning is clear (one version dir = one model + one portfolio + one universe).

**Cons:**

- You must keep the extracted set complete (if something else is needed later, you add it to the extract list).
- The **PnL calculator** is separate (see below).

**PnL calculator:** The calculator is logic (often depends on curves, market data, rade_sr). Three sub-options:

1. **Serialize the calculator** (e.g. pickle) alongside portfolio/universe — only viable if the calculator and all its dependencies are picklable and the same code exists at load time; can be fragile across environments.
2. **Calculator factory at inference time** — Do not persist the calculator. At inference, config (or user) supplies a path or module that, given portfolio + universe (loaded from artifacts), returns a callable that computes elementary PnL from scenario data. So: rade_ml loads portfolio + universe from artifacts; calls into a thin rade_sr hook: “build_calculator(portfolio, universe)” or “load_calculator_from(path)”. That keeps heavy logic in rade_sr but data in rade_ml.
3. **Save calculator config / version** — Persist only a reference (e.g. calculator version or config name). At inference, that reference plus portfolio + universe (from artifacts) is passed to rade_sr to reconstruct the calculator (e.g. from a user-defined path or a known rade_sr API). Again, data lives in artifacts; calculator construction stays in rade_sr.

Recommendation: **Option A for data** (extract portfolio + elementary trade universe into registry artifacts). For the calculator, use **2 or 3** so rade_ml stays agnostic of rade_sr’s internals and you avoid fragile pickling.

### Option B: Save the whole job in artifacts

**What:** At training time, save the entire job object (e.g. `job.pt`) under the registry version. Eval/inference load the job and pull portfolio, universe, and (if possible) calculator from it.

**Pros:**

- Nothing is missed; one artifact = everything training had.

**Cons:**

- Heavy (job can be large); ties artifact format to the current job structure from rade_sr. If the job schema changes, old artifacts can break. Blurs ownership: the “registry artifact” is literally the preprocessing job.

**Verdict:** Prefer **Option A** for a clear separation: rade_ml owns a minimal, stable inference payload (portfolio + universe + model + graph/encoder/scalers); rade_sr owns job structure and calculator construction.

### Option C: User-defined path at eval/inference time

**What:** Do not save portfolio/universe at training. At eval/inference, the user supplies a path (or config) that points to the output of a preprocessing run; the pipeline loads portfolio/universe (and maybe calculator) from there.

**Pros:**

- No heavy write at training; no duplication of large objects.

**Cons:**

- Eval/inference depend on that path and on rade_sr being available at load time. Reproducibility and versioning are harder (which path? which preprocessing version?). Less self-contained.

**Verdict:** Use only if you explicitly want “inference always reads latest preprocessing output from a known location.” For “this model version was trained with this portfolio/universe,” **Option A** is cleaner.

### Summary (standalone and ensemble)

| Approach | Save at training | Eval/inference load | rade_sr coupling |
|----------|------------------|----------------------|------------------|
| **A: Extract portfolio + universe** | Yes (minimal) | From registry | Only for calculator factory/config (optional) |
| **B: Save whole job** | Yes (full job) | From registry | Artifact is rade_sr object |
| **C: User path** | No | From user path | Eval/inference call rade_sr or read its output |

**Recommended:** **Option A** — extract asset portfolio and elementary trade universe from the job during training and save them as part of the version’s inference artifacts. Use a small, rade_ml-defined contract (e.g. `load_inference_artifacts(version_dir)` → dict with `portfolio`, `elementary_universe`, plus graph_builder, encoder, scalers as today). For the PnL calculator, require a factory or config at inference time (or a serialized calculator only if it is simple and stable). Same pattern applies to **ensemble**: each member version can store its own inference artifacts (portfolio + universe for that cluster’s training job); the ensemble inference pipeline loads per member from each member’s registry version.

---

## Cloud Training (Google Cloud / Vertex AI)

### Overview

Add cloud-based training support via Google Cloud Vertex AI (Option 2: custom Docker
container). This is the standard front-office approach: reproducible, auditable,
cost-efficient (pay only for training time), and integrates with GCS for artifact storage.

### Folder Structure

```
src/rade_ml_pt/
├── cloud/                              # All cloud/infra code
│   ├── __init__.py
│   │
│   ├── config.py                       # CloudConfig dataclass
│   │                                   #   (project_id, region, bucket, machine_type,
│   │                                   #    accelerator, image_uri)
│   │
│   ├── storage/                        # Abstract away GCS vs local filesystem
│   │   ├── __init__.py
│   │   ├── base.py                     # ArtifactStore ABC (save, load, list, exists)
│   │   ├── local.py                    # LocalStore -- wraps current pathlib behaviour
│   │   └── gcs.py                      # GcsStore -- same interface, reads/writes gs://
│   │
│   ├── vertex/                         # Vertex AI job submission and monitoring
│   │   ├── __init__.py
│   │   ├── job.py                      # submit_training_job(), monitor_job(), cancel_job()
│   │   ├── config.py                   # VertexJobConfig (machine_type, accelerator,
│   │   │                               #   replica_count, timeout, env_vars)
│   │   └── utils.py                    # Build worker pool spec, parse job status
│   │
│   └── docker/                         # Dockerfile + build helpers
│       ├── Dockerfile                  # Multi-stage: pytorch-gpu base + project code
│       ├── .dockerignore
│       └── build.py                    # Build image, tag, push to GCR/Artifact Registry
│
scripts/                                # Top-level CLI convenience scripts
├── cloud_train.py                      # Parse args, build CloudConfig, submit Vertex job
├── cloud_eval.py                       # Submit eval job to Vertex
└── cloud_build.py                      # Build + push Docker image
```

### Key Abstraction: ArtifactStore

The most important piece is `cloud/storage/`. Currently the framework uses raw `Path()`
strings for `artifacts_dir`, `registry_dir`, etc. The `ArtifactStore` ABC abstracts
filesystem operations so the same pipeline code works locally and on GCS:

```python
class ArtifactStore(ABC):
    def save(self, data: bytes, key: str) -> None: ...
    def load(self, key: str) -> bytes: ...
    def save_file(self, local_path: Path, key: str) -> None: ...
    def download_file(self, key: str, local_path: Path) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list_keys(self, prefix: str) -> List[str]: ...
```

- `LocalStore` wraps current pathlib logic (backward compatible, zero behaviour change).
- `GcsStore` uses `google-cloud-storage` client for GCS read/write.

### Changes to Existing Code

| File | Change | Impact |
|------|--------|--------|
| `PipelineConfig` | Add optional `artifact_store` field | Backward compatible -- defaults to `LocalStore(artifacts_dir)` |
| `ModelRegistry` | Accept `ArtifactStore` instead of path string | Same interface, reads/writes via the store |
| `TrainPipeline.run()` | Use `self.store.save_file()` instead of raw `Path()` writes | ~5-10 lines |
| `EvalPipeline.run()` | Same pattern for loading | ~5-10 lines |

Model, layers, trainer, callbacks, data pipeline, and plots are **unchanged**.

### Dependencies

```
google-cloud-storage       # GCS read/write
google-cloud-aiplatform    # Vertex AI job submission
```

Both are optional cloud-only dependencies. Keep in a separate `requirements-cloud.txt`
or as an extras group (`pip install rade_ml_pt[cloud]`).

### Implementation Order

1. `cloud/storage/` -- foundation; once ArtifactStore works, everything plugs in.
2. `cloud/docker/` -- get a working container that runs training locally via `docker run`.
3. `cloud/vertex/` -- submit that container to Vertex AI.
4. `scripts/` -- CLI wrappers for convenience.

### Expected GPU Speedup

| Hardware | Approx. epoch time | Speedup vs CPU |
|----------|-------------------|----------------|
| CPU (current) | ~60s | 1x |
| T4 GPU | ~5–10s | 6–12x |
| A100 GPU | ~2–4s | 15–30x |

---

## Ensemble Model

### Overview

The ensemble model combines N members (each trained on a separate cluster of trades) into
a single prediction surface. A `TradeRouter` assigns each target trade to its cluster's
model; predictions are concatenated (disjoint clusters) or aggregated (overlapping). The
ensemble is registered as a first-class version in the registry.

**Model-agnostic:** The ensemble layer is not coupled to Hybrid GNN-RNN. Each cluster's
pipeline class is configurable via `EnsembleConfig.pipeline_class` (a dotpath string
resolved at runtime via `importlib`). Members can be Hybrid GNN-RNN, RNN-only, GNN-only,
or any `BaseModel` subclass. Different clusters can use different architectures.

See `docs/ensemble_implementation.md` for full implementation details including folder
structure, artifacts layout, visualization flow, and UI dashboard integration.

### Folder Structure (summary)

```
src/rade_ml_pt/
├── ensemble/
│   ├── config.py           # EnsembleConfig (member_configs, cluster_mapping, aggregation)
│   ├── builder.py          # EnsembleBuilder (load members, validate coverage)
│   ├── model.py            # EnsembleModel (predict, route, aggregate)
│   ├── router.py           # TradeRouter (trade → cluster mapping)
│   ├── aggregation.py      # concat / weighted_mean / stacking strategies
│   ├── registry.py         # EnsembleRegistry (register, load, tag, list)
│   ├── metrics.py          # Ensemble-level metric aggregation for UI
│   └── plots.py            # Member comparison, cluster heatmap, version comparison
│
├── pipelines/ensemble/
│   ├── train.py            # EnsembleTrainPipeline (orchestrate N member trains)
│   ├── eval.py             # EnsembleEvalPipeline (route, aggregate, per-member + ensemble metrics)
│   └── infer.py            # EnsembleInferencePipeline (route new trades/scenarios, aggregate)
```

### UI Dashboard (summary)

The dashboard is a **read-only** consumer of artifacts produced by offline pipelines
(training, evaluation). Only inference runs live in the UI.

| Page | Purpose | Cluster filtering |
|------|---------|-------------------|
| **Overview** | Active version, key metrics, worst trade, thumbnails | No (ensemble-level) |
| **Model Performance** | Training/eval plots, per-cluster breakdown table, cluster drill-down | Yes (click row) |
| **Inference** | Upload scenarios/trades, run predictions, download CSV | Yes (dropdown) |
| **Risk & Analytics** | PnL timeline, tail metrics, worst trades, trade drill-down, version comparison | Yes (dropdown + trade click) |

The key artifact enabling cluster filtering is `trade_cluster_map.json` which maps every
trade ID to its cluster. See `docs/ensemble_implementation.md` for the full visualization
flow and artifacts layout.

### Implementation Order

1. `ensemble/config.py` + `router.py` — config and trade routing (foundation).
2. `ensemble/model.py` + `aggregation.py` — EnsembleModel with predict/route/aggregate.
3. `ensemble/builder.py` + `registry.py` — build from members, register/load.
4. `pipelines/ensemble/train.py` — orchestrate N member trains + ensemble registration.
5. `pipelines/ensemble/eval.py` — route eval data, per-member + ensemble metrics.
6. `ensemble/metrics.py` + `plots.py` — aggregation helpers and visualisations.
7. `pipelines/ensemble/infer.py` — inference with routing for new trades/scenarios.
8. Dash UI dashboard (separate repo or `dash_app/` folder).
