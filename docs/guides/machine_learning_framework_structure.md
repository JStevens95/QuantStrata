# Machine Learning Framework Structure

This guide walks through `src/machine_learning/`: how folders interact, which parts are **static** (generic framework) and which are **model-dependent**, and how to add new models (e.g. a hybrid GNN-RNN) while keeping the same simple, flexible structure.

---

## 1. High-level layout

```
src/machine_learning/
├── core/              # STATIC: protocols, config, base classes, tracking
├── data/              # MIXED: generic dataset + per-model builders
│   ├── dataset.py     # STATIC: TFDataset, NormalizationStats, split/batch
│   ├── types.py       # STATIC: MLDataset, feature types
│   ├── common/        # STATIC: shared encoders, graph builders
│   ├── pricing/       # MODEL-DEPENDENT: build_pricing_data → train/val/test
│   ├── calibration/   # MODEL-DEPENDENT: build_calibration_dataset
│   └── gnn_rnn_hybrid/# MODEL-DEPENDENT: build_gnn_data → train/val/proj
├── models/            # MODEL-DEPENDENT: one subfolder per architecture
│   ├── pricing/       # MLP pricer, etc.
│   └── gnn_rnn_hybrid/# HybridGnnRnn + layers (GNN, RNN, fusion, attention)
├── training/          # STATIC: Trainer (Keras), fit loop, callbacks
├── pipelines/         # STATIC: run_training, evaluate_model, load/save, tuning
├── evaluation/        # STATIC: Evaluator, metrics, EvaluationResult
├── inference/         # STATIC: save_model, load_model, Predictor
├── tuning/            # STATIC: search space, run_tuning
├── registry/         # STATIC: model versioning, stages (staging/production)
├── calibration/       # MODEL-AWARE: TrainingManager (wires GNN-RNN to data)
└── utilities/         # STATIC: TradeAttributeEncoder, TradeGraphBuilder
```

---

## 2. Static (framework) vs model-dependent

### 2.1 Static – same for every model

These do **not** depend on whether you use an MLP pricer, GNN-RNN, or a new architecture. They define the contract and wiring.

| Folder / file | Role |
|---------------|------|
| **core/** | `Trainable` protocol (forward, compute_loss, get/set_parameters), `TrainingConfig`, `BaseModel` / `PricingModel` / `PortfolioModel`, callbacks, experiment tracking (MLflow, WandB, in-memory). |
| **training/** | `Trainer` (Keras), `TrainingResult`, `fit_model`. Uses `TrainingConfig`; works with any `tf.keras.Model` (or `Trainable` via pipelines). |
| **pipelines/** | `run_training(model, data, config)`, `evaluate_model()`, `load_model()` / `save_model()` / `predict()`, `run_tuning()`. Framework-agnostic where possible (e.g. `Trainable` protocol). |
| **evaluation/** | `Evaluator`, `EvaluationResult`, regression metrics (MSE, MAE, R², etc.), optional pricing-specific metrics. |
| **inference/** | `ModelArtifact`, `Predictor`, `BatchPredictor`; save/load via SavedModel and optional metadata (e.g. normalisation stats). |
| **tuning/** | Search space definitions, `run_tuning()` to drive hyperparameter search. |
| **registry/** | `ModelRegistry`, `ModelStage` (staging/production/archived), versioning, artifact paths. |
| **data/dataset.py** | `TFDataset`, `NormalizationStats`, train/val/test split, batching, normalisation. Generic for tabular (feature vector + target). |
| **data/types.py** | `MLDataset`, `PricingFeatures`, `CalibrationFeatures` – shared type definitions. |
| **data/common/** | Re-exports `TradeAttributeEncoder`, `TradeGraphBuilder` from `utilities/`. Shared across models that use trade graphs. |
| **utilities/** | `TradeAttributeEncoder`, `TradeGraphBuilder` – used by GNN-RNN (and any future graph-based model). |

So: **core, training, pipelines, evaluation, inference, tuning, registry, data/dataset, data/types, data/common, utilities** are the static, reusable framework.

### 2.2 Model-dependent – one per “model family”

Each model (or family) has its own **data builder** and **model definition**. The framework only expects:

- **Data**: a builder that returns a **consistent result type** (e.g. `train_ds`, `val_ds`, optional `test_ds` / `proj_ds`) and optional stats/metadata.
- **Model**: a `tf.keras.Model` (or something that implements `Trainable`) that can be trained with the same `Trainer` / `run_training` and evaluated with `Evaluator` / `evaluate_model`.

| Folder | Role | Contract |
|--------|------|----------|
| **data/pricing/** | Data for MLP (and similar) pricers. | `build_pricing_data()` → `PricingDataResult(train_ds, val_ds, test_ds, feature_stats, target_stats, metadata)`. Uses generic `TFDataset` / `create_pricing_dataset`. |
| **data/calibration/** | Data for calibration nets (e.g. vol surface → params). | `build_calibration_dataset()` → `CalibrationDataResult(...)`. |
| **data/gnn_rnn_hybrid/** | Data for GNN-RNN hybrid. | `build_gnn_data()` → `GnnDataResult(train_ds, val_ds, proj_ds, metadata)`. Uses synthetic or FX portfolio data, `gnn_inputs_to_tf_dataset`; graph + temporal inputs. |
| **models/pricing/** | MLP pricer (and related). | `MLPPricer`, `create_mlp_pricer()`; subclasses `PricingModel`; input = feature vector, output = price. |
| **models/gnn_rnn_hybrid/** | Hybrid GNN-RNN for PnL. | `HybridGnnRnn`, `default_hybrid_model_config()`; custom `call()` with dict inputs (trade_features, adjacency_matrix, pnl_history, indices). |

So: **data/pricing**, **data/calibration**, **data/gnn_rnn_hybrid**, **models/pricing**, **models/gnn_rnn_hybrid** are the model-dependent parts. Adding a new model = add a new `data/<model>/` and `models/<model>/` that follow the same contracts.

### 2.3 Model-aware glue (optional)

| Folder | Role |
|--------|------|
| **calibration/** | `TrainingManager`: knows about `HybridGnnRnn` and its data (training_ds, validation_ds). Wires that one model to its datasets and training config. You can have similar “manager” modules for other models if you want a single entry point. |

This is the only place that explicitly ties “GNN-RNN” to “this data format”. The rest of the framework stays generic.

---

## 3. How the pieces interact

### 3.1 Data flow (generic contract)

- **Generic**: `TFDataset` (and helpers) produce `tf.data.Dataset` with batches of `(features, targets)` or, for GNN-RNN, batches of dicts `(inputs_dict, targets)`.
- **Per-model**: Each `data/<model>/build.py` implements a `build_*_data()` that:
  - Generates or loads raw data (model-specific).
  - Normalises / preprocesses (model-specific).
  - Builds `tf.data.Dataset` (same API: batched, optionally shuffled).
  - Returns a **result dataclass** (e.g. `PricingDataResult`, `GnnDataResult`) so the rest of the pipeline only sees `train_ds`, `val_ds`, etc.

So: **preprocessing and input shape are model-dependent**; **the fact that you get train_ds/val_ds (and maybe test_ds/proj_ds) is the static contract**.

### 3.2 Training flow

- **Keras path**: `Trainer(model, TrainingConfig).fit(train_ds, val_ds)` – used by pricing and by the GNN-RNN when trained via Keras.
- **Protocol path**: `run_training(model, train_data, val_data, config)` in `pipelines/training.py` uses the `Trainable` protocol (forward, compute_loss, get/set_parameters). So any model that implements `Trainable` can be trained by the same loop without knowing the architecture.

So: **training loop and config are static**; **the model’s forward/loss and the data batch format are model-dependent**.

### 3.3 Evaluation and inference

- **evaluation**: `Evaluator(model).evaluate(test_ds)` or `evaluate_model(model, test_ds, ...)` – works on any model that can `predict()` (or `forward()`) on the same batch structure that the data builder produces.
- **inference**: `save_model()` / `load_model()` (SavedModel); `Predictor` / `BatchPredictor` for batch prediction. Custom layers/models (e.g. GNN-RNN) are registered so load gives back the same class.

So: **evaluation and inference APIs are static**; **batch structure and model class are model-dependent**.

### 3.4 End-to-end for GNN-RNN

1. **Data**: `build_gnn_data(...)` → `GnnDataResult(train_ds, val_ds, proj_ds, metadata)`. Inputs are dicts (trade_features, adjacency_matrix, pnl_history, indices).
2. **Model**: `HybridGnnRnn(model_config)` – built from `models/gnn_rnn_hybrid/` (layers: GNN, RNN, fusion, attention, projection).
3. **Training**: Either `TrainingManager` (calibration/training_manager.py) or `Trainer` + your own fit call; both use the same train_ds/val_ds.
4. **Evaluation**: Same `Evaluator` / `evaluate_model()` on `proj_ds` or a test set.
5. **Inference**: `save_model()` / `load_model()`; `Predictor` with the same input dict structure.

The framework (pipelines, training, evaluation, inference) does not need to know that the model is GNN-RNN; it only needs datasets and a model that can be called on those batches.

---

## 4. Where to add a new model (e.g. your hybrid GNN-RNN)

To add a new model family while reusing the same framework:

1. **data/<your_model>/**  
   - Implement `build_*_data()` that returns a result with at least `train_ds`, `val_ds` (and optionally `test_ds` or `proj_ds`).  
   - Do all input construction and preprocessing here (graphs, sequences, tabular, etc.).  
   - Use `data/common` (e.g. `TradeGraphBuilder`, `TradeAttributeEncoder`) if you share the same notion of “trade graph”.

2. **models/<your_model>/**  
   - Implement your architecture (e.g. `tf.keras.Model` or a class that implements `Trainable`).  
   - Use a config dict or dataclass (e.g. `default_*_model_config()`).  
   - If you use custom layers, put them under `models/<your_model>/layers/` (as in `gnn_rnn_hybrid/layers/`).

3. **Optional glue**  
   - If you want a single entry point (like `TrainingManager` for GNN-RNN), add a small module that imports your model and your `build_*_data`, and calls `Trainer` or `run_training` with the right config.

4. **Registry / inference**  
   - Reuse existing `save_model` / `load_model` and `ModelRegistry`; register custom classes so loading returns the right type.

No change is required in **core**, **training**, **pipelines**, **evaluation**, **inference**, **tuning**, or **registry** unless you need a new callback type or a new protocol method.

---

## 5. Summary table

| Layer | Static (framework) | Model-dependent |
|-------|--------------------|------------------|
| **core** | Protocols, config, base classes, tracking | — |
| **data** | dataset.py, types, common | pricing/, calibration/, gnn_rnn_hybrid/ (and your new builders) |
| **models** | — | pricing/, gnn_rnn_hybrid/ (and your new model + layers) |
| **training** | Trainer, fit, callbacks | — |
| **pipelines** | run_training, evaluate, load/save, tuning | — |
| **evaluation** | Evaluator, metrics, result type | — |
| **inference** | save/load, Predictor | custom_objects when loading your model class |
| **tuning** | search space, run_tuning | — |
| **registry** | versioning, stages | — |
| **calibration** | — | TrainingManager (GNN-RNN wiring) |
| **utilities** | encoders, graph builder | — |

This keeps the framework **simple and generic** (same training/evaluation/inference/registry for everything) while giving **full flexibility** in data preprocessing and model architecture under `data/<model>/` and `models/<model>/`.
