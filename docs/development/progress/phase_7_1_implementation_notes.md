# Phase 7.1 Implementation Notes: Roadmap vs Codebase

**Purpose:** Align roadmap.md (Phase 7.1) with the current folder structure and list **code that needs review/replacement** and **elements not yet implemented**. Folder structure is kept as-is.

---

## 1. Folder Structure (Kept the Same)

Current layout:

```
src/machine_learning/
├── core/                    # protocols, types, base, config, callbacks
├── data/
│   ├── dataset.py           # TFDataset, create_pricing_dataset, create_calibration_dataset
│   ├── types.py             # MLDataset, PricingFeatures, CalibrationFeatures
│   ├── common/              # TradeAttributeEncoder, TradeGraphBuilder (re-export from utilities)
│   ├── pricing/             # build_pricing_data, build_pricing_dataset_from_mc/analytic
│   ├── calibration/         # build_calibration_dataset, CalibrationDataResult
│   ├── portfolio.py         # build_gnn_dataset_from_portfolio, gnn_inputs_to_tf_dataset (re-export)
│   └── gnn_rnn_hybrid/      # build_gnn_data, GnnDataResult, synthetic, dataset_utils, etc.
├── pipelines/               # training.py, evaluation.py, inference.py, tuning.py (NumPy/Trainable)
├── training/                # Trainer (TensorFlow-native)
├── evaluation/              # Evaluator, metrics
├── inference/               # model_io (save_model, load_model), predictor
├── calibration/             # TrainingManager (Keras/HybridGnnRnn-specific)
├── models/                  # pricing/, gnn_rnn_hybrid/
└── utilities/               # trade_attribute_encoder, trade_graph_builder
```

---

## 2. Code That Is Issues / Needs Replacing (For Your Review)

These are inconsistencies or legacy names that may need to be replaced or refactored.

| Item | Location | Issue | Suggested action |
|------|----------|--------|------------------|
| **Pipeline vs inference** | `pipelines/inference.py` (NumPy, Trainable, JSON params) vs `inference/model_io.py` (TF SavedModel, Keras) | Two inference paths: (1) Generic `save_model`/`load_model`/`predict` in `pipelines/inference.py` for `Trainable` + JSON weights; (2) TensorFlow `save_model`/`load_model` in `inference/model_io.py`. Hybrid GNN and MLP pricer use the TF path. | Decide: deprecate `pipelines/inference` for Keras models and use `inference/model_io` only; or keep both and document when to use which. |
| **TrainingManager vs Trainer** | `calibration/training_manager.py` vs `training/trainer.py` | `TrainingManager` is used by the Hybrid GNN-LSTM tutorial (model-specific, Keras fit); `Trainer` is the generic TensorFlow training loop. Roadmap says "generic ML training pipeline". | Either make HybridGnnRnn trainable via generic `Trainer` (e.g. KerasTrainableAdapter) and document, or keep TrainingManager as the model-specific entry point and document that Phase 7.1 "generic pipeline" is fulfilled by `Trainer` + `pipelines/training` for other models. |
| **Training config types** | `core/types.py` has `TrainingConfig`; `core/config.py` has `TrainingConfig` (and EarlyStoppingConfig, CheckpointConfig, etc.) | Two different config dataclasses: `core/types.TrainingConfig` for pipeline (NumPy/Trainable); `core/config.TrainingConfig` for TF Trainer/Keras. | Unify or clearly separate: e.g. document `core/types.TrainingConfig` for pipeline/Trainable, `core/config.TrainingConfig` for TF Trainer; consider renaming one to `TFTrainingConfig` or `PipelineTrainingConfig` to avoid confusion. |
| **build_pricing_dataset_from_mc / from_analytic** | `data/pricing/build.py` | Wrappers delegate to `create_pricing_dataset` with optional `pricing_fn`. For full MC/analytic integration they may need real pricer instance or path generator instead of synthetic-only. | Review: ensure from_mc/from_analytic accept real MC paths or pricer callables and integrate with library pricers when used in production. |
| **build_calibration_dataset** | `data/calibration/build.py` | Uses `create_calibration_dataset` (synthetic IV surface from random Heston/SABR params). `forward_model_fn` is optional and not yet wired into the synthetic path. | Review signature and integration with real calibration workflow; wire forward_model_fn if you need params → observables for real calibration. |
| **Component reference doc** | `docs/architecture/component_reference.md` | Previously referred to `pipeline/` (training, evaluation, inference) and `data/pricing.py`, `data/calibration.py`, `data/portfolio.py`. Actual layout uses `pipelines/` and `data/pricing/`, `data/calibration/`, `data/portfolio.py`. | Updated in this pass to match current folders and export names (see Section 4 below). |

---

## 3. Elements Not Implemented Yet (Now Added)

The following were missing and have been added without changing the folder structure.

| Roadmap / progress item | Added where | What was added |
|-------------------------|------------|----------------|
| **build_pricing_dataset_from_mc** | `data/pricing/build.py` | `build_pricing_dataset_from_mc(n_samples, paths_or_fn, ...)` → returns `MLDataset`. Uses `create_pricing_dataset` with an optional pricing function (e.g. MC-based). For full MC integration, pass a callable that computes price from MC paths. |
| **build_pricing_dataset_from_analytic** | `data/pricing/build.py` | `build_pricing_dataset_from_analytic(n_samples, pricer_fn, ...)` → returns `MLDataset`. Uses `create_pricing_dataset(..., pricing_fn=pricer_fn)`. |
| **build_calibration_dataset** | `data/calibration/build.py` | `build_calibration_dataset(...)` that returns an `MLDataset` for calibration; delegates to `create_calibration_dataset`. `CalibrationDataResult` and `forward_model_fn` parameter for future forward-model integration. |
| **data/portfolio.py** | `data/portfolio.py` | Re-exports `build_gnn_dataset_from_portfolio` and `gnn_inputs_to_tf_dataset` from `data/gnn_rnn_hybrid/dataset_utils.py` so roadmap-style imports work. |
| **tests/unit/machine_learning/pipelines/test_inference.py** | Already present | Minimal test for save_model → load_model → predict with KerasTrainableAdapter. |
| **tests/unit/machine_learning/data/calibration** | Added | Minimal test for `build_calibration_dataset` (see Section 5). |

---

## 4. Component Reference Update (machine_learning section)

`docs/architecture/component_reference.md` Section 11 (machine_learning) has been updated to:

- Use **pipelines/** (not pipeline/) with training.py, evaluation.py, inference.py, tuning.py.
- Use **data/dataset.py**, **data/pricing/** (build.py), **data/calibration/** (build.py), **data/portfolio.py**, **data/gnn_rnn_hybrid/**.
- Include **training/trainer.py**, **evaluation/** (evaluator, metrics), **inference/** (model_io, predictor).
- List key exports: Trainable, run_training, evaluate_model, save_model/load_model/predict (pipelines), model_io (inference), build_pricing_data, build_pricing_dataset_from_mc/from_analytic, build_calibration_dataset, build_gnn_dataset_from_portfolio, gnn_inputs_to_tf_dataset, build_gnn_data, TrainingManager.

---

## 5. Roadmap Checklist (Phase 7.1) – Status

| Roadmap item | Status | Notes |
|--------------|--------|-------|
| Generic ML training pipeline | Done | `pipelines/training.py` (Trainable + NumPy), `training/trainer.py` (TF); TrainingManager for Hybrid GNN. |
| Data preparation for ML | Done | dataset.py, pricing/build.py, gnn_rnn_hybrid/build.py; calibration/build.py and portfolio.py in place. |
| Standardised ML evaluation outputs | Done | `pipelines/evaluation.py`, `evaluation/evaluator.py`, `evaluation/metrics.py`. |
| Generalised ML inference pipeline | Done | `pipelines/inference.py` (generic Trainable), `inference/model_io.py` (TF/Keras). |
| ML-based pricing | Done | `models/pricing/`, `create_pricing_dataset`, `build_pricing_data`; from_mc/from_analytic in data/pricing/build.py. |
| ML calibration | Done | `create_calibration_dataset` in dataset.py, `build_calibration_dataset` in data/calibration/build.py. |
| Hybrid GNN-LSTM full revaluation pricer | Done | `models/gnn_rnn_hybrid/`, `data/gnn_rnn_hybrid/`, TrainingManager; integrate with generic pipeline per "Code issues" above. |

---

## 6. Summary for Review

- **Replace / refactor:** Unify or document the two inference paths (pipelines vs inference/model_io), the two training paths (TrainingManager vs Trainer), and the two TrainingConfig types. Component reference updated to match current layout.
- **New code to review:** `data/pricing/build.py` (from_mc, from_analytic), `data/calibration/build.py` (build_calibration_dataset), `data/portfolio.py` (re-exports). Ensure from_mc/from_analytic and build_calibration_dataset match how you plan to use MC/analytic pricers and calibration in production.
