# Machine Learning Framework (Phase 7.1)

Technical reference for the QuantStrata ML integration: pipelines, data preparation, models, and inference.

**Status:** Phase 7.1 complete. See `docs/development/progress/phase_7_1_implementation_notes.md` for alignment with the roadmap and review items.

---

## 1. Overview

The ML framework is **model-agnostic** and reusable:

1. **Build model instance** — e.g. NN pricer, calibration net, Hybrid GNN-LSTM.
2. **Prepare data** — Standardised builders (pricing, calibration, GNN) output `tf.data.Dataset` or `MLDataset`.
3. **Train** — Generic pipeline (`pipelines/training`, `training/trainer`) or model-specific `TrainingManager` (Hybrid GNN).
4. **Evaluate** — Common metrics and logging (`evaluation/`, `pipelines/evaluation`).
5. **Inference** — Load model and predict (`pipelines/inference`, `inference/model_io`).

---

## 2. Module Layout

| Path | Purpose |
|------|---------|
| `core/protocols.py` | `Trainable` protocol, `KerasTrainableAdapter` |
| `core/types.py` | `TrainingConfig`, `TrainingResult`, `EvaluationResult` (pipeline) |
| `core/config.py` | TF/Keras training config (EarlyStopping, Checkpoint, Optimizer) |
| `pipelines/training.py` | `run_training()`, `TrainingLoop` (Trainable + NumPy) |
| `pipelines/evaluation.py` | `evaluate_model()` |
| `pipelines/inference.py` | `save_model()`, `load_model()`, `predict()` (Trainable + JSON) |
| `training/trainer.py` | TensorFlow-native `Trainer` |
| `evaluation/` | `Evaluator`, `metrics` |
| `inference/model_io.py` | TF/Keras save/load (SavedModel, weights) |
| `inference/predictor.py` | Inference utilities |
| `data/dataset.py` | `TFDataset`, `create_pricing_dataset`, `create_calibration_dataset` |
| `data/pricing/build.py` | `build_pricing_data`, `build_pricing_dataset_from_mc`, `from_analytic` |
| `data/calibration/build.py` | `build_calibration_dataset`, `CalibrationDataResult` |
| `data/portfolio.py` | Re-exports `build_gnn_dataset_from_portfolio`, `gnn_inputs_to_tf_dataset` |
| `data/gnn_rnn_hybrid/` | `build_gnn_data`, `GnnDataResult`, synthetic, dataset_utils |
| `calibration/training_manager.py` | `TrainingManager`, `TrainingConfiguration` (Keras/HybridGnnRnn) |
| `models/pricing/` | NN pricer model |
| `models/gnn_rnn_hybrid/` | Hybrid GNN-RNN (layers, config, model) |
| `utilities/` | `TradeAttributeEncoder`, `TradeGraphBuilder` |

---

## 3. Training Paths

- **Generic (Trainable):** `run_training(model, train_data, config)` in `pipelines/training.py` — any model implementing `Trainable` (e.g. via `KerasTrainableAdapter`).
- **TensorFlow-native:** `Trainer` in `training/trainer.py` — Keras models with `model.fit()`-style loop, checkpointing, callbacks.
- **Hybrid GNN-LSTM:** `TrainingManager` in `calibration/training_manager.py` — builds `HybridGnnRnn`, compiles, fits with `model.fit()`, stores history; use `manager.run(stages=[TrainingConfiguration(...)])`.

---

## 4. Data Contracts

- **Pricing:** `build_pricing_data()` → `PricingDataResult(train_ds, val_ds, test_ds, feature_stats, target_stats, metadata)`. From MC/analytic: `build_pricing_dataset_from_mc()`, `build_pricing_dataset_from_analytic()` → `MLDataset`.
- **Calibration:** `build_calibration_dataset()` → `MLDataset` (features = IV/market observables, targets = model parameters). `CalibrationDataResult` for structured output with `to_ml_dataset()`.
- **GNN-RNN:** `build_gnn_data(use_synthetic=True|False)` → `GnnDataResult(train_ds, val_ds, proj_ds, metadata)`. Inputs: `trade_features`, `adjacency_matrix`, `pnl_history`, `target_indices`, `elementary_indices`.

---

## 5. Inference Paths

- **Trainable (JSON):** `save_model()`, `load_model()`, `predict()` in `pipelines/inference.py` — serialises parameters to JSON and restores via a factory; good for generic Trainable models.
- **TF/Keras:** `inference/model_io.py` — `save_model` / `load_model` for full SavedModel or weights (`.weights.h5`); use for Hybrid GNN and Keras pricers.

---

## 6. Configuration Notes

- **Two `TrainingConfig` types:** `core/types.TrainingConfig` for the pipeline (NumPy/Trainable); `core/config.TrainingConfig` (and related) for TF Trainer. `TrainingConfiguration` in `calibration/training_manager.py` is used only by `TrainingManager`. See phase 7.1 implementation notes for optional unification.
- **Hybrid GNN:** `default_hybrid_model_config(n_targets=...)` from `models/gnn_rnn_hybrid/config.py`; pass to `HybridGnnRnn(model_config)` and to `TrainingManager(..., model_config=...)`.

---

## 7. References

- Progress: `docs/development/progress/phase_7_1_implementation_notes.md`, `phase_7_1_machine_learning_integration.md`
- Component reference: `docs/architecture/component_reference.md` (Section 11 — machine_learning)
- Tutorials: `docs/tutorials/machine_learning/` (ML lifecycle, Hybrid GNN-LSTM)
- Guide: [ML Framework Guide](../../guides/machine_learning/ml_framework.md)
