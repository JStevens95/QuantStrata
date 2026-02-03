# Phase 7.1: Machine Learning Integration - Progress Report

**Status:** In Progress (Tasks 1–4 Complete)  
**Started:** January 2026

---

## Overview

Phase 7.1 implements a **model-agnostic ML framework** for QuantStrata: generic training/evaluation/inference pipelines, standardised data preparation, ML-based pricing and calibration, and completion of the GNN-RNN hybrid pricer.

The goal is: **build model instance → prepare data → generic training loop → standardised evaluation → generalised inference** — reusable across NN pricers, calibration nets, and the GNN-LSTM pricer.

---

## Architecture

```
src/m_learning/
├── __init__.py
├── core/                          # Protocols and types
│   ├── __init__.py
│   ├── protocols.py               # Trainable protocol
│   └── types.py                   # TrainingConfig, EvaluationResult, etc.
├── pipeline/                      # Generic ML pipeline
│   ├── __init__.py
│   ├── training.py                # run_training(), TrainingLoop
│   ├── evaluation.py              # evaluate_model(), metrics
│   └── inference.py               # load_model(), predict()
├── data/                          # Data preparation
│   ├── __init__.py
│   ├── types.py                   # MLDataset, feature/target schemas
│   ├── pricing.py                 # MC/analytic → pricing dataset
│   ├── calibration.py             # Market → calibration dataset
│   └── portfolio.py               # Portfolio → GNN inputs
├── calibration/                   # (existing) TrainingManager
│   └── training_manager.py
├── models/                        # (existing) GNN-RNN hybrid
│   └── gnn_rnn_hybrid/
└── utilities/                     # (existing) TradeGraphBuilder, encoder
    ├── trade_attribute_encoder.py
    └── trade_graph_builder.py
```

---

## Implementation Tasks

### 1. Trainable Protocol + Generic Training Pipeline

**Status:** COMPLETE

| Component | File | Status |
|-----------|------|--------|
| Trainable protocol | `src/m_learning/core/protocols.py` | ✅ |
| KerasTrainableAdapter | `src/m_learning/core/protocols.py` | ✅ |
| Training config types | `src/m_learning/core/types.py` | ✅ |
| Generic training loop | `src/m_learning/pipeline/training.py` | ✅ |
| Unit tests | `tests/unit/m_learning/pipeline/test_training.py` | ✅ |
| Unit tests | `tests/unit/m_learning/core/test_types.py` | ✅ |

**Deliverables:**
- [x] `Trainable` protocol (forward/call, compute_loss, get/set_parameters)
- [x] `KerasTrainableAdapter` for wrapping Keras models
- [x] `TrainingConfig` dataclass (epochs, lr, checkpoint path, early stopping, etc.)
- [x] `run_training(model, train_data, config)` function
- [x] `TrainingLoop` class with batching, validation, checkpointing
- [x] Checkpointing (save best, periodic)
- [x] Early stopping
- [x] Epoch logging (loss, val_loss)
- [x] Unit tests (9 tests for types, 10 tests for training)

---

### 2. Data Preparation for ML

**Status:** COMPLETE

| Component | File | Status |
|-----------|------|--------|
| MLDataset type | `src/m_learning/data/types.py` | ✅ |
| PricingFeatures schema | `src/m_learning/data/types.py` | ✅ |
| CalibrationFeatures schema | `src/m_learning/data/types.py` | ✅ |
| Pricing data adapter (MC) | `src/m_learning/data/pricing.py` | ✅ |
| Pricing data adapter (analytic) | `src/m_learning/data/pricing.py` | ✅ |
| Calibration data adapter | `src/m_learning/data/calibration.py` | ✅ |
| Portfolio/GNN adapter | `src/m_learning/data/portfolio.py` | ✅ |
| Unit tests | `tests/unit/m_learning/data/` | ✅ |

**Deliverables:**
- [x] `MLDataset` (features, targets, feature_names, split method)
- [x] `PricingFeatures` (spot, strike, vol, rate, expiry, option_type)
- [x] `CalibrationFeatures` (market_quotes, strikes, expiries, spot)
- [x] `build_pricing_dataset_from_mc()` — MC paths → pricing dataset
- [x] `build_pricing_dataset_from_analytic()` — analytic pricer → pricing dataset
- [x] `build_calibration_dataset()` — forward model → calibration dataset
- [x] `build_gnn_dataset_from_portfolio()` — portfolio → GNN inputs
- [x] `gnn_inputs_to_tf_dataset()` — convert to TensorFlow Dataset
- [x] Unit tests (21 tests for data types and pricing)

---

### 3. Standardised ML Evaluation Outputs

**Status:** COMPLETE

| Component | File | Status |
|-----------|------|--------|
| EvaluationResult type | `src/m_learning/core/types.py` | ✅ |
| evaluate_model() | `src/m_learning/pipeline/evaluation.py` | ✅ |
| Metric functions | `src/m_learning/pipeline/evaluation.py` | ✅ |
| Serialisation | `src/m_learning/core/types.py` | ✅ |
| Unit tests | `tests/unit/m_learning/pipeline/test_evaluation.py` | ✅ |

**Deliverables:**
- [x] `EvaluationResult` (loss, metrics, loss_curves, pricing_error, metadata)
- [x] `evaluate_model(model, data, config)` function
- [x] Built-in metrics: MSE, MAE, RMSE, MAPE, R²
- [x] Benchmark comparison (pricing_error vs analytic pricer)
- [x] JSON serialisation of results
- [x] Unit tests (10 tests for evaluation)

---

### 4. Generalised ML Inference Pipeline

**Status:** COMPLETE

| Component | File | Status |
|-----------|------|--------|
| save_model() | `src/m_learning/pipeline/inference.py` | ✅ |
| load_model() | `src/m_learning/pipeline/inference.py` | ✅ |
| predict() | `src/m_learning/pipeline/inference.py` | ✅ |
| Artifact convention | `src/m_learning/pipeline/inference.py` (docstring) | ✅ |
| Integration test | `tests/unit/m_learning/pipeline/test_inference.py` | ✅ |

**Deliverables:**
- [x] `save_model(model, artifact_dir, config, metadata)` — save parameters + config + metadata
- [x] `load_model(artifact_dir, model_factory)` → model instance
- [x] `predict(model, inputs, batch_size)` → outputs (with optional batching)
- [x] Artifact layout: `parameters.json`, `config.json`, `metadata.json`
- [x] End-to-end test: train → save → load → predict
- [x] Unit tests (10 tests for inference)

---

### 5–7. ML Pricing, Calibration, GNN Pricer

(To be completed after 1–4)

---

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_types.py` (core) | 9 | ✅ |
| `test_training.py` | 10 | ✅ |
| `test_evaluation.py` | 10 | ✅ |
| `test_inference.py` | 10 | ✅ |
| `test_types.py` (data) | 10 | ✅ |
| `test_pricing.py` | 9 | ✅ |
| **Total** | **58** | ✅ |

---

## Documentation

- Tutorial (TensorFlow): `docs/tutorials/m_learning/ml_pipeline_tensorflow.ipynb` ✅
- Tutorial (NumPy): `docs/tutorials/m_learning/ml_pipeline_introduction.ipynb` ✅
- Reference: `docs/reference/m_learning/ml_framework.md` (planned)
- Guide: `docs/guides/m_learning/ml_pipeline.md` (planned)

## TensorFlow-Native Refactoring (Completed)

The ML module has been refactored to be fully TensorFlow-native:

### New Structure
```
src/m_learning/
├── core/
│   ├── base.py         # BaseModel, PricingModel, CalibrationModel, PortfolioModel
│   ├── config.py       # TrainingConfig, OptimizerConfig, LRScheduleConfig, etc.
│   ├── callbacks.py    # MetricsLogger, PricingErrorCallback, TrainingProgressCallback
│   └── types.py        # Legacy types (backward compatibility)
├── data/
│   ├── dataset.py      # TFDataset, NormalizationStats, create_pricing_dataset
│   └── ...             # Legacy adapters
├── models/
│   ├── pricing/
│   │   └── mlp_pricer.py  # MLPPricer, ResidualMLPPricer
│   └── gnn_rnn_hybrid/    # Fixed imports
├── training/
│   └── trainer.py      # Trainer class, TrainingResult
├── evaluation/
│   ├── evaluator.py    # Evaluator class, EvaluationResult
│   └── metrics.py      # PricingMetrics, CalibrationMetrics
└── inference/
    ├── model_io.py     # save_model, load_model, ModelArtifact
    └── predictor.py    # Predictor, BatchPredictor
```

### Key Features
- `tf.data.Dataset` integration for efficient batching
- `tf.keras.Model` base classes with metadata tracking
- TensorFlow SavedModel format for serialization
- Automatic normalization/denormalization
- Greeks via automatic differentiation
- Uncertainty estimation via MC Dropout

---

## Files Changed

### New Files (Tasks 1–4)
- `src/m_learning/core/__init__.py` — Core exports
- `src/m_learning/core/protocols.py` — Trainable protocol, KerasTrainableAdapter
- `src/m_learning/core/types.py` — TrainingConfig, TrainingResult, EvaluationResult, CheckpointInfo
- `src/m_learning/pipeline/__init__.py` — Pipeline exports
- `src/m_learning/pipeline/training.py` — run_training(), TrainingLoop
- `src/m_learning/pipeline/evaluation.py` — evaluate_model(), metric functions
- `src/m_learning/pipeline/inference.py` — save_model(), load_model(), predict()
- `src/m_learning/data/__init__.py` — Data exports
- `src/m_learning/data/types.py` — MLDataset, PricingFeatures, CalibrationFeatures
- `src/m_learning/data/pricing.py` — build_pricing_dataset_from_mc/analytic
- `src/m_learning/data/calibration.py` — build_calibration_dataset
- `src/m_learning/data/portfolio.py` — build_gnn_dataset_from_portfolio, gnn_inputs_to_tf_dataset
- `tests/unit/m_learning/__init__.py`
- `tests/unit/m_learning/core/__init__.py`
- `tests/unit/m_learning/core/test_types.py` — 9 tests
- `tests/unit/m_learning/pipeline/__init__.py`
- `tests/unit/m_learning/pipeline/test_training.py` — 10 tests
- `tests/unit/m_learning/pipeline/test_evaluation.py` — 10 tests
- `tests/unit/m_learning/pipeline/test_inference.py` — 10 tests
- `tests/unit/m_learning/data/__init__.py`
- `tests/unit/m_learning/data/test_types.py` — 10 tests
- `tests/unit/m_learning/data/test_pricing.py` — 9 tests

### Modified Files
- `docs/development/roadmap.md` (to be updated with checkboxes)
