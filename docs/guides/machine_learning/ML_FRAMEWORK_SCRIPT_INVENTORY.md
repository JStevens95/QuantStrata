# Machine Learning Framework — Comprehensive Script Inventory

Master list of all scripts for a generic ML framework, with HybridGnnRnn-specific wiring and implementation status.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✓ | HybridGnnRnn uses / depends on this |
| ✗ | Not used by HybridGnnRnn (generic or other models) |
| 🔌 | Requires plug-in/wiring for HybridGnnRnn |

---

## 1. Core (Framework Infrastructure)

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `core/__init__.py` | Exports base classes, configs | ✓ | BaseModel, configs |
| `core/base.py` | Abstract model bases | ✓ | BaseModel, PricingModel, CalibrationModel, PortfolioModel |
| `core/config.py` | Training/config dataclasses | ✓ | TrainingConfig, OptimizerConfig, EarlyStoppingConfig, CheckpointConfig, ModelConfig |
| `core/protocols.py` | Trainable protocol, adapters | ✓ | Trainable, KerasTrainableAdapter |
| `core/types.py` | Training types | ✓ | TrainingConfig, TrainingResult, CheckpointInfo |
| `core/tracking.py` | Experiment tracking | ✓ | ExperimentTracker, MLflowTracker, WandBTracker, InMemoryTracker |

*Note: callbacks.py lives in `training/callbacks.py`.*

---

## 2. Data

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `data/__init__.py` | Exports datasets, builders | ✓ | build_pricing_data, build_gnn_data, TFDataset |
| `data/dataset.py` | TF dataset wrapper, normalization | ✓ | TFDataset, NormalizationStats |
| `data/types.py` | Data types | ✓ | MLDataset, PricingFeatures, CalibrationFeatures |
| `data/common/__init__.py` | Shared data helpers | ✗ | — |
| `data/pricing/__init__.py` | Pricing data exports | ✗ | — |
| `data/pricing/build.py` | Pricing data builder | ✗ | build_pricing_data |
| `data/calibration/__init__.py` | Calibration data exports | ✗ | — |
| `data/calibration/build.py` | Calibration dataset builder | ✗ | build_calibration_dataset |
| `data/gnn_rnn_hybrid/__init__.py` | GNN data exports | ✓ | build_gnn_data, GnnDataResult |
| `data/gnn_rnn_hybrid/build.py` | **GNN data builder** | ✓ | build_gnn_data, GnnDataResult |
| `data/gnn_rnn_hybrid/synthetic.py` | Synthetic GNN data | ✓ | generate_synthetic_gnn_data, SyntheticGnnData |
| `data/gnn_rnn_hybrid/dataset_utils.py` | GNN → tf.data conversion | ✓ | gnn_inputs_to_tf_dataset |
| `data/gnn_rnn_hybrid/portfolio_builder.py` | FX portfolio → GNN format | ✓ | build_fx_gnn_data, GnnPortfolioData |

---

## 3. Models

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `models/__init__.py` | Exports models | ✓ | MLPPricer, HybridGnnRnn |
| `models/pricing/__init__.py` | Pricing model exports | ✗ | — |
| `models/pricing/model.py` | MLP pricer | ✗ | MLPPricer |
| `models/pricing/config.py` | Pricing config | ✗ | — |
| `models/gnn_rnn_hybrid/__init__.py` | GNN model exports | ✓ | HybridGnnRnn, default_hybrid_model_config |
| `models/gnn_rnn_hybrid/model.py` | **HybridGnnRnn** | ✓ | HybridGnnRnn(BaseModel) |
| `models/gnn_rnn_hybrid/config.py` | GNN config | ✓ | default_hybrid_model_config |
| `models/gnn_rnn_hybrid/layers/__init__.py` | Layer exports | ✓ | GnnBlock, RnnBlock, FusionLayer, etc. |
| `models/gnn_rnn_hybrid/layers/gnn_layers.py` | GNN blocks | ✓ | GnnBlock, GraphSage, MixedGraphSage |
| `models/gnn_rnn_hybrid/layers/rnn_layers.py` | RNN blocks | ✓ | RnnBlock |
| `models/gnn_rnn_hybrid/layers/fusion_layer.py` | Fusion layer | ✓ | FusionLayer |
| `models/gnn_rnn_hybrid/layers/attention_layer.py` | Target attention | ✓ | TargetAttentionLayer |
| `models/gnn_rnn_hybrid/layers/projection_layer.py` | PnL projection | ✓ | TargetPnlOutput |

---

## 4. Utilities

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `utilities/__init__.py` | Exports utilities | ✓ | — |
| `utilities/trade_attribute_encoder.py` | Encode trade attributes | ✓ | TradeAttributeEncoder |
| `utilities/trade_graph_builder.py` | Build adjacency matrix | ✓ | TradeGraphBuilder |

---

## 5. Training (Generic for All Models)

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `training/__init__.py` | Exports Trainer | ✓ | — |
| `training/callbacks.py` | Keras callbacks | ✓ | get_standard_callbacks |
| `training/trainer.py` | Generic Keras trainer | ✓ | Trainer, TrainingResult, fit_model |

*TrainingManager omitted — use Trainer for all models including HybridGnnRnn.*

---

## 7. Pipelines

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `pipelines/__init__.py` | Exports pipelines | ✓ | — |
| `pipelines/training.py` | Generic training loop | ✓ | run_training, TrainingLoop (Trainable protocol) |
| `pipelines/evaluation.py` | Evaluation pipeline | ✓ | run_evaluation |
| `pipelines/inference.py` | Inference pipeline | ✓ | run_inference, save_model, load_model |
| `pipelines/tuning.py` | Optuna tuning pipeline | ✓ | run_optuna_tuning integration |

**🔌 HybridGnnRnn:** Trainer.fit() and TrainingManager both work; pipelines use tf.data.Dataset (dict inputs OK).

---

## 8. Evaluation

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `evaluation/__init__.py` | Exports Evaluator | ✓ | — |
| `evaluation/evaluator.py` | Evaluator, EvaluationResult | ✓ | Evaluator, evaluate, plot_predictions, plot_residuals |
| `evaluation/metrics.py` | Metric computation | ✓ | compute_metrics, PricingMetrics |

**🔌 HybridGnnRnn:** Evaluator.evaluate() expects TFDataset or (features, targets). For GNN dict inputs, add branch for `tf.data.Dataset` or provide `GnnEvaluator` / extend `evaluate()` to accept dataset iterator.

---

## 9. Inference

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `inference/__init__.py` | Exports save/load, Predictor | ✓ | — |
| `inference/model_io.py` | Save/load models | ✓ | save_model, load_model, ModelArtifact |
| `inference/predictor.py` | Prediction wrapper | ✓ | Predictor, BatchPredictor |

**🔌 HybridGnnRnn:** `Predictor` assumes `features: np.ndarray`. For GNN: add `predict_dict()` or `predict_from_dataset()`, or extend `predict()` to accept `Dict[str, np.ndarray]` / `tf.data.Dataset`.

---

## 10. Registry

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `registry/__init__.py` | Exports ModelRegistry | ✓ | — |
| `registry/registry.py` | Model versioning | ✓ | ModelRegistry, ModelVersion, ModelStage |

---

## 11. Tuning

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `tuning/__init__.py` | Exports tuning | ✓ | — |
| `tuning/search_space.py` | Optuna search space | ✓ | SearchSpace, run_optuna_tuning |

---

## 12. Features (Hedge-Fund Additions)

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `features/__init__.py` | Exports feature layer | ✓ | — |
| `features/schema.py` | Feature schemas | ✓ | FeatureSchema, GnnFeatureSchema |
| `features/registry.py` | Feature transform registry | ✓ | FeatureRegistry, get_registry |
| `features/transforms/__init__.py` | Transform exports | ✓ | — |
| `features/transforms/standardiser.py` | Z-score/minmax | ✓ | Standardiser |
| `features/gnn.py` | GNN feature builders | ✓ | Re-export or wrap TradeAttributeEncoder, TradeGraphBuilder |

---

## 13. Validation (Hedge-Fund Additions)

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `validation/__init__.py` | Exports validators | ✓ | — |
| `validation/base.py` | ValidationResult, Validator protocol | ✓ | — |
| `validation/inputs.py` | validate_features, validate_targets | ✓ | — |
| `validation/schema.py` | validate_against_schema | ✓ | — |
| `validation/deployment_gates.py` | DeploymentGate, ThresholdGate | ✓ | — |
| `validation/models/gnn_rnn_hybrid.py` | GNN-specific validation | ✓ | Optional: graph structure checks |

---

## 14. Monitoring (Hedge-Fund Additions)

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `monitoring/__init__.py` | Exports monitoring | ✓ | — |
| `monitoring/baselines.py` | BaselineStore, save/load | ✓ | — |
| `monitoring/drift.py` | check_feature_drift | ✓ | — |
| `monitoring/logger.py` | PredictionLogger | ✓ | — |
| `monitoring/alerts.py` | AlertHandler | ✓ | — |

---

## 15. Ensemble (Hedge-Fund Additions)

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `ensemble/__init__.py` | Exports ensemble | ✓ | — |
| `ensemble/registry.py` | KeyedModelRegistry | ✓ | — |
| `ensemble/ensemble.py` | KeyedModelEnsemble | ✓ | — |
| `ensemble/routers/__init__.py` | Router exports | ✓ | — |
| `ensemble/routers/cluster.py` | ClusterRouter | ✓ | key_for_trade, keys_for_scenarios |

---

## 16. Logging (Optional)

| Script | Purpose | GNN? | Contents |
|--------|---------|------|----------|
| `logging/__init__.py` | Exports configure_ml_logging | ✓ | — |
| `logging/setup.py` | configure_ml_logging | ✓ | Console + file logging |

---

## HybridGnnRnn Integration Checklist

Use this to verify each component is wired for HybridGnnRnn:

| Component | Wire-In Required |
|-----------|-------------------|
| **TrainingManager.build_model** | `if stage.model == "HybridGnnRnn": return HybridGnnRnn(model_config, name=stage.name)` |
| **build_gnn_data** | Returns train_ds, val_ds, proj_ds (dict inputs) |
| **Evaluator** | Support tf.data.Dataset with dict inputs, or GnnEvaluator |
| **Predictor** | Support dict inputs or predict_from_dataset() |
| **ModelRegistry** | Register HybridGnnRnn artifacts (generic) |
| **Tracking** | Log model_config, architecture in run params (generic) |
| **ClusterRouter** | Implement key_for_trade(trade) → cluster_id for per-cluster models |

---

## Desired Implementation Phases

Uses **generic Trainer** for all models (no TrainingManager). Callbacks live in `training/callbacks.py`.

---

### Phase 1: Core (Framework Foundation)

| Order | Script | Purpose |
|-------|--------|---------|
| 1.1 | `core/__init__.py` | Exports |
| 1.2 | `core/base.py` | BaseModel, PricingModel, CalibrationModel, PortfolioModel |
| 1.3 | `core/config.py` | TrainingConfig, OptimizerConfig, EarlyStoppingConfig, CheckpointConfig |
| 1.4 | `core/protocols.py` | Trainable protocol, KerasTrainableAdapter |
| 1.5 | `core/types.py` | TrainingResult, CheckpointInfo |
| 1.6 | `core/tracking.py` | ExperimentTracker, MLflow, W&B, InMemory |

---

### Phase 2: Utilities & Data

| Order | Script | Purpose |
|-------|--------|---------|
| 2.1 | `utilities/__init__.py` | Exports |
| 2.2 | `utilities/trade_attribute_encoder.py` | TradeAttributeEncoder |
| 2.3 | `utilities/trade_graph_builder.py` | TradeGraphBuilder |
| 2.4 | `data/__init__.py` | Exports |
| 2.5 | `data/types.py` | MLDataset, PricingFeatures, CalibrationFeatures |
| 2.6 | `data/dataset.py` | TFDataset, NormalizationStats |
| 2.7 | `data/common/__init__.py` | — |
| 2.8 | `data/gnn_rnn_hybrid/__init__.py` | Exports |
| 2.9 | `data/gnn_rnn_hybrid/synthetic.py` | generate_synthetic_gnn_data |
| 2.10 | `data/gnn_rnn_hybrid/dataset_utils.py` | gnn_inputs_to_tf_dataset |
| 2.11 | `data/gnn_rnn_hybrid/build.py` | build_gnn_data |

---

### Phase 3: Models

| Order | Script | Purpose |
|-------|--------|---------|
| 3.1 | `models/__init__.py` | Exports |
| 3.2 | `models/gnn_rnn_hybrid/__init__.py` | Exports |
| 3.3 | `models/gnn_rnn_hybrid/config.py` | default_hybrid_model_config |
| 3.4 | `models/gnn_rnn_hybrid/model.py` | HybridGnnRnn(BaseModel) |
| 3.5 | `models/gnn_rnn_hybrid/layers/__init__.py` | Layer exports |
| 3.6 | `models/gnn_rnn_hybrid/layers/gnn_layers.py` | GnnBlock, GraphSage |
| 3.7 | `models/gnn_rnn_hybrid/layers/rnn_layers.py` | RnnBlock |
| 3.8 | `models/gnn_rnn_hybrid/layers/fusion_layer.py` | FusionLayer |
| 3.9 | `models/gnn_rnn_hybrid/layers/attention_layer.py` | TargetAttentionLayer |
| 3.10 | `models/gnn_rnn_hybrid/layers/projection_layer.py` | TargetPnlOutput |

---

### Phase 4: Training (Generic Trainer)

| Order | Script | Purpose |
|-------|--------|---------|
| 4.1 | `training/__init__.py` | Exports |
| 4.2 | `training/callbacks.py` | get_standard_callbacks (move from core) |
| 4.3 | `training/trainer.py` | Trainer, TrainingResult, fit_model |

---

### Phase 5: Inference & Evaluation

| Order | Script | Purpose |
|-------|--------|---------|
| 5.1 | `inference/__init__.py` | Exports |
| 5.2 | `inference/model_io.py` | save_model, load_model, ModelArtifact |
| 5.3 | `inference/predictor.py` | Predictor, BatchPredictor (+ GNN dict support) |
| 5.4 | `evaluation/__init__.py` | Exports |
| 5.5 | `evaluation/metrics.py` | compute_metrics |
| 5.6 | `evaluation/evaluator.py` | Evaluator (+ GNN dataset support) |

---

### Phase 6: Pipelines

| Order | Script | Purpose |
|-------|--------|---------|
| 6.1 | `pipelines/__init__.py` | Exports |
| 6.2 | `pipelines/training.py` | run_training, TrainingLoop |
| 6.3 | `pipelines/evaluation.py` | run_evaluation |
| 6.4 | `pipelines/inference.py` | run_inference |

---

### Phase 7: Registry & Tracking

| Order | Script | Purpose |
|-------|--------|---------|
| 7.1 | `registry/__init__.py` | Exports |
| 7.2 | `registry/registry.py` | ModelRegistry, ModelVersion, ModelStage |
| 7.3 | (Optional) Model builder registry | build_model(model_type, **kwargs) |

---

### Phase 8: Tuning

| Order | Script | Purpose |
|-------|--------|---------|
| 8.1 | `tuning/__init__.py` | Exports |
| 8.2 | `tuning/search_space.py` | SearchSpace, run_optuna_tuning |

---

### Phase 9: Hedge-Fund Additions

| Order | Script | Purpose |
|-------|--------|---------|
| 9.1 | `features/` | schema, registry, standardiser |
| 9.2 | `validation/` | base, inputs, schema, deployment_gates |
| 9.3 | `monitoring/` | baselines, drift, logger, alerts |
| 9.4 | `ensemble/` | registry, ensemble, routers |
| 9.5 | `logging/setup.py` | configure_ml_logging |

---

### Phase 10: Optional / Deferred

| Script | Note |
|--------|------|
| `data/pricing/` | When adding MLPPricer |
| `data/calibration/` | When adding calibration models |
| `models/pricing/` | When adding MLPPricer |
| `calibration/training_manager.py` | Omit — use generic Trainer |

---

## Milestone Checkpoints

| Milestone | After Phase | Verification |
|-----------|-------------|--------------|
| **M1: Model runs forward pass** | Phase 3 | `model(inputs)` produces output |
| **M2: End-to-end training** | Phase 4 | `trainer.fit(train_ds, val_ds)` completes |
| **M3: Save & load** | Phase 5 | save_model → load_model → predict |
| **M4: Full pipeline** | Phase 6 | run_training, run_evaluation, run_inference |
| **M5: Production-ready** | Phase 9 | features, validation, monitoring, ensemble |  

---

## File Count Summary

| Category | Scripts |
|----------|--------|
| Core | 6 |
| Data | 12 |
| Models | 10 (+ layers) |
| Utilities | 3 |
| Training | 3 (incl. callbacks) |
| Pipelines | 5 |
| Evaluation | 3 |
| Inference | 3 |
| Registry | 2 |
| Tuning | 2 |
| Features | 6 |
| Validation | 5 |
| Monitoring | 5 |
| Ensemble | 4 |
| Logging | 2 |
| **Total** | **~66 scripts** |
