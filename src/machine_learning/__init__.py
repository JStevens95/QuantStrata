"""
QuantStrata Machine Learning Module.

A comprehensive, production-grade TensorFlow-native framework for ML-based
pricing, calibration, and portfolio risk prediction.  Designed to support
both simple models (MLP pricers) and complex architectures (GNN-RNN hybrids)
through a unified pipeline.

Architecture:
    src/machine_learning/
    ├── core/           # Base classes, configuration, result types, tracking
    ├── data/           # Data builders, sklearn scalers, tf.data pipelines
    │   ├── pricing/    # Option pricing data builder
    │   ├── calibration/# Calibration data builder
    │   └── gnn_rnn_hybrid/  # GNN/RNN portfolio data builder
    ├── models/         # Neural network architectures
    │   ├── pricing/    # Option pricing models (MLP, attention)
    │   ├── calibration/# Model calibration networks
    │   └── gnn_rnn_hybrid/  # Portfolio P&L prediction
    ├── training/       # Trainer, callbacks, LR schedules
    ├── evaluation/     # Evaluator (sklearn.metrics + domain metrics)
    ├── inference/      # Predictor, BatchPredictor, model I/O (joblib scalers)
    ├── validation/     # Pre-deployment quality gates
    ├── monitoring/     # Production drift detection
    ├── pipelines/      # Framework-agnostic training/eval/inference loops
    ├── registry/       # Model registry and versioning
    └── utilities/      # Graph building, encoders

Quick Start:
    from src.machine_learning import (
        # Data
        build_tf_dataset, create_pricing_dataset, build_pricing_data,
        # Models
        MLPPricer, create_mlp_pricer,
        # Training
        Trainer, TrainingConfig,
        # Evaluation
        Evaluator, evaluate_model,
        # Inference
        save_model, load_model, Predictor,
        # Validation
        ValidationGate, run_validation,
        # Monitoring
        DriftDetector,
    )

    # 1. Build dataset (sklearn scalers + tf.data)
    data = build_pricing_data(n_samples=10_000, seed=42)

    # 2. Build model
    model = create_mlp_pricer(n_features=6, hidden_units=[128, 64, 32])

    # 3. Train
    config = TrainingConfig(epochs=100, batch_size=256)
    trainer = Trainer(model, config)
    result = trainer.fit(data.train_ds, data.val_ds)

    # 4. Evaluate
    eval_result = evaluate_model(model, data.test_ds, target_scaler=data.target_scaler)
    print(eval_result.summary())

    # 5. Validate before deployment
    report = run_validation(eval_result, min_r2=0.95, max_mae=0.02)
    assert report.passed, "Model failed validation gates"

    # 6. Save (scalers persisted via joblib)
    save_model(model, "models/my_pricer",
               feature_scaler=data.feature_scaler,
               target_scaler=data.target_scaler)

Plug-and-play:
    The same pipeline (Trainer, Evaluator, save_model/load_model, Predictor) works
    for any tf.keras.Model — including models with dict inputs (GNN/graph models).
    Subclass BaseModel or PricingModel for optional metadata and helpers.

See Also:
    - Tutorial: docs/tutorials/machine_learning/ml_pipeline_tensorflow.ipynb
    - Architecture: docs/architecture/ecosystem_diagrams.md
"""

# Core components
from src.machine_learning.core import (
    BaseModel,
    PricingModel,
    CalibrationModel,
    PortfolioModel,
    TrainingConfig,
    OptimizerConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    ModelConfig,
)

# Data utilities
from src.machine_learning.data import (
    build_tf_dataset,
    SyntheticData,
    create_pricing_dataset,
    create_calibration_dataset,
    build_pricing_data,
    build_gnn_data,
    PricingDataResult,
    GnnDataResult,
    TradeAttributeEncoder,
    TradeGraphBuilder,
    DatasetManifest,
)

# Models
from src.machine_learning.models import (
    MLPPricer,
    create_mlp_pricer,
)

# Training
from src.machine_learning.training import (
    Trainer,
    TrainingResult,
    fit_model,
)

# Evaluation
from src.machine_learning.evaluation import (
    Evaluator,
    EvaluationResult,
    evaluate_model,
    compute_metrics,
    PricingMetrics,
)

# Inference
from src.machine_learning.inference import (
    save_model,
    load_model,
    ModelArtifact,
    Predictor,
    BatchPredictor,
    create_serving_function,
)

# Validation gates
from src.machine_learning.validation import (
    ValidationGate,
    ValidationReport,
    ValidationCheck,
    run_validation,
)

# Production monitoring
from src.machine_learning.monitoring import (
    DriftDetector,
    DriftReport,
    DriftCheckResult,
)

__version__ = "4.0.0"

__all__ = [
    # Core
    "BaseModel",
    "PricingModel",
    "CalibrationModel",
    "PortfolioModel",
    "TrainingConfig",
    "OptimizerConfig",
    "EarlyStoppingConfig",
    "CheckpointConfig",
    "ModelConfig",
    # Data
    "build_tf_dataset",
    "SyntheticData",
    "create_pricing_dataset",
    "create_calibration_dataset",
    "build_pricing_data",
    "build_gnn_data",
    "PricingDataResult",
    "GnnDataResult",
    "TradeAttributeEncoder",
    "TradeGraphBuilder",
    "DatasetManifest",
    # Models
    "MLPPricer",
    "create_mlp_pricer",
    # Training
    "Trainer",
    "TrainingResult",
    "fit_model",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    "evaluate_model",
    "compute_metrics",
    "PricingMetrics",
    # Inference
    "save_model",
    "load_model",
    "ModelArtifact",
    "Predictor",
    "BatchPredictor",
    "create_serving_function",
    # Validation
    "ValidationGate",
    "ValidationReport",
    "ValidationCheck",
    "run_validation",
    # Monitoring
    "DriftDetector",
    "DriftReport",
    "DriftCheckResult",
]
