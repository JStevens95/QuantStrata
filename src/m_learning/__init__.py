"""
QuantStrata Machine Learning Module.

A comprehensive TensorFlow-native framework for ML-based pricing,
calibration, and portfolio risk prediction.

Architecture:
    src/m_learning/
    ├── core/           # Base classes, configuration, callbacks
    ├── data/           # Dataset utilities, normalization
    ├── models/         # Neural network architectures
    │   ├── pricing/    # Option pricing models (MLP, attention)
    │   ├── calibration/# Model calibration networks
    │   └── gnn_rnn_hybrid/  # Portfolio P&L prediction
    ├── training/       # Trainer class, training loops
    ├── evaluation/     # Metrics, evaluator
    ├── inference/      # Model I/O, prediction
    └── utilities/      # Graph building, encoders

Quick Start:
    from src.m_learning import (
        # Data
        TFDataset, create_pricing_dataset,
        # Models
        MLPPricer, create_mlp_pricer,
        # Training
        Trainer, TrainingConfig,
        # Evaluation
        Evaluator, evaluate_model,
        # Inference
        save_model, load_model, Predictor,
    )
    
    # 1. Create dataset
    dataset = create_pricing_dataset(n_samples=10000, seed=42)
    dataset.normalize_features().normalize_targets()
    train, val, test = dataset.split()
    
    # 2. Build model
    model = create_mlp_pricer(n_features=6, hidden_units=[128, 64, 32])
    
    # 3. Train
    config = TrainingConfig(epochs=100, batch_size=256)
    trainer = Trainer(model, config)
    result = trainer.fit(train, val)
    
    # 4. Evaluate
    eval_result = evaluate_model(model, test, target_scaler=dataset.target_stats)
    print(eval_result.summary())
    
    # 5. Save & Deploy
    save_model(model, "models/my_pricer", 
               feature_stats=dataset.feature_stats,
               target_stats=dataset.target_stats)

Plug-and-play:
    The same pipeline (Trainer, Evaluator, save_model/load_model, Predictor) works
    for any tf.keras.Model. Subclass BaseModel or PricingModel for optional metadata
    and helpers; custom_objects is used when loading custom classes.

See Also:
    - Tutorial: docs/tutorials/m_learning/ml_pipeline_tensorflow.ipynb
    - Architecture: docs/architecture/ecosystem_diagrams.md
"""

# Core components
from src.m_learning.core import (
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
from src.m_learning.data import (
    TFDataset,
    NormalizationStats,
    create_pricing_dataset,
    create_calibration_dataset,
    MLDataset,
    build_pricing_data,
    build_gnn_data,
    PricingDataResult,
    GnnDataResult,
    TradeAttributeEncoder,
    TradeGraphBuilder,
)

# Models
from src.m_learning.models import (
    MLPPricer,
    create_mlp_pricer,
)

# Training
from src.m_learning.training import (
    Trainer,
    TrainingResult,
    fit_model,
)

# Evaluation
from src.m_learning.evaluation import (
    Evaluator,
    EvaluationResult,
    evaluate_model,
    compute_metrics,
    PricingMetrics,
)

# Inference
from src.m_learning.inference import (
    save_model,
    load_model,
    ModelArtifact,
    Predictor,
    BatchPredictor,
)

__version__ = "2.0.0"

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
    "TFDataset",
    "NormalizationStats",
    "create_pricing_dataset",
    "create_calibration_dataset",
    "MLDataset",
    "build_pricing_data",
    "build_gnn_data",
    "PricingDataResult",
    "GnnDataResult",
    "TradeAttributeEncoder",
    "TradeGraphBuilder",
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
]
