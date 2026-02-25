#!/usr/bin/env python3
"""
End-to-End Training Example: Hybrid GNN-RNN Model
==================================================

This script walks through every component of the Hybrid GNN-RNN training
pipeline using synthetic data.  It is designed to:

  1. Validate that every stage of the framework is wired correctly.
  2. Serve as living documentation of what each component expects and produces.
  3. Be runnable out-of-the-box:  ``python examples/rade_ml/hybrid_gnn_rnn/01_train_hybrid_gnn_rnn.py``

Architecture overview
---------------------
The Hybrid GNN-RNN model predicts **target trade PnL** from a portfolio of
elementary trades.  It combines:

  - **GNN**  – learns structural relationships between trades via a k-NN graph
    built from encoded trade attributes (moneyness, delta, vega, product type …).
  - **RNN**  – captures temporal dynamics from historical PnL sequences.
  - **Fusion + Attention + Projection** – merges GNN and RNN embeddings,
    attends to target trades, and projects to PnL predictions.

Pipeline stages
---------------
  0. **Synthetic data generation** – creates realistic trade PnL and attributes.
  1. **Data build** – PnL standardisation, dimensionality reduction, attribute
     encoding, graph construction, tf.data.Dataset creation.
  2. **Model instantiation** – HybridGnnRnn Keras model (uncompiled).
  3. **Training** – compile (optimizer, loss, metrics) and fit via Trainer.
     Uses tf.distribute.Strategy when strategy="auto" (GPU if available).
     Set mixed_precision=True and xla_compile=True for faster GPU training.
  4. **Training results** – loss history, best epoch, model summary.
  5. **Evaluation** – Evaluator with RMSE, MAE, R², MAPE and residual statistics.
  6. **Model registration** – persist model + metadata to ModelRegistry.
  7. **Inference** – load from registry via InferenceRunner, predict on test batch.
  8. **Training plots** – 4-panel figure displayed on screen (not saved).
"""
from __future__ import annotations

import sys
import logging
import tempfile
from pathlib import Path

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Reproducibility & TF environment setup
# ---------------------------------------------------------------------------
# Must happen BEFORE any TensorFlow graph operations or dataset creation.
#
# 1. TF_DETERMINISTIC_OPS / enable_op_determinism  – forces all TF ops to
#    use deterministic algorithms (disables non-deterministic GPU reductions).
# 2. enable_debug_mode – TF 2.20 on macOS ARM64: PrefetchDataset background
#    threads can deadlock; debug mode forces synchronous dataset ops.
# 3. set_random_seed – seeds Python random, numpy, and TF global RNG in one
#    call so every source of randomness starts from the same state.
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
tf.config.experimental.enable_op_determinism()
tf.data.experimental.enable_debug_mode()
tf.keras.utils.set_random_seed(42)

import numpy as np
import pandas as pd

from src.rade_ml.data.io import CacheLoader

# Configure logging so every pipeline component is visible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("example.hybrid_gnn_rnn")


# ======================================================================
# 0.  Synthetic Data Generation
# ======================================================================
#
# In production the "job" dict points to real data files.  Here we
# fabricate a small but structurally complete dataset so every pipeline
# stage exercises its logic.
#
# Required data artefacts:
#
#   elementary_pnl  – pd.DataFrame [n_scenarios x n_elementary]
#       Daily PnL for each elementary (hedging) trade.
#       Columns: trade_id strings in format "UNDERLYING|PRODUCT_TYPE|id".
#       Index: integer scenario index (0 … n_scenarios - 1).
#
#   target_pnl      – pd.DataFrame [n_scenarios x n_target]
#       Daily PnL for each target trade (portfolio positions to predict).
#       Same column format and scenario index as elementary_pnl.
#
#   elementary_attribs / target_attribs – Dict[str, List[Any]]
#       Per-trade attribute dictionary with keys:
#         trade_id              – list of str  (must match PnL column order)
#         moneyness             – list of float
#         yrs_to_maturity       – list of float
#         delta                 – list of float
#         vega                  – list of float
#         product_type          – list of str   (e.g. "vanilla_option")
#         product_subtype       – list of str   (e.g. "european")
#         trade_type            – list of str   "option" for elementary, "option" for target
#         underlying_risk_factors – list of list[str]  (e.g. [["FX"]])
#
# The trade_id format "UNDERLYING|PRODUCT_TYPE|id" is critical because
# dimension_reduction groups trades by underlying×product_type before
# running basis selection (SVD + pivoted QR).
# ======================================================================

def make_synthetic_data(
    workdir: Path,
    n_scenarios: int = 200,
    n_elementary: int = 20,
    n_target: int = 4,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic PnL and attribute files mimicking a real trade universe.

    Returns a ``job`` dict with ``cluster_info`` paths — the same structure
    that a production job scheduler would produce.
    """
    rng = np.random.RandomState(seed)

    # -- Trade IDs --
    # Format: UNDERLYING|PRODUCT_TYPE|numeric_id
    # We create two underlyings and two product types so dimension_reduction
    # has multiple groups to process independently.
    underlyings = ["EURUSD", "GBPUSD"]
    product_types = ["vanilla_option", "forward"]

    elem_ids = []
    idx = 1
    for und in underlyings:
        for prod in product_types:
            n_per_group = n_elementary // (len(underlyings) * len(product_types))
            for i in range(n_per_group):
                elem_ids.append(f"{und}|{prod}|{idx}")
                idx += 1
    # pad if needed
    while len(elem_ids) < n_elementary:
        elem_ids.append(f"EURUSD|vanilla_option|{idx}")
        idx += 1
    elem_ids = elem_ids[:n_elementary]

    tgt_ids = [f"EURUSD|vanilla_option|tgt_{i+1}" for i in range(n_target)]

    logger.info(f"Trade universe: {n_elementary} elementary + {n_target} target = {n_elementary + n_target} total")
    logger.info(f"Elementary IDs (sample): {elem_ids[:4]} ...")
    logger.info(f"Target IDs: {tgt_ids}")

    # -- PnL DataFrames --
    # Small Gaussian noise imitates daily PnL changes.
    # Target PnL is constructed as a noisy linear combination of elementary PnL
    # so there is a learnable signal for the model.
    elem_pnl_arr = rng.randn(n_scenarios, n_elementary).astype(np.float32) * 0.01
    mix_weights = rng.randn(n_elementary, n_target).astype(np.float32) * 0.3
    tgt_pnl_arr = (elem_pnl_arr @ mix_weights + rng.randn(n_scenarios, n_target).astype(np.float32) * 0.002)

    elem_pnl = pd.DataFrame(elem_pnl_arr, columns=elem_ids)
    tgt_pnl = pd.DataFrame(tgt_pnl_arr, columns=tgt_ids)

    logger.info(f"PnL shapes: elementary {elem_pnl.shape}, target {tgt_pnl.shape}")

    # -- Attribute Dictionaries --
    def _make_attrs(trade_ids: list, trade_type: str = "option") -> dict:
        n = len(trade_ids)
        return {
            "trade_id": trade_ids,
            "moneyness": rng.uniform(0.8, 1.2, n).tolist(),
            "yrs_to_maturity": rng.uniform(0.1, 2.0, n).tolist(),
            "delta": rng.uniform(-1.0, 1.0, n).tolist(),
            "vega": rng.uniform(0.0, 0.5, n).tolist(),
            "product_type": [tid.split("|")[1] for tid in trade_ids],
            "product_subtype": ["european"] * n,
            "trade_type": [trade_type] * n,
            "underlying_risk_factors": [["FX"]] * n,
        }

    elem_attrs = _make_attrs(elem_ids, trade_type="elementary")
    tgt_attrs = _make_attrs(tgt_ids, trade_type="target")

    logger.info(f"Attribute keys: {list(elem_attrs.keys())}")

    # -- Write to disk (pickle) --
    workdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "elementary_pnl_path": str(workdir / "elem_pnl.pkl"),
        "target_pnl_path": str(workdir / "tgt_pnl.pkl"),
        "elementary_attribs_path": str(workdir / "elem_attrs.pkl"),
        "target_attribs_path": str(workdir / "tgt_attrs.pkl"),
    }
    for key, path in paths.items():
        data = {
            "elementary_pnl_path": elem_pnl,
            "target_pnl_path": tgt_pnl,
            "elementary_attribs_path": elem_attrs,
            "target_attribs_path": tgt_attrs,
        }[key]
        CacheLoader.save_data(data, path)

    logger.info(f"Data written to {workdir}")

    # The job dict mimics what a production scheduler provides.
    # "cluster_info" keys must match load_data() in build.py.
    return {"cluster_info": paths}


# ======================================================================
# 1.  Configuration
# ======================================================================
#
# Three configuration objects govern the pipeline:
#
#   PipelineConfig       – top-level: references training_config, data_config,
#                          model_config, infrastructure paths, and metadata.
#
#   HybridGnnRnnDataConfig – data pipeline settings:
#     - DataPipelineConfig base: batch_size, shuffle, validation_split, etc.
#     - FolderEnvironmentConfig: root output folder.
#     - DimensionalityConfig: basis selection (var_threshold, method).
#     - GraphBuilderConfig: k-NN graph params (k, alpha weights, quotas).
#     - AttributeEncoderConfig: which trade attributes to encode and how.
#
#   TrainingConfig       – training loop settings:
#     - epochs, loss, metrics.
#     - OptimizerConfig: name + hyperparameters → builds a Keras optimizer.
#     - EarlyStoppingConfig: patience, monitor, restore_best_weights.
#     - GPU options: strategy, mixed_precision, xla_compile (see build_configs).
#     - Optional: LrScheduleConfig, CheckpointConfig, ReduceLrConfig.
# ======================================================================

def build_configs(workdir: Path, job: dict) -> "PipelineConfig":
    """Build the full pipeline configuration with explanations."""
    from src.rade_ml.pipelines.config import PipelineConfig
    from src.rade_ml.data.hybrid_gnn_rnn.config import (
        HybridGnnRnnDataConfig,
        FolderEnvironmentConfig,
        DimensionalityConfig,
        BasisSelectionConfig,
        GraphBuilderConfig,
        AttributeEncoderConfig,
    )
    from src.rade_ml.core.config import TrainingConfig, OptimizerConfig, EarlyStoppingConfig, ReduceLrConfig

    artifacts_dir = workdir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -- Data configuration --
    data_config = HybridGnnRnnDataConfig(
        # FolderEnvironmentConfig: root folder for intermediate file outputs.
        folders=FolderEnvironmentConfig(root_folder=str(workdir)),

        # Data splitting: 75% train, 15% validation, 10% test.
        validation_split=0.10,
        test_split=0.05,

        # Sequence length: number of consecutive PnL days per sample.
        # seq_length=1 means each sample is a single-day snapshot.
        seq_length=1,

        # tf.data.Dataset settings.
        batch_size=16,
        shuffle=True,
        cache=False,
        drop_remainder=False,

        # PnL standardisation: "standard" uses StandardScaler (zero-mean, unit-var).
        transform_type="standard",

        # Dimensionality reduction: basis selection finds a representative
        # subset of elementary trades using SVD + pivoted QR decomposition.
        # var_threshold: fraction of variance to retain (lower = fewer trades).
        # max_components: hard cap on selected trades per underlying×product group.
        dimensionality=DimensionalityConfig(
            reduction_mode="basis_selection",
            basis_selection=BasisSelectionConfig(
                var_threshold=0.95,
                method="pca",
                max_components=10,
            ),
        ),

        # Graph builder: k-NN adjacency with Gaussian RBF kernel.
        # k: neighbours per node (clamped to available trades if too large).
        # alpha_*: feature importance weights for distance calculation.
        graph_builder=GraphBuilderConfig(
            k=3,
            distance_metric="euclidean",
            include_quota=False,
            alpha_moneyness=1.0,
            alpha_maturity=1.0,
            alpha_delta=1.0,
            alpha_vega=1.0,
            alpha_prod_type=1.0,
            alpha_prod_subtype=0.5,
            alpha_underlying=1.0,
            alpha_underlying_rf=0.5,
        ),

        # Attribute encoder: defines which raw trade fields to encode.
        attribute_encoder=AttributeEncoderConfig(
            numeric_keys=["moneyness", "yrs_to_maturity", "delta", "vega"],
            categorical_keys=["product_type", "product_subtype", "trade_type"],
            multi_label_keys=["underlying_risk_factors"],
            num_decay_terms=3,
        ),

        # Diagnostic flags (disabled for speed).
        plot_trade_graph=False,
        plot_pnl_distribution=False,
        save_intermediate_files=False,

        seed=42,
    )

    # -- Training configuration --
    training_config = TrainingConfig(
        epochs=500,
        loss="mse",
        metrics=["mse", "mae"],

        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.999,
        ),

        early_stopping=EarlyStoppingConfig(
            patience=30,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        ),

        lr_reduction=ReduceLrConfig(
            monitor="val_loss",
            mode="min",
            initial_lr=1e-3,
            patience=10,
            factor=0.8,
            min_lr=1e-6,
        ),

        # GPU / performance options (no accuracy impact):
        #   strategy: "auto" uses GPU if available, else CPU. Also "mirrored" (multi-GPU),
        #     "one_device_gpu", "one_device_cpu". None = default device placement.
        strategy="auto",
        #   mixed_precision: fp16 compute where supported (~1.5-2x speed, ~50% less memory).
        mixed_precision=True,
        #   xla_compile: JIT-compile the model for op fusion (~10-30% speedup).
        xla_compile=True,

        verbose=True,
    )

    # -- Pipeline configuration --
    config = PipelineConfig(
        training_config=training_config.to_dict(),
        data_config=data_config,
        model_config=None,     # uses default_model_config()
        registry_dir=None,     # no model registry for this example
        tracking_dir=None,     # no experiment tracker for this example
        artifacts_dir=str(artifacts_dir),
        metadata={
            "job": job,
            "run_name": "hybrid_gnn_rnn_example",
            "generate_training_report": False,
        },
    )

    logger.info("Configuration built")
    logger.info(f"  Data config: batch_size={data_config.batch_size}, "
                f"val_split={data_config.validation_split}, "
                f"test_split={data_config.test_split}, "
                f"seq_length={data_config.seq_length}")
    logger.info(f"  Training config: epochs={training_config.epochs}, "
                f"loss={training_config.loss}, "
                f"optimizer={training_config.optimizer.name} "
                f"(lr={training_config.optimizer.learning_rate}), "
                f"strategy={training_config.strategy}, "
                f"mixed_precision={training_config.mixed_precision}, "
                f"xla_compile={training_config.xla_compile}")

    return config


# ======================================================================
# 2.  Step-by-Step Pipeline Execution
# ======================================================================
#
# This section runs each stage individually so you can inspect
# intermediate outputs.  The same sequence is orchestrated automatically
# by HybridGnnRnnTrainPipeline.run().
# ======================================================================

def run_step_by_step(config: "PipelineConfig") -> None:
    """Execute each pipeline stage individually with detailed logging."""
    import tensorflow as tf
    from src.rade_ml.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline
    from src.rade_ml.training.trainer import Trainer, setup_training_environment

    pipeline = HybridGnnRnnTrainPipeline(config)
    training_config = pipeline._resolve_training_config()
    seed = pipeline._resolve_seed()
    setup_training_environment(training_config, seed)

    # ---- Stage 1: Build Data ----
    #
    # Internally calls build_dataset() which runs:
    #   0. load_data()           – reads PnL DataFrames and attribute dicts from disk.
    #   1. standardise_pnl_history() – fits StandardScaler on training split, transforms all.
    #   2. dimension_reduction() – basis selection per underlying×product_type group.
    #   3. encode_trade_attributes() – numeric scaling + one-hot + multi-label encoding.
    #   4. build_trade_graph()   – k-NN adjacency with RBF kernel weights.
    #   5. _make_ds()            – windowed PnL sequences → tf.data.Dataset.
    #
    # Returns HybridGnnRnnResult containing:
    #   .train_ds / .val_ds / .test_ds  – tf.data.Datasets ready for model.fit()
    #   .elementary_pnl                 – DataFrame of reduced elementary PnL
    #   .target_pnl                     – DataFrame of target PnL
    #   .elementary_attributes          – attribute dict (post-reduction)
    #   .target_attributes              – attribute dict
    #   .builder                        – TradeGraphBuilder instance
    #   .graph_results                  – adjacency SparseTensor, indices, etc.
    #   .encoder                        – fitted TradeAttributeEncoder
    #   .encoder_results                – encoded features (combined_features array)
    #   .metadata                       – scenario splits, trade indices, inverse transforms
    print("\n" + "=" * 70)
    print("STAGE 1: BUILD DATA")
    print("=" * 70)

    data_result = pipeline.build_data(config)

    print(f"\n  Data result type: {type(data_result).__name__}")
    print(f"  Elementary PnL shape: {data_result.elementary_pnl.shape}")
    print(f"  Target PnL shape:     {data_result.target_pnl.shape}")
    print(f"  Elementary trade IDs:  {list(data_result.elementary_pnl.columns[:3])} ...")
    print(f"  Target trade IDs:      {list(data_result.target_pnl.columns)}")

    print(f"\n  Metadata keys: {sorted(data_result.metadata.keys())}")
    print(f"  Train scenarios: {data_result.metadata['train_size']:.0%} "
          f"({len(data_result.metadata['train_starts'])} windows)")
    print(f"  Val scenarios:   {data_result.metadata['val_size']:.0%} "
          f"({len(data_result.metadata['val_starts'])} windows)")
    print(f"  Test scenarios:  {data_result.metadata['test_size']:.0%} "
          f"({len(data_result.metadata['test_starts'])} windows)")
    print(f"  Elementary idx:  {data_result.metadata['elementary_idx']}")
    print(f"  Target idx:      {data_result.metadata['target_idx']}")

    # Inspect one batch to confirm tensor shapes and key names.
    # The adjacency flows through the dataset as three dense component arrays
    # (indices, values, dense_shape) instead of a tf.SparseTensor — this
    # avoids tf.data map-closure tracing issues while keeping O(nnz) memory.
    # The model reconstructs the SparseTensor from components in call().
    print("\n  --- tf.data.Dataset batch inspection ---")
    for inputs_batch, targets_batch in data_result.train_ds.take(1):
        print(f"  Input keys:     {sorted(inputs_batch.keys())}")
        for k, v in sorted(inputs_batch.items()):
            print(f"    {k:25s} -> {v.shape}")
        print(f"  Targets shape:  {targets_batch.shape}")

    # Show sparse efficiency.
    adj_sp = data_result.graph_results["adjacency_matrix"]
    n_trades = adj_sp.dense_shape[0].numpy()
    nnz = adj_sp.values.shape[0]
    print(f"\n  Adjacency: {n_trades}x{n_trades} sparse, nnz={nnz} "
          f"(density={nnz / (n_trades * n_trades):.1%}, "
          f"sparse={nnz * 12 / 1024:.1f}KB vs dense={n_trades * n_trades * 4 / 1024:.1f}KB)")

    # ---- Stage 2: Build Model ----
    #
    # Instantiates HybridGnnRnn with default_model_config().
    # The model is NOT compiled here — compilation is deferred to the Trainer
    # so optimizer/loss/metrics come from a single TrainingConfig source of truth.
    #
    # Model.call() expects a dict with these keys (all provided by the dataset):
    #   "trade_features"         [n_trades, feature_dim]   (static, dense)
    #   "pnl_history"            [batch, seq_len, n_elem]  (variable, dense)
    #   "adjacency_indices"      [nnz, 2]                  (static, dense — sparse component)
    #   "adjacency_values"       [nnz]                     (static, dense — sparse component)
    #   "adjacency_dense_shape"  [2]                        (static, dense — sparse component)
    #   "elementary_indices"     [n_elementary]             (static, dense)
    #   "target_indices"         [n_target]                 (static, dense)
    #
    # The model reconstructs tf.SparseTensor from the three adjacency components.
    #
    # Model architecture (from default_model_config):
    #   GnnBlock       – 2-layer mixed GraphSAGE, 128 units, ReLU
    #   RnnBlock       – 2-layer LSTM, 128 units, ReLU
    #   FusionLayer    – gated fusion, 64 units, sigmoid gate
    #   AttentionLayer – standard attention, 32 units, tanh
    #   ProjectionLayer – target PnL output, 32 units, GELU
    #
    # Output: [batch, n_target] — predicted PnL for each target trade.
    print("\n" + "=" * 70)
    print("STAGE 2: BUILD MODEL")
    print("=" * 70)

    model = pipeline.build_model(config, data_result)

    print(f"\n  Model class: {type(model).__name__}")
    print(f"  Model name:  {model.name}")
    print(f"  Compiled:    {getattr(model, '_is_compiled', False)}")
    print(f"  Sub-blocks:  GnnBlock, RnnBlock, FusionLayer, AttentionLayer, ProjectionLayer")

    # ---- Stage 3: Trainer (Compile + Fit) ----
    #
    # setup_training_environment was already called above (before build_model).
    # The Trainer handles:
    #   a) Compiling the model using TrainingConfig:
    #      - OptimizerConfig.build() → tf.keras.optimizers.Adam
    #      - loss: "mse" → tf.keras.losses.MeanSquaredError
    #      - metrics: ["mse", "mae"]
    #   b) Building callbacks from config:
    #      - EarlyStoppingConfig → tf.keras.callbacks.EarlyStopping
    #   c) Running model.fit() with train_data and val_data.
    #   d) Returning TrainingResult with history, best epoch, timing.
    print("\n" + "=" * 70)
    print("STAGE 3: COMPILE + TRAIN")
    print("=" * 70)

    trainer = Trainer(model=model, config=training_config, seed=seed)

    print(f"\n  Training config:")
    print(f"    Epochs:          {training_config.epochs}")
    print(f"    Loss:            {training_config.loss}")
    print(f"    Optimizer:       {training_config.optimizer.name} (lr={training_config.optimizer.learning_rate})")
    print(f"    Early stopping:  patience={training_config.early_stopping.patience}, "
          f"monitor={training_config.early_stopping.monitor}")
    print(f"    Metrics:         {training_config.metrics}")

    print("\n  Compiling model...")
    trainer.compile()
    print(f"  Model compiled: optimizer={model.optimizer.__class__.__name__}, "
          f"loss={model.loss.__class__.__name__}")

    print("\n  Starting training...")
    result = trainer.fit(
        train_data=data_result.train_ds,
        val_data=data_result.val_ds,
    )

    # ---- Stage 4: Results ----
    print("\n" + "=" * 70)
    print("STAGE 4: TRAINING RESULTS")
    print("=" * 70)

    print(f"\n  Training time:   {result.training_time_seconds:.1f}s")
    print(f"  Final epoch:     {result.final_epoch}")
    print(f"  Best epoch:      {result.best_epoch}")
    print(f"  Best train loss: {result.best_train_loss:.6f}")
    print(f"  Best val loss:   {result.best_val_loss:.6f}")
    print(f"  Stopped early:   {result.stopped_early}")

    print(f"\n  Loss history:")
    for epoch_i, (train_l, val_l) in enumerate(
        zip(result.history["loss"], result.history.get("val_loss", [])),
        start=1,
    ):
        marker = " <-- best" if epoch_i == result.best_epoch else ""
        print(f"    Epoch {epoch_i:3d}:  train={train_l:.6f}  val={val_l:.6f}{marker}")

    print(f"\n  Model summary:")
    summary = result.model_summary or {}
    print(f"    Trainable params:     {summary.get('trainable_params', 'N/A'):,}")
    print(f"    Non-trainable params: {summary.get('non_trainable_params', 'N/A'):,}")
    print(f"    Layers:               {summary.get('layers', 'N/A')}")

    # ---- Stage 5: Evaluation ----
    #
    # The Evaluator goes beyond trainer.evaluate():
    #   - Collects raw predictions and ground-truth targets.
    #   - Computes residuals (pred − target) for every sample.
    #   - Adds aggregate residual statistics (MAE, max, P95, P99).
    #   - Accepts additional user-supplied metric functions.
    #
    # Returns an EvaluationResult with .metrics, .predictions, .targets,
    # .residuals, and .dataset_info — enough to drive any downstream
    # reporting, alerting, or drift-detection pipeline.
    print("\n" + "=" * 70)
    print("STAGE 5: EVALUATION")
    print("=" * 70)

    from src.rade_ml.evaluation.evaluator import Evaluator
    from src.rade_ml.evaluation.metrics import rmse, mae, r_squared, mape

    eval_ds = data_result.test_ds if data_result.test_ds is not None else data_result.val_ds
    evaluator = Evaluator(model)
    eval_result = evaluator.run(
        eval_ds,
        additional_metrics={"rmse": rmse, "mae": mae, "r_squared": r_squared, "mape": mape},
    )

    print(f"\n  {eval_result.summary()}")
    print(f"\n  Prediction shape:  {eval_result.predictions.shape}")
    print(f"  Target shape:      {eval_result.targets.shape}")
    print(f"  Residual stats:")
    for k, v in sorted(eval_result.metrics.items()):
        if k.startswith("residual_") or k in ("rmse", "mae", "r_squared", "mape"):
            print(f"    {k:20s}: {v:.6f}")

    # ---- Stage 6: Model Registration ----
    #
    # The ModelRegistry persists trained models alongside their metadata
    # (training metrics, config, tags) so that downstream consumers —
    # evaluation scripts, inference pipelines, dashboards — can load any
    # version by tag or version string.
    #
    # register() saves:
    #   <registry_dir>/<version>/model.keras   — full Keras model
    #   <registry_dir>/<version>/meta.json     — RegistryEntry metadata
    #   <registry_dir>/index.json              — tag → version mapping
    print("\n" + "=" * 70)
    print("STAGE 6: MODEL REGISTRATION")
    print("=" * 70)

    from src.rade_ml.registry.store import ModelRegistry

    registry_dir = Path(config.artifacts_dir) / "registry"
    registry = ModelRegistry(str(registry_dir))

    entry = registry.register(
        model=model,
        training_result=result,
        tags=["hybrid_gnn_rnn", "latest"],
        description="Hybrid GNN-RNN example model (synthetic data)",
    )

    print(f"\n  Registry dir:  {registry_dir}")
    print(f"  Version:       {entry.version}")
    print(f"  Tags:          {entry.tags}")
    print(f"  Model dir:     {entry.model_dir}")
    print(f"  Best epoch:    {entry.best_epoch}")
    print(f"  Metrics:       {entry.metrics}")

    # ---- Stage 7: Inference ----
    #
    # The InferenceRunner is a model-agnostic forward-pass wrapper.
    # It loads a model from the registry (or a direct path), runs
    # prediction, and wraps outputs in a provenance-enriched
    # InferenceResult (including input hash, timing, version info).
    #
    # In production the full HybridGnnRnnInferencePipeline would:
    #   1. Load the graph builder and encoder from saved artefacts.
    #   2. Optionally extend the graph for new target trades.
    #   3. Prepare model-ready input tensors.
    #   4. Feed them to the InferenceRunner.
    #
    # Here we demonstrate the runner directly with one batch from
    # the test dataset — the same tensors the model was evaluated on.
    print("\n" + "=" * 70)
    print("STAGE 7: INFERENCE (via InferenceRunner)")
    print("=" * 70)

    from src.rade_ml.inference.runner import InferenceRunner
    import src.rade_ml.models.hybrid_gnn_rnn.model  # noqa: F401 – registers custom classes with Keras

    runner = InferenceRunner.from_registry(registry, version_or_tag="latest")
    print(f"\n  Loaded model version: {runner.model_version}")
    print(f"  Model path:           {runner.model_path}")

    infer_ds = eval_ds.take(1).map(lambda x, _y: x)
    infer_result = runner.predict(
        inputs=infer_ds,
        sample_ids=[f"trade_{i}" for i in range(len(data_result.target_pnl.columns))],
        metadata={"source": "example_script", "dataset": "synthetic_test"},
    )

    print(f"\n  InferenceResult:")
    print(f"    n_samples:       {infer_result.n_samples}")
    print(f"    latency:         {infer_result.latency_seconds:.4f}s")
    print(f"    input_hash:      {infer_result.input_hash[:16]}...")
    print(f"    model_version:   {infer_result.model_version}")
    preds = infer_result.predictions
    print(f"    prediction shape: {preds.shape}")
    print(f"    prediction stats: mean={preds.mean():.6f}, "
          f"std={preds.std():.6f}, "
          f"min={preds.min():.6f}, "
          f"max={preds.max():.6f}")

    # ---- Stage 8: Training Plots ----
    #
    # Display training dynamics on screen (not saved to disk).
    # Four panels: loss curves, train-val gap, val/train ratio, other metrics.
    # This is shown last so the interactive matplotlib window does not block
    # the preceding console output.
    print("\n" + "=" * 70)
    print("STAGE 8: TRAINING PLOTS (displaying on screen)")
    print("=" * 70)
    print("\n  Showing 4-panel training dynamics figure...")
    print("  Close the plot window to finish.\n")

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    from src.rade_ml.training.plots import show_training_plots
    show_training_plots(result)

    return result


# ======================================================================
# 3.  Full Pipeline Run (single-call)
# ======================================================================
#
# The same sequence above is available as a single call:
#   pipeline = HybridGnnRnnTrainPipeline(config)
#   result = pipeline.run()
#
# This is shown below for completeness.
# ======================================================================

def run_pipeline(config: "PipelineConfig") -> None:
    """Run the complete pipeline in one call."""
    from src.rade_ml.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline

    print("\n" + "=" * 70)
    print("FULL PIPELINE: HybridGnnRnnTrainPipeline.run()")
    print("=" * 70)

    pipeline = HybridGnnRnnTrainPipeline(config)
    result = pipeline.run()

    print(f"\n  Result: best_epoch={result.best_epoch}, "
          f"best_val_loss={result.best_val_loss:.6f}, "
          f"time={result.training_time_seconds:.1f}s")

    return result


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="rade_ml_hybrid_gnn_rnn_"))
    logger.info(f"Working directory: {workdir}")

    # 0. Generate synthetic data.
    job = make_synthetic_data(workdir, n_scenarios=200, n_elementary=20, n_target=4)

    # 1. Build configuration.
    config = build_configs(workdir, job)

    # 2. Run step-by-step (detailed output).
    result = run_step_by_step(config)

    # Uncomment to also run the single-call pipeline:
    # run_pipeline(config)

    logger.info(f"All artifacts in: {workdir}")


if __name__ == "__main__":
    main()
