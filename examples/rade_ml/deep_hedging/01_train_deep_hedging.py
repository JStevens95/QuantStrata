"""
Deep Hedging -- End-to-end training example.

Demonstrates the full rade_ml workflow:
    1. Configure market simulation and model
    2. Build data (GBM paths for a European call)
    3. Train a Deep Hedging model with CVaR loss
    4. Register the trained model
    5. Track the experiment

Run:
    python examples/rade_ml/deep_hedging/01_train_deep_hedging.py
"""
import logging
import numpy as np

from src.rade_ml.core.config import TrainingConfig, OptimizerConfig, EarlyStoppingConfig
from src.rade_ml.data.deep_hedging.config import (
    DeepHedgingDataConfig,
    MarketDynamicsConfig,
    OptionConfig,
    SimulationConfig,
)
from src.rade_ml.data.deep_hedging.build import build_deep_hedging_data
from src.rade_ml.models.deep_hedging.model import DeepHedgingModel
from src.rade_ml.models.deep_hedging.config import default_model_config
from src.rade_ml.models.deep_hedging.layers.risk_measure import CVaRLoss
from src.rade_ml.training.trainer import Trainer
from src.rade_ml.registry.store import ModelRegistry
from src.rade_ml.tracking.tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------ #
    # 1. Configure
    # ------------------------------------------------------------------ #
    data_config = DeepHedgingDataConfig(
        market=MarketDynamicsConfig(
            model="gbm",
            spot_0=100.0,
            risk_free_rate=0.05,
            volatility=0.2,
        ),
        option=OptionConfig(
            option_type="call",
            strike=100.0,
            maturity_years=0.25,
        ),
        simulation=SimulationConfig(
            num_paths=50_000,
            num_steps=63,
            seed=42,
        ),
        batch_size=256,
        shuffle=True,
    )

    model_config = default_model_config()
    model_config["general"]["transaction_cost_rate"] = 0.001

    training_config = TrainingConfig(
        epochs=30,
        loss="mse",  # overridden by CVaR below
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
        early_stopping=EarlyStoppingConfig(patience=5, monitor="val_loss"),
        seed=42,
    )

    # ------------------------------------------------------------------ #
    # 2. Build data
    # ------------------------------------------------------------------ #
    logger.info("Building deep hedging datasets...")
    data_result = build_deep_hedging_data(data_config)
    logger.info(f"Train: {data_result.metadata['n_train']} | "
                f"Val: {data_result.metadata['n_val']} | "
                f"Test: {data_result.metadata['n_test']}")

    # ------------------------------------------------------------------ #
    # 3. Build & train model
    # ------------------------------------------------------------------ #
    model = DeepHedgingModel(config=model_config)

    cvar_loss = CVaRLoss(alpha=0.95)
    model.compile(optimizer="adam", loss=cvar_loss)

    trainer = Trainer(model=model, config=training_config)
    trainer._is_compiled = True  # already compiled with custom loss

    result = trainer.fit(
        train_data=data_result.train_ds,
        val_data=data_result.val_ds,
    )

    logger.info(f"Training complete | best_epoch={result.best_epoch} | "
                f"best_val_loss={result.best_val_loss:.6f}")

    # ------------------------------------------------------------------ #
    # 4. Register model
    # ------------------------------------------------------------------ #
    registry = ModelRegistry("./artifacts/deep_hedging/registry")
    entry = registry.register(model, result, tags=["deep_hedging", "gbm_call"])
    logger.info(f"Registered model version: {entry.version}")

    # ------------------------------------------------------------------ #
    # 5. Track experiment
    # ------------------------------------------------------------------ #
    tracker = ExperimentTracker("./artifacts/deep_hedging/experiments")
    run = tracker.start_run(name="deep_hedging_gbm_call", tags=["deep_hedging"])
    run.log_config(training_config)
    run.log_result(result)
    run.set_model_version(entry.version)
    tracker.end_run(run)
    logger.info(f"Experiment run saved: {run.run_id}")

    # ------------------------------------------------------------------ #
    # Quick sanity check on test set
    # ------------------------------------------------------------------ #
    pnl_list = []
    for batch_x, _ in data_result.test_ds:
        pnl_batch = model(batch_x, training=False).numpy()
        pnl_list.append(pnl_batch)
    pnl = np.concatenate(pnl_list)

    logger.info(f"Test P&L stats: mean={pnl.mean():.4f}, std={pnl.std():.4f}, "
                f"P5={np.percentile(pnl, 5):.4f}, P95={np.percentile(pnl, 95):.4f}")


if __name__ == "__main__":
    main()
