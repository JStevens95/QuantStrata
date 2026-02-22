"""
Deep Hedging -- Evaluation and comparison to Black-Scholes delta hedging.

Loads a trained model from the registry, evaluates it on test scenarios,
and compares the hedging P&L distribution against a Black-Scholes delta hedge.

Run:
    python examples/rade_ml/deep_hedging/02_evaluate_deep_hedging.py
"""
import logging
import numpy as np

from src.rade_ml.data.deep_hedging.config import (
    DeepHedgingDataConfig,
    MarketDynamicsConfig,
    OptionConfig,
    SimulationConfig,
)
from src.rade_ml.data.deep_hedging.build import build_deep_hedging_data
from src.rade_ml.evaluation.evaluator import Evaluator
from src.rade_ml.evaluation.metrics import rmse, mae
from src.rade_ml.registry.store import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_bs_delta_hedge_pnl(
    spot_paths: np.ndarray,
    bs_deltas: np.ndarray,
    payoffs: np.ndarray,
    tc_rate: float = 0.001,
) -> np.ndarray:
    """
    Compute the terminal P&L of a Black-Scholes delta hedging strategy.

    Uses the same transaction cost model as the deep hedging model for
    a fair comparison.
    """
    num_paths, num_steps_plus_1 = spot_paths.shape
    cumulative_gain = np.zeros(num_paths)
    cumulative_cost = np.zeros(num_paths)
    prev_delta = np.zeros(num_paths)

    for t in range(num_steps_plus_1 - 1):
        delta_t = bs_deltas[:, t]
        spot_t = spot_paths[:, t]
        spot_next = spot_paths[:, t + 1]

        cumulative_gain += delta_t * (spot_next - spot_t)
        cumulative_cost += tc_rate * np.abs(delta_t - prev_delta) * spot_t
        prev_delta = delta_t

    return cumulative_gain - cumulative_cost - payoffs


def main():
    # ------------------------------------------------------------------ #
    # 1. Configure (same as training for consistent comparison)
    # ------------------------------------------------------------------ #
    data_config = DeepHedgingDataConfig(
        market=MarketDynamicsConfig(model="gbm", spot_0=100.0, volatility=0.2),
        option=OptionConfig(option_type="call", strike=100.0, maturity_years=0.25),
        simulation=SimulationConfig(num_paths=50_000, num_steps=63, seed=42),
        batch_size=256,
        shuffle=False,
    )

    # ------------------------------------------------------------------ #
    # 2. Build data
    # ------------------------------------------------------------------ #
    data_result = build_deep_hedging_data(data_config)

    # ------------------------------------------------------------------ #
    # 3. Load model from registry
    # ------------------------------------------------------------------ #
    registry = ModelRegistry("./artifacts/deep_hedging/registry")
    model, entry = registry.load("deep_hedging")
    logger.info(f"Loaded model version: {entry.version}")

    # ------------------------------------------------------------------ #
    # 4. Evaluate with Evaluator
    # ------------------------------------------------------------------ #
    evaluator = Evaluator(model)
    eval_result = evaluator.run(
        data_result.test_ds,
        additional_metrics={"rmse": rmse, "mae": mae},
    )
    logger.info(f"Evaluation summary:\n{eval_result.summary()}")

    # ------------------------------------------------------------------ #
    # 5. Compare to BS delta hedge
    # ------------------------------------------------------------------ #
    deep_pnl = eval_result.predictions.flatten()

    spot_test = data_result.metadata["spot_paths_test"]
    bs_deltas_test = data_result.metadata["bs_deltas_test"]
    payoffs_test = data_result.metadata["payoffs_test"]

    bs_pnl = compute_bs_delta_hedge_pnl(
        spot_test, bs_deltas_test, payoffs_test, tc_rate=0.001,
    )

    def cvar_95(x):
        threshold = np.percentile(x, 5)
        return np.mean(x[x <= threshold])

    print("\n" + "=" * 60)
    print("HEDGING STRATEGY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Deep Hedge':>15} {'BS Delta':>15}")
    print("-" * 60)
    print(f"{'Mean P&L':<25} {deep_pnl.mean():>15.4f} {bs_pnl.mean():>15.4f}")
    print(f"{'Std P&L':<25} {deep_pnl.std():>15.4f} {bs_pnl.std():>15.4f}")
    print(f"{'VaR 95%':<25} {np.percentile(-deep_pnl, 95):>15.4f} {np.percentile(-bs_pnl, 95):>15.4f}")
    print(f"{'CVaR 95%':<25} {-cvar_95(deep_pnl):>15.4f} {-cvar_95(bs_pnl):>15.4f}")
    print(f"{'Min P&L':<25} {deep_pnl.min():>15.4f} {bs_pnl.min():>15.4f}")
    print(f"{'Max P&L':<25} {deep_pnl.max():>15.4f} {bs_pnl.max():>15.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
