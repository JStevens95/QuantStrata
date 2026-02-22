"""
Data builder for the Deep Hedging model.

Orchestrates: simulate price paths -> construct feature tensors -> split ->
build tf.data.Datasets via the generic build_tf_dataset() backbone.

Feature tensor at each timestep t:
    [spot_t / S0,  log_moneyness,  time_to_expiry,  bs_delta_t,  implied_vol]

The builder returns a DataBuildResult containing train/val/test datasets that
yield ({price_paths, payoffs}, targets=zeros) tuples -- compatible with the
standard Trainer and Evaluator.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.rade_ml.data.result import DataBuildResult
from src.rade_ml.data.dataset import build_tf_dataset
from src.rade_ml.data.deep_hedging.config import DeepHedgingDataConfig
from src.rade_ml.data.deep_hedging.simulators import GBMSimulator, HestonSimulator

logger = logging.getLogger(__name__)


def build_deep_hedging_data(
    config: Optional[DeepHedgingDataConfig] = None,
) -> DataBuildResult:
    """
    Build train/val/test datasets for the Deep Hedging model.

    :param config: data pipeline configuration (uses defaults if None)
    :return: DataBuildResult with train_ds, val_ds, test_ds
    """
    if config is None:
        config = DeepHedgingDataConfig()

    mkt = config.market
    opt = config.option
    sim = config.simulation

    logger.info(
        f"Simulating {sim.num_paths} paths | model={mkt.model} | "
        f"T={opt.maturity_years}y | steps={sim.num_steps}"
    )

    # 1. simulate price paths
    if mkt.model.lower() == "gbm":
        simulator = GBMSimulator(
            spot_0=mkt.spot_0,
            risk_free_rate=mkt.risk_free_rate,
            dividend_yield=mkt.dividend_yield,
            volatility=mkt.volatility,
        )
    elif mkt.model.lower() == "heston":
        simulator = HestonSimulator(
            spot_0=mkt.spot_0,
            v0=mkt.v0,
            risk_free_rate=mkt.risk_free_rate,
            dividend_yield=mkt.dividend_yield,
            kappa=mkt.kappa,
            theta=mkt.theta,
            xi=mkt.xi,
            rho=mkt.rho,
        )
    else:
        raise ValueError(f"Unknown market model: {mkt.model}")

    result = simulator.simulate(
        maturity=opt.maturity_years,
        num_steps=sim.num_steps,
        num_paths=sim.num_paths,
        seed=sim.seed,
        strike=opt.strike,
        option_type=opt.option_type,
    )

    # 2. construct feature tensor: [num_paths, num_steps+1, num_features]
    features = _build_feature_tensor(result, config)
    payoffs = result.payoffs

    # 3. train / val / test split
    n = sim.num_paths
    n_val = int(n * config.validation_split)
    n_test = int(n * config.test_split)
    n_train = n - n_val - n_test

    idx = np.random.default_rng(sim.seed).permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # 4. build tf.data.Datasets
    train_ds = _build_ds(features, payoffs, train_idx, config)
    val_ds = _build_ds(features, payoffs, val_idx, config)
    test_ds = _build_ds(features, payoffs, test_idx, config)

    metadata = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "num_steps": sim.num_steps,
        "num_features": features.shape[-1],
        "market_model": mkt.model,
        "strike": opt.strike,
        "maturity": opt.maturity_years,
    }

    # store simulation result and BS deltas in metadata for evaluation
    if result.bs_deltas is not None:
        metadata["bs_deltas_test"] = result.bs_deltas[test_idx]
    metadata["spot_paths_test"] = result.spot_paths[test_idx]
    metadata["payoffs_test"] = payoffs[test_idx]

    logger.info(
        f"Data built: train={n_train}, val={n_val}, test={n_test} | "
        f"features={features.shape[-1]}d"
    )

    return DataBuildResult(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        metadata=metadata,
    )


def _build_feature_tensor(result, config: DeepHedgingDataConfig) -> np.ndarray:
    """
    Construct the per-timestep feature tensor from simulation output.

    Features per timestep:
        0: normalised spot (S_t / S_0)
        1: log-moneyness  ln(S_t / K)
        2: time-to-expiry  (T - t) / T
        3: BS delta (if available, else 0)
        4: implied vol (constant for GBM, instantaneous for Heston)
    """
    spot = result.spot_paths
    S0 = spot[:, 0:1]
    K = config.option.strike
    T = config.option.maturity_years
    times = result.times

    norm_spot = spot / S0
    log_moneyness = np.log(spot / K + 1e-8)
    time_to_expiry = np.broadcast_to(
        (T - times)[np.newaxis, :],
        spot.shape,
    ) / max(T, 1e-8)

    features_list = [
        norm_spot[:, :, np.newaxis],
        log_moneyness[:, :, np.newaxis],
        time_to_expiry[:, :, np.newaxis],
    ]

    if result.bs_deltas is not None:
        features_list.append(result.bs_deltas[:, :, np.newaxis])

    if result.vol_paths is not None:
        features_list.append(np.sqrt(np.maximum(result.vol_paths, 0.0))[:, :, np.newaxis])
    else:
        vol_const = np.full_like(spot, config.market.volatility)
        features_list.append(vol_const[:, :, np.newaxis])

    features = np.concatenate(features_list, axis=-1).astype(np.float32)
    return features


def _build_ds(features, payoffs, indices, config):
    """Build a tf.data.Dataset for the given split indices."""
    feat_split = features[indices]
    pay_split = payoffs[indices]
    targets = np.zeros(len(indices), dtype=np.float32)

    variable_inputs = {
        "price_paths": feat_split,
        "payoffs": pay_split,
    }
    return build_tf_dataset(variable_inputs, targets, config)
