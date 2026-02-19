"""
Dataset utilities for QuantStrata ML framework.

This module provides:
    - ``build_tf_dataset``: thin helper to wrap arrays/dicts into a batched,
      shuffled, prefetched ``tf.data.Dataset`` pipeline.
    - ``create_pricing_dataset``: synthetic option pricing data generator.
    - ``create_calibration_dataset``: synthetic calibration data generator.

Normalisation is **not** handled here — use ``sklearn.preprocessing.StandardScaler``
(or ``MinMaxScaler``) in your data builder before calling ``build_tf_dataset``.

Splitting is **not** handled here — use ``sklearn.model_selection.train_test_split``
in your data builder.

Usage:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from src.machine_learning.data.dataset import build_tf_dataset, create_pricing_dataset

    # Generate synthetic data
    data = create_pricing_dataset(n_samples=10000, seed=42)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        data.features, data.targets, test_size=0.2, random_state=42,
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build tf.data pipelines
    train_ds = build_tf_dataset(X_train, y_train, batch_size=256, shuffle=True)
    test_ds  = build_tf_dataset(X_test, y_test, batch_size=256, shuffle=False)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore


# ---------------------------------------------------------------------------
# Core helper: build tf.data.Dataset from arrays or dicts
# ---------------------------------------------------------------------------


def _build_static_tensors(
    static_inputs: Dict[str, np.ndarray],
    ensure_float32: bool,
) -> Dict[str, "tf.Tensor"]:
    """
    Convert static NumPy arrays to TF constants for the map closure.

    Static inputs (e.g. trade_features, adjacency_matrix) have no sample dimension
    and are shared across all batches. Converting once to tf.constant avoids
    repeated conversion inside the map and keeps them in the graph.
    """
    result = {}
    for key, val in static_inputs.items():
        arr = np.asarray(val)
        # Cast to model-compatible dtypes (float32 for floats, int32 for indices)
        if ensure_float32 and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int32)
        result[key] = tf.constant(arr)
    return result


def build_tf_dataset(
    variable_inputs: Union[np.ndarray, Dict[str, np.ndarray]],
    targets: np.ndarray,
    static_inputs: Optional[Dict[str, np.ndarray]] = None,
    variable_input_key: str = "features",
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer: Optional[int] = None,
    cache: bool = True,
    drop_remainder: bool = False,
    ensure_float32: bool = True,
) -> "tf.data.Dataset":
    """
    Build a batched, prefetched ``tf.data.Dataset`` for Keras training.

    Supports two patterns:
        1. **Simple** (pricing, calibration): only ``variable_inputs`` and
           ``targets``. Each sample is a row; no static data.
        2. **Static + variable** (GNN-RNN, graph models): ``variable_inputs``
           are per-sample (batched); ``static_inputs`` are shared across all
           samples and injected into every batch via a ``map``.

    No explicit tensor conversion is needed — ``from_tensor_slices`` accepts
    NumPy arrays and converts lazily. Static inputs are converted to tensors
    once for the map closure.

    Parameters
    ----------
    variable_inputs : np.ndarray or Dict[str, np.ndarray]
        Per-sample data with first dimension ``n_samples``.  Passed to
        ``from_tensor_slices``.  When combined with ``static_inputs``, use
        a dict for clarity (e.g. ``{"elem_pnl_history": arr}``).
    targets : np.ndarray
        Target array, shape ``(n_samples,)`` or ``(n_samples, n_outputs)``.
    static_inputs : dict, optional
        Arrays with no sample dimension (e.g. trade_features, adjacency_matrix).
        Injected into every batch via ``map``.  When None, only variable data.
    variable_input_key : str
        Key for ``variable_inputs`` when it is an ndarray and
        ``static_inputs`` is not None.  Ignored when ``variable_inputs``
        is a dict.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Shuffle before batching.
    shuffle_buffer : int, optional
        Shuffle buffer size.  Defaults to ``min(n_samples, 50_000)``.
    cache : bool
        Cache dataset in memory after first pass.
    drop_remainder : bool
        Drop final incomplete batch (useful for fixed-size training).
    ensure_float32 : bool
        Cast float inputs and targets to float32 for model compatibility.

    Returns
    -------
    tf.data.Dataset
        Yields ``(inputs_dict, targets)`` or ``(inputs, targets)`` when no
        static data.  Ready for ``model.fit()`` / ``model.evaluate()``.

    Examples
    --------
    Simple (pricing):
        train_ds = build_tf_dataset(X_train, y_train, batch_size=256)

    GNN with static graph:
        train_ds = build_tf_dataset(
            variable_inputs={"elem_pnl_history": elem_pnl_train},
            targets=target_pnl_train,
            static_inputs={
                "trade_features": trade_features,
                "adjacency_matrix": adj,
                "elementary_indices": elem_idx,
                "target_indices": target_idx,
            },
            batch_size=32,
        )
    """
    if tf is None:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    # -------------------------------------------------------------------------
    # Pipeline overview (order matters):
    #   validate → from_tensor_slices → [cache] → [shuffle] → batch → [map static] → prefetch
    # -------------------------------------------------------------------------

    # --- Step 1: Validate and prepare targets ---
    # Targets must be 1D or 2D with first dim = n_samples. Cast to float32 if requested.
    targets = np.asarray(targets)
    n_samples = len(targets)

    def _ensure_dtype(arr: np.ndarray) -> np.ndarray:
        """Cast float arrays to float32 when requested (models typically expect float32)."""
        arr = np.asarray(arr)
        if ensure_float32 and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        return arr

    targets = _ensure_dtype(targets)

    # --- Step 2: Validate and prepare variable inputs ---
    # Variable inputs change per sample (first dim = n_samples). Two formats:
    #   - ndarray: Simple case (pricing, calibration). Shape (n_samples, n_features).
    #   - dict: Multi-input models (e.g. GNN). Keys map to arrays with same n_samples.
    if isinstance(variable_inputs, dict):
        variable_inputs = {k: _ensure_dtype(v) for k, v in variable_inputs.items()}
        for k, v in variable_inputs.items():
            if len(v) != n_samples:
                raise ValueError(
                    f"variable_inputs['{k}'] first dim {len(v)} != targets {n_samples}"
                )
    else:
        variable_inputs = _ensure_dtype(variable_inputs)
        if len(variable_inputs) != n_samples:
            raise ValueError(
                f"variable_inputs first dim {len(variable_inputs)} != targets {n_samples}"
            )

    # --- Step 3: Create base dataset ---
    # from_tensor_slices slices along the first dimension. TF accepts NumPy natively;
    # no explicit tf.convert_to_tensor needed — conversion happens lazily when iterated.
    ds = tf.data.Dataset.from_tensor_slices((variable_inputs, targets))

    # --- Step 4: Cache (optional) ---
    # Cache stores dataset in memory after first pass. Speeds up later epochs; disable for
    # large datasets that won't fit in RAM.
    if cache:
        ds = ds.cache()

    # --- Step 5: Shuffle (optional) ---
    # Shuffle before batching so each mini-batch sees varied samples. Buffer limits RAM;
    # larger = better randomness, more memory. Default caps at 50k. Skip when empty.
    if shuffle and n_samples > 0:
        buffer = shuffle_buffer if shuffle_buffer is not None else min(n_samples, 50_000)
        ds = ds.shuffle(buffer_size=max(1, buffer), reshuffle_each_iteration=True)

    # --- Step 6: Batch ---
    # Group samples into mini-batches for efficient GPU transfer. drop_remainder=True
    # drops the last incomplete batch (useful when the model expects fixed batch size).
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    # --- Step 7: Inject static inputs (GNN / graph models only) ---
    # Static inputs (e.g. trade_features, adjacency_matrix) have no sample dimension —
    # they are the same for every sample. Convert once to tf.constant; the map adds them
    # to every batch. Without this, the model would need static data separately. The map
    # merges static + variable into a single inputs dict for model.call(inputs_dict).
    if static_inputs:
        static_tensors = _build_static_tensors(static_inputs, ensure_float32)

        def merge_static(var_batch: Any, tgt_batch: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
            if isinstance(var_batch, dict):
                merged = {**static_tensors, **var_batch}
            else:
                merged = {**static_tensors, variable_input_key: var_batch}
            return merged, tgt_batch

        ds = ds.map(merge_static, num_parallel_calls=tf.data.AUTOTUNE)

    # --- 8. Prefetch ---
    # Overlaps data loading with training; improves GPU utilisation.
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ---------------------------------------------------------------------------
# Lightweight data container returned by synthetic generators
# ---------------------------------------------------------------------------

@dataclass
class SyntheticData:
    """
    Lightweight container for synthetic dataset arrays.

    Returned by ``create_pricing_dataset`` and ``create_calibration_dataset``.
    This is a pure data container — it does **not** normalise, split, or convert
    to ``tf.data.Dataset``.  Use sklearn and ``build_tf_dataset`` for that.

    Attributes
    ----------
    features : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    targets : np.ndarray
        Target array of shape ``(n_samples,)`` or ``(n_samples, n_outputs)``.
    feature_names : list of str
        Human-readable names for each feature column.
    target_names : list of str
        Human-readable names for each target column.
    metadata : dict
        Generation parameters (ranges, seed, pricing method, etc.).
    """

    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        n_feat = self.features.shape[1] if self.features.ndim > 1 else 1
        n_tgt = self.targets.shape[-1] if self.targets.ndim > 1 else 1
        return (
            f"SyntheticData(n_samples={len(self)}, "
            f"n_features={n_feat}, n_targets={n_tgt})"
        )


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def create_pricing_dataset(
    n_samples: int = 10_000,
    spot_range: Tuple[float, float] = (80.0, 120.0),
    strike_range: Tuple[float, float] = (80.0, 120.0),
    vol_range: Tuple[float, float] = (0.1, 0.5),
    rate_range: Tuple[float, float] = (0.01, 0.10),
    expiry_range: Tuple[float, float] = (0.1, 2.0),
    seed: Optional[int] = None,
    pricing_fn: Optional[callable] = None,
) -> SyntheticData:
    """
    Generate a synthetic option pricing dataset.

    Creates random option parameters and computes prices using either a
    provided pricing function or a Black-Scholes closed-form.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    spot_range : tuple of float
        ``(min, max)`` for spot price.
    strike_range : tuple of float
        ``(min, max)`` for strike price.
    vol_range : tuple of float
        ``(min, max)`` for volatility.
    rate_range : tuple of float
        ``(min, max)`` for risk-free rate.
    expiry_range : tuple of float
        ``(min, max)`` for time to expiry.
    seed : int, optional
        Random seed for reproducibility.
    pricing_fn : callable, optional
        Custom pricing function ``(spot, strike, vol, rate, expiry, is_call) -> price``.

    Returns
    -------
    SyntheticData
        Features, targets, column names, and generation metadata.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random parameters
    spot = np.random.uniform(*spot_range, n_samples)
    strike = np.random.uniform(*strike_range, n_samples)
    vol = np.random.uniform(*vol_range, n_samples)
    rate = np.random.uniform(*rate_range, n_samples)
    expiry = np.random.uniform(*expiry_range, n_samples)
    is_call = np.random.choice([0.0, 1.0], n_samples)

    features = np.stack([spot, strike, vol, rate, expiry, is_call], axis=1).astype(np.float32)

    # Compute prices
    if pricing_fn is not None:
        prices = np.array(
            [pricing_fn(s, k, v, r, t, c) for s, k, v, r, t, c in features],
            dtype=np.float32,
        )
    else:
        prices = _black_scholes_price(spot, strike, vol, rate, expiry, is_call)

    return SyntheticData(
        features=features,
        targets=prices,
        feature_names=["spot", "strike", "volatility", "rate", "time_to_expiry", "is_call"],
        target_names=["price"],
        metadata={
            "n_samples": n_samples,
            "spot_range": spot_range,
            "strike_range": strike_range,
            "vol_range": vol_range,
            "rate_range": rate_range,
            "expiry_range": expiry_range,
            "seed": seed,
            "pricing_method": "custom" if pricing_fn else "black_scholes",
        },
    )


def _black_scholes_price(
    spot: np.ndarray,
    strike: np.ndarray,
    vol: np.ndarray,
    rate: np.ndarray,
    expiry: np.ndarray,
    is_call: np.ndarray,
) -> np.ndarray:
    """Closed-form Black-Scholes call/put price."""
    from scipy.stats import norm

    sqrt_t = np.sqrt(expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * expiry) / (vol * sqrt_t + 1e-8)
    d2 = d1 - vol * sqrt_t

    call_price = spot * norm.cdf(d1) - strike * np.exp(-rate * expiry) * norm.cdf(d2)
    put_price = strike * np.exp(-rate * expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    prices = np.where(is_call > 0.5, call_price, put_price)
    return prices.astype(np.float32)


def create_calibration_dataset(
    n_samples: int = 5000,
    n_strikes: int = 10,
    n_expiries: int = 5,
    model: str = "heston",
    seed: Optional[int] = None,
) -> SyntheticData:
    """
    Generate a synthetic calibration dataset.

    Creates random model parameters, generates implied-volatility surfaces,
    and returns ``(IV surface, model parameters)`` pairs for calibration training.

    Parameters
    ----------
    n_samples : int
        Number of parameter sets to generate.
    n_strikes : int
        Number of strike points in the IV surface.
    n_expiries : int
        Number of expiry points in the IV surface.
    model : str
        Target model (``"heston"`` or ``"sabr"``).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SyntheticData
        Features (flattened IV surfaces), targets (model parameters), column
        names, and generation metadata.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random model parameters
    if model == "heston":
        v0 = np.random.uniform(0.01, 0.1, n_samples)
        kappa = np.random.uniform(0.5, 5.0, n_samples)
        theta = np.random.uniform(0.01, 0.1, n_samples)
        sigma = np.random.uniform(0.1, 0.8, n_samples)
        rho = np.random.uniform(-0.9, -0.1, n_samples)

        params = np.stack([v0, kappa, theta, sigma, rho], axis=1)
        param_names = ["v0", "kappa", "theta", "sigma", "rho"]

    elif model == "sabr":
        alpha = np.random.uniform(0.1, 0.5, n_samples)
        beta = np.random.uniform(0.3, 1.0, n_samples)
        rho = np.random.uniform(-0.8, 0.0, n_samples)
        nu = np.random.uniform(0.1, 0.8, n_samples)

        params = np.stack([alpha, beta, rho, nu], axis=1)
        param_names = ["alpha", "beta", "rho", "nu"]
    else:
        raise ValueError(f"Unknown model: {model}")

    # Generate synthetic IV surfaces
    moneyness = np.linspace(0.8, 1.2, n_strikes)
    expiries = np.linspace(0.1, 2.0, n_expiries)

    features_list = []
    for i in range(n_samples):
        base_vol = np.sqrt(params[i, 0] if model == "heston" else params[i, 0])
        skew = params[i, -1] if model == "heston" else params[i, 2]

        iv_surface = np.zeros((n_strikes, n_expiries))
        for j, m in enumerate(moneyness):
            for k, t in enumerate(expiries):
                iv = base_vol * (1 + 0.1 * skew * (m - 1.0) + 0.05 * np.sqrt(t))
                iv_surface[j, k] = max(iv, 0.01)

        features_list.append(iv_surface.flatten())

    features = np.array(features_list, dtype=np.float32)

    feature_names = [
        f"iv_m{m:.2f}_t{t:.2f}" for m in moneyness for t in expiries
    ]

    return SyntheticData(
        features=features,
        targets=params.astype(np.float32),
        feature_names=feature_names,
        target_names=param_names,
        metadata={
            "n_samples": n_samples,
            "n_strikes": n_strikes,
            "n_expiries": n_expiries,
            "model": model,
            "moneyness": moneyness.tolist(),
            "expiries": expiries.tolist(),
            "seed": seed,
        },
    )


__all__ = [
    "build_tf_dataset",
    "SyntheticData",
    "create_pricing_dataset",
    "create_calibration_dataset",
]
