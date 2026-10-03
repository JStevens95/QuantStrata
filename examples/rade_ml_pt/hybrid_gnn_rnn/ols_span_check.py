"""
OLS span check: can the elementary PnL basis linearly replicate the targets?

This is the decisive diagnostic for "why can't the model fit linear products?".
It fits an ordinary least-squares map  Y ≈ X·W + b  on the *full, original-unit*
cluster PnL (elementary -> target) and reports R² / MSE on a held-out split.

Interpretation
--------------
* test R² ≈ 1 (MSE ~1e-3 or lower)
      -> the basis DOES span the targets. Any gap the neural model shows is an
         OPTIMISATION/ARCHITECTURE problem (additive deep stack crowding out the
         linear head), NOT a data problem. Fix on the model side (gate/warm-start
         the polynomial head).

* test R² noticeably < 1 (e.g. ~0.91, matching a scaled val MSE ~0.087)
      -> the basis does NOT span the targets even for least squares. This is a
         DATA/structure problem: the elementary "forward" is likely a linearised
         delta rather than a full-reval FX-forward instrument, so the bilinear
         spot×discount cross term (ΔS·Δdf) is missing. Fix on the data side
         (add a co-terminal full forward instrument) or add a cross-gamma term.

Usage
-----
    python ols_span_check.py \
        --elementary /path/to/cluster/elementary_pnl.parquet \
        --target     /path/to/cluster/target_pnl.parquet

(Point at the RAW cluster cache files so you get the full pre-reduction universe.
These are the same paths used as job["elementary_pnl_path"] / ["target_pnl_path"].)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _r2_per_column(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target R² = 1 - SS_res / SS_tot (column-wise)."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    # Guard constant columns (SS_tot == 0): define R²=1 if residual is ~0 else 0.
    out = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    return out


def ols_span_check(
    elementary_path: str,
    target_path: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    # ---- load full, original-unit PnL (scenarios x trades) ----
    X_df = pd.read_parquet(elementary_path)
    Y_df = pd.read_parquet(target_path)

    # align on the shared scenario index (rows) defensively.
    common = X_df.index.intersection(Y_df.index)
    if len(common) == 0:
        # Fall back to positional alignment if indices don't match.
        n = min(len(X_df), len(Y_df))
        X = X_df.to_numpy(dtype=np.float64)[:n]
        Y = Y_df.to_numpy(dtype=np.float64)[:n]
    else:
        X = X_df.loc[common].to_numpy(dtype=np.float64)
        Y = Y_df.loc[common].to_numpy(dtype=np.float64)

    n_scen, n_elem = X.shape
    _, n_tgt = Y.shape
    print(f"scenarios={n_scen}  elementary={n_elem}  targets={n_tgt}")
    if n_scen <= n_elem:
        print(
            f"  WARNING: n_scenarios ({n_scen}) <= n_elementary ({n_elem}). "
            f"The linear system is under-determined; train R² will look perfect "
            f"by memorisation. Trust the TEST R² only."
        )

    # ---- train/test split (mirror the model's random scenario split) ----
    idx = np.arange(n_scen)
    tr, te = train_test_split(idx, test_size=test_size, shuffle=True, random_state=seed)

    # ---- fit OLS with intercept: augment X with a ones column ----
    Xtr = np.concatenate([X[tr], np.ones((len(tr), 1))], axis=1)
    Xte = np.concatenate([X[te], np.ones((len(te), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xtr, Y[tr], rcond=None)        # [(n_elem+1), n_tgt]

    pred_tr = Xtr @ W
    pred_te = Xte @ W

    # ---- metrics ----
    mse_tr = float(((Y[tr] - pred_tr) ** 2).mean())
    mse_te = float(((Y[te] - pred_te) ** 2).mean())
    r2_tr = _r2_per_column(Y[tr], pred_tr)
    r2_te = _r2_per_column(Y[te], pred_te)

    # scaled MSE: divide each column error by its (train) std so it is comparable
    # to the model's `standard`-space val MSE.
    col_std = Y[tr].std(axis=0)
    col_std = np.where(col_std > 1e-12, col_std, 1.0)
    scaled_mse_te = float((((Y[te] - pred_te) / col_std) ** 2).mean())

    print("\n=== OLS span check (original units) ===")
    print(f"train  MSE={mse_tr:.6g}   R² min={np.nanmin(r2_tr):.5f}  median={np.nanmedian(r2_tr):.5f}")
    print(f"test   MSE={mse_te:.6g}   R² min={np.nanmin(r2_te):.5f}  median={np.nanmedian(r2_te):.5f}")
    print(f"test   scaled-MSE (per-target std-normalised) = {scaled_mse_te:.5f}")
    print("       ^ compare this directly with the model's scaled val MSE (e.g. 0.087)")

    # worst targets to eyeball where the basis (if anywhere) breaks down.
    order = np.argsort(np.nan_to_num(r2_te, nan=-np.inf))
    worst = order[:10]
    cols = list(Y_df.columns)
    print("\nworst 10 targets by test R²:")
    for j in worst:
        name = cols[j] if j < len(cols) else f"target_{j}"
        print(f"  {name:<40s} R²={r2_te[j]:.4f}")

    print("\n--- verdict ---")
    if scaled_mse_te < 0.01:
        print("Basis SPANS the targets (test scaled-MSE < 0.01). The model gap is")
        print("architectural: gate/warm-start the polynomial head so it dominates.")
    else:
        print("Basis does NOT span the targets at OLS. This is structural — likely a")
        print("missing spot×discount cross term (linearised forward instead of full")
        print("reval). Add a co-terminal full forward instrument or a cross-gamma term.")


def main() -> None:
    ap = argparse.ArgumentParser(description="OLS span check for elementary->target PnL.")
    ap.add_argument("--elementary", required=True, help="Path to elementary_pnl parquet (full, raw).")
    ap.add_argument("--target", required=True, help="Path to target_pnl parquet (raw).")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ols_span_check(args.elementary, args.target, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()
