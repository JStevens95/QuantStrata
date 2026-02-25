"""
Module is responsible for reducing dimensionality of a dataset to enhance machine learning capabilities.
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.linalg import qr
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from typing import Optional, List, Literal, Union

from src.rade_ml.validation.exceptions import UndefinedReductionType, UndefinedComputationMethod

from src.rade_ml.data.hybrid_gnn_rnn.config import DimensionalityConfig, BasisSelectionConfig


def basis_selection_reduction(
        pnl_df: pd.DataFrame, config: BasisSelectionConfig, seed: int = 42, file_prefix: Optional[str] = None,
        save_path: Optional[str] = None
):
    """
    Compute a minimal basis of trades with 2 steps
        1. determine effective rank 'r' via SVD to meet variance threshold.
        2. use pivoted QR to pick the top-r actual trades.

    :param pnl_df: dataframe of pnl time-series where index is scenarios and columns are trades.
    :param config: configuration for basis selection dimensionality reduction.
    :param file_prefix: file name prefix
    :param seed: random state seed for reproducibility.
    :param save_path: [optional] file path to save plots.
    :return:
    """

    # 1. compute effective rank (intrinsic dimension)
    r: int = compute_effective_rank(
        pnl_df=pnl_df, seed=seed, var_threshold=config.var_threshold, method=config.method,
        max_components=config.max_components,
    )

    # 2. select r basis trades.
    basis: List[str] = select_minimal_basis(pnl_df=pnl_df, k=r, weight_tail=config.weight_tail)

    # 3. [optional] plot cumulative variance.
    if save_path:
        plot_cumulative_variance(
            pnl_df=pnl_df, var_thresholds=[0.95, config.var_threshold], method=config.method,
            max_components=config.max_components, save_path=save_path, file_prefix=file_prefix,
        )
    return basis

def compute_effective_rank(
        pnl_df: pd.DataFrame, seed: int, var_threshold: float = 0.995, method: Literal["svd", "pca", "svds"] = "pca",
        max_components: int = 50
) -> int:
    """
    Determine the minimum number of singular components 'r' needed to explain at selected total variance.

    :param pnl_df: dataframe of pnl time-series where index is scenarios and columns are trades.
    :param seed: random state seed for reproducibility.
    :param var_threshold: cumulative variance threshold.
    :param method: computation method.
    :param max_components: maximum number of components.
    :return:
    """
    # convert dataframe to numpy array for linear algebra.
    x: np.ndarray = pnl_df.values
    t, n = x.shape
    if method == "pca":
        # pca autp-selects n components to meet variance threshold.
        pca = PCA(n_components=var_threshold, svd_solver="full", random_state=seed)
        pca.fit(x)
        return pca.n_components_
    elif method == "svd":
        # determine component cap
        n_comp = min(max_components, t - 1, n - 1)

        # randomised svd for top-n components singular values.
        _, s, _ = randomized_svd(x, n_components=n_comp, random_state=seed)
    elif method == "svds":
        # determine component cap
        n_comp = min(max_components, t - 1, n - 1)

        # arpack compute k largest, returns ascending.
        u, s, vt = svds(x, k=n_comp)
        s = np.sort(s)[::-1]
    else:
        raise UndefinedComputationMethod(f"Undefined computation method: {method}")
    # calculate variances.
    variances = s**2
    cumvar = np.cumsum(variances) / np.sum(variances)

    # find minimal r
    r: int = int(np.searchsorted(cumvar, var_threshold)) + 1
    return r


def plot_cumulative_variance(
        pnl_df: pd.DataFrame, var_thresholds: List[float] = [0.95, 0.99], method: Literal["svd", "pca", "svds"] = "pca",
        max_components: int = 50, save_path: Optional[str] = None, file_prefix: Optional[str] = None,
) -> None:
    """
    Plot cumulative variance and print effective rank for thresholds.
    :param pnl_df: dataframe of pnl time-series where index is scenarios and columns are trades.
    :param var_thresholds: list of cumulative variance thresholds.
    :param method: computation method.
    :param max_components: maximum number of components.
    :param save_path: path to save plots.
    :param file_prefix: file name prefix
    :return:
    """

def run_dimensionality_reduction(
        pnl_df: pd.DataFrame, config: DimensionalityConfig, seed: int = 42, save_path: Optional[str] = None,
        file_prefix: Optional[str] = None,
) -> List[str]:
    """
    Run dimensionality reduction with model / parameters specified by the user.

    :param pnl_df: dataframe of pnl time-series where index is scenarios and columns are trades.
    :param config: general dimensionality reduction configuration.
    :param seed: random state seed for reproducibility.
    :param save_path: [optional] path to save plots.
    :param file_prefix: [optional] file name prefix
    :return:
    """
    # run dimensionality reduction for specified type.
    if config.reduction_mode.lower() == 'basis_selection':
        # TODO: validate inputs are correct and available.
        

        # run reduction.
        selected_trades = basis_selection_reduction(
            pnl_df=pnl_df, config=config.basis_selection, seed=seed, save_path=save_path, file_prefix=file_prefix
        )
    else:
        raise UndefinedReductionType(f"Undefined dimensionality reduction mode: {config.reduction_mode}")
    return selected_trades

def select_minimal_basis(pnl_df: pd.DataFrame, k: int, weight_tail: float = 1.0) -> List[str]:
    """
    Select k columns (trades) from pnl dataframe that best span its subspace via pivoted QR.

    :param pnl_df: dataframe of pnl time-series where index is scenarios and columns are trades.
    :param k: number of basis trades to select.
    :param weight_tail: factor to emphasize extreme pnl scenarios.
    :return:
    """
    # extract underlying array.
    x: np.ndarray = pnl_df.values

    # 1. tail weighting boost rows with extreme absoulte pnl
    if weight_tail != 1.0:
        # compute maximum absolute pnl per scenario,
        row_max: np.ndarray = np.max(np.abs(x), axis=1)

        # define threshold: mu + 2*sigma of these maxima
        threshold: float = np.mean(row_max) + 1.645 * np.std(row_max)

        # create weights
        weights: np.ndarray = np.sqrt(np.where(row_max > threshold, weight_tail, 1.0))

        # apply row weights
        x = x * weights[:, None]

    # 2. pivoted qr, economic mode with column pivoting.
    _, _, pivots = qr(x, mode='economic', pivoting=True)

    # 3. select first k pivoting column by their original names.
    selected: List[str] = [pnl_df.columns[i] for i in pivots]
    return selected

