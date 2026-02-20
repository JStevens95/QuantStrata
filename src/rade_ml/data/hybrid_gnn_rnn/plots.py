"""
This module holds all visualisations and plots that are unique to HybridGnnRnn model.
"""
from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, Union


def plot_kde_distributions(df: pd.DataFrame, save_path: Union[str, Path]) -> None:
    """
    Plot kernel density estimates to compare distributions of calibration and validation periods and elementary vs
    target trades

    :param df: Dataframe containing elementary pnl target pnl and period columns.
    :param save_path: file path to save plots.
    :return:
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # KDE for elementary pnl
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='elementary_pnl', hue='period', fill=True, common_norm=False, palette='crest', alpha=0.5,
        linewidth=0
    )
    plt.title('Elementary PnL Distribution: Training vs Validation')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.savefig(Path(save_path, 'elem_distribution.png'))

    # CKDE for elementary pnl.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='elementary_pnl', hue='period', fill=True, common_norm=False, cumulative=True, palette='crest',
        alpha=0.5,
    )
    plt.title('Elementary PnL Cumulative Distribution: Training vs Validation')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.savefig(Path(save_path, 'elem_c_distribution.png'))

    # KDE for target pnl.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='target_pnl', hue='period', fill=True, common_norm=False, palette='crest', alpha=0.5, linewidth=0
    )
    plt.title('Target PnL Distribution: Training vs Validation')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.savefig(Path(save_path, 'targ_distribution.png'))

    # CKDE for target pnl.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='target_pnl', hue='period', common_norm=False, alpha=0.5, cumulative=True, common_grid=True,
        palette='crest',
    )
    plt.title('Target PnL Cumulative Distribution: Training vs Validation')
    plt.xlabel('PnL')
    plt.ylabel('Density')
    plt.savefig(Path(save_path, 'targ_c_distribution.png'))

    # KDE for elementary and target pnl.
    plt.figure(figsize=(12, 6))
    sns.kdeplot(
        data=df, x='elementary_pnl', y='target_pnl', hue='period', fill=True, common_norm=False, palette='crest',
        alpha=0.5, linewidth=0
    )
    plt.title('Elementary vs Target PnL Distribution: Training vs Validation')
    plt.xlabel('Elementary PnL')
    plt.ylabel('Target PnL')
    plt.savefig(Path(save_path, 'elem_targ_distribution.png'))


def plot_pnl_distribution(
        elementary_pnl: pd.DataFrame, target_pnl: pd.DataFrame, metadata: Dict[str, Any], save_path: Union[str, Path]
) -> None:
    """
    Plot showing pnl distribution across different sample periods.

    :param elementary_pnl: dataframe of elementary trade pnl history.
    :param target_pnl: dataframe of target trade pnl history.
    :param metadata: metadata from transformation concerning different sample periods.
    :param save_path: file path to save plots
    :return:
    """
    # assert pnl scenario dimensions align.
    assert elementary_pnl.shape[0] == target_pnl.shape[0], "Scenario mismatch between elementary & target pnl."

    # aggregate pnl across trades for each scenario.
    elem_pnl_agg = elementary_pnl.sum(axis=1)
    target_pnl_agg = target_pnl.sum(axis=1)

    # create dataframe for plotting.
    sample_df = pd.DataFrame(
        {
        "scenario_id": elem_pnl_agg.index,
        "elementary_pnl": elem_pnl_agg,
        "target_pnl": target_pnl_agg,
        }
    ).reset_index(drop=True)

    # create column to detail whether scenario is in training or validation or test.
    train_set = set(metadata["train_indices"])
    sample_df["period"] = sample_df.index.map(
        lambda idx: "training" if idx in train_set else "validation"
    )

    # plot kernel density estimate distribution.
    plot_kde_distributions(df=sample_df, save_path=save_path)

