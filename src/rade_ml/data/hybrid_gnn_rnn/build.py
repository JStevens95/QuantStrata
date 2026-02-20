"""
Hybrid GNN-RNN model data builder.

Produces train / val/ test tf.data.Dataset splits for hybrid gnn_rnn model. All pipeline settings are read from
HybridGnnRnnDataConfig.

Usage:
    cfg = HybridGnnRnnDataConfig(
        batch_size=32, cache=False, shuffle=True, dimensionality=DimensionalityConfig(...),
        graph_builder=GraphBuilderConfig(...), attribute_encoder=AttributeEncoderConfig(...)
    )
    result = build_data()
    trainer.fit(result.train_ds, result.val_ds)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from itertools import product
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from sklearn.model_selection import train_test_split

from src.rade_ml.features.transforms.standardiser import get_transformer
from src.rade_ml.features.transforms.dimensionality import run_dimensionality_reduction

from src.rade_ml.data.io import CacheLoader
from src.rade_ml.data.dataset import build_tf_dataset
from src.rade_ml.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig, AttributeEncoderConfig, GraphBuilderConfig

from src.rade_ml.data.hybrid_gnn_rnn.plots import plot_pnl_distribution

from src.rade_ml.utilities.attribute_encoder import TradeAttributeEncoder
from src.rade_ml.utilities.graph_builder import TradeGraphBuilder

# define module level logging.
logger = logging.getLogger(__name__)


@dataclass
class HybridGnnRnnResult:
    """Result of build_dataset() and tf.data.Dataset splits for Hybrid GNN-RNN model."""

    train_ds: tf.data.Dataset
    val_ds: Optional[tf.data.Dataset] = None
    test_ds: Optional[tf.data.Dataset] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_dataset():
    """

    :return:
    """
    # --- 0.


def dimension_reduction(pnl_df: pd.DataFrame, config: HybridGnnRnnDataConfig, seed: int = 42) -> Dict[str, Any]:
    """
    Reduce the dimensionality of trade universe to a highly informative and relevatn subset.

    :param pnl_df: dataframe of pnl time-series where index is scenarios and columns are trades.
    :param config: hybrid gnn rnn data pipeline configuration.
    :param seed:  random state seed for reproducibility.
    :return:
    """
    # define list of selected trades.
    selected_trades = list()

    # extract underlying and product type.
    underlying = sorted(list(set(x.split('|')[0] for x in pnl_df.columns.to_list())))
    prod_type = sorted(list(set(x.split('|')[1] for x in pnl_df.columns.to_list())))

    # loop through each underlying + product combination in trade universe.
    for und, prod in product(underlying, prod_type):
        # filter trades for underlying and product type.
        filt_pnl_df = pnl_df[
            [k for k in pnl_df.columns.to_list() if k.split("|")[0] == und if k.split("|")[1] == prod_type]
        ]

        # run dimensionality reduction
        selected_trades += run_dimensionality_reduction(
            pnl_df=filt_pnl_df, config=config.dimensionality, seed=seed, save_path=None,
            file_prefix=f"{und.upper()}_{prod.upper()}"
        )
    # extract set of trades removed.
    removed_trades = list(set(
        [c for c in pnl_df.columns.to_list() if c not in selected_trades]
    ))

    return {
        "selected_trades": selected_trades,
        "removed_trades": removed_trades,
    }



def load_data(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load required data from file paths.
    :param job: dictionary containing job details including cluster info, attributes, assets and elementary trades.
    :return:
    """
    return {
        "target_pnl": CacheLoader.load(file_path=job["target_pnl_path"]),
        "elementary_pnl": CacheLoader.load(file_path=job["elementary_pnl_path"]),
        "target_attribs": CacheLoader.load(file_path=job["target_attribs_path"]),
        "elementary_attribs": CacheLoader.load(file_path=job["elementary_attribs_path"]),

    }


def prepare_input_data(job: Dict[str, Any], config: HybridGnnRnnDataConfig):
    """
    preprocessing of input data for Hybrid GNN-RNN model.
    :param job: dictionary containing job details including cluster info, attributes, assets and elementary trades.
    :param config: HybridGnnRnnDataConfig configuration for data pipeline for hybrid gnn rnn model.
    :return:
    """
    # extract cluster information from specified job.
    cluster_info = job["cluster_info"]

    # ---- 0. extract data from provided job. ----
    data_dict = load_data(job=cluster_info)

    # ---- 1. standardisation of pnl history ----
    scaled_data_dict = standardise_pnl_history(
        elementary_pnl=data_dict["elementary_pnl_df"], target_pnl=data_dict["target_pnl_df"], config=config
    )

    # ------ 1.1. diagnostic plot of training and validation distributions. ------
    if config.plot_pnl_distribution:
        plot_pnl_distribution(
            elementary_pnl=scaled_data_dict["elementary_pnl_df"], target_pnl=scaled_data_dict["target_pnl_df"],
            metadata=scaled_data_dict["metadata"], save_path=""
        )

    # ---- 2. apply dimensionality reduction to elementary trades. ----
    trade_reduction = dimension_reduction(
        pnl_df=scaled_data_dict["elementary_pnl_df"], config=config, seed=config.seed
    )

    # ------ 2.1. update elementary trade attributes to reflect reduced population ------
    elem_attribs = _update_trade_attribs(
        trade_attribs=scaled_data_dict["elementary_attribs"], selected_trades=scaled_data_dict["selected_trades"],
        save_path=None
    )

    # ------ 2.2. update elementary trade pnl to reflect reduced population. ------
    elem_pnl = _update_trade_pnl(
        trade_pnl=scaled_data_dict["elementary_pnl"], selected_trades=scaled_data_dict["selected_trades"],
        save_path=None
    )

    # ------ 2.3. validate input data ------

    # ---- 3. encode trade attributes ----

    # ---- 4. construct trade graph ----


def standardise_pnl_history(
        elementary_pnl: pd.DataFrame, target_pnl: pd.DataFrame, config: HybridGnnRnnDataConfig
) -> Dict[str, Any]:
    """
    Standardise pnl history for training and validation.

    Combine into pnl time-series according to data pipeline configuration, fit to scala to training data and
    then transform validation data.
    :param elementary_pnl: dataframe of elementary pnl history
    :param target_pnl: dataframe of target pnl history
    :param config: hybrid gnn-rnn data pipeline configuration.
    :return:
    """
    # ---- 0. split scenario into train / val / test periods.
    pnl_periods = split_pnl_periods(
        elementary_pnl=elementary_pnl, target_pnl=target_pnl, val_split=config.validation_split,
        test_split=config.test_split, shuffle=config.shuffle, seq_length=config.seq_length, seed=config.seed
    )
    elementary_pnl_arr = elementary_pnl.to_numpy(copy=False)
    target_pnl_arr = target_pnl.to_numpy(copy=False)

    # ---- 1. create transformer for elementary & target trades ----
    # TODO: check this is correct - should be one transformer.
    elementary_transformer = get_transformer(transform_type=config.transform_type)
    target_transformer = get_transformer(transform_type=config.transform_type)

    # ---- 2. fit transformer on training data only - implementing epsilon threshold for scaling ----
    training_idx = pnl_periods["train_ends"]
    elementary_transformer.fit(elementary_pnl_arr[training_idx, :])
    target_transformer.fit(target_pnl_arr[training_idx, :])

    # ---- 3. transform full pnl history (train/val/test) ----
    elementary_pnl_scaled = elementary_transformer.transform(elementary_pnl_arr)
    target_pnl_scaled = target_transformer.transform(target_pnl_arr)

    # ---- 4. build metadata dictionary ----
    metadata = {
        "scenarios": elementary_pnl.index.tolist(),
        "elementary_ids": elementary_pnl.columns.tolist(),
        "elementary_idx": np.arange(0, elementary_pnl.shape[-1]),
        "target_ids": target_pnl.columns.tolist(),
        "target_idx": np.arange(elementary_pnl.shape[-1], elementary_pnl.shape[-1] + target_pnl.shape[-1]),
        "train_scenarios": [elementary_pnl.index[i] for i in pnl_periods["train_indices"].tolist()],
        "train_end_scenarios": [elementary_pnl.index[i] for i in pnl_periods["train_ends"].tolist()],
        "val_scenarios": [elementary_pnl.index[i] for i in pnl_periods["val_indices"].tolist()],
        "val_ends_scenarios": [elementary_pnl.index[i] for i in pnl_periods["val_ends"].tolist()],
        "test_scenarios": [elementary_pnl.index[i] for i in pnl_periods["test_indices"].tolist()],
        "test_end_scenarios": [elementary_pnl.index[i] for i in pnl_periods["test_ends"].tolist()],
        "elementary_pnl_transformer": elementary_transformer,
        "target_pnl_transformer": target_transformer,
    }
    metadata |= pnl_periods
    return {
        "elementary_pnl_df": pd.DataFrame(
            elementary_pnl_scaled, columns=elementary_pnl.columns.tolist(), index=elementary_pnl.index.to_list(),
        ),
        "target_pnl_df": pd.DataFrame(
            target_pnl_scaled, columns=target_pnl.columns.tolist(), index=elementary_pnl.index.to_list(),
        ),
        "metadata": metadata,
    }


def split_pnl_periods(
        elementary_pnl: pd.DataFrame, target_pnl: pd.DataFrame, val_split: Optional[float] = None,
        test_split: Optional[float] = None, shuffle: bool = False, seq_length: Optional[int] = 1,
        seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Simple chronological train/val/test pnl splitting.
    :param elementary_pnl: dataframe of elementary pnl history
    :param target_pnl: dataframe of target pnl history
    :param val_split: fraction of sample used for validation dataset
    :param test_split: fraction of sample used for testing dataset
    :param shuffle: whether to shuffle the pnl history
    :param seq_length: length of pnl sequence to generate.
    :param seed: random state seed for reproducibility.
    :return:
    """
    # ------ 0.1. input validation ------
    if not isinstance(elementary_pnl, pd.DataFrame) or not isinstance(target_pnl, pd.DataFrame):
        raise TypeError("Elementary & Target PnL must be pandas dataframes.")
    if elementary_pnl.shape[0] != target_pnl.shape[0]:
        raise ValueError("Elementary & Target pnl must have the same number of scenarios.")
    if not elementary_pnl.index.equals(target_pnl.index):
        raise ValueError("Elementary & Target pnl must have the same scenario index.")
    if seq_length < 1:
        raise ValueError("Sequence length must be greater than 0.")
    if not (0.0 <= val_split < 1.0) or not (0.0 <= test_split < 1.0) or (val_split + test_split < 1.0):
        raise ValueError("Validation and Test splits must be in the range [0.0, 1.0] and not be greater than 1.0.")

    # ------ 0.2. basic sizes / index arrays ------
    num_scenarios, num_elementary = elementary_pnl.shape
    _, num_target = target_pnl.shape
    scenario_idx = np.arange(num_scenarios)

    # ---- 1. allocate test indices, if specified ----
    train_val_idx, test_idx = train_test_split(scenario_idx, test_size=test_split, shuffle=shuffle, random_state=seed)
    logger.info(
        f"[test_split: {test_split}] Allocated {len(test_idx)} scenarios for testing: {test_idx[0]}...{test_idx[-1]}"
    )

    # ---- 2. allocate training / validation indices ----
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_split, shuffle=shuffle, random_state=seed)
    logger.info(
        f"[train_split: {1 - val_split - test_split}] Allocated {len(train_idx)} scenarios for testing: {train_idx[0]}"
        f"...{train_idx[-1]}"
    )
    logger.info(
        f"[val_split: {val_split}] Allocated {len(val_idx)} scenarios for testing: {val_idx[0]}...{val_idx[-1]}"
    )

    # ---- 3. adjust indices for sequential length ----
    train_starts, train_ends = window_starts_from_days(scenario_idx=train_idx, sequence_length=seq_length)
    val_starts, val_ends = window_starts_from_days(scenario_idx=val_idx, sequence_length=seq_length)
    test_starts, test_ends = window_starts_from_days(scenario_idx=test_idx, sequence_length=seq_length)

    return {
        "train_indices": train_idx, "val_indices": val_idx, "test_indices": test_idx, "train_starts": train_starts,
        "val_starts": val_starts, "test_starts": test_starts, "train_ends": train_ends, "val_ends": val_ends,
        "test_ends": test_ends, "sequence_length": seq_length, "test_size": test_split, "val_size": val_split,
        "train_size": (1 - (test_split + val_split))
    }


def window_starts_from_days(scenario_idx: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an ordered array of absolute scenario indices (days), return start and end indices for all valid sequences.

    :param scenario_idx:
    :param sequence_length:
    :return:
    """
    # create array of indices.
    scens = np.asarray(scenario_idx, dtype=int)

    # return empty array if indices are empty & if sequence length is 1 (trivial case).
    if scenario_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if sequence_length == 1:
        starts = scens.copy()
        ends = scens.copy()
        return starts, ends

    # find break points where consecutive days are not continuous.
    diffs = np.diff(scens)
    gap_positions = np.nonzero(diffs != 1)[0]   # positions of 'gaps' in days (index into days)

    # run start and end positions (indices into 'scens' array)
    run_start_pos = np.concatenate(([0], gap_positions + 1))
    run_end_pos = np.concatenate((gap_positions, [scens.size - 1]))

    starts_list = []
    ends_list = []
    for rs, re in zip(run_start_pos, run_end_pos):
        run_len = re - rs + 1
        n_valid = run_len - sequence_length + 1
        if n_valid > 0:
            # take the first n valid day values of this run as start indices.
            run_starts = scens[rs: rs + n_valid]
            run_ends = run_starts + (sequence_length - 1)
            starts_list.append(run_starts)
            ends_list.append(run_ends)
    if not starts_list or not ends_list:
        return np.empty((0, ), dtype=int), np.empty((0, ), dtype=int)
    starts = np.concatenate(starts_list).astype(int)
    ends = np.concatenate(ends_list).astype(int)
    return starts, ends


def _update_trade_attributes(
        trade_attribs: Dict[str, List[Any]], selected_trades: List[str], id_key: str = "trade_id",
        save_path: Optional[Union[str, Path]] = None
) -> Dict[str, List[Any]]:
    """
    Update trade attributes to reflect the trades which have been removed via dimensionality reduction.

    :param trade_attribs: dictionary of trade attributes.
    :param selected_trades: list of trade ids to keep
    :param id_key: key in dictionary for trade id
    :param save_path: path to save updated trade attributes.
    :return:
    """
    if id_key not in trade_attribs:
        raise KeyError(f"{id_key} missing in trade attribute keys={list(trade_attribs.keys())}.")

    # convert the current ID sequence to a plain python list for easy indexing / lookup.
    current_trade_ids: List[str] = list(trade_attribs[id_key])

    # total number of trades currently present.
    total_trades: int = len(current_trade_ids)

    # --- Build mapping from trade id --> row index is the current attributes ---

    # create dictionary mapping each trade_id to its current row position.
    id_to_position: Dict[str, int] = {tid: i for i, tid in enumerate(selected_trades)}

    # validate that every selected IR actually exists in the source attributes
    missing_ids = [tid for tid in selected_trades if tid not in id_to_position]
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        more = " ..." if len(missing_ids) > 10 else ""
        raise ValueError(f"Selected IDs unavailable in attributes: {preview}{more}")

    # guard against accidental duplicates in the required selection.
    if len(set(selected_trades)) != len(selected_trades):
        # collect which IDs are duplicated to help the caller fix upstream issues.
        dupes = [tid for tid in set(selected_trades) if selected_trades(tid) > 1]
        preview = ", ".join(dupes[:10])
        more = " ..." if len(dupes) > 10 else ""
        raise ValueError(f"Selected IDs unavailable in attributes: {preview}{more}")

    # build the row-index array to keep, in exactly the same order as 'selected_trades'
    keep_idx = [id_to_position[tid] for tid in selected_trades]

    # prepare output dict (we don't mutate the input; produce a clean copy)
    out: Dict[str, Any] = {}

    # --- main filtering / reordering logic ---

    # iterate over every key/value in the attribute dict and filter / reorder by rows.
    for k, v in trade_attribs.items():
        # strict: every key must be per trade 1D
        try:
            lgnth = len(v)
        except Exception:
            raise TypeError(f"Key '{k}' is not a 1D per-trade list")
        if lgnth != total_trades:
            raise ValueError(f"Key '{k}' length {lgnth} != total trade length {total_trades}")

        # reorder to keep idx keep list type.
        out[k] = [v[i] for i in keep_idx]

    out[id_key] = list(selected_trades)
    if out[id_key] != list(selected_trades):
        raise AssertionError("Post-filter trade_id order mismatch")
    if save_path:
        CacheLoader.save_data(data=out, file_path=save_path)
    return out


def _update_trade_pnl(
        trade_pnl: pd.DataFrame, selected_trades: List[str], save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Update trade pnl time-series to reflect the trades which have been removed via dimensionality reduction.

    :param trade_pnl: dataframe of trades pnl time-series
    :param selected_trades: list of trade ids to keep
    :param save_path: path to save updated trade attributes.
    :return:
    """
