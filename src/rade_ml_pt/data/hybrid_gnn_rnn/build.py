"""
Hybrid GNN-RNN model data builder (PyTorch).

Produces train / val / test DataLoader splits for hybrid gnn_rnn model. All pipeline settings are read from
HybridGnnRnnDataConfig.

Usage:
    cfg = HybridGnnRnnDataConfig(
        batch_size=32, cache=False, shuffle=True, dimensionality=DimensionalityConfig(...),
        graph_builder=GraphBuilderConfig(...), attribute_encoder=AttributeEncoderConfig(...)
    )
    result = build_dataset(job, cfg)
    trainer.fit(result.train_ds, result.val_ds)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from sklearn.model_selection import train_test_split

from src.rade_ml_pt.features.transforms.standardiser import get_transformer
from src.rade_ml_pt.features.transforms.dimensionality import run_dimensionality_reduction

from src.rade_ml_pt.data.io import CacheLoader
from src.rade_ml_pt.data.result import DataBuildResult
from src.rade_ml_pt.data.dataset import build_dataloader
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig, AttributeEncoderConfig, GraphBuilderConfig

from src.rade_ml_pt.utilities.attribute_encoder import TradeAttributeEncoder
from src.rade_ml_pt.utilities.graph_builder import TradeGraphBuilder

# define module level logging.
logger = logging.getLogger(__name__)


def _set_feature_names(transformer: Any, names: np.ndarray) -> None:
    """Set ``feature_names_in_`` on a transformer, handling sklearn Pipelines.

    In sklearn >=1.2 ``Pipeline.feature_names_in_`` is a read-only property
    that delegates to the first step.  Setting it directly raises
    ``AttributeError``, so for Pipelines we set the attribute on the first
    step instead — the property then exposes it transparently.
    """
    from sklearn.pipeline import Pipeline
    if isinstance(transformer, Pipeline):
        transformer.steps[0][1].feature_names_in_ = names
    else:
        transformer.feature_names_in_ = names


@dataclass
class HybridGnnRnnResult(DataBuildResult):
    """Result of build_dataset() and DataLoader splits for Hybrid GNN-RNN model."""
    # cluster information for training run -- from input job.
    cluster_info: Dict[str, Any] = None

    # data configuration -- configuration to build input data.
    data_config: HybridGnnRnnDataConfig = None

    # elementary result objects
    elementary_pnl: Optional[pd.DataFrame] = None
    elementary_attributes: Optional[Dict[str, List[Any]]] = None

    # target result objects
    target_pnl: Optional[pd.DataFrame] = None
    target_attributes: Optional[Dict[str, List[Any]]] = None

    # graph builder objects
    graph_builder: Optional[TradeGraphBuilder] = None
    graph_results: Optional[Dict[str, Any]] = None

    # attribute encoder objects
    encoder: Optional[TradeAttributeEncoder] = None
    encoder_results: Optional[Dict[str, Any]] = None


def build_dataset(
        job: Dict[str, Any], config: HybridGnnRnnDataConfig
) -> HybridGnnRnnResult:
    """
    Model-specific function responsible for generating and processing job information to give ml input data.

    Pipeline:
        0. Load raw trade pnl & attributes from job paths.
        1. Standardise PnL history (fit scaler on train, transform all).
        2. Dimensionality reduction on elementary trades.
        3. Encode trade attributes (one-hot, multi-label, numeric scaling).
        4. Build trade graph (k-NN adjacency with RBF kernel).
        5. Construct DataLoaders for train / val / test splits.

    :param job: dictionary containing job details, including cluster_info, attributes, assets and elementary trades.
    :param config: model-specific configuration for data pipeline for hybrid gnn rnn model.
    :return:
    """
    # ---- 0. load raw data ----
    cluster_info = job["cluster_info"]
    data_dict = load_data(job=cluster_info)

    # ---- 1. standardisation of pnl history ----
    scaled_data_dict = standardise_pnl_history(
        elementary_pnl=data_dict["elementary_pnl"], target_pnl=data_dict["target_pnl"], config=config,
    )

    # ---- 2. apply dimensionality reduction to elementary trades. ----
    trade_reduction = dimension_reduction(
        pnl_df=scaled_data_dict["elementary_pnl_scaled"], config=config, seed=config.seed
    )

    # ------ 2.1. update elementary attributes & pnl to reflect reduced population. ------
    elementary_attributes = _update_trade_attributes(
        trade_attribs=data_dict["elementary_attribs"], selected_trades=trade_reduction["selected_trades"],
    )
    elementary_pnl = _update_trade_pnl(
        trade_pnl=scaled_data_dict["elementary_pnl_scaled"], selected_trades=trade_reduction["selected_trades"],
    )
    target_attributes, target_pnl = data_dict["target_attribs"], scaled_data_dict["target_pnl_scaled"]

    # ------ 2.2. validate input data ------
    _validate_input_data(elementary_attributes, elementary_pnl.to_numpy(), target_attributes, target_pnl.to_numpy())

    # ------ 2.3. build metadata ------
    metadata = build_metadata(
        metadata={**scaled_data_dict["metadata"], **trade_reduction}, elementary_pnl=elementary_pnl,
        target_pnl=target_pnl
    )

    # ---- 3. encode trade attributes ----
    encoder, encoder_results = encode_trade_attributes(
        config=config.attribute_encoder, elementary_attrs=elementary_attributes, target_attrs=target_attributes
    )

    # ---- 4. construct trade graph ----
    graph_builder, graph_result = build_trade_graph(
        config=config.graph_builder, encoded_features=encoder_results["combined_encoded"]
    )

    # ---- 5. build DataLoader splits ----
    # extract adjacency components as numpy arrays for static inputs
    elem_pnl = elementary_pnl.to_numpy()
    tgt_pnl = target_pnl.to_numpy()

    static_inputs = {
        "trade_features": encoder_results["combined_features"],
        "adjacency_indices": graph_result["sparse_indices"],
        "adjacency_values": graph_result["sparse_values"],
        "adjacency_dense_shape": np.array(graph_result["sparse_shape"], dtype=np.int64),
        "elementary_indices": metadata["elementary_idx"],
        "target_indices": metadata["target_idx"],
    }

    # training period
    logger.info("Building train dataloader...")
    train_ds = _make_dataloader(
        config=config, trade_pnl=elem_pnl, target_pnl=tgt_pnl,
        period_starts=metadata["train_starts"], static_inputs=static_inputs,
    )
    logger.info("Train dataloader built.")

    # validation period (if available / specified)
    val_ds = _make_dataloader(
        config=config, trade_pnl=elem_pnl, target_pnl=tgt_pnl,
        period_starts=metadata["val_starts"], static_inputs=static_inputs
    ) if metadata["val_size"] > 0.0 else None

    # testing period (if available / specified)
    test_ds = _make_dataloader(
        config=config, trade_pnl=elem_pnl, target_pnl=tgt_pnl,
        period_starts=metadata["test_starts"], static_inputs=static_inputs
    ) if metadata["test_size"] > 0.0 else None

    return HybridGnnRnnResult(
        # cluster information.
        cluster_info=cluster_info,

        # data configuration
        data_config=config,

        # elementary trade objects
        elementary_pnl=elementary_pnl, elementary_attributes=elementary_attributes,

        # target trade objects
        target_pnl=target_pnl, target_attributes=target_attributes,

        # graph builder objects
        graph_builder=graph_builder, graph_results=graph_result,

        # attribute encoder objects
        encoder=encoder, encoder_results=encoder_results,

        # DataLoader splits
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,

        # general metadata
        metadata=metadata
    )


def build_trade_graph(
        config: GraphBuilderConfig, encoded_features: Dict[str, np.ndarray],
) -> Tuple[TradeGraphBuilder, Dict[str, Any]]:
    """
    Construct the k-NN trade relationship graph from encoded attributes.

    :param config: graph builder configuration.
    :param encoded_features: output of TradeAttributeEncoder.fit_transform().
    :return: (graph_builder instance, graph result dict with sparse adjacency).
    """
    # initiate graph builder object.
    builder = TradeGraphBuilder(
        distance_metric=config.distance_metric, k=config.k, alpha_moneyness=config.alpha_moneyness,
        alpha_maturity=config.alpha_maturity, alpha_delta=config.alpha_delta, alpha_vega=config.alpha_vega,
        alpha_prod_type=config.alpha_prod_type, alpha_prod_subtype=config.alpha_prod_subtype,
        alpha_underlying=config.alpha_underlying, alpha_underlying_rf=config.alpha_underlying_rf,
        p_min_elementary=config.p_min_elementary, q_min_target=config.q_min_target,
    )

    # build trade graph
    graph_result = builder.build_graph(encoded_trades=encoded_features, include_quota=config.include_quota)
    return builder, graph_result


def build_metadata(metadata: Dict[str, Any], elementary_pnl: pd.DataFrame, target_pnl: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a consolidated metadata to be used in training/evaluation & inference pipelines.

    :param metadata: collection of metadata for sorting.
    :param elementary_pnl: dataframe of elementary trades pnl time-series.
    :param target_pnl:  dataframe of target trades pnl time-series.
    :return:
    """
    n_scenarios, n_elementary = elementary_pnl.to_numpy().shape
    _, n_targets = target_pnl.to_numpy().shape
    return {
        # scenario / split tracking.
        "scenarios": metadata["scenarios"],
        "scenario_idx": np.arange(len(elementary_pnl)).tolist(),
        "sequence_length": metadata["sequence_length"],

        # training periods.
        "train_indices": metadata["train_indices"],
        "train_starts": metadata["train_starts"],
        "train_ends": metadata["train_ends"],
        "train_size": metadata["train_size"],
        "train_scenarios": metadata["train_scenarios"],
        "train_end_scenarios": metadata["train_end_scenarios"],

        # validation periods.
        "val_indices": metadata["val_indices"],
        "val_starts": metadata["val_starts"],
        "val_ends": metadata["val_ends"],
        "val_size": metadata["val_size"],
        "val_scenarios": metadata["val_scenarios"],
        "val_end_scenarios": metadata["val_end_scenarios"],

        # testing periods.
        "test_indices": metadata["test_indices"],
        "test_starts": metadata["test_starts"],
        "test_ends": metadata["test_ends"],
        "test_size": metadata["test_size"],
        "test_scenarios": metadata["test_scenarios"],
        "test_end_scenarios": metadata["test_end_scenarios"],

        # trade universe (plain Python lists for JSON serialization)
        "elementary_idx": list(range(n_elementary)),
        "elementary_ids": list(elementary_pnl.columns),

        "target_idx": list(range(n_elementary, n_elementary + n_targets)),
        "target_ids": list(target_pnl.columns),

        # inverse transforms
        "elementary_pnl_transformer": metadata["elementary_pnl_transformer"],
        "target_pnl_transformer": metadata["target_pnl_transformer"],
        "selected_trades": metadata["selected_trades"],
        "removed_trades": metadata["removed_trades"],
    }


def dimension_reduction(pnl_df: pd.DataFrame, config: HybridGnnRnnDataConfig, seed: int = 42) -> Dict[str, Any]:
    """
    Reduce the dimensionality of trade universe to a highly informative and relevant subset.

    :param pnl_df: dataframe of pnl time-series where index is scenarios and columns are trades.
    :param config: hybrid gnn rnn data pipeline configuration.
    :param seed:  random state seed for reproducibility.
    :return:
    """
    # define list of selected trades.
    selected_trades = list()

    if config.dimensionality.reduction_mode.lower() == "none":
        selected_trades = list(pnl_df.columns)
    else:
        # extract underlying and product type.
        underlying = sorted(list(set(x.split('|')[0] for x in pnl_df.columns.to_list())))
        prod_type = sorted(list(set(x.split('|')[1] for x in pnl_df.columns.to_list())))

        # loop through each underlying + product combination in trade universe.
        for und, prod in product(underlying, prod_type):
            # filter trades for underlying and product type.
            filt_pnl_df = pnl_df[
                [k for k in pnl_df.columns.to_list() if k.split("|")[0] == und and k.split("|")[1] == prod]
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


def encode_trade_attributes(
        config: AttributeEncoderConfig, elementary_attrs: Dict[str, List[np.ndarray]],
        target_attrs: Dict[str, List[np.ndarray]], ttm_key: str = "yrs_to_maturity",
) -> Tuple[TradeAttributeEncoder, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    """
    Encode combined elementary + target trade attributes.

    :param config: attribute encoder configuration.
    :param elementary_attrs: filtered elementary trade attributes.
    :param target_attrs: target trade attributes.
    :param ttm_key: attribute name relating to maturity.
    :return: (encoder, encoded_trades, num_elementary, num_target, elementary_idx, target_idx).
    """
    # initiate trade attribute encoder
    encoder = TradeAttributeEncoder(
        numeric_keys=config.numeric_keys, categorical_keys=config.categorical_keys, ttm_key=ttm_key,
        multi_label_keys=config.multi_label_keys, num_decay_terms=config.num_decay_terms,
    )

    # fit encoder to assess attributes across target and elementary trades.
    encoder.fit(elem_attrs=elementary_attrs, target_attrs=target_attrs)

    # encode target trade attributes.
    target_encoded = encoder.transform(trade_attrs=target_attrs)
    logger.info(
        f"Target trade attribute encoded: {len(target_attrs[ttm_key])} trades, across "
        f"{len(list(target_attrs.keys()))} attributes."
    )

    # encode elementary trade attributes.
    elementary_encoded = encoder.transform(trade_attrs=elementary_attrs)
    logger.info(
        f"Elementary trade attribute encoded: {len(elementary_attrs[ttm_key])} trades, across "
        f"{len(list(elementary_attrs.keys()))} attributes."
    )

    # combine encoded features.
    combined_encoded = _combine_encoded_trades(elementary_encoded, target_encoded)

    encoder_results = {
        "elementary_encoded": elementary_encoded, "target_encoded": target_encoded,
        "combined_encoded": combined_encoded, "combined_features": combined_encoded["combined_features"]
    }
    return encoder, encoder_results


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


def standardise_pnl_history(
        elementary_pnl: pd.DataFrame, target_pnl: pd.DataFrame, config: HybridGnnRnnDataConfig
) -> Dict[str, Any]:
    """
    Standardise pnl history for training and validation.

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
    elementary_transformer = get_transformer(transform_type=config.transform_type)
    target_transformer = get_transformer(transform_type=config.transform_type)

    # ---- 2. fit transformer on training data only ----
    training_idx = pnl_periods["train_indices"]
    elementary_transformer.fit(elementary_pnl_arr[training_idx, :])
    _set_feature_names(elementary_transformer, np.array(elementary_pnl.columns.to_list()))
    target_transformer.fit(target_pnl_arr[training_idx, :])
    _set_feature_names(target_transformer, np.array(target_pnl.columns.to_list()))

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
        "val_end_scenarios": [elementary_pnl.index[i] for i in pnl_periods["val_ends"].tolist()],
        "test_scenarios": [elementary_pnl.index[i] for i in pnl_periods["test_indices"].tolist()],
        "test_end_scenarios": [elementary_pnl.index[i] for i in pnl_periods["test_ends"].tolist()],
        "elementary_pnl_transformer": elementary_transformer,
        "target_pnl_transformer": target_transformer,
    }
    metadata |= pnl_periods
    return {
        "elementary_pnl_scaled": pd.DataFrame(
            elementary_pnl_scaled, columns=elementary_pnl.columns.tolist(), index=elementary_pnl.index.to_list(),
        ),
        "target_pnl_scaled": pd.DataFrame(
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
    val_split = val_split if val_split is not None else 0.0
    test_split = test_split if test_split is not None else 0.0

    # ------ 0.1. input validation ------
    if not isinstance(elementary_pnl, pd.DataFrame) or not isinstance(target_pnl, pd.DataFrame):
        raise TypeError("Elementary & Target PnL must be pandas dataframes.")
    if elementary_pnl.shape[0] != target_pnl.shape[0]:
        raise ValueError("Elementary & Target pnl must have the same number of scenarios.")
    if not elementary_pnl.index.equals(target_pnl.index):
        raise ValueError("Elementary & Target pnl must have the same scenario index.")
    if seq_length < 1:
        raise ValueError("Sequence length must be greater than 0.")
    if not (0.0 <= val_split < 1.0) or not (0.0 <= test_split < 1.0) or (val_split + test_split >= 1.0):
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
        f"[train_split: {1 - val_split - test_split}] Allocated {len(train_idx)} scenarios for training: {train_idx[0]}"
        f"...{train_idx[-1]}"
    )
    logger.info(
        f"[val_split: {val_split}] Allocated {len(val_idx)} scenarios for validation: {val_idx[0]}...{val_idx[-1]}"
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
    gap_positions = np.nonzero(diffs != 1)[0]

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


def _build_pnl_sequences(
        elementary_pnl: np.ndarray, target_pnl: np.ndarray, period_starts: np.ndarray, sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build windowed pnl sequences from start indices from pnl time-series.

    :param elementary_pnl: array of pnl time-series for elementary trades [s, n_e]
    :param target_pnl: array of pnl time-series for target trades [s, n_t]
    :param period_starts: start indices for specified period and sequence.
    :param sequence_length: sequence length of windowing
    :return:
    """
    # ---- input checks & conversions ----
    elementary_arr = np.asarray(elementary_pnl, dtype=np.float32)
    target_arr = np.asarray(target_pnl, dtype=np.float32)
    if elementary_arr.ndim != 2 or target_arr.ndim != 2:
        raise ValueError("Elementary & Target pnl arrays must be 2-dimensional.")

    # ---- build windows ----
    if sequence_length == 1:
        # each window is a single day (s, 1, e)
        windows_view = elementary_arr.reshape(elementary_arr.shape[0], 1, elementary_arr.shape[1])
        elem_seq_arr = windows_view[period_starts].astype(np.float32, copy=False)
        tgt_arr = target_arr[period_starts].astype(np.float32, copy=False)
    else:
        # sliding window view gives shape (s - seq + 1, seq, e)
        n_windows = elementary_arr.shape[0] - sequence_length + 1
        elementary_arr = np.ascontiguousarray(elementary_arr)
        windows_view = np.lib.stride_tricks.sliding_window_view(elementary_arr, window_shape=sequence_length, axis=0)
        windows_view = np.moveaxis(windows_view, 1, 2)
        if np.any(period_starts < 0) or np.any(period_starts >= n_windows):
            bad = period_starts[(period_starts < 0) | (period_starts >= n_windows)]
            raise ValueError(f"Some starts are out of range for sliding window (0, ..{n_windows - 1}) --> {bad}.")
        # select windows and labels.
        elem_seq_arr = windows_view[period_starts].astype(np.float32, copy=False)
        tgt_arr = target_arr[period_starts + sequence_length - 1].astype(np.float32, copy=False)
    return elem_seq_arr, tgt_arr


def _combine_encoded_trades(
        elementary_encoded: Dict[str, np.ndarray], target_encoded: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Combine encoded elementary and target trade attributes into a single dict (elementary first).
    :param elementary_encoded: encoded elementary trade attributes.
    :param target_encoded: encoded target trade attributes.
    :return:
    """
    combined = {}
    for key in elementary_encoded:
        if isinstance(elementary_encoded[key], np.ndarray):
            combined[key] = np.concatenate([elementary_encoded[key], target_encoded[key]], axis=0)

            # check combined features shape.
            if combined[key].ndim == 1:
                expected_shape = elementary_encoded[key].shape[0] + target_encoded[key].shape[0]
                assert combined[key].shape[0] == expected_shape, f"Shape mismatch in combined features for key: {key}"
            else:
                expected_shape_0 = elementary_encoded[key].shape[0] + target_encoded[key].shape[0]
                expected_shape_1 = elementary_encoded[key].shape[1]
                assert combined[key].shape[0] == expected_shape_0 and combined[key].shape[1] == expected_shape_1, \
                    f"Shape mismatch in combined features for key: {key}"
    return combined


def _make_dataloader(
        config: HybridGnnRnnDataConfig, trade_pnl: np.ndarray, target_pnl: np.ndarray, period_starts: np.ndarray,
        static_inputs: Optional[Dict[str, np.ndarray]] = None
) -> Any:
    """
    Build a DataLoader for one period (train / val / test).

    Variable inputs (pnl_history) are per-sample and batched. Static inputs
    (trade_features, adjacency components, indices) are shared across all
    samples and injected into every batch.

    :param config: HybridGnnRnnDataConfig configuration
    :param trade_pnl: array of pnl time-series for elementary trades [s, n_e]
    :param target_pnl: array of pnl time-series for target trades [s, n_t]
    :param period_starts: start indices for specified period and sequence.
    :param static_inputs: dict of static arrays injected into every batch.
    :return: PyTorch DataLoader yielding (input_dict, target_pnl) batches.
    """
    elem_seq, tgt_seq = _build_pnl_sequences(
        elementary_pnl=trade_pnl, target_pnl=target_pnl, period_starts=period_starts, sequence_length=config.seq_length,
    )

    dataloader = build_dataloader(
        variable_inputs={"pnl_history": elem_seq},
        targets=tgt_seq,
        config=config,
        static_inputs=static_inputs,
    )
    return dataloader


def _update_trade_attributes(
        trade_attribs: Dict[str, List[Any]], selected_trades: List[str], id_key: str = "trade_id",
) -> Dict[str, List[Any]]:
    """
    Update trade attributes to reflect the trades which have been removed via dimensionality reduction.

    :param trade_attribs: dictionary of trade attributes.
    :param selected_trades: list of trade ids to keep
    :param id_key: key in dictionary for trade id
    :return:
    """
    if id_key not in trade_attribs:
        raise KeyError(f"{id_key} missing in trade attribute keys={list(trade_attribs.keys())}.")

    # convert the current ID sequence to a plain python list for easy indexing / lookup.
    current_trade_ids: List[str] = list(trade_attribs[id_key])

    # total number of trades currently present.
    total_trades: int = len(current_trade_ids)

    # --- Build mapping from trade id --> row index in the current attributes ---
    id_to_position: Dict[str, int] = {tid: i for i, tid in enumerate(current_trade_ids)}

    # validate that every selected trade actually exists in the source attributes
    missing_ids = [tid for tid in selected_trades if tid not in id_to_position]
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        more = " ..." if len(missing_ids) > 10 else ""
        raise ValueError(f"Selected IDs unavailable in attributes: {preview}{more}")

    # guard against accidental duplicates in the required selection.
    if len(set(selected_trades)) != len(selected_trades):
        dupes = [tid for tid in set(selected_trades) if selected_trades.count(tid) > 1]
        preview = ", ".join(dupes[:10])
        more = " ..." if len(dupes) > 10 else ""
        raise ValueError(f"Selected IDs unavailable in attributes: {preview}{more}")

    # build the row-index array to keep, in exactly the same order as 'selected_trades'
    keep_idx = [id_to_position[tid] for tid in selected_trades]

    # prepare output dict (we don't mutate the input; produce a clean copy)
    out: Dict[str, Any] = {}

    # --- main filtering / reordering logic ---
    for k, v in trade_attribs.items():
        try:
            lgnth = len(v)
        except Exception:
            raise TypeError(f"Key '{k}' is not a 1D per-trade list")
        if lgnth != total_trades:
            raise ValueError(f"Key '{k}' length {lgnth} != total trade length {total_trades}")

        # reorder to keep idx keep list type.
        out[k] = [v[i] for i in keep_idx]

    out[id_key] = list(selected_trades)
    return out


def _update_trade_pnl(
        trade_pnl: pd.DataFrame, selected_trades: List[str],
) -> pd.DataFrame:
    """
    Update trade pnl time-series to reflect the trades which have been removed via dimensionality reduction.

    :param trade_pnl: PnL time-series [scenarios x trades].
    :param selected_trades: trade ids to keep (defines output column order).
    :return: filtered dataframe with columns ordered by selected_trades.
    """
    missing = set(selected_trades) - set(trade_pnl.columns)
    if missing:
        raise ValueError(f"Trades missing from PnL columns: {sorted(missing)[:10]}")
    filtered = trade_pnl.reindex(columns=selected_trades)
    logger.info(f"Filtered PnL: {trade_pnl.shape[1]} -> {filtered.shape[1]} trades")
    return filtered


def _validate_input_data(
        elementary_trades: Any, elementary_pnl: Any, target_trades: Any, target_pnl: Any
) -> None:
    """
    Validate input data for processing.

    :param elementary_trades: dictionary of trade attributes for elementary trades.
    :param elementary_pnl: dataframe containing pnl time-series for elementary trades.
    :param target_trades: dictionary of trade attributes for target trades.
    :param target_pnl: dataframe containing pnl time-series for target trades.
    :return:
    """
    # check that all inputs are provided.
    if any(elem is None for elem in [elementary_trades, elementary_pnl, target_trades, target_pnl]):
        logger.error("All inputs (elementary_trades, elementary_pnl, target_trades, target_pnl) must be provided.")
        raise ValueError("All inputs (elementary_trades, elementary_pnl, target_trades, target_pnl) must be provided.")

    # check that elementary trades and target trades are dictionary of lists.
    if not isinstance(elementary_trades, dict) or not all(isinstance(elementary_trades[t], list ) for t in elementary_trades):
        logger.error("elementary_trades must be a dictionary of lists.")
        raise TypeError("elementary_trades must be a dictionary of lists.")
    if not isinstance(target_trades, dict) or not all(isinstance(target_trades[t], list) for t in target_trades):
        logger.error("target_trades must be a dictionary of lists.")
        raise TypeError("target_trades must be a dictionary of lists.")

    # check that elementary pnl and target pnl are numpy arrays.
    if not isinstance(elementary_pnl, np.ndarray) or not isinstance(target_pnl, np.ndarray):
        logger.error("elementary_pnl & target_pnl must be a numpy array.")
        raise TypeError("elementary_pnl & target_pnl must be a numpy array.")

    # check that pnl arrays have appropriate dimensions.
    if elementary_pnl.size == 0 or target_pnl.size == 0:
        logger.error("elementary_pnl and target_pnl must have the same size.")
        raise ValueError("elementary_pnl and target_pnl must have the same size.")
