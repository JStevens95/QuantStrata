"""
Inference pipeline for the Hybrid GNN-RNN model (PyTorch).

Loads a registered model, prepares inputs via prepare_inputs() which branches
on detected mode (new scenarios vs new trades), and returns PnL predictions.

Input mode is inferred from config.metadata["inference"]; the user does not
set input_mode explicitly.

Design: load inference context once (read-only), then build static_dict and
pnl_history per mode; a single _build_model_input_dict() assembles the 7-key
model input (same names as training, no targets).
"""
from __future__ import annotations

import os
import copy
import json
import joblib
import pandas as pd
import logging

import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

from src.rade_ml_pt.data.result import DataBuildResult
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml_pt.utilities.graph_builder import TradeGraphBuilder
from src.rade_ml_pt.utilities.attribute_encoder import TradeAttributeEncoder

from src.rade_ml_pt.pipelines.config import PipelineConfig
from src.rade_ml_pt.pipelines.base import InferencePipeline
from src.rade_ml_pt.core.types import InferenceResult

# static replication specific imports outside of rade_ml_pt
from src.rade_sr.market_data_manager.shock_manager import ShockHistory
from src.rade_sr.elementary_trades.pnl_calculator import PnlFactory

if TYPE_CHECKING:
    from src.rade_ml_pt.ensemble.model import EnsembleModel

# define module level logging.
logger = logging.getLogger(__name__)

# keys that indicate new scenarios are given.
_SCENARIO_KEYS = frozenset({"new_scenario_dir"})

# keys that indicate new trades are given.
_TRADE_KEYS = frozenset({"new_trade_path"})


@dataclass
class InferenceContext:
    """Metadata required for Inference context."""
    # data configuration context.
    data_config: Optional[Dict[str, Any]] = None

    # graph builder context.
    graph_builder: Optional[TradeGraphBuilder] = None
    graph_results: Optional[Dict[str, Any]] = None

    # attribute encoder context.
    encoder: Optional[TradeAttributeEncoder] = None
    encoder_results: Optional[Dict[str, Any]] = None

    # elementary trade context.
    elementary_pnl: Optional[pd.DataFrame] = None
    elementary_attributes: Optional[Dict[str, Any]] = None
    elementary_scaler: Optional[Any] = None

    # target trade context.
    target_attributes: Optional[Dict[str, Any]] = None
    target_scaler: Optional[Any] = None

    # cluster specific context.
    cluster_info: Optional[Dict[str, Any]] = None
    # ``cluster_assets`` is lazy: only loaded for clusters routed to the
    # affected (re-pricing) path.  ``cluster_rf_keys`` is the
    # lightweight sidecar that lets us decide routing without paying
    # the cluster_assets deserialisation cost.
    cluster_assets: Optional[Dict[str, Any]] = None
    cluster_rf_keys: Optional[Dict[str, List[str]]] = None
    _cluster_assets_path: Optional[Path] = None
    cluster_elem_trades: Optional[Dict[str, Any]] = None

    # trade universe context.
    trade_universe: Optional[Dict[str, Any]] = None

    # metadata context.
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceInputData:
    """Input data requirements to build inference data-loader."""

    # trade features.
    trade_features: Optional[Any] = None

    # sparse adjacency components.
    adjacency_indices: np.ndarray = None
    adjacency_values: np.ndarray = None
    adjacency_dense_shape: List[int] = None

    # elementary trade inputs.
    elementary_attributes: Dict[str, Any] = None
    elementary_indices: List[int] = None
    elementary_ids: List[str] = None
    elementary_pnl: pd.DataFrame = None

    # target trade inputs.
    target_attributes: Dict[str, Any] = None
    target_indices: List[int] = None
    target_ids: List[str] = None

    # inference data-loader
    infer_inputs: Optional[Any] = None


def load_inference_context_from_dir(version_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load all inference artifacts from a member's version directory.

    Standalone equivalent of
    ``HybridGnnRnnInferencePipeline.load_inference_context`` that takes a
    directory path directly, so callers (e.g. ensemble pipelines) can load a
    member's context without instantiating the full inference pipeline.

    Parameters
    ----------
    version_dir : str or Path
        Registry version directory containing saved model artifacts.

    Returns
    -------
    dict
        Raw context dict whose keys match those produced by
        ``HybridGnnRnnInferencePipeline.load_inference_context``.

    Notes
    -----
    Emits one INFO log per file with size (MB) and load wall-time, plus
    a per-cluster summary at the end.  The cluster identifier in the log
    is the ``version_dir.name`` (e.g. ``cluster-A-v1``); when called from
    a parallel context loader these tags let you distinguish interleaved
    log lines per cluster.
    """
    import time

    version_dir = Path(version_dir)
    cluster_tag = version_dir.name
    t_total = time.perf_counter()

    def _load_one(name: str, fname: str) -> Any:
        """Load a single artefact and log its size + wall-time.

        Returns the loaded object, or ``None`` if the file is absent /
        has an unrecognised extension.  Errors are re-raised with the
        file name attached so the caller can identify the offender.
        """
        path = version_dir / fname
        if not path.exists():
            return None
        size_mb = path.stat().st_size / (1024 * 1024)
        t0 = time.perf_counter()
        try:
            if path.suffix in (".joblib", ".pkl"):
                obj = joblib.load(str(path))
            elif path.suffix == ".json":
                with open(path) as f:
                    obj = json.load(f)
            elif path.suffix == ".parquet":
                obj = pd.read_parquet(path=str(path))
            else:
                return None
        except Exception as exc:
            logger.error(
                "  [%s] FAILED %s (%.1f MB): %s",
                cluster_tag, fname, size_mb, exc,
            )
            raise
        dt = time.perf_counter() - t0
        logger.info(
            "  [%s] %-30s %7.1f MB  %6.2fs",
            cluster_tag, fname, size_mb, dt,
        )
        return obj

    context: Dict[str, Any] = {}

    # Encoder + graph builder emit their own INFO logs from
    # ``TradeAttributeEncoder.load`` / ``TradeGraphBuilder.load`` (one
    # line each saying "loaded from <path>").  We add timing on top so
    # the cluster tag and file size are visible alongside those lines.
    t = time.perf_counter()
    enc_path = version_dir / "encoder.pkl"
    context["encoder"] = TradeAttributeEncoder.load(file_path=enc_path)
    logger.info(
        "  [%s] %-30s %7.1f MB  %6.2fs",
        cluster_tag, "encoder.pkl",
        enc_path.stat().st_size / (1024 * 1024),
        time.perf_counter() - t,
    )

    t = time.perf_counter()
    gb_path = version_dir / "graph_builder.pkl"
    context["graph_builder"] = TradeGraphBuilder.load(file_path=gb_path)
    logger.info(
        "  [%s] %-30s %7.1f MB  %6.2fs",
        cluster_tag, "graph_builder.pkl",
        gb_path.stat().st_size / (1024 * 1024),
        time.perf_counter() - t,
    )

    context["data_config"] = HybridGnnRnnDataConfig.from_json(
        path=version_dir / "data_config.json"
    )

    # Order chosen so the largest, most-likely-to-hang files are loaded
    # first — easier to spot in the log when something is wrong.
    #
    # NOTE: ``cluster_assets.joblib`` is deliberately *not* eager-loaded.
    # It's the single biggest file (heavy nested Asset objects) and is
    # only needed for clusters routed to the affected (re-pricing) path
    # at inference time.  We instead read the tiny ``cluster_rf_keys``
    # sidecar (or bootstrap one from cluster_assets on legacy artefacts)
    # so the routing decision in ``intersecting_risk_factors`` doesn't
    # force the slow load.
    for name, fname in [
        ("graph_results",       "graph_results.joblib"),
        ("encoder_results",     "encoder_results.joblib"),
        ("elementary_pnl",      "elementary_pnl.parquet"),
        ("cluster_elem_trades", "cluster_elem_trades.joblib"),
        ("cluster_info",        "cluster_info.joblib"),
        ("elementary_scaler",   "elementary_scaler.pkl"),
        ("target_scaler",       "target_scaler.pkl"),
        ("elementary_attribs",  "elementary_attributes.json"),
        ("target_attribs",      "target_attributes.json"),
        ("trade_universe",      "trade_universe.json"),
    ]:
        obj = _load_one(name, fname)
        if obj is not None:
            context[name] = obj

    # ------------------------------------------------------------------
    # Lazy cluster_assets: load the tiny RF-keys sidecar now (so routing
    # is instant), defer the heavy joblib until re-pricing actually
    # needs it.  Two paths:
    #
    #   * Sidecar present (future training, or after first run on legacy
    #     artefacts): just read the JSON.
    #   * Sidecar absent but cluster_assets.joblib present (legacy
    #     artefacts on first encounter): pay the slow load once, extract
    #     the keys, write the sidecar so subsequent runs are fast,
    #     release the loaded portfolio.
    # ------------------------------------------------------------------
    sidecar_path = version_dir / "cluster_rf_keys.json"
    assets_path  = version_dir / "cluster_assets.joblib"
    context["_cluster_assets_path"] = assets_path

    if sidecar_path.exists():
        t = time.perf_counter()
        size_mb = sidecar_path.stat().st_size / (1024 * 1024)
        with open(sidecar_path) as f:
            context["cluster_rf_keys"] = json.load(f)
        logger.info(
            "  [%s] %-30s %7.1f MB  %6.2fs  (cluster_assets deferred)",
            cluster_tag, "cluster_rf_keys.json", size_mb,
            time.perf_counter() - t,
        )
    elif assets_path.exists():
        logger.warning(
            "  [%s] no cluster_rf_keys.json — bootstrapping sidecar from "
            "cluster_assets.joblib (slow, one-time)",
            cluster_tag,
        )
        t = time.perf_counter()
        portfolio = joblib.load(str(assets_path))
        keys: Dict[str, List[str]] = {
            asset_name: list(asset.risk_factor_shocks.keys())
            for asset_name, asset in portfolio.items()
        }
        del portfolio  # release the heavy graph — we only wanted the keys
        with open(sidecar_path, "w") as f:
            json.dump(keys, f, indent=2)
        context["cluster_rf_keys"] = keys
        logger.info(
            "  [%s] %-30s              bootstrapped in %.2fs (sidecar written)",
            cluster_tag, "cluster_rf_keys.json",
            time.perf_counter() - t,
        )

    logger.info(
        "[%s] inference context loaded in %.2fs",
        cluster_tag, time.perf_counter() - t_total,
    )
    return context


class HybridGnnRnnInferencePipeline(InferencePipeline):
    """
    Concrete inference pipeline for Hybrid GNN-RNN.

    prepare_inputs() detects mode from the data provided and branches to
    _prepare_new_scenarios_inputs() or _prepare_new_trade_inputs().
    """

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self._inference_context: Optional[InferenceContext] = None

    def get_result_cls(self) -> type:
        return InferenceResult

    def build_inference_context(self) -> InferenceContext:
        """Build and cache InferenceContext instance (loaded once, reused by post_infer)."""
        if self._inference_context is not None:
            return self._inference_context

        context = self.load_inference_context()
        self._inference_context = InferenceContext(
            data_config=context.get("data_config"), encoder=context.get("encoder"),
            encoder_results=context.get("encoder_results"), graph_builder=context.get("graph_builder"),
            graph_results=context.get("graph_results"), elementary_pnl=context.get("elementary_pnl"),
            elementary_scaler=context.get("elementary_scaler"), elementary_attributes=context.get("elementary_attribs"),
            target_scaler=context.get("target_scaler"), target_attributes=context.get("target_attribs"),
            trade_universe=context.get("trade_universe"), cluster_info=context.get("cluster_info"),
            cluster_assets=context.get("cluster_assets"), cluster_elem_trades=context.get("cluster_elem_trades"),
            cluster_rf_keys=context.get("cluster_rf_keys"),
            _cluster_assets_path=context.get("_cluster_assets_path"),
        )
        return self._inference_context

    @staticmethod
    def calculate_elementary_pnl(
            asset_portfolio: Dict[str, Any], elementary_trades: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Calculate elementary trade pnl for new scenario shocks."""
        # 0. initiate output dictionary.
        output = {}

        # 1. extract keys to calculate pnl.
        elementary_rfs = list(elementary_trades.keys())
        for rf in elementary_rfs:
            # 1.1. extract asset object instance from asset portfolio.
            asset  = asset_portfolio[rf]

            # 1.2. initiate shock history object.
            shock_obj = ShockHistory(asset)

            # 1.3. initiate asset specific variables / functions.
            shock_obj.initiate_asset_shocks()
            shock_obj.asset_functions["populate_shocks"]()

            # 1.4. calculate asset pnl vectors.
            pnl_vectors = PnlFactory.initiate_pnl_calculator(
                asset=asset, elem_trades=elementary_trades[rf], shock_history=shock_obj.asset_statics["shock_history"]
            )

            # 1.5. update output with asset elementary trade pnl.
            output.update({asset.asset_name: pnl_vectors})
        return output

    @staticmethod
    def _detect_input_mode(infer_meta: Dict[str, Any]) -> str:
        """
        Infer input mode from config.metadata["inference"] keys; no user defined mode flag.

        - If any scenario-like key is present and no trade-like key has a non-empty value -> new_scenarios.
        - If any trade-like key is present and no scenario-like key has a non-empty value -> new_trades.
        """
        has_scenario = any(infer_meta.get(k) not in (None, [], {}) for k in _SCENARIO_KEYS if k in infer_meta)
        has_trade = any(infer_meta.get(k) not in (None, [], {}) for k in _TRADE_KEYS if k in infer_meta)

        if has_scenario and not has_trade:
            return "new_scenarios"
        if has_trade and not has_scenario:
            return "new_trades"
        raise ValueError(f"Could not determine inference input mode: {list(infer_meta.keys())}")

    @staticmethod
    def _inject_unchanged_inputs(context: InferenceContext, mode: str) -> InferenceInputData:
        """Inject inference input data that is unchanged specific to the inference mode detected."""
        input_obj = InferenceInputData()
        if mode == "new_scenarios":
            input_obj.elementary_indices = context.trade_universe["elementary_idx"]
            input_obj.elementary_ids = context.trade_universe["elementary_ids"]
            input_obj.trade_features = context.encoder_results["combined_features"]
            input_obj.adjacency_indices = context.graph_results["sparse_indices"]
            input_obj.adjacency_values = context.graph_results["sparse_values"]
            input_obj.adjacency_dense_shape = context.graph_results["sparse_shape"]
            input_obj.target_indices = context.trade_universe["target_idx"]
            input_obj.target_ids = context.trade_universe["target_ids"]
            return input_obj
        elif mode == "new_trades":
            raise NotImplementedError
        raise ValueError(f"Could not determine inference input mode: {mode}")

    def load_inference_context(self) -> Dict[str, Any]:
        """
        Load all artifacts needed for inference into a single read-only context.

        Thin wrapper around :func:`load_inference_context_from_dir` so the
        standalone single-cluster pipeline and the ensemble pipeline share
        the same loader (including the lazy ``cluster_assets`` + sidecar
        scheme and per-file timing logs).
        """
        version_dir = Path(self._loaded_runner.model_path)
        return load_inference_context_from_dir(version_dir)

    @staticmethod
    def load_new_scenarios(path: str) -> Dict[str, pd.DataFrame]:
        """Load new scenario shocks files from new scenario dir."""
        # create dictionary to hold new risk factor scenario shocks.
        loaded_shocks = {}

        # loop through files in folder to load shocks.
        loaded_shocks.update({
            k.replace(".csv", ""): pd.read_csv(os.path.join(path, k), index_col=0).to_dict(orient="index")
            for k in os.listdir(path)
        })
        return loaded_shocks

    def prepare_inputs(self, config: PipelineConfig) -> Dict[str, Any]:
        """Build mode-ready inputs from the pipeline config."""
        # 0. load inference context from artifacts registry.
        context = self.build_inference_context()

        # extract inference metadata from pipeline configuration.
        infer_meta = config.metadata.get("inference", {})
        mode = self._detect_input_mode(infer_meta)

        # 1. prepare inference inputs for detected mode.
        if mode == "new_scenarios":
            inputs = self._prepare_new_scenarios_inputs(config=config, context=context)
        elif mode == "new_trades":
            inputs = self._prepare_new_trade_inputs(config=config, context=context)
        else:
            raise ValueError(f"Undefined Inference mode, got: {mode}")
        return inputs

    def _prepare_new_scenarios_inputs(
            self, config: PipelineConfig, context: InferenceContext
    ) -> Dict[str, Any]:
        """
        Build mode-ready inputs — incorporating new risk-factor scenario shocks.

        Steps
        -----
        1. Load new risk-factor shock CSVs.
        2. Inject unchanged static inputs (trade_features, adjacency, indices).
        3. Deep-copy asset portfolio with new shocks injected.
        4. Filter elementary trades to match cluster's reduced population.
        5. Calculate elementary PnL for new scenarios.
        6. Concatenate elementary PnL across assets, reorder to training column order.
        7. Standardise elementary PnL using saved scaler.
        8. Store scaled PnL on InferenceInputData.
        9. Build PnL sequences with same windowing as training.
        10. Assemble 7-key model input dict and return base-class contract.
        """
        # 1. load and format new risk-factor shock data.
        new_scenario_dir = config.metadata["inference"].get("new_scenario_dir")
        new_scenario_shocks = self.load_new_scenarios(new_scenario_dir)

        # 2. inject all unchanged inputs into InferenceInputData
        inputs = self._inject_unchanged_inputs(context, mode="new_scenarios")

        # 3a. lazy-load cluster_assets if the context was built with the
        #     deferred loader (cluster_assets.joblib skipped at load time
        #     to avoid the heavy deserialisation cost for clusters that
        #     turn out to be unaffected).
        if context.cluster_assets is None and context._cluster_assets_path is not None:
            logger.info(
                "Loading cluster_assets for standalone re-pricing (%s)",
                Path(context._cluster_assets_path).parent.name,
            )
            context.cluster_assets = joblib.load(str(context._cluster_assets_path))

        # 3b. insert new shocks into respective asset objects.
        new_asset_portfolio = self._update_asset_portfolio(context.cluster_assets, new_scenario_shocks)

        # 4. pre-filter each asset's elementary trades to match cluster's reduced population.
        elem_trades = {
            z: [x for x in context.cluster_elem_trades[z] if x["id"] in inputs.elementary_ids]
            for z in context.cluster_elem_trades.keys()
        }

        # 5. calculate elementary trade pnl for new scenario shocks.
        asset_elementary_pnl = self.calculate_elementary_pnl(
            asset_portfolio=new_asset_portfolio, elementary_trades=elem_trades
        )

        # 6. combine elementary pnl across all cluster assets.
        new_elementary_pnl = pd.concat(asset_elementary_pnl.values(), axis=1)
        new_elementary_pnl = new_elementary_pnl[context.elementary_attributes["trade_id"]]

        # 7. standardise elementary pnl.
        new_elementary_pnl_scaled = self._standardise_pnl(
            pnl_unscaled=new_elementary_pnl, scaler=context.elementary_scaler
        )

        # 8. inject new scaled elementary pnl dataframe into inputs.
        inputs.elementary_pnl = pd.DataFrame(
            new_elementary_pnl_scaled, columns=context.elementary_pnl.columns.tolist(),
            index=new_elementary_pnl.index.tolist(),
        )

        # 9. build PnL sequences — same windowing as training.
        elem_seq = self.build_new_pnl_sequences(
            elementary_pnl=inputs.elementary_pnl,
            seq_length=context.data_config.seq_length,
            n_targets=len(inputs.target_indices),
        )

        # 10. assemble 7-key model input dict and base-class contract.
        return self.build_model_inputs(
            elem_seq=elem_seq,
            inputs=inputs,
            seq_length=context.data_config.seq_length,
        )

    def post_infer(self, result: InferenceResult, config: PipelineConfig) -> None:
        """
        Inverse-scale predictions back to original PnL units.

        The model outputs predictions in the scaled space that training used.
        Applying the target scaler's inverse_transform converts them back to
        real PnL values.
        """
        context = self.build_inference_context()
        target_scaler = context.target_scaler
        if target_scaler is None:
            logger.warning("No target_scaler found — predictions remain in scaled space.")
            return

        raw = result.predictions
        n_scaler_features = len(target_scaler.feature_names_in_)

        if raw.shape[1] == n_scaler_features:
            result.predictions = target_scaler.inverse_transform(raw)
        elif raw.shape[1] < n_scaler_features:
            padded = np.zeros((raw.shape[0], n_scaler_features), dtype=np.float32)
            padded[:, :raw.shape[1]] = raw
            unscaled = target_scaler.inverse_transform(padded)
            result.predictions = unscaled[:, :raw.shape[1]]
        else:
            logger.warning(
                f"Prediction width ({raw.shape[1]}) > scaler features ({n_scaler_features}). "
                f"Applying inverse_transform to first {n_scaler_features} columns only."
            )
            head = target_scaler.inverse_transform(raw[:, :n_scaler_features])
            result.predictions = np.concatenate([head, raw[:, n_scaler_features:]], axis=1)

        logger.info(
            f"Inverse-scaled {raw.shape[0]} predictions "
            f"({raw.shape[1]} targets) to original PnL units."
        )

    def post_infer_plots(self, result: "InferenceResult", data_result: "DataBuildResult"):
        """Run HybridGnnRnn specific plots for inference."""

    @staticmethod
    def build_new_pnl_sequences(
            elementary_pnl: pd.DataFrame, seq_length: int, n_targets: int,
    ) -> np.ndarray:
        """
        Build windowed elementary PnL sequences for inference, using the same
        windowing logic as training.

        :param elementary_pnl: scaled elementary PnL [n_scenarios, n_elementary].
        :param seq_length: sequence length (must match training config).
        :param n_targets: number of target trades (for placeholder shape).
        :return: elementary sequences [n_windows, seq_length, n_elementary].
        """
        from src.rade_ml_pt.data.hybrid_gnn_rnn.build import (
            _build_pnl_sequences, window_starts_from_days,
        )

        n_scenarios = elementary_pnl.shape[0]
        inference_starts, _ = window_starts_from_days(
            scenario_idx=np.arange(n_scenarios), sequence_length=seq_length,
        )
        if inference_starts.size == 0:
            raise ValueError(
                f"No valid inference windows: {n_scenarios} scenarios with seq_length={seq_length}. "
                f"Need at least {seq_length} contiguous scenarios."
            )

        target_placeholder = np.zeros((n_scenarios, n_targets), dtype=np.float32)
        elem_seq, _ = _build_pnl_sequences(
            elementary_pnl=elementary_pnl.to_numpy(),
            target_pnl=target_placeholder,
            period_starts=inference_starts,
            sequence_length=seq_length,
        )
        logger.info(
            f"Built {elem_seq.shape[0]} inference sequences "
            f"(seq_length={seq_length}, n_elementary={elem_seq.shape[2]})"
        )
        return elem_seq

    @staticmethod
    def build_model_inputs(
            elem_seq: np.ndarray, inputs: InferenceInputData, seq_length: int,
    ) -> Dict[str, Any]:
        """
        Assemble the 7-key model input dict and wrap in the base-class contract.

        :param elem_seq: windowed elementary PnL [n_windows, seq_length, n_elementary].
        :param inputs: populated InferenceInputData with static inputs.
        :param seq_length: sequence length used for windowing.
        :return: dict with "inputs", "sample_ids", and "metadata" keys.
        """
        model_inputs = {
            "pnl_history": elem_seq,
            "trade_features": inputs.trade_features,
            "adjacency_indices": inputs.adjacency_indices,
            "adjacency_values": inputs.adjacency_values,
            "adjacency_dense_shape": np.array(inputs.adjacency_dense_shape, dtype=np.int64),
            "elementary_indices": np.array(inputs.elementary_indices, dtype=np.int64),
            "target_indices": np.array(inputs.target_indices, dtype=np.int64),
        }
        return {
            "inputs": model_inputs,
            "sample_ids": inputs.target_ids,
            "metadata": {
                "mode": "new_scenarios",
                "n_scenarios": elem_seq.shape[0],
                "seq_length": seq_length,
                "elementary_ids": inputs.elementary_ids,
                "target_ids": inputs.target_ids,
            },
        }

    @staticmethod
    def _standardise_pnl(pnl_unscaled: pd.DataFrame, scaler: Any) -> np.ndarray:
        """Transform elementary pnl into scaled space, consistent with training."""
        # extract index of feature names.
        feat_index = pd.Index(scaler.feature_names_in_).get_indexer(pnl_unscaled.columns.tolist()).tolist()

        # define padded shape.
        padded = np.zeros((pnl_unscaled.shape[0], len(scaler.feature_names_in_)))
        padded[:, feat_index] = pnl_unscaled.to_numpy()
        pnl_scaled = scaler.transform(padded)
        return pnl_scaled[:, feat_index]

    @staticmethod
    def _update_asset_portfolio(
            asset_portfolio: Dict[str, Any], new_scenario_shocks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update asset portfolio object with new shock data.

        After injection, validates that all risk factors across all assets have
        the same number of scenarios. Mixed counts (e.g. some RF with 100 new
        scenarios, others with the original 1000) would cause PnL calculation
        to fail or produce incorrect results.
        """
        # 0. define new asset portfolio via deep copy.
        new_asset_portfolio = copy.deepcopy(asset_portfolio)

        # 1. loop through each asset in the portfolio.
        for k in new_asset_portfolio.keys():
            # 1.1. extract list of risk factors for this asset.
            asset_rfs = list(new_asset_portfolio[k].risk_factor_shocks.keys())

            # 1.2. check if new shock rf are in asset, if so update with new shocks.
            common_rfs = [rf for rf in asset_rfs if rf in new_scenario_shocks]
            if common_rfs:
                for rf in common_rfs:
                    # 1.3. replace old risk factor shocks with new rf shocks.
                    new_asset_portfolio[k].risk_factor_shocks[rf] = new_scenario_shocks[rf]
                    logger.info(f"Injected new scenario shocks into asset portfolio for risk-factor: {rf}")

        # 2. validate consistent scenario counts across all risk factors.
        scenario_counts: Dict[str, int] = {}
        for asset_name, asset in new_asset_portfolio.items():
            for rf_name, rf_shocks in asset.risk_factor_shocks.items():
                scenario_counts[f"{asset_name}/{rf_name}"] = len(rf_shocks)

        unique_counts = set(scenario_counts.values())
        if len(unique_counts) > 1:
            examples = {n: [] for n in unique_counts}
            for rf_key, count in scenario_counts.items():
                if len(examples[count]) < 3:
                    examples[count].append(rf_key)
            detail = "; ".join(
                f"{n} scenarios: [{', '.join(rfs)}]" for n, rfs in sorted(examples.items())
            )
            raise ValueError(
                f"Inconsistent scenario counts across risk factors after shock injection. "
                f"All risk factors must have the same number of scenarios. Found: {detail}"
            )

        return new_asset_portfolio

    # ==================================================================
    # Cluster-level classification (per-cluster, stateless)
    #
    # Pure functions over an InferenceContext.  Used by the ensemble
    # pipeline's validate_scenarios() to decide which cluster takes
    # the affected (re-pricing) vs unaffected (cheap-lookup) path,
    # and by a standalone cluster pipeline that wants the same checks.
    # ==================================================================

    @staticmethod
    def intersecting_risk_factors(
        ctx: InferenceContext,
        shock_risk_factors: Iterable[str],
    ) -> List[str]:
        """Sorted intersection of cluster RFs and shocked RFs.

        Reads exclusively from ``ctx.cluster_rf_keys`` — the lightweight
        sidecar populated at load time by
        :func:`load_inference_context_from_dir` (from either a
        pre-built ``cluster_rf_keys.json`` or a one-time bootstrap of
        ``cluster_assets.joblib`` for legacy artefacts).

        The heavier ``ctx.cluster_assets`` is deliberately *not*
        consulted here so the routing decision in
        ``validate_scenarios`` does not force the slow joblib load.

        Returns
        -------
        List[str]
            Sorted intersection — empty list iff the cluster is
            unaffected by the loaded shocks (or has no RF metadata
            available).  Callers typically take
            ``bool(intersecting_risk_factors(...))`` for the
            ``is_affected`` predicate.
        """
        if ctx.cluster_rf_keys is None:
            return []
        cluster_rfs: set[str] = set()
        for rf_list in ctx.cluster_rf_keys.values():
            cluster_rfs.update(rf_list)
        return sorted(cluster_rfs & set(shock_risk_factors))

    @staticmethod
    def missing_scenario_labels(
        ctx: InferenceContext,
        scenario_labels: List[str],
    ) -> List[str]:
        """Scenario labels missing from this cluster's historical PnL.

        Cheap-path eligibility check: when the cluster is unaffected,
        inference reads ``ctx.elementary_pnl.loc[scenario_labels]``,
        which requires every requested label to be present in the
        parquet's index.  Returns the labels that aren't — an empty
        list means the cluster is cheap-path eligible.

        Returns
        -------
        List[str]
            Missing labels (in input order).  Returns *all* requested
            labels when ``ctx.elementary_pnl`` is absent or has a
            numeric (``RangeIndex``) — those can't satisfy label
            lookups even if they have the right row count.
        """
        if ctx.elementary_pnl is None:
            return list(scenario_labels)
        if isinstance(ctx.elementary_pnl.index, pd.RangeIndex):
            return list(scenario_labels)
        existing = set(ctx.elementary_pnl.index)
        return [lab for lab in scenario_labels if lab not in existing]

    # ==================================================================
    # Cluster-level input builders (per-cluster, mode-specific)
    #
    # Called once per cluster by the ensemble pipeline, or once by a
    # standalone cluster pipeline.  Stateless — no `self` access, no
    # ensemble-level state assumed.  Everything they need is on the
    # supplied `ctx` (InferenceContext) plus mode-specific args.
    # ==================================================================

    @staticmethod
    def build_new_scenario_inputs(
        ctx: InferenceContext,
        new_scenario_shocks: Dict[str, Dict[Any, Any]],
        scenario_labels: List[str],
        is_affected: bool,
    ) -> Dict[str, Any]:
        """Build the 7-key model input dict for one cluster, new_scenarios mode.

        Dispatches on ``is_affected``:

          * **True**  → :meth:`_build_affected_inputs` — re-prices the
            cluster's elementary PnL under the new shocks (expensive).
          * **False** → :meth:`_build_unaffected_inputs` — looks up
            the cluster's historical elementary PnL by scenario label
            (cheap: no re-pricing, no re-scaling).

        The caller (ensemble's ``_build_new_scenarios_inputs``, or a
        standalone cluster pipeline) decides ``is_affected`` — typically
        via :meth:`is_cluster_affected`.

        Parameters
        ----------
        ctx : InferenceContext
            Per-cluster loaded inference context (assets, trades,
            scaler, elementary_pnl, data_config).
        new_scenario_shocks : Dict[str, Dict[Any, Any]]
            ``{rf_name: {scenario_label: shock_value}}`` for every
            shocked risk factor.  Used only by the affected path.
        scenario_labels : List[str]
            Canonical ordered list of scenario labels.  Used by the
            unaffected path to look up the historical PnL slice.
        is_affected : bool
            Whether this cluster's risk factors intersect the
            shocked-RF set.

        Returns
        -------
        Dict[str, Any]
            ``{"inputs", "sample_ids", "metadata"}`` from
            :meth:`build_model_inputs`.  The caller forwards
            ``result["inputs"]`` to the ensemble forward pass.
        """
        if is_affected:
            return HybridGnnRnnInferencePipeline._build_affected_inputs(
                ctx, new_scenario_shocks,
            )
        return HybridGnnRnnInferencePipeline._build_unaffected_inputs(
            ctx, scenario_labels,
        )

    @staticmethod
    def _build_affected_inputs(
        ctx: InferenceContext,
        new_scenario_shocks: Dict[str, Dict[Any, Any]],
    ) -> Dict[str, Any]:
        """Re-pricing path for clusters whose RFs are shocked.

        Eight composing steps:

          1. ``_inject_unchanged_inputs`` — load static inputs
             (trade features, adjacency, indices).
          2. Lazy-load ``ctx.cluster_assets`` from
             ``ctx._cluster_assets_path`` if not already in memory
             (deferred from ``load()`` to avoid paying the heavy joblib
             cost for clusters that turn out to be unaffected).
          3. ``_update_asset_portfolio`` — deep-copy + inject shocks.
          4. Filter elementary trades to the cluster's reduced
             population (training-time selection).
          5. ``calculate_elementary_pnl`` — static-replication kernels.
             This is the expensive step.
          6. ``pd.concat`` + column reorder to training schema.
          7. ``_standardise_pnl`` — fit-time scaler applied with
             padding for the reduced-population case.
          8. ``build_new_pnl_sequences`` + ``build_model_inputs``.

        After ``_update_asset_portfolio`` (which deep-copies and returns
        a fresh portfolio) we release ``ctx.cluster_assets`` so the
        streaming inference loop only ever holds one cluster's heavy
        asset graph resident at a time.
        """
        inputs = HybridGnnRnnInferencePipeline._inject_unchanged_inputs(
            ctx, mode="new_scenarios",
        )

        # Lazy load — pays the slow joblib cost ONLY for affected
        # clusters, ONLY when we reach the re-pricing path.
        if ctx.cluster_assets is None:
            if ctx._cluster_assets_path is None:
                raise RuntimeError(
                    "cluster_assets required for re-pricing but no "
                    "_cluster_assets_path on InferenceContext — was the "
                    "context built via load_inference_context_from_dir?"
                )
            logger.info(
                "Loading cluster_assets for affected cluster (%s)",
                Path(ctx._cluster_assets_path).parent.name,
            )
            ctx.cluster_assets = joblib.load(str(ctx._cluster_assets_path))

        new_asset_portfolio = HybridGnnRnnInferencePipeline._update_asset_portfolio(
            ctx.cluster_assets, new_scenario_shocks,
        )

        # Free the original cluster_assets — _update_asset_portfolio has
        # already produced an independent deep-copy in new_asset_portfolio,
        # so the next cluster in the streaming loop can start without
        # this one's ~100-300 MB still resident.
        ctx.cluster_assets = None

        # Filter elementary trades to the cluster's reduced population.
        elem_trades = {
            z: [
                x for x in ctx.cluster_elem_trades[z]
                if x["id"] in inputs.elementary_ids
            ]
            for z in ctx.cluster_elem_trades.keys()
        }

        asset_elementary_pnl = HybridGnnRnnInferencePipeline.calculate_elementary_pnl(
            asset_portfolio=new_asset_portfolio,
            elementary_trades=elem_trades,
        )

        new_pnl = pd.concat(asset_elementary_pnl.values(), axis=1)
        new_pnl = new_pnl[ctx.elementary_attributes["trade_id"]]

        new_pnl_scaled = HybridGnnRnnInferencePipeline._standardise_pnl(
            pnl_unscaled=new_pnl, scaler=ctx.elementary_scaler,
        )

        inputs.elementary_pnl = pd.DataFrame(
            new_pnl_scaled,
            columns=ctx.elementary_pnl.columns.tolist(),
            index=new_pnl.index.tolist(),
        )

        elem_seq = HybridGnnRnnInferencePipeline.build_new_pnl_sequences(
            elementary_pnl=inputs.elementary_pnl,
            seq_length=ctx.data_config.seq_length,
            n_targets=len(inputs.target_indices),
        )

        return HybridGnnRnnInferencePipeline.build_model_inputs(
            elem_seq=elem_seq,
            inputs=inputs,
            seq_length=ctx.data_config.seq_length,
        )

    @staticmethod
    def _build_unaffected_inputs(
        ctx: InferenceContext,
        scenario_labels: List[str],
    ) -> Dict[str, Any]:
        """Historical-lookup path for clusters whose RFs are not shocked.

        Since this cluster's risk factors don't intersect any of the
        loaded scenario shocks, the elementary PnL under the "new"
        scenarios is — by definition — identical to the historical
        values cached on disk during training.  We therefore skip
        re-pricing and re-scaling entirely:

          1. ``_inject_unchanged_inputs`` — same static inputs as
             the affected path.
          2. ``ctx.elementary_pnl.loc[scenario_labels]`` — direct
             lookup (eligibility was verified by
             :meth:`missing_scenario_labels` during ``validate_*``,
             so missing labels can't reach here).
          3. ``build_new_pnl_sequences`` — same windowing.
          4. ``build_model_inputs`` — same assembly.
        """
        inputs = HybridGnnRnnInferencePipeline._inject_unchanged_inputs(
            ctx, mode="new_scenarios",
        )

        inputs.elementary_pnl = ctx.elementary_pnl.loc[scenario_labels, :]

        elem_seq = HybridGnnRnnInferencePipeline.build_new_pnl_sequences(
            elementary_pnl=inputs.elementary_pnl,
            seq_length=ctx.data_config.seq_length,
            n_targets=len(inputs.target_indices),
        )

        return HybridGnnRnnInferencePipeline.build_model_inputs(
            elem_seq=elem_seq,
            inputs=inputs,
            seq_length=ctx.data_config.seq_length,
        )

    # ==================================================================
    # Post-prediction transforms
    #
    # The scaled → original math lives in the neutral utility module
    # ``src.rade_ml_pt.utilities.predictions_transform`` so it can be
    # shared with the eval pipeline *without* the eval side having to
    # import inference (or wrap a stub ``InferenceContext`` around a
    # scaler + attribute dict).  The static method below is kept on
    # the pipeline class as a thin facade so every existing inference
    # call site keeps its ``ctx``-shaped contract.
    # ==================================================================

    @staticmethod
    def transform_predictions(
        cluster_id:      str,
        cluster_preds:   np.ndarray,
        scenario_labels: Optional[List[str]],
        ctx:             InferenceContext,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Per-cluster post-prediction transformation: scaled → original.

        Thin facade over
        :func:`src.rade_ml_pt.utilities.predictions_transform.transform_to_original`.
        Pulls ``target_scaler`` and ``target_attributes`` off the
        :class:`InferenceContext` and forwards everything else verbatim,
        so :meth:`EnsembleInferencePipeline._post_infer_cluster` and the
        rest of the inference stack keep their existing call sites
        unchanged.

        Parameters
        ----------
        cluster_id : str
            Used to tag the per-scenario summary frame.
        cluster_preds : np.ndarray
            Raw model output for the cluster.  Shape
            ``[n_scenarios, n_trades]`` — rows are scenarios, columns
            are trades.  This is the convention used everywhere in
            ``ensemble/aggregation``, ``ensemble/session``, and
            ``pipelines/ensemble/eval``.
        scenario_labels : list of str or None
            Per-row labels (length ``n_scenarios``); used as the row
            index of both wide frames and the ``scenario_label``
            column of the summary frame.  ``None`` ⇒ positional
            ``0..n_scenarios-1`` index.  The ensemble pipeline passes
            the canonical ``ValidationReport.scenario_labels`` here so
            trade-level parquet output is indexed by scenario label.
        ctx : InferenceContext
            Must have ``target_scaler`` and ``target_attributes``
            (containing ``TradeKey`` / ``trade_id`` and
            ``NotionalSign``) populated.

        Returns
        -------
        scaled_wide : pd.DataFrame
            Shape ``[n_scenarios, n_trades]``.  Predictions in the
            model's **scaled** output space.  Index =
            ``scenario_label``; columns = trade IDs in the scaler's
            canonical order.
        original_wide : pd.DataFrame
            Same shape / index / columns as ``scaled_wide`` —
            inverse-scaled AND notional-sign-restored.
        summary_df : pd.DataFrame
            One row per scenario; the per-scenario aggregate stats
            documented in
            :func:`~src.rade_ml_pt.utilities.predictions_transform.build_scenario_summary`.

        Raises
        ------
        ValueError
            If ``cluster_preds`` is not 2-D or ``scenario_labels``
            length doesn't match ``n_scenarios``.
        RuntimeError
            If ``ctx.target_scaler`` is None / has no
            ``feature_names_in_``, or ``ctx.target_attributes`` is
            missing / misaligned.
        """
        # Local import — keeps utilities free of inference-pipeline cycles
        # and avoids importing the utility at module load time for callers
        # that don't transform predictions.
        from src.rade_ml_pt.utilities.predictions_transform import (
            transform_to_original,
        )

        return transform_to_original(
            cluster_id        = cluster_id,
            cluster_preds     = cluster_preds,
            scenario_labels   = scenario_labels,
            scaler            = ctx.target_scaler,
            target_attributes = ctx.target_attributes,
            strict_notional   = True,
        )

    # ==================================================================
    # Memory-bounded forward pass
    #
    # ``EnsembleModel.predict_member`` is intentionally model-agnostic:
    # it takes whatever input the member expects, moves it to the right
    # device, and returns predictions.  It has no way to know that, for
    # the Hybrid GNN-RNN contract, only ``pnl_history`` carries the
    # per-scenario batch axis — and therefore no way to safely chunk
    # the forward pass to bound activation memory.
    #
    # The static helper below sits in this pipeline (where the Hybrid
    # GNN-RNN contract is the source of truth) and provides chunked
    # prediction for callers that need bounded peak memory, without
    # leaking model-specific knowledge into ``EnsembleModel``.
    # ==================================================================

    @staticmethod
    def predict_member_chunked(
        ensemble:       "EnsembleModel",
        cluster_id:     str,
        cluster_inputs: Dict[str, Any],
        batch_size:     int = 128,
    ) -> np.ndarray:
        """Memory-bounded forward pass for a single cluster.

        Splits ``cluster_inputs["pnl_history"]`` along the scenario
        (window) axis into ``batch_size``-sized blocks, runs each
        through :meth:`EnsembleModel.predict_member`, and concatenates
        the outputs.  Static inputs (trade features, adjacency, etc.)
        are passed through to every chunk unchanged — slicing the
        underlying numpy / torch arrays returns a view, so per-chunk
        construction is zero-copy.

        Keeps :class:`EnsembleModel` model-agnostic: all knowledge of
        which input key is batched (``pnl_history``) lives here, next
        to the rest of the Hybrid GNN-RNN contract.

        Parameters
        ----------
        ensemble : EnsembleModel
            Orchestrator that owns the member models.  Used only via
            its model-agnostic :meth:`predict_member` API.
        cluster_id : str
            Member to dispatch to.  Must be a key of
            ``ensemble.members``.
        cluster_inputs : dict
            The 7-key Hybrid GNN-RNN input dict produced by
            :meth:`build_model_inputs`.  Must contain ``"pnl_history"``.
        batch_size : int, default 128
            Scenarios per forward pass.  Forward-pass activation peak
            scales linearly with this value.  Tune down for tight RAM
            budgets, up for higher throughput on devices with spare
            memory.

        Returns
        -------
        np.ndarray
            Predictions of shape ``[n_scenarios, n_targets]`` in
            scaled (model-output) space.  Identical to the result of
            calling ``ensemble.predict_member(cluster_id, cluster_inputs)``
            directly, only with bounded peak memory.

        Raises
        ------
        KeyError
            If ``cluster_inputs`` is missing ``"pnl_history"``.
        """
        if "pnl_history" not in cluster_inputs:
            raise KeyError(
                "predict_member_chunked: cluster_inputs must contain "
                "'pnl_history' (the per-scenario batched tensor)."
            )

        pnl_history = cluster_inputs["pnl_history"]
        n_scenarios = pnl_history.shape[0]

        # Fast path — fits in one chunk, no slicing or concat overhead.
        if n_scenarios <= batch_size:
            return ensemble.predict_member(cluster_id, cluster_inputs)

        # Non-batched inputs are reused as-is across every chunk.  Slicing
        # ``pnl_history`` returns a view in numpy / torch so per-chunk dict
        # construction stays zero-copy.
        static_inputs = {
            k: v for k, v in cluster_inputs.items() if k != "pnl_history"
        }

        chunks: List[np.ndarray] = []
        for start in range(0, n_scenarios, batch_size):
            end          = min(start + batch_size, n_scenarios)
            chunk_inputs = {
                **static_inputs,
                "pnl_history": pnl_history[start:end],
            }
            chunks.append(ensemble.predict_member(cluster_id, chunk_inputs))

        return np.concatenate(chunks, axis=0)
