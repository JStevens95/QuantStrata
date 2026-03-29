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
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

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
    pass

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
    cluster_assets: Optional[Dict[str, Any]] = None
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
    """
    version_dir = Path(version_dir)
    context: Dict[str, Any] = {
        "encoder": TradeAttributeEncoder.load(file_path=version_dir / "encoder.pkl"),
        "graph_builder": TradeGraphBuilder.load(file_path=version_dir / "graph_builder.pkl"),
        "data_config": HybridGnnRnnDataConfig.from_json(path=version_dir / "data_config.json"),
    }
    for name, fname in [
        ("graph_results", "graph_results.joblib"),
        ("encoder_results", "encoder_results.joblib"),
        ("elementary_pnl", "elementary_pnl.parquet"),
        ("elementary_attribs", "elementary_attributes.json"),
        ("elementary_scaler", "elementary_scaler.pkl"),
        ("target_attribs", "target_attributes.json"),
        ("target_scaler", "target_scaler.pkl"),
        ("trade_universe", "trade_universe.json"),
        ("cluster_info", "cluster_info.joblib"),
        ("cluster_assets", "cluster_assets.joblib"),
        ("cluster_elem_trades", "cluster_elem_trades.joblib"),
    ]:
        p = version_dir / fname
        if p.exists() and (str(p).endswith(".joblib") or str(p).endswith(".pkl")):
            context[name] = joblib.load(str(p))
        if p.exists() and str(p).endswith(".json"):
            with open(p) as f:
                context[name] = json.load(f)
        if p.exists() and str(p).endswith(".parquet"):
            context[name] = pd.read_parquet(path=str(p))
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

        Does not mutate any loaded objects. Caller uses a context to build static data and pnl history per mode and
        then combine static and variable input data into a single dictionary for inference.
        """
        # create version directory for artifact loading.
        version_dir = Path(self._loaded_runner.model_path)

        # define inference context.
        context: Dict[str, Any] = {
            "encoder": TradeAttributeEncoder.load(file_path=version_dir / "encoder.pkl"),
            "graph_builder": TradeGraphBuilder.load(file_path=version_dir / "graph_builder.pkl"),
            "data_config": HybridGnnRnnDataConfig.from_json(path=version_dir / "data_config.json"),
        }
        if version_dir is not None:
            for name, fname in [
                ("graph_results", "graph_results.joblib"),
                ("encoder_results", "encoder_results.joblib"),
                ("elementary_pnl", "elementary_pnl.parquet"),
                ("elementary_attribs", "elementary_attributes.json"),
                ("elementary_scaler", "elementary_scaler.pkl"),
                ("target_attribs", "target_attributes.json"),
                ("target_scaler", "target_scaler.pkl"),
                ("trade_universe", "trade_universe.json"),
                ("cluster_info", "cluster_info.joblib"),
                ("cluster_assets", "cluster_assets.joblib"),
                ("cluster_elem_trades", "cluster_elem_trades.joblib"),
            ]:
                p = version_dir / fname
                if p.exists() and (str(p).endswith(".joblib") or str(p).endswith(".pkl")):
                    context[name] = joblib.load(str(p))
                if p.exists() and str(p).endswith(".json"):
                    with open(p) as f:
                        context[name] = json.load(f)
                if p.exists() and str(p).endswith(".parquet"):
                    context[name] = pd.read_parquet(path=str(p))
        return context

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

        # 3. insert new shocks into respective asset objects.
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
