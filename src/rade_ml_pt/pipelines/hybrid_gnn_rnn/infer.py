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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.rade_ml_pt.data.result import DataBuildResult
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


class HybridGnnRnnInferencePipeline(InferencePipeline):
    """
    Concrete inference pipeline for Hybrid GNN-RNN.

    prepare_inputs() detects mode from the data provided and branches to
    _prepare_new_scenarios_inputs() or _prepare_new_trade_inputs().
    """

    def get_result_cls(self) -> type:
        return InferenceResult

    def build_inference_context(self) -> InferenceContext:
        """Build InferenceContext instance."""
        # load context data.
        context = self.load_inference_context()
        return InferenceContext(
            data_config=context.get("data_config"), encoder=context.get("encoder"),
            encoder_results=context.get("encoder_results"), graph_builder=context.get("graph_builder"),
            graph_results=context.get("graph_results"), elementary_pnl=context.get("elementary_pnl"),
            elementary_scaler=context.get("elementary_scaler"), elementary_attributes=context.get("elementary_attribs"),
            target_scaler=context.get("target_scaler"), target_attributes=context.get("target_attribs"),
            trade_universe=context.get("trade_universe"), cluster_info=context.get("cluster_info"),
            cluster_assets=context.get("cluster_assets"), cluster_elem_trades=context.get("cluster_elem_trades"),
        )

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
        }
        if version_dir is not None:
            for name, fname in [
                ("data_config", "data_config.json"),
                ("graph_results", "graph_results.joblib"),
                ("encoder_results", "encoder_results.joblib"),
                ("elementary_pnl", "elementary_pnl.parquet"),
                ("elementary_attribs", "elementary_attributes.json"),
                ("elementary_scaler", "elementary_scaler.pkl"),
                ("target_attribs", "target_attributes.json"),
                ("trade_universe", "trade_universe.json"),
                ("cluster_info", "cluster_info.joblib"),
                ("cluster_assets", "cluster_assets.joblib"),
                ("cluster_elem_trades", "cluster_elem_trades.joblib"),
            ]:
                p = version_dir / fname
                if p.exists() and str(p).endswith(".joblib") or str(p).endswith(".pkl"):
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
        context = self.load_inference_context()

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
        Build mode-ready inputs - incorporating new risk-factor scenario shocks.

        Steps
        -----

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
            index=context.elementary_pnl.index.tolist()
        )

        # 9. build inference dataloader.

        # 10. construct final inference inputs.
        output = {}
        return output

    def post_infer(self, result: InferenceResult, config: PipelineConfig) -> None:
        """Inference specific analytics after predictions."""

    def post_infer_plots(self, result: "InferenceResult", data_result: "DataBuildResult"):
        """Run HybridGnnRnn specific plots for inference."""

    @staticmethod
    def _standardise_pnl(pnl_unscaled: pd.DataFrame, scaler: Any) -> np.ndarray:
        """Transform elementary pnl into scaled space, consistent with training."""
        # extract index of feature names.
        feat_index = pd.Index(scaler.feature_names_in_).get_indexer(pnl_unscaled.columns.tolist()).tolist()

        # define padded shape.
        padded = np.zeros((pnl_unscaled.shape[0], len(scaler.mean_)))
        padded[:, feat_index] = pnl_unscaled.to_numpy()
        pnl_scaled = scaler.transform(padded)
        return pnl_scaled[:, feat_index]

    @staticmethod
    def _update_asset_portfolio(
            asset_portfolio: Dict[str, Any], new_scenario_shocks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update asset portfolio object with new shock data."""
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
        return new_asset_portfolio
