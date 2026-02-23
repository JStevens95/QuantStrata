"""
Inference pipeline for the Deep Hedging model.

Loads a registered model, runs a new simulation for the desired scenarios,
and returns hedging P&L predictions via InferenceRunner.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

import numpy as np

from src.rade_ml.pipelines.base import InferencePipeline
from src.rade_ml.pipelines.config import PipelineConfig
from src.rade_ml.data.deep_hedging.config import DeepHedgingDataConfig
from src.rade_ml.data.deep_hedging.build import build_deep_hedging_data
from src.rade_ml.core.types import DeepHedgingInferenceResult

if TYPE_CHECKING:
    from src.rade_ml.core.types import InferenceResult

logger = logging.getLogger(__name__)


class DeepHedgingInferencePipeline(InferencePipeline):
    """
    Concrete inference pipeline for Deep Hedging.

    Implements prepare_inputs: simulates new market scenarios and returns
    model-ready tensors for the InferenceRunner.
    """

    def get_result_cls(self) -> type:
        return DeepHedgingInferenceResult

    def prepare_inputs(self, config: PipelineConfig) -> Dict[str, Any]:
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = DeepHedgingDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = DeepHedgingDataConfig()

        result = build_deep_hedging_data(data_config)

        # collect all test data into numpy arrays
        price_paths_list, payoffs_list = [], []
        for batch_x, _ in result.test_ds:
            price_paths_list.append(batch_x["price_paths"].numpy())
            payoffs_list.append(batch_x["payoffs"].numpy())

        inputs = {
            "price_paths": np.concatenate(price_paths_list, axis=0),
            "payoffs": np.concatenate(payoffs_list, axis=0),
        }

        return {
            "inputs": inputs,
            "metadata": {
                "market_model": data_config.market.model,
                "strike": data_config.option.strike,
                "maturity": data_config.option.maturity_years,
            },
        }

    def post_infer(
        self,
        result: "InferenceResult",
        config: PipelineConfig,
    ) -> None:
        if result.predictions is not None:
            pnl = result.predictions
            logger.info(
                f"Deep Hedging inference | scenarios={result.scenario_count} | "
                f"mean_pnl={np.mean(pnl):.4f} | std_pnl={np.std(pnl):.4f}"
            )
