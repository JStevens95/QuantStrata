"""
Evaluation pipeline for the Deep Hedging model.

Loads a registered model, builds test data, and runs evaluation through the
generic Evaluator.  The post_eval hook adds deep-hedging-specific diagnostics
(e.g. comparison to Black-Scholes delta hedge).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.rade_ml.pipelines.base import EvalPipeline
from src.rade_ml.pipelines.config import PipelineConfig
from src.rade_ml.data.deep_hedging.config import DeepHedgingDataConfig
from src.rade_ml.data.deep_hedging.build import build_deep_hedging_data

if TYPE_CHECKING:
    from src.rade_ml.core.types import EvaluationResult
    from src.rade_ml.data.result import DataBuildResult

logger = logging.getLogger(__name__)


class DeepHedgingEvalPipeline(EvalPipeline):
    """
    Concrete evaluation pipeline for Deep Hedging.

    Implements the required build_data hook and an optional post_eval hook
    that logs deep-hedging-specific metrics.
    """

    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = DeepHedgingDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = DeepHedgingDataConfig()
        return build_deep_hedging_data(data_config)

    def post_eval(
        self,
        eval_result: "EvaluationResult",
        config: PipelineConfig,
    ) -> None:
        if eval_result.metrics:
            logger.info(
                f"Deep Hedging evaluation | "
                f"loss={eval_result.loss:.6f} | "
                f"residual_p95={eval_result.metrics.get('residual_p95', 'N/A')}"
            )
