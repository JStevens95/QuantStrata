"""
Evaluation pipeline for the Hybrid GNN-RNN model.

Loads a registered model, builds test data, and runs evaluation through the
generic Evaluator. The post_eval hook adds GNN-RNN-specific diagnostics
(e.g. per-target-trade residual analysis).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.rade_ml.pipelines.base import EvalPipeline
from src.rade_ml.pipelines.config import PipelineConfig
from src.rade_ml.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml.data.hybrid_gnn_rnn.build import build_dataset

if TYPE_CHECKING:
    from src.rade_ml.core.types import EvaluationResult
    from src.rade_ml.data.result import DataBuildResult

logger = logging.getLogger(__name__)


class HybridGnnRnnEvalPipeline(EvalPipeline):
    """
    Concrete evaluation pipeline for Hybrid GNN-RNN.

    Implements the required build_data hook and an optional post_eval hook
    that logs per-target-trade evaluation diagnostics.
    """

    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = HybridGnnRnnDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = HybridGnnRnnDataConfig()

        job = config.metadata.get("job", {})
        return build_dataset(config=data_config, job=job)

    def post_eval(
        self,
        eval_result: "EvaluationResult",
        config: PipelineConfig,
    ) -> None:
        if eval_result.metrics:
            logger.info(
                f"Hybrid GNN-RNN evaluation | "
                f"loss={eval_result.loss:.6f} | "
                f"residual_mae={eval_result.metrics.get('residual_mae', 'N/A')} | "
                f"residual_p95={eval_result.metrics.get('residual_p95', 'N/A')}"
            )

        if eval_result.residuals is not None and eval_result.residuals.ndim == 2:
            per_trade_mae = np.mean(np.abs(eval_result.residuals), axis=0)
            worst_idx = int(np.argmax(per_trade_mae))
            logger.info(
                f"Per-target MAE: mean={np.mean(per_trade_mae):.6f}, "
                f"worst_trade_idx={worst_idx} (mae={per_trade_mae[worst_idx]:.6f})"
            )
