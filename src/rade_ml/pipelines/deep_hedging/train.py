"""
Training pipeline for the Deep Hedging model.

Wires model-specific build_data and build_model hooks into the generic
TrainPipeline orchestration (data -> model -> Trainer.fit -> register -> track).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.rade_ml.pipelines.base import TrainPipeline
from src.rade_ml.pipelines.config import PipelineConfig
from src.rade_ml.data.deep_hedging.config import DeepHedgingDataConfig
from src.rade_ml.data.deep_hedging.build import build_deep_hedging_data
from src.rade_ml.models.deep_hedging.model import DeepHedgingModel
from src.rade_ml.models.deep_hedging.layers.risk_measure import CVaRLoss, EntropicRiskLoss

if TYPE_CHECKING:
    import tensorflow as tf
    from src.rade_ml.data.result import DataBuildResult

logger = logging.getLogger(__name__)


class DeepHedgingTrainPipeline(TrainPipeline):
    """
    Concrete training pipeline for Deep Hedging.

    Implements the two required abstract hooks:
        - build_data:  simulate market paths and build tf.data.Datasets
        - build_model: instantiate DeepHedgingModel and compile with CVaR loss
    """

    def build_data(self, config: PipelineConfig) -> "DataBuildResult":
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = DeepHedgingDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = DeepHedgingDataConfig()
        return build_deep_hedging_data(data_config)

    def build_model(
        self,
        config: PipelineConfig,
        data_result: "DataBuildResult",
    ) -> "tf.keras.Model":
        model_config = config.model_config or {}
        model = DeepHedgingModel(config=model_config)

        risk_cfg = model_config.get("risk_measure", {})
        risk_type = risk_cfg.get("type", "cvar")

        if risk_type == "cvar":
            loss = CVaRLoss(alpha=risk_cfg.get("alpha", 0.95))
        elif risk_type == "entropic":
            loss = EntropicRiskLoss(risk_aversion=risk_cfg.get("lagrange_multiplier", 1.0))
        else:
            raise ValueError(f"Unknown risk measure: {risk_type}")

        import tensorflow as tf
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss,
        )
        logger.info(f"Deep Hedging model compiled with {risk_type} loss")
        return model
