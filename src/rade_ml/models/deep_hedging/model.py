"""
Deep Hedging Model.

Learns an optimal derivative hedging strategy by minimising a tail-risk measure
(CVaR / entropic risk) of the terminal hedging P&L under real-world frictions
(transaction costs, discrete rebalancing).

The model wraps a StrategyRollout layer that orchestrates per-timestep feature
encoding, recurrent policy decisions, and differentiable P&L accounting.

Reference:
    Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019).
    "Deep Hedging." Quantitative Finance, 19(8), 1271-1291.
"""
from __future__ import annotations

import logging
import tensorflow as tf
from typing import Dict, Any, Union

from src.rade_ml.core.base import BaseModel
from src.rade_ml.validation.base import validate_dict_keys
from src.rade_ml.models.deep_hedging.layers.strategy_layer import StrategyRollout

logger = logging.getLogger(__name__)


class DeepHedgingModel(BaseModel):
    """
    Deep Hedging model for learning optimal derivative hedging strategies.

    Integrates a Gated Residual Network encoder, GRU-based hedging policy, and
    differentiable P&L accounting.  Trained end-to-end by minimising a risk
    measure (CVaR or entropic) of the terminal hedging error.
    """

    def __init__(self, config: Dict[str, Any], name: str = "deep_hedging", **kwargs: Any) -> None:
        """
        Initialise the DeepHedgingModel.

        :param config: model configuration dictionary with keys:
            general, encoder, policy, risk_measure
        :param name: model name
        :param kwargs: passed to BaseModel
        """
        super().__init__(name=name, **kwargs)

        self.model_config = config
        self.general_config = config.get("general", {})
        self.encoder_config = config.get("encoder", {})
        self.policy_config = config.get("policy", {})
        self.risk_measure_config = config.get("risk_measure", {})

        num_instruments = self.general_config.get("num_hedging_instruments", 1)
        tc_rate = self.general_config.get("transaction_cost_rate", 0.001)
        position_limit = self.general_config.get("position_limit", None)

        self.strategy = StrategyRollout(
            encoder_config=self.encoder_config,
            policy_config=self.policy_config,
            num_instruments=num_instruments,
            transaction_cost_rate=tc_rate,
            position_limit=position_limit,
            name=f"{self.name}_strategy",
        )

    def build(self, input_shape: Dict[str, tf.TensorShape]) -> None:
        """
        Build the model graph.

        :param input_shape: dictionary of input tensor shapes:
            - price_paths: [batch, timesteps, num_features]
            - payoffs: [batch]
        """
        logger.info("Building Deep Hedging model layers.")
        validate_dict_keys(input_dict=input_shape, keys=["price_paths", "payoffs"])
        logger.info("Deep Hedging model built successfully.")

    def call(
        self,
        inputs: Dict[str, Union[tf.Tensor, tf.SparseTensor]],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass of the Deep Hedging model.

        Orchestrates:
            1. Input validation
            2. Strategy rollout (encode -> policy -> accumulate P&L per timestep)

        :param inputs: dictionary of inputs:
            - price_paths: [batch, T, num_features] -- feature 0 is spot price
            - payoffs: [batch] -- derivative payoff at maturity
        :param training: whether in training mode
        :return: terminal hedging P&L [batch]
        """
        validate_dict_keys(input_dict=inputs, keys=["price_paths", "payoffs"])

        price_paths = inputs["price_paths"]
        payoffs = inputs["payoffs"]

        terminal_pnl = self.strategy((price_paths, payoffs), training=training)
        return terminal_pnl

    @staticmethod
    def compute_output_shape(input_shape: Dict[str, tf.TensorShape]) -> tf.TensorShape:
        """Compute output shape: [batch]."""
        price_shape = input_shape["price_paths"]
        return tf.TensorShape([price_shape[0]])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config["model_config"] = self.model_config
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "DeepHedgingModel":
        """Instantiate from configuration."""
        return super().from_config(config, **kwargs)
