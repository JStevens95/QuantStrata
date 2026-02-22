"""
Strategy rollout layer for deep hedging.

Orchestrates the temporal hedging loop: at each rebalancing date the layer
encodes market features, queries the policy for hedge ratios, computes
transaction costs, and accumulates the hedging P&L.

The full rollout is differentiable, allowing gradients to flow from the
terminal risk measure back through every hedging decision.

P&L accounting:
    hedging_gain_t = delta_t * (S_{t+1} - S_t)
    cost_t         = tc * |delta_t - delta_{t-1}| * S_t
    terminal_pnl   = sum(hedging_gains) - sum(costs) - payoff
"""
import logging
import tensorflow as tf
from typing import Any, Dict, Optional, Tuple

from src.rade_ml.models.deep_hedging.layers.feature_encoder import GatedResidualNetwork
from src.rade_ml.models.deep_hedging.layers.policy_network import HedgingPolicy

logger = logging.getLogger(__name__)


class StrategyRollout(tf.keras.layers.Layer):
    """
    Differentiable hedging strategy rollout across T rebalancing dates.

    Encapsulates the feature encoder, hedging policy, and P&L accounting
    into a single layer that maps (price_paths, payoffs) -> terminal hedging P&L.
    """

    def __init__(
        self,
        encoder_config: Dict[str, Any],
        policy_config: Dict[str, Any],
        num_instruments: int = 1,
        transaction_cost_rate: float = 0.001,
        position_limit: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.encoder_config = encoder_config
        self.policy_config = policy_config
        self.num_instruments = num_instruments
        self.tc_rate = transaction_cost_rate
        self.position_limit = position_limit

        self.encoder = GatedResidualNetwork(
            layer_config=encoder_config, name=f"{self.name}_encoder"
        )
        self.policy = HedgingPolicy(
            layer_config=policy_config,
            num_instruments=num_instruments,
            name=f"{self.name}_policy",
        )

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Execute the hedging rollout.

        :param inputs: tuple of:
            - price_paths: market features at each rebalancing date [batch, T, num_features].
              Feature 0 must be the spot price used for P&L and cost accounting.
            - payoffs: derivative payoff at maturity [batch]
        :param training: training flag
        :return: terminal hedging P&L [batch] (positive = profit for hedger)
        """
        price_paths, payoffs = inputs

        batch_size = tf.shape(price_paths)[0]
        num_steps = tf.shape(price_paths)[1]

        # extract spot prices (feature index 0) for P&L calculation
        spot_prices = price_paths[:, :, 0]

        # initialise rollout state
        states = self.policy.get_initial_state(batch_size)
        prev_delta = tf.zeros([batch_size, self.num_instruments])
        cumulative_hedge_gain = tf.zeros([batch_size])
        cumulative_cost = tf.zeros([batch_size])

        # Use Python range when time dim is known (graph-mode compatible). tf.range fails in
        # graph mode because AutoGraph cannot iterate over a symbolic tensor.
        T = price_paths.shape[1]
        if T is not None:
            step_indices = range(int(T) - 1)
        else:
            raise ValueError(
                "StrategyRollout requires a known time dimension (price_paths.shape[1]) for "
                "graph-mode compatibility. Use a dataset with fixed num_steps, or ensure "
                "batch() preserves the static shape (e.g. from arrays with shape [N, T, F])."
            )

        for t in step_indices:
            features_t = price_paths[:, t, :]
            encoded_t = self.encoder(features_t, training=training)

            # augment encoding with current position information
            augmented = tf.concat([encoded_t, prev_delta], axis=-1)

            delta_t, states = self.policy.step(augmented, states, training=training)

            if self.position_limit is not None:
                delta_t = tf.clip_by_value(delta_t, -self.position_limit, self.position_limit)

            # P&L from hedging: delta_t * (S_{t+1} - S_t)
            spot_t = spot_prices[:, t]
            spot_next = spot_prices[:, t + 1]
            price_change = spot_next - spot_t

            hedge_gain_t = tf.reduce_sum(delta_t * price_change[:, tf.newaxis], axis=-1)
            cumulative_hedge_gain += hedge_gain_t

            # proportional transaction costs: tc * |delta_change| * S_t
            delta_change = tf.abs(delta_t - prev_delta)
            cost_t = self.tc_rate * tf.reduce_sum(delta_change * spot_t[:, tf.newaxis], axis=-1)
            cumulative_cost += cost_t

            prev_delta = delta_t

        # terminal P&L = hedging gains - costs - payoff
        terminal_pnl = cumulative_hedge_gain - cumulative_cost - payoffs
        return terminal_pnl

    def compute_output_shape(self, input_shape):
        price_shape, _ = input_shape
        return tf.TensorShape([price_shape[0]])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "encoder_config": self.encoder_config,
            "policy_config": self.policy_config,
            "num_instruments": self.num_instruments,
            "transaction_cost_rate": self.tc_rate,
            "position_limit": self.position_limit,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StrategyRollout":
        return cls(**config)
