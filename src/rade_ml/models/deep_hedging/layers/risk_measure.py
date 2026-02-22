"""
Differentiable risk measure losses for deep hedging.

Standard supervised losses (MSE, MAE) are not appropriate for hedging because
they penalise upside and downside symmetrically.  In practice, hedgers care
about tail risk -- the worst-case losses.

This module provides two risk measures:
    - CVaRLoss:          Conditional Value-at-Risk (Expected Shortfall) via the
                         Rockafellar-Uryasev dual representation.
    - EntropicRiskLoss:  Exponential risk measure that penalises variance and
                         higher moments asymmetrically.

Both are implemented as Keras Loss subclasses so they integrate directly with
model.compile().
"""
import tensorflow as tf
from typing import Any, Dict


class CVaRLoss(tf.keras.losses.Loss):
    """
    Conditional Value-at-Risk (Expected Shortfall) loss.

    Minimises the average loss in the worst (1-alpha) fraction of outcomes.
    Uses the Rockafellar-Uryasev dual representation for differentiability:

        CVaR_alpha(X) = min_z { z + 1/(1-alpha) * E[max(X - z, 0)] }

    where X represents the hedging P&L (positive = loss from hedger's perspective),
    and z (the VaR level) is optimised jointly with the model parameters.

    Parameters
    ----------
    alpha : float
        Confidence level, typically 0.95 or 0.99.
    """

    def __init__(self, alpha: float = 0.95, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute CVaR loss.

        The model outputs hedging P&L (where negative means the hedger lost money).
        We define loss = -pnl (so positive values are bad), then compute CVaR on that.

        :param y_true: dummy targets (zeros), unused in the risk-measure formulation
        :param y_pred: hedging P&L [batch]
        :return: scalar CVaR loss
        """
        losses = -tf.squeeze(y_pred)

        k = tf.cast(
            tf.math.ceil(tf.cast(tf.shape(losses)[0], tf.float32) * self.alpha),
            tf.int32,
        )
        k = tf.minimum(k, tf.shape(losses)[0] - 1)

        sorted_losses = tf.sort(losses, direction="ASCENDING")
        var_alpha = sorted_losses[k]

        exceedances = tf.nn.relu(losses - var_alpha)
        cvar = var_alpha + tf.reduce_mean(exceedances) / (1.0 - self.alpha + 1e-8)
        return cvar

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["alpha"] = self.alpha
        return config


class EntropicRiskLoss(tf.keras.losses.Loss):
    """
    Entropic (exponential) risk measure loss.

    Defined as:  rho(X) = (1/lambda) * log(E[exp(lambda * X)])

    where X = -pnl (losses) and lambda controls risk aversion.
    As lambda -> 0 the measure converges to E[X] (risk-neutral).
    As lambda -> inf it converges to the essential supremum (worst case).

    Parameters
    ----------
    risk_aversion : float
        The lambda parameter controlling tail sensitivity.
    """

    def __init__(self, risk_aversion: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.risk_aversion = risk_aversion

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute entropic risk measure.

        :param y_true: dummy targets (zeros)
        :param y_pred: hedging P&L [batch]
        :return: scalar entropic risk
        """
        losses = -tf.squeeze(y_pred)

        # log-sum-exp trick for numerical stability
        max_loss = tf.stop_gradient(tf.reduce_max(losses))
        shifted = self.risk_aversion * (losses - max_loss)
        log_expectation = max_loss + tf.math.log(
            tf.reduce_mean(tf.exp(shifted)) + 1e-12
        )
        return log_expectation / (self.risk_aversion + 1e-12)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["risk_aversion"] = self.risk_aversion
        return config
