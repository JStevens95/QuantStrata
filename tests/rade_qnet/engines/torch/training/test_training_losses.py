"""
Tests for the loss registry and the non-standard objectives.

``mse``, ``mae`` and ``huber`` are thin wrappers over torch and are tested
only for the properties the registry promises. The two objectives worth real
tests are the ones a user would otherwise write themselves and get subtly
wrong.

**Asymmetric loss.** Under-predicting a risk figure leaves a position
unhedged; over-predicting it costs some funding. A symmetric loss asserts
those cost the same, which for a risk number is false. The direction of the
asymmetry is the thing to test, because getting it backwards produces a model
that systematically under-hedges while the loss curve looks healthy -- the
loss is still minimised, just for the wrong objective.

**Quantile loss.** Pinball loss fits a conditional quantile. The tests assert
the identity that anchors it: at the median it equals half the mean absolute
error. That is a closed-form relationship, so it catches a factor-of-two error
that no amount of staring at a training curve would reveal.
"""

from __future__ import annotations

import pytest
import torch

from src.rade_qnet.core.runtime.errors import EngineError
from src.rade_qnet.engines.torch.training.losses import (
    asymmetric_loss,
    build_loss,
    quantile_loss,
    register_loss,
    registered_losses,
)


class TestTheRegistry:
    """Name to callable, with the options bound at build time."""

    def test_the_built_in_objectives_are_available(self):
        """So a spec can name one without the user registering anything."""
        assert {"mse", "mae", "huber", "asymmetric", "quantile"} <= set(registered_losses())

    def test_an_unknown_name_is_reported_with_the_alternatives(self):
        """
        Because the fix is almost always a typo, and the list is short.

        A bare ``KeyError: 'rmse'`` does not say that ``mse`` exists.
        """
        with pytest.raises(EngineError) as caught:
            build_loss("rmse")
        assert "mse" in str(caught.value)

    def test_options_are_bound_when_the_loss_is_built(self):
        """
        Not read per batch, so a configured loss is a plain two-argument call.

        The loop should not have to carry the options alongside the callable,
        and the inner loop should not pay for a dictionary lookup per batch.
        """
        loss = build_loss("quantile", options={"quantile": 0.9})
        predictions = torch.zeros(4, 1)
        target = torch.ones(4, 1)
        assert loss(predictions, target).item() == pytest.approx(0.9)

    def test_a_custom_loss_can_be_registered(self):
        """
        Which is how a user plugs in an objective the framework never saw.

        Registered under a name, so it can then be selected from a config file
        like any built-in -- that is the point of the registry over passing a
        callable around.
        """
        register_loss("always_two", lambda predictions, target: torch.tensor(2.0), replace=True)
        assert build_loss("always_two")(torch.zeros(1), torch.zeros(1)).item() == 2.0

    def test_registering_over_an_existing_name_is_refused(self):
        """
        Unless ``replace`` is passed, because a shadowed ``mse`` is invisible.

        Two libraries both registering their own ``mse`` at import time would
        otherwise mean the objective depends on import order.
        """
        with pytest.raises(Exception, match=r"already|exists|replace"):
            register_loss("mse", lambda predictions, target: torch.tensor(0.0))


class TestAsymmetricLoss:
    """The direction of the asymmetry, which is the easy thing to invert."""

    def test_under_prediction_costs_more_than_over_prediction(self):
        """
        The defining property, asserted on equal-magnitude errors.

        Inverting this produces a model that systematically under-hedges, and
        the loss curve looks exactly as healthy either way -- it is still
        being minimised, just for the opposite objective.
        """
        target = torch.ones(4, 1)
        under = asymmetric_loss(torch.zeros(4, 1), target, under_penalty=3.0)
        over = asymmetric_loss(torch.full((4, 1), 2.0), target, under_penalty=3.0)
        assert under > over

    def test_the_penalty_is_the_exact_multiplier(self):
        """
        So a configured penalty of three means three, not roughly three.

        An error of one in each direction, so the ratio is the multiplier with
        nothing else mixed in.
        """
        target = torch.ones(1, 1)
        under = asymmetric_loss(torch.zeros(1, 1), target, under_penalty=3.0)
        over = asymmetric_loss(torch.full((1, 1), 2.0), target, under_penalty=3.0)
        assert (under / over).item() == pytest.approx(3.0)

    def test_a_penalty_of_one_is_the_mean_absolute_error(self):
        """
        The degenerate case, which anchors the scale of the loss.

        It means a user comparing an asymmetric run against an ``mae``
        baseline is comparing numbers in the same units.
        """
        predictions = torch.tensor([[0.0], [3.0], [1.5]])
        target = torch.ones(3, 1)
        assert asymmetric_loss(predictions, target, under_penalty=1.0).item() == (
            pytest.approx(torch.nn.functional.l1_loss(predictions, target).item())
        )

    def test_a_perfect_prediction_costs_nothing(self):
        """
        So the loss has no offset to confuse a comparison between runs.

        A floor that is not zero makes early stopping's minimum delta mean
        something different at every scale, and a reported loss that cannot
        reach zero is impossible to read against another run's.
        """
        assert asymmetric_loss(torch.ones(4, 1), torch.ones(4, 1)).item() == 0.0

    def test_the_loss_is_differentiable(self):
        """
        Otherwise it cannot be trained against at all.

        A loss assembled with a comparison in the wrong place can detach the
        graph, and the symptom is a model that never updates -- the same
        symptom as defect 6, from a different cause.
        """
        predictions = torch.zeros(4, 1, requires_grad=True)
        asymmetric_loss(predictions, torch.ones(4, 1)).backward()
        assert predictions.grad is not None
        assert torch.any(predictions.grad != 0)


class TestQuantileLoss:
    """Pinball loss, anchored on its closed-form relationship to the MAE."""

    def test_the_median_is_half_the_mean_absolute_error(self):
        """
        An exact identity, which catches a factor-of-two error outright.

        No training curve would reveal a loss that is uniformly twice what it
        should be: it still decreases, and the only visible effect is that the
        learning rate wants to be half as large.
        """
        predictions = torch.tensor([[0.0], [3.0], [1.5]])
        target = torch.ones(3, 1)
        expected = 0.5 * torch.nn.functional.l1_loss(predictions, target).item()
        assert quantile_loss(predictions, target, quantile=0.5).item() == (pytest.approx(expected))

    def test_a_high_quantile_penalises_under_prediction(self):
        """
        Which is what makes the fitted quantile sit above the median.

        The asymmetry *is* the mechanism: there is nothing else in the loss
        that could push the prediction upwards.
        """
        target = torch.ones(4, 1)
        under = quantile_loss(torch.zeros(4, 1), target, quantile=0.9)
        over = quantile_loss(torch.full((4, 1), 2.0), target, quantile=0.9)
        assert under > over

    def test_a_low_quantile_penalises_over_prediction(self):
        """The mirror image, so the parameter is not simply a magnitude."""
        target = torch.ones(4, 1)
        under = quantile_loss(torch.zeros(4, 1), target, quantile=0.1)
        over = quantile_loss(torch.full((4, 1), 2.0), target, quantile=0.1)
        assert over > under

    def test_the_penalty_ratio_is_the_quantile_ratio(self):
        """
        At the ninetieth percentile, nine to one.

        Stated exactly, because a quantile loss built from the complement --
        ``1 - q`` where ``q`` belongs -- fits the hundred-minus-q percentile
        and nothing in the output says so.
        """
        target = torch.ones(1, 1)
        under = quantile_loss(torch.zeros(1, 1), target, quantile=0.9)
        over = quantile_loss(torch.full((1, 1), 2.0), target, quantile=0.9)
        assert (under / over).item() == pytest.approx(9.0)

    def test_a_quantile_outside_the_unit_interval_is_refused(self):
        """
        Because the loss ceases to be a loss: one side goes negative.

        A negative contribution means the optimiser can reduce the objective
        without bound by predicting ever more wrongly in that direction.
        """
        with pytest.raises(Exception, match="quantile"):
            quantile_loss(torch.zeros(1, 1), torch.ones(1, 1), quantile=1.5)
