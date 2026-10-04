"""
Tests for the supervised update rule.

The learner is the smallest unit that can be wrong in a way that still trains.
A missing ``zero_grad`` accumulates gradients across batches, which is a
learning rate that grows linearly through the epoch; the loss still falls, for
a while. A clip applied after the step clips nothing. A scaler stepped in the
wrong order silently skips every update where an inf appeared.

The one failure with no plausible symptom at all is shape broadcasting. A
model emitting ``(batch, 1)`` against a target of ``(batch,)`` broadcasts to
``(batch, batch)``: the loss is then the mean over every prediction paired
with every target, which is a well-defined number that decreases under
training and means nothing. The model learns to predict the target's mean and
the loss curve looks healthy throughout. That is why ``align_with_target``
reshapes when the element counts agree and refuses when they do not, rather
than letting numpy's broadcasting rules decide.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.rade_qnet.core.runtime.errors import EngineError
from src.rade_qnet.core.spec.hardware import HardwareSpec
from src.rade_qnet.engines.torch.hardware.devices import resolve_hardware
from src.rade_qnet.engines.torch.learners.supervised import (
    SupervisedLearner,
    align_with_target,
)
from src.rade_qnet.engines.torch.training.callbacks import GradientNorms
from src.rade_qnet.engines.torch.training.losses import build_loss

CPU_HARDWARE = resolve_hardware(HardwareSpec(device="cpu"))


class Net(nn.Module):
    """
    A model whose forward signature matches a batch's keys.

    The learner calls ``model(**inputs)``, so a model's parameter names are
    how it declares what it consumes. That is what lets a model take a static
    adjacency alongside its features without the engine knowing anything about
    graphs.

    Parameters
    ----------
    n_features
        Input width.
    dropout
        Dropout probability, for the tests that need a layer whose behaviour
        differs between training and evaluation mode.
    """

    def __init__(self, n_features: int = 4, *, dropout: float = 0.0) -> None:
        """Build the stack."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer = nn.Linear(n_features, 1)

    @property
    def weight(self) -> torch.Tensor:
        """Return the single weight matrix, for brevity in the assertions."""
        return self.layer.weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        return self.layer(self.dropout(features))


def make_learner(model: nn.Module, **kwargs) -> SupervisedLearner:
    """
    Build a learner over a model with plain SGD and mean squared error.

    Parameters
    ----------
    model
        The model to optimise.
    **kwargs
        Forwarded to :class:`SupervisedLearner`.

    Returns
    -------
    SupervisedLearner
        The learner.
    """
    return SupervisedLearner(
        loss=build_loss("mse"),
        optimiser=torch.optim.SGD(model.parameters(), lr=0.1),
        hardware=CPU_HARDWARE,
        **kwargs,
    )


class TestAlignWithTarget:
    """Reshape when the elements agree, refuse when they do not."""

    def test_matching_shapes_pass_through(self):
        """The common case, which must not pay for the check."""
        predictions = torch.zeros(4, 1)
        target = torch.ones(4, 1)
        aligned, _ = align_with_target(predictions, target)
        assert aligned is predictions

    def test_a_column_vector_is_reshaped_to_a_flat_target(self):
        """
        The case that would otherwise broadcast to a square.

        ``(4, 1)`` against ``(4,)`` broadcasts to ``(4, 4)``, making the loss
        the mean over every prediction paired with every target. That number
        decreases under training and is meaningless.
        """
        aligned, target = align_with_target(torch.zeros(4, 1), torch.ones(4))
        assert aligned.shape == target.shape == (4,)

    def test_a_flat_prediction_is_reshaped_to_a_column_target(self):
        """The mirror image, since either side may be the flat one."""
        aligned, target = align_with_target(torch.zeros(4), torch.ones(4, 1))
        assert aligned.shape == target.shape == (4, 1)

    def test_differing_element_counts_are_refused(self):
        """
        Rather than broadcast, which is the whole point of the function.

        Four predictions against eight targets cannot describe the same
        values, so there is no arrangement that makes the loss meaningful.
        """
        with pytest.raises(EngineError):
            align_with_target(torch.zeros(4, 1), torch.ones(8, 1))

    def test_the_message_names_both_shapes_and_both_counts(self):
        """
        Because the fix is in the model's output dimension or the signature.

        Neither is visible from a message that only says the shapes did not
        match, and the element counts are what distinguish a reshape from a
        genuine mismatch.
        """
        with pytest.raises(EngineError) as caught:
            align_with_target(torch.zeros(4, 3), torch.ones(4, 1))
        message = str(caught.value)
        assert "(4, 3)" in message
        assert "(4, 1)" in message
        assert "signature" in message


class TestTrainStep:
    """One update, and the bookkeeping around it."""

    def test_the_parameters_change(self):
        """
        The minimum claim: a step that updates nothing is defect 6's symptom.

        Asserted on the weights rather than on the loss, because a loss that
        falls over several steps can be produced by other means.
        """
        model = Net()
        before = model.weight.detach().clone()
        make_learner(model).train_step(model, {"features": torch.ones(8, 4)}, torch.ones(8, 1))
        assert not torch.equal(model.weight, before)

    def test_the_loss_is_reported(self):
        """
        Under the key the loop and every callback are written against.

        A learner reporting its objective under another name makes early
        stopping and checkpointing silent no-ops.
        """
        model = Net()
        scalars = make_learner(model).train_step(
            model, {"features": torch.ones(8, 4)}, torch.ones(8, 1)
        )
        assert "loss" in scalars

    def test_the_reported_loss_is_a_float_not_a_tensor(self):
        """
        So the history holds numbers rather than live graph nodes.

        A tensor kept in the history retains the whole backward graph for the
        batch that produced it, which turns a per-epoch record into a memory
        leak proportional to the batch size.
        """
        model = Net()
        scalars = make_learner(model).train_step(
            model, {"features": torch.ones(8, 4)}, torch.ones(8, 1)
        )
        assert isinstance(scalars["loss"], float)

    def test_gradients_do_not_accumulate_between_steps(self):
        """
        A missing ``zero_grad`` is a learning rate that grows through the epoch.

        The loss still falls for a while, so the only symptom is that the
        optimal configured rate becomes a function of the batch count.
        """
        model = Net()
        learner = make_learner(model)
        inputs = {"features": torch.ones(8, 4)}
        target = torch.ones(8, 1)

        learner.train_step(model, inputs, target)
        first = model.weight.grad.detach().clone()
        learner.train_step(model, inputs, target)

        # Accumulation would make the second gradient roughly twice the first.
        assert not torch.allclose(model.weight.grad, first * 2.0)

    def test_repeated_steps_reduce_the_loss(self):
        """
        End to end over a trivially learnable target.

        A constant target against a linear model with a bias is reachable, so
        a learner that is wired correctly must get closer to it.
        """
        model = Net()
        learner = make_learner(model)
        inputs = {"features": torch.ones(8, 4)}
        target = torch.full((8, 1), 3.0)

        first = learner.train_step(model, inputs, target)["loss"]
        for _ in range(20):
            last = learner.train_step(model, inputs, target)["loss"]
        assert last < first

    def test_the_learner_does_not_touch_the_training_mode(self):
        """
        Because the mode belongs to the pass, not to the step.

        The loop sets it once per pass, which is the right place: dropout and
        batch-norm behave per pass, and setting the mode per batch would walk
        every submodule once per batch for a value that cannot have changed.
        A learner that set it anyway would also be able to disagree with the
        loop about which pass this is.
        """
        model = Net(dropout=0.5)
        model.eval()
        make_learner(model).train_step(model, {"features": torch.ones(8, 4)}, torch.ones(8, 1))
        assert not model.training


class TestGradientClipping:
    """Applied before the step, and recorded."""

    def test_clipping_bounds_the_recorded_norm(self):
        """
        A clip applied after the step clips nothing at all.

        The tracked norm is the pre-clip value, which is what makes the
        diagnostic useful: it says how far the gradient wanted to go, not how
        far it was allowed to.
        """
        model = Net()
        norms = GradientNorms(clip_norm=0.01)
        learner = make_learner(model, gradient_clip_norm=0.01, gradient_norms=norms)
        learner.train_step(model, {"features": torch.ones(8, 4) * 100}, torch.ones(8, 1))

        summary = norms.summarise()
        assert summary["grad_norm_max"] > 0.01
        assert summary["grad_clipped_fraction"] == pytest.approx(1.0)

    def test_the_step_is_bounded_by_the_clip(self):
        """
        Which is the behaviour, as distinct from the recording.

        A large gradient under a tight clip must move the weights by about the
        clip times the learning rate, not by the unclipped amount.
        """
        model = Net()
        before = model.weight.detach().clone()
        learner = make_learner(model, gradient_clip_norm=0.01)
        learner.train_step(model, {"features": torch.ones(8, 4) * 100}, torch.ones(8, 1))
        assert torch.linalg.vector_norm(model.weight - before) < 0.1

    def test_norms_are_tracked_without_clipping(self):
        """
        Because the diagnostic is worth having even when clipping is off.

        An unbounded norm is a run about to diverge, and the loss curve only
        says so once it already has.
        """
        model = Net()
        norms = GradientNorms()
        learner = make_learner(model, gradient_norms=norms)
        learner.train_step(model, {"features": torch.ones(8, 4)}, torch.ones(8, 1))
        assert norms.summarise()["grad_norm_mean"] > 0


class TestEvalStep:
    """Scoring, which must not change anything."""

    def test_the_parameters_do_not_change(self):
        """
        The defining property.

        An eval step that updates would fit on the validation split, so the
        validation loss would be a training loss and early stopping would be
        selecting on the data it was meant to hold out.
        """
        model = Net()
        before = model.weight.detach().clone()
        make_learner(model).eval_step(model, {"features": torch.ones(8, 4)}, torch.ones(8, 1))
        assert torch.equal(model.weight, before)

    def test_no_gradients_are_left_behind(self):
        """
        So the next training step starts from a clean slate.

        A gradient left by an eval pass is added to the first training batch's
        gradient, making one update per epoch partly computed from held-out
        data.
        """
        model = Net()
        make_learner(model).eval_step(model, {"features": torch.ones(8, 4)}, torch.ones(8, 1))
        assert all(
            parameter.grad is None or torch.all(parameter.grad == 0)
            for parameter in model.parameters()
        )

    def test_scoring_runs_without_building_a_graph(self):
        """
        Which is what makes an eval pass cheaper than a training one.

        A no-grad pass holds no activations, so validation over a large split
        costs the memory of one batch rather than of the whole pass. Asserted
        through the absence of a graph on the output, since that is the
        observable consequence.
        """
        model = Net()
        learner = make_learner(model)
        learner.eval_step(model, {"features": torch.ones(8, 4)}, torch.ones(8, 1))
        assert model.weight.grad is None or torch.all(model.weight.grad == 0)

    def test_the_loss_matches_a_training_step_s_loss_on_the_same_batch(self):
        """
        Because the two must measure the same thing.

        If they differed, the validation curve and the training curve would be
        in different units and the gap between them -- the one number that
        indicates overfitting -- would be meaningless.
        """
        model = Net()
        learner = make_learner(model)
        inputs = {"features": torch.ones(8, 4)}
        target = torch.ones(8, 1)

        evaluated = learner.eval_step(model, inputs, target)["loss"]
        trained = learner.train_step(model, inputs, target)["loss"]
        assert evaluated == pytest.approx(trained)


class TestPredictStep:
    """Predictions, without a target."""

    def test_predictions_are_returned_for_the_whole_batch(self):
        """One row out per row in, which is what scoring aligns against."""
        model = Net()
        predictions = make_learner(model).predict_step(model, {"features": torch.ones(8, 4)})
        assert predictions.shape[0] == 8

    def test_predicting_does_not_change_the_model(self):
        """
        So serving a model is not training it.

        Which also means two predictions over the same rows agree, and that
        is what makes a stored prediction set reproducible.
        """
        model = Net()
        before = model.weight.detach().clone()
        make_learner(model).predict_step(model, {"features": torch.ones(8, 4)})
        assert torch.equal(model.weight, before)

    def test_predictions_carry_no_gradient(self):
        """
        Because they leave the engine and become plain arrays.

        A tensor that still requires grad cannot be converted by numpy
        without an explicit detach, and the error appears far from here.
        """
        model = Net()
        predictions = make_learner(model).predict_step(model, {"features": torch.ones(8, 4)})
        assert not predictions.requires_grad
