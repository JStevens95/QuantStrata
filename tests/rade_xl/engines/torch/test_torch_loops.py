"""
Tests for the epoch loop.

The loop's job is bookkeeping, and the tests here are mostly about the
bookkeeping being honest, because a loop that trains correctly while recording
the wrong numbers is worse than one that fails. Every downstream decision --
early stopping, checkpoint selection, the comparison between two runs -- is
made from this history and nothing else.

Two refusals are deliberate and get their own tests. A learner that reports no
``loss`` is rejected rather than defaulted to zero, because a zero loss is the
best possible value and would make epoch one the permanent best epoch. And an
unbounded source is rejected outright: it has no notion of a pass, so an epoch
count over it is arbitrary and silently becomes the denominator of every
averaged metric.

The loop is driven here with a stub learner rather than the real supervised
one, so that a failure in these tests is a failure in the loop. The learner
has its own tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from torch import nn

from src.rade_xl.core.runtime.errors import EngineError
from src.rade_xl.core.spec.training import CheckpointSpec, EarlyStoppingSpec
from src.rade_xl.engines.torch.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    GradientNorms,
)
from src.rade_xl.engines.torch.loops import fit_epochs
from src.rade_xl.testkit.fixtures import SyntheticTensorSource

CPU = torch.device("cpu")


@dataclass
class StubLearner:
    """
    A learner that reports a scripted loss per epoch.

    Parameters
    ----------
    losses
        One loss per epoch. The last value is repeated if the loop runs for
        more epochs than there are entries.
    steps_per_epoch
        Batches in the training source, which is how the stub works out which
        epoch it is in. The loop deliberately tells a learner nothing about
        epochs -- a learner's whole remit is one step -- so a stub that wants
        to script a per-epoch loss has to count.
    extra
        Additional scalars reported alongside the loss, to check they reach
        the history.
    omit_loss
        Report no ``loss`` key at all, which the loop must refuse.
    """

    losses: list[float] = field(default_factory=lambda: [1.0])
    steps_per_epoch: int = 4
    extra: dict[str, float] = field(default_factory=dict)
    omit_loss: bool = False
    train_calls: int = 0
    eval_calls: int = 0
    epoch: int = 0

    def _scalars(self) -> dict[str, float]:
        """Return this epoch's scripted scalars."""
        if self.omit_loss:
            return dict(self.extra)
        index = min(self.epoch, len(self.losses) - 1)
        return {"loss": self.losses[index], **self.extra}

    def train_step(self, model, inputs, target):
        """Count the call, advance the epoch if the pass rolled over, report."""
        del model, inputs, target
        self.train_calls += 1
        # Validation runs after the training pass, so `eval_step` reads the
        # epoch this sets rather than tracking its own.
        self.epoch = (self.train_calls - 1) // self.steps_per_epoch
        return self._scalars()

    def eval_step(self, model, inputs, target):
        """Count the call and return the scripted scalars."""
        del model, inputs, target
        self.eval_calls += 1
        return self._scalars()


@pytest.fixture
def sources():
    """
    Provide a train and validation source.

    Returns
    -------
    dict
        Sources by split name.
    """
    rng = np.random.default_rng(0)

    def source(n_samples):
        """Build one source of a given size."""
        return SyntheticTensorSource(
            features=rng.normal(size=(n_samples, 4)).astype(np.float32),
            targets=rng.normal(size=(n_samples, 1)).astype(np.float32),
            batch_size=16,
        )

    return {"train": source(64), "validation": source(32)}


def run(sources, learner, *, epochs=3, **kwargs):
    """
    Drive the loop with a stub learner.

    Parameters
    ----------
    sources
        Sources by split name.
    learner
        The stub learner.
    epochs
        Epoch budget.
    **kwargs
        Forwarded to :func:`fit_epochs`.

    Returns
    -------
    FitOutcome
        The loop's outcome.
    """
    return fit_epochs(
        nn.Linear(4, 1),
        sources,
        learner=learner,
        signature=sources["train"].signature,
        device=CPU,
        epochs=epochs,
        **kwargs,
    )


class TestTheHistory:
    """What a report, a comparison and a checkpoint are all read from."""

    def test_one_record_per_epoch(self, sources):
        """So the history length is the epoch count, not an approximation."""
        outcome = run(sources, StubLearner(), epochs=4)
        assert len(outcome.history) == 4

    def test_the_epochs_are_numbered_in_order_from_zero(self, sources):
        """
        Zero-based, as the contract declares, and monotonic.

        The numbering is zero-based in the data and one-based only where it is
        shown to a person, so that a stored history indexes like any other
        sequence. Monotonic matters because a report plots the history as a
        time series, and an out-of-order record draws a line going backwards.
        """
        outcome = run(sources, StubLearner(), epochs=3)
        assert [record.epoch for record in outcome.history] == [0, 1, 2]

    def test_the_recorded_loss_is_the_mean_over_the_pass(self, sources):
        """
        Not the last batch's, which is noisy by a factor of the batch count.

        A last-batch loss makes a training curve look far rougher than the run
        actually was, and makes early stopping fire on batch-level noise.
        """
        outcome = run(sources, StubLearner(losses=[2.5]), epochs=1)
        assert outcome.history[0].train_loss == pytest.approx(2.5)

    def test_additional_learner_scalars_reach_the_history(self, sources):
        """
        So a learner can report its own diagnostics without a loop change.

        That is the point of returning a dictionary rather than a float: an
        RL update rule reporting an entropy term needs no new plumbing.
        """
        outcome = run(sources, StubLearner(extra={"penalty": 0.25}), epochs=1)
        assert outcome.history[0].metrics["penalty"] == pytest.approx(0.25)

    def test_validation_scalars_are_prefixed_and_training_ones_are_not(self, sources):
        """
        So the same scalar from the two passes cannot collide in one mapping.

        Training is the unprefixed namespace because it is the one every run
        has; a collision would silently leave whichever pass wrote last.
        """
        outcome = run(sources, StubLearner(extra={"penalty": 0.25}), epochs=1)
        assert {"penalty", "val_penalty"} <= set(outcome.history[0].metrics)

    def test_each_epoch_records_its_own_duration(self, sources):
        """
        Because a run that slows down over time is a leak.

        And a leak is invisible in a total, which is the only other number
        that records duration.
        """
        outcome = run(sources, StubLearner(), epochs=2)
        assert all(record.seconds > 0 for record in outcome.history)

    def test_a_validation_loss_is_recorded_when_there_is_a_split(self, sources):
        """Which is what early stopping and checkpointing monitor."""
        outcome = run(sources, StubLearner(), epochs=1)
        assert outcome.history[0].val_loss is not None

    def test_no_validation_loss_is_recorded_when_there_is_no_split(self, sources):
        """
        ``None`` rather than the training loss, so the two are never confused.

        Copying the training loss into the validation field would make a
        report claim a generalisation estimate that does not exist.
        """
        outcome = run({"train": sources["train"]}, StubLearner(), epochs=1)
        assert outcome.history[0].val_loss is None


class TestPasses:
    """Which source gets walked, how often, and in what mode."""

    def test_the_training_source_is_walked_once_per_epoch(self, sources):
        """
        Four batches of sixteen over sixty-four samples, three epochs.

        Asserted exactly, because a loop that walks the source twice trains
        for twice the configured budget and reports the configured one.
        """
        learner = StubLearner()
        run(sources, learner, epochs=3)
        assert learner.train_calls == 4 * 3

    def test_the_validation_source_is_walked_once_per_epoch(self, sources):
        """Two batches of sixteen over thirty-two samples, three epochs."""
        learner = StubLearner()
        run(sources, learner, epochs=3)
        assert learner.eval_calls == 2 * 3

    def test_the_model_is_left_in_evaluation_mode(self, sources):
        """
        Because the next thing to touch it is scoring, not training.

        A model left in training mode scores with dropout active and batch-norm
        updating its running statistics from the test set -- which is both a
        worse score and a leak, and neither is visible in the output.
        """
        model = nn.Linear(4, 1)
        fit_epochs(
            model,
            sources,
            learner=StubLearner(),
            signature=sources["train"].signature,
            device=CPU,
            epochs=1,
        )
        assert not model.training


class TestRefusals:
    """Where a default would be worse than a failure."""

    def test_a_learner_reporting_no_loss_is_refused(self, sources):
        """
        Rather than defaulting to zero.

        Zero is the best achievable loss, so a default would make epoch one
        the permanent best epoch, early stopping fire immediately, and the
        checkpoint hold the initial random weights.
        """
        with pytest.raises(EngineError, match="loss"):
            run(sources, StubLearner(omit_loss=True), epochs=1)

    def test_a_missing_training_source_is_refused(self, sources):
        """
        Naming the splits that were supplied, since the cause is a typo.

        An empty loop would otherwise produce a complete history of epochs
        that each trained on nothing.
        """
        with pytest.raises(EngineError, match="train"):
            fit_epochs(
                nn.Linear(4, 1),
                {"validation": sources["validation"]},
                learner=StubLearner(),
                signature=sources["validation"].signature,
                device=CPU,
                epochs=1,
            )

    def test_a_diverged_run_is_stopped_at_the_epoch_it_diverged_in(self, sources):
        """
        Because no later epoch can recover from non-finite parameters.

        Once the loss is NaN the weights are too, and every subsequent epoch
        computes NaN gradients over NaN weights. Left to finish its budget,
        the run fails two stages later in scoring with a message about
        non-finite predictions -- which points at the metrics rather than at
        the epoch the divergence happened in.
        """
        with pytest.raises(EngineError, match="no longer finite"):
            run(sources, StubLearner(losses=[1.0, float("nan")]), epochs=5)

    def test_the_message_names_the_usual_causes(self, sources):
        """
        Because the fix is upstream of the loop in every case.

        A non-finite input, a learning rate too high for the objective, or a
        loss dividing by something that reached zero -- none of which is
        findable from "the loss was NaN".
        """
        with pytest.raises(EngineError) as caught:
            run(sources, StubLearner(losses=[float("inf")]), epochs=2)
        assert "learning rate" in str(caught.value)

    def test_an_unbounded_source_is_refused(self, sources):
        """
        Because it has no notion of a pass.

        An epoch count over an unbounded source is arbitrary, and it silently
        becomes the denominator of every averaged metric -- so the reported
        loss would be a mean over a window nobody chose.
        """
        rng = np.random.default_rng(1)
        unbounded = SyntheticTensorSource(
            features=rng.normal(size=(32, 4)).astype(np.float32),
            targets=rng.normal(size=(32, 1)).astype(np.float32),
            unbounded=True,
        )
        with pytest.raises(EngineError, match=r"unbounded|steps_per_epoch"):
            run({"train": unbounded}, StubLearner())


class TestCallbackIntegration:
    """The loop's contract with the callbacks it is handed."""

    def test_early_stopping_shortens_the_run(self, sources):
        """
        And the history records only the epochs that ran.

        A history padded to the full budget would make a report draw a flat
        tail that never happened.
        """
        learner = StubLearner(losses=[1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        outcome = run(
            sources,
            learner,
            epochs=6,
            callbacks=[
                EarlyStopping(
                    EarlyStoppingSpec(enabled=True, monitor="val_loss", patience=1),
                    has_validation=True,
                )
            ],
        )
        assert len(outcome.history) < 6
        assert outcome.stopped_early

    def test_the_best_epoch_comes_from_the_checkpoint(self, sources):
        """
        Taken from the tracker that chose it, not recomputed.

        Recomputing it means the reported best epoch agrees with the restored
        weights only while two pieces of code agree about the monitor and the
        direction.
        """
        model = nn.Linear(4, 1)
        checkpoint = BestCheckpoint(
            CheckpointSpec(enabled=True, monitor="val_loss"),
            model=model,
            has_validation=True,
        )
        outcome = fit_epochs(
            model,
            sources,
            learner=StubLearner(losses=[3.0, 1.0, 2.0]),
            signature=sources["train"].signature,
            device=CPU,
            epochs=3,
            callbacks=[checkpoint],
            checkpoint=checkpoint,
        )
        assert outcome.best_epoch == 1

    def test_gradient_norms_are_folded_into_the_epoch_metrics(self, sources):
        """
        So the diagnostic lands in the same history as the loss.

        Kept separately it would need its own writer, its own file and its own
        alignment to the epoch numbers.
        """
        norms = GradientNorms()
        norms.record(1.5)
        outcome = run(sources, StubLearner(), epochs=1, gradient_norms=norms)
        assert "grad_norm_mean" in outcome.history[0].metrics

    def test_the_learning_rate_is_recorded(self, sources):
        """
        Because a resumed or swept run differs from its sibling mainly here.

        Without it the history cannot explain why two runs of the same config
        produced different curves.
        """
        outcome = run(sources, StubLearner(), epochs=1, learning_rate=0.01)
        assert outcome.history[0].learning_rate == pytest.approx(0.01)


class TestTheOutcome:
    """The summary the pipeline persists and reports from."""

    def test_a_run_that_finishes_its_budget_is_not_marked_early(self, sources):
        """So "stopped early" means it, and a report can say why a run is short."""
        outcome = run(sources, StubLearner(losses=[3.0, 2.0, 1.0]), epochs=3)
        assert not outcome.stopped_early

    def test_the_outcome_names_the_monitored_metric(self, sources):
        """
        Taken from the checkpoint, so it cannot disagree with the one used.

        A reported monitor that differs from the one that selected the best
        epoch would make the report's explanation of the choice wrong.
        """
        model = nn.Linear(4, 1)
        checkpoint = BestCheckpoint(
            CheckpointSpec(enabled=True, monitor="val_loss"),
            model=model,
            has_validation=True,
        )
        outcome = fit_epochs(
            model,
            sources,
            learner=StubLearner(),
            signature=sources["train"].signature,
            device=CPU,
            epochs=2,
            callbacks=[checkpoint],
            checkpoint=checkpoint,
        )
        assert outcome.monitor == "val_loss"

    def test_the_total_duration_is_recorded(self, sources):
        """For a cost estimate, which per-epoch times do not directly give."""
        assert run(sources, StubLearner(), epochs=2).total_seconds > 0
