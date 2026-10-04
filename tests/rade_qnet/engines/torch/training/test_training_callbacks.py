"""
Tests for the epoch-end callbacks.

Callbacks are where a training run's most expensive silent failures live,
because every one of them has a plausible no-op form. An early stopper
watching a metric that is never produced simply never fires, and the run goes
to its full epoch budget -- which is indistinguishable from a run that needed
every epoch. A checkpoint that snapshots by reference restores the last epoch
rather than the best, and the history still names the best epoch correctly.

So the tests here are mostly about the *difference between doing nothing and
doing the right thing*, which is exactly what is hard to see in a log.

Two deliberate design points get their own tests. A monitor whose metric is
absent raises rather than returning "no improvement", because the second makes
the callback a no-op for the whole run. And a NaN is never an improvement,
because ``nan < best`` and ``best < nan`` are both false, so a diverged run
would otherwise keep whichever epoch happened to be compared first.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.rade_qnet.core.runtime.errors import EngineError
from src.rade_qnet.core.spec.training import (
    CheckpointSpec,
    EarlyStoppingSpec,
    SchedulerSpec,
)
from src.rade_qnet.engines.torch.training.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    EpochContext,
    GradientNorms,
    LearningRateSchedule,
    Monitor,
)


def take_a_step(optimiser: torch.optim.Optimizer) -> None:
    """
    Take one real optimiser step.

    Torch warns if a scheduler steps before the optimiser ever has, because
    that skips the first value of the schedule. The warning is correct, so the
    tests here do what a real loop does rather than suppressing it.

    Parameters
    ----------
    optimiser
        The optimiser to step.
    """
    for group in optimiser.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter)
    optimiser.step()


def context(epoch: int, *, train_loss: float = 1.0, val_loss: float | None = None):
    """
    Build an epoch context.

    Parameters
    ----------
    epoch
        One-based epoch number.
    train_loss
        Training loss for the epoch.
    val_loss
        Validation loss, or ``None`` for a run without a validation split.

    Returns
    -------
    EpochContext
        The context a callback would be handed.
    """
    return EpochContext(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
    )


class TestMonitor:
    """Whether a watched metric improved, and by enough."""

    def test_the_first_finite_value_is_always_an_improvement(self):
        """
        Otherwise there is no baseline and nothing ever improves.

        A monitor initialised to zero in ``min`` mode would reject every real
        loss, so patience would expire on the first epoch.
        """
        monitor = Monitor(monitor="val_loss")
        assert monitor.update(context(1, val_loss=5.0))

    def test_a_smaller_value_improves_in_min_mode(self):
        """The ordinary case for a loss."""
        monitor = Monitor(monitor="val_loss")
        monitor.update(context(1, val_loss=5.0))
        assert monitor.update(context(2, val_loss=4.0))

    def test_a_larger_value_improves_in_max_mode(self):
        """
        Which is what a metric like directional accuracy needs.

        A monitor that only ever minimised would early-stop an
        accuracy-monitored run at the epoch where accuracy was worst.
        """
        monitor = Monitor(monitor="val_loss", mode="max")
        monitor.update(context(1, val_loss=0.5))
        assert monitor.update(context(2, val_loss=0.6))

    def test_an_improvement_below_the_minimum_delta_does_not_count(self):
        """
        So a run does not continue for fifty epochs on fourth-decimal noise.

        Without it, patience effectively never expires on a plateau, because
        random variation produces a new "best" every few epochs.
        """
        monitor = Monitor(monitor="val_loss", min_delta=0.01)
        monitor.update(context(1, val_loss=5.0))
        assert not monitor.update(context(2, val_loss=4.999))

    def test_a_nan_is_never_an_improvement(self):
        """
        Because the comparisons do not order it.

        ``nan < best`` is false and ``best < nan`` is also false, so a naive
        comparison keeps whichever epoch happened to be compared first -- and
        in ``max`` mode a NaN can win outright, which makes a diverged run's
        checkpoint the diverged weights.
        """
        monitor = Monitor(monitor="val_loss")
        monitor.update(context(1, val_loss=5.0))
        assert not monitor.update(context(2, val_loss=float("nan")))
        assert monitor.best_epoch == 1

    def test_the_best_epoch_is_recorded(self):
        """So the report can name it without recomputing it from history."""
        monitor = Monitor(monitor="val_loss")
        for epoch, value in enumerate([5.0, 3.0, 4.0], start=1):
            monitor.update(context(epoch, val_loss=value))
        assert monitor.best_epoch == 2
        assert monitor.best_value == 3.0

    def test_an_absent_metric_raises_rather_than_reporting_no_improvement(self):
        """
        The deliberate design point.

        Treating it as "no improvement" turns the callback into a silent no-op
        for the whole run: patience expires on epoch one with the monitor
        never having had a value, or never expires at all. Either way the
        reason is invisible in the log.
        """
        monitor = Monitor(monitor="val_loss")
        with pytest.raises(EngineError, match="no validation split"):
            monitor.update(context(1, val_loss=None))


class TestEarlyStopping:
    """Stopping, and refusing to be a no-op."""

    def test_a_run_stops_after_patience_is_exhausted(self):
        """
        Counted from the last improvement, not from the start.

        Patience counted from epoch zero would stop every run at the same
        epoch regardless of what it was doing.
        """
        stopper = EarlyStopping(
            EarlyStoppingSpec(enabled=True, monitor="val_loss", patience=2),
            has_validation=True,
        )
        for epoch, value in enumerate([5.0, 4.0, 4.5, 4.6], start=1):
            stopper.on_epoch_end(context(epoch, val_loss=value))
        assert stopper.should_stop

    def test_an_improvement_resets_the_patience_counter(self):
        """
        Otherwise a long run with slow progress is cut off mid-descent.

        Four epochs with an improvement at the third, against a patience of
        two: a non-resetting counter would have stopped at epoch three.
        """
        stopper = EarlyStopping(
            EarlyStoppingSpec(enabled=True, monitor="val_loss", patience=2),
            has_validation=True,
        )
        for epoch, value in enumerate([5.0, 5.1, 4.0, 4.1], start=1):
            stopper.on_epoch_end(context(epoch, val_loss=value))
        assert not stopper.should_stop

    def test_monitoring_a_validation_metric_without_one_fails_at_construction(self):
        """
        Before any data is read, which is the point.

        Discovered at the end of the first epoch instead, it has already cost
        an epoch; discovered never, it is a callback that silently does
        nothing for the entire run.
        """
        with pytest.raises(EngineError):
            EarlyStopping(
                EarlyStoppingSpec(enabled=True, monitor="val_loss"),
                has_validation=False,
            )

    def test_a_disabled_stopper_never_stops(self):
        """So the feature can be turned off without removing the callback."""
        stopper = EarlyStopping(EarlyStoppingSpec(enabled=False, patience=1), has_validation=True)
        for epoch in range(1, 6):
            stopper.on_epoch_end(context(epoch, val_loss=10.0))
        assert not stopper.should_stop


class TestBestCheckpoint:
    """Restoring the best epoch, which it must actually do."""

    def test_the_best_weights_are_restored_at_the_end(self):
        """
        The headline behaviour, asserted on the predictions.

        A checkpoint that snapshots by reference restores the *last* epoch,
        and the training history still names the best epoch correctly -- so
        the only visible symptom is a final score that does not match the one
        early stopping selected on.
        """
        model = nn.Linear(2, 1)
        checkpoint = BestCheckpoint(
            CheckpointSpec(enabled=True, monitor="val_loss"),
            model=model,
            has_validation=True,
        )

        checkpoint.on_epoch_end(context(1, val_loss=1.0))
        best = model(torch.ones(1, 2)).item()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        checkpoint.on_epoch_end(context(2, val_loss=2.0))

        assert checkpoint.restore_best()
        assert model(torch.ones(1, 2)).item() == pytest.approx(best)

    def test_a_later_improvement_replaces_the_snapshot(self):
        """
        So the restored weights are the best ones, not the first ones.

        A snapshot taken only once would restore epoch one from every run,
        which looks like a model that cannot learn.
        """
        model = nn.Linear(2, 1)
        checkpoint = BestCheckpoint(
            CheckpointSpec(enabled=True, monitor="val_loss"),
            model=model,
            has_validation=True,
        )

        checkpoint.on_epoch_end(context(1, val_loss=2.0))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        improved = model(torch.ones(1, 2)).item()
        checkpoint.on_epoch_end(context(2, val_loss=1.0))

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(5.0)
        checkpoint.restore_best()

        assert model(torch.ones(1, 2)).item() == pytest.approx(improved)

    def test_restoring_without_a_snapshot_reports_that_it_did_nothing(self):
        """
        Rather than claiming success.

        The caller logs "restored the best epoch", and that line must not
        appear when no epoch was ever recorded.
        """
        checkpoint = BestCheckpoint(
            CheckpointSpec(enabled=False), model=nn.Linear(2, 1), has_validation=True
        )
        assert not checkpoint.restore_best()

    def test_a_disabled_checkpoint_leaves_the_final_weights_alone(self):
        """So turning the feature off means the last epoch is what is kept."""
        model = nn.Linear(2, 1)
        checkpoint = BestCheckpoint(CheckpointSpec(enabled=False), model=model, has_validation=True)
        checkpoint.on_epoch_end(context(1, val_loss=1.0))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        final = model(torch.ones(1, 2)).item()
        checkpoint.restore_best()
        assert model(torch.ones(1, 2)).item() == pytest.approx(final)


class TestLearningRateSchedule:
    """The rate the optimiser is actually using, not the one configured."""

    def test_a_cosine_schedule_decays_the_rate(self):
        """
        Over the configured horizon, so the final rate is near the floor.

        A schedule annealing over an estimated epoch count finishes at the
        wrong rate, which looks like a badly chosen initial rate.
        """
        model = nn.Linear(2, 1)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
        schedule = LearningRateSchedule(
            SchedulerSpec(kind="cosine"),
            optimiser=optimiser,
            epochs=10,
        )
        take_a_step(optimiser)
        start = schedule.current_rate
        for epoch in range(1, 11):
            schedule.on_epoch_end(context(epoch))
        assert schedule.current_rate < start

    def test_the_reported_rate_is_the_optimiser_s_rate(self):
        """
        Read from the parameter group rather than recomputed.

        A recomputed value agrees with the real one only while the formula and
        the scheduler agree, and the history would be reporting a number the
        optimiser never used.
        """
        model = nn.Linear(2, 1)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.05)
        schedule = LearningRateSchedule(SchedulerSpec(kind="cosine"), optimiser=optimiser, epochs=4)
        take_a_step(optimiser)
        schedule.on_epoch_end(context(1))
        assert schedule.current_rate == optimiser.param_groups[0]["lr"]

    def test_a_constant_schedule_holds_the_rate(self):
        """So the configured rate is what the run uses, start to finish."""
        model = nn.Linear(2, 1)
        optimiser = torch.optim.SGD(model.parameters(), lr=0.05)
        schedule = LearningRateSchedule(SchedulerSpec(kind="none"), optimiser=optimiser, epochs=4)
        for epoch in range(1, 5):
            schedule.on_epoch_end(context(epoch))
        assert schedule.current_rate == pytest.approx(0.05)


class TestGradientNorms:
    """The diagnostic that distinguishes a dead network from a converged one."""

    def test_the_summary_reports_the_mean_and_the_maximum(self):
        """
        Both, because they fail differently.

        A rising maximum with a flat mean is a few exploding samples; a
        collapsing mean is a dead network. A loss curve shows neither.
        """
        norms = GradientNorms()
        for value in (1.0, 2.0, 3.0):
            norms.record(value)
        summary = norms.summarise()
        assert summary["grad_norm_mean"] == pytest.approx(2.0)
        assert summary["grad_norm_max"] == pytest.approx(3.0)

    def test_summarising_resets_the_window(self):
        """
        So an epoch's figures describe that epoch.

        A cumulative mean over the whole run flattens exactly the trend the
        diagnostic exists to show. The reset happens in ``summarise`` rather
        than at epoch end because the loop reads the summary *into* the epoch
        context, so epoch end is already too late.
        """
        norms = GradientNorms()
        norms.record(100.0)
        norms.summarise()
        norms.record(1.0)
        assert norms.summarise()["grad_norm_max"] == pytest.approx(1.0)

    def test_the_clipped_fraction_is_reported_when_clipping_is_on(self):
        """
        So a report can show how often the clip threshold actually bound.

        A threshold that binds on every step is not clipping outliers, it is
        rescaling every update -- which caps the effective learning rate at
        something other than the configured one.
        """
        norms = GradientNorms(clip_norm=2.0)
        for value in (1.0, 3.0, 5.0, 1.0):
            norms.record(value)
        assert norms.summarise()["grad_clipped_fraction"] == pytest.approx(0.5)

    def test_an_epoch_with_no_recorded_norms_summarises_to_nothing(self):
        """
        Rather than to zero, which would read as a dead network.

        Zero is a meaningful and alarming value here, so it must not be the
        value that means "not measured".
        """
        assert GradientNorms().summarise() == {}
