# `tests/rade_qnet/engines/torch/training`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 1 | 32 | `ea056fce628eee7f` |
| 2 | `test_training_callbacks.py` | 402 | 15291 | `97c783702a4278fc` |
| 3 | `test_training_checkpoint.py` | 225 | 8428 | `07ea0986a0ceb441` |
| 4 | `test_training_loops.py` | 1003 | 34249 | `eee06920985be093` |
| 5 | `test_training_losses.py` | 209 | 8874 | `ad6de8c53529ae7d` |

---

## 1. `tests/rade_qnet/engines/torch/training/__init__.py`

32 bytes · SHA-256 `ea056fce628eee7f`

```python
"""Mirror of torch.training."""
```

---

## 2. `tests/rade_qnet/engines/torch/training/test_training_callbacks.py`

15291 bytes · SHA-256 `97c783702a4278fc`

```python
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

from src.rade_qnet.core.lifecycle.errors import EngineError
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
```

---

## 3. `tests/rade_qnet/engines/torch/training/test_training_checkpoint.py`

8428 bytes · SHA-256 `07ea0986a0ceb441`

```python
"""
Tests for weight serialisation -- defect 10.

Defect 10 is that checkpoints were loaded with ``weights_only=False``, which
makes ``torch.load`` run the pickle machinery and therefore execute arbitrary
code from the file. A model artifact is a *data* file: it travels between
machines, sits in object storage, and is fetched by a serving process. Giving
it the privileges of a Python script is a straightforward remote code
execution path, and it is enabled by a default rather than by a decision.

Loading with ``weights_only=True`` closes it, and the consequence is that a
checkpoint containing pickled *modules* -- rather than a plain state
dictionary -- can no longer be read. That is the correct trade: such a
checkpoint cannot be loaded safely at all, and the fix is to regenerate it
from the bundle rather than to re-enable the unsafe path.

The second group of tests here is about the snapshot being a *copy*. A state
dictionary is a dictionary of views onto the live parameters, so a snapshot
taken by reference is not a snapshot: the next optimiser step rewrites it in
place, and "restore the best epoch" restores the last epoch instead. Nothing
about that is visible in the training history, which still reports the best
epoch correctly.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.rade_qnet.core.lifecycle.errors import EngineError
from src.rade_qnet.engines.base import ModelHandle
from src.rade_qnet.engines.torch.training.checkpoint import (
    load_weights,
    restore_state_dict,
    save_weights,
    snapshot_state_dict,
)


def make_model(seed: int = 0) -> nn.Module:
    """
    Build a small deterministic model.

    Parameters
    ----------
    seed
        Seed for the initial weights, so two models can be made to differ.

    Returns
    -------
    torch.nn.Module
        The model.
    """
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


class TestSnapshotIsACopy:
    """A snapshot by reference is not a snapshot."""

    def test_a_snapshot_survives_a_parameter_update(self):
        """
        The defect that makes "restore the best epoch" restore the last one.

        ``state_dict`` returns views onto the live tensors, so a snapshot
        taken without detaching and copying is rewritten in place by the next
        optimiser step. The history still names the right best epoch, so
        nothing reports the problem.
        """
        model = make_model()
        snapshot = snapshot_state_dict(model)
        before = snapshot["0.weight"].clone()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)

        assert torch.equal(snapshot["0.weight"], before)

    def test_a_snapshot_is_on_the_cpu(self):
        """
        So a device-resident snapshot does not hold accelerator memory.

        A tuning sweep keeping one snapshot per trial on the GPU runs out of
        memory at a trial count that looks arbitrary.
        """
        model = make_model()
        snapshot = snapshot_state_dict(model)
        assert all(tensor.device.type == "cpu" for tensor in snapshot.values())

    def test_a_snapshot_carries_no_gradients(self):
        """
        It is weights, not a point in an optimisation.

        A snapshot holding its gradient graph keeps the whole forward pass's
        activations alive, which turns one retained snapshot into a memory
        leak proportional to the batch size.
        """
        model = make_model()
        inputs = torch.ones(2, 4)
        model(inputs).sum().backward()
        snapshot = snapshot_state_dict(model)
        assert all(not tensor.requires_grad for tensor in snapshot.values())


class TestRestore:
    """Putting a snapshot back, exactly."""

    def test_a_restored_model_predicts_identically(self):
        """
        Which is the property early stopping depends on.

        Restoring approximately would mean the model that gets saved is not
        the model whose validation score chose it.
        """
        model = make_model()
        inputs = torch.ones(3, 4)
        expected = model(inputs)

        snapshot = snapshot_state_dict(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.5)
        restore_state_dict(model, snapshot)

        assert torch.equal(model(inputs), expected)

    def test_a_mismatched_snapshot_is_refused(self):
        """
        Strictly, so a partial restore cannot happen quietly.

        A non-strict load silently leaves any unmatched parameter at its
        current value, which produces a model that is part best-epoch and part
        last-epoch -- and scores somewhere between the two.
        """
        wide = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
        with pytest.raises(Exception, match=r"size mismatch|shape|Error"):
            restore_state_dict(wide, snapshot_state_dict(make_model()))


class TestFileRoundTrip:
    """Save, reload into a fresh model, predict the same."""

    def test_weights_round_trip_bit_for_bit(self, tmp_path):
        """
        Not approximately: a served model and a scored model are the same model.

        A round trip that lost precision would make the difference show up as
        a discrepancy between a backtest and live trading, which is the last
        place anyone looks for a serialisation bug.
        """
        model = make_model(seed=1)
        inputs = torch.ones(3, 4)
        expected = model(inputs)

        path = tmp_path / "weights.pt"
        save_weights(ModelHandle(model=model, unwrapped=model), path)

        fresh = make_model(seed=2)
        assert not torch.equal(fresh(inputs), expected)
        load_weights(fresh, path)
        assert torch.equal(fresh(inputs), expected)

    def test_a_wrapper_prefix_is_stripped(self, tmp_path):
        """
        So a bundle written from a distributed run reopens without one.

        ``DistributedDataParallel`` prefixes every key with ``module.`` and
        ``torch.compile`` with ``_orig_mod.``. A bundle carrying those keys
        can only be loaded by reconstructing the same wrapper -- which the
        reader of the bundle has no way to know about.
        """
        model = make_model()
        wrapped = nn.Sequential(model)  # Stands in for a wrapper's nesting.
        path = tmp_path / "weights.pt"
        save_weights(ModelHandle(model=wrapped, unwrapped=model), path)

        fresh = make_model(seed=3)
        load_weights(fresh, path)
        assert torch.equal(fresh(torch.ones(2, 4)), model(torch.ones(2, 4)))

    def test_a_manifest_is_written_beside_the_weights(self, tmp_path):
        """
        So a weights file can be identified without loading it.

        Which matters because loading it is the operation the format exists to
        make safe to refuse.
        """
        model = make_model()
        path = tmp_path / "weights.pt"
        save_weights(ModelHandle(model=model, unwrapped=model), path)
        assert (tmp_path / "weights.json").exists()


class TestDefectTenPickledCheckpointsAreRefused:
    """``weights_only=True``, and a message that says what to do instead."""

    def test_a_checkpoint_containing_a_module_is_refused(self, tmp_path):
        """
        The remote code execution path, closed.

        A pickled module in a checkpoint means loading the file executes code
        from it. A model artifact travels between machines and is fetched by a
        serving process, so that is not a theoretical concern.
        """
        path = tmp_path / "pickled.pt"
        torch.save(make_model(), path)
        with pytest.raises(EngineError):
            load_weights(make_model(), path)

    def test_the_message_points_at_regeneration(self, tmp_path):
        """
        Rather than at the flag that would re-open the hole.

        Torch's own error helpfully suggests ``weights_only=False``, which is
        precisely the thing not to do -- so the message is replaced rather
        than passed through.
        """
        path = tmp_path / "pickled.pt"
        torch.save(make_model(), path)
        with pytest.raises(EngineError) as caught:
            load_weights(make_model(), path)
        assert "weights_only=False" not in str(caught.value)
        assert "regenerat" in str(caught.value).lower()
```

---

## 4. `tests/rade_qnet/engines/torch/training/test_training_loops.py`

34249 bytes · SHA-256 `eee06920985be093`

```python
"""
Tests for the two loop drivers.

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

The step driver is tested in the second half of this module, and the mirror
image of that second refusal is one of its tests: ``fit_steps`` rejects a
*bounded* source, because a step budget over a dataset would cut it off
mid-pass or silently repeat it. Between them the two refusals are what make
the source's ``steps_per_epoch`` the single thing that selects a driver.

Both drivers are exercised here with stub learners rather than the real ones,
so that a failure in these tests is a failure in a driver. The learners have
their own tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from torch import nn

from src.rade_qnet.core.contract.data import TARGET_KEY
from src.rade_qnet.core.contract.signature import SpaceSpec
from src.rade_qnet.core.lifecycle.errors import EngineError
from src.rade_qnet.core.spec.training import CheckpointSpec, EarlyStoppingSpec
from src.rade_qnet.engines.torch.training.callbacks import (
    BestCheckpoint,
    EarlyStopping,
    GradientNorms,
)
from src.rade_qnet.engines.torch.training.loops import fit_epochs, fit_steps
from src.rade_qnet.sources.batching.rollout import RolloutSource
from src.rade_qnet.sources.environment import StepOutcome
from src.rade_qnet.testkit.fixtures import SyntheticTensorSource

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


@dataclass
class StubPolicyLearner:
    """
    A ``PolicyLearner`` that records what it was handed.

    Stubbed for the same reason :class:`StubLearner` is: a failure in the
    tests below should be a failure in the driver, not in an algorithm.

    Parameters
    ----------
    losses
        Loss to report for each successive update, cycling when exhausted.
    """

    losses: list[float] = field(default_factory=lambda: [1.0])
    batches: list[int] = field(default_factory=list)
    keys: list[tuple[str, ...]] = field(default_factory=list)
    updates: int = 0

    def act(self, policy, observation):
        """
        Choose a constant action.

        Parameters
        ----------
        policy
            Unused.
        observation
            Unused.

        Returns
        -------
        numpy.int64
            Always zero.
        """
        del policy, observation
        return np.int64(0)

    def update(self, policy, experience):
        """
        Record the batch and report the next scripted loss.

        Parameters
        ----------
        policy
            Unused.
        experience
            Recorded, so a test can assert what the driver passed.

        Returns
        -------
        dict
            The scripted loss, plus a counter that averages to one.
        """
        del policy
        self.batches.append(int(experience[TARGET_KEY].shape[0]))
        self.keys.append(tuple(sorted(experience)))
        loss = self.losses[self.updates % len(self.losses)]
        self.updates += 1
        return {"loss": loss, "updates": 1.0}


class StepCounter:
    """
    An environment that rewards one per step and terminates every third.

    Parameters
    ----------
    episode_length
        Steps before termination.
    """

    observation_space = SpaceSpec(kind="box", shape=(1,), dtype="float32")
    action_space = SpaceSpec(kind="discrete", n=2)

    def __init__(self, episode_length: int = 3) -> None:
        self.episode_length = episode_length
        self.index = 0

    def reset(self, *, seed=None):
        """
        Start an episode.

        Parameters
        ----------
        seed
            Unused.

        Returns
        -------
        numpy.ndarray
            The zero observation.
        """
        del seed
        self.index = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        """
        Advance one step.

        Parameters
        ----------
        action
            Unused.

        Returns
        -------
        StepOutcome
            A unit reward, terminating at the episode length.
        """
        del action
        self.index += 1
        return StepOutcome(
            np.full(1, self.index, dtype=np.float32),
            reward=1.0,
            terminated=self.index >= self.episode_length,
        )


def rollout(learner, *, batch_size=4, episode_length=3):
    """
    Build a rollout source driven by a learner's ``act``.

    Parameters
    ----------
    learner
        Supplies the action selector.
    batch_size
        Transitions per batch, and therefore per update.
    episode_length
        Steps before the environment terminates.

    Returns
    -------
    RolloutSource
        An unbounded source of experience.
    """
    policy = nn.Linear(1, 2)
    return RolloutSource(
        StepCounter(episode_length=episode_length),
        act=lambda observation: learner.act(policy, observation),
        batch_size=batch_size,
    )


class TestTheStepBudget:
    """How many steps run, and how they are grouped into records."""

    def test_the_budget_is_counted_in_transitions(self):
        """
        Not in updates, and not from a configured batch size.

        The source decides how much experience one update gets; the driver
        counts what it was handed. A second number here could disagree, and
        the run would simply stop at the wrong time.
        """
        learner = StubPolicyLearner()
        source = rollout(learner, batch_size=4)

        fit_steps(
            nn.Linear(1, 2),
            source,
            learner=learner,
            device=CPU,
            total_steps=12,
            report_every_steps=12,
        )
        assert learner.updates == 3
        assert learner.batches == [4, 4, 4]

    def test_one_record_per_block_not_per_update(self):
        """
        A history with one entry per update would be unreadable.

        It would also make early stopping fire on the noise of a single
        batch, which for an interactive run is a very small sample.
        """
        learner = StubPolicyLearner()
        outcome = fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2),
            learner=learner,
            device=CPU,
            total_steps=12,
            report_every_steps=6,
        )
        assert learner.updates == 6
        assert outcome.n_epochs == 2

    def test_the_final_block_is_cut_to_the_budget(self):
        """
        A budget that is not a whole number of blocks stops on the budget.

        Overrunning would make two runs with different block sizes do
        different amounts of work for the same declared budget.
        """
        learner = StubPolicyLearner()
        fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2),
            learner=learner,
            device=CPU,
            total_steps=10,
            report_every_steps=4,
        )
        assert learner.updates == 5

    def test_block_indices_are_consecutive_from_zero(self):
        """So a reader can line a record up against the run's progress."""
        learner = StubPolicyLearner()
        outcome = fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2),
            learner=learner,
            device=CPU,
            total_steps=12,
            report_every_steps=4,
        )
        assert [record.epoch for record in outcome.history] == [0, 1, 2]


class TestWhatTheLearnerReceives:
    """The experience arrives whole, on the device, not pre-split."""

    def test_the_whole_transition_is_passed(self):
        """
        Including the keys a policy-gradient method would ignore.

        A driver that split the batch would have to know which algorithm it
        was driving, which is exactly the coupling the learner split removes.
        """
        learner = StubPolicyLearner()
        fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=4),
            learner=learner,
            device=CPU,
            total_steps=4,
            report_every_steps=4,
        )
        assert learner.keys[0] == (
            "action",
            "next_observation",
            "observation",
            "target",
            "terminated",
            "truncated",
        )

    def test_the_batch_arrives_as_tensors(self):
        """Converted once by the driver, so no learner converts its own."""

        class Inspect(StubPolicyLearner):
            def update(self, policy, experience):
                """Assert every entry is a tensor, then defer."""
                assert all(isinstance(value, torch.Tensor) for value in experience.values())
                return super().update(policy, experience)

        learner = Inspect()
        fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2),
            learner=learner,
            device=CPU,
            total_steps=2,
            report_every_steps=2,
        )
        assert learner.updates == 1


class TestTheHistoryOfAnInteractiveRun:
    """What a block's record says, and what it deliberately does not."""

    def test_the_block_loss_is_the_mean_over_its_updates(self):
        """Not the last update's, which would make the history a random walk."""
        learner = StubPolicyLearner(losses=[1.0, 3.0])
        outcome = fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2),
            learner=learner,
            device=CPU,
            total_steps=4,
            report_every_steps=4,
        )
        assert outcome.history[0].train_loss == pytest.approx(2.0)

    def test_there_is_no_validation_loss(self):
        """
        Left ``None``, because an environment has no held-out split.

        Reporting the training loss there would make a monitor watching
        ``val_loss`` believe it was watching validation.
        """
        learner = StubPolicyLearner()
        outcome = fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2),
            learner=learner,
            device=CPU,
            total_steps=4,
            report_every_steps=2,
        )
        assert all(record.val_loss is None for record in outcome.history)

    def test_the_episode_return_is_folded_in(self):
        """
        The interpretable number, taken from the source's own bookkeeping.

        A loss is barely comparable between runs; a mean episode return is
        what answers "is this agent any good".
        """
        learner = StubPolicyLearner()
        outcome = fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=3, episode_length=3),
            learner=learner,
            device=CPU,
            total_steps=6,
            report_every_steps=6,
        )
        assert outcome.history[0].metrics["episode_return"] == pytest.approx(3.0)

    def test_a_source_that_cannot_summarise_is_simply_not_asked(self):
        """
        The capability is optional, so a replay source need not invent one.

        Checked with ``isinstance`` and skipped when absent, which is the
        framework's standard capability pattern.
        """

        class Bare:
            steps_per_epoch = None

            def batches(self):
                """Yield constant experience forever."""
                while True:
                    yield {TARGET_KEY: np.ones(2, dtype=np.float32)}

        learner = StubPolicyLearner()
        outcome = fit_steps(
            nn.Linear(1, 2),
            Bare(),
            learner=learner,
            device=CPU,
            total_steps=4,
            report_every_steps=4,
        )
        assert "episode_return" not in outcome.history[0].metrics

    def test_other_scalars_are_averaged_into_the_metrics(self):
        """A learner's extra keys survive, as they do for the epoch driver."""
        learner = StubPolicyLearner()
        outcome = fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2),
            learner=learner,
            device=CPU,
            total_steps=4,
            report_every_steps=4,
        )
        assert outcome.history[0].metrics["updates"] == pytest.approx(1.0)


class TestTheStepDriverRefusals:
    """Each driver refuses the other's source, and says which to use."""

    def test_a_bounded_source_is_refused(self):
        """
        The mirror of ``fit_epochs`` refusing an unbounded one.

        A step budget over a dataset would cut it off mid-pass or silently
        repeat it, and either way the reported numbers would describe
        something other than the dataset.
        """

        class Bounded:
            steps_per_epoch = 10

            def batches(self):
                """Never called."""
                raise AssertionError("the source should be refused before use")

        learner = StubPolicyLearner()
        with pytest.raises(EngineError, match="Use fit_epochs instead"):
            fit_steps(
                nn.Linear(1, 2),
                Bounded(),
                learner=learner,
                device=CPU,
                total_steps=4,
                report_every_steps=2,
            )

    def test_a_learner_reporting_no_loss_is_refused(self):
        """
        Rather than defaulted to zero.

        Zero is the best possible value, so a default would make the first
        block the permanent best block -- the same reasoning as for the
        epoch driver.
        """

        class Silent(StubPolicyLearner):
            def update(self, policy, experience):
                """Report a scalar that is not the objective."""
                del policy, experience
                return {"entropy": 0.5}

        learner = Silent()
        with pytest.raises(EngineError, match="loss"):
            fit_steps(
                nn.Linear(1, 2),
                rollout(learner, batch_size=2),
                learner=learner,
                device=CPU,
                total_steps=2,
                report_every_steps=2,
            )

    def test_a_non_positive_budget_is_refused(self):
        """A run that does nothing is a configuration error, not a no-op."""
        learner = StubPolicyLearner()
        with pytest.raises(EngineError, match="positive"):
            fit_steps(
                nn.Linear(1, 2),
                rollout(learner, batch_size=2),
                learner=learner,
                device=CPU,
                total_steps=0,
                report_every_steps=2,
            )

    def test_a_non_finite_loss_stops_the_run(self):
        """
        The parameters are already NaN and no later update recovers.

        Raised at the block boundary rather than carried forward, so a
        checkpoint is never taken from weights known to be broken.
        """
        learner = StubPolicyLearner(losses=[float("nan")])
        with pytest.raises(EngineError, match="no longer finite"):
            fit_steps(
                nn.Linear(1, 2),
                rollout(learner, batch_size=2),
                learner=learner,
                device=CPU,
                total_steps=2,
                report_every_steps=2,
            )

    def test_experience_with_no_reward_is_refused(self):
        """The budget is counted in transitions, so a batch must carry rewards."""

        class Rewardless:
            steps_per_epoch = None

            def batches(self):
                """Yield a batch with no target."""
                while True:
                    yield {"observation": np.zeros((2, 1), dtype=np.float32)}

        learner = StubPolicyLearner()

        class Tolerant(StubPolicyLearner):
            def update(self, policy, experience):
                """Report a loss without inspecting the batch."""
                del policy, experience
                return {"loss": 1.0}

        learner = Tolerant()
        with pytest.raises(EngineError, match="not experience"):
            fit_steps(
                nn.Linear(1, 2),
                Rewardless(),
                learner=learner,
                device=CPU,
                total_steps=2,
                report_every_steps=2,
            )


class TestTheStepDriverAndCallbacks:
    """Early stopping works unchanged, which is why the split was worth keeping."""

    def test_early_stopping_ends_the_run_between_blocks(self):
        """
        The same callbacks serve both drivers.

        Consulted per block rather than per update, so a stopping decision
        is made from a sample worth deciding on.

        ``has_validation=False`` is the honest setting for an interactive
        run, and the callback's existing guard already refuses a ``val_*``
        monitor in that case -- so the one monitor an interactive run could
        have watched by mistake is rejected without this driver adding a
        check of its own.

        The blocks here are deliberately longer than an episode. A block in
        which no episode finished reports no episode return -- empty rather
        than zero, by design -- and a callback monitoring it then fails with
        ``nothing monitors 'episode_return'``. That message is accurate and
        the constraint is real: a block has to be long enough to contain an
        episode before an episode statistic can be monitored.
        """
        learner = StubPolicyLearner(losses=[5.0])
        stopping = EarlyStopping(
            EarlyStoppingSpec(
                enabled=True,
                patience=1,
                # The metric worth watching in an interactive run, and it
                # reaches the callback only because the driver folds the
                # source's episode summary into each block's metrics.
                monitor="episode_return",
                mode="max",
            ),
            has_validation=False,
        )

        unstopped = fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2, episode_length=3),
            learner=learner,
            device=CPU,
            total_steps=60,
            report_every_steps=6,
        )
        assert unstopped.n_epochs == 10
        assert not unstopped.stopped_early

        stopped = fit_steps(
            nn.Linear(1, 2),
            rollout(learner, batch_size=2, episode_length=3),
            learner=learner,
            device=CPU,
            total_steps=60,
            report_every_steps=6,
            callbacks=[stopping],
        )
        assert stopped.stopped_early
        assert stopped.n_epochs < unstopped.n_epochs
```

---

## 5. `tests/rade_qnet/engines/torch/training/test_training_losses.py`

8874 bytes · SHA-256 `ad6de8c53529ae7d`

```python
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

from src.rade_qnet.core.lifecycle.errors import EngineError
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
```

