"""
The two loop drivers: ``fit_epochs`` over a dataset, ``fit_steps`` over an environment.

The library split described in ``ARCHITECTURE.md`` §6: the loop decides *when*
to step, validate, checkpoint and stop; the learner decides *what* one update
means. A supervised regression, a deep Q-network and a pathwise hedging
objective are then three learners sharing a driver, rather than three
training scripts.

Why the learner protocols are declared here
-------------------------------------------
The loop is the consumer, so the loop declares the interface it needs. The
same dependency inversion puts ``Catalog`` and ``Tracker`` in
``core.runtime.context`` rather than in ``storage``: the party that depends on
a capability owns its definition, and the party that provides it satisfies a
contract it need not import.

Two drivers, selected by the source
-----------------------------------
:attr:`~rade_qnet.core.contract.source.BatchSource.steps_per_epoch` returning
``None`` means the source is unbounded, and that single value selects the
driver. A fixed dataset has a meaningful notion of a pass, so it is driven by
epochs. An environment does not, so it is driven by a step budget. The source
declares which it is; the engine does not guess.

Each driver refuses the other's source rather than coping, because the
invented number -- a pass length for an environment, a step budget for a
dataset -- would silently become the denominator of every reported metric and
the period of every schedule.

Record: why there are two learner protocols
-------------------------------------------
``PHASE_7_REINFORCEMENT_LEARNING.md`` §3.1 claimed Phase 7 would add its
learners "against an unchanged loop", and said that if a second loop were
needed, *that* was the finding rather than a licence to fork. So, recorded as
the finding:

The second **driver** was never the surprise -- ``core.contract.source`` has
specified since Phase 1 that an unbounded source "is driven by ``fit_steps``",
and :func:`fit_epochs` has always refused one by name. What did not survive
contact is the **learner protocol**. :class:`Learner` takes
``(model, inputs, target)``, because :func:`fit_epochs` splits each batch into
inputs and a target using the signature's declared target.

A policy has no target. ``PolicySignature`` has an observation space and an
action space and nothing resembling one, and what an interactive update
consumes is a whole transition -- observation, action, reward, next
observation, and the two episode-end flags -- not a pair. Passing a reward as
``target`` would type-check and would be a lie: a reward is not a label, and
a learner reading it as one is the exact confusion the paradigm split exists
to prevent.

Hence :class:`PolicyLearner`, with its own two methods. The alternative --
widening :class:`Learner` to make ``target`` optional and adding the five
other fields -- would push the branch *into every supervised learner*, which
is the coupling the split was for. The loop/learner split itself held: both
drivers below are the same shape, and the pipeline is a sibling of the
supervised one rather than a forked lifecycle.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from ....core.contract.result import EpochRecord, FitOutcome
from ....core.runtime.errors import EngineError
from ....core.runtime.logging import get_logger
from ..loaders import TARGET_KEY, StaticInputs, to_device_batches, to_tensor
from .callbacks import EpochContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ....core.contract.signature import InputSignature
    from ....core.contract.source import BatchSource
    from .callbacks import BestCheckpoint, Callback, GradientNorms

__all__ = ["ExperienceSummary", "Learner", "PolicyLearner", "fit_epochs", "fit_steps"]

_LOGGER = get_logger(__name__)

#: Key every learner reports its scalar objective under.
LOSS_KEY = "loss"

#: Key a learner uses to flag a step it skipped for a non-finite loss.
_NON_FINITE_KEY = "non_finite_steps"


class Learner(Protocol):
    """
    What one optimisation step means, for supervised learning.

    Implemented by :class:`~.learners.supervised.SupervisedLearner`. An
    interactive update rule implements :class:`PolicyLearner` instead -- see
    the record in the module docstring for why the two are separate.
    """

    def train_step(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, float]:
        """
        Perform one update and report its scalars.

        Parameters
        ----------
        model
            The prepared model.
        inputs
            Batch inputs.
        target
            Observed values.

        Returns
        -------
        dict
            Must carry ``loss``; any other key is averaged into the epoch's
            metrics.
        """
        ...

    def eval_step(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute scalars for one batch without updating anything.

        Parameters
        ----------
        model
            The prepared model.
        inputs
            Batch inputs.
        target
            Observed values.

        Returns
        -------
        dict
            Must carry ``loss``.
        """
        ...


class PolicyLearner(Protocol):
    """
    What one interactive update means, and how the policy chooses an action.

    Two methods rather than :class:`Learner`'s two, because interaction needs
    a different pair.

    :meth:`act` is here, and not on the model, because exploration is a
    property of the *algorithm*: the same network is acted on greedily by one
    learner, epsilon-greedily by another, and sampled from by a third. Putting
    it on the policy would force every network to carry every exploration
    scheme, and would make a saved policy's behaviour depend on which
    algorithm had trained it.

    There is no ``eval_step``. A dataset has a held-out split; an environment
    does not, and the only honest way to evaluate a policy is to run more
    episodes with exploration turned off. That is a different *amount of
    interaction*, not a different step, so it belongs to the pipeline which
    can collect it -- not here.
    """

    def act(self, policy: torch.nn.Module, observation: object) -> object:
        """
        Choose one action for one observation.

        Called by the collector, not by the driver, and therefore outside any
        gradient graph: an implementation must detach or use ``no_grad``, or
        the graph of every action taken is retained until the update, which
        for a replay buffer means retaining it for the whole run.

        Parameters
        ----------
        policy
            The prepared policy.
        observation
            One observation, already placed on the policy's device by the
            engine and not batched. Device placement happens there rather
            than here for the same reason it does for a batch: a learner
            that had to do its own would get it wrong once per learner, and
            the symptom -- a forward pass failing inside a linear layer -- is
            a long way from the cause.

        Returns
        -------
        object
            One action the environment will accept. Not necessarily a
            tensor: a discrete environment wants an index.
        """
        ...

    def update(
        self,
        policy: torch.nn.Module,
        experience: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Perform one update from a batch of experience and report its scalars.

        Parameters
        ----------
        policy
            The prepared policy.
        experience
            One batch of transitions, on the device, keyed as the source's
            signature declares: observations, actions, next observations, the
            two episode-end flags, and the reward under
            :data:`~rade_qnet.core.contract.data.TARGET_KEY`.

            Passed whole rather than pre-split, because what an algorithm
            reads varies -- a policy-gradient method ignores the next
            observation entirely, a value-based one depends on it -- and a
            driver that split it would have to know which algorithm it was
            driving.

        Returns
        -------
        dict
            Must carry ``loss``; any other key is averaged into the reported
            block's metrics. A learner that performs no update still reports
            one, so that "nothing happened" is visible rather than absent.
        """
        ...


def _mean_of(totals: Mapping[str, float], n_batches: int) -> dict[str, float]:
    """
    Average a pass's accumulated scalars.

    Parameters
    ----------
    totals
        Summed scalars.
    n_batches
        Number of batches summed.

    Returns
    -------
    dict
        Means, or an empty mapping when no batch ran.
    """
    if n_batches == 0:
        return {}
    # The non-finite counter is a count and not an average: "0.03 non-finite
    # steps" is meaningless, "3 non-finite steps" is actionable.
    return {
        name: total if name == _NON_FINITE_KEY else total / n_batches
        for name, total in totals.items()
    }


def _run_pass(
    source: BatchSource,
    *,
    model: torch.nn.Module,
    learner: Learner,
    signature: InputSignature,
    device: torch.device,
    static: StaticInputs,
    training: bool,
) -> dict[str, float]:
    """
    Run one pass over a source and return its mean scalars.

    Parameters
    ----------
    source
        The batch source.
    model
        The prepared model.
    learner
        The update rule.
    signature
        The declared interface.
    device
        Where tensors are placed.
    static
        Already-uploaded static inputs.
    training
        Whether to update parameters. Also sets the model's mode, which is
        what makes dropout and batch normalisation behave correctly -- an
        evaluation pass left in training mode reports a loss computed with
        dropout active and batch statistics from the held-out batch.

    Returns
    -------
    dict
        Mean scalars for the pass.
    """
    model.train(training)
    step = learner.train_step if training else learner.eval_step

    totals: dict[str, float] = {}
    n_batches = 0
    for inputs, target in to_device_batches(
        source, signature=signature, device=device, static=static
    ):
        for name, value in step(model, inputs, target).items():
            totals[name] = totals.get(name, 0.0) + value
        n_batches += 1

    return _mean_of(totals, n_batches)


def fit_epochs(
    model: torch.nn.Module,
    sources: Mapping[str, BatchSource],
    *,
    learner: Learner,
    signature: InputSignature,
    device: torch.device,
    epochs: int,
    callbacks: Sequence[Callback] = (),
    gradient_norms: GradientNorms | None = None,
    checkpoint: BestCheckpoint | None = None,
    learning_rate: float | None = None,
) -> FitOutcome:
    """
    Train for a budget of passes over the training source.

    Parameters
    ----------
    model
        The prepared model. Must already be materialised: an optimiser built
        over unmaterialised parameters tracks nothing, which is defect 6.
    sources
        Batch sources by split name. ``train`` is required.
    learner
        The update rule.
    signature
        The declared interface.
    device
        Where tensors are placed.
    epochs
        Maximum passes over the training source.
    callbacks
        Consulted at the end of each epoch, in order. Any one of them may end
        the run.
    gradient_norms
        Tracker whose per-epoch summary is folded into the epoch's metrics.
    checkpoint
        Best-weight tracker, asked to restore at the end. Passed separately
        from ``callbacks`` because the loop needs its answer -- which epoch
        was best, and whether its weights were restored -- for the outcome,
        and a bare ``Callback`` cannot be asked. It also supplies the name
        recorded as the monitored metric, which is taken from it rather than
        passed separately so the reported monitor cannot disagree with the one
        that actually chose the best epoch.
    learning_rate
        Rate to record against the first epoch when no schedule is in use.

    Returns
    -------
    FitOutcome
        The epoch history, the best epoch, and whether the run stopped early
        and restored the best weights.

    Raises
    ------
    EngineError
        If there is no training source, if it is unbounded, or if a learner
        reports no ``loss``.
    """
    if "train" not in sources:
        raise EngineError(f"fit_epochs needs a 'train' source; received {sorted(sources)}")

    train_source = sources["train"]
    if train_source.steps_per_epoch is None:
        raise EngineError(
            "fit_epochs cannot drive an unbounded source, which reports "
            "steps_per_epoch=None. An unbounded source has no notion of a pass, "
            "so an epoch count would be arbitrary and would silently become the "
            "denominator of every reported metric. Use fit_steps instead"
        )

    validation_source = sources.get("validation")
    static = StaticInputs.from_source(train_source, signature=signature, device=device)

    history: list[EpochRecord] = []
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(epochs):
        epoch_started = time.perf_counter()

        train_scalars = _run_pass(
            train_source,
            model=model,
            learner=learner,
            signature=signature,
            device=device,
            static=static,
            training=True,
        )
        if LOSS_KEY not in train_scalars:
            raise EngineError(
                f"the learner reported {sorted(train_scalars)} for epoch {epoch} "
                f"but not {LOSS_KEY!r}; every learner must report its objective "
                f"under that key, because the loop and the callbacks are written "
                f"against it"
            )

        # A non-finite training loss means the parameters are already NaN, and
        # nothing in a training loop can recover from that: every subsequent
        # epoch computes NaN gradients over NaN weights.  Stopping here rather
        # than finishing the budget matters because the failure otherwise
        # surfaces in the evaluate stage, as a message about non-finite
        # predictions -- which points at the metrics rather than at the epoch
        # the run actually diverged in.
        if not math.isfinite(train_scalars[LOSS_KEY]):
            raise EngineError(
                f"the training loss became {train_scalars[LOSS_KEY]} at epoch "
                f"{epoch + 1}, so the parameters are no longer finite and no "
                f"later epoch can recover. The usual causes are a non-finite "
                f"input reaching the model, a learning rate too high for the "
                f"objective, or a loss that divides by a quantity that reached "
                f"zero. Check the data build's quality metrics first, then "
                f"lower the learning rate or enable gradient clipping"
            )

        val_scalars = (
            _run_pass(
                validation_source,
                model=model,
                learner=learner,
                signature=signature,
                device=device,
                static=static,
                training=False,
            )
            if validation_source is not None
            else {}
        )

        # Validation scalars are prefixed, so `loss` from the two passes
        # cannot collide in one metrics mapping.
        metrics = {name: value for name, value in train_scalars.items() if name != LOSS_KEY}
        metrics.update(
            {f"val_{name}": value for name, value in val_scalars.items() if name != LOSS_KEY}
        )
        if gradient_norms is not None:
            metrics.update(gradient_norms.summarise())

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_scalars[LOSS_KEY],
            val_loss=val_scalars.get(LOSS_KEY),
            metrics=metrics,
            learning_rate=learning_rate,
            seconds=time.perf_counter() - epoch_started,
        )
        history.append(record)
        _log_epoch(record, epochs)

        context = _context_from(record)
        # Checkpointing runs before the other callbacks so that the snapshot
        # is taken from the epoch's own weights, whatever a schedule does next.
        if checkpoint is not None:
            checkpoint.on_epoch_end(context)
        for callback in callbacks:
            callback.on_epoch_end(context)

        if any(callback.should_stop for callback in callbacks):
            stopped_early = True
            break

        # Read after the callbacks, so a schedule's change is recorded against
        # the epoch it takes effect in rather than the one that caused it.
        learning_rate = _current_rate(callbacks, default=learning_rate)

    restored = checkpoint.restore_best() if checkpoint is not None else False
    best_epoch = checkpoint.monitor.best_epoch if checkpoint is not None else None
    best_value = checkpoint.monitor.best_value if checkpoint is not None else None
    monitor = checkpoint.monitor.monitor if checkpoint is not None else "val_loss"

    outcome = FitOutcome(
        history=tuple(history),
        monitor=monitor,
        best_epoch=best_epoch,
        best_monitor_value=best_value,
        stopped_early=stopped_early,
        restored_best=restored,
        total_seconds=time.perf_counter() - started,
    )
    # `EpochRecord.epoch` is zero-based, but every per-epoch line above was
    # logged one-based in the `n of m` form.  Reporting the raw index here
    # would put "epoch 1/3" and "best ... at epoch 0" in the same log, which
    # reads as an off-by-one in the training rather than in the numbering.
    _LOGGER.info(
        "fit finished: %d epoch(s) in %.1fs, best %s=%s at epoch %s%s",
        outcome.n_epochs,
        outcome.total_seconds,
        monitor,
        None if best_value is None else f"{best_value:.6g}",
        "n/a" if best_epoch is None else best_epoch + 1,
        " (weights restored)" if restored else "",
    )
    return outcome


def _context_from(record: EpochRecord) -> EpochContext:
    """
    Build the callback context for a finished epoch.

    Parameters
    ----------
    record
        The epoch's record.

    Returns
    -------
    EpochContext
        The context callbacks are given.
    """
    return EpochContext(
        epoch=record.epoch,
        train_loss=record.train_loss,
        val_loss=record.val_loss,
        metrics=record.metrics,
        learning_rate=record.learning_rate,
    )


def _current_rate(callbacks: Sequence[Callback], *, default: float | None) -> float | None:
    """
    Return the learning rate a schedule callback reports, if there is one.

    Parameters
    ----------
    callbacks
        The callbacks in use.
    default
        Rate to return when no callback reports one.

    Returns
    -------
    float or None
        The current rate.
    """
    for callback in callbacks:
        rate = getattr(callback, "current_rate", None)
        if rate is not None:
            return float(rate)
    return default


def _log_epoch(record: EpochRecord, epochs: int) -> None:
    """
    Log one epoch's headline numbers.

    Parameters
    ----------
    record
        The epoch's record.
    epochs
        The epoch budget, for the ``n of m`` form.
    """
    validation = "" if record.val_loss is None else f" val_loss={record.val_loss:.6g}"
    _LOGGER.info(
        "epoch %d/%d train_loss=%.6g%s (%.2fs)",
        record.epoch + 1,
        epochs,
        record.train_loss,
        validation,
        record.seconds,
    )


@runtime_checkable
class ExperienceSummary(Protocol):
    """
    An opt-in capability: a source that can summarise the episodes it has seen.

    Checked with ``isinstance`` and skipped when absent, which is the
    framework's standard capability pattern -- see
    ``core.capability.protocols``. Optional rather than part of
    :class:`~rade_qnet.core.contract.source.BatchSource` because a replay
    source serving stored transitions has no episodes of its own to report,
    and widening the source protocol would force it to invent an answer.
    """

    def episode_summary(self) -> Mapping[str, float]:
        """
        Summarise the finished episodes.

        Returns
        -------
        Mapping
            Named statistics, or empty when no episode has finished.
        """
        ...


def fit_steps(
    policy: torch.nn.Module,
    source: BatchSource,
    *,
    learner: PolicyLearner,
    device: torch.device,
    total_steps: int,
    report_every_steps: int,
    callbacks: Sequence[Callback] = (),
    gradient_norms: GradientNorms | None = None,
    checkpoint: BestCheckpoint | None = None,
    learning_rate: float | None = None,
) -> FitOutcome:
    """
    Train for a budget of environment steps against an unbounded source.

    The interactive counterpart of :func:`fit_epochs`, and deliberately the
    same shape: collect, update, record, consult the callbacks, maybe stop.
    What differs is only the unit of progress -- steps taken rather than
    passes completed -- and that the experience arrives whole rather than
    split into inputs and a target.

    How the budget is counted
    ------------------------
    One update per batch, and the steps in that batch are counted from the
    batch itself rather than from a configured batch size. The source decides
    how much experience one update gets; the driver counts what it was
    handed. A second number here could disagree with the source's, and the
    disagreement would be invisible -- the run would simply stop at the wrong
    time.

    Parameters
    ----------
    policy
        The prepared policy. Must already be materialised, for the reason in
        :func:`fit_epochs`.
    source
        An unbounded source of experience, normally a
        :class:`~rade_qnet.sources.batching.rollout.RolloutSource`.
    learner
        The update rule, which also chooses the actions.
    device
        Where tensors are placed.
    total_steps
        Environment steps to run before stopping.
    report_every_steps
        Steps per reported block. One :class:`EpochRecord` is emitted per
        block, and the callbacks are consulted once per block.

        Blocks rather than updates because an interactive run does thousands
        of small updates, and a history with one entry each would be useless
        to read and would make early stopping fire on the noise of a single
        batch.
    callbacks
        Consulted at the end of each block, in order. Any one of them may end
        the run.
    gradient_norms
        Tracker whose summary is folded into each block's metrics.
    checkpoint
        Best-weight tracker. For an interactive run the metric worth
        monitoring is the episode return, not a loss, because a policy-
        gradient loss falling says almost nothing about whether the agent
        improved.
    learning_rate
        Rate to record against the first block when no schedule is in use.

    Returns
    -------
    FitOutcome
        The block history, the best block, and whether the run stopped early
        and restored the best weights.

    Raises
    ------
    EngineError
        If the source is bounded, if the budget or block size is not
        positive, if a learner reports no ``loss``, or if the loss becomes
        non-finite.
    """
    if source.steps_per_epoch is not None:
        raise EngineError(
            f"fit_steps drives an unbounded source, but this one reports "
            f"steps_per_epoch={source.steps_per_epoch}. A bounded source has a "
            f"meaningful pass, so a step budget would cut it off mid-dataset or "
            f"silently repeat it. Use fit_epochs instead"
        )
    if total_steps <= 0 or report_every_steps <= 0:
        raise EngineError(
            f"fit_steps needs positive total_steps and report_every_steps; "
            f"received {total_steps} and {report_every_steps}"
        )

    history: list[EpochRecord] = []
    stopped_early = False
    started = time.perf_counter()

    steps_taken = 0
    block = 0
    block_started = time.perf_counter()
    totals: dict[str, float] = {}
    updates_in_block = 0
    next_report = min(report_every_steps, total_steps)

    policy.train(True)
    for batch in source.batches():
        experience = {name: to_tensor(value, device=device) for name, value in batch.items()}
        scalars = learner.update(policy, experience)
        if LOSS_KEY not in scalars:
            raise EngineError(
                f"the learner reported {sorted(scalars)} for the update at step "
                f"{steps_taken} but not {LOSS_KEY!r}; every learner must report "
                f"its objective under that key, because the loop and the "
                f"callbacks are written against it"
            )

        for name, value in scalars.items():
            totals[name] = totals.get(name, 0.0) + value
        updates_in_block += 1
        steps_taken += _steps_in(experience)

        if steps_taken < next_report:
            continue

        record = _block_record(
            block,
            totals=totals,
            updates=updates_in_block,
            source=source,
            gradient_norms=gradient_norms,
            learning_rate=learning_rate,
            seconds=time.perf_counter() - block_started,
        )
        # Non-finite for the same reason as in `fit_epochs`: the parameters
        # are already NaN and no later update recovers.  Raised here rather
        # than after the callbacks so a checkpoint is never taken from
        # weights that are known to be broken.
        if not math.isfinite(record.train_loss):
            raise EngineError(
                f"the training loss became {record.train_loss} after "
                f"{steps_taken} step(s), so the parameters are no longer finite "
                f"and no later update can recover. In an interactive run the "
                f"usual causes are an unbounded reward, a value target that "
                f"bootstrapped past a terminal state, or a learning rate too "
                f"high for the objective. Check the episode return first: if it "
                f"diverged before the loss did, the reward is the problem"
            )

        history.append(record)
        _log_block(record, steps_taken=steps_taken, total_steps=total_steps)

        context = _context_from(record)
        if checkpoint is not None:
            checkpoint.on_epoch_end(context)
        for callback in callbacks:
            callback.on_epoch_end(context)

        if any(callback.should_stop for callback in callbacks):
            stopped_early = True
            break

        learning_rate = _current_rate(callbacks, default=learning_rate)

        if steps_taken >= total_steps:
            break

        block += 1
        block_started = time.perf_counter()
        totals = {}
        updates_in_block = 0
        next_report = min(steps_taken + report_every_steps, total_steps)

    restored = checkpoint.restore_best() if checkpoint is not None else False
    outcome = FitOutcome(
        history=tuple(history),
        monitor=checkpoint.monitor.monitor if checkpoint is not None else LOSS_KEY,
        best_epoch=checkpoint.monitor.best_epoch if checkpoint is not None else None,
        best_monitor_value=checkpoint.monitor.best_value if checkpoint is not None else None,
        stopped_early=stopped_early,
        restored_best=restored,
        total_seconds=time.perf_counter() - started,
    )
    _LOGGER.info(
        "fit finished: %d step(s) over %d block(s) in %.1fs%s",
        steps_taken,
        outcome.n_epochs,
        outcome.total_seconds,
        " (weights restored)" if restored else "",
    )
    return outcome


def _steps_in(experience: Mapping[str, torch.Tensor]) -> int:
    """
    Count the transitions in one batch of experience.

    Read from the target -- the reward -- because it is the one entry
    guaranteed to be exactly one value per transition. An observation may
    carry extra dimensions, and a source may add keys this driver has never
    heard of.

    Parameters
    ----------
    experience
        One batch, on the device.

    Returns
    -------
    int
        The number of transitions.

    Raises
    ------
    EngineError
        If the batch carries no reward, which means it is not experience.
    """
    reward = experience.get(TARGET_KEY)
    if reward is None:
        raise EngineError(
            f"the experience batch has {sorted(experience)} but no {TARGET_KEY!r}; "
            f"fit_steps counts its budget in transitions, and a batch with no "
            f"reward is not experience"
        )
    return int(reward.shape[0]) if reward.ndim else 1


def _block_record(
    block: int,
    *,
    totals: Mapping[str, float],
    updates: int,
    source: BatchSource,
    gradient_norms: GradientNorms | None,
    learning_rate: float | None,
    seconds: float,
) -> EpochRecord:
    """
    Summarise one block of updates as a history record.

    Parameters
    ----------
    block
        Zero-based block index.
    totals
        Scalars summed over the block's updates.
    updates
        How many updates the block contained.
    source
        Asked for its episode summary, if it can give one.
    gradient_norms
        Tracker whose summary is folded in, if present.
    learning_rate
        Rate in effect for the block.
    seconds
        Wall-clock duration of the block.

    Returns
    -------
    EpochRecord
        The block's record, with ``val_loss`` left ``None`` -- an environment
        has no held-out split, and reporting the training loss there would
        make a monitor watching it think it was watching validation.
    """
    means = _mean_of(totals, updates)
    metrics = {name: value for name, value in means.items() if name != LOSS_KEY}
    if isinstance(source, ExperienceSummary):
        metrics.update(source.episode_summary())
    if gradient_norms is not None:
        metrics.update(gradient_norms.summarise())

    return EpochRecord(
        epoch=block,
        train_loss=means[LOSS_KEY],
        val_loss=None,
        metrics=metrics,
        learning_rate=learning_rate,
        seconds=seconds,
    )


def _log_block(record: EpochRecord, *, steps_taken: int, total_steps: int) -> None:
    """
    Log one block's headline numbers.

    Parameters
    ----------
    record
        The block's record.
    steps_taken
        Steps completed so far, logged instead of a block index because the
        budget is in steps and that is what a reader is tracking.
    total_steps
        The step budget.
    """
    episode_return = record.metrics.get("episode_return")
    reported = "" if episode_return is None else f" episode_return={episode_return:.6g}"
    _LOGGER.info(
        "step %d/%d loss=%.6g%s (%.2fs)",
        steps_taken,
        total_steps,
        record.train_loss,
        reported,
        record.seconds,
    )
