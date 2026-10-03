"""
``fit_epochs`` -- passes over a finite source, driving a learner.

The library split described in ``ARCHITECTURE.md`` §6: the loop decides *when*
to step, validate, checkpoint and stop; the learner decides *what* one update
means. A supervised regression, a deep Q-network and a pathwise hedging
objective are then three learners sharing this one function, rather than three
training scripts.

Why the ``Learner`` protocol is declared here
---------------------------------------------
The loop is the consumer, so the loop declares the interface it needs. The
same dependency inversion puts ``Catalog`` and ``Tracker`` in
``core.runtime.context`` rather than in ``storage``: the party that depends on
a capability owns its definition, and the party that provides it satisfies a
contract it need not import.

Two drivers, one of them deferred
---------------------------------
:attr:`~rade_xl.core.contract.source.BatchSource.steps_per_epoch` returning
``None`` means the source is unbounded, and that single value selects the
driver. A fixed dataset has a meaningful notion of a pass, so it is driven
here by epochs. An environment does not, so Phase 7 adds ``fit_steps``.

:func:`fit_epochs` refuses an unbounded source rather than inventing a pass
length for it, because the invented number would silently become the
denominator of every reported metric and the period of every schedule.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Protocol

import torch

from ...core.contract.result import EpochRecord, FitOutcome
from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger
from .callbacks import EpochContext
from .loaders import StaticInputs, to_device_batches

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ...core.contract.signature import InputSignature
    from ...core.contract.source import BatchSource
    from .callbacks import BestCheckpoint, Callback, GradientNorms

__all__ = ["Learner", "fit_epochs"]

_LOGGER = get_logger(__name__)

#: Key every learner reports its scalar objective under.
LOSS_KEY = "loss"

#: Key a learner uses to flag a step it skipped for a non-finite loss.
_NON_FINITE_KEY = "non_finite_steps"


class Learner(Protocol):
    """
    What one optimisation step means.

    Implemented by :class:`~.learners.supervised.SupervisedLearner` and, from
    Phase 7, by the reinforcement-learning update rules.
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
