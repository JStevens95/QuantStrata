"""
What the fit, evaluate and predict stages return.

One field recurs and is the most important thing in this module:
``in_original_units``. Every result that carries numbers a user will read
carries a flag asserting those numbers are in the units of the original target
rather than in whatever space the model happened to train in.

It is a field rather than a convention because it is checked. The conformance
suite asserts it is true on anything a pipeline reports, which turns "we
remember to invert the target scaling" from a habit into a property. A mean
absolute error of 0.03 is either excellent or meaningless depending on this
one boolean, and the number alone does not say which.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from ..lifecycle.errors import ContractError, SpecError
from .base import ContractModel

__all__ = [
    "EpochRecord",
    "EvalResult",
    "EvaluationResult",
    "FitOutcome",
    "Predictions",
    "TrainingResult",
    "TrialRecord",
    "TuningResult",
]


class EpochRecord(ContractModel):
    """
    One epoch's outcome.

    Parameters
    ----------
    epoch
        Zero-based epoch index.
    train_loss
        Mean training loss over the epoch.
    val_loss
        Mean validation loss, or ``None`` when there is no validation split.
    metrics
        Any additional metrics computed for the epoch.
    learning_rate
        Rate in effect, recorded so a schedule's behaviour is visible in the
        saved history rather than having to be inferred from the loss curve.
    seconds
        Wall time for the epoch.
    """

    epoch: int = Field(ge=0)
    train_loss: float
    val_loss: float | None = None
    metrics: Mapping[str, float] = Field(default_factory=dict)
    learning_rate: float | None = Field(default=None, gt=0.0)
    seconds: float = Field(default=0.0, ge=0.0)


class FitOutcome(ContractModel):
    """
    What a fit produced, whatever engine performed it.

    A gradient loop over two hundred epochs and a single boosted-tree call both
    report through this type. The tree engine emits one record per boosting
    round, so a "learning curve" means the same thing for both and the same
    report renders either.

    Parameters
    ----------
    history
        One record per epoch or boosting round, in order.
    monitor
        Metric used to identify the best epoch.
    best_epoch
        Index of the best epoch, or ``None`` if nothing was monitored.
    best_monitor_value
        Value of the monitored metric at the best epoch.
    stopped_early
        Whether training ended before the epoch budget was exhausted.
    restored_best
        Whether the best epoch's parameters were restored at the end. When
        false, the reported metrics and the saved weights describe different
        models, so this is recorded rather than assumed.
    total_seconds
        Wall time for the whole fit.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if ``best_epoch`` does not identify a record
        in ``history``.
    """

    history: tuple[EpochRecord, ...] = ()
    monitor: str = "val_loss"
    best_epoch: int | None = Field(default=None, ge=0)
    best_monitor_value: float | None = None
    stopped_early: bool = False
    restored_best: bool = False
    total_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_best_epoch_exists(self) -> FitOutcome:
        """
        Reject a best epoch that is not present in the history.

        Returns
        -------
        FitOutcome
            The validated outcome.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if ``best_epoch`` indexes no record. This catches an off-by-one
            between a callback's epoch counter and the history it recorded,
            which would otherwise surface as the wrong checkpoint being
            restored.
        """
        if self.best_epoch is None:
            return self
        if not any(record.epoch == self.best_epoch for record in self.history):
            recorded = [record.epoch for record in self.history]
            raise SpecError(
                f"best_epoch={self.best_epoch} is not present in the history "
                f"(recorded epochs: {recorded})"
            )
        return self

    @property
    def n_epochs(self) -> int:
        """Number of records in the history."""
        return len(self.history)

    @property
    def best_record(self) -> EpochRecord | None:
        """The best epoch's record, or ``None`` if there is no best epoch."""
        if self.best_epoch is None:
            return None
        return next(record for record in self.history if record.epoch == self.best_epoch)

    def curve(self, name: Literal["train_loss", "val_loss"]) -> tuple[float | None, ...]:
        """
        Return one series from the history, for plotting.

        Parameters
        ----------
        name
            Which series to extract.

        Returns
        -------
        tuple
            One value per record. ``val_loss`` may contain ``None`` entries,
            which a plotting function must handle rather than treating as
            zero -- a gap in a curve and a loss of zero look very different
            and mean very different things.
        """
        return tuple(getattr(record, name) for record in self.history)


class EvalResult(ContractModel):
    """
    Metrics for one split.

    Parameters
    ----------
    split
        Which split was scored.
    metrics
        Metric name to value.
    n_samples
        Number of samples scored, so a metric can be weighted or pooled
        correctly across splits.
    in_original_units
        Whether the metrics are in the original target units. See the module
        docstring: this is checked, not assumed.
    baseline_metrics
        The same metrics for a naive reference model on the same split.
        Carried alongside rather than reported separately, because a headline
        metric without a reference point is not interpretable.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if a metric value is not finite.
    """

    split: str
    metrics: Mapping[str, float]
    n_samples: int = Field(ge=0)
    in_original_units: bool = True
    baseline_metrics: Mapping[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_metrics_are_finite(self) -> EvalResult:
        """
        Reject non-finite metric values.

        Returns
        -------
        EvalResult
            The validated result.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any metric is ``NaN`` or infinite. Such a value almost always
            means a degenerate split or a divide-by-zero inside a metric, and
            it is far cheaper to diagnose here than after it has been written
            to a bundle and compared against other runs.
        """
        offenders = sorted(name for name, value in self.metrics.items() if not np.isfinite(value))
        if offenders:
            raise SpecError(
                f"metric(s) {offenders} for split {self.split!r} are not finite; "
                f"this usually means a degenerate split or a division by zero "
                f"inside a metric"
            )
        return self


class TrainingResult(ContractModel):
    """
    The complete outcome of a training run.

    Parameters
    ----------
    fit
        What the fit itself produced.
    evaluations
        Metrics per split, keyed by split name.
    seed
        The seed actually applied, recorded so a run can be repeated without
        re-deriving it.
    notes
        Free-form annotations, such as which compatibility flags were active.
    bundle_directory
        Where the trained model was saved, or ``None`` if nothing was
        written. Recorded because a result that reports a model's metrics
        but not the model is awkward in exactly the case that matters:
        training and then evaluating, predicting or promoting it. Without
        it the caller has to reconstruct a path from the run id, the model
        name and the version, which is three things to get wrong and a
        private layout to depend on.
    """

    fit: FitOutcome
    evaluations: Mapping[str, EvalResult] = Field(default_factory=dict)
    seed: int = Field(default=0, ge=0)
    notes: Mapping[str, str] = Field(default_factory=dict)
    bundle_directory: str | None = None

    def metric(self, split: str, name: str) -> float:
        """
        Return one metric for one split.

        Parameters
        ----------
        split
            Split name.
        name
            Metric name.

        Returns
        -------
        float
            The metric value.

        Raises
        ------
        ContractError
            If the split or the metric is absent, with the available names
            listed.
        """
        if split not in self.evaluations:
            raise ContractError(
                f"no evaluation for split {split!r}; "
                f"evaluated splits are {sorted(self.evaluations)}"
            )
        metrics = self.evaluations[split].metrics
        if name not in metrics:
            raise ContractError(
                f"no metric {name!r} for split {split!r}; available: {sorted(metrics)}"
            )
        return metrics[name]

    def headline(self, *, split: str = "test") -> Mapping[str, float]:
        """
        Return the metrics most runs are judged on.

        Parameters
        ----------
        split
            Split to report. Falls back to validation, then train, so a run
            configured without a test split still has a headline.

        Returns
        -------
        Mapping
            Metrics for the first available split, or empty if none were
            evaluated.
        """
        for candidate in (split, "validation", "train"):
            if candidate in self.evaluations:
                return self.evaluations[candidate].metrics
        return {}


class EvaluationResult(ContractModel):
    """
    The outcome of scoring a *saved* model, with the provenance to read it by.

    Distinct from :class:`TrainingResult` because the two answer different
    questions. A training result says what a run produced; an evaluation
    result says what was scored, from which bundle, against which data. The
    second half is not decoration. A metric detached from the source it was
    computed over cannot be compared to another metric with any confidence,
    and the comparison people most want to make -- "is this model still as
    good as it was?" -- is exactly the one that goes wrong when the data
    changed underneath without anybody recording it.

    Which is why :attr:`source_changed` is a field rather than something a
    reader is expected to work out. Re-scoring on new data is a legitimate
    and common thing to do; doing it while believing you reproduced an old
    number is not.

    Parameters
    ----------
    evaluations
        Metrics per split, keyed by split name.
    model_name
        The name the model is registered under, from the bundle's manifest.
    bundle_version
        Which version of the bundle was scored.
    spec_digest
        The digest of the specification the model was trained from.
    source_fingerprint
        Fingerprint of the data actually scored against.
    trained_on_fingerprint
        Fingerprint of the data the model was trained on.
    source_changed
        Whether the two fingerprints differ. Redundant by construction, and
        kept anyway: a reader scanning a result should not have to compare
        two hashes to learn whether the headline number means what they
        assume it means.
    notes
        Free-form annotations.
    """

    evaluations: Mapping[str, EvalResult] = Field(default_factory=dict)
    model_name: str = ""
    bundle_version: int = Field(default=1, ge=1)
    spec_digest: str = ""
    source_fingerprint: str = ""
    trained_on_fingerprint: str = ""
    source_changed: bool = False
    notes: Mapping[str, str] = Field(default_factory=dict)

    def metric(self, split: str, name: str) -> float:
        """
        Return one metric for one split.

        Parameters
        ----------
        split
            Split name.
        name
            Metric name.

        Returns
        -------
        float
            The metric value.

        Raises
        ------
        ContractError
            If the split or the metric is absent, with the available names
            listed.
        """
        if split not in self.evaluations:
            raise ContractError(
                f"no evaluation for split {split!r}; "
                f"evaluated splits are {sorted(self.evaluations)}"
            )
        metrics = self.evaluations[split].metrics
        if name not in metrics:
            raise ContractError(
                f"no metric {name!r} for split {split!r}; available: {sorted(metrics)}"
            )
        return metrics[name]

    def describe(self) -> str:
        """
        Return a one-line summary, for a log and a report header.

        Returns
        -------
        str
            Names the source change when there is one, in the same words
            every layer of this phase uses for it, so that the warning is
            recognisable wherever it surfaces.
        """
        scored = ", ".join(
            f"{split}={self.evaluations[split].metrics.get('mae', float('nan')):.6g}"
            for split in sorted(self.evaluations)
        )
        suffix = (
            ""
            if not self.source_changed
            else (
                f" [CHANGED SOURCE {self.trained_on_fingerprint[:8]} -> "
                f"{self.source_fingerprint[:8]}; not a reproduction]"
            )
        )
        return f"evaluated {self.model_name} v{self.bundle_version}: mae {scored}{suffix}"


class TrialRecord(ContractModel):
    """
    One trial of a search: what was proposed, and what came of it.

    A failed trial is a record rather than an absence, for the same reason a
    failed job is. A search that dropped its failures and reported the best
    of the rest would look exactly like a search in which everything
    succeeded -- and if eleven of forty failed, the conclusion drawn from
    the other twenty-nine is probably wrong, with nothing in the report to
    suggest it.

    Parameters
    ----------
    trial
        The trial number, which is also its position in the search.
    overrides
        The flat proposal, dotted path to value. Flat rather than nested
        because this is what a reader compares across trials, and a nested
        fragment has to be mentally flattened before two of them can be read
        side by side.
    status
        Whether it finished.
    objective
        The metric the search optimises, on the objective split. ``None``
        for a failed trial, and also for one that produced no such metric --
        which is a distinct situation from scoring badly, and must not be
        collapsed into a score of zero.
    metrics
        Every metric on the objective split, so the winner can be examined
        on more than the one number it was chosen by.
    bundle_directory
        Where the trial's model was written, relative to nothing -- an
        absolute path, because a search's trials live under the search and a
        caller holding only this record has no base to resolve against.
        Recorded so the winner can be *used* rather than retrained, which is
        the difference between acting on a search and repeating it.
    wall_seconds
        How long it took, successes and failures alike.
    failure_kind
        Exception class name, for a failed trial.
    failure_message
        Exception message, for a failed trial.
    """

    trial: int = Field(ge=0)
    overrides: Mapping[str, Any] = Field(default_factory=dict)
    status: Literal["succeeded", "failed"] = "succeeded"
    objective: float | None = None
    metrics: Mapping[str, float] = Field(default_factory=dict)
    bundle_directory: str | None = None
    wall_seconds: float = Field(default=0.0, ge=0.0)
    failure_kind: str | None = None
    failure_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """
        Return whether the trial finished.

        Returns
        -------
        bool
            True if it did.
        """
        return self.status == "succeeded"


class TuningResult(ContractModel):
    """
    Every trial of a search, and which one won.

    Parameters
    ----------
    trials
        The trials, in the order they ran.
    best_trial
        The winning trial number.
    objective
        The metric that was optimised.
    direction
        Whether it was minimised or maximised. Recorded because a column of
        objective values cannot be read without it -- and a reader who
        assumes the wrong direction concludes the opposite of the truth.
    objective_split
        Which split the objective was read from. Recorded for the same
        reason: a winner selected on test means something different from one
        selected on validation, and only one of them is a held-out estimate.
    name
        A label for the search.
    """

    trials: tuple[TrialRecord, ...] = ()
    best_trial: int = Field(default=0, ge=0)
    objective: str = "mae"
    direction: Literal["minimise", "maximise"] = "minimise"
    objective_split: str = "validation"
    name: str | None = None

    @property
    def best(self) -> TrialRecord:
        """
        Return the winning trial.

        Returns
        -------
        TrialRecord
            The winner.

        Raises
        ------
        ContractError
            If the search recorded no trials.
        """
        for record in self.trials:
            if record.trial == self.best_trial:
                return record
        raise ContractError(
            f"the search names trial {self.best_trial} as its best, but holds "
            f"{len(self.trials)} trial(s) and none with that number"
        )

    @property
    def failed(self) -> tuple[TrialRecord, ...]:
        """
        Return the trials that did not finish.

        Returns
        -------
        tuple of TrialRecord
            The failures, in trial order.
        """
        return tuple(record for record in self.trials if not record.succeeded)

    def describe(self) -> str:
        """
        Return a one-line summary, naming any failures.

        Returns
        -------
        str
            The winner, its objective, and how many trials failed -- which
            is said even when none did, because "0 failed" is information
            and a missing clause is not.
        """
        best = self.best
        return (
            f"best of {len(self.trials)} trial(s): #{best.trial} with "
            f"{self.objective}={best.objective:.6g} on {self.objective_split} "
            f"({self.direction}d); {len(self.failed)} failed"
        )


@dataclass(frozen=True, slots=True)
class Predictions:
    """
    Model output, with enough provenance to be acted on.

    A dataclass rather than a contract model because it carries arrays. Its
    metadata is small enough to be written beside it when predictions are
    persisted.

    Parameters
    ----------
    values
        Predicted values.
    in_original_units
        Whether the values are in the original target units. See the module
        docstring.
    entity_ids
        Identifier per prediction, where one exists. Without this a prediction
        cannot be attributed to a real instrument, which makes it unusable
        downstream however accurate it is.
    scenario_indices
        Scenario index per prediction.
    bundle_version
        Which bundle produced these predictions.
    provenance
        Additional annotations, such as the spec digest and the time of
        inference.

    Raises
    ------
    ContractError
        If the identifier or index lengths disagree with the values.
    """

    values: NDArray[np.floating]
    in_original_units: bool = True
    entity_ids: tuple[str, ...] | None = None
    scenario_indices: NDArray[np.int64] | None = None
    bundle_version: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that any per-prediction metadata aligns with the values."""
        n_predictions = self.values.shape[0] if self.values.ndim else 0
        if self.entity_ids is not None and len(self.entity_ids) != n_predictions:
            raise ContractError(
                f"entity_ids has {len(self.entity_ids)} entries but there are "
                f"{n_predictions} predictions"
            )
        if self.scenario_indices is not None and self.scenario_indices.shape[0] != n_predictions:
            raise ContractError(
                f"scenario_indices has {self.scenario_indices.shape[0]} entries but "
                f"there are {n_predictions} predictions"
            )

    @property
    def n_predictions(self) -> int:
        """Number of predictions."""
        return int(self.values.shape[0]) if self.values.ndim else 0
