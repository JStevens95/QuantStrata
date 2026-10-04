"""
Tests for the result contracts.

The field doing the most work here is ``in_original_units``. A mean absolute
error of 0.03 is either excellent or meaningless depending on that one
boolean, and the number alone does not say which -- so it travels with the
number rather than being remembered.

The rest of this module is about rejecting results that are wrong in ways that
still render: a ``best_epoch`` that indexes nothing, a ``NaN`` metric, a
prediction array longer than its entity identifiers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from src.rade_qnet.core.contract.result import (
    EpochRecord,
    EvalResult,
    FitOutcome,
    Predictions,
    TrainingResult,
)
from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.testkit.fixtures import make_training_result


def _history(*losses, val=True):
    """
    Build an epoch history from a sequence of training losses.

    Parameters
    ----------
    losses
        One training loss per epoch.
    val
        Whether to record a validation loss alongside each.

    Returns
    -------
    tuple of EpochRecord
        The history.
    """
    return tuple(
        EpochRecord(epoch=index, train_loss=loss, val_loss=loss + 0.1 if val else None)
        for index, loss in enumerate(losses)
    )


class TestEpochRecord:
    """One epoch, recorded in enough detail to diagnose a run."""

    def test_a_minimal_record_is_accepted(self):
        """Training loss is the only thing every engine has."""
        assert EpochRecord(epoch=0, train_loss=1.0).val_loss is None

    def test_a_negative_epoch_index_is_rejected(self):
        """Epochs are zero-based and count upward."""
        with pytest.raises(ValidationError):
            EpochRecord(epoch=-1, train_loss=1.0)

    def test_the_learning_rate_is_recorded(self):
        """
        So a schedule's behaviour is visible in the saved history.

        Otherwise it has to be inferred from the shape of the loss curve,
        which is guesswork.
        """
        assert EpochRecord(epoch=0, train_loss=1.0, learning_rate=1e-3).learning_rate == 1e-3

    def test_a_zero_learning_rate_is_rejected(self):
        """
        A rate of zero means no epoch after this one changes anything.

        Recording it without comment would make a frozen run look like a
        converged one.
        """
        with pytest.raises(ValidationError):
            EpochRecord(epoch=0, train_loss=1.0, learning_rate=0.0)

    def test_negative_wall_time_is_rejected(self):
        """Not a quantity that can be negative."""
        with pytest.raises(ValidationError):
            EpochRecord(epoch=0, train_loss=1.0, seconds=-1.0)


class TestFitOutcome:
    """One type for a gradient loop and a boosting run alike."""

    def test_an_empty_history_is_accepted(self):
        """
        A fit that has not run yet is representable.

        Which is what lets a pipeline construct the outcome before the loop
        rather than assembling it from fragments afterwards.
        """
        assert FitOutcome().n_epochs == 0

    def test_the_best_record_is_resolved_by_epoch_number(self):
        """
        Looked up by ``epoch``, not by position in the tuple.

        An engine that records every other epoch would otherwise return the
        wrong record.
        """
        outcome = FitOutcome(
            history=(
                EpochRecord(epoch=0, train_loss=1.0),
                EpochRecord(epoch=2, train_loss=0.5),
            ),
            best_epoch=2,
        )
        assert outcome.best_record.train_loss == 0.5

    def test_a_best_epoch_absent_from_the_history_is_rejected(self):
        """
        Catches an off-by-one between a callback and the history.

        Which would otherwise surface as the wrong checkpoint being restored
        -- a run that reports epoch 40's metrics with epoch 41's weights.
        """
        with pytest.raises(ValidationError, match="best_epoch"):
            FitOutcome(history=_history(1.0, 0.5), best_epoch=7)

    def test_no_best_epoch_is_permitted(self):
        """A run that monitored nothing has no best epoch."""
        assert FitOutcome(history=_history(1.0)).best_record is None

    def test_restoring_the_best_epoch_is_recorded_not_assumed(self):
        """
        When false, the reported metrics and the saved weights disagree.

        That is a legitimate configuration, but it has to be visible in the
        bundle or the two will be compared as though they described the same
        model.
        """
        assert FitOutcome().restored_best is False

    def test_a_curve_preserves_missing_validation_losses(self):
        """
        ``None`` rather than zero, because they mean different things.

        A gap in a curve and a loss of zero look nothing alike to a reader
        and nothing alike to a plotting function.
        """
        outcome = FitOutcome(history=_history(1.0, 0.5, val=False))
        assert outcome.curve("val_loss") == (None, None)

    def test_a_curve_returns_one_value_per_record(self):
        """The series a report plots."""
        assert FitOutcome(history=_history(1.0, 0.8, 0.5)).curve("train_loss") == (1.0, 0.8, 0.5)

    def test_it_round_trips_through_json(self):
        """Written into a bundle, so it must reload exactly."""
        outcome = FitOutcome(history=_history(1.0, 0.5), best_epoch=1, stopped_early=True)
        assert FitOutcome.model_validate_json(outcome.model_dump_json()) == outcome


class TestEvalResult:
    """Metrics, with the two things that make them interpretable."""

    def test_metrics_are_assumed_to_be_in_original_units(self):
        """
        The safe default is the one that is true for most models.

        A model that transforms its target has to say so, which is the right
        way round: silence should not mean "these numbers are in a space
        nobody can interpret".
        """
        assert EvalResult(split="test", metrics={"mae": 0.1}, n_samples=10).in_original_units

    def test_model_internal_units_are_expressible(self):
        """
        Because sometimes they genuinely are internal.

        Hiding that would be worse than reporting it.
        """
        result = EvalResult(
            split="test", metrics={"mae": 0.1}, n_samples=10, in_original_units=False
        )
        assert not result.in_original_units

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_metric_is_rejected(self, value):
        """
        Almost always a degenerate split or a divide-by-zero in a metric.

        Far cheaper to diagnose here than after it has been written to a
        bundle and compared against other runs.
        """
        with pytest.raises(ValidationError, match="finite"):
            EvalResult(split="test", metrics={"mae": value}, n_samples=10)

    def test_the_offending_metric_is_named(self):
        """So the reader does not have to inspect every metric."""
        with pytest.raises(ValidationError, match="r2"):
            EvalResult(
                split="test",
                metrics={"mae": 0.1, "r2": float("nan")},
                n_samples=10,
            )

    def test_baseline_metrics_travel_with_the_result(self):
        """
        A headline metric without a reference point is not interpretable.

        Carrying the baseline alongside means a report cannot show one
        without the other.
        """
        result = EvalResult(
            split="test",
            metrics={"mae": 0.1},
            n_samples=10,
            baseline_metrics={"mae": 0.5},
        )
        assert result.baseline_metrics["mae"] == 0.5

    def test_an_empty_split_is_representable(self):
        """Zero samples is a fact to report, not an error to raise."""
        assert EvalResult(split="test", metrics={}, n_samples=0).n_samples == 0


class TestTrainingResult:
    """The complete outcome, and its lookups."""

    def test_a_metric_is_retrieved_by_split_and_name(self):
        """The normal access path."""
        result = make_training_result()
        assert isinstance(result.metric("test", "mae"), float)

    def test_an_unevaluated_split_lists_what_was_evaluated(self):
        """
        The usual cause is a run configured without that split.

        Listing the evaluated splits answers the question immediately.
        """
        with pytest.raises(ContractError, match="evaluated splits"):
            make_training_result().metric("holdout", "mae")

    def test_an_unknown_metric_lists_what_is_available(self):
        """
        Metric names vary by task.

        A caller asking for ``accuracy`` on a regression run should see what
        it could have asked for instead.
        """
        with pytest.raises(ContractError, match="available"):
            make_training_result().metric("test", "accuracy")

    def test_the_headline_prefers_the_test_split(self):
        """What a run is normally judged on."""
        result = make_training_result()
        assert result.headline() == result.evaluations["test"].metrics

    def test_the_headline_falls_back_when_there_is_no_test_split(self):
        """
        A run configured without held-out test data still has a headline.

        Returning nothing would make the summary page blank for a legitimate
        configuration.
        """
        result = make_training_result()
        without_test = result.model_copy(
            update={"evaluations": {"validation": result.evaluations["validation"]}}
        )
        assert without_test.headline() == result.evaluations["validation"].metrics

    def test_the_headline_is_empty_when_nothing_was_evaluated(self):
        """An empty mapping rather than an exception, since this is a report path."""
        assert TrainingResult(fit=FitOutcome()).headline() == {}

    def test_it_round_trips_through_json(self):
        """The whole result is written into a bundle."""
        result = make_training_result()
        assert TrainingResult.model_validate_json(result.model_dump_json()) == result


class TestPredictions:
    """Output that can be acted on, which means output that can be attributed."""

    def test_values_alone_are_sufficient(self):
        """Not every problem has entity identifiers."""
        assert Predictions(values=np.zeros(5)).n_predictions == 5

    def test_misaligned_entity_ids_are_rejected(self):
        """
        A prediction attributed to the wrong instrument is worse than none.

        It is actionable and wrong, which is the most expensive combination.
        """
        with pytest.raises(ContractError, match="entity_ids"):
            Predictions(values=np.zeros(5), entity_ids=("EURUSD",))

    def test_misaligned_scenario_indices_are_rejected(self):
        """The same failure along the time axis."""
        with pytest.raises(ContractError, match="scenario_indices"):
            Predictions(values=np.zeros(5), scenario_indices=np.array([0, 1], dtype=np.int64))

    def test_aligned_metadata_is_accepted(self):
        """Both axes, correctly supplied."""
        predictions = Predictions(
            values=np.zeros(2),
            entity_ids=("EURUSD", "USDJPY"),
            scenario_indices=np.array([10, 11], dtype=np.int64),
        )
        assert predictions.n_predictions == 2

    def test_the_originating_bundle_is_recorded(self):
        """
        So a prediction can be traced to the model that made it.

        Without it, a prediction file found six months later is unattributable.
        """
        assert Predictions(values=np.zeros(1), bundle_version="demo/job/v3").bundle_version == (
            "demo/job/v3"
        )

    def test_predictions_are_frozen(self):
        """Output must not be edited in place by a downstream consumer."""
        with pytest.raises(AttributeError):
            Predictions(values=np.zeros(1)).values = np.ones(1)

    def test_a_non_finite_prediction_is_permitted(self):
        """
        Unlike a metric, which is rejected.

        A model that diverged genuinely produced ``NaN``, and hiding that
        would be a lie. A metric computed *from* such output, on the other
        hand, is meaningless and is rejected where it is constructed.
        """
        assert math.isnan(float(Predictions(values=np.array([np.nan])).values[0]))
