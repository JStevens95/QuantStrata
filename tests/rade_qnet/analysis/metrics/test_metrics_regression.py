"""
Tests for the regression metrics.

Two of these tests matter more than the others.

Flattening: a prediction of shape ``(n, 1)`` against a target of shape
``(n,)`` must compare row-wise. Without the reshape, numpy broadcasts them to
``(n, n)`` and averages the error over every pairing, returning a plausible
number that is complete nonsense. It is one of the easiest mistakes to ship
and one of the hardest to notice.

No silent NaN handling: dropping missing rows changes the denominator, so a
model that failed to predict for half the universe can report a *better* mean
error than one that predicted for all of it, with nothing in the output
hinting that it happened.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.analysis.metrics.regression import (
    baseline_metrics,
    bias,
    directional_accuracy,
    mean_absolute_error,
    r_squared,
    regression_metrics,
    root_mean_squared_error,
)
from src.rade_qnet.core.lifecycle.errors import ContractError

ALL_METRICS = (
    mean_absolute_error,
    root_mean_squared_error,
    bias,
    r_squared,
    directional_accuracy,
)


class TestKnownValues:
    """Each metric, against a case worked out by hand."""

    def test_mean_absolute_error(self):
        """Errors of 1, 1 and 4 average to two."""
        predicted = np.array([1.0, 3.0, 6.0])
        observed = np.array([2.0, 2.0, 2.0])
        assert mean_absolute_error(predicted, observed) == pytest.approx(2.0)

    def test_root_mean_squared_error_penalises_large_misses(self):
        """
        Squared errors of 1, 1 and 16 give a root mean of about 2.45.

        Larger than the mean absolute error of 2.0, and the gap is the
        point: it says the errors are dominated by one large miss rather
        than spread evenly.
        """
        predicted = np.array([1.0, 3.0, 6.0])
        observed = np.array([2.0, 2.0, 2.0])
        assert root_mean_squared_error(predicted, observed) == pytest.approx(2.449489, abs=1e-5)

    def test_bias_is_signed(self):
        """
        Positive means over-prediction.

        A model can have a small absolute error and a large bias, which for
        a financial target is a materially different problem from being
        imprecise in both directions.
        """
        assert bias(np.array([3.0, 3.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)

    def test_offsetting_errors_cancel_in_the_bias(self):
        """
        Which is exactly why bias is reported beside the absolute error.

        Alone, a bias of zero here would suggest a perfect model.
        """
        predicted, observed = np.array([1.0, 3.0]), np.array([2.0, 2.0])
        assert bias(predicted, observed) == pytest.approx(0.0)
        assert mean_absolute_error(predicted, observed) == pytest.approx(1.0)

    def test_a_perfect_prediction_scores_perfectly(self):
        """The sanity check that catches a sign error in any of them."""
        values = np.array([1.0, -2.0, 3.0])
        assert mean_absolute_error(values, values) == 0.0
        assert root_mean_squared_error(values, values) == 0.0
        assert bias(values, values) == 0.0
        assert r_squared(values, values) == pytest.approx(1.0)


class TestRSquared:
    """The one metric whose edge cases carry real information."""

    def test_predicting_the_mean_scores_zero(self):
        """The definitional anchor."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.full_like(observed, observed.mean())
        assert r_squared(predicted, observed) == pytest.approx(0.0)

    def test_a_worse_than_mean_model_scores_negative(self):
        """
        Not clipped to zero, because the negative value is the diagnosis.

        "Worse than predicting the average" is precisely what a reader needs
        to be told, and clipping would hide it behind a zero that looks like
        a merely uninformative model.
        """
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([10.0, -10.0, 10.0])
        assert r_squared(predicted, observed) < 0.0

    def test_a_constant_target_returns_zero_rather_than_raising(self):
        """
        The statistic is undefined: the denominator is zero.

        A constant target legitimately occurs in a short held-out window, so
        raising would fail a valid run. Zero reads as "no explanatory
        power", which is the honest summary.
        """
        assert r_squared(np.array([1.0, 2.0]), np.array([5.0, 5.0])) == 0.0


class TestDirectionalAccuracy:
    """Sign agreement, with the zero-target case handled explicitly."""

    def test_all_signs_correct_scores_one(self):
        """Magnitude is irrelevant to this metric by design."""
        predicted = np.array([0.1, -5.0, 2.0])
        observed = np.array([9.0, -0.2, 3.0])
        assert directional_accuracy(predicted, observed) == pytest.approx(1.0)

    def test_all_signs_wrong_scores_zero(self):
        """The other end of the range."""
        assert directional_accuracy(np.array([1.0, -1.0]), np.array([-1.0, 1.0])) == 0.0

    def test_zero_targets_are_excluded_from_both_sides(self):
        """
        No direction was available to get right.

        Counting them as misses would penalise a model for a question it was
        never asked, and counting them as hits would inflate the score.
        """
        predicted = np.array([1.0, 1.0, 1.0])
        observed = np.array([1.0, 0.0, -1.0])
        assert directional_accuracy(predicted, observed) == pytest.approx(0.5)

    def test_an_all_zero_target_returns_zero(self):
        """
        Rather than dividing by an empty denominator.

        Which happens in a flat window, and should not crash a report.
        """
        assert directional_accuracy(np.array([1.0, 2.0]), np.zeros(2)) == 0.0


class TestShapeHandling:
    """The broadcasting trap, closed."""

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_a_column_vector_compares_row_wise(self, metric):
        """
        The mistake this guards against returns a plausible wrong number.

        Without the reshape, ``(n, 1)`` against ``(n,)`` broadcasts to
        ``(n, n)`` and averages the error over every pairing.
        """
        observed = np.array([1.0, 2.0, 3.0])
        assert metric(observed.reshape(-1, 1), observed) == pytest.approx(
            metric(observed, observed)
        )

    def test_the_broadcast_result_would_have_differed(self):
        """
        Demonstrating that the trap is real, not hypothetical.

        If flattening were a no-op here, this test would be pointless -- so
        it pins the fact that the naive computation gives a different
        answer.
        """
        predicted = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
        observed = np.array([3.0, 2.0, 1.0])
        broadcast = float(np.mean(np.abs(predicted - observed)))
        assert mean_absolute_error(predicted, observed) != pytest.approx(broadcast)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_mismatched_lengths_are_rejected(self, metric):
        """
        Checked after flattening, so the message is about real counts.

        A length mismatch means the predictions and the targets came from
        different rows, which no metric can paper over.
        """
        with pytest.raises(ContractError, match="different lengths"):
            metric(np.zeros(3), np.zeros(4))

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_an_empty_array_is_rejected(self, metric):
        """
        A mean over nothing is not zero, it is undefined.

        Returning zero would make an empty split look like a perfect one.
        """
        with pytest.raises(ContractError, match="empty"):
            metric(np.array([]), np.array([]))

    def test_integer_input_is_accepted(self):
        """
        A target read from a CSV may arrive as integers.

        Refusing it would push a cast into every caller.
        """
        assert mean_absolute_error(np.array([1, 2]), np.array([2, 2])) == pytest.approx(0.5)


class TestNonFiniteHandling:
    """Nothing is dropped, because dropping changes the denominator."""

    @pytest.mark.parametrize("metric", ALL_METRICS)
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_prediction_is_rejected(self, metric, bad):
        """
        Nothing is dropped, however convenient dropping would be.

        A model that failed to predict for half the universe would otherwise
        report a better mean error than one that predicted for all of it.
        """
        with pytest.raises(ContractError, match="non-finite"):
            metric(np.array([1.0, bad]), np.array([1.0, 1.0]))

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_a_non_finite_target_is_rejected(self, metric):
        """A missing observation is as disqualifying as a missing prediction."""
        with pytest.raises(ContractError, match="non-finite"):
            metric(np.array([1.0, 1.0]), np.array([1.0, np.nan]))

    def test_the_error_says_which_side_and_how_many(self):
        """
        So the reader knows whether to look at the model or the data.

        And whether it is one row or forty thousand.
        """
        with pytest.raises(ContractError, match="targets contain 2"):
            mean_absolute_error(np.ones(3), np.array([1.0, np.nan, np.nan]))


class TestTheStandardSet:
    """One call, so every run is comparable."""

    def test_every_metric_is_present(self):
        """
        Fixed keys, because a report and a catalog both index by them.

        A model choosing its own metric names would make two runs
        incomparable.
        """
        metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert set(metrics) == {"mae", "rmse", "bias", "r2", "directional_accuracy"}

    def test_every_value_is_a_plain_float(self):
        """
        Not a numpy scalar, which does not round-trip through JSON.

        A bundle holding ``np.float64`` would fail to serialise at the last
        step of a long run.
        """
        metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert all(type(value) is float for value in metrics.values())

    def test_the_values_match_the_individual_functions(self):
        """The convenience wrapper must not drift from what it wraps."""
        predicted, observed = np.array([1.0, 2.0]), np.array([1.5, 2.5])
        assert regression_metrics(predicted, observed)["mae"] == mean_absolute_error(
            predicted, observed
        )


class TestBaselines:
    """A reference point, and the leak it is careful not to introduce."""

    def test_the_zero_baseline_predicts_nothing(self):
        """
        The no-information forecast for a return-like target.

        Its mean absolute error is therefore the mean absolute target.
        """
        observed = np.array([1.0, -3.0])
        assert baseline_metrics(observed)["mae"] == pytest.approx(2.0)

    def test_the_mean_baseline_uses_the_training_mean(self):
        """
        Not the scored split's own mean.

        Using the latter would give the baseline information the model never
        had, making it an unbeatable and meaningless reference.
        """
        scored = np.array([10.0, 10.0])
        metrics = baseline_metrics(scored, strategy="mean", train_targets=np.array([0.0, 2.0]))
        assert metrics["mae"] == pytest.approx(9.0)

    def test_the_mean_baseline_requires_training_targets(self):
        """
        Rather than silently falling back to the scored split.

        The fallback is exactly the leak above, so it has to be an error.
        """
        with pytest.raises(ContractError, match="train_targets"):
            baseline_metrics(np.array([1.0]), strategy="mean")

    def test_an_empty_training_split_is_rejected(self):
        """A mean over nothing cannot serve as a baseline."""
        with pytest.raises(ContractError, match="training target"):
            baseline_metrics(np.array([1.0]), strategy="mean", train_targets=np.array([]))

    def test_an_unknown_strategy_lists_the_options(self):
        """A typo should not fall through to a default baseline."""
        with pytest.raises(ContractError, match="'zero' or 'mean'"):
            baseline_metrics(np.array([1.0]), strategy="median")

    def test_a_baseline_reports_the_same_keys_as_a_model(self):
        """
        Which is what makes the two comparable side by side.

        A baseline with different keys could not be rendered in the same
        table.
        """
        observed = np.array([1.0, -3.0])
        assert set(baseline_metrics(observed)) == set(
            regression_metrics(np.zeros_like(observed), observed)
        )
