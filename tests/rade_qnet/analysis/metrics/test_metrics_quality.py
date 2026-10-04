"""
Tests for the data-quality metrics.

These metrics exist because "the model got worse" and "the data got worse" are
the two explanations for a drop in performance, and only one of them is a
modelling problem. Without them, a feed that quietly started forward-filling a
price looks exactly like a model that stopped generalising.

The design decisions worth pinning down are the degenerate cases, because each
one has a plausible wrong answer that is actively misleading:

An empty input is *complete*, not empty-and-therefore-zero. Nothing is missing
from nothing, and reporting zero would make an empty split look like the worst
data in the run.

An infinity counts as missing. It is not a measurement, and counting it as
present reports a column as complete while every mean computed from it is also
infinite.

A pair of NaNs is not a repeat. Missingness is completeness's business, and
counting a gap as staleness too would penalise the same gap twice.

And staleness is order-dependent by construction, so it must be computed on a
source whose order is stable -- which is why the pipeline routes it through the
ordered view rather than the training source.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.analysis.metrics.quality import (
    column_completeness,
    completeness,
    entity_coverage,
    quality_metrics,
    quality_warnings,
    staleness,
)


class TestCompleteness:
    """The fraction of entries that are present and usable."""

    def test_a_full_matrix_is_completely_complete(self):
        """The baseline, so the scale has a known top."""
        assert completeness(np.ones((4, 3))) == 1.0

    def test_missing_entries_reduce_the_fraction(self):
        """
        Counted over entries rather than rows.

        Per-row would conflate "one column is always absent" with "every
        column is occasionally absent", and those need different fixes.
        """
        values = np.ones((2, 2))
        values[0, 0] = np.nan
        assert completeness(values) == pytest.approx(0.75)

    def test_an_infinity_counts_as_missing(self):
        """
        Because it is not a measurement.

        Counted as present, a column holding one infinity reports as complete
        while every mean computed from it is also infinite -- so the metric
        would be saying the data is fine at exactly the moment it is not.
        """
        values = np.ones((2, 2))
        values[0, 0] = np.inf
        assert completeness(values) == pytest.approx(0.75)

    def test_an_empty_input_is_complete(self):
        """
        Because nothing is missing from nothing.

        Zero would make an empty split read as the worst data in the run, and
        a report would then lead with a quality alarm about a split that has
        no rows -- which is a different problem, reported elsewhere.
        """
        assert completeness(np.zeros((0, 3))) == 1.0

    def test_a_flat_series_is_accepted(self):
        """So the same function scores the target and the feature matrix."""
        assert completeness(np.array([1.0, np.nan, 3.0, 4.0])) == pytest.approx(0.75)


class TestColumnCompleteness:
    """Per column, because the overall figure hides the shape of the gap."""

    def test_one_fraction_per_column(self):
        """So a report can name the offending column rather than the matrix."""
        assert column_completeness(np.ones((5, 3))).shape == (3,)

    def test_an_entirely_absent_column_is_identified(self):
        """
        The case the overall figure buries.

        Ninety per cent overall is unremarkable spread evenly and serious when
        one column in ten is entirely absent, and only the per-column figures
        distinguish the two.
        """
        values = np.ones((10, 2))
        values[:, 1] = np.nan
        assert column_completeness(values).tolist() == [1.0, 0.0]

    def test_an_empty_matrix_reports_every_column_complete(self):
        """Consistent with the overall figure, for the same reason."""
        assert column_completeness(np.zeros((0, 3))).tolist() == [1.0, 1.0, 1.0]


class TestStaleness:
    """Consecutive repeats, which is how a frozen feed shows up."""

    def test_a_varying_series_is_not_stale(self):
        """The baseline, so a healthy feed scores zero."""
        assert staleness(np.arange(10.0).reshape(-1, 1)) == 0.0

    def test_a_constant_series_is_entirely_stale(self):
        """
        Which is what a feed that stopped updating looks like.

        A forward-filled price is indistinguishable from a real one in every
        other metric: it is present, finite, and in range.
        """
        assert staleness(np.ones((10, 1))) == 1.0

    def test_the_fraction_is_over_consecutive_pairs(self):
        """
        Nine pairs across ten rows, so a short series is not over-penalised.

        Dividing by the row count instead would cap the metric below one and
        make "entirely stale" unreachable.
        """
        values = np.array([[1.0], [1.0], [2.0], [3.0]])
        assert staleness(values) == pytest.approx(1 / 3)

    def test_a_pair_of_gaps_is_not_counted_as_a_repeat(self):
        """
        Because missingness is completeness's business.

        Counting a gap as staleness as well penalises the same gap twice, and
        a column that is entirely absent would then report as entirely stale
        -- which points at the feed rather than at the gap.
        """
        values = np.array([[np.nan], [np.nan], [np.nan]])
        assert staleness(values) == 0.0

    def test_a_single_row_is_not_stale(self):
        """
        Because consecutive repetition is undefined with nothing to repeat.

        One is the obvious wrong answer here and it would make every
        single-row split report a frozen feed.
        """
        assert staleness(np.ones((1, 3))) == 0.0

    def test_staleness_depends_on_the_order(self):
        """
        Which is why it must be computed on a stable source.

        A shuffled source gives a different answer every pass, so the metric
        is routed through the ordered view -- the same mechanism the scoring
        alignment needs, for the same underlying reason.
        """
        ordered = np.array([[1.0], [1.0], [2.0], [2.0]])
        shuffled = np.array([[1.0], [2.0], [1.0], [2.0]])
        assert staleness(ordered) != staleness(shuffled)


class TestEntityCoverage:
    """How much of the universe the data actually describes."""

    def test_full_coverage_of_the_universe(self):
        """The baseline for a multi-entity run."""
        assert entity_coverage(("EUR", "USD"), universe=("EUR", "USD")) == 1.0

    def test_a_missing_entity_reduces_the_coverage(self):
        """
        Which is the multi-cluster failure that has no other symptom.

        A portfolio run that quietly dropped one instrument produces a
        complete-looking result for every instrument it did cover, and
        nothing else in the output counts them.
        """
        assert entity_coverage(("EUR",), universe=("EUR", "USD")) == pytest.approx(0.5)

    def test_extra_entities_do_not_inflate_the_coverage(self):
        """
        Because coverage is of the universe, not of the data.

        Intersecting rather than counting means a run carrying an instrument
        nobody asked for cannot report above full coverage.
        """
        assert entity_coverage(("EUR", "USD", "JPY"), universe=("EUR", "USD")) == 1.0

    def test_no_universe_means_full_coverage(self):
        """
        Because there is nothing to fall short of.

        Zero would make every tabular problem -- which has no entity axis at
        all -- report a coverage alarm it cannot act on.
        """
        assert entity_coverage(("EUR",), universe=None) == 1.0

    def test_no_entities_and_no_universe_is_still_full(self):
        """The tabular case, stated directly."""
        assert entity_coverage(None) == 1.0


class TestQualityMetrics:
    """The whole set, which is what gets persisted into the lineage."""

    def test_every_value_is_a_finite_fraction(self):
        """
        Because they are compared across runs and written into a report.

        A metric that can exceed one or be NaN cannot be thresholded, and a
        threshold is the only thing that turns a number into an alert.
        """
        rng = np.random.default_rng(0)
        metrics = quality_metrics(rng.normal(size=(50, 4)))
        assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics.values())

    def test_the_target_is_scored_separately_from_the_features(self):
        """
        Because a gap in the target removes a training example entirely.

        A gap in one feature of one row does not: the row still trains, just
        with one input imputed. Averaging the two together would hide the
        difference.
        """
        rng = np.random.default_rng(1)
        metrics = quality_metrics(rng.normal(size=(20, 3)), target=np.array([np.nan, *np.ones(19)]))
        assert metrics["target_completeness"] < 1.0
        assert metrics["feature_completeness"] == 1.0

    def test_the_worst_column_is_reported_alongside_the_overall_figure(self):
        """
        So one absent column cannot hide behind a healthy average.

        This is the single most useful number in the set, because it is the
        one that names something actionable.
        """
        values = np.ones((10, 4))
        values[:, 2] = np.nan
        metrics = quality_metrics(values)
        assert metrics["worst_column_completeness"] == 0.0
        assert metrics["feature_completeness"] == pytest.approx(0.75)

    def test_target_metrics_are_absent_when_no_target_is_given(self):
        """
        Rather than defaulting to one.

        A default would claim a perfect target in an unsupervised run, and a
        comparison against a supervised run would then show no difference
        where there is nothing to compare.
        """
        metrics = quality_metrics(np.ones((5, 2)))
        assert "target_completeness" not in metrics


class TestQualityWarnings:
    """Thresholds turned into sentences, which is what a report prints."""

    def test_clean_data_produces_no_warnings(self):
        """
        So a report with no quality section means there was nothing to say.

        A warning that always fires is a warning nobody reads.
        """
        assert quality_warnings(quality_metrics(np.ones((10, 3)) * np.arange(10)[:, None])) == ()

    def test_poor_completeness_is_warned_about(self):
        """Because it is the condition most likely to invalidate a result."""
        values = np.ones((10, 4))
        values[:, :3] = np.nan
        assert quality_warnings(quality_metrics(values))

    def test_a_frozen_feed_is_warned_about(self):
        """
        Which no other metric catches.

        Every value is present, finite and plausible; only the consecutive
        repetition says the feed stopped moving.
        """
        assert quality_warnings(quality_metrics(np.ones((20, 3))))

    def test_the_warnings_are_sentences_rather_than_codes(self):
        """
        Because the report writes them verbatim for a person to read.

        A code would need a lookup table maintained alongside it, and the
        table would drift from the thresholds.
        """
        warnings = quality_warnings(quality_metrics(np.ones((20, 3))))
        assert all(len(warning.split()) > 3 for warning in warnings)

    def test_an_absent_metric_is_not_warned_about(self):
        """
        So an unsupervised run produces no target warnings.

        Treating absence as zero would make every run without a target report
        a completely missing one.
        """
        assert quality_warnings({"feature_completeness": 1.0}) == ()
