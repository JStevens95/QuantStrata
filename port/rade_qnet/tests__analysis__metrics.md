# `tranql/models/rade/rade_qnet/tests/analysis/metrics`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 21 | 846 | `815690298ea5fb56` |
| 2 | `test_metrics_drift.py` | 310 | 12289 | `2b6c0fa30b125934` |
| 3 | `test_metrics_quality.py` | 306 | 11841 | `0c70ea96252aee07` |
| 4 | `test_metrics_regression.py` | 331 | 12842 | `c512c2990d0fee32` |

---

## 1. `tranql/models/rade/rade_qnet/tests/analysis/metrics/__init__.py`

846 bytes · SHA-256 `815690298ea5fb56`

```python
"""
Tests for ``rade_qnet.analysis.metrics``.

Each metric is checked against a value computed by hand, not against a second
implementation. Comparing two implementations of the same formula only proves
they agree, which they will even when both are wrong.

Planned modules
---------------
``test_metrics_regression.py``
    Error and agreement measures, including the degenerate cases -- a constant
    target, a perfect prediction, a single observation -- which is where most
    metric implementations divide by zero.  [Phase 1]
``test_metrics_quality.py``
    Completeness, staleness and coverage.  [Phase 2]
``test_metrics_drift.py``
    Distribution distance, including the identical-distribution case that must
    report zero.  [Phase 5]
``test_metrics_episode.py``
    Episode return, length and risk-adjusted statistics.  [Phase 7]
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/analysis/metrics/test_metrics_drift.py`

12289 bytes · SHA-256 `2b6c0fa30b125934`

```python
"""
Tests for the drift measures.

Drift is a warning system, and a warning system has two ways to fail. It can
miss a real move, which is the failure everyone designs against. It can also
fire on nothing, which is the failure that actually breaks it: an alert that
cries wolf every morning gets filtered into a folder nobody opens, and then
it misses the real move too.

So the tests come in pairs. For each measure there is a test that it stays
quiet on two samples from the same distribution, and a test that it fires on
a genuine shift. The degenerate cases -- a constant baseline, an empty
sample, a single non-finite value -- get their own tests, because each one
has an arithmetic trap behind it that produces ``inf`` or ``NaN`` rather than
an error, and a ``NaN`` in a drift report reads as "no drift".
"""

from __future__ import annotations

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.analysis.metrics.drift import (
    DRIFT_THRESHOLDS,
    drift_metrics,
    drift_warnings,
    kolmogorov_smirnov,
    mean_shift,
    population_stability_index,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError

#: Large enough that sampling noise sits well below every threshold, so a
#: "quiet on identical distributions" test is asserting the measure rather
#: than the sample size.
N_SAMPLES = 4000


@pytest.fixture
def baseline():
    """
    Draw a reference sample.

    Returns
    -------
    numpy.ndarray
        Standard normal values.
    """
    return np.random.default_rng(0).normal(size=N_SAMPLES)


@pytest.fixture
def same(baseline):
    """
    Draw a second sample from the same distribution.

    A fresh draw rather than a copy. Comparing a sample against itself tests
    that the measure is zero at exactly one point; comparing it against an
    independent draw tests that it is *small* across sampling noise, which
    is the property a daily drift report depends on.

    Returns
    -------
    numpy.ndarray
        Another standard normal sample.
    """
    del baseline
    return np.random.default_rng(1).normal(size=N_SAMPLES)


class TestIdenticalDistributions:
    """The degenerate case, and the one a false alarm shows up in."""

    def test_drift_of_identical_distributions_is_zero(self, baseline):
        """Exactly zero, for every measure, against the sample itself."""
        assert population_stability_index(baseline, baseline) == 0.0
        assert kolmogorov_smirnov(baseline, baseline) == 0.0
        assert mean_shift(baseline, baseline) == 0.0

    def test_an_independent_draw_stays_below_every_threshold(self, baseline, same):
        """
        Which is the property a daily report depends on.

        A measure that fired on ordinary sampling noise would be filtered
        into a folder nobody opens, and would then miss the real move too.
        """
        assert population_stability_index(baseline, same) < DRIFT_THRESHOLDS["psi"]
        assert kolmogorov_smirnov(baseline, same) < DRIFT_THRESHOLDS["ks"]
        assert mean_shift(baseline, same) < DRIFT_THRESHOLDS["mean_shift"]

    def test_no_warning_is_raised_for_an_undrifted_feature(self, baseline, same):
        """The quiet path produces no text at all."""
        metrics = drift_metrics(baseline.reshape(-1, 1), same.reshape(-1, 1), feature_names=("x",))

        assert drift_warnings(metrics) == ()


class TestRealShifts:
    """Each measure catches what it is there to catch."""

    def test_a_location_shift_is_caught(self, baseline, same):
        """The easiest case, and the one all three should agree on."""
        moved = same + 3.0

        assert population_stability_index(baseline, moved) > DRIFT_THRESHOLDS["psi"]
        assert kolmogorov_smirnov(baseline, moved) > DRIFT_THRESHOLDS["ks"]
        assert mean_shift(baseline, moved) > DRIFT_THRESHOLDS["mean_shift"]

    def test_a_variance_shift_is_caught_without_a_mean_shift(self, baseline, same):
        """
        Which is why three measures are computed rather than one.

        A distribution that triples its spread while keeping its centre has
        moved enormously, and the readable measure -- how far the centre
        moved -- reports nothing at all.
        """
        wider = same * 3.0

        assert mean_shift(baseline, wider) < DRIFT_THRESHOLDS["mean_shift"]
        assert population_stability_index(baseline, wider) > DRIFT_THRESHOLDS["psi"]
        assert kolmogorov_smirnov(baseline, wider) > DRIFT_THRESHOLDS["ks"]

    def test_the_measures_grow_with_the_shift(self, baseline, same):
        """
        Monotonic, so the number means something rather than merely firing.

        A binary alarm would be satisfied by any threshold; a reader
        comparing today's 0.3 against last week's 0.9 needs the ordering to
        be real.
        """
        small = population_stability_index(baseline, same + 0.5)
        large = population_stability_index(baseline, same + 2.0)

        assert small < large

    def test_a_shift_beyond_the_training_range_is_not_dropped(self, baseline):
        """
        The most extreme evidence of drift must not fall outside the bins.

        Binning on interior edges only is what keeps a live value beyond
        anything seen in training inside the edge bin. Clipping it away
        would make the clearest case the one thing the measure cannot see.
        """
        far = np.full(N_SAMPLES, 50.0)

        assert population_stability_index(baseline, far) > DRIFT_THRESHOLDS["psi"]


class TestBinningOnTheBaseline:
    """Why the reference distribution fixes the scale."""

    def test_the_measure_is_not_symmetric_in_its_arguments(self, baseline, same):
        """
        Because one side is the reference and the other is not.

        A symmetric measure would be binning on the pooled data, which lets
        the live sample move the edges -- and that is how a large shift
        produces a small index: the edges follow the data, each bin keeps
        its share, and the measure reports that nothing happened.
        """
        shifted = same * 5.0
        forward = population_stability_index(baseline, shifted)
        backward = population_stability_index(shifted, baseline)

        assert forward != pytest.approx(backward)

    def test_the_spread_is_taken_from_the_baseline(self, baseline):
        """
        So a drifting input cannot flatter itself with its own variance.

        The live sample here is both shifted and far wider. Standardising
        by the pooled spread would divide the shift by the inflated
        standard deviation and report a small move.
        """
        wide_and_moved = np.random.default_rng(2).normal(size=N_SAMPLES) * 20.0 + 5.0

        assert mean_shift(baseline, wide_and_moved) > DRIFT_THRESHOLDS["mean_shift"]


class TestDegenerateInputs:
    """Each of these has an arithmetic trap that returns a number."""

    def test_an_empty_bin_does_not_make_the_index_infinite(self, baseline):
        """
        A floor rather than an infinity.

        Without it, one unoccupied bin makes the whole index infinite --
        both useless as a number and indistinguishable from a genuine
        catastrophe.
        """
        narrow = np.random.default_rng(3).normal(size=N_SAMPLES) * 0.01

        value = population_stability_index(baseline, narrow)
        assert np.isfinite(value)
        assert value > DRIFT_THRESHOLDS["psi"]

    def test_a_constant_baseline_that_did_not_move_is_quiet(self):
        """Rather than dividing by a zero spread."""
        constant = np.full(100, 2.0)

        assert population_stability_index(constant, np.full(50, 2.0)) == 0.0
        assert mean_shift(constant, np.full(50, 2.0)) == 0.0

    def test_a_constant_baseline_that_moved_is_total_drift(self):
        """
        Infinite, which is the honest answer rather than an error.

        An input that was always 2.0 and is now always 3.0 has moved by an
        unmeasurable number of its own standard deviations, because it had
        none.
        """
        constant = np.full(100, 2.0)

        assert population_stability_index(constant, np.full(50, 3.0)) == float("inf")
        assert mean_shift(constant, np.full(50, 3.0)) == float("inf")

    def test_non_finite_values_are_dropped_rather_than_propagated(self, baseline, same):
        """
        One ``NaN`` must not turn every measure into ``NaN``.

        Which would read as "no drift could be computed" in a report with
        no way to distinguish that from "no drift" -- and missingness is a
        quality concern, reported separately.
        """
        polluted = same.copy()
        polluted[0] = np.nan
        polluted[1] = np.inf

        assert np.isfinite(population_stability_index(baseline, polluted))
        assert np.isfinite(kolmogorov_smirnov(baseline, polluted))
        assert np.isfinite(mean_shift(baseline, polluted))

    def test_an_all_missing_sample_is_refused(self, baseline):
        """Rather than reporting zero drift against nothing."""
        with pytest.raises(ContractError, match=r"needs finite values on both sides"):
            population_stability_index(baseline, np.full(10, np.nan))

    def test_an_empty_sample_is_refused(self, baseline):
        """It has no distribution to compare."""
        with pytest.raises(ContractError, match=r"needs finite values on both sides"):
            kolmogorov_smirnov(baseline, np.array([]))


class TestPerFeature:
    """Drift is almost never uniform, so it is never averaged."""

    def test_only_the_drifted_feature_is_flagged(self):
        """
        One input going stale while the rest hold steady is the common case.

        An average over all of them would hide exactly that.
        """
        rng = np.random.default_rng(4)
        base = rng.normal(size=(N_SAMPLES, 3))
        live = rng.normal(size=(N_SAMPLES, 3))
        live[:, 1] += 4.0

        metrics = drift_metrics(base, live, feature_names=("a", "b", "c"))

        assert metrics["b"]["psi"] > DRIFT_THRESHOLDS["psi"]
        assert metrics["a"]["psi"] < DRIFT_THRESHOLDS["psi"]
        assert metrics["c"]["psi"] < DRIFT_THRESHOLDS["psi"]

    def test_the_warning_names_the_feature(self):
        """A report saying ``feature_7`` drifted sends the reader counting."""
        rng = np.random.default_rng(5)
        base = rng.normal(size=(N_SAMPLES, 2))
        live = rng.normal(size=(N_SAMPLES, 2))
        live[:, 0] += 4.0

        warnings = drift_warnings(drift_metrics(base, live, feature_names=("spread", "vol")))

        assert len(warnings) == 1
        assert warnings[0].startswith("spread has drifted")

    def test_warnings_are_ordered_worst_first(self):
        """So a truncated list keeps the part worth keeping."""
        rng = np.random.default_rng(6)
        base = rng.normal(size=(N_SAMPLES, 2))
        live = rng.normal(size=(N_SAMPLES, 2))
        live[:, 0] += 1.5
        live[:, 1] += 6.0

        warnings = drift_warnings(drift_metrics(base, live, feature_names=("mild", "severe")))

        assert len(warnings) == 2
        assert warnings[0].startswith("severe")

    def test_a_column_count_mismatch_is_refused(self):
        """Rather than comparing column three against an unrelated column three."""
        with pytest.raises(ContractError, match=r"drift is computed column by column"):
            drift_metrics(np.zeros((10, 3)), np.zeros((10, 2)))

    def test_a_name_count_mismatch_is_refused(self):
        """Otherwise every label after the first mistake is wrong."""
        with pytest.raises(ContractError, match=r"feature name"):
            drift_metrics(np.zeros((10, 3)), np.zeros((10, 3)), feature_names=("a", "b"))

    def test_unnamed_features_get_positional_labels(self):
        """Legible, though not actionable -- which the docstring admits."""
        metrics = drift_metrics(np.zeros((10, 2)), np.zeros((10, 2)))

        assert set(metrics) == {"feature_0", "feature_1"}

    def test_a_single_feature_may_be_passed_as_a_vector(self):
        """So the common case does not need a reshape at every call site."""
        rng = np.random.default_rng(7)
        metrics = drift_metrics(rng.normal(size=200), rng.normal(size=200))

        assert set(metrics) == {"feature_0"}
```

---

## 3. `tranql/models/rade/rade_qnet/tests/analysis/metrics/test_metrics_quality.py`

11841 bytes · SHA-256 `0c70ea96252aee07`

```python
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

from tranql.models.rade.rade_qnet.rade_qnet.analysis.metrics.quality import (
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
```

---

## 4. `tranql/models/rade/rade_qnet/tests/analysis/metrics/test_metrics_regression.py`

12842 bytes · SHA-256 `c512c2990d0fee32`

```python
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

from tranql.models.rade.rade_qnet.rade_qnet.analysis.metrics.regression import (
    baseline_metrics,
    bias,
    directional_accuracy,
    mean_absolute_error,
    r_squared,
    regression_metrics,
    root_mean_squared_error,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError

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
```

