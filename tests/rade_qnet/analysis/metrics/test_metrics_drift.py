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

from src.rade_qnet.analysis.metrics.drift import (
    DRIFT_THRESHOLDS,
    drift_metrics,
    drift_warnings,
    kolmogorov_smirnov,
    mean_shift,
    population_stability_index,
)
from src.rade_qnet.core.runtime.errors import ContractError

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
        metrics = drift_metrics(
            baseline.reshape(-1, 1), same.reshape(-1, 1), feature_names=("x",)
        )

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
