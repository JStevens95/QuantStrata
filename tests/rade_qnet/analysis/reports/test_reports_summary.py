"""
Tests for the default summary report.

This is the only report enabled by default, so it has to answer the questions
someone actually asks when they open a run directory: what was trained, from
what configuration, against what data, how did it do, and is that any better
than doing nothing.

Two things here are not cosmetic. The page must flag metrics that are *not* in
original target units, because a mean absolute error of 0.03 is excellent or
meaningless depending on that one fact. And the page must say whether the best
weights were restored, because when they were not, the reported metrics and
the saved weights describe different models.

The report also has to survive degenerate runs -- a single epoch, no
evaluation, no validation split -- since reports are never load-bearing and a
summary that raised would turn one failure into two.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.rade_qnet.analysis.reports.base import ReportContext
from src.rade_qnet.analysis.reports.summary import SUMMARY_FILENAME, SummaryReport
from src.rade_qnet.core.contract.result import EvalResult, FitOutcome, TrainingResult
from src.rade_qnet.core.runtime.components import get_report
from src.rade_qnet.testkit.fixtures import make_model_bundle, make_training_result


def _render(tmp_path, bundle=None, **context_fields):
    """
    Render the summary and return the page text.

    Parameters
    ----------
    tmp_path
        Directory to render into.
    bundle
        The bundle to summarise, defaulting to the standard fixture.
    context_fields
        Overrides for the report context.

    Returns
    -------
    tuple of (str, ReportOutcome)
        The page text and the outcome.
    """
    context = ReportContext(
        bundle=bundle if bundle is not None else make_model_bundle(with_manifest=True),
        directory=tmp_path,
        **context_fields,
    )
    outcome = SummaryReport().render_safely(context)
    assert outcome.succeeded, outcome.skipped_reason
    return (tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"), outcome


@pytest.fixture
def page(tmp_path):
    """
    Provide the rendered page for a complete run.

    Returns
    -------
    str
        The Markdown.
    """
    return _render(tmp_path)[0]


class TestRegistration:
    """Reached by the name a specification uses."""

    def test_it_is_registered_as_summary(self):
        """
        Which is the name the default report specification carries.

        A mismatch would mean the default run produces no report at all.
        """
        assert get_report("summary") is SummaryReport


class TestStructure:
    """The questions the page has to answer."""

    @pytest.mark.parametrize(
        "section", ["## Configuration", "## Data", "## Training", "## Metrics"]
    )
    def test_each_section_is_present(self, page, section):
        """
        One section per question a reader opens the directory with.

        A page missing one of these sends them back to the raw JSON.
        """
        assert section in page

    def test_the_page_is_titled_with_the_bundle_identifier(self, page):
        """
        So a page copied out of its directory is still attributable.

        Which happens constantly: summaries get pasted into tickets.
        """
        assert "# Run summary: synthetic/v1" in page

    def test_an_unwritten_bundle_is_labelled_rather_than_crashing(self, tmp_path):
        """
        A report may be rendered before the bundle is persisted.

        Raising on the missing manifest would make the summary unavailable
        in exactly the case where someone is debugging a failing write.
        """
        text, _ = _render(tmp_path, bundle=make_model_bundle())
        assert "(unwritten)" in text

    def test_the_page_ends_with_a_single_newline(self, page):
        """
        So two runs' pages diff cleanly.

        Trailing-whitespace noise would swamp the real differences.
        """
        assert page.endswith("\n")
        assert not page.endswith("\n\n")


class TestConfigurationSection:
    """What was run, in terms the reader typed."""

    def test_the_model_is_described_readably(self, page):
        """
        Not as a pydantic repr.

        ``name='demo' params={}`` leaking into a summary page reads as a
        debugging artefact, which is exactly how it was first found.
        """
        assert "params={}" not in page
        assert "name=" not in page

    def test_the_applied_seed_is_recorded(self, page):
        """
        The seed actually used, not the one requested.

        A run that derived its seed from a job identifier must report the
        derived value, or the run cannot be repeated.
        """
        assert "seed applied" in page

    def test_the_spec_digest_is_recorded(self, page):
        """
        The configuration is identified, not merely described.

        So two runs claiming the same configuration can be shown to have had
        it, rather than taken at their word.
        """
        assert "spec digest" in page


class TestDataSection:
    """Where the data came from, without pasting the data in."""

    def test_split_sizes_are_reported(self, page):
        """
        Sizes rather than indices.

        The indices are in the bundle for reproducibility; ten thousand
        integers in a summary page would make the page useless.
        """
        assert "train scenarios" in page
        assert "[" not in page.split("## Training")[0].split("## Data")[1]

    def test_the_source_fingerprint_is_reported(self, page):
        """Which is what makes a stale cache entry detectable."""
        assert "source fingerprint" in page

    def test_notes_are_surfaced(self, tmp_path):
        """
        A reward-shaping change or a parity flag must be visible.

        Otherwise two runs that are not comparable look comparable.
        """
        bundle = make_model_bundle()
        noted = replace(
            bundle,
            lineage=bundle.lineage.model_copy(update={"notes": {"reward_shaping": "v2"}}),
        )
        text, _ = _render(tmp_path, bundle=noted)
        assert "reward_shaping" in text


class TestTrainingSection:
    """How the fit went, including the part everyone forgets."""

    def test_whether_the_best_weights_were_restored_is_stated(self, page):
        """
        The thing a reader must know before comparing two runs.

        When it is "no", the reported metrics and the saved weights
        describe different models.
        """
        assert "best weights restored" in page

    def test_early_stopping_is_stated(self, page):
        """
        An epoch count alone does not say whether the budget was spent.

        "Fifty epochs" and "stopped at fifty of two hundred" are different
        runs with the same epoch count.
        """
        assert "stopped early" in page

    def test_the_monitored_metric_is_named(self, page):
        """
        "Best epoch 40" means nothing without knowing best by what.

        And the answer differs between a run with a validation split and
        one without.
        """
        assert "monitored metric" in page


class TestMetricsSection:
    """Numbers, with the two things that make them interpretable."""

    def test_each_evaluated_split_gets_its_own_table(self, page):
        """So validation and test are not silently conflated."""
        assert "### validation" in page
        assert "### test" in page

    def test_baselines_appear_beside_the_model(self, page):
        """
        A headline metric without a reference point is not interpretable.

        0.004 is excellent or useless depending entirely on the target's
        scale.
        """
        assert "baseline" in page

    def test_a_missing_baseline_renders_as_a_dash(self, tmp_path):
        """
        Rather than as a blank cell or a zero.

        A zero would read as a baseline that scored perfectly, which is the
        opposite of the truth.
        """
        result = TrainingResult(
            fit=FitOutcome(),
            evaluations={"test": EvalResult(split="test", metrics={"mae": 0.1}, n_samples=5)},
        )
        text, _ = _render(tmp_path, bundle=make_model_bundle(result=result))
        assert "| mae | 0.1 | - |" in text

    def test_model_internal_units_are_flagged_prominently(self, tmp_path):
        """
        The single most important warning this page can carry.

        A mean absolute error of 0.03 in standardised space is a number
        nobody can act on, and it looks perfectly reasonable.
        """
        result = TrainingResult(
            fit=FitOutcome(),
            evaluations={
                "test": EvalResult(
                    split="test",
                    metrics={"mae": 0.03},
                    n_samples=5,
                    in_original_units=False,
                )
            },
        )
        text, _ = _render(tmp_path, bundle=make_model_bundle(result=result))
        assert "not comparable across runs" in text

    def test_original_units_are_stated_too(self, page):
        """
        Silence would be ambiguous.

        A reader should not have to know that the absence of a warning is
        itself the reassurance.
        """
        assert "original target units" in page

    def test_a_run_with_no_evaluation_says_so(self, tmp_path):
        """
        Rather than omitting the section.

        A missing section reads as a framework bug; a sentence reads as a
        fact about the run.
        """
        result = TrainingResult(fit=FitOutcome())
        text, _ = _render(tmp_path, bundle=make_model_bundle(result=result))
        assert "No evaluation was recorded" in text

    def test_the_sample_count_is_reported_per_split(self, page):
        """
        So a metric computed over nine rows is not read as a verdict.

        Which is a real risk for a job set member with little history.
        """
        assert "samples" in page


class TestFigures:
    """Written beside the page, and linked relatively."""

    def test_the_training_curve_is_written(self, tmp_path):
        """The figure every run gets, for every engine."""
        _render(tmp_path)
        assert (tmp_path / "training_curve.png").is_file()

    def test_a_metric_figure_is_written_per_split(self, tmp_path):
        """Matching the per-split tables."""
        _render(tmp_path)
        assert (tmp_path / "metrics_test.png").is_file()
        assert (tmp_path / "metrics_validation.png").is_file()

    def test_figures_are_linked_relatively(self, page):
        """
        So the directory can be moved or archived and still render.

        An absolute path from a build machine is a dead link everywhere
        else.
        """
        assert "![training curve](training_curve.png)" in page

    def test_the_figure_format_follows_the_specification(self, tmp_path):
        """One run's figures should not mix formats."""
        _render(tmp_path, figure_format="svg")
        assert (tmp_path / "training_curve.svg").is_file()

    def test_every_written_file_is_reported(self, tmp_path):
        """
        Including the figures, not just the page.

        The outcome's paths are what get published as artifacts, so a
        figure omitted here is a figure missing from the tracker.
        """
        _, outcome = _render(tmp_path)
        assert len(outcome.paths) == 4

    def test_the_page_is_listed_first(self, tmp_path):
        """
        It is the entry point, so it should lead any listing.

        A tracker showing three figures before the page they belong to
        buries the thing worth opening.
        """
        _, outcome = _render(tmp_path)
        assert outcome.paths[0].name == SUMMARY_FILENAME


class TestDegenerateRuns:
    """A summary that raised would turn one failure into two."""

    def test_a_single_epoch_fit_omits_the_curve(self, tmp_path):
        """
        A one-point line is not a curve, and a one-shot engine has one.

        Plotting it would imply a trend that cannot exist.
        """
        result = make_training_result(n_epochs=1)
        _render(tmp_path, bundle=make_model_bundle(result=result))
        assert not (tmp_path / "training_curve.png").exists()

    def test_a_fit_with_no_history_still_renders(self, tmp_path):
        """
        A boosted-tree fit that recorded nothing is still a finished run.

        And a run that failed before its first epoch still deserves a page.
        """
        result = TrainingResult(fit=FitOutcome())
        text, _ = _render(tmp_path, bundle=make_model_bundle(result=result))
        assert "## Training" in text

    def test_a_run_without_validation_renders_a_single_series(self, tmp_path):
        """
        A legitimate configuration -- a final refit on everything.

        Plotting an empty validation series would put a phantom entry in
        the legend.
        """
        result = make_training_result(with_validation=False)
        _render(tmp_path, bundle=make_model_bundle(result=result))
        assert (tmp_path / "training_curve.png").is_file()

    def test_the_page_contains_no_unresolved_placeholders(self, page):
        """
        Catching a format string that was never filled in.

        Which renders as plausible text and is easy to miss in review.
        """
        assert "{" not in page
        assert "None" not in page
