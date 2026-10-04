"""
Tests for the report base and the rule it enforces.

A report is never load-bearing. Reporting code is the least-tested code in any
pipeline, because its output is read by people rather than asserted on by
tests -- which makes it the stage most likely to raise on an unusual input: an
empty validation split, a metric absent for one job, a single-epoch history.
If a report could fail a run, the least important stage in the pipeline would
be the one that destroys four hours of training.

So most of this module is about failure: every way a report can go wrong, and
confirmation that none of them propagates. The one thing that must *not* be
swallowed is the reason, because a missing section nobody explains reads as a
bug in the framework.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rade_qnet.analysis.reports.base import Report, ReportContext, ReportOutcome
from src.rade_qnet.testkit.fixtures import make_model_bundle, make_run_context


class WritingReport(Report):
    """A report that writes one file."""

    component_name = "writing"

    def render(self, context):
        """Write a single file and return it."""
        path = context.directory / "report.md"
        path.write_text("content", encoding="utf-8")
        return [path]


class FailingReport(Report):
    """A report that raises partway through."""

    component_name = "failing"

    def render(self, context):
        """Raise the kind of error an unusual input produces."""
        raise KeyError("validation")


class SkippingReport(Report):
    """A report that declines deliberately."""

    component_name = "skipping"

    def render(self, context):
        """Decline with a clear reason."""
        raise LookupError("there is no test split to attribute")


@pytest.fixture
def context(tmp_path):
    """
    Provide a report context pointing at a temporary directory.

    Returns
    -------
    ReportContext
        The context.
    """
    return ReportContext(bundle=make_model_bundle(), directory=tmp_path / "reports")


class TestOutcome:
    """The record of what a report did."""

    def test_a_report_with_paths_succeeded(self):
        """Success is the absence of a reason, not a separate flag."""
        assert ReportOutcome(name="demo", paths=(Path("a.md"),)).succeeded

    def test_a_report_with_a_reason_did_not(self):
        """
        Two fields cannot disagree if only one of them is authoritative.

        A separate boolean would eventually be set inconsistently with the
        reason beside it.
        """
        assert not ReportOutcome(name="demo", skipped_reason="no test split").succeeded

    def test_a_report_that_wrote_nothing_still_succeeded(self):
        """
        Writing no files is not the same as failing.

        A report that legitimately had nothing to say should not be
        presented as broken.
        """
        assert ReportOutcome(name="demo").succeeded

    def test_the_outcome_is_frozen(self):
        """A record of what happened, not a mutable scratchpad."""
        with pytest.raises(AttributeError):
            ReportOutcome(name="demo").skipped_reason = "changed"


class TestTheContract:
    """What a subclass must provide, and what it is given."""

    def test_render_is_abstract(self):
        """
        A report that renders nothing is not a report.

        And the failure belongs at class-definition time.
        """
        assert Report.render.__isabstractmethod__

    def test_an_incomplete_report_cannot_be_instantiated(self):
        """The requirement, demonstrated."""

        class Incomplete(Report):
            """Declares a name and nothing else."""

            component_name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_the_context_carries_only_what_a_report_needs(self, context):
        """
        A narrow bundle of inputs rather than the whole pipeline.

        Which is what stops a report reaching into pipeline internals and
        quietly becoming load-bearing after all.
        """
        assert context.bundle is not None
        assert not hasattr(context, "pipeline")

    def test_figure_settings_come_from_the_specification(self, tmp_path):
        """
        So one run's figures are consistent across every report.

        A report choosing its own format would produce a document mixing
        PNG and SVG for no reason a reader could see.
        """
        narrow = ReportContext(
            bundle=make_model_bundle(),
            directory=tmp_path,
            figure_format="svg",
            figure_dpi=300,
        )
        assert (narrow.figure_format, narrow.figure_dpi) == ("svg", 300)

    def test_extras_are_available_for_model_specific_reports(self, tmp_path):
        """
        A model's own pipeline may publish values its own report reads.

        Framework reports never touch this, which keeps the extension point
        from becoming a back channel into shared code.
        """
        narrow = ReportContext(
            bundle=make_model_bundle(),
            directory=tmp_path,
            extras={"attribution": [1, 2, 3]},
        )
        assert narrow.extras["attribution"] == [1, 2, 3]


class TestSafeRendering:
    """The method the pipeline actually calls."""

    def test_a_successful_report_returns_its_paths(self, context):
        """The normal case."""
        outcome = WritingReport().render_safely(context)
        assert outcome.succeeded
        assert outcome.paths[0].read_text(encoding="utf-8") == "content"

    def test_the_output_directory_is_created(self, context):
        """
        A report author should not have to remember to create it.

        Forgetting produces a ``FileNotFoundError`` that is then swallowed,
        so the report silently disappears.
        """
        WritingReport().render_safely(context)
        assert context.directory.is_dir()

    def test_a_failing_report_does_not_raise(self, context):
        """
        The central guarantee.

        Four hours of training must not be lost to a ``KeyError`` in a
        plotting helper.
        """
        assert not FailingReport().render_safely(context).succeeded

    def test_the_failure_reason_is_preserved(self, context):
        """
        Swallowed is not the same as hidden.

        A run summary should say why a section is missing, because a
        missing section nobody explains is read as a framework bug.
        """
        outcome = FailingReport().render_safely(context)
        assert "KeyError" in outcome.skipped_reason

    def test_a_deliberate_skip_reads_clearly(self, context):
        """
        What a report author should do instead of relying on a ``KeyError``.

        "There is no test split to attribute" is an explanation; a stack
        trace in a log is not.
        """
        outcome = SkippingReport().render_safely(context)
        assert "no test split" in outcome.skipped_reason

    def test_the_failure_is_logged_with_the_report_named(self, context, caplog):
        """
        So the log says which report, out of a dozen, went wrong.

        Without the name, a warning about a ``KeyError`` sends the reader
        through every report in the run.
        """
        with caplog.at_level("WARNING"):
            FailingReport().render_safely(context)
        assert "failing" in caplog.text

    def test_the_outcome_carries_the_registered_name(self, context):
        """
        Not the class name, which a user never typed.

        The summary refers to reports by the name in the specification.
        """
        assert WritingReport().render_safely(context).name == "writing"

    def test_an_unregistered_report_falls_back_to_its_class_name(self, context):
        """
        A report rendered from a notebook has no registered name.

        Raising on the missing attribute would make the safety net itself
        the thing that fails.
        """

        class Unregistered(Report):
            """Never passed through the decorator."""

            def render(self, context):
                """Write nothing."""
                return []

        assert Unregistered().render_safely(context).name == "Unregistered"


class TestArtifactPublication:
    """Written files reach the tracker, and failing to do so is harmless."""

    def test_written_files_are_published_as_artifacts(self, context, tmp_path):
        """
        So a tracked run links to its report without the report knowing how.

        Which keeps tracking out of every report author's concern.
        """
        run = make_run_context(output_directory=tmp_path / "run")
        outcome = WritingReport().render_safely(context, run)
        assert outcome.succeeded

    def test_a_report_rendered_outside_a_run_still_works(self, context):
        """
        ``run`` is optional, for a notebook or an ad-hoc rerun.

        Requiring a full run context to render a report would make
        reproducing one from a saved bundle unnecessarily hard.
        """
        assert WritingReport().render_safely(context, None).succeeded

    def test_a_failing_report_publishes_nothing(self, context, tmp_path):
        """
        A skipped report has no artifacts to link.

        Publishing a path that was never written would leave a dead link in
        the tracker.
        """
        run = make_run_context(output_directory=tmp_path / "run")
        assert FailingReport().render_safely(context, run).paths == ()
