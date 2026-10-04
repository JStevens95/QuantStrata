"""
Tests for the reports specification.

``fail_fast`` exists so that "a report never fails a run" is the default rather
than the only behaviour. Defaulting it to false is deliberate: reporting code
is the least-tested code in any pipeline, and a figure that cannot render
should not discard four hours of training. A caller who genuinely wants a
report failure to be fatal has to say so, which makes it reviewable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.core.spec.reports import ReportsSpec


class TestDefaults:
    """Defaults produce a useful run with no configuration."""

    def test_it_constructs_with_no_arguments(self):
        """Reached through a ``default_factory``, so it must be bare."""
        assert ReportsSpec() is not None

    def test_the_summary_report_is_enabled_by_default(self):
        """
        A run with no report configuration still explains itself.

        The summary needs only the bundle, so it works for every model and
        every engine -- which is what makes it safe as the default.
        """
        assert ReportsSpec().enabled == ("summary",)

    def test_report_failures_are_not_fatal_by_default(self):
        """The rule stated in the module docstring."""
        assert ReportsSpec().fail_fast is False


class TestEnabledReports:
    """The enabled list is validated, not merely accepted."""

    def test_an_empty_list_is_accepted(self):
        """
        Disabling reporting entirely is legitimate.

        A tuning sweep of four hundred trials does not want four hundred
        summary pages.
        """
        assert ReportsSpec(enabled=()).enabled == ()

    def test_duplicates_are_rejected(self):
        """
        A report named twice would run twice and overwrite its own output.

        The second run's figures replace the first's, so the duplicate is
        pure waste -- and it usually means the user merged two configurations
        by hand.
        """
        with pytest.raises(ValidationError, match="summary"):
            ReportsSpec(enabled=("summary", "curves", "summary"))

    def test_several_distinct_reports_are_accepted(self):
        """The normal case."""
        assert ReportsSpec(enabled=("summary", "curves")).enabled == ("summary", "curves")

    def test_order_is_preserved(self):
        """
        Reports run in the order given.

        Not sorted, because a user may want the expensive one last so the
        cheap ones have already been written if it fails.
        """
        assert ReportsSpec(enabled=("curves", "summary")).enabled == ("curves", "summary")


class TestFigureSettings:
    """Figure settings are bounded, so a typo cannot produce a useless file."""

    @pytest.mark.parametrize("figure_format", ["png", "svg", "pdf"])
    def test_supported_formats_are_accepted(self, figure_format):
        """All three are lossless or vector."""
        assert ReportsSpec(figure_format=figure_format).figure_format == figure_format

    def test_a_lossy_format_is_rejected(self):
        """
        A figure that will be read carefully should not be a JPEG.

        Restricting the set at the spec level means no report has to defend
        against it.
        """
        with pytest.raises(ValidationError):
            ReportsSpec(figure_format="jpg")

    def test_a_sensible_resolution_is_accepted(self):
        """The normal case."""
        assert ReportsSpec(figure_dpi=300).figure_dpi == 300

    @pytest.mark.parametrize("dpi", [10, 1200])
    def test_an_unusable_resolution_is_rejected(self, dpi):
        """
        Bounded on both sides.

        Ten is illegible; twelve hundred produces a hundred-megabyte PNG that
        nothing will open. Neither is what the user meant.
        """
        with pytest.raises(ValidationError):
            ReportsSpec(figure_dpi=dpi)


class TestStrictness:
    """The usual spec guarantees."""

    def test_an_unknown_key_is_rejected(self):
        """A typo fails at load."""
        with pytest.raises(ValidationError):
            ReportsSpec(enable=("summary",))

    def test_the_spec_is_frozen(self):
        """A report cannot enable itself mid-run."""
        spec = ReportsSpec()
        with pytest.raises(ValidationError):
            spec.fail_fast = True

    def test_round_trip_is_exact(self):
        """So the bundle records what actually ran."""
        spec = ReportsSpec(
            enabled=("summary", "curves"),
            directory_name="figures",
            fail_fast=True,
            figure_format="svg",
            figure_dpi=200,
        )
        assert ReportsSpec.model_validate(spec.model_dump()) == spec
