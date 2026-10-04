"""
Tests for the data-quality report.

This is the page that answers "did the data change" when a model's score
drops. Without it, a feed that quietly started forward-filling a price looks
exactly like a model that stopped generalising, and the two have completely
different fixes.

The load-bearing part is the provenance section, which records *both* digests:
the source fingerprint and the specification digest. With only one of them,
"the model got worse" and "the data got worse" are indistinguishable -- two
runs that differ are known to differ, but not in which respect.

The splits section names the boundary-gap shortfall explicitly, because that
is the one number proving a sequence-windowed chronological split did not
leak. If the shares add up to the full scenario count while the sequence
length is above one, something read across a boundary.

Everything here degrades to a stated absence rather than to an error or a
zero. A quality report that failed would remove the evidence at exactly the
moment it was needed.
"""

from __future__ import annotations

from src.rade_qnet.analysis.reports.base import ReportContext
from src.rade_qnet.analysis.reports.quality import QUALITY_FILENAME, QualityReport
from src.rade_qnet.core.lifecycle.components import get_report
from src.rade_qnet.testkit.fixtures import make_lineage, make_model_bundle


def render(tmp_path, *, lineage=None, bundle=None) -> tuple[str | None, object]:
    """
    Render the report and return its page text and outcome.

    Parameters
    ----------
    tmp_path
        Directory to render into.
    lineage
        Lineage to report on, defaulting to the standard fixture.
    bundle
        A complete bundle, overriding ``lineage``.

    Returns
    -------
    tuple
        The page text -- ``None`` when the report was skipped -- and the
        report outcome.
    """
    if bundle is None:
        bundle = make_model_bundle(lineage=lineage if lineage is not None else make_lineage())
    outcome = QualityReport().render_safely(ReportContext(bundle=bundle, directory=tmp_path))
    page = tmp_path / QUALITY_FILENAME
    return (page.read_text(encoding="utf-8") if page.exists() else None), outcome


class TestRegistration:
    """Reached by the name a specification uses."""

    def test_it_is_registered_as_quality(self):
        """So a run can enable the page from a configuration file."""
        assert get_report("quality") is QualityReport


class TestProvenance:
    """Both digests, because one of them cannot tell the two causes apart."""

    def test_the_source_fingerprint_is_recorded(self, tmp_path):
        """
        So two runs over different data are known to be over different data.

        Without it, a score that moved has no attributable cause: the config
        is identical and the data is unlabelled.
        """
        page, _ = render(tmp_path, lineage=make_lineage())
        assert "synthetic" in page

    def test_the_specification_digest_is_recorded_too(self, tmp_path):
        """
        Because the pair is what distinguishes the two explanations.

        With only the data fingerprint, "the model got worse" and "the data
        got worse" look the same; with only the spec digest, so do they.
        """
        page, _ = render(tmp_path, lineage=make_lineage())
        assert "## Provenance" in page
        assert make_lineage().spec_digest[:8] in page

    def test_the_scenario_count_is_recorded(self, tmp_path):
        """
        Because a split share means nothing without the total it is of.

        Seventy per cent of four hundred scenarios and seventy per cent of
        four are the same share and not the same run.
        """
        page, _ = render(tmp_path, lineage=make_lineage())
        assert str(make_lineage().n_scenarios) in page

    def test_lineage_notes_are_reproduced_when_present(self, tmp_path):
        """
        Because they are how a data module explains a decision it made.

        A module that dropped a column for a stated reason should have that
        reason survive into the report, rather than into a log line nobody
        kept.
        """
        page, _ = render(tmp_path, lineage=make_lineage(notes={"dropped": "column 3 was constant"}))
        assert "column 3 was constant" in page


class TestSplits:
    """The shares, and the gap that proves the split did not leak."""

    def test_every_split_is_listed_with_its_size(self, tmp_path):
        """
        So the shares can be checked against what was configured.

        A fraction that silently rounded to zero produces an empty split, and
        the size is where that shows.
        """
        page, _ = render(tmp_path, lineage=make_lineage())
        assert "train" in page
        assert "## Splits" in page

    def test_the_boundary_gap_shortfall_is_named(self, tmp_path):
        """
        Because it is the one number proving a windowed split did not leak.

        If the split sizes add up to the full scenario count while the
        sequence length is above one, some sample read across a boundary --
        and that is a leak with no other symptom.
        """
        # Splits covering 90 of 100 scenarios, so ten belong to no split --
        # which is what a boundary gap looks like in the lineage.
        lineage = make_lineage().model_copy(
            update={
                "split_indices": {
                    "train": tuple(range(0, 70)),
                    "validation": tuple(range(75, 85)),
                    "test": tuple(range(90, 100)),
                }
            }
        )
        page, _ = render(tmp_path, lineage=lineage)
        assert "boundary gap" in page.lower()
        assert "10 scenario" in page

    def test_splits_that_cover_everything_report_no_gap(self, tmp_path):
        """
        Because a gap sentence that always appears says nothing.

        A sequence length of one needs no gap, and claiming one would make
        the reassurance meaningless in the runs that do need it.
        """
        page, _ = render(tmp_path, lineage=make_lineage())
        assert "boundary gap" not in page.lower()

    def test_a_lineage_with_no_split_indices_says_so(self, tmp_path):
        """
        Rather than rendering an empty table.

        An empty table reads as "no splits", which is a different and much
        more alarming claim than "the indices were not recorded".
        """
        page, _ = render(
            tmp_path,
            lineage=make_lineage().model_copy(update={"split_indices": {}}),
        )
        assert "no split indices" in page.lower()


class TestQualityMetrics:
    """The numbers that distinguish a data problem from a model problem."""

    def test_recorded_metrics_are_tabulated(self, tmp_path):
        """
        So they are readable without parsing the lineage.

        These are the numbers compared across runs, and a table is what makes
        a comparison by eye possible.
        """
        page, _ = render(
            tmp_path,
            lineage=make_lineage().model_copy(update={"quality": {"feature_completeness": 0.95}}),
        )
        assert "feature_completeness" in page

    def test_a_fractional_metric_is_shown_as_a_percentage(self, tmp_path):
        """
        Because 0.95 completeness and 95% completeness read very differently.

        A reader scanning for a problem spots "72%" far faster than "0.72",
        and these metrics are all bounded fractions by construction.
        """
        page, _ = render(
            tmp_path,
            lineage=make_lineage().model_copy(update={"quality": {"feature_completeness": 0.95}}),
        )
        assert "95" in page

    def test_poor_quality_produces_a_warning_section(self, tmp_path):
        """
        Because a number in a table is not a finding.

        The warnings are phrased for a person and written verbatim, so the
        page states the concern rather than leaving it to be inferred from a
        threshold the reader has to know.
        """
        page, _ = render(
            tmp_path,
            lineage=make_lineage().model_copy(
                update={"quality": {"feature_completeness": 0.2, "feature_staleness": 0.9}}
            ),
        )
        assert "warning" in page.lower()

    def test_a_run_with_no_quality_metrics_says_so(self, tmp_path):
        """
        Rather than reporting zeros.

        Zero completeness is the worst possible value, so a default would
        turn "not measured" into an alarm about entirely absent data.
        """
        page, _ = render(tmp_path, lineage=make_lineage())
        assert "no quality metrics" in page.lower()


class TestSignatureAndState:
    """What the model was served, and what was fitted to serve it."""

    def test_the_input_signature_is_recorded(self, tmp_path):
        """
        Because it is the interface the stored weights expect.

        A bundle reloaded against differently shaped data fails inside a
        forward pass, and the signature is what makes the mismatch checkable
        beforehand.
        """
        page, _ = render(tmp_path)
        assert "## Input signature" in page

    def test_the_fitted_transform_state_is_described(self, tmp_path):
        """
        Because an inference run must reuse it, not refit it.

        A state refitted at inference time standardises against the serving
        window's own statistics, which is a different transform from the one
        the model was trained through.
        """
        page, _ = render(tmp_path)
        assert "## Fitted transform state" in page


class TestDegenerateRuns:
    """A quality report that failed would remove the evidence."""

    def test_the_page_renders_for_a_minimal_run(self, tmp_path):
        """
        With every optional section stated as absent.

        Which is the state of a smoke test, and the run most likely to be the
        first thing anyone renders.
        """
        _, outcome = render(tmp_path)
        assert outcome.succeeded

    def test_the_written_path_is_reported(self, tmp_path):
        """
        Because it goes into the run manifest.

        A manifest naming a file that was never written makes a later
        consumer fail with nothing to say which stage should have produced
        it.
        """
        _, outcome = render(tmp_path)
        assert [path.name for path in outcome.paths] == [QUALITY_FILENAME]
