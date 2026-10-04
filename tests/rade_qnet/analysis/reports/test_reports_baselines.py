"""
Tests for the baselines report.

A metric on its own is not a result. A mean absolute error of 0.004 on a daily
return series is roughly what predicting zero achieves, and a graph-temporal
network that cannot beat a persistence forecast has not earned the compute it
cost. This report is the page that makes that comparison impossible to skip.

The parts worth testing are the places where a wrong answer would be
*flattering*. The verdict has to know which direction is better for each
metric, and where it does not know, it has to say so rather than guess -- a
guessed direction turns a worse-than-baseline result into a better-than one
with no indication. The ratio has to cope with a baseline of zero, which is
not hypothetical: a persistence baseline on a stationary series can have
exactly zero bias.

And the page has to render for a run with no baselines at all, because a
report is never load-bearing.
"""

from __future__ import annotations

from src.rade_qnet.analysis.reports.base import ReportContext
from src.rade_qnet.analysis.reports.baselines import BASELINES_FILENAME, BaselinesReport
from src.rade_qnet.core.contract.result import EvalResult, TrainingResult
from src.rade_qnet.core.lifecycle.components import get_report
from src.rade_qnet.testkit.fixtures import make_model_bundle, make_training_result


def render(
    tmp_path,
    *,
    evaluations: dict[str, EvalResult] | None = None,
) -> tuple[str | None, object]:
    """
    Render the report and return its page text and outcome.

    Parameters
    ----------
    tmp_path
        Directory to render into.
    evaluations
        Split evaluations to report on. ``None`` uses the standard fixture.

    Returns
    -------
    tuple
        The page text -- ``None`` when the report was skipped -- and the
        report outcome.
    """
    result: TrainingResult = make_training_result()
    if evaluations is not None:
        result = result.model_copy(update={"evaluations": evaluations})

    outcome = BaselinesReport().render_safely(
        ReportContext(bundle=make_model_bundle(result=result), directory=tmp_path)
    )
    page = tmp_path / BASELINES_FILENAME
    return (page.read_text(encoding="utf-8") if page.exists() else None), outcome


def evaluation(
    split: str = "test",
    *,
    metrics: dict[str, float] | None = None,
    baseline_metrics: dict[str, float] | None = None,
) -> EvalResult:
    """
    Build one split's evaluation.

    Parameters
    ----------
    split
        Split name.
    metrics
        The model's metrics.
    baseline_metrics
        The baseline's metrics.

    Returns
    -------
    EvalResult
        The evaluation.
    """
    return EvalResult(
        split=split,
        metrics=metrics if metrics is not None else {"mae": 0.5, "r2": 0.8},
        n_samples=100,
        baseline_metrics=baseline_metrics
        if baseline_metrics is not None
        else {"mae": 1.0, "r2": 0.1},
    )


class TestRegistration:
    """Reached by the name a specification uses."""

    def test_it_is_registered_as_baselines(self):
        """So a run can enable the comparison from a configuration file."""
        assert get_report("baselines") is BaselinesReport


class TestTheComparison:
    """Both numbers, side by side, with a direction."""

    def test_both_the_model_and_the_baseline_appear(self, tmp_path):
        """
        Because a ratio alone hides the scale.

        Twice as good as a baseline is excellent or irrelevant depending on
        whether the baseline was any good, and the only way to tell is to see
        both numbers.
        """
        page, _ = render(tmp_path, evaluations={"test": evaluation()})
        # The metric's own row carries both figures: 0.5 for the model and 1
        # for the baseline, in the same line.
        row = next(line for line in page.splitlines() if line.startswith("| mae |"))
        assert "0.5" in row
        assert "1" in row.split("|")[3]

    def test_a_better_model_is_reported_as_better(self, tmp_path):
        """
        For a metric where lower is better.

        Getting the direction wrong here is the single most misleading thing
        this report could do, because the verdict is what a reader takes away.
        """
        page, _ = render(
            tmp_path,
            evaluations={"test": evaluation(metrics={"mae": 0.5}, baseline_metrics={"mae": 1.0})},
        )
        assert "better" in page.lower()

    def test_a_worse_model_is_reported_as_worse(self, tmp_path):
        """
        Which is the result the report exists to surface.

        A model that loses to a persistence forecast has not earned the
        compute it cost, and the page must say so plainly rather than leaving
        a reader to compare two numbers.
        """
        page, _ = render(
            tmp_path,
            evaluations={"test": evaluation(metrics={"mae": 2.0}, baseline_metrics={"mae": 1.0})},
        )
        assert "worse" in page.lower()

    def test_a_higher_is_better_metric_is_judged_the_other_way(self, tmp_path):
        """
        Because the direction is per metric, not global.

        An r-squared of 0.8 against a baseline's 0.1 is a good result, and a
        single global direction would report it as a loss.
        """
        page, _ = render(
            tmp_path,
            evaluations={"test": evaluation(metrics={"r2": 0.8}, baseline_metrics={"r2": 0.1})},
        )
        assert "worse" not in page.lower()

    def test_an_unknown_metric_direction_is_not_guessed(self, tmp_path):
        """
        The deliberate omission.

        A guessed direction converts a worse-than-baseline result into a
        better-than one with nothing in the output to signal the guess. An
        em dash says "this needs a human", which is the honest answer.
        """
        page, _ = render(
            tmp_path,
            evaluations={
                "test": evaluation(
                    metrics={"custom_score": 0.5}, baseline_metrics={"custom_score": 1.0}
                )
            },
        )
        assert "better" not in page.lower()
        assert "worse" not in page.lower()


class TestRatios:
    """The arithmetic, including the case that divides by zero."""

    def test_a_ratio_is_reported(self, tmp_path):
        """
        Because "half the error" is more readable than two decimals.

        It is the form a summary sentence can use, and the form that survives
        a change of units.
        """
        page, _ = render(
            tmp_path,
            evaluations={"test": evaluation(metrics={"mae": 0.5}, baseline_metrics={"mae": 1.0})},
        )
        assert "0.5" in page

    def test_a_zero_baseline_does_not_produce_an_infinity(self, tmp_path):
        """
        And this is not hypothetical.

        A persistence baseline on a stationary series can have exactly zero
        bias, and a page reporting ``inf`` or ``nan`` next to a real metric
        reads as a broken report rather than as an undefined ratio.
        """
        page, _ = render(
            tmp_path,
            evaluations={
                "test": evaluation(metrics={"abs_bias": 0.5}, baseline_metrics={"abs_bias": 0.0})
            },
        )
        # Checked per cell rather than over the whole page, since "inf" is a
        # substring of "information" in the preamble.
        cells = {
            cell.strip().lower()
            for line in page.splitlines()
            if line.startswith("|")
            for cell in line.split("|")
        }
        assert not cells & {"inf", "-inf", "nan"}

    def test_a_metric_the_baseline_did_not_produce_is_still_shown(self, tmp_path):
        """
        Because the model's own number is worth reading regardless.

        Dropping it would make a page whose contents depend on which
        baselines happened to be configured.
        """
        page, _ = render(
            tmp_path,
            evaluations={
                "test": evaluation(
                    metrics={"mae": 0.5, "directional_accuracy": 0.6},
                    baseline_metrics={"mae": 1.0},
                )
            },
        )
        assert "directional_accuracy" in page


class TestStructure:
    """One section per split, in a stable order."""

    def test_every_evaluated_split_gets_a_section(self, tmp_path):
        """
        Because the splits can disagree, and the disagreement is the point.

        A model that beats the baseline on validation and loses on test is
        the most important case this page can show, and a single combined
        section would average it away.
        """
        page, _ = render(
            tmp_path,
            evaluations={
                "validation": evaluation("validation"),
                "test": evaluation("test"),
            },
        )
        assert "validation" in page
        assert "test" in page

    def test_the_page_opens_with_a_summary_sentence(self, tmp_path):
        """
        So the answer is readable without reading the tables.

        This page exists to make one comparison unavoidable, and a reader who
        stops after the first line should still have got it.
        """
        page, _ = render(tmp_path, evaluations={"test": evaluation()})
        assert len(page.split("\n")[0].split()) > 1

    def test_metrics_not_in_original_units_are_flagged(self, tmp_path):
        """
        Because the comparison is then between two scaled numbers.

        A mean absolute error of 0.03 in standardised units says nothing
        about the size of the error in the units anyone cares about, and the
        same is true of the baseline it is compared against.
        """
        page, _ = render(
            tmp_path,
            evaluations={
                "test": EvalResult(
                    split="test",
                    metrics={"mae": 0.03},
                    n_samples=100,
                    in_original_units=False,
                    baseline_metrics={"mae": 0.05},
                )
            },
        )
        assert "unit" in page.lower()


class TestDegenerateRuns:
    """A report is never load-bearing."""

    def test_a_run_with_no_baselines_is_skipped_with_a_reason(self, tmp_path):
        """
        Rather than writing a page of empty tables.

        The reason goes into the run manifest, so a later reader can tell a
        report that was not configured from one that failed.
        """
        _, outcome = render(tmp_path, evaluations={"test": evaluation(baseline_metrics={})})
        assert not outcome.succeeded
        assert outcome.skipped_reason

    def test_a_run_with_no_evaluations_is_skipped(self, tmp_path):
        """
        Because there is nothing to compare.

        Which is the state of a run that trained and was never scored, and
        that is a legitimate thing to do.
        """
        _, outcome = render(tmp_path, evaluations={})
        assert not outcome.succeeded

    def test_the_written_path_is_reported(self, tmp_path):
        """
        Because it goes into the run manifest.

        A manifest naming a file that was never written makes a later
        consumer fail on a missing file with nothing to say why.
        """
        _, outcome = render(tmp_path, evaluations={"test": evaluation()})
        assert all(path.exists() for path in outcome.paths)
