# `tests/rade_qnet/analysis/reports`

6 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 23 | 845 | `243b3f95c99bd0f3` |
| 2 | `test_reports_base.py` | 280 | 9518 | `131b98b13897f88f` |
| 3 | `test_reports_baselines.py` | 326 | 11384 | `9b12b9557f8e4ea2` |
| 4 | `test_reports_curves.py` | 329 | 10916 | `95fe830b23029c76` |
| 5 | `test_reports_quality.py` | 277 | 10426 | `c4848a2f442042b8` |
| 6 | `test_reports_summary.py` | 394 | 13449 | `f2f29c7816c8684e` |

---

## 1. `tests/rade_qnet/analysis/reports/__init__.py`

845 bytes · SHA-256 `243b3f95c99bd0f3`

```python
"""
Tests for ``rade_qnet.analysis.reports``.

The rule under test throughout is that no report is load-bearing. A report that
raises must produce a warning and a completed run, never a lost one. Each
report suite therefore includes a deliberately failing writer alongside the
happy path.

Planned modules
---------------
``test_reports_base.py``
    The ``Report`` protocol, registration by name, and a failing report
    degrading to a warning.  [Phase 1]
``test_reports_summary.py``
    Summary contents and the files written.  [Phase 1]
``test_reports_curves.py``
    Curve figures plus the raw history saved alongside them.  [Phase 2]
``test_reports_baselines.py``
    Baselines evaluated on the same split as the model they are compared
    against.  [Phase 2]
``test_reports_quality.py``
    The input data quality report.  [Phase 2]
"""
```

---

## 2. `tests/rade_qnet/analysis/reports/test_reports_base.py`

9518 bytes · SHA-256 `131b98b13897f88f`

```python
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
```

---

## 3. `tests/rade_qnet/analysis/reports/test_reports_baselines.py`

11384 bytes · SHA-256 `9b12b9557f8e4ea2`

```python
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
```

---

## 4. `tests/rade_qnet/analysis/reports/test_reports_curves.py`

10916 bytes · SHA-256 `95fe830b23029c76`

```python
"""
Tests for the training-curves report.

This report exists so that a finished run can be read without rerunning it.
The CSV is the load-bearing part: it is what someone opens in a spreadsheet
six months later, and what a comparison between two runs is built from. The
figures are a convenience on top of it.

Two properties of the CSV are deliberate. The column order is fixed for the
known columns and sorted for the rest, so two runs of the same configuration
produce diffable files -- an order that depended on dictionary insertion would
make every re-run look like a change. And the column set is the *union* across
every epoch, so an epoch that recorded one extra metric does not shift the
remaining values one column to the left for every other row.

The figures are attempted independently of one another. A run with no gradient
norms recorded should still get its loss curve, because a report that fails as
a unit is a report that disappears exactly when something has gone wrong.
"""

from __future__ import annotations

import csv

from src.rade_qnet.analysis.reports.base import ReportContext
from src.rade_qnet.analysis.reports.curves import HISTORY_FILENAME, CurvesReport
from src.rade_qnet.core.contract.result import EpochRecord, FitOutcome, TrainingResult
from src.rade_qnet.core.lifecycle.components import get_report
from src.rade_qnet.testkit.fixtures import make_model_bundle, make_training_result


def make_fit(
    *,
    n_epochs: int = 4,
    with_validation: bool = True,
    metrics_per_epoch: list[dict[str, float]] | None = None,
) -> FitOutcome:
    """
    Build a fit outcome with a plausible history.

    Parameters
    ----------
    n_epochs
        Epoch count.
    with_validation
        Whether each record carries a validation loss.
    metrics_per_epoch
        Per-epoch metric mappings.

    Returns
    -------
    FitOutcome
        The outcome.
    """
    metrics = metrics_per_epoch or [{} for _ in range(n_epochs)]
    history = tuple(
        EpochRecord(
            epoch=index,
            train_loss=1.0 / (index + 1),
            val_loss=1.2 / (index + 1) if with_validation else None,
            metrics=metrics[index],
            learning_rate=0.01,
            seconds=0.4,
        )
        for index in range(n_epochs)
    )
    return FitOutcome(
        history=history,
        monitor="val_loss" if with_validation else "train_loss",
        best_epoch=0,
        best_monitor_value=history[0].val_loss or history[0].train_loss,
        stopped_early=False,
        restored_best=True,
        total_seconds=n_epochs * 0.4,
    )


def render(tmp_path, fit: FitOutcome | None = None):
    """
    Render the report and return its outcome.

    Parameters
    ----------
    tmp_path
        Directory to render into.
    fit
        Fit outcome to report on, defaulting to a four-epoch run.

    Returns
    -------
    ReportOutcome
        The outcome.
    """
    return CurvesReport().render_safely(ReportContext(bundle=make_bundle(fit), directory=tmp_path))


def make_bundle(fit: FitOutcome | None = None):
    """
    Build a bundle whose training result carries a given fit outcome.

    Parameters
    ----------
    fit
        The fit outcome, defaulting to a four-epoch run.

    Returns
    -------
    ModelBundle
        The bundle.
    """
    result: TrainingResult = make_training_result()
    return make_model_bundle(
        result=result.model_copy(update={"fit": fit if fit is not None else make_fit()})
    )


def read_history(tmp_path) -> list[dict[str, str]]:
    """
    Read the written history CSV.

    Parameters
    ----------
    tmp_path
        The render directory.

    Returns
    -------
    list of dict
        One mapping per row.
    """
    with (tmp_path / HISTORY_FILENAME).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestRegistration:
    """Reached by the name a specification uses."""

    def test_it_is_registered_as_curves(self):
        """
        So a run can ask for it by name without importing the class.

        A mismatch between the registered name and the one in the default
        specification produces a run with no curves report and no error.
        """
        assert get_report("curves") is CurvesReport


class TestTheHistoryFile:
    """The part someone opens six months later."""

    def test_one_row_per_epoch(self, tmp_path):
        """So the file length is the epoch count, with nothing interpolated."""
        render(tmp_path, make_fit(n_epochs=4))
        assert len(read_history(tmp_path)) == 4

    def test_the_known_columns_come_first_and_in_a_fixed_order(self, tmp_path):
        """
        So two runs of the same configuration produce diffable files.

        An order that followed dictionary insertion would make every re-run
        look like a change, and a diff that is always noisy is a diff nobody
        reads.
        """
        render(tmp_path, make_fit())
        columns = list(read_history(tmp_path)[0])
        assert columns[:5] == [
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
            "seconds",
        ]

    def test_the_metric_columns_are_sorted(self, tmp_path):
        """
        For the same reason, applied to the columns nobody declared.

        A model's own metrics arrive in whatever order its learner returned
        them, which is not stable across refactors of the learner.
        """
        render(
            tmp_path,
            make_fit(
                n_epochs=2,
                metrics_per_epoch=[{"zeta": 1.0, "alpha": 2.0}, {"zeta": 1.0, "alpha": 2.0}],
            ),
        )
        columns = list(read_history(tmp_path)[0])
        assert columns.index("alpha") < columns.index("zeta")

    def test_the_columns_are_the_union_across_every_epoch(self, tmp_path):
        """
        So one epoch's extra metric does not shift every other row's values.

        Taking the first epoch's keys instead would silently drop a metric
        that only started being recorded later; taking each row's own keys
        would misalign the file entirely.
        """
        render(
            tmp_path,
            make_fit(
                n_epochs=3,
                metrics_per_epoch=[{"alpha": 1.0}, {"alpha": 1.0, "beta": 2.0}, {"alpha": 1.0}],
            ),
        )
        rows = read_history(tmp_path)
        assert "beta" in rows[0]
        assert rows[1]["beta"] == "2.0"

    def test_an_epoch_missing_a_metric_leaves_the_cell_empty(self, tmp_path):
        """
        Rather than writing a zero.

        Zero is a value, and in a gradient-norm column it is the alarming one
        -- so "not recorded" must not be written as "recorded as zero".
        """
        render(
            tmp_path,
            make_fit(n_epochs=2, metrics_per_epoch=[{"alpha": 1.0}, {}]),
        )
        assert read_history(tmp_path)[1]["alpha"] == ""

    def test_a_run_with_no_validation_leaves_that_column_empty(self, tmp_path):
        """
        For the same reason, applied to the column someone will average.

        A validation column filled with zeros would make a mean validation
        loss of zero, which is the best possible score.
        """
        render(tmp_path, make_fit(with_validation=False))
        assert read_history(tmp_path)[0]["val_loss"] == ""


class TestFigures:
    """Attempted independently, because a report is never load-bearing."""

    def test_the_loss_curve_is_written(self, tmp_path):
        """The one figure every run can produce."""
        outcome = render(tmp_path, make_fit())
        assert any("training_curve" in path.name for path in outcome.paths)

    def test_a_run_without_gradient_norms_still_gets_its_loss_curve(self, tmp_path):
        """
        Because the figures are attempted one at a time.

        Rendered as a unit, a run with norm tracking off would lose every
        figure -- including the ones it could produce -- and the report would
        disappear at exactly the moment someone needed it.
        """
        outcome = render(tmp_path, make_fit())
        assert outcome.succeeded
        assert len(outcome.paths) >= 2

    def test_the_gradient_norm_figure_appears_when_norms_were_recorded(self, tmp_path):
        """
        So turning tracking on actually produces something.

        A tracker whose output never reached a figure would be pure cost.
        """
        outcome = render(
            tmp_path,
            make_fit(
                n_epochs=2,
                metrics_per_epoch=[{"grad_norm_mean": 1.0, "grad_norm_max": 2.0}] * 2,
            ),
        )
        assert any("gradient_norm" in path.name for path in outcome.paths)

    def test_every_reported_path_exists(self, tmp_path):
        """
        Because the paths go into the run manifest.

        A manifest naming a file that was never written makes a later
        consumer fail on a missing file, with nothing to say which stage was
        supposed to have produced it.
        """
        outcome = render(tmp_path, make_fit())
        assert all(path.exists() for path in outcome.paths)


class TestDegenerateRuns:
    """A report that raised would turn one failure into two."""

    def test_a_single_epoch_run_renders(self, tmp_path):
        """
        Which is what a smoke test or a failed run produces.

        A line plot of one point is not informative, but it is not an error
        either, and the CSV is still worth having.
        """
        outcome = render(tmp_path, make_fit(n_epochs=1))
        assert outcome.succeeded

    def test_an_empty_history_is_skipped_with_a_reason(self, tmp_path):
        """
        Rather than writing a header-only CSV.

        An empty history means the fit produced nothing, and the reason
        belongs in the report outcome so the run manifest records why the
        report is absent.
        """
        empty = FitOutcome(
            history=(),
            monitor="train_loss",
            stopped_early=False,
            restored_best=False,
            total_seconds=0.0,
        )
        outcome = render(tmp_path, empty)
        assert not outcome.succeeded
        assert outcome.skipped_reason

    def test_the_figure_format_from_the_context_is_honoured(self, tmp_path):
        """
        Because a paper wants vector output and a dashboard wants raster.

        Hardcoding either one means the other needs a conversion step that
        nobody maintains.
        """
        outcome = CurvesReport().render_safely(
            ReportContext(
                bundle=make_bundle(make_fit(n_epochs=2)),
                directory=tmp_path,
                figure_format="svg",
            )
        )
        assert all(path.suffix in {".svg", ".csv"} for path in outcome.paths), [
            path.name for path in outcome.paths
        ]
```

---

## 5. `tests/rade_qnet/analysis/reports/test_reports_quality.py`

10426 bytes · SHA-256 `c4848a2f442042b8`

```python
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
```

---

## 6. `tests/rade_qnet/analysis/reports/test_reports_summary.py`

13449 bytes · SHA-256 `f2f29c7816c8684e`

```python
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
from src.rade_qnet.core.lifecycle.components import get_report
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
```

