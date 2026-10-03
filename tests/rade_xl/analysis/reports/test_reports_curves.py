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

from src.rade_xl.analysis.reports.base import ReportContext
from src.rade_xl.analysis.reports.curves import HISTORY_FILENAME, CurvesReport
from src.rade_xl.core.contract.result import EpochRecord, FitOutcome, TrainingResult
from src.rade_xl.core.runtime.components import get_report
from src.rade_xl.testkit.fixtures import make_model_bundle, make_training_result


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
