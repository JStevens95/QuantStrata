"""
The training-curve report: the figures, and the numbers behind them.

Writes the diagnostic figures *and* the raw per-epoch history as CSV. The CSV
is not a convenience, and it is the reason this report exists separately from
the summary.

A figure answers the question you thought to ask when you made it. Six months
later the question is different -- was the validation loss already rising at
epoch 40, what was the learning rate when the gradient norm spiked -- and a
PNG cannot be re-interrogated. The history is small, so keeping it costs
nothing and makes the run re-analysable with any tool.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from ...core.runtime.logging import get_logger
from ..visuals.export import save_figure
from ..visuals.primitives import training_curve_figure
from ..visuals.training import (
    epoch_timing_figure,
    gradient_norm_figure,
    learning_rate_figure,
    training_diagnostics_figure,
)
from .base import Report, ReportContext, report

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ...core.contract.result import FitOutcome

__all__ = ["CurvesReport"]

_LOGGER = get_logger(__name__)

#: File the per-epoch history is written to.
HISTORY_FILENAME = "training_history.csv"

#: Columns that come from an ``EpochRecord``'s own fields rather than from its
#: metrics mapping. Listed first so the CSV reads in a sensible order.
_FIXED_COLUMNS = ("epoch", "train_loss", "val_loss", "learning_rate", "seconds")


@report("curves")
class CurvesReport(Report):
    """
    Training diagnostics as figures, plus the raw history as CSV.

    Reads only the bundle's training result, so it works for every engine --
    a boosted-tree fit reports one record per round through the same
    :class:`~rade_xl.core.contract.result.FitOutcome` and renders identically.
    """

    def render(self, context: ReportContext) -> Sequence[Path]:
        """
        Write the history and whichever figures the history supports.

        Parameters
        ----------
        context
            The bundle to report on, and where to write.

        Returns
        -------
        Sequence of Path
            The CSV, and the figures written beside it.

        Raises
        ------
        LookupError
            If the fit recorded no epochs. Raised deliberately rather than
            left to fail incidentally, so the skip reason reads as an
            explanation instead of a stack trace -- see
            :meth:`~.base.Report.render`.
        """
        outcome = context.bundle.result.fit
        if not outcome.history:
            raise LookupError(
                "this run recorded no epoch history, so there are no curves to "
                "draw. A fit that failed before its first epoch completed produces "
                "this"
            )

        written = [self._write_history(outcome, context)]
        written.extend(self._write_figures(outcome, context))
        return tuple(written)

    @staticmethod
    def _write_history(outcome: FitOutcome, context: ReportContext) -> Path:
        """
        Write the per-epoch history as CSV.

        Parameters
        ----------
        outcome
            The fit outcome.
        context
            Where to write.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        # The union of every record's metric keys, so a metric that only
        # appeared after epoch ten still gets a column.  Sorted for a stable
        # column order between runs, which is what lets two histories be
        # diffed.
        metric_columns = sorted({name for record in outcome.history for name in record.metrics})
        columns = [*_FIXED_COLUMNS, *metric_columns]

        path = context.directory / HISTORY_FILENAME
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, restval="")
            writer.writeheader()
            for record in outcome.history:
                row: dict[str, object] = {
                    "epoch": record.epoch,
                    "train_loss": record.train_loss,
                    "val_loss": "" if record.val_loss is None else record.val_loss,
                    "learning_rate": ("" if record.learning_rate is None else record.learning_rate),
                    "seconds": record.seconds,
                }
                row.update(record.metrics)
                writer.writerow(row)

        _LOGGER.debug("wrote %d epoch(s) of history to %s", outcome.n_epochs, path)
        return path

    @staticmethod
    def _write_figures(outcome: FitOutcome, context: ReportContext) -> list[Path]:
        """
        Write every figure the history supports.

        Each factory is attempted independently and a failure skips only that
        figure. A report never fails a run, and within a report one
        unavailable diagnostic should not cost the others -- a fit with no
        recorded gradient norms still has a perfectly good loss curve.

        Parameters
        ----------
        outcome
            The fit outcome.
        context
            Where to write, and in what format.

        Returns
        -------
        list of Path
            The figures written.
        """
        factories = {
            "training_curve": lambda: training_curve_figure(
                outcome.curve("train_loss"),
                outcome.curve("val_loss") if _has_validation(outcome) else (),
                best_epoch=outcome.best_epoch,
            ),
            "training_diagnostics": lambda: training_diagnostics_figure(outcome),
            "learning_rate": lambda: learning_rate_figure(outcome),
            "gradient_norm": lambda: gradient_norm_figure(outcome),
            "epoch_timing": lambda: epoch_timing_figure(outcome),
        }

        written: list[Path] = []
        for name, factory in factories.items():
            try:
                figure = factory()
            except Exception as error:  # Broad: one figure must not cost the rest.
                _LOGGER.debug("skipped the %s figure: [%s] %s", name, type(error).__name__, error)
                continue
            written.append(
                save_figure(
                    figure,
                    context.directory,
                    name,
                    figure_format=context.figure_format,
                    dpi=context.figure_dpi,
                )
            )
        return written


def _has_validation(outcome: FitOutcome) -> bool:
    """
    Return whether any epoch recorded a validation loss.

    Parameters
    ----------
    outcome
        The fit outcome.

    Returns
    -------
    bool
        True if at least one epoch has one.
    """
    return any(record.val_loss is not None for record in outcome.history)
