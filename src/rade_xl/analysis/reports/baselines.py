"""
The baseline report: every headline metric next to the score of doing nothing.

A mean absolute error of 0.03 is either excellent or embarrassing and the
number alone does not say which. The only thing that makes it interpretable is
what a predictor with no information scores on the same split.

Why this is a report rather than a note in the summary
------------------------------------------------------
:class:`~.summary.SummaryReport` already prints baseline metrics where the
evaluation carried them. This report exists for the case where it did not, and
for the comparison the summary has no room for: the *ratio* per metric, and an
explicit verdict on whether the model beat the reference at all.

The verdict is the point. A model that loses to predicting zero is a result,
not an error, and it should be stated in words in a file somebody will read --
not left implicit in two numbers a few lines apart. The framework's own
baselines exist for the same reason: a graph-temporal network that cannot beat
a ridge regression has not earned its complexity.

Where the baseline numbers come from
------------------------------------
From :attr:`~rade_xl.core.contract.result.EvalResult.baseline_metrics`, which
the evaluate stage computes. This report does not compute them itself, and
that is deliberate: a baseline recomputed here would need the targets, and a
report that needs the data cannot be re-rendered from a saved bundle months
later. Reading what the evaluation recorded keeps the report a pure function of
the bundle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.runtime.logging import get_logger
from ..visuals.export import save_figure
from ..visuals.primitives import metric_comparison_figure
from .base import Report, ReportContext, report

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from ...core.contract.result import EvalResult

__all__ = ["BaselinesReport"]

_LOGGER = get_logger(__name__)

#: File the report writes.
BASELINES_FILENAME = "baselines.md"

#: Metrics where a *lower* value is better. Needed because the verdict is a
#: comparison and the direction is not inferable from the name -- an r-squared
#: of 0.9 beating a baseline's 0.1 and a mean absolute error of 0.1 beating a
#: baseline's 0.9 are both wins.
_LOWER_IS_BETTER = frozenset(
    {"mae", "mean_absolute_error", "rmse", "root_mean_squared_error", "abs_bias"}
)

#: Metrics where a higher value is better.
_HIGHER_IS_BETTER = frozenset({"r2", "r_squared", "directional_accuracy"})


@report("baselines")
class BaselinesReport(Report):
    """
    A Markdown comparison of the model against its naive reference.

    Reads only the bundle, so it re-renders from a saved run without the data
    or the model.
    """

    def render(self, context: ReportContext) -> Sequence[Path]:
        """
        Write the comparison page and one figure per split.

        Parameters
        ----------
        context
            The bundle to report on, and where to write.

        Returns
        -------
        Sequence of Path
            The Markdown page, and any figures written beside it.

        Raises
        ------
        LookupError
            If no split recorded baseline metrics. A deliberate skip: without
            a reference there is nothing to compare, and the correct outcome
            is a clear explanation rather than a page of empty tables.
        """
        evaluations = context.bundle.result.evaluations
        comparable = {
            name: evaluation
            for name, evaluation in evaluations.items()
            if evaluation.baseline_metrics
        }
        if not comparable:
            raise LookupError(
                f"no split recorded baseline metrics, so there is nothing to "
                f"compare against. Scored splits were {sorted(evaluations)}; the "
                f"evaluate stage populates EvalResult.baseline_metrics"
            )

        sections = ["# Baseline comparison", self._preamble()]
        for name in sorted(comparable):
            sections.append(self._split_section(name, comparable[name]))

        written = list(self._write_figures(comparable, context))
        page = context.directory / BASELINES_FILENAME
        page.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
        return (page, *written)

    @staticmethod
    def _preamble() -> str:
        """
        Return the explanatory paragraph.

        Returns
        -------
        str
            Markdown text.
        """
        return (
            "Every metric below is shown against a naive predictor evaluated on "
            "the same split. The baseline uses only information the model also "
            "had, so beating it is the minimum bar rather than an achievement."
        )

    def _split_section(self, split: str, evaluation: EvalResult) -> str:
        """
        Return one split's comparison table and verdict.

        Parameters
        ----------
        split
            Split name.
        evaluation
            The split's metrics and baseline metrics.

        Returns
        -------
        str
            Markdown text.
        """
        lines = [
            f"## {split} ({evaluation.n_samples} samples)",
            "",
            "| metric | model | baseline | ratio | verdict |",
            "| --- | --- | --- | --- | --- |",
        ]

        wins = 0
        comparable = 0
        for name in sorted(evaluation.metrics):
            model_value = evaluation.metrics[name]
            if name not in evaluation.baseline_metrics:
                lines.append(f"| {name} | {model_value:.6g} | — | — | — |")
                continue

            baseline_value = evaluation.baseline_metrics[name]
            ratio = self._ratio(model_value, baseline_value)
            verdict = self._verdict(name, model_value, baseline_value)
            if verdict != "—":
                comparable += 1
                wins += verdict == "better"
            lines.append(
                f"| {name} | {model_value:.6g} | {baseline_value:.6g} | {ratio} | {verdict} |"
            )

        lines.extend(("", self._summary_sentence(split, wins=wins, total=comparable)))
        if not evaluation.in_original_units:
            lines.extend(
                (
                    "",
                    "> **These metrics are not in the original target units.** "
                    "They were computed in transformed space, so their magnitudes "
                    "are not directly interpretable.",
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _ratio(model_value: float, baseline_value: float) -> str:
        """
        Return the model-to-baseline ratio, as text.

        Parameters
        ----------
        model_value
            The model's metric.
        baseline_value
            The baseline's metric.

        Returns
        -------
        str
            The ratio, or an em dash when the baseline is zero. A zero
            baseline makes the ratio undefined rather than infinite, and
            printing ``inf`` would read as a result.
        """
        if baseline_value == 0.0:
            return "—"
        return f"{model_value / baseline_value:.3f}"

    @staticmethod
    def _verdict(name: str, model_value: float, baseline_value: float) -> str:
        """
        Return whether the model beat the baseline on one metric.

        Parameters
        ----------
        name
            Metric name.
        model_value
            The model's metric.
        baseline_value
            The baseline's metric.

        Returns
        -------
        str
            ``better``, ``worse``, ``tied``, or an em dash when the metric's
            direction is unknown. An unknown direction is reported as unknown
            rather than guessed, because guessing it backwards would print a
            confident and wrong verdict.
        """
        if name in _LOWER_IS_BETTER:
            better = model_value < baseline_value
        elif name in _HIGHER_IS_BETTER:
            better = model_value > baseline_value
        else:
            return "—"

        if model_value == baseline_value:
            return "tied"
        return "better" if better else "worse"

    @staticmethod
    def _summary_sentence(split: str, *, wins: int, total: int) -> str:
        """
        Return the plain-language verdict for one split.

        Parameters
        ----------
        split
            Split name.
        wins
            Metrics the model won.
        total
            Metrics with a known direction.

        Returns
        -------
        str
            Markdown text.
        """
        if total == 0:
            return (
                "No metric here has a known direction, so no verdict is given. "
                "Add the metric to the direction tables in this module to get one."
            )
        if wins == total:
            return f"The model beats the baseline on all {total} comparable metric(s)."
        if wins == 0:
            return (
                f"**The model does not beat the baseline on any of the {total} "
                f"comparable metric(s) for {split}.** This is a result, not an "
                f"error: on this split the model carries no advantage over a "
                f"predictor with no information."
            )
        return (
            f"The model beats the baseline on {wins} of {total} comparable "
            f"metric(s) for {split}; the remainder are worse or tied."
        )

    @staticmethod
    def _write_figures(evaluations: Mapping[str, EvalResult], context: ReportContext) -> list[Path]:
        """
        Write one metric-comparison figure per split.

        Parameters
        ----------
        evaluations
            The comparable evaluations.
        context
            Where to write, and in what format.

        Returns
        -------
        list of Path
            The figures written.
        """
        written: list[Path] = []
        for name in sorted(evaluations):
            evaluation = evaluations[name]
            try:
                figure = metric_comparison_figure(
                    evaluation.metrics,
                    evaluation.baseline_metrics,
                    title=f"{name}: model against baseline",
                )
            except Exception as error:  # Broad: one figure must not cost the page.
                _LOGGER.debug(
                    "skipped the %s comparison figure: [%s] %s",
                    name,
                    type(error).__name__,
                    error,
                )
                continue
            written.append(
                save_figure(
                    figure,
                    context.directory,
                    f"baseline_{name}",
                    figure_format=context.figure_format,
                    dpi=context.figure_dpi,
                )
            )
        return written
