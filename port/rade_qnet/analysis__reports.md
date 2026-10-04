# `src/rade_qnet/analysis/reports`

6 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 61 | 2487 | `67191a456d8c17a9` |
| 2 | `base.py` | 189 | 6357 | `25d759f42d87b0a2` |
| 3 | `baselines.py` | 323 | 11108 | `fb7a6c6ed77475da` |
| 4 | `curves.py` | 202 | 6918 | `9db36b4306e4f1d9` |
| 5 | `quality.py` | 317 | 10915 | `233491a91e32745d` |
| 6 | `summary.py` | 329 | 10659 | `225ff04319798e23` |

---

## 1. `src/rade_qnet/analysis/reports/__init__.py`

2487 bytes · SHA-256 `67191a456d8c17a9`

```python
"""
Report writers -- what gets persisted for a run.

A report is declared in the specification, resolved by name, and invoked by a
pipeline with the run context and whatever contract it needs.  It calls into
``metrics`` and ``visuals`` and writes its output beneath the run directory.

Two properties are deliberate.  A report is **optional**: no report may be
required for a run to succeed, and a failing report degrades to a warning
rather than discarding a completed training run.  A report is **declarative**:
enabling one is a specification change, not a pipeline change.

Modules
-------
``base.py``
    The ``Report`` base, the ``ReportContext`` it is given, the
    ``ReportOutcome`` it returns, and the ``@report`` registration decorator.
    ``render_safely`` is what a pipeline calls, and it is what makes "a report
    never fails a run" true rather than aspirational.  [Phase 1, delivered]
``summary.py``
    The human-readable run summary: spec digest, data lineage, training
    outcome, headline metrics against their baselines, and the training curve.
    The only report enabled by default.  [Phase 1, delivered]
``curves.py``
    Training and validation curves plus the raw history as CSV, so the numbers
    behind a figure remain available to a question nobody thought to ask when
    the figure was drawn.  [Phase 2, delivered]
``baselines.py``
    Each headline metric against the naive reference scored on the same split,
    with an explicit verdict on whether the model beat it.  [Phase 2,
    delivered]
``quality.py``
    What the model was trained on: provenance digests, split shares, input
    signature, quality metrics and the fitted transform state.  [Phase 2,
    delivered]

Registration
------------
Importing this package registers all four reports. The import is eager
rather than lazy on purpose: a registry populated only once someone happens
to have imported the right submodule is the classic source of "no report
named 'summary'" from a specification that is perfectly correct, and the
only reliable cure is for registration to be a consequence of importing the
package that owns it.
"""

from .base import Report, ReportContext, ReportOutcome
from .baselines import BaselinesReport
from .curves import CurvesReport
from .quality import QualityReport
from .summary import SummaryReport

__all__ = [
    "BaselinesReport",
    "CurvesReport",
    "QualityReport",
    "Report",
    "ReportContext",
    "ReportOutcome",
    "SummaryReport",
]
```

---

## 2. `src/rade_qnet/analysis/reports/base.py`

6357 bytes · SHA-256 `25d759f42d87b0a2`

```python
"""
The report base, and the rule that keeps reports safe to run.

**A report is never load-bearing.** Nothing downstream may read a report's
output, and no run may fail because a report failed. A report is a rendering
of information that already exists in the bundle -- it is not where that
information lives.

Why that rule is enforced rather than merely stated
---------------------------------------------------
Reporting code is the least-tested code in any pipeline, because its output is
read by people rather than asserted on by tests. It is therefore the most
likely stage to raise on an unusual input: an empty validation split, a metric
absent for one job, a single-epoch history. If a report can fail a run, the
least important stage in the pipeline becomes the one that destroys four hours
of training.

:meth:`Report.render_safely` is the method the pipeline calls. It catches
everything, logs it with the report named, and reports the failure as a
skipped report rather than as a failed run. A caller who genuinely wants a
report failure to be fatal sets ``fail_fast`` on
:class:`~rade_qnet.core.spec.reports.ReportsSpec`, which is an explicit,
reviewable choice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.runtime.components import report
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from ...core.contract.bundle import ModelBundle
    from ...core.runtime.context import RunContext

__all__ = ["Report", "ReportOutcome", "report"]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReportOutcome:
    """
    What a report produced, or why it produced nothing.

    Parameters
    ----------
    name
        Registered report name.
    paths
        Files written. Empty when the report was skipped.
    skipped_reason
        Why nothing was written, or ``None`` if the report succeeded.
        Recorded rather than discarded so a run summary can say "the
        attribution report was skipped because there was no test split"
        instead of silently omitting it -- a missing section that nobody
        explains is read as a bug in the framework.
    """

    name: str
    paths: tuple[Path, ...] = ()
    skipped_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the report ran and wrote its output."""
        return self.skipped_reason is None


@dataclass(frozen=True, slots=True)
class ReportContext:
    """
    What a report is given to work with.

    A narrow, explicit bundle of inputs rather than the whole pipeline, which
    is what stops a report from reaching into pipeline internals and quietly
    becoming load-bearing.

    Parameters
    ----------
    bundle
        The finished run.
    directory
        Where this report should write. Already created.
    figure_format
        Format for any figures, from the run's report specification.
    figure_dpi
        Resolution for raster figures.
    extras
        Additional named values a model's own pipeline chose to publish, for
        model-specific reports. Framework reports never read this.
    """

    bundle: ModelBundle
    directory: Path
    figure_format: str = "png"
    figure_dpi: int = 150
    extras: dict[str, object] = field(default_factory=dict)


class Report(ABC):
    """
    Base class for report writers.

    Subclasses implement :meth:`render` and are registered with
    :func:`~rade_qnet.core.runtime.components.report`. The pipeline calls
    :meth:`render_safely`, never :meth:`render` directly.

    Attributes
    ----------
    component_name
        Registered name, set by the decorator.
    """

    component_name: str

    @abstractmethod
    def render(self, context: ReportContext) -> Sequence[Path]:
        """
        Write the report and return the files written.

        May raise freely. :meth:`render_safely` is what the pipeline calls,
        and it converts an exception into a skipped report. That means a
        report author does not have to defend against every unusual input --
        which is just as well, since the unusual inputs are the ones nobody
        anticipates.

        Implementations should still *skip deliberately* where they can, by
        raising :class:`LookupError` or returning an empty sequence with a
        reason, rather than relying on an incidental ``KeyError``. A
        deliberate skip produces a clear explanation; an incidental one
        produces a stack trace in a log.

        Parameters
        ----------
        context
            What to report on, and where to write it.

        Returns
        -------
        Sequence of Path
            The files written.
        """

    def render_safely(self, context: ReportContext, run: RunContext | None = None) -> ReportOutcome:
        """
        Run :meth:`render`, converting any failure into a skipped report.

        Parameters
        ----------
        context
            What to report on, and where to write it.
        run
            The run context, used to publish written files as artifacts. May
            be ``None`` when a report is rendered outside a run, such as from
            a notebook.

        Returns
        -------
        ReportOutcome
            The files written, or the reason nothing was.
        """
        name = getattr(self, "component_name", type(self).__name__)
        try:
            context.directory.mkdir(parents=True, exist_ok=True)
            paths = tuple(self.render(context))
        except Exception as error:  # Broad by design: see the module docstring.
            _LOGGER.warning(
                "report %r failed and was skipped: [%s] %s",
                name,
                type(error).__name__,
                error,
                exc_info=True,
            )
            return ReportOutcome(
                name=name,
                skipped_reason=f"{type(error).__name__}: {error}",
            )

        if run is not None:
            for path in paths:
                run.track_artifact(path, name=f"{name}/{path.name}")
        _LOGGER.info("report %r wrote %d file(s)", name, len(paths))
        return ReportOutcome(name=name, paths=paths)
```

---

## 3. `src/rade_qnet/analysis/reports/baselines.py`

11108 bytes · SHA-256 `fb7a6c6ed77475da`

```python
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
From :attr:`~rade_qnet.core.contract.result.EvalResult.baseline_metrics`, which
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
```

---

## 4. `src/rade_qnet/analysis/reports/curves.py`

6918 bytes · SHA-256 `9db36b4306e4f1d9`

```python
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
    :class:`~rade_qnet.core.contract.result.FitOutcome` and renders identically.
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
```

---

## 5. `src/rade_qnet/analysis/reports/quality.py`

10915 bytes · SHA-256 `233491a91e32745d`

```python
"""
The data-quality report: what the model was actually trained on.

Reads the lineage the data stage recorded -- scenario and entity counts, the
exact split indices, the quality metrics, the fitted transform state -- and
turns it into a page that answers "was this dataset fit to train on" without
needing the dataset.

Why the quality numbers belong in the run directory
---------------------------------------------------
A model trained on a feature that was 70% missing will train, converge, and
produce metrics. Nothing in the loss curve or the test score says the feature
was mostly absent; the model simply learned from whatever was there and the
run looks healthy.

The failure shows up weeks later when the feature's availability changes and
the model's behaviour changes with it. By then the run directory is the only
evidence of what the training data looked like, so the quality summary has to
be written at training time or it is lost.

Why warnings rather than failures
---------------------------------
A run is not stopped for low completeness, because the threshold that is
unacceptable for one dataset is normal for another -- a sparse corporate-action
feature is 95% missing by nature. The report states the number, names the
threshold it crossed, and leaves the judgement to the reader, which is the
only place the judgement can correctly be made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..metrics.quality import quality_warnings
from .base import Report, ReportContext, report

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from ...core.contract.data import DataLineage
    from ...core.contract.signature import InputSignature

__all__ = ["QualityReport"]

#: File the report writes.
QUALITY_FILENAME = "data_quality.md"

#: Words that mark a quality metric as a fraction of one, and so better read as
#: a percentage.  Matched anywhere in the name rather than as a prefix, because
#: the metrics are qualified by what they measure -- ``feature_completeness``,
#: ``worst_column_completeness``, ``entity_coverage``.
_FRACTIONAL_TOKENS = ("completeness", "staleness", "coverage")


@report("quality")
class QualityReport(Report):
    """
    A Markdown page describing the dataset a run was trained on.

    Reads only the bundle's lineage, signature and fitted state, so it
    re-renders from a saved run without the data.
    """

    def render(self, context: ReportContext) -> Sequence[Path]:
        """
        Write the data-quality page.

        Parameters
        ----------
        context
            The bundle to report on, and where to write.

        Returns
        -------
        Sequence of Path
            The Markdown page.
        """
        lineage = context.bundle.lineage
        sections = [
            "# Data quality",
            self._provenance_section(lineage),
            self._split_section(lineage),
            self._signature_section(context.bundle.signature),
            self._quality_section(lineage.quality),
            self._state_section(context),
        ]
        page = context.directory / QUALITY_FILENAME
        page.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
        return (page,)

    @staticmethod
    def _provenance_section(lineage: DataLineage) -> str:
        """
        Return the dataset dimensions and the digests that identify it.

        The two digests are the part worth keeping. ``source_fingerprint``
        changing between two runs means the input data changed;
        ``spec_digest`` changing means the configuration did. Being able to
        tell those apart is the difference between "the model got worse" and
        "the data got worse".

        Parameters
        ----------
        lineage
            The run's data lineage.

        Returns
        -------
        str
            Markdown text.
        """
        lines = [
            "## Provenance",
            "",
            "| property | value |",
            "| --- | --- |",
            f"| scenarios | {lineage.n_scenarios} |",
            f"| entities | {'n/a' if lineage.n_entities is None else lineage.n_entities} |",
            f"| source fingerprint | `{lineage.source_fingerprint}` |",
            f"| spec digest | `{lineage.spec_digest}` |",
            f"| framework version | {lineage.framework_version} |",
            f"| built at | {lineage.created_at.isoformat()} |",
        ]
        if lineage.notes:
            lines.extend(("", "### Notes", ""))
            lines.extend(f"- **{name}**: {lineage.notes[name]}" for name in sorted(lineage.notes))
        return "\n".join(lines)

    @staticmethod
    def _split_section(lineage: DataLineage) -> str:
        """
        Return the split sizes and their shares of the dataset.

        The share matters as much as the count. A 2% validation split on a
        short history is a handful of scenarios, and an early-stopping
        decision made on a handful of scenarios is noise -- which is visible
        in the percentage and easy to miss in the raw count.

        Parameters
        ----------
        lineage
            The run's data lineage.

        Returns
        -------
        str
            Markdown text.
        """
        sizes = lineage.split_sizes
        if not sizes:
            return (
                "## Splits\n\nThe lineage recorded no split indices, so the "
                "train/validation/test shares of this dataset are unknown."
            )

        lines = [
            "## Splits",
            "",
            "| split | scenarios | share of dataset |",
            "| --- | --- | --- |",
        ]
        for name in sorted(sizes):
            count = sizes[name]
            share = count / lineage.n_scenarios if lineage.n_scenarios else 0.0
            lines.append(f"| {name} | {count} | {share:.1%} |")

        # The shortfall is the boundary gap: scenarios discarded so that no
        # sequence window straddles a split edge.  Naming it explicitly stops
        # the missing rows looking like an arithmetic error.
        discarded = lineage.n_scenarios - sum(sizes.values())
        if discarded > 0:
            lines.extend(
                (
                    "",
                    f"{discarded} scenario(s) belong to no split. These are the "
                    f"boundary gaps between splits, discarded so that no sequence "
                    f"window spans two splits and no training row can see a "
                    f"validation observation.",
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _signature_section(signature: InputSignature) -> str:
        """
        Return the input signature the model was built against.

        Parameters
        ----------
        signature
            The run's input signature.

        Returns
        -------
        str
            Markdown text.
        """
        lines = [
            "## Input signature",
            "",
            "The shapes the model was built against. A saved model can only be "
            "served data matching these.",
            "",
            "| input | kind | shape |",
            "| --- | --- | --- |",
        ]
        for name in sorted(signature.dynamic):
            lines.append(f"| {name} | dynamic | `{signature.dynamic[name].describe()}` |")
        for name in sorted(signature.static):
            lines.append(f"| {name} | static | `{signature.static[name].describe()}` |")
        lines.append(f"| target | target | `{signature.target.describe()}` |")
        return "\n".join(lines)

    @staticmethod
    def _quality_section(metrics: Mapping[str, float]) -> str:
        """
        Return the recorded quality metrics, with any warnings called out.

        Parameters
        ----------
        metrics
            The lineage's quality metrics.

        Returns
        -------
        str
            Markdown text.
        """
        if not metrics:
            return (
                "## Quality metrics\n\nThis run recorded no quality metrics, so "
                "completeness and staleness are unknown. The training pipeline "
                "populates `DataLineage.quality` from "
                "`rade_qnet.analysis.metrics.quality.quality_metrics`."
            )

        lines = ["## Quality metrics", "", "| metric | value |", "| --- | --- |"]
        lines.extend(
            f"| {name} | {_format_metric(name, metrics[name])} |" for name in sorted(metrics)
        )

        warnings = quality_warnings(metrics)
        if warnings:
            lines.extend(("", "### Warnings", ""))
            lines.extend(f"- {warning}" for warning in warnings)
            lines.extend(
                (
                    "",
                    "These did not stop the run. Whether they matter depends on the "
                    "dataset -- see this module's docstring.",
                )
            )
        else:
            lines.extend(("", "No quality thresholds were crossed."))
        return "\n".join(lines)

    @staticmethod
    def _state_section(context: ReportContext) -> str:
        """
        Return a description of the fitted transform state.

        Included because the state is what makes a prediction reproducible: a
        model served with a different scaling mean is a different model, and
        this is the only human-readable record of which one was used.

        Parameters
        ----------
        context
            The bundle to report on.

        Returns
        -------
        str
            Markdown text.
        """
        description = context.bundle.state.describe()
        if not description:
            return (
                "## Fitted transform state\n\nThis run fitted no transform state, "
                "so inputs were used as supplied."
            )

        lines = [
            "## Fitted transform state",
            "",
            "Fitted on the training split only. Serving a model with different "
            "state than it was trained with silently changes its predictions.",
            "",
            "| property | value |",
            "| --- | --- |",
        ]
        lines.extend(f"| {name} | {description[name]} |" for name in sorted(description))
        return "\n".join(lines)


def _format_metric(name: str, value: float) -> str:
    """
    Format one quality metric for display.

    Parameters
    ----------
    name
        Metric name.
    value
        Metric value.

    Returns
    -------
    str
        A percentage for the metrics that are fractions by construction, and a
        plain number otherwise -- printing an entity count as ``1200.0%``
        would be worse than useless.
    """
    if any(token in name for token in _FRACTIONAL_TOKENS):
        return f"{value:.1%}"
    return f"{value:.6g}"
```

---

## 6. `src/rade_qnet/analysis/reports/summary.py`

10659 bytes · SHA-256 `225ff04319798e23`

```python
"""
The default report: one Markdown page describing a run.

Enabled by default, and deliberately the only report that is. It answers the
questions someone actually asks when they open a run directory: what was
trained, from what configuration, against what data, how did it do, and is
that any better than doing nothing.

Markdown rather than HTML or a notebook, for three reasons: it renders in
every code host and editor without a build step, it diffs legibly so two runs
can be compared with ordinary tools, and it stays readable as plain text in a
terminal over SSH -- which is where these files are read more often than
anyone expects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...core.runtime.logging import get_logger
from ..visuals.export import save_figure
from ..visuals.primitives import metric_comparison_figure, training_curve_figure
from .base import Report, ReportContext, report

__all__ = ["SummaryReport"]

_LOGGER = get_logger(__name__)

#: File the report writes.
SUMMARY_FILENAME = "summary.md"


@report("summary")
class SummaryReport(Report):
    """
    A one-page Markdown summary of a run, with the training curve.

    Reads only the bundle. It needs no engine, no data and no model object, so
    it works for every model and every engine without modification -- and can
    be re-rendered from a saved bundle months later.
    """

    def render(self, context: ReportContext) -> Sequence[Path]:
        """
        Write the summary page and its figures.

        Parameters
        ----------
        context
            The bundle to summarise, and where to write.

        Returns
        -------
        Sequence of Path
            The Markdown page, and any figures written beside it.
        """
        written: list[Path] = []
        sections: list[str] = []

        sections.append(self._heading(context))
        sections.append(self._configuration_section(context))
        sections.append(self._data_section(context))
        sections.append(self._training_section(context))
        sections.append(self._metrics_section(context))

        figure_paths = self._write_figures(context)
        written.extend(figure_paths)
        if figure_paths:
            sections.append(self._figures_section(figure_paths))

        page = context.directory / SUMMARY_FILENAME
        page.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
        written.insert(0, page)
        return tuple(written)

    def _heading(self, context: ReportContext) -> str:
        """
        Return the page title and identifying line.

        Parameters
        ----------
        context
            The report context.

        Returns
        -------
        str
            Markdown.
        """
        bundle = context.bundle
        identifier = bundle.manifest.identifier if bundle.manifest else "(unwritten)"
        return f"# Run summary: {identifier}"

    def _configuration_section(self, context: ReportContext) -> str:
        """
        Return the configuration section.

        Parameters
        ----------
        context
            The report context.

        Returns
        -------
        str
            Markdown.
        """
        bundle = context.bundle
        rows = {
            "model": bundle.spec.model.describe(),
            "task": bundle.spec.task,
            "seed applied": bundle.result.seed,
            "spec digest": bundle.lineage.spec_digest,
            "framework version": bundle.lineage.framework_version,
        }
        if bundle.manifest is not None:
            rows["engine"] = bundle.manifest.engine
            rows["version"] = bundle.manifest.version
            rows["written at"] = bundle.manifest.created_at.isoformat(timespec="seconds")
        return "## Configuration\n\n" + _table(("field", "value"), rows.items())

    def _data_section(self, context: ReportContext) -> str:
        """
        Return the data lineage section.

        Parameters
        ----------
        context
            The report context.

        Returns
        -------
        str
            Markdown.
        """
        lineage = context.bundle.lineage
        rows: list[tuple[str, object]] = [
            ("source fingerprint", lineage.source_fingerprint),
            ("scenarios", lineage.n_scenarios),
        ]
        if lineage.n_entities is not None:
            rows.append(("entities", lineage.n_entities))
        # Split *sizes* rather than the indices themselves: the indices are in
        # the bundle for reproducibility, but pasting ten thousand integers
        # into a summary page makes the page useless.
        for split, indices in sorted(lineage.split_indices.items()):
            rows.append((f"{split} scenarios", len(indices)))
        for key, value in sorted(lineage.notes.items()):
            rows.append((f"note: {key}", value))
        return "## Data\n\n" + _table(("field", "value"), rows)

    def _training_section(self, context: ReportContext) -> str:
        """
        Return the training outcome section.

        Parameters
        ----------
        context
            The report context.

        Returns
        -------
        str
            Markdown.
        """
        fit = context.bundle.result.fit
        rows: list[tuple[str, object]] = [
            ("epochs recorded", fit.n_epochs),
            ("monitored metric", fit.monitor),
            ("best epoch", "-" if fit.best_epoch is None else fit.best_epoch),
            (
                "best value",
                "-" if fit.best_monitor_value is None else f"{fit.best_monitor_value:.6g}",
            ),
            ("stopped early", "yes" if fit.stopped_early else "no"),
            # Stated explicitly because when this is "no", the reported
            # metrics and the saved weights describe different models -- which
            # a reader must know before comparing this run with another.
            ("best weights restored", "yes" if fit.restored_best else "no"),
            ("total time", f"{fit.total_seconds:.1f}s"),
        ]
        return "## Training\n\n" + _table(("field", "value"), rows)

    def _metrics_section(self, context: ReportContext) -> str:
        """
        Return the metrics section, with baselines where available.

        Parameters
        ----------
        context
            The report context.

        Returns
        -------
        str
            Markdown.
        """
        evaluations = context.bundle.result.evaluations
        if not evaluations:
            return "## Metrics\n\nNo evaluation was recorded for this run."

        lines: list[str] = ["## Metrics"]
        for split in sorted(evaluations):
            evaluation = evaluations[split]
            units = (
                "original target units"
                if evaluation.in_original_units
                else "**model-internal units -- not comparable across runs**"
            )
            lines.append(f"### {split} ({evaluation.n_samples} samples, {units})")
            headers = ("metric", "model", "baseline")
            rows = [
                (
                    name,
                    f"{evaluation.metrics[name]:.6g}",
                    (
                        f"{evaluation.baseline_metrics[name]:.6g}"
                        if name in evaluation.baseline_metrics
                        else "-"
                    ),
                )
                for name in sorted(evaluation.metrics)
            ]
            lines.append(_table(headers, rows))
        return "\n\n".join(lines)

    def _write_figures(self, context: ReportContext) -> tuple[Path, ...]:
        """
        Write the figures this report includes.

        Parameters
        ----------
        context
            The report context.

        Returns
        -------
        tuple of Path
            The figures written. Empty if there was nothing to plot -- a
            one-shot fit with no recorded history, for instance.
        """
        bundle = context.bundle
        fit = bundle.result.fit
        written: list[Path] = []

        if fit.n_epochs > 1:
            figure = training_curve_figure(
                fit.curve("train_loss"),
                fit.curve("val_loss") if any(r.val_loss is not None for r in fit.history) else (),
                best_epoch=fit.best_epoch,
            )
            written.append(
                save_figure(
                    figure,
                    context.directory,
                    "training_curve",
                    figure_format=context.figure_format,
                    dpi=context.figure_dpi,
                )
            )

        for split, evaluation in sorted(bundle.result.evaluations.items()):
            if not evaluation.metrics:
                continue
            figure = metric_comparison_figure(
                evaluation.metrics,
                evaluation.baseline_metrics or None,
                title=f"{split} metrics",
            )
            written.append(
                save_figure(
                    figure,
                    context.directory,
                    f"metrics_{split}",
                    figure_format=context.figure_format,
                    dpi=context.figure_dpi,
                )
            )
        return tuple(written)

    def _figures_section(self, paths: Sequence[Path]) -> str:
        """
        Return the figures section, embedding each figure.

        Parameters
        ----------
        paths
            The figures written.

        Returns
        -------
        str
            Markdown.
        """
        lines = ["## Figures"]
        for path in paths:
            # Relative links, so the directory can be moved or archived and
            # the page still renders.
            lines.append(f"![{path.stem.replace('_', ' ')}]({path.name})")
        return "\n\n".join(lines)


def _table(
    headers: Sequence[str],
    rows: Sequence[tuple[object, ...]] | Mapping[str, object],
) -> str:
    """
    Render a Markdown table.

    Parameters
    ----------
    headers
        Column headers.
    rows
        Row tuples, or a mapping rendered as two columns.

    Returns
    -------
    str
        A Markdown table.
    """
    materialised = [tuple(row) for row in (rows.items() if isinstance(rows, Mapping) else rows)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in materialised)
    return "\n".join(lines)
```

