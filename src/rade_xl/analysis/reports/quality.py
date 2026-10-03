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
                "`rade_xl.analysis.metrics.quality.quality_metrics`."
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
