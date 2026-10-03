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
