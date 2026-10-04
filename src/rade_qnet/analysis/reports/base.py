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

from ...core.lifecycle.components import report
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from ...core.contract.bundle import ModelBundle
    from ...core.lifecycle.context import RunContext

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
    :func:`~rade_qnet.core.lifecycle.components.report`. The pipeline calls
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
