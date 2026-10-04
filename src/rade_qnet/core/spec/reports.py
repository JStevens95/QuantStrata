"""
Which report writers a run produces.

Reports are declarative: enabling one is a specification change, resolved by
name through the component registry, not a pipeline change.

The default for ``fail_fast`` is the important setting in this module. A report
that raises produces a warning and the run completes. Discarding four hours of
training because a figure failed to render is not a trade anyone would make
deliberately, so it is not the default -- but it is available, because during
development a silently skipped report is worse than a loud one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec

__all__ = ["ReportsSpec"]


class ReportsSpec(Spec):
    """
    Report selection and output settings.

    Parameters
    ----------
    enabled
        Registered report names, in the order they will run. A tuple rather
        than a set because order is occasionally meaningful -- a summary
        report that lists the figures produced must run after them.
    directory_name
        Subdirectory of the run directory that reports write beneath.
    fail_fast
        Whether a failing report should fail the run. See the module
        docstring.
    figure_format
        Image format for saved figures.
    figure_dpi
        Resolution for raster formats. Ignored for ``svg`` and ``pdf``.
    """

    enabled: tuple[str, ...] = ("summary",)
    directory_name: str = "reports"
    fail_fast: bool = False
    figure_format: Literal["png", "svg", "pdf"] = "png"
    figure_dpi: int = Field(default=150, ge=50, le=600)

    @model_validator(mode="after")
    def _reject_duplicate_reports(self) -> ReportsSpec:
        """
        Reject a report listed more than once.

        A duplicate would run twice and overwrite its own output, so it is
        always a mistake -- usually a merge artefact from combining job-set
        defaults with a per-job override.

        Returns
        -------
        ReportsSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any name appears more than once.
        """
        seen: set[str] = set()
        duplicates = sorted({name for name in self.enabled if name in seen or seen.add(name)})
        if duplicates:
            raise SpecError(f"reports listed more than once: {', '.join(duplicates)}")
        return self
