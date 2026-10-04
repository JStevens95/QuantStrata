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
