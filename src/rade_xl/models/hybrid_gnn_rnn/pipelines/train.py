"""
The hybrid network's training pipeline.

It adds one report and changes nothing else.

Why that is the whole file
---------------------------
The framework's training sequence -- resolve, build data, declare
signature, build model, materialise, prepare hardware, fit, evaluate,
persist, report -- is the sequence this model needs. Every stage it would
have wanted to customise turned out to be customisable from the spec or
from the model definition instead, which is the outcome the framework was
designed for and is worth stating plainly rather than leaving as an
absence.

In particular, three things that *look* like they need a custom pipeline
do not:

- fitting scalers on training rows only, which the data module's split
  handling already enforces along the scenario axis while leaving the
  entity axis free;
- building the graph over the full universe, which is the same mechanism
  seen from the other side -- the entity axis is not a leakage axis;
- caching the graph embedding across an evaluation pass, which the
  ``Precomputable`` capability handles without the pipeline knowing.

If this file ever has to replace a stage rather than extend a hook, that
is a finding about the framework's step granularity and belongs in Phase 2
rather than here. A pipeline override that reimplements a stage is how a
general framework acquires its first special case.
"""

from __future__ import annotations

from ....orchestration.pipelines.train import TrainPipeline
from ..reports import HybridGraphReport

__all__ = ["HYBRID_REPORTS", "HybridTrainPipeline"]

#: Reports this model contributes regardless of what the spec asks for.
#:
#: Not merely defaulted, because a default can be overwritten by a spec that
#: lists its own reports -- and a user narrowing the report set to save time
#: should not thereby switch off the only diagnostic that can tell them the
#: graph is broken. The graph is the model's largest assumption and the one
#: least visible in any metric.
#:
#: Taken from the report class rather than written as a string, which also
#: makes this import load-bearing: enabling the pipeline is what guarantees
#: the report is registered by the time the pipeline resolves it.
HYBRID_REPORTS: tuple[str, ...] = (HybridGraphReport.component_name,)


class HybridTrainPipeline(TrainPipeline):
    """
    The framework's training pipeline, plus the graph diagnostics report.

    A tier-2 override: it extends a hook and leaves the stage sequence
    untouched.
    """

    def report_names(self) -> tuple[str, ...]:
        """
        Return the spec's reports followed by this model's own.

        Appended rather than prepended so the spec's ordering is preserved
        and a summary report that lists the figures produced still runs
        where the user put it. Duplicates are dropped rather than rejected:
        a user who names ``hybrid_graph`` explicitly has asked for
        something already guaranteed, which is a harmless redundancy and
        not an error worth failing a run over.

        Returns
        -------
        tuple of str
            Registered report names, in order, without repeats.
        """
        ordered = dict.fromkeys((*super().report_names(), *HYBRID_REPORTS))
        return tuple(ordered)
