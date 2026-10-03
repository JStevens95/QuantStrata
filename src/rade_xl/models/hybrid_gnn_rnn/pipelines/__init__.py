"""
Pipeline overrides for the hybrid graph-temporal network.

A model may override any of the four lifecycle pipelines independently,
which is why there is a file per pipeline rather than one file for all of
them. A model that needs a custom evaluation but standard training should
not have to open a file that also contains training code.

Every override here subclasses the framework base and extends a hook. None
of them replaces a stage, and that is the measure of whether the framework's
step granularity is right: an override that reimplements a stage is how a
general framework acquires its first special case.

Modules
-------
``train.py``
    Adds the graph diagnostics report. The report is contributed regardless
    of what the spec asks for, because a user narrowing the report set
    should not thereby switch off the only diagnostic that can tell them
    the graph is broken.
``eval.py``
    Adds a per-target breakdown to the result. A replication model's
    aggregate error hides the one target it cannot replicate, which is
    usually the one a desk most wants flagged.
``tune.py``
    Discards proposals whose block width does not divide by its head count,
    so no trial is spent discovering a configuration that cannot build.

Why there is no ``infer.py``
-----------------------------
It was planned, to handle target instruments unseen during training by
transferring from their nearest neighbours in the learned output space.
That is still the right feature and it is not implementable yet.

An entity absent from training is also absent from the fitted state a
reloaded model applies. For this model that means no graph node, no entry
in ``target_indices`` and no row in the encoded attribute table -- all three
come from the saved ``HybridState``, not from the inference source. There is
nothing to transfer *to*, and inventing a node at inference time would make
the transfer a function of a graph the model was never trained against.

The framework side of this is recorded as deviation 8.3 of the Phase 5
charter, which also explains why ``Inductive`` deliberately stayed a
declaration rather than growing a hook nothing could implement. The
inference pipeline is correct as it stands: it refuses an unseen entity
rather than returning a confident default, which is the behaviour that
matters until the mechanism exists.
"""

from .eval import HybridEvalPipeline
from .train import HybridTrainPipeline
from .tune import HybridTunePipeline

__all__ = ["HybridEvalPipeline", "HybridTrainPipeline", "HybridTunePipeline"]
