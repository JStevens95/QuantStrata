"""
Who coordinates the work.

``orchestration`` is the conductor.  It holds the four pipelines that define the
model lifecycle, the fan-out that runs one model across a list of jobs, and the
placement layer that decides which process or GPU each job lands on.

The separation inside this package is between *sequence* and *placement*:

``pipelines``
    The ordered stages of a single run.  Deterministic and placement-agnostic.
``jobs``
    One model, many jobs -- shared defaults, per-job overrides (including
    different architecture complexity per job), and the aggregated summary.
``compute``
    Where a unit of work executes: in-process, across processes, across GPUs,
    or on a cluster.

Keeping these apart is what makes a job set reproducible: running sequentially
and running across eight processes must produce byte-identical artifacts, which
is only achievable if the pipeline cannot observe its own placement.

Dependency rule
---------------
May import: ``core``, ``sources``, ``engines``, ``storage``, ``analysis``.
May not import: ``models``.  A pipeline resolves a model through the registry,
never by importing it, so the framework never depends on the model library.
"""

__all__: tuple[str, ...] = ()
