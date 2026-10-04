"""
What every run can prove about itself, whatever it computed.

Three guarantees, and they are the three things a model-risk review asks for.
Together they are the difference between a number and a number you can defend.

``seeding``
    The same numbers, twice.  The same specification on the same data
    produces the same weights, and a fanned-out job set produces the same
    results run in parallel as run one after another.

``hashing``
    The same identity, on another machine, in a year.  A digest of equal
    inputs must be equal in a different process, on a different day --
    which Python's built-in ``hash`` does not give you, because string
    hashing is salted per interpreter.  These digests are cache keys and
    bundle provenance, so an unstable one quietly means a cache that never
    hits and a lineage that cannot be checked.

``logging``
    The narrative, legible among forty concurrent runs.  Identifiers live in
    context variables rather than in a logger threaded through every
    signature, so code that has nothing to do with logging does not acquire
    a logging parameter.

Why these are not "utilities"
------------------------------
They were, in a ``runtime`` package that also held the registry, the pipeline
base and the run context -- eight modules whose only shared property was
being needed everywhere.  That is a bin, and a bin is where things go to stop
being findable.

What separates these three from the rest of that package is *who reads them*.
:mod:`rade_qnet.core.lifecycle` is read by someone extending the framework:
how do I register a model, override a stage, attach a hook.  This package is
read by someone who has to answer for a result: why is this number what it
is, and can you produce it again.

Dependency note
---------------
``seeding`` seeds what is installed without importing it.  ``core`` may not
depend on a training library, yet seeding PyTorch is exactly the kind of
thing that belongs here, so an engine registers a seeding callback when it is
imported and ``seed_everything`` calls whatever has registered.  The torch
half lives in :mod:`rade_qnet.engines.torch.hardware.determinism`, because
what it seeds is CUDA generators and cuDNN algorithm selection -- device
facts, not framework facts.
"""

__all__: tuple[str, ...] = ()
