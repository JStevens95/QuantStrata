"""
How optimisation actually happens -- one adapter per library.

An engine is the only place in the framework that is allowed to know about a
specific training library.  It answers a fixed set of questions for its
library: how is a model instantiated, how is data materialised into the
library's native form, how is a fit run, how is a checkpoint written, how is a
prediction produced.

Because the pipelines above talk only to this interface, adding a backend is a
new sub-package and a registration line -- not a change to any pipeline.

Sub-packages
------------
``torch``
    PyTorch.  The richest engine: gradient-based loops, learners for both
    supervised and reinforcement learning, mixed precision, distributed
    training and lazy-parameter materialisation.
``xgboost``
    Gradient-boosted trees.  Fits in one call, so it implements the engine
    interface with a no-op training loop.
``sklearn``
    scikit-learn estimators, for baselines and sanity checks.

Modules
-------
``base.py``
    The ``Engine`` protocol every adapter satisfies, ``EngineCapabilities``
    (what a pipeline may ask of an engine) and ``ModelHandle`` (a prepared
    model and its apparatus).  Scheduled for Phase 1 and delivered in Phase 2
    instead: Phase 1 built the pipeline skeleton against the *model
    definition*, so nothing consumed an engine interface and writing one would
    have meant designing a contract with no consumer -- the risk that phase's
    own risk table names as its largest.  [Phase 2]

Dependency rule
---------------
May import: ``core``, ``sources``.
May not import: ``orchestration``, ``models``, ``analysis``.
An engine never decides *where* it runs -- that is ``orchestration.compute`` --
and never writes a report.
"""

__all__: tuple[str, ...] = ()
