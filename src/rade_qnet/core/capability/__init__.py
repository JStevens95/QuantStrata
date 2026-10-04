"""
The model-facing interface: what a model must provide, and what it may.

A model registers itself with the framework by subclassing a base in
``definition.py``.  The base demands very little -- build a model object from a
spec and a signature -- which is what keeps a simple model to a single file.

Anything beyond that minimum is an *opt-in capability*: a narrow protocol a
model may implement to unlock extra behaviour.  The framework checks for a
capability with a runtime ``isinstance`` test, so a model never pays for a
feature it does not use, and adding a capability never breaks existing models.

Modules
-------
``definition.py``
    ``ModelDefinition`` and its two specialisations, ``PredictorDefinition``
    (learns from a fixed dataset) and ``PolicyDefinition`` (learns by
    interacting with an environment), plus the ``@model`` registration
    decorator re-exported from ``core.runtime.components`` so a model author
    needs one import.  [Phase 1, delivered]
``protocols.py``
    The opt-in capabilities:

    ``StaticInputs``
        The model consumes inputs that are constant across every batch (a
        graph, entity features, index arrays).  Lets the framework upload them
        to the device once instead of once per batch.
    ``Precomputable``
        An expensive encoding of the static inputs can be computed once per
        evaluation pass and reused.
    ``CustomStep``
        The model owns its loss computation, for multi-objective or otherwise
        non-standard training.
    ``Routable``
        The model can serve as a member of a job set, declaring which targets
        it was actually trained on.
    ``Inductive``
        The model can predict for entities that were absent during training.
    [Phase 1 delivered, extended in Phase 3]

Planned modules
---------------
``simple.py``
    Convenience bases (``TabularModel`` and friends) that supply a standard
    data module and split so a straightforward model needs no data code at
    all.  [Phase 2]
"""

__all__: tuple[str, ...] = ()
