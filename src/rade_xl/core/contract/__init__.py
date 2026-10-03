"""
Typed payloads passed between pipeline stages.

A contract is the promise one stage makes to the next.  Because the promise is
a declared type rather than a convention, a stage can be overridden, cached,
replayed or executed in another process without the neighbouring stages
needing to change.

Contracts are generic over the engine where it matters: a PyTorch run carries
``TensorBatchData``, as every other engine's does, and all of them travel
inside the same ``DataBundle`` and through the same pipeline.

Which modules are pydantic and which are dataclasses is governed by one rule,
stated and justified in ``base.py``: pydantic where the contract must
round-trip through JSON, a frozen dataclass where it holds live data.

Modules
-------
``base.py``
    ``ContractModel``, the strict immutable pydantic base for contract
    metadata, and the documented policy on how validation failures surface.
    [Phase 1, delivered]
``signature.py``
    ``TensorSpec``, ``InputSignature`` (static inputs, dynamic inputs and the
    target) and ``PolicySignature`` (observation and action spaces).  A
    signature is what lets a model be rebuilt from a saved bundle without
    re-running the data build.  [Phase 1, delivered]
``state.py``
    ``FittedState`` -- the abstract base for everything a model fits during
    training and needs again at inference time.  Replaces ad-hoc sidecar files
    with one typed, self-describing object.  [Phase 1, delivered]
``data.py``
    ``DataBundle`` (splits, fitted state, signature, lineage), the engine-side
    payload ``TensorBatchData``, ``SplitIndices`` and
    ``DataLineage`` (source fingerprint, split indices, spec hash).
    [Phase 1, delivered]
``source.py``
    ``BatchSource``, the single protocol every training loop consumes.  This is
    the unification point: fixed datasets, on-policy rollouts, replay buffers
    and differentiable simulators all satisfy it.  [Phase 1, delivered]
``result.py``
    ``EpochRecord``, ``FitOutcome``, ``EvalResult``, ``TrainingResult`` and
    ``Predictions``.  [Phase 1, delivered]
``bundle.py``
    ``ModelBundle`` (what a finished run produces), ``SavedBundle`` (what it
    looks like on disk), ``Manifest`` and ``ManifestEntry``.
    [Phase 1, delivered]

Planned modules
---------------
``experience.py``
    ``Transition``, ``Trajectory`` and ``EpisodeStats`` -- the interactive
    learning payloads.  ``ActionResult`` lands here too, rather than in
    ``result.py``, because it is meaningless outside an interactive run.
    [Phase 7]
"""

__all__: tuple[str, ...] = ()
