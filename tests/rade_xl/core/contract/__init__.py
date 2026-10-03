"""
Tests for ``rade_xl.core.contract`` -- the typed stage hand-offs.

A contract is only useful if it is enforced, so these tests check that a
malformed payload is rejected at construction rather than surfacing as a shape
error inside a training loop. They also pin the behaviour that makes contracts
portable: a contract must survive being written to disk and read back, because
that is how a job crosses a process boundary and how a bundle is reloaded
months later.

Planned modules
---------------
``test_contract_signature.py``
    ``TensorSpec`` and ``InputSignature`` construction and equality -- the
    basis for rebuilding a model from a bundle without the original data.
    [Phase 1]
``test_contract_data.py``
    ``DataBundle``, ``TensorBatchData`` and ``DataLineage``.
    [Phase 1]
``test_contract_source.py``
    The ``BatchSource`` protocol, checked against a trivial implementation to
    prove the protocol is satisfiable without inheritance.  [Phase 1]
``test_contract_result.py``
    ``FitOutcome``, ``TrainingResult``, ``EvalResult`` and ``Predictions``.
    [Phase 1]
``test_contract_bundle.py``
    ``ModelBundle`` and ``Manifest``.  [Phase 1]
``test_contract_state.py``
    ``FittedState`` save and load round-trips, and target inversion.  [Phase 1]
``test_contract_experience.py``
    ``Transition``, ``Trajectory`` and ``EpisodeStats``.  [Phase 7]
"""
