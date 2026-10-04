"""
Placeholder for model test suites.

Model-specific tests are deliberately deferred: the framework is being proven
model-independently first, so that when the flagship is refactored into it, a
failure is attributable to the model rather than to foundations still in flux.

This package exists so the mirror is complete and the gap is visible. When a
model lands, its suite is added beneath here following the source layout --
``models/hybrid_gnn_rnn/layers/test_layers_gnn.py`` and so on -- and the
mirroring rule in ``test_scaffold.py`` is extended to cover it.

What a model suite must contain is not left to taste. Every model is run
through ``rade_qnet.testkit.conformance``, and a model refactored from an
existing implementation is additionally held to the parity levels in
``src/rade_qnet/docs/phases/PHASE_0_BASELINE.md``.
"""
