"""
Tests for ``rade_xl.engines.xgboost`` -- the gradient-boosting engine.

This suite is also a test of the framework's abstractions. XGBoost trains in
one call, so if it can satisfy the engine contract and produce the same
artifacts as a neural network, the contract is genuinely engine-agnostic.

Planned modules
---------------
``test_xgboost_engine.py``
    ``DMatrix`` materialisation from a ``DataBundle``, native early stopping
    against the validation split, the boosting history reported through
    ``FitOutcome``, and a persisted model that reloads and predicts
    identically.  [Phase 6]
"""
