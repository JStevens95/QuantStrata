"""
Tests for ``rade_qnet.analysis.metrics``.

Each metric is checked against a value computed by hand, not against a second
implementation. Comparing two implementations of the same formula only proves
they agree, which they will even when both are wrong.

Planned modules
---------------
``test_metrics_regression.py``
    Error and agreement measures, including the degenerate cases -- a constant
    target, a perfect prediction, a single observation -- which is where most
    metric implementations divide by zero.  [Phase 1]
``test_metrics_quality.py``
    Completeness, staleness and coverage.  [Phase 2]
``test_metrics_drift.py``
    Distribution distance, including the identical-distribution case that must
    report zero.  [Phase 5]
``test_metrics_episode.py``
    Episode return, length and risk-adjusted statistics.  [Phase 7]
"""
