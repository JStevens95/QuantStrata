"""
Metrics -- pure functions over arrays.

Every metric takes arrays and returns numbers.  No metric reads configuration,
touches the filesystem or holds state, which makes each one independently
testable against a hand-computed value.

One rule is enforced by the pipelines that call these: metrics are always
computed in the *original* target units.  A mean absolute error reported in
standardised space is not a quantity anyone can act on, so the evaluate
pipeline inverts target transforms before this package is reached.

Modules
-------
``regression.py``
    Error and agreement measures: mean absolute error, root mean squared
    error, bias, coefficient of determination, directional accuracy, and the
    naive baselines every headline metric is reported against.
    [Phase 1, delivered]

Planned modules
---------------
``episode.py``
    Episode return, length, terminal wealth and risk-adjusted statistics for
    interactive runs.  [Phase 7]
``drift.py``
    Distribution distance between a baseline and a live window, for monitoring
    a deployed model.  [Phase 5]
``quality.py``
    Input data quality: completeness, staleness and coverage.  A model trained
    on 40 per cent missing history should say so in its own artifacts.
    [Phase 2]
"""

__all__: tuple[str, ...] = ()
