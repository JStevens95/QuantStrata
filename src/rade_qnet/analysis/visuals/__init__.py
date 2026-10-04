"""
Figure factories -- shared across the whole framework.

Every function here has the same shape: it accepts data and styling, and
returns a figure.  It does not save, show, close or configure anything global.
That purity is what makes these functions reusable from a report writer, a
notebook, a dashboard or a unit test, and it is what makes them testable at all
-- a test can assert on axis labels, series count and data limits without
rendering to a file.

Modules
-------
``style.py``
    The house style: palettes, figure sizes, fonts and a context manager that
    applies them without mutating global state for the rest of the process.
    Also the reason this package never imports ``pyplot``.
    [Phase 1, delivered]
``figures.py``
    Shared building blocks -- training curve, prediction scatter with a parity
    line, residual histogram, metric comparison against a baseline -- that the
    higher-level modules compose.  [Phase 1, delivered]
``export.py``
    The one place that writes a figure to disk, in a consistent format and
    resolution.  Separate from the factories by design.  [Phase 1, delivered]
``data.py``
    Data diagnostics: split layout over the scenario axis, target distribution
    per split, missingness map, feature correlation.  The split layout is the
    one artifact where a leaky split is visible at a glance.
    [Phase 2, delivered]
``jobset.py``
    Cross-job comparison: ranking, metric dispersion, status overview and
    wall time.  Forty loss curves answer nothing, because nobody reads forty
    loss curves; each of these collapses a whole set into one view.
    [Phase 4, delivered]
``training.py``
    Learning-rate trace, gradient-norm trace, per-epoch timing, and the
    stacked three-panel diagnostic that relates them.  The loss curve itself
    lives in ``figures.py``, since every engine produces one.
    [Phase 2, delivered]

Planned modules
---------------
``graph.py``
    Graph structure: adjacency sparsity, degree distribution, neighbourhood
    weight profiles, layout of a sampled subgraph.  [Phase 3]
``evaluation.py``
    Predicted against actual, residual diagnostics, error by bucket, baseline
    comparison.  [Phase 5]
``tuning.py``
    Trial history, parameter importance, parallel coordinates.  [Phase 5]
``episodes.py``
    Reward curves, action trajectories, policy-value surfaces, hedging error
    paths.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
