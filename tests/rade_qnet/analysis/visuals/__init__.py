"""
Tests for ``rade_qnet.analysis.visuals``.

Every visual is checked for the two properties that make it reusable: it
returns a figure, and it writes nothing. The second is asserted explicitly
against a temporary working directory, because a plotting function that saves
as a side effect cannot be used from a dashboard, a notebook or another test.

Planned modules
---------------
``test_visuals_style.py``
    The house style applies and -- critically -- restores global plotting state
    on exit, so one figure cannot change the appearance of the next.
    [Phase 1]
``test_visuals_primitives.py``
    The shared building blocks, including empty and single-point inputs.
    [Phase 1]
``test_visuals_data.py``
    Split layout, target distribution and missingness diagnostics.  [Phase 2]
``test_visuals_training.py``
    Learning curves, including a run that stopped early.  [Phase 2]
``test_visuals_graph.py``
    Sparsity, degree distribution and subgraph layout.  [Phase 3]
``test_visuals_jobset.py``
    Cross-job comparison, including a set with a failed job.  [Phase 4]
``test_visuals_evaluation.py``
    Predicted against actual, residuals and baseline comparison.  [Phase 5]
``test_visuals_tuning.py``
    Trial history and parameter importance.  [Phase 5]
``test_visuals_episodes.py``
    Reward curves and action trajectories.  [Phase 7]
``test_visuals_export.py``
    Writing a figure: format, resolution and atomic replacement.  [Phase 1]
"""
