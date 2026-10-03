"""
Tests for ``rade_xl.analysis.reports``.

The rule under test throughout is that no report is load-bearing. A report that
raises must produce a warning and a completed run, never a lost one. Each
report suite therefore includes a deliberately failing writer alongside the
happy path.

Planned modules
---------------
``test_reports_base.py``
    The ``Report`` protocol, registration by name, and a failing report
    degrading to a warning.  [Phase 1]
``test_reports_summary.py``
    Summary contents and the files written.  [Phase 1]
``test_reports_curves.py``
    Curve figures plus the raw history saved alongside them.  [Phase 2]
``test_reports_baselines.py``
    Baselines evaluated on the same split as the model they are compared
    against.  [Phase 2]
``test_reports_quality.py``
    The input data quality report.  [Phase 2]
"""
