"""
Tests for ``rade_qnet.core.spec`` -- configuration schemas.

The properties under test are the ones users depend on without knowing it: an
unknown key is rejected rather than ignored, an invalid combination fails at
load time rather than three hours into a run, and a spec round-trips through
``model_dump`` and back to an equal object -- which is what makes a persisted
spec a faithful record of how a model was produced.

Planned modules
---------------
``test_spec_run.py``
    ``RunSpec`` validation and discrimination on ``task``.  [Phase 1]
``test_spec_data.py``
    Source, split and loader specs.  [Phase 1]
``test_spec_training.py``
    Engine-discriminated training specs.  [Phase 1]
``test_spec_hardware.py``
    Device, precision and determinism resolution.  [Phase 1]
``test_spec_reports.py``
    Report selection.  [Phase 1]
``test_spec_jobs.py``
    Job and job-set specs.  [Phase 4]
``test_spec_merge.py``
    Deep merge of defaults with per-job overrides, including the cases where a
    nested override must not discard sibling defaults.  [Phase 4]
"""
