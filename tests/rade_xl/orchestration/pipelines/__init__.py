"""
Tests for ``rade_xl.orchestration.pipelines`` -- the four lifecycle pipelines.

Pipelines are tested against synthetic models and sources from
``rade_xl.testkit.fixtures``, never against real data. That keeps the suite
fast and, more importantly, keeps it testing the pipeline rather than the
model: a failure here is unambiguously the framework's fault.

The overriding tests matter as much as the happy paths. Each of the four
customisation tiers is exercised, because the promise that a model can replace
one step without reimplementing training is only credible if something checks
it.

Planned modules
---------------
``test_pipelines_train.py``
    Stage ordering, the artifacts produced, and each override tier: spec only,
    added reports, a single replaced step, and a fully replaced ``run()``.
    [Phase 1, extended in Phase 2]
``test_pipelines_evaluate.py``
    Rebuilding a source from saved lineage, and metrics reported in original
    target units rather than transformed space.  [Phase 5]
``test_pipelines_infer.py``
    Predicting for entities unseen during training, and prediction provenance
    pointing back to a specific bundle.  [Phase 5]
``test_pipelines_tune.py``
    Trial generation, the data build being cached across trials rather than
    repeated, and best-trial selection.  [Phase 5]
"""
