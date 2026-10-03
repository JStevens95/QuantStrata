"""
The four pipelines that define a model's lifecycle.

Each pipeline is a template method: ``run()`` calls a sequence of small,
individually typed, individually overridable steps.  The granularity is the
point.  A model author who wants a different loss override one step; they do
not reimplement training.

There are four customisation tiers, and a model should use the lowest one that
works:

1. **Spec only.**  Change configuration; write no code.
2. **Add reports or hooks.**  Extra artifacts and instrumentation, no pipeline
   subclass.
3. **Override one step.**  Keep the sequence, replace a single stage.
4. **Override ``run()``.**  Reserved for genuinely different sequences; the
   conformance suite still applies.

Planned modules
---------------
``train.py``
    ``TrainPipeline``: resolve spec, build source, build model, materialise,
    fit, evaluate, package bundle, write reports, register.  [Phase 1 skeleton,
    Phase 2 complete]
``evaluate.py``
    ``EvaluatePipeline``: load bundle, rebuild source from saved lineage,
    predict, invert target transforms, compute metrics, write reports.
    [Phase 5]
``infer.py``
    ``InferPipeline``: load bundle, prepare inputs for unseen entities,
    predict, emit predictions with provenance.  [Phase 5]
``tune.py``
    ``TunePipeline``: propose trials, run a short train per trial against a
    cached data build, select the best, optionally refit.  [Phase 5]
"""

__all__: tuple[str, ...] = ()
