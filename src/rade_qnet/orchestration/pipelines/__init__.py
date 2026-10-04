"""
The pipelines that define a model's lifecycle.

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

Modules
-------
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
``reinforce.py``
    ``ReinforcePipeline``: resolve spec, build environment, declare spaces,
    build policy, materialise, collect experience, fit a step budget, package
    bundle, write reports, register.  [Phase 7]

Two training pipelines, not a fork
----------------------------------
``reinforce.py`` is a *sibling* of ``train.py``: the same stage names in the
same order, the same ``TrainingResult``, the same bundle layout, the same four
customisation tiers.  Three stages genuinely differ -- there is no dataset to
build, no target to declare and no pass to make -- and each is a different
*kind* of thing rather than a different parameter.

A single class covering both would branch on ``spec.task`` in those three
stages, and a model author overriding ``build_data`` would then have to know
which branch they were in.  The tiers above only work if a stage means one
thing.  The full reasoning is in ``reinforce.py`` and in
``PHASE_7_REINFORCEMENT_LEARNING.md`` §3.1.
"""

__all__: tuple[str, ...] = ()
