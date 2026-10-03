"""
Tests for ``rade_xl.engines.torch`` -- the PyTorch engine.

Tests here use tiny models on the CPU and assert exact or tightly-toleranced
numbers, so the suite stays fast enough to run on every change. Behaviour that
genuinely needs a GPU or multiple processes is marked and skipped when the
hardware is absent, rather than being left untested and unmentioned.

Planned modules
---------------
``test_torch_engine.py``
    Build, materialise, fit, checkpoint and predict against the engine
    contract.  [Phase 2]
``test_torch_loops.py``
    ``fit_epochs`` and ``fit_steps``: early stopping fires at the right epoch,
    the best checkpoint is the one restored, and a resumed run continues from
    the right step.  [Phase 2]
``test_torch_callbacks.py``
    Callback ordering and interaction, particularly early stopping together
    with learning-rate scheduling.  [Phase 2]
``test_torch_losses.py``
    Each loss against hand-computed values.  [Phase 2]
``test_torch_hardware.py``
    Device and precision resolution from a spec, including the fallback when
    requested hardware is unavailable.  [Phase 2]
``test_torch_materialise.py``
    Lazy parameters acquire concrete shapes before an optimiser, a checkpoint
    or a distributed wrapper touches them -- the ordering defect this engine
    exists to fix.  [Phase 2]
``test_torch_checkpoint.py``
    Checkpoints round-trip as state dictionaries, and load without executing
    pickled code.  [Phase 2]
``test_torch_loaders.py``
    Collation, worker configuration, and static inputs uploaded once rather
    than compared per sample.  [Phase 2]
``test_torch_distributed.py``
    Setup and teardown ordering. Skipped unless multiple devices are present.
    [Phase 2]
``test_torch_predictor.py``
    Batched inference, and that the precompute path gives results identical to
    the plain path.  [Phase 5]
``test_torch_risk.py``
    Differentiable risk measures against analytic values.  [Phase 7]
"""
