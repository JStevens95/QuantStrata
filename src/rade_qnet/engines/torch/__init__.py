"""
The PyTorch engine.

This is the framework's primary engine and the one the flagship model uses.
Its central idea is a split of concerns that most training code leaves tangled:

*The loop* decides when to step, validate, checkpoint and stop.
*The learner* decides what a single update means.

A supervised regression, a deep Q-network and a pathwise hedging objective are
then three learners sharing one loop, rather than three training scripts.

Modules
-------
``engine.py``
    The ``Engine`` implementation: build, materialise, fit, checkpoint,
    predict.  [Phase 2]
``loops.py``
    ``fit_epochs`` -- passes over a finite dataset -- and the ``Learner``
    protocol it drives.  ``fit_steps``, for a fixed number of updates against
    an unbounded source, arrives with reinforcement learning.  Supervised
    learning naturally wants the first, reinforcement learning the second.
    [Phase 2 / Phase 7]
``learners/``
    Update rules, one per algorithm.
``callbacks.py``
    Early stopping, checkpointing, learning-rate scheduling, gradient-norm
    tracking.  [Phase 2]
``losses.py``
    The loss registry, including the asymmetric and quantile objectives used
    for P&L work.  A loss is engine code because the *learner* consumes it,
    so the P&L-shaped objectives live here rather than in ``domains`` -- the
    dependency rule forbids the alternative, and rightly: it would make this
    engine unusable without the P&L package installed.  [Phase 2]
``hardware.py``
    Device selection, autocast and precision policy, and ``torch.compile``
    application -- resolved from ``HardwareSpec``.  [Phase 2]
``distributed.py``
    Distributed data-parallel setup and teardown.  Wrapping happens *after*
    lazy parameters are materialised, which is the ordering the previous
    implementation got wrong.  [Phase 2]
``materialise.py``
    Runs one dummy forward pass from the input signature so lazy modules
    acquire concrete shapes before any optimiser, checkpoint or distributed
    wrapper touches them.  [Phase 2]
``checkpoint.py``
    Checkpoints as ``state_dict`` payloads rather than pickled modules, so a
    saved model survives a refactor and can be loaded without executing
    arbitrary code.  [Phase 2]
``loaders.py``
    Conversion of a ``BatchSource``'s NumPy batches into device-resident
    tensors.  Static inputs are kept out of per-sample collation and uploaded
    to the device once.  No ``DataLoader`` is constructed: a ``BatchSource``
    already yields whole batches, so prefetching across processes would mean
    pickling the source for no gain on in-memory arrays.  [Phase 2]

Planned modules
---------------
``risk.py``
    Differentiable risk measures (mean-variance, conditional value at risk,
    entropic) used as objectives by the pathwise learner.  [Phase 7]
``predictor.py``
    Batched inference, including the precompute path for models that can cache
    an encoding of their static inputs.  [Phase 5]

Importing this package registers its components
-----------------------------------------------
Importing ``rade_qnet.engines.torch`` registers :class:`TorchEngine` under the
name ``"torch"`` and :class:`SupervisedLearner` under ``"supervised"``, which
is what lets a specification name them as strings.  It also registers
:func:`~rade_qnet.engines.torch.seeding.seed_torch`, without which
``seed_everything`` would leave Torch unseeded -- so two runs of one
configuration would differ in every weight initialisation while both reported
the same seed.

The import is eager rather than lazy on purpose.  A registry populated only
once someone happens to have imported the right module is the classic source
of "no engine named 'torch'" from a run whose configuration is perfectly
correct, and the only reliable cure is for the registration to be a
consequence of importing the package that owns it.  Importing this package
already implies that ``torch`` is installed, so nothing is paid by a host that
does not use it.
"""

from .engine import TorchEngine
from .learners.supervised import SupervisedLearner

# Importing the name is what registers the seeder, since registration happens
# at that module's import.
from .seeding import seed_torch

__all__ = ["SupervisedLearner", "TorchEngine", "seed_torch"]
