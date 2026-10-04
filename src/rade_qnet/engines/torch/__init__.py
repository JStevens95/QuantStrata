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
    The ``Engine`` implementation, and the only file here that satisfies it:
    build, materialise, fit, checkpoint, predict.  [Phase 2]
``materialise.py``
    Runs one dummy forward pass from the input signature so lazy modules
    acquire concrete shapes before any optimiser, checkpoint or distributed
    wrapper touches them.  [Phase 2]
``loaders.py``
    Conversion of a ``BatchSource``'s NumPy batches into device-resident
    tensors.  Static inputs are kept out of per-sample collation and uploaded
    to the device once.  No ``DataLoader`` is constructed: a ``BatchSource``
    already yields whole batches, so prefetching across processes would mean
    pickling the source for no gain on in-memory arrays.  [Phase 2]
``predictor.py``
    Batched inference, including the precompute path for models that can cache
    an encoding of their static inputs.  [Phase 5]

Sub-packages
------------
``training/``
    How a fit is executed: the two drivers, the callbacks that watch them, the
    losses they optimise and the checkpoints they write.
``learners/``
    What one update means -- one module per algorithm.
``hardware/``
    Where the computation runs, and whether it runs the same way twice:
    device resolution, distributed training, and PyTorch seeding.

The four modules above are the four verbs an engine performs -- build, feed,
fit, predict -- and the three sub-packages are the parts of *fit* that are
large enough to have their own vocabulary.  That shape is the engine template:
``engine.py`` is required and every other name is drawn from this list, which
``tests/rade_qnet/engines/test_engine_layout.py`` enforces.  The progression
across the three engines is itself informative -- xgboost is one file because
it fits in a single call and owns no loop, sklearn adds nothing but shares the
hoisted ``engines/loaders.py``, and only PyTorch needs all of it.

Planned modules
---------------
``training/risk.py``
    Differentiable risk measures (mean-variance, conditional value at risk,
    entropic) used as objectives by the pathwise learner.  [Phase 7]

Importing this package registers its components
-----------------------------------------------
Importing ``rade_qnet.engines.torch`` registers :class:`TorchEngine` under the
name ``"torch"`` and :class:`SupervisedLearner` under ``"supervised"``, which
is what lets a specification name them as strings.  It also registers
:func:`~rade_qnet.engines.torch.hardware.determinism.seed_torch`, without which
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

# Importing the name is what registers the seeder, since registration happens
# at that module's import.
from .hardware.determinism import seed_torch
from .learners.random import RandomLearner
from .learners.supervised import SupervisedLearner

__all__ = ["RandomLearner", "SupervisedLearner", "TorchEngine", "seed_torch"]
