"""
The XGBoost engine.

The hardest test of the engine contract, which is why it exists.  A booster
fits in one call with no loop, no optimiser, no gradient and no device --
almost every concept the Torch engine introduced is absent.  If the
``Engine`` protocol can accommodate that and still produce the same bundle,
the same metrics and the same reports as a neural network, then it describes
training rather than describing PyTorch.

Modules
-------
``engine.py``
    Translates a boosting run into the framework's contract: one
    ``EpochRecord`` per round, the library's own early stopping, and
    persistence in XGBoost's documented JSON format rather than a pickle.

An optional dependency
----------------------
``xgboost`` is not required to use ``rade_qnet``.  This package is where the
optionality lives: importing it imports the library, so a host that never
trains trees never pays for it, and a host that does gets a clear
``ImportError`` here rather than an obscure failure part-way through a run.

On macOS the wheel additionally needs an OpenMP runtime that ``pip`` does not
install -- ``brew install libomp`` -- which is a second reason it is not a
hard requirement.

That OpenMP runtime is also why this module imports Torch
---------------------------------------------------------
On macOS the Homebrew ``libomp`` that XGBoost links against and the
``libomp`` that the PyTorch wheel bundles are two *different images* of the
same library, loaded at two different addresses.  Whichever is loaded first
wins the process, and the second one's thread pool is then unusable: the
main thread waits on one runtime's barrier while the worker threads belong
to the other, and the process hangs forever with no error and no traceback.

Import order decides it, and only import order -- not call order.  With
``import torch`` first, both libraries work.  With ``import xgboost`` first,
the *first* Torch operation that opens a parallel region never returns.

So this package imports Torch before XGBoost when Torch is installed.  It is
an unpleasant thing for one engine to do about another and it is still the
right call, because the alternative is a silent hang in exactly the workflow
this framework is for: comparing a tree against a network in one process.  A
user cannot diagnose that from the symptom; it takes sampling the process
and noticing two copies of ``libomp`` in the image list.

The cost is bounded.  Nothing is imported that is not already installed, so
a host with trees and no networks pays nothing, and a host with both was
going to import Torch anyway.  Recorded as a deviation in
``PHASE_6_ADDITIONAL_ENGINES.md`` §8.2.

Importing this package registers its components
-----------------------------------------------
Importing ``rade_qnet.engines.xgboost`` registers :class:`XGBoostEngine` under
the name ``"xgboost"``, which is what lets a specification name it as a
string.
"""

from importlib.util import find_spec

# Deliberately before the engine import below, which is what pulls in
# ``xgboost``. See "That OpenMP runtime is also why this module imports
# Torch" above: this ordering is load-bearing, not stylistic, and a linter
# or an import-sorter that moves it reintroduces a hang with no error.
if find_spec("torch") is not None:  # pragma: no cover - depends on the host
    import torch as _torch

from .engine import ENGINE_NAME, BoosterModel, XGBoostEngine

__all__ = ["ENGINE_NAME", "BoosterModel", "XGBoostEngine"]
