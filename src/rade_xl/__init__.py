"""
A model-independent framework for machine learning and reinforcement learning.

The framework owns the *lifecycle* of a model (configuration, data, training,
persistence, evaluation, reporting and fan-out across many jobs).  A model
author supplies only the parts that are genuinely specific to their model: the
architecture, how its data is built, and what fitted state it needs at
inference time.  Everything else is inherited.

Design goals
------------
1. **One lifecycle, many models.**  A ridge regression and a graph-temporal
   deep network move through identical stages and produce identical artifacts.
2. **Typed hand-offs.**  Every stage returns a declared contract, so a stage
   can be replaced without reading the stage that follows it.
3. **Scales both ways.**  A simple model costs one small file; a complex model
   may override any stage without forking the framework.
4. **Reproducible by construction.**  Specs, seeds, data fingerprints and code
   versions are captured in every saved bundle.

Package map
-----------
``core``
    The framework's vocabulary: configuration schemas, stage contracts,
    optional model capabilities and the run-time primitives that execute a
    pipeline.  Depends on no machine-learning library.
``sources``
    Where a training signal comes from: fixed datasets, interactive
    environments, and the batching layer that turns either into a uniform
    stream of training batches.
``engines``
    How optimisation actually happens, one adapter per library
    (PyTorch, XGBoost, scikit-learn).
``orchestration``
    Who coordinates the work: the pipeline bases, fan-out across a list of
    jobs, and placement of those jobs onto cores, GPUs or a cluster.
``storage``
    The system of record: versioned model bundles, the catalog and the
    experiment tracker.
``analysis``
    Understanding a run: metrics, pure plotting functions and the report
    writers that persist artifacts to disk.
``models``
    The model library.  ``hybrid_gnn_rnn`` is the flagship; ``baselines``
    holds deliberately simple models that keep the framework honest.
``domains``
    Business context (P&L replication, hedging, trading).  Nothing in the
    layers above may import this package.
``testkit``
    Tools that prove a model conforms to the framework's contracts.  Not
    part of the production runtime.
``api``
    The front door: ``train``, ``train_jobs`` and ``train_portfolio``.  A
    module rather than a package, because it adds no behaviour -- it only
    assembles pieces a caller could assemble by hand.  It lives at the top
    level because it reaches across layers that may not see each other:
    ``orchestration`` may not import ``domains``, but expanding a book into
    jobs and then running them needs both.

Import convention
-----------------
Modules inside ``rade_xl`` import each other with **explicit relative
imports** (``from ..core.contract import DataBundle``).  This keeps the
package relocatable: it behaves identically whether it is imported as
``src.rade_xl`` from the repository root or as ``rade_xl`` from an installed
distribution.  Code outside the package should use whichever absolute path
matches how it was installed.

Documentation
-------------
``docs/ARCHITECTURE.md``
    What the framework does and how the pieces fit together.
``docs/IMPLEMENTATION.md``
    The phased build plan and the definition of done for each phase.
``docs/CODING_STANDARDS.md``
    The standard every contribution is held to.
"""

__version__ = "0.1.0.dev0"

# The public surface is populated as each phase lands; see
# ``docs/IMPLEMENTATION.md``.  Keeping it explicit (rather than re-exporting
# everything) means ``from rade_xl import *`` can never leak internals.
__all__: tuple[str, ...] = ("__version__",)
