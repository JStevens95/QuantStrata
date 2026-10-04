"""
The flagship model: a hybrid graph-temporal network for P&L replication.

The model learns to express a target instrument's profit and loss as a function
of a set of elementary instruments.  Two structures carry the signal and the
architecture keeps them separate before combining them:

*Cross-sectional structure.*  Instruments relate to each other through shared
attributes (currency pair, tenor, product).  A graph block attends over an
instrument's neighbourhood in that space.

*Temporal structure.*  Each instrument carries a P&L history.  A recurrent
block summarises it.

A fusion layer combines the two representations, a target-attention layer lets
each target instrument weight the elementary instruments it actually depends
on, and the output layer produces the replicated P&L.

Why this model drives the framework's design
--------------------------------------------
It exercises almost every hard requirement at once: static inputs that are
constant across batches, lazy parameter shapes known only after the data build,
fitted state beyond model weights, two fitting axes with different leakage
rules, an expensive encoding worth precomputing once per evaluation pass, and
fan-out across many jobs with different architecture complexity per job.  A
framework that hosts this cleanly will host most things.

Modules
-------
``model.py``
    The network.  Architecture only -- composes the blocks in ``layers/`` and
    defines the forward pass.  [Phase 3]
``layers/``
    The architectural blocks, one per file.  [Phase 3]
``register.py``
    The framework declaration: binds the model to its spec, data module, fitted
    state class and pipeline overrides.  [Phase 3]
``spec.py``
    The model and data specifications.  [Phase 3]
``state.py``
    The ``FittedState`` subclass: target scalers, the selected elementary
    basis, the entity encoder and the graph.  [Phase 3]
``data.py``
    The data module: load, scale, reduce, encode, build the graph, split,
    package.  [Phase 3]
``features/``
    Model-specific feature construction (entity attribute encoding, graph
    construction).  [Phase 3]
``pipelines/``
    Stage overrides for this model.  [Phase 3 / Phase 5]
``reports.py``, ``visuals.py``
    Artifacts specific to this model: the graph diagnostics page and the
    three figures behind it.

Reference
---------
The behaviour this model must reproduce is specified by the parity plan in
``docs/phases/PHASE_0_BASELINE.md``, which pins a golden fixture captured from
the existing ``rade_ml_pt`` implementation.
"""

__all__: tuple[str, ...] = ()

from .register import HybridGnnRnnModel

__all__ = ["HybridGnnRnnModel"]
