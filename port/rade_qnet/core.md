# `tranql/models/rade/rade_qnet/rade_qnet/core`

1 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 53 | 2379 | `93489b048e7a52fd` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/core/__init__.py`

2379 bytes · SHA-256 `93489b048e7a52fd`

```python
"""
The framework's vocabulary and run-time primitives.

``core`` defines *what things are called and what shape they have*.  It holds
no training logic, no file formats and no plotting.  Crucially it imports no
machine-learning library, which is what allows a specification to be parsed,
validated and hashed in a process that has never loaded PyTorch.

Everything in the framework depends on ``core``; ``core`` depends on nothing
inside ``rade_qnet``.

Sub-packages
------------
``spec``
    Configuration schemas -- the declarative objects a user writes as YAML or
    constructs in Python to describe a run.
``contract``
    The typed payloads passed between pipeline stages.  A stage's signature is
    its contract with the rest of the framework.
``authoring``
    The model-facing interface: the base class a model registers with, plus
    the optional capabilities it may implement to unlock extra behaviour.
    Named for its audience -- this is the only sub-package a model author has
    to read.
``lifecycle``
    How a run is assembled and how it is overridden: component lookup, the
    template-method pipeline base, the ambient run context, hooks, and the
    error hierarchy.  Read by somebody extending the framework.
``provenance``
    What every run can prove about itself: seeding, stable hashing, and
    contextual logging.  Read by somebody who has to answer for a result.

Why five, and why these five
-----------------------------
``lifecycle`` and ``provenance`` were one package called ``runtime``, holding
eight modules whose only shared property was being needed everywhere.  That
is a bin, and the name told a reader nothing: a sub-package of ``core`` being
described as "run-time" does not narrow anything down.

The line between them is *who opens the file*.  Someone registering a model,
overriding a stage or attaching a hook reads ``lifecycle``.  Someone asked
why Tuesday's number differs from Monday's reads ``provenance`` -- the seed,
the digest and the log are the three things that answer it.  Those are
different people on different days, and a package boundary that matches that
is worth more than one that matches an import graph.

Dependency rule
---------------
May import: the standard library, ``pydantic`` and ``numpy``.
May not import: any other ``rade_qnet`` package, or any training library.
"""

__all__: tuple[str, ...] = ()
```

