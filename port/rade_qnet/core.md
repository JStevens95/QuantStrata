# `src/rade_qnet/core`

1 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 33 | 1277 | `5288c88f526124bc` |

---

## 1. `src/rade_qnet/core/__init__.py`

1277 bytes · SHA-256 `5288c88f526124bc`

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
``capability``
    The model-facing interface: the base class a model registers with, plus
    the optional protocols a model may implement to unlock extra behaviour.
``runtime``
    The machinery that executes a pipeline: run context, the instrumented step
    runner, hooks, component lookup, seeding and error types.

Dependency rule
---------------
May import: the standard library, ``pydantic`` and ``numpy``.
May not import: any other ``rade_qnet`` package, or any training library.
"""

__all__: tuple[str, ...] = ()
```

