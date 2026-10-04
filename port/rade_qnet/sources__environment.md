# `src/rade_qnet/sources/environment`

1 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 41 | 1563 | `3cf2edd3f52440a4` |

---

## 1. `src/rade_qnet/sources/environment/__init__.py`

1563 bytes · SHA-256 `3cf2edd3f52440a4`

```python
"""
Interactive data: environments an agent learns by acting in.

Two kinds of environment are first-class citizens here, and keeping them
distinct matters:

``Environment``
    The familiar step-and-reward interface.  Gradients, if any, come from the
    learner's own estimator.
``DifferentiableEnvironment``
    The dynamics are themselves differentiable, so a loss can be
    back-propagated straight through a simulated path.  Hedging and replication
    problems are naturally of this form, and forcing them into a
    transition-based interface discards the exact gradients that make them
    tractable.

Most reinforcement-learning libraries model only the first.  Supporting the
second directly is a deliberate choice for a quantitative-finance framework.

Planned modules
---------------
``protocol.py``
    ``Environment`` and ``DifferentiableEnvironment``.  [Phase 7]
``spaces.py``
    Observation and action space descriptions, and the ``PolicySignature``
    derived from them.  [Phase 7]
``vector.py``
    Batched environments stepped in lockstep, synchronously or in worker
    processes.  [Phase 7]
``wrappers.py``
    Composable observation, action and reward wrappers.  Each records itself in
    the run lineage, so a reward-shaping change is visible in the bundle.
    [Phase 7]
``recording.py``
    Episode capture to disk, producing the transition tables that
    ``sources.dataset.transitions`` reads back for offline learning.  [Phase 7]
``adapters/``
    Bridges to third-party environment interfaces.
"""

__all__: tuple[str, ...] = ()
```

