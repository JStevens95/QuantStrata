# `src/rade_qnet/domains`

1 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 33 | 1251 | `ab3c59729154868e` |

---

## 1. `src/rade_qnet/domains/__init__.py`

1251 bytes · SHA-256 `ab3c59729154868e`

```python
"""
Business context -- the only place that knows what the numbers mean.

Everywhere else in the framework, a target is an array.  Here it is a profit
and loss series for a specific instrument on a specific desk.  Separating the
two is what lets the framework be reused on a problem nobody has thought of
yet, and it is also what stops a currency convention from ending up hard-coded
in a training loop.

A domain contributes three kinds of thing: adapters that read the firm's data
into framework sources, metrics that are meaningful to that desk, and
environments for problems in that domain.  It contributes no pipelines and no
engines.

Sub-packages
------------
``pnl``
    P&L replication: portfolio and universe adapters, cluster definitions, and
    replication-quality metrics.  The domain the flagship model serves.
``hedging``
    Differentiable hedging environments and the risk objectives they optimise.
    [Phase 7]
``trading``
    Execution and trading-signal problems.  [later]

Dependency rule
---------------
May import: ``core``, ``sources``, ``analysis``.
Nothing in the framework may import this package.  If a framework module needs
something from here, the abstraction is in the wrong place.
"""

__all__: tuple[str, ...] = ()
```

