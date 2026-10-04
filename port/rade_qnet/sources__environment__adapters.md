# `src/rade_qnet/sources/environment/adapters`

1 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 16 | 580 | `d4274fd3c2191e72` |

---

## 1. `src/rade_qnet/sources/environment/adapters/__init__.py`

580 bytes · SHA-256 `d4274fd3c2191e72`

```python
"""
Bridges from third-party environment interfaces to ``Environment``.

Adapters are thin and one-directional: they translate an external interface
into the framework's protocol and nothing more.  Keeping them isolated here
means an upstream API change is a single-file fix, and the framework never
takes a hard dependency on an external environment library.

Planned modules
---------------
``gymnasium.py``
    Wraps a Gymnasium environment, mapping its spaces onto framework spaces and
    normalising the step-return convention.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
```

